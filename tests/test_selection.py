"""Validation-based checkpoint selection: criteria, persistence, resume, and no use of the test split."""

import csv
import json

import numpy as np
import pytest
import torch
from physicsnemo.utils import load_checkpoint

from stokes_ft import selection
from stokes_ft.config import load_config
from stokes_ft.model import build_meshgraphnet
from stokes_ft.study import run_study


def test_criteria_are_the_fixed_definitions():
    e = {"u": 0.2, "v": 0.4, "p": 0.5}
    assert selection.CRITERIA["velocity"](e) == pytest.approx(0.3)
    assert selection.CRITERIA["pressure"](e) == pytest.approx(0.5)
    assert selection.CRITERIA["joint"](e) == pytest.approx(0.4)


def test_ties_keep_the_earlier_epoch():
    state = {n: {"score": f({"u": 0.2, "v": 0.4, "p": 0.5}), "epoch": 1} for n, f in selection.CRITERIA.items()}
    assert selection.improved(state, {"u": 0.2, "v": 0.4, "p": 0.5}) == []
    assert selection.improved(state, {"u": 0.2, "v": 0.4, "p": 0.4}) == ["pressure", "joint"]


@pytest.fixture(scope="module")
def selected_runs(tiny_root, monkeypatch_module):
    root, overrides = tiny_root
    splits = []
    import stokes_ft.engine as engine

    real = engine.make_dataset

    def recording(cfg, split, *a, **k):
        splits.append(split)
        return real(cfg, split, *a, **k)

    monkeypatch_module.setattr(engine, "make_dataset", recording)
    base = overrides + ["epochs=4", "lr_decay_rate=0.9", "+training.select_checkpoints=true"]
    full = load_config("smoke", base + ["name=sel_full"])
    paused = load_config("smoke", base + ["name=sel_paused"])
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    full_dir = run_study(full, stages=["train"], runs_root=root / "runs")
    paused_dir = run_study(paused, stages=["train"], runs_root=root / "runs", stop_after_epoch=2)
    run_study(paused, stages=["train"], resume=True, runs_root=root / "runs")
    torch.set_num_threads(threads)
    return full, full_dir, paused_dir, splits


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def _history(run_dir):
    with open(run_dir / "history.csv") as f:
        return [{k: float(r[f"val_rel_l2_{k}"]) for k in "uvp"} | {"epoch": int(r["epoch"])} for r in csv.DictReader(f)]


def test_selected_epochs_are_the_validation_minima(selected_runs):
    _, full_dir, _, _ = selected_runs
    hist = _history(full_dir)
    for name, score in selection.CRITERIA.items():
        record = json.loads((full_dir / "checkpoints" / "selected" / name / "selection.json").read_text())
        scores = [score(h) for h in hist]
        assert record["epoch"] == hist[int(np.argmin(scores))]["epoch"]  # argmin: first of ties
        assert record["score"] == pytest.approx(min(scores))


def test_selected_weights_load(selected_runs):
    cfg, full_dir, _, _ = selected_runs
    for name in selection.CRITERIA:
        record = json.loads((full_dir / "checkpoints" / "selected" / name / "selection.json").read_text())
        model = build_meshgraphnet(cfg)
        assert load_checkpoint(str(full_dir / "checkpoints" / "selected" / name), models=model) == record["epoch"]


def test_selection_survives_pause_and_resume(selected_runs):
    _, full_dir, paused_dir, _ = selected_runs
    for name in selection.CRITERIA:
        a = json.loads((full_dir / "checkpoints" / "selected" / name / "selection.json").read_text())
        b = json.loads((paused_dir / "checkpoints" / "selected" / name / "selection.json").read_text())
        assert a["epoch"] == b["epoch"] and a["score"] == b["score"]


def test_training_never_loads_the_test_split(selected_runs):
    *_, splits = selected_runs
    assert splits and set(splits) <= {"train", "validation"}


def test_selection_after_a_lost_epoch_is_removed(tmp_path):
    (tmp_path / "pressure").mkdir()
    (tmp_path / "pressure" / "selection.json").write_text(json.dumps({"score": 0.1, "epoch": 7}))
    (tmp_path / "joint").mkdir()
    (tmp_path / "joint" / "selection.json").write_text(json.dumps({"score": 0.2, "epoch": 3}))
    messages = []
    state = selection.resume(tmp_path, 5, log=messages.append)
    assert "pressure" not in state and state["_since_pressure"] == 5 and state["joint"]["epoch"] == 3
    assert not (tmp_path / "pressure").exists() and any("WARNING" in m for m in messages)
