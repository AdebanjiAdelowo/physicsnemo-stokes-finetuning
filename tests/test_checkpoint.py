"""Verified checkpoints, pause and resume, and carrying a run between sessions (CPU, synthetic data)."""

import csv
import json
import random
import shutil

import numpy as np
import pytest
import torch

from stokes_ft import checkpoint
from stokes_ft.config import load_config
from stokes_ft.engine import train_data_driven
from stokes_ft.session_state import pack, unpack
from stokes_ft.study import run_study


def _losses(run_dir):
    with open(run_dir / "history.csv") as f:
        return np.array([[float(r[k]) for k in ("loss", "val_rel_l2_u", "val_rel_l2_v", "val_rel_l2_p")]
                         for r in csv.DictReader(f)])


@pytest.fixture(scope="module")
def runs(tiny_root):
    """An uninterrupted 3-epoch run and the same run paused after epoch 1, then resumed."""
    root, overrides = tiny_root
    base = overrides + ["epochs=3", "lr_decay_rate=0.9"]
    full = load_config("smoke", base + ["name=ckpt_full"])
    paused = load_config("smoke", base + ["name=ckpt_paused"])
    full_dir = run_study(full, stages=["train"], runs_root=root / "runs")
    paused_dir = run_study(paused, stages=["train"], runs_root=root / "runs", stop_after_epoch=1)
    status_after_pause = json.loads((paused_dir / "status.json").read_text())
    metrics_after_pause = (paused_dir / "train_metrics.json").exists()
    run_study(paused, stages=["train"], resume=True, runs_root=root / "runs")
    return root, full, full_dir, paused, paused_dir, status_after_pause, metrics_after_pause


def test_pause_writes_a_verified_checkpoint_and_no_completion_record(runs):
    *_, status, metrics_written = runs
    assert status["state"] == "paused" and status["epochs_completed"] == 1
    assert not metrics_written


def test_resumed_run_matches_the_uninterrupted_run(runs):
    _, _, full_dir, _, paused_dir, *_ = runs
    record = json.loads((paused_dir / "train_metrics.json").read_text())
    assert record["status"] == "complete" and record["resumed_from_epoch"] == 1
    assert record["resumed_from_verified_checkpoint"] is True
    assert json.loads((paused_dir / "status.json").read_text())["state"] == "complete"
    np.testing.assert_allclose(_losses(paused_dir), _losses(full_dir), rtol=1e-4)


def test_newest_three_checkpoints_are_kept_with_manifests(runs):
    _, _, full_dir, *_ = runs
    ckpt = full_dir / "checkpoints" / "data_driven"
    assert [m.name for m in checkpoint.manifests(ckpt)] == ["manifest.0.3.json", "manifest.0.2.json", "manifest.0.1.json"]
    for m in checkpoint.manifests(ckpt):
        assert checkpoint.verify(m, None) == (True, "ok")
    assert not [p for p in ckpt.iterdir() if p.name.startswith(".tmp-")]


def _copy(root, src, name):
    dst = root / "runs" / name
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    (dst / "train_metrics.json").unlink(missing_ok=True)
    return dst


def test_corrupted_newest_checkpoint_falls_back_to_an_older_one(runs):
    root, full, full_dir, *_ = runs
    run = _copy(root, full_dir, "ckpt_corrupt")
    with open(run / "checkpoints" / "data_driven" / "checkpoint.0.3.pt", "ab") as f:
        f.write(b"corruption")
    cfg = full.copy()
    cfg.name = "ckpt_corrupt"
    messages = []
    record = train_data_driven(cfg, run, torch.device("cpu"), resume=True, log=messages.append)
    assert record["resumed_from_epoch"] == 2
    assert any("checksum mismatch in checkpoint.0.3.pt" in m for m in messages)


def test_checkpoint_without_manifest_is_ignored(runs):
    root, full, full_dir, *_ = runs
    run = _copy(root, full_dir, "ckpt_incomplete")
    (run / "checkpoints" / "data_driven" / "manifest.0.3.json").unlink()  # write interrupted before the commit point
    cfg = full.copy()
    cfg.name = "ckpt_incomplete"
    record = train_data_driven(cfg, run, torch.device("cpu"), resume=True, log=lambda *_: None)
    assert record["resumed_from_epoch"] == 2


def test_incompatible_configuration_is_refused(runs):
    root, full, full_dir, *_ = runs
    run = _copy(root, full_dir, "ckpt_other")
    other = full.copy()
    other.name, other.lr = "ckpt_other", full.lr * 2
    with pytest.raises(checkpoint.IncompatibleCheckpoint, match="config_sha256"):
        train_data_driven(other, run, torch.device("cpu"), resume=True, log=lambda *_: None)


def test_changed_normalisation_statistics_are_refused(runs):
    root, full, full_dir, *_ = runs
    m = json.loads((full_dir / "checkpoints" / "data_driven" / "manifest.0.3.json").read_text())
    fp = dict(m["fingerprint"], node_stats_sha256="0" * 64)
    with pytest.raises(checkpoint.IncompatibleCheckpoint, match="node_stats_sha256"):
        checkpoint.verify(full_dir / "checkpoints" / "data_driven" / "manifest.0.3.json", fp)


def test_random_number_generator_states_round_trip():
    random.seed(1), np.random.seed(1), torch.manual_seed(1)
    state = checkpoint._rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(1).item())
    random.seed(2), np.random.seed(2), torch.manual_seed(2)
    checkpoint._set_rng_state(state)
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == expected


def test_session_state_round_trip_and_refusals(runs, tmp_path):
    root, _, _, _, paused_dir, *_ = runs
    tar = tmp_path / "state.tar"
    summary = pack(paused_dir, tar)
    assert summary["status"]["state"] == "complete" and all(c["ok"] for c in summary["checkpoints"])
    restored = unpack(tar, tmp_path / "next_session")
    assert all(c["ok"] for c in restored["checkpoints"])
    assert _losses(tmp_path / "next_session" / paused_dir.name).tolist() == _losses(paused_dir).tolist()
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        unpack(tar, tmp_path / "next_session")
    with open(tar, "ab") as f:
        f.write(b"x")
    with pytest.raises(RuntimeError, match="does not match its sidecar"):
        unpack(tar, tmp_path / "third_session")


def test_resumed_run_is_bit_identical_on_a_deterministic_device(runs):
    """On the CPU training is deterministic, so pausing and resuming must not change a single bit:
    every stored history value and every final weight equals the uninterrupted run's."""
    from physicsnemo.utils import load_checkpoint

    from stokes_ft.model import build_meshgraphnet

    _, full, full_dir, paused, paused_dir, *_ = runs

    def rows(run_dir):
        with open(run_dir / "history.csv") as f:
            return [{k: v for k, v in r.items() if k != "seconds"} for r in csv.DictReader(f)]

    assert rows(paused_dir) == rows(full_dir)
    a, b = build_meshgraphnet(full), build_meshgraphnet(paused)
    load_checkpoint(str(full_dir / "checkpoints" / "data_driven"), models=a, epoch=full.epochs)
    load_checkpoint(str(paused_dir / "checkpoints" / "data_driven"), models=b, epoch=paused.epochs)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
