"""The whole study on a 20-mesh synthetic dataset: both stages, evaluation, resume."""

import csv
import json

import numpy as np
import pytest

from stokes_ft.data import read_sample
from stokes_ft.evaluate import TABLE
from stokes_ft.study import run_study


@pytest.fixture(scope="module")
def study(tiny_root):
    from stokes_ft.config import load_config

    root, overrides = tiny_root
    cfg = load_config("smoke", overrides + ["name=pipeline"])
    return cfg, run_study(cfg, runs_root=root / "runs"), root


def test_checkpoints_of_both_stages_are_kept(study):
    cfg, run_dir, _ = study
    names = sorted(p.name for p in (run_dir / "checkpoints" / "data_driven").iterdir())
    assert names == [f"MeshGraphNet.0.{cfg.epochs}.mdlus", f"checkpoint.0.{cfg.epochs}.pt"]
    for index in cfg.finetune.samples:
        assert (run_dir / "checkpoints" / "finetune" / f"fourier_dnn_{index}.pt").exists()


def test_both_stages_are_stored_on_the_same_mesh(study):
    cfg, run_dir, _ = study
    predict = json.loads((run_dir / "predict_metrics.json").read_text())
    meta = json.loads((run_dir / "study_metadata.json").read_text())
    assert predict["test_files"] == [f"res_{n}.vtp" for n in meta["dataset"]["test_numbers_used"]]
    for index in cfg.finetune.samples:
        sample = read_sample(run_dir / "predictions" / f"graph_{index}.vtp")
        source = read_sample(run_dir.parents[1] / "data" / "raw" / "results" / predict["test_files"][index])
        np.testing.assert_array_equal(sample["coords"], source["coords"])
        np.testing.assert_array_equal(sample["ref_u"], source["ref_u"])
        for prefix in ("gnn", "filtered"):
            assert sample[f"{prefix}_u"].shape == sample["ref_u"].shape
        assert not np.allclose(sample["gnn_u"], sample["filtered_u"])


def test_result_files(study):
    cfg, run_dir, _ = study
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["result_status"] == "UNLABELLED" and summary["samples"] == len(cfg.finetune.samples)
    for key, _, _ in TABLE:
        c = summary["comparison"][key]
        assert np.isclose(c["change"], c["physics_finetuned"] - c["data_driven"])
        assert 0 <= c["samples_improved"] <= summary["samples"]
    # the reference has no prediction error and satisfies the boundary conditions
    assert summary["comparison"]["rel_l2_velocity"]["reference"] is None
    assert summary["comparison"]["bc_boundary"]["reference"] < 1e-12
    assert "autodiff_momentum" in summary["stages"]["physics_finetuned"]
    assert "autodiff_momentum" not in summary["stages"]["data_driven"]
    assert summary["compute"]["data_driven_parameters"] > summary["compute"]["finetune_parameters_per_mesh"] == 49923

    rows = list(csv.DictReader(open(run_dir / "per_sample.csv")))
    assert len(rows) == 3 * len(cfg.finetune.samples)
    assert {r["stage"] for r in rows} == {"reference", "data_driven", "physics_finetuned"}
    assert len(list(csv.reader(open(run_dir / "summary.csv")))) == len(TABLE) + 1

    fields = np.load(run_dir / "fields.npz")
    i = cfg.evaluation.plot_samples[0]
    n = len(fields[f"s{i}_coords"])
    for stage in ("reference", "data_driven", "physics_finetuned"):
        for key in ("u", "v", "p", "continuity", "momentum_x", "momentum_y"):
            assert fields[f"s{i}_{stage}_{key}"].shape == (n,)

    history = list(csv.DictReader(open(run_dir / "history.csv")))
    assert [int(r["epoch"]) for r in history] == list(range(1, cfg.epochs + 1))
    ft = list(csv.DictReader(open(run_dir / "finetune" / f"history_{i}.csv")))
    assert int(ft[-1]["iteration"]) == cfg.pi_iters


def test_metadata(study):
    cfg, run_dir, _ = study
    meta = json.loads((run_dir / "study_metadata.json").read_text())
    session = meta["sessions"][0]
    for key in ("timestamp_utc", "git_commit", "git_dirty", "device", "python_version", "torch_version",
                "physicsnemo", "torch_geometric_version", "torch_scatter_version", "platform"):
        assert key in session
    assert session["physicsnemo"]["commit"] == "b45a5c810c741e6b41f8515be24c51121f8fc21f"
    assert meta["seed"] == cfg.seed and meta["config"]["pi_iters"] == cfg.pi_iters
    assert meta["dataset"]["split_sizes"] == {"train": 16, "validation": 2, "test": 2}


def test_resume_skips_finished_stages_and_refuses_another_config(study, capsys):
    cfg, run_dir, root = study
    before = (run_dir / "predictions" / "graph_0.vtp").stat().st_mtime_ns
    run_study(cfg, resume=True, runs_root=root / "runs")
    out = capsys.readouterr().out
    assert "already complete" in out and "predictions already written" in out and "already fine-tuned" in out
    assert (run_dir / "predictions" / "graph_0.vtp").stat().st_mtime_ns == before
    assert len(json.loads((run_dir / "study_metadata.json").read_text())["sessions"]) == 2
    with pytest.raises(RuntimeError, match="holds checkpoints"):
        run_study(cfg, stages=["train"], runs_root=root / "runs")
    other = cfg.copy()
    other.pi_iters = cfg.pi_iters + 1
    with pytest.raises(RuntimeError, match="another configuration"):
        run_study(other, stages=["evaluate"], resume=True, runs_root=root / "runs")


def test_same_seed_repeats_the_data_driven_stage_to_rounding(study, tiny_root):
    """Initialisation and sample order are seeded. Scatter-add accumulation order is not fixed,
    so two runs agree to rounding and are not bit-for-bit identical."""
    from stokes_ft.config import load_config

    cfg, run_dir, root = study
    again = load_config("smoke", tiny_root[1] + ["name=pipeline_again"])
    other_dir = run_study(again, stages=["train"], runs_root=root / "runs")
    a = [float(r["loss"]) for r in csv.DictReader(open(run_dir / "history.csv"))]
    b = [float(r["loss"]) for r in csv.DictReader(open(other_dir / "history.csv"))]
    np.testing.assert_allclose(a, b, rtol=1e-4)
