"""Consistency of the experiment configurations, and the new model variants through the pipeline."""

import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from stokes_ft.config import load_config
from stokes_ft.study import run_study


def resolved(name, overrides=()):
    return OmegaConf.to_container(load_config(name, list(overrides)), resolve=True)


def test_upstream_derived_configs_have_no_model_variant_key():
    # runs are resumable only with an identical resolved configuration: adding keys to official,
    # local or full would make the stored local study impossible to resume (G1)
    for name in ("official", "local", "full", "smoke"):
        assert "model_variant" not in resolved(name) and "processor_repeats" not in resolved(name)


def test_arch_differs_from_local_only_in_the_controlled_keys():
    a, b = resolved("local"), resolved("arch")
    changed = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert changed == {"name", "model_variant", "processor_repeats", "finetune", "evaluation"}
    assert b["finetune"]["samples"] == [] and b["finetune"]["log_every"] == a["finetune"]["log_every"]
    for key in ("epochs", "num_training_samples", "lr", "seed", "data", "hidden_dim_node_encoder", "batch_size"):
        assert a[key] == b[key]


def test_full_is_not_reduced():
    full = resolved("full")
    assert (full["epochs"], full["num_training_samples"], full["pi_iters"]) == (500, 500, 10000)
    assert full["finetune"]["samples"] == list(range(10)) and full["num_test_samples"] == 10


@pytest.mark.parametrize("variant", ["global_latent", "shared_depth"])
def test_variant_trains_checkpoints_resumes_and_predicts(tiny_root, variant):
    root, overrides = tiny_root
    cfg = load_config("smoke", overrides + [f"name=arch_{variant}", f"+model_variant={variant}", "finetune.samples=[]"])
    run_dir = run_study(cfg, stages=["train", "predict"], runs_root=root / "runs")
    record = json.loads((run_dir / "train_metrics.json").read_text())
    assert record["epochs"] == cfg.epochs and np.isfinite(record["final_train_loss"])
    # resume: the completed stage is recognised and not retrained
    before = (run_dir / "train_metrics.json").read_text()
    run_study(cfg, stages=["train", "predict"], runs_root=root / "runs", resume=True)
    assert (run_dir / "train_metrics.json").read_text() == before
    assert len(list((run_dir / "predictions").glob("graph_*.vtp"))) == cfg.num_test_samples
