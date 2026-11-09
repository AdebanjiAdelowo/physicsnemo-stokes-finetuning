import pytest
import torch
from omegaconf import OmegaConf

from stokes_ft.config import REPO_ROOT, load_config
from stokes_ft.model import build_meshgraphnet, count_parameters


def test_full_arch_keeps_historical_training_controls():
    original = OmegaConf.to_container(load_config("full_select"), resolve=True)
    full = OmegaConf.to_container(load_config("full_arch"), resolve=True)
    for key in ("name", "model_variant", "num_test_samples", "finetune", "evaluation"):
        original.pop(key, None)
        full.pop(key, None)
    assert full == original
    cfg = load_config("full_arch")
    assert cfg.num_test_samples == 100 and cfg.training.select_checkpoints


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matched_seeds_have_identical_initial_shared_weights(seed):
    torch.manual_seed(seed)
    base = build_meshgraphnet(load_config("full_arch", [f"seed={seed}"]))
    torch.manual_seed(seed)
    global_model = build_meshgraphnet(load_config("full_arch", [f"seed={seed}", "model_variant=global_latent"]))
    for key, value in base.state_dict().items():
        assert torch.equal(value, global_model.state_dict()[key]), key
    assert count_parameters(base) == 2_531_587
    assert count_parameters(global_model) == 2_564_611


@pytest.mark.parametrize("arm", ["baseline", "global_latent"])
def test_probe_resumes_and_checks_selections_on_cpu(arm, tiny_root, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "scripts"))
    from full_arch_probe import probe_arm

    root, overrides = tiny_root
    cfg = load_config("full_arch", overrides + ["device=cpu", f"model_variant={arm}"])
    work = root / "probe" / arm
    work.mkdir(parents=True)
    record = probe_arm(cfg, work, torch.device("cpu"))
    assert record["opened_splits"] == ["train", "validation"]
    assert record["train_metrics"]["resumed_from_epoch"] == 1
    assert set(record["selected"]) == {"pressure", "velocity", "joint"}
    assert record["validation_inference_seconds_per_mesh"] > 0
    assert record["peak_allocated_cuda_mib"] is None
