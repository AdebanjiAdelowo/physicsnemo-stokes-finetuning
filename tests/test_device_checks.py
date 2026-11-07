import torch

from stokes_ft.device_checks import run_device_checks


def test_device_checks_pass_on_cpu(tiny_cfg, tiny_root):
    # run directories inside the dataset root: the loader needs a digit-free relative path to the data
    out = run_device_checks(tiny_cfg, tiny_root[0] / "runs" / "device_checks", torch.device("cpu"), log=lambda *_: None)
    assert out["resumed_flag"] and out["resume_epochs_in_history"] == 2 and out["checkpoint_epoch"] == 2
    assert out["resume_max_relative_difference"] < 1e-3
    assert out["repeat_max_relative_difference"] < 1e-3
    assert out["checkpoint_outputs_identical"] and out["gradients_finite"] and out["gradient_norm"] > 0
    assert out["passed"]
