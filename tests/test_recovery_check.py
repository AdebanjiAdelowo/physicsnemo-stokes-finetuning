"""The cross-session recovery test on CPU: two simulated sessions that share only the packed state."""

import torch

from stokes_ft.config import load_config
from stokes_ft.recovery_check import first_session, second_session
from stokes_ft.session_state import pack, unpack


def test_recovery_across_simulated_sessions(tiny_root, tmp_path):
    root, overrides = tiny_root
    cfg = {name: load_config("smoke", overrides + ["epochs=5", "training.checkpoint_every=2", f"name=recovery_{name}"])
           for name in ("resumed", "reference", "repeat")}
    first = root / "sessionone"
    out = first_session(cfg["resumed"], cfg["reference"], first, pause_epoch=3)
    assert out["resumed_status"]["state"] == "paused" and out["reference_epochs"] == 5
    for name in ("resumed", "reference"):
        pack(first / f"recovery_{name}", tmp_path / f"state-{name}.tar")
    second = root / "sessiontwo"
    for name in ("resumed", "reference"):
        unpack(tmp_path / f"state-{name}.tar", second)
    result = second_session(cfg["resumed"], cfg["reference"], cfg["repeat"], second, 3, torch.device("cpu"))
    assert result["checkpoints_restored"] == ["manifest.0.3.json", "manifest.0.2.json"]
    assert result["corruption_drill"]["detected"] and result["corruption_drill"]["fell_back_to_older"]
    assert result["incompatibility_drill"]["refused"]
    assert result["resumed_from_epoch"] == 3 and result["resumed_from_verified_checkpoint"]
    assert result["resumed_vs_reference_max_relative_difference"] < 1e-4
    assert result["passed"]
