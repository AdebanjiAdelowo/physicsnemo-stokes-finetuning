"""The cross-session recovery test on CPU: two simulated sessions that share only the packed state."""

import torch

from stokes_ft.config import load_config
from stokes_ft.recovery_check import first_session, second_session
from stokes_ft.session_state import pack, unpack


def test_recovery_across_simulated_sessions(tiny_root, tmp_path):
    root, overrides = tiny_root
    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # deterministic CPU reductions, for the bit-identity criterion
    cfg = {name: load_config("smoke", overrides + ["epochs=5", "training.checkpoint_every=2", f"name=recovery_{name}"])
           for name in ("resumed", "reference", "repeat", "repeatb", "repeatc")}
    first = root / "sessionone"
    out = first_session(cfg["resumed"], cfg["reference"], first, pause_epoch=3)
    assert out["resumed_status"]["state"] == "paused" and out["reference_epochs"] == 5
    for name in ("resumed", "reference"):
        pack(first / f"recovery_{name}", tmp_path / f"state-{name}.tar")
    second = root / "sessiontwo"
    for name in ("resumed", "reference"):
        unpack(tmp_path / f"state-{name}.tar", second)
    repeats = [cfg[n] for n in ("repeat", "repeatb", "repeatc")]
    result = second_session(cfg["resumed"], cfg["reference"], repeats, second, 3, torch.device("cpu"))
    assert result["checkpoints_restored"] == ["manifest.0.3.json", "manifest.0.2.json"]
    assert result["corruption_drill"]["detected"] and result["corruption_drill"]["fell_back_to_older"]
    assert result["incompatibility_drill"]["refused"]
    assert result["resumed_from_epoch"] == 3 and result["resumed_from_verified_checkpoint"]
    assert result["resumed_vs_reference_max_relative_difference"] < 1e-4
    assert result["criterion_v2"]["epochs_compared"] == [4, 5]
    assert len(result["criterion_v2"]["repeats_vs_reference_after_pause"]) == 3
    assert result["criterion_v1"]["passed"] and result["criterion_v2"]["passed"] and result["passed"]
    # on the CPU training is deterministic: the bit-identity criterion must hold as well
    assert result["criterion_v3"]["resumed_bit_identical_to_reference"]
    assert result["criterion_v3"]["verdict"] == "PASSED"
    torch.set_num_threads(threads)
