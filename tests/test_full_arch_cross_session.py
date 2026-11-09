import pytest
import torch

from stokes_ft.config import REPO_ROOT, load_config
from stokes_ft.data import prepare_split
from stokes_ft.session_state import pack, unpack


@pytest.mark.parametrize("arm", ["baseline", "global_latent"])
def test_exact_boundary_and_failure_injections(arm, tiny_root, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "scripts"))
    from full_arch_cross_session import compare, drills, inspect_state, replay, seed_all, train

    root, overrides = tiny_root
    first, second = root / f"{arm}-first", root / f"{arm}-second"
    cfg = {path: load_config("full_arch", overrides + [
        "device=cpu", f"model_variant={arm}", f"name={arm}-{path}",
    ]) for path in ("reference", "resumed")}
    split = prepare_split(cfg["resumed"])
    for path, config in cfg.items():
        seed_all(config.seed)
        run = train(config, first, stop_after_epoch=1 if path == "resumed" else config.epochs)
        archive = root / f"{arm}-{path}.tar"
        pack(run, archive)
        unpack(archive, second)
    paused = second / cfg["resumed"].name
    initial = inspect_state(paused, cfg["resumed"], split, None, 1, int(torch.cuda.is_available()) * torch.cuda.device_count())
    scratch = root / f"{arm}-drills"
    scratch.mkdir()
    devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    checks = drills(paused, cfg["resumed"], split, None, root / f"{arm}-resumed.tar", scratch, devices)
    assert set(checks) == {"archive_checksum", "incomplete", "dataset", "epoch", "update", "model_variant", "lr"}
    final, restored = replay(initial, cfg["resumed"], second)
    assert all(x["identical"] for x in restored.values())
    a = inspect_state(second / cfg["reference"].name, cfg["reference"], split, None, 2, devices)
    b = inspect_state(final, cfg["resumed"], split, None, 2, devices)
    assert a["updates"] == b["updates"] == 8
    for key in ("model", "optimizer", "scheduler", "rng", "history", "selections"):
        assert compare(a[key], b[key])["identical"], key
