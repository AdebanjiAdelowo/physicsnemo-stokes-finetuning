"""Verified, atomic checkpoints of the data-driven stage, for continuation across sessions.

A checkpoint of epoch ``e`` in ``<run>/checkpoints/data_driven`` consists of

* ``checkpoint.0.<e>.pt`` and ``<Model>.0.<e>.mdlus``: PhysicsNeMo's files (model, optimiser,
  scheduler, AMP scaler, epoch);
* ``rng.0.<e>.pt``: the Python, NumPy, torch and CUDA random-number-generator states;
* ``history.0.<e>.csv``: the per-epoch history up to ``e``;
* ``manifest.0.<e>.json``: SHA-256 of each of the files above and the run's fingerprint.

All files are first written to a temporary directory and then moved into place, the manifest last,
so a checkpoint without a manifest is incomplete and is never loaded. The newest ``keep``
checkpoints are retained. Loading verifies every checksum of the newest checkpoint and falls back to
the next older one if a file is missing or corrupted; it refuses (raises) if the fingerprint
differs from the current run, because continuing would mix configurations, datasets or
architectures. Checkpoints written before this module existed (no manifest) are loaded only when no
manifest exists at all, and are reported as unverified.

The fingerprint covers the resolved configuration without local paths, device and run name, the simulation
numbers of the training and validation files, the SHA-256 of the normalisation statistics, and the
names and shapes of all model parameters.
"""

import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from physicsnemo.utils import load_checkpoint, save_checkpoint

from .data import sha256_of as sha256_file
from .provenance import write_json


class IncompatibleCheckpoint(RuntimeError):
    """The checkpoint belongs to a different configuration, dataset or architecture."""


class NoValidCheckpoint(RuntimeError):
    """Checkpoints exist but none passes verification."""


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def fingerprint(cfg, run_dir: Path, model, train_numbers, validation_numbers) -> dict:
    config = OmegaConf.to_container(cfg, resolve=True)
    for key in ("device", "name"):  # neither affects training
        config.pop(key, None)
    config.pop("data", None)  # local paths; the dataset is identified by the file numbers below
    return {
        "config_sha256": _digest(config),
        "seed": cfg.seed,
        "train_numbers_sha256": _digest(list(train_numbers)),
        "validation_numbers_sha256": _digest(list(validation_numbers)),
        "node_stats_sha256": sha256_file(Path(run_dir) / "node_stats.json"),
        "edge_stats_sha256": sha256_file(Path(run_dir) / "edge_stats.json"),
        "architecture_sha256": _digest([(k, list(v.shape)) for k, v in model.state_dict().items()]),
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }


def _rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])


def _epoch_of(path: Path) -> int:
    return int(path.name.split(".")[-2])


def manifests(ckpt_dir: Path) -> list[Path]:
    """Complete checkpoints, newest first."""
    return sorted(Path(ckpt_dir).glob("manifest.0.*.json"), key=_epoch_of, reverse=True)


def save(ckpt_dir: Path, epoch: int, model, optimizer, scheduler, scaler, history_csv: Path, fp: dict,
         keep: int = 3) -> Path:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_dir / f".tmp-{epoch}-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    save_checkpoint(str(tmp), models=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch)
    torch.save(_rng_state(), tmp / f"rng.0.{epoch}.pt")
    shutil.copy(history_csv, tmp / f"history.0.{epoch}.csv")
    files = {p.name: sha256_file(p) for p in sorted(tmp.iterdir())}
    if not any(n.startswith("checkpoint.") for n in files) or not any(n.endswith(".mdlus") for n in files):
        raise RuntimeError(f"PhysicsNeMo did not write the expected checkpoint files: {sorted(files)}")
    for name in files:
        os.replace(tmp / name, ckpt_dir / name)
    manifest = {"epoch": epoch, "files": files, "fingerprint": fp, "written_unix": time.time()}
    write_json(tmp / "manifest.json", manifest)
    os.replace(tmp / "manifest.json", ckpt_dir / f"manifest.0.{epoch}.json")  # the commit point
    shutil.rmtree(tmp, ignore_errors=True)
    _prune(ckpt_dir, keep)
    return ckpt_dir / f"manifest.0.{epoch}.json"


def _prune(ckpt_dir: Path, keep: int) -> None:
    kept = {_epoch_of(m) for m in manifests(ckpt_dir)[:keep]}
    for path in ckpt_dir.iterdir():
        if path.name.startswith(".tmp-"):
            shutil.rmtree(path, ignore_errors=True)
            continue
        parts = path.name.split(".")
        if len(parts) >= 3 and parts[-2].isdigit() and int(parts[-2]) not in kept:
            path.unlink()


def verify(manifest_path: Path, fp: dict | None) -> tuple[bool, str]:
    """(usable, reason). Raises IncompatibleCheckpoint if the fingerprint differs."""
    try:
        manifest = json.loads(Path(manifest_path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        return False, f"unreadable manifest: {error}"
    if fp is not None and manifest["fingerprint"] != fp:
        diff = sorted(k for k in fp if manifest["fingerprint"].get(k) != fp[k])
        raise IncompatibleCheckpoint(f"{manifest_path.name}: fingerprint differs in {diff}; refusing to continue")
    for name, digest in manifest["files"].items():
        path = Path(manifest_path).parent / name
        if not path.exists():
            return False, f"missing {name}"
        if sha256_file(path) != digest:
            return False, f"checksum mismatch in {name}"
    return True, "ok"


def load_latest(ckpt_dir: Path, fp: dict, model, optimizer, scheduler, scaler, device, log=print) -> dict:
    """Restore the newest verified checkpoint. Returns {"epoch", "history_csv", "verified", "skipped"}."""
    ckpt_dir = Path(ckpt_dir)
    found = manifests(ckpt_dir) if ckpt_dir.exists() else []
    skipped = []
    for manifest_path in found:
        ok, reason = verify(manifest_path, fp)
        if not ok:
            log(f"checkpoint {manifest_path.name} not usable ({reason}); trying an older one")
            skipped.append({"manifest": manifest_path.name, "reason": reason})
            continue
        epoch = _epoch_of(manifest_path)
        loaded = load_checkpoint(str(ckpt_dir), models=model, optimizer=optimizer, scheduler=scheduler,
                                 scaler=scaler, epoch=epoch, device=device)
        if loaded != epoch:
            raise NoValidCheckpoint(f"PhysicsNeMo loaded epoch {loaded}, expected {epoch}")
        _set_rng_state(torch.load(ckpt_dir / f"rng.0.{epoch}.pt", weights_only=False))
        log(f"resumed from verified checkpoint of epoch {epoch}")
        return {"epoch": epoch, "history_csv": ckpt_dir / f"history.0.{epoch}.csv", "verified": True, "skipped": skipped}
    if found:
        raise NoValidCheckpoint(f"no checkpoint in {ckpt_dir} passed verification: {skipped}")
    legacy = ckpt_dir.exists() and any(p.name.startswith("checkpoint.") for p in ckpt_dir.iterdir())
    epoch = load_checkpoint(str(ckpt_dir), models=model, optimizer=optimizer, scheduler=scheduler,
                            scaler=scaler, device=device) if legacy else 0
    if legacy:
        log(f"resumed from an UNVERIFIED legacy checkpoint of epoch {epoch} (written before manifests existed)")
    return {"epoch": epoch, "history_csv": None, "verified": not legacy, "skipped": skipped}
