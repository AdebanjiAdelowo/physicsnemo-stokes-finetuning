"""Carry a run directory from one session to the next (``scripts/session_state.py``).

On Kaggle a session keeps only what is in ``/kaggle/working`` when it ends normally, and the next
session reads that output, read-only, as a kernel source under ``/kaggle/input``. ``pack`` writes
the whole run directory (checkpoints with their manifests, history, normalisation statistics,
metadata, predictions and fine-tuning records) to one tar file with a SHA-256 sidecar and a small
JSON summary. ``unpack`` checks the tar against its sidecar, refuses to overwrite an existing run,
extracts with tarfile's ``data`` filter (no absolute paths, links or escapes), and verifies every
checkpoint manifest. Whether the restored checkpoint belongs to the same configuration, dataset and
architecture is checked when training resumes (``checkpoint.load_latest``).
"""

import json
import os
import tarfile
from pathlib import Path

from . import checkpoint
from .data import sha256_of
from .provenance import write_json


def _verify_checkpoints(run_dir: Path) -> list[dict]:
    ckpt_dir = Path(run_dir) / "checkpoints" / "data_driven"
    found = checkpoint.manifests(ckpt_dir) if ckpt_dir.exists() else []
    return [{"manifest": m.name, "ok": ok, "reason": reason}
            for m in found for ok, reason in [checkpoint.verify(m, None)]]


def pack(run_dir: Path, out_tar: Path) -> dict:
    run_dir, out_tar = Path(run_dir), Path(out_tar)
    checks = _verify_checkpoints(run_dir)
    if checks and not checks[0]["ok"]:
        raise RuntimeError(f"the newest checkpoint of {run_dir} does not verify: {checks[0]}")
    partial = out_tar.with_name(out_tar.name + ".partial")
    with tarfile.open(partial, "w") as tar:
        tar.add(run_dir, arcname=run_dir.name)
    os.replace(partial, out_tar)
    digest = sha256_of(out_tar)
    Path(f"{out_tar}.sha256").write_text(f"{digest}  {out_tar.name}\n")
    status_path = run_dir / "status.json"
    summary = {"run": run_dir.name, "tar": out_tar.name, "sha256": digest, "bytes": out_tar.stat().st_size,
               "status": json.loads(status_path.read_text()) if status_path.exists() else None,
               "checkpoints": checks}
    write_json(Path(f"{out_tar}.json"), summary)
    return summary


def unpack(tar_path: Path, runs_root: Path) -> dict:
    tar_path, runs_root = Path(tar_path), Path(runs_root)
    expected = Path(f"{tar_path}.sha256").read_text().split()[0]
    found = sha256_of(tar_path)
    if found != expected:
        raise RuntimeError(f"{tar_path.name}: SHA-256 {found} does not match its sidecar {expected}")
    with tarfile.open(tar_path) as tar:
        names = tar.getnames()
        top = {n.split("/")[0] for n in names}
        if len(top) != 1:
            raise RuntimeError(f"{tar_path.name} holds more than one top-level directory: {sorted(top)}")
        run_dir = runs_root / top.pop()
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError(f"{run_dir} already exists; refusing to overwrite it")
        runs_root.mkdir(parents=True, exist_ok=True)
        tar.extractall(runs_root, filter="data")
    checks = _verify_checkpoints(run_dir)
    if checks and not checks[0]["ok"]:
        raise RuntimeError(f"the newest checkpoint restored from {tar_path.name} does not verify: {checks[0]}")
    return {"run_dir": str(run_dir), "sha256": found, "checkpoints": checks}
