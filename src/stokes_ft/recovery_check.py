"""Cross-session recovery test of the data-driven stage (``scripts/recovery_check.py``).

Session one trains ``resumed`` and pauses it after ``pause_epoch`` (a verified checkpoint is
written even if the schedule would not write one), and trains ``reference`` without interruption.
Both run directories leave the session as tar files (``session_state.pack``). Session two, a
separate execution that receives them only through the persisted output, restores them
(``session_state.unpack``) and then:

1. **corruption drill**: on a copy of ``resumed``, appends bytes to the newest checkpoint; resuming
   must skip it, report the checksum mismatch and continue from the next older checkpoint;
2. **incompatibility drill**: on another copy, resuming with a different learning rate must raise
   ``IncompatibleCheckpoint``;
3. resumes ``resumed`` to the end;
4. trains the ``repeat`` runs without interruption in this session, to measure the run-to-run spread;
5. compares the per-epoch training loss and validation errors of ``resumed`` and each repeat with
   ``reference``.

Scatter-add is not bit-reproducible on CUDA, so equality is not required.

Criterion version 1 (fixed before the first CUDA execution, 2026-11-08; FAILED there): largest
relative difference from ``reference`` over all epochs and the four series below ``TOLERANCE``
(1e-2) or at most twice that of one repeat. It counted epochs before the pause, which an
interruption cannot affect, and estimated the spread from a single repeat; the failing 2.31 %
arose at epoch 3, before the pause. It is still computed and reported (``criterion_v1``).

Criterion version 2 (fixed before its first execution, after version 1 failed; the one used for
the pass/fail decision): both drills behave as stated; ``resumed`` restarts from the verified
checkpoint of ``pause_epoch``; and over the epochs after the pause only, the largest relative
difference of ``resumed`` from ``reference`` is below ``TOLERANCE`` or at most ``SPREAD_FACTOR``
(1.5) times the largest such difference of the ``REPEATS`` (3) uninterrupted repeats.
"""

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from . import checkpoint
from .engine import train_data_driven
from .study import run_study

TOLERANCE = 1e-2
SPREAD_FACTOR = 1.5
REPEATS = 3
SERIES = ("loss", "val_rel_l2_u", "val_rel_l2_v", "val_rel_l2_p")


def history(run_dir: Path) -> np.ndarray:
    with open(Path(run_dir) / "history.csv") as f:
        return np.array([[float(r[k]) for k in SERIES] for r in csv.DictReader(f)])


def max_rel_diff(a: np.ndarray, b: np.ndarray, from_epoch: int = 0) -> float:
    """Largest relative difference over epochs ``from_epoch + 1`` onwards and all series."""
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a[from_epoch:] - b[from_epoch:]) / np.maximum(np.abs(b[from_epoch:]), 1e-30)))


def first_session(resumed_cfg, reference_cfg, runs_root: Path, pause_epoch: int) -> dict:
    run_study(resumed_cfg, stages=["train"], runs_root=runs_root, stop_after_epoch=pause_epoch)
    run_study(reference_cfg, stages=["train"], runs_root=runs_root)
    status = json.loads((Path(runs_root) / resumed_cfg.name / "status.json").read_text())
    return {"resumed_status": status, "pause_epoch": pause_epoch,
            "reference_epochs": len(history(Path(runs_root) / reference_cfg.name))}


def _drill_copy(runs_root: Path, source: str, name: str) -> Path:
    dst = Path(runs_root) / name
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(Path(runs_root) / source, dst)
    return dst


def second_session(resumed_cfg, reference_cfg, repeat_cfgs, runs_root: Path, pause_epoch: int, device) -> dict:
    runs_root = Path(runs_root)
    ckpt = runs_root / resumed_cfg.name / "checkpoints" / "data_driven"
    restored = [m.name for m in checkpoint.manifests(ckpt)]

    corrupt = _drill_copy(runs_root, resumed_cfg.name, "drillcorrupt")
    with open(corrupt / "checkpoints" / "data_driven" / f"checkpoint.0.{pause_epoch}.pt", "ab") as f:
        f.write(b"corrupted on purpose")
    messages = []
    drill = train_data_driven(resumed_cfg, corrupt, device, resume=True, log=messages.append, stop=lambda e: True)
    fallback = next((m for m in messages if "not usable" in m), None)
    resumed_line = next((m for m in messages if m.startswith("resumed from verified checkpoint")), "")
    corruption = {"detected": fallback is not None and "checksum mismatch" in fallback, "message": fallback,
                  "resumed_line": resumed_line, "fell_back_to_older": f"epoch {pause_epoch}" not in resumed_line
                  and "verified checkpoint of epoch" in resumed_line, "status": drill.get("status")}

    other = _drill_copy(runs_root, resumed_cfg.name, "drillother")
    other_cfg = resumed_cfg.copy()
    other_cfg.lr = resumed_cfg.lr * 2
    try:
        train_data_driven(other_cfg, other, device, resume=True, log=lambda *_: None)
        incompatible = {"refused": False, "message": None}
    except checkpoint.IncompatibleCheckpoint as error:
        incompatible = {"refused": True, "message": str(error)}
    shutil.rmtree(corrupt), shutil.rmtree(other)

    run_study(resumed_cfg, stages=["train"], resume=True, runs_root=runs_root)
    record = json.loads((runs_root / resumed_cfg.name / "train_metrics.json").read_text())
    for repeat_cfg in repeat_cfgs:
        run_study(repeat_cfg, stages=["train"], runs_root=runs_root)
    ref = history(runs_root / reference_cfg.name)
    resumed_h = history(runs_root / resumed_cfg.name)
    repeat_h = [history(runs_root / c.name) for c in repeat_cfgs]
    resumed_diff = max_rel_diff(resumed_h, ref)
    repeat_diff = max_rel_diff(repeat_h[0], ref)
    resumed_after = max_rel_diff(resumed_h, ref, pause_epoch)
    repeats_after = [max_rel_diff(h, ref, pause_epoch) for h in repeat_h]
    mechanics = bool(corruption["detected"] and corruption["fell_back_to_older"] and incompatible["refused"]
                     and record["resumed_from_epoch"] == pause_epoch and record["resumed_from_verified_checkpoint"])
    criterion_v1 = {"resumed_vs_reference_all_epochs": resumed_diff, "first_repeat_vs_reference_all_epochs": repeat_diff,
                    "passed": mechanics and (resumed_diff < TOLERANCE or resumed_diff <= 2 * repeat_diff)}
    criterion_v2 = {"epochs_compared": list(range(pause_epoch + 1, len(ref) + 1)),
                    "resumed_vs_reference_after_pause": resumed_after,
                    "repeats_vs_reference_after_pause": repeats_after,
                    "threshold": max(TOLERANCE, SPREAD_FACTOR * max(repeats_after)),
                    "passed": mechanics and (resumed_after < TOLERANCE or resumed_after <= SPREAD_FACTOR * max(repeats_after))}
    out = {
        "checkpoints_restored": restored, "corruption_drill": corruption, "incompatibility_drill": incompatible,
        "resumed_from_epoch": record["resumed_from_epoch"],
        "resumed_from_verified_checkpoint": record["resumed_from_verified_checkpoint"],
        "resumed_vs_reference_max_relative_difference": resumed_diff,
        "repeat_vs_reference_max_relative_difference": repeat_diff,
        "tolerance": TOLERANCE, "spread_factor": SPREAD_FACTOR, "series": list(SERIES),
        "mechanics_passed": mechanics, "criterion_v1": criterion_v1, "criterion_v2": criterion_v2,
        "histories": {"reference": ref.tolist(), "resumed": resumed_h.tolist(),
                      "repeats": [h.tolist() for h in repeat_h]},
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    out["passed"] = criterion_v2["passed"]
    return out
