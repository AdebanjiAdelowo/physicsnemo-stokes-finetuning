"""Checks of the training pipeline on one device, run before a long study (``scripts/device_checks.py``).

* **Resume after an interruption**: a two-epoch run paused after epoch one (verified checkpoint,
  ``checkpoint.py``) and resumed must match an uninterrupted two-epoch run to rounding.
* **Repeatability**: two uninterrupted runs with the same seed must agree to rounding (scatter-add
  accumulation order is not fixed, so they are not required to be bit-for-bit identical).
* **Checkpoint round trip**: two models loaded from the same checkpoint have bit-identical weights,
  and their outputs agree to rounding (``OUTPUT_TOLERANCE``; scatter-add is not bit-reproducible on
  CUDA, so even one model called twice differs in the last bits; that difference is reported).
* **Gradients**: one backward pass from the loaded model gives finite, nonzero gradients for every
  parameter that requires one.

The numbers verify the pipeline; they are not results.
"""

import csv
import shutil
from pathlib import Path

import numpy as np
import torch
from physicsnemo.utils import load_checkpoint

from .data import make_dataset, prepare_split
from .engine import train_data_driven
from .model import build_meshgraphnet

TOLERANCE = 1e-3
OUTPUT_TOLERANCE = 1e-4  # max |difference| / max |output| between two loads of one checkpoint


def _history(run_dir: Path) -> dict:
    with open(run_dir / "history.csv") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in ("loss", "val_rel_l2_u", "val_rel_l2_v", "val_rel_l2_p")}


def _max_rel_diff(a: dict, b: dict) -> float:
    return float(max(np.max(np.abs(a[k] - b[k]) / np.maximum(np.abs(b[k]), 1e-30)) for k in b))


def run_device_checks(cfg, out_dir: Path, device: torch.device, log=print) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    prepare_split(cfg)
    two = cfg.copy()
    two.epochs, two.lr_decay_rate = 2, 0.99
    two.training.checkpoint_every = 1

    interrupted, uninterrupted, repeat = out_dir / "interrupted", out_dir / "uninterrupted", out_dir / "repeat"
    first = train_data_driven(two, interrupted, device, log=log, stop=lambda epoch: epoch >= 1)
    resumed = train_data_driven(two, interrupted, device, resume=True, log=log)
    reference = train_data_driven(two, uninterrupted, device, log=log)
    train_data_driven(two, repeat, device, log=log)
    h_int, h_ref, h_rep = _history(interrupted), _history(uninterrupted), _history(repeat)

    def load():
        model = build_meshgraphnet(two).to(device)
        epoch = load_checkpoint(str(uninterrupted / "checkpoints" / "data_driven"), models=model, device=device)
        return model, epoch

    a, epoch = load()
    b, _ = load()
    graph = next(iter(make_dataset(two, "train", uninterrupted))).to(device)
    a.eval(), b.eval()
    sa, sb = a.state_dict(), b.state_dict()
    weights_identical = sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    with torch.no_grad():
        ya, ya2, yb = (m(graph.x, graph.edge_attr, graph) for m in (a, a, b))
    # scatter-add accumulation order is not fixed on CUDA: even one model called twice is not
    # bit-identical, so outputs are compared to rounding and the difference is reported
    scale = float(ya.abs().max())
    output_diff = float((ya - yb).abs().max()) / scale
    repeat_call_diff = float((ya - ya2).abs().max()) / scale
    identical = weights_identical and output_diff < OUTPUT_TOLERANCE
    a.train()
    loss = torch.nn.functional.mse_loss(a(graph.x, graph.edge_attr, graph), graph.y)
    loss.backward()
    grads = [p.grad for p in a.parameters() if p.requires_grad]
    grads_finite = all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
    grad_norm = float(torch.sqrt(sum((g.double() ** 2).sum() for g in grads if g is not None)))

    finite = all(np.isfinite(v).all() for h in (h_int, h_ref, h_rep) for v in h.values())
    out = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "epochs": 2, "lr_decay_rate": 0.99, "tolerance": TOLERANCE,
        "interrupted_after_epoch": first["epochs_completed"], "first_status": first["status"],
        "resumed_flag": resumed["resumed"], "resumed_from_verified_checkpoint": resumed["resumed_from_verified_checkpoint"],
        "resume_epochs_in_history": len(h_int["loss"]),
        "resume_max_relative_difference": _max_rel_diff(h_int, h_ref),
        "repeat_max_relative_difference": _max_rel_diff(h_rep, h_ref),
        "checkpoint_epoch": epoch, "checkpoint_weights_identical": weights_identical,
        "checkpoint_output_max_relative_difference": output_diff,
        "same_model_repeat_call_max_relative_difference": repeat_call_diff,
        "checkpoint_round_trip_ok": identical,
        "gradients_finite": grads_finite, "gradient_norm": grad_norm,
        "all_losses_finite": finite,
        "losses": {k: v.tolist() for k, v in h_ref.items()},
        "peak_cuda_memory_mb": reference["peak_cuda_memory_mb"],
    }
    out["passed"] = bool(
        out["first_status"] == "paused" and out["resumed_flag"] and out["resumed_from_verified_checkpoint"]
        and out["resume_epochs_in_history"] == 2 and out["checkpoint_epoch"] == 2
        and out["resume_max_relative_difference"] < TOLERANCE and out["repeat_max_relative_difference"] < TOLERANCE
        and identical and grads_finite and grad_norm > 0 and finite
    )
    return out
