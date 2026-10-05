"""Error and residual norms."""

import numpy as np


def relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    """||pred - target||_2 / ||target||_2 over all entries, as a fraction."""
    return float(np.linalg.norm(pred - target) / np.linalg.norm(target))


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else float("nan")


def field_errors(pred: dict, ref: dict) -> dict:
    """Relative L2 errors of one sample. ``pred`` and ``ref`` map u, v, p to (N, 1) arrays.

    ``velocity`` stacks u and v; ``total`` stacks u, v and p in their physical units.
    """
    out = {key: relative_l2(pred[key], ref[key]) for key in ("u", "v", "p")}
    out["velocity"] = relative_l2(np.hstack([pred["u"], pred["v"]]), np.hstack([ref["u"], ref["v"]]))
    out["total"] = relative_l2(
        np.hstack([pred["u"], pred["v"], pred["p"]]), np.hstack([ref["u"], ref["v"], ref["p"]])
    )
    return out
