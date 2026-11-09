"""Checkpoint selection on the validation split, with criteria fixed before any selected run.

Enabled by ``training.select_checkpoints: true`` (absent from the historical configurations, which
are therefore unchanged). After every epoch the validation relative L2 errors of the normalised
fields, ``e_u``, ``e_v``, ``e_p``, are scored by:

* ``velocity``: ``(e_u + e_v) / 2``;
* ``pressure``: ``e_p``;
* ``joint``: ``0.5 * (e_u + e_v) / 2 + 0.5 * e_p`` (equal weight of velocity and pressure).

When a score is strictly lower than the best so far (ties keep the earlier epoch), the model
weights of that epoch are written to ``checkpoints/selected/<criterion>/`` with ``selection.json``
(epoch, the three errors, the score), through a temporary directory that replaces the old one. The
final epoch is the fourth candidate and is the regular checkpoint. Only the validation split enters;
the test split is never loaded during training.

On resumption from epoch ``r``, a selection of an epoch after ``r`` (possible only after falling
back past a damaged checkpoint) belongs to epochs that are trained again; it is removed with a
warning, and that criterion is tracked again from ``r``; ``selection.json`` records the epoch from
which its criterion was tracked (``tracked_since``).
"""

import json
import os
import shutil
from pathlib import Path

from physicsnemo.utils import save_checkpoint

from .provenance import write_json

CRITERIA = {
    "velocity": lambda e: (e["u"] + e["v"]) / 2,
    "pressure": lambda e: e["p"],
    "joint": lambda e: 0.5 * (e["u"] + e["v"]) / 2 + 0.5 * e["p"],
}


def load_state(selected_dir: Path) -> dict:
    """Best score and epoch per criterion from the stored selections (empty when none)."""
    state = {}
    for name in CRITERIA:
        path = Path(selected_dir) / name / "selection.json"
        if path.exists():
            record = json.loads(path.read_text())
            state[name] = {"score": record["score"], "epoch": record["epoch"]}
    return state


def resume(selected_dir: Path, epoch: int, log=print) -> dict:
    """Selection state for training resumed after ``epoch`` (see the module docstring)."""
    state = load_state(selected_dir)
    for name, record in list(state.items()):
        if record["epoch"] > epoch:
            log(f"WARNING: selection '{name}' of epoch {record['epoch']} is after the restored epoch {epoch}; "
                f"removed, tracked again from epoch {epoch}")
            shutil.rmtree(Path(selected_dir) / name)
            del state[name]
            state[f"_since_{name}"] = epoch
    return state


def improved(state: dict, errors: dict) -> list[str]:
    """Criteria whose score for ``errors`` is strictly lower than the stored best."""
    return [name for name, score in CRITERIA.items() if name not in state or score(errors) < state[name]["score"]]


def update(selected_dir: Path, state: dict, epoch: int, errors: dict, model) -> list[str]:
    """Save the weights of ``epoch`` for every improved criterion; returns those criteria."""
    names = improved(state, errors)
    for name in names:
        target = Path(selected_dir) / name
        tmp = Path(selected_dir) / f".tmp-{name}-{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        save_checkpoint(str(tmp), models=model, epoch=epoch)
        record = {"criterion": name, "epoch": epoch, "errors": {k: float(errors[k]) for k in "uvp"},
                  "score": float(CRITERIA[name](errors)), "tracked_since": state.get(f"_since_{name}", 0)}
        write_json(tmp / "selection.json", record)
        old = Path(selected_dir) / f".old-{name}-{os.getpid()}"
        if target.exists():
            os.replace(target, old)
        os.replace(tmp, target)
        shutil.rmtree(old, ignore_errors=True)
        state[name] = {"score": record["score"], "epoch": epoch}
    return names
