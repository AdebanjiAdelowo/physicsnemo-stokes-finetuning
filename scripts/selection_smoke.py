"""Smoke check of validation-based checkpoint selection on one device, before a selected full run.

    python scripts/selection_smoke.py device=cuda

On the smoke configuration with ``training.select_checkpoints`` and four epochs, run ``sel_smoke``
paused after epoch 2 and resumed, and ``sel_smoke_ref`` without interruption. Checks: training loads
only the train and validation splits (the dataset constructor is wrapped and every split recorded);
for each criterion of ``stokes_ft.selection`` the selected epoch and score equal the minimum of the
criterion over the validation history (first of ties) in both runs; the selected and retained
checkpoints load; the newest retained checkpoints verify; nothing is written outside the two run
directories under ``runs/``. Writes ``runs/selection_smoke.json``; exits with status 1 on failure.
The numbers check the pipeline; they are not results.
"""

import csv
import json
import sys

import numpy as np
import torch
from physicsnemo.utils import load_checkpoint

from _bootstrap import ROOT
import stokes_ft.engine as engine
from stokes_ft import checkpoint, selection
from stokes_ft.config import load_config
from stokes_ft.model import build_meshgraphnet
from stokes_ft.provenance import collect_metadata, resolve_device, write_json
from stokes_ft.study import run_study


def main() -> None:
    overrides = sys.argv[1:]
    splits = []
    real = engine.make_dataset

    def recording(cfg, split, *a, **k):
        splits.append(split)
        return real(cfg, split, *a, **k)

    engine.make_dataset = recording
    runs = ROOT / "runs"
    before = {p.name for p in runs.iterdir()} if runs.exists() else set()
    base = overrides + ["epochs=4", "+training.select_checkpoints=true"]
    cfg = {name: load_config("smoke", base + [f"name={name}"]) for name in ("sel_smoke", "sel_smoke_ref")}
    for name in cfg:
        if (runs / name).exists():
            raise SystemExit(f"runs/{name} exists; remove it first")
    run_study(cfg["sel_smoke"], stages=["train"], stop_after_epoch=2)
    run_study(cfg["sel_smoke"], stages=["train"], resume=True)
    run_study(cfg["sel_smoke_ref"], stages=["train"])
    engine.make_dataset = real
    device = resolve_device(cfg["sel_smoke"].device)

    checks = {"training_splits": sorted(set(splits)), "only_train_and_validation": set(splits) <= {"train", "validation"}}
    for name, c in cfg.items():
        run_dir = runs / name
        with open(run_dir / "history.csv") as f:
            hist = [{k: float(r[f"val_rel_l2_{k}"]) for k in "uvp"} | {"epoch": int(r["epoch"])} for r in csv.DictReader(f)]
        per = {}
        for crit, score in selection.CRITERIA.items():
            record = json.loads((run_dir / "checkpoints" / "selected" / crit / "selection.json").read_text())
            scores = [score(h) for h in hist]
            model = build_meshgraphnet(c)
            loaded = load_checkpoint(str(run_dir / "checkpoints" / "selected" / crit), models=model, device=device)
            per[crit] = {"epoch": record["epoch"], "expected_epoch": hist[int(np.argmin(scores))]["epoch"],
                         "score": record["score"], "expected_score": float(min(scores)), "loaded_epoch": loaded,
                         "ok": record["epoch"] == hist[int(np.argmin(scores))]["epoch"] and loaded == record["epoch"]
                         and abs(record["score"] - min(scores)) < 1e-12}
        retained = [m.name for m in checkpoint.manifests(run_dir / "checkpoints" / "data_driven")]
        verified = all(checkpoint.verify(run_dir / "checkpoints" / "data_driven" / m, None)[0] for m in retained)
        checks[name] = {"selected": per, "retained": retained, "retained_verify": verified,
                        "status": json.loads((run_dir / "status.json").read_text())["state"]}
    a, b = checks["sel_smoke"]["selected"], checks["sel_smoke_ref"]["selected"]
    checks["selected_epochs_equal_after_resume"] = all(a[k]["epoch"] == b[k]["epoch"] for k in a)
    after = {p.name for p in runs.iterdir()}
    checks["new_entries_in_runs"] = sorted(after - before)
    checks["output_isolated"] = after - before == set(cfg)
    passed = (checks["only_train_and_validation"] and checks["output_isolated"]
              and all(v["ok"] for n in cfg for v in checks[n]["selected"].values())
              and all(checks[n]["retained_verify"] and checks[n]["status"] == "complete" for n in cfg))
    checks["selected_epochs_equal_after_resume_note"] = "reported only: on CUDA the two runs are not bit-identical"
    out = checks | {"passed": passed, "environment": collect_metadata(device),
                    "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    write_json(runs / "selection_smoke.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "environment"}, indent=1))
    print("SELECTION SMOKE PASSED" if passed else "SELECTION SMOKE FAILED")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
