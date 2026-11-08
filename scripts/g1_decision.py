"""Apply the G1 decision rules, fixed in docs/EXPERIMENT_GAPS.md at commit 57926cb before any
mesh 2 to 9 was fine-tuned, to the completed local study. Written to results/validation/g1_decision.json.

    python scripts/g1_decision.py [--run runs/local] [--out results/validation/g1_decision.json]

Primary endpoint: velocity relative L2 error, fine-tuned against data-driven, paired per mesh,
over all 10 meshes including 0 and 1.

* "Generalises" if at least 9 of 10 meshes improve (two-sided sign test p = 0.0215; the
  document's "p at most 0.021" is this value rounded) AND the bootstrap 95 % interval of the
  mean per-mesh relative change lies entirely below 0.
* Otherwise "generalisation not established".

Falsification test of the receptive-field explanation: contradicted if the fine-tuned inlet
pressure level has slope >= 0.5 against the reference level AND correlation >= 0.8 over the
10 meshes. Secondary measures are descriptive. This script was committed before the G1 results
existed; it must not be edited to change a rule.
"""

import argparse
import csv
import json

import numpy as np

from _bootstrap import ROOT
from stokes_ft.data import INFLOW, read_sample
from stokes_ft.evaluate import paired_uncertainty
from stokes_ft.provenance import write_json

RULE_COMMIT = "57926cb"
SECONDARY = ["rel_l2_u", "rel_l2_v", "rel_l2_p", "rel_l2_total", "res_momentum", "res_continuity",
             "bc_inflow", "bc_noslip", "bc_outflow_pressure"]


def decide(before: np.ndarray, after: np.ndarray) -> dict:
    u = paired_uncertainty(before, after)
    improved = int((after < before).sum())
    low, high = u["relative_change_mean_ci95"]
    generalises = len(before) == 10 and improved >= 9 and high < 0
    return {"meshes": len(before), "improved": improved, "uncertainty": u,
            "conclusion": "GENERALISES across this test set (local, undertrained surrogate only)" if generalises
            else "GENERALISATION NOT ESTABLISHED: report per-mesh results, keep the two-mesh scoping"}


def falsification(ref_level: np.ndarray, tuned_level: np.ndarray) -> dict:
    slope = float(np.polyfit(ref_level, tuned_level, 1)[0])
    corr = float(np.corrcoef(ref_level, tuned_level)[0, 1])
    return {"slope": slope, "correlation": corr, "contradicted": bool(slope >= 0.5 and corr >= 0.8)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="runs/local")
    parser.add_argument("--out", default="results/validation/g1_decision.json")  # output path only; no rule
    args = parser.parse_args()
    run = ROOT / args.run
    with open(run / "per_sample.csv") as f:
        rows = list(csv.DictReader(f))
    by = {(int(r["index"]), r["stage"]): r for r in rows}
    meshes = sorted(i for (i, s) in by if s == "physics_finetuned")
    if meshes != list(range(10)):
        raise SystemExit(f"G1 needs all 10 meshes evaluated; found {meshes}")
    get = lambda stage, key: np.array([float(by[(i, stage)][key]) for i in meshes])  # noqa: E731
    primary = decide(get("data_driven", "rel_l2_velocity"), get("physics_finetuned", "rel_l2_velocity"))

    levels = {"reference": [], "data_driven": [], "physics_finetuned": []}
    for i in meshes:
        s = read_sample(run / "predictions" / f"graph_{i}.vtp")
        inlet = s["marker"] == INFLOW
        for stage, prefix in (("reference", "ref"), ("data_driven", "gnn"), ("physics_finetuned", "filtered")):
            levels[stage].append(float(s[f"{prefix}_p"][inlet].mean()))
    ref = np.array(levels["reference"])

    secondary = {}
    for key in SECONDARY:
        b, a = get("data_driven", key), get("physics_finetuned", key)
        entry = {"data_driven_mean": float(b.mean()), "physics_finetuned_mean": float(a.mean()), "improved": int((a < b).sum()),
                 "reference_floor_mean": float(get("reference", key).mean()) if key.startswith("res_") else None}
        if np.all(b > 0):
            entry["uncertainty"] = paired_uncertainty(b, a)
        secondary[key] = entry
    records = [json.loads((run / "finetune" / f"finetune_{i}.json").read_text()) for i in meshes]
    out = {
        "rules_fixed_at_commit": RULE_COMMIT, "run": args.run, "meshes": meshes,
        "primary_velocity_rel_l2": primary | {"per_mesh": {"data_driven": get("data_driven", "rel_l2_velocity").tolist(),
                                                            "physics_finetuned": get("physics_finetuned", "rel_l2_velocity").tolist()}},
        "falsification_inlet_level": falsification(ref, np.array(levels["physics_finetuned"]))
        | {"data_driven": falsification(ref, np.array(levels["data_driven"])), "levels": levels},
        "secondary": secondary,
        "finetune_runs": [{"index": r["index"], "seed": r["seed"], "iterations": r["iterations"], "seconds": r["seconds"],
                           "final_loss_finite": bool(np.isfinite(r["final"]["loss_total"]))} for r in records],
    }
    write_json(ROOT / args.out, out)
    print(json.dumps({k: out[k] for k in ("primary_velocity_rel_l2", "falsification_inlet_level")}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
