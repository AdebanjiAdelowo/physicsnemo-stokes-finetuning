"""Verdicts on the pre-registered predictions of the checkpoint-selection experiment (G13).

    python scripts/selection_decision.py [--run full_select]

Committed before the selected run produced any result. Inputs: the outputs of
``scripts/pressure_diagnostics.py`` for the final checkpoint (``checkpoints/data_driven``, epoch 500)
and for ``checkpoints/selected/{pressure,velocity,joint}`` of the run, each on train:500,
validation:10 and test:100. Writes ``results/validation/selection_decision.json``.

Predictions (docs/PRESSURE_INVESTIGATION.md, section 9, commit 172469f), made operational here:

* **H1, pressure overfitting.** SUPPORTED if the pressure-selected checkpoint's mean test pressure
  relative L2 error is lower than that of the final checkpoint of the same run AND its ratio of mean
  test to mean training pressure error is below 3.4 (the ratio of the original full run).
  CONTRADICTED if its test pressure error is not lower. UNRESOLVED if it is lower but the ratio is
  not below 3.4, or if the pressure criterion selected the final epoch itself (no earlier checkpoint).
* **H2, receptive-field limit.** SUPPORTED if the inlet-level correlation with the reference on the
  test meshes is below 0.5 for all four checkpoints; CONTRADICTED otherwise.
* **H3, velocity retention.** SUPPORTED if the velocity-selected checkpoint's mean test velocity
  relative L2 error is within 10 % of the final checkpoint's of the same run (|a - b| / b <= 0.10);
  CONTRADICTED otherwise. The original full run's value (0.121) is reported, not ruled on.

Uncertainty: paired differences against the final checkpoint over the 100 test meshes (bootstrap
95 % interval of the mean relative change, sign test), from ``stokes_ft.evaluate.paired_uncertainty``.
They describe variation across test geometries for one trained network, not across training runs.
"""

import argparse
import json

import numpy as np

from _bootstrap import ROOT
from stokes_ft.evaluate import paired_uncertainty
from stokes_ft.provenance import write_json

ORIGINAL_RATIO = 3.4
ORIGINAL_VELOCITY = 0.121
CANDIDATES = ("final", "pressure", "velocity", "joint")


def load(run: str, candidate: str) -> dict:
    folder = ROOT / "results" / "validation"
    pattern = f"pressure_diag_{run}_e*.json" if candidate == "final" else f"pressure_diag_{run}_selected_{candidate}_e*.json"
    found = [p for p in folder.glob(pattern) if not p.stem.endswith(("_finetuned", "_perturbation"))]
    if len(found) != 1:
        raise SystemExit(f"expected one file matching {pattern}, found {[p.name for p in found]}")
    return json.loads(found[0].read_text())


def metrics(d: dict) -> dict:
    sp = d["splits"]
    t, tr, va = sp["test"]["summary"], sp["train"]["summary"], sp["validation"]["summary"]
    m = lambda s, k: s[k]["mean"]  # noqa: E731
    return {
        "epoch": d["epoch"],
        "test_velocity": m(t, "rel_l2_velocity"), "test_pressure": m(t, "rel_l2_p"),
        "train_velocity": m(tr, "rel_l2_velocity"), "train_pressure": m(tr, "rel_l2_p"),
        "validation_velocity": m(va, "rel_l2_velocity"), "validation_pressure": m(va, "rel_l2_p"),
        "test_over_train_pressure": m(t, "rel_l2_p") / m(tr, "rel_l2_p"),
        "test_gauge_free_pressure": m(t, "pred_l2_p_gauge_free"), "test_pressure_gradient": m(t, "pred_grad_p"),
        "test_upstream_share": m(t, "pred_share_upstream"),
        "test_inlet_slope": t["pred_inlet_level"]["slope"], "test_inlet_correlation": t["pred_inlet_level"]["correlation"],
        "test_inlet_rms": t["pred_inlet_level"]["rms_error"],
        "train_inlet_correlation": tr["pred_inlet_level"]["correlation"],
        "test_recovered_pressure": m(t, "recovered_l2_p"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="full_select")
    args = parser.parse_args()
    data = {c: load(args.run, c) for c in CANDIDATES}
    numbers = {c: tuple(d["splits"]["test"]["numbers"]) for c, d in data.items()}
    if len(set(numbers.values())) != 1:
        raise SystemExit("the candidates were not evaluated on the same test meshes")
    table = {c: metrics(d) for c, d in data.items()}
    per_mesh = {c: {k: np.array([r[k] for r in d["splits"]["test"]["per_mesh"]]) for k in ("rel_l2_p", "rel_l2_velocity")}
                for c, d in data.items()}
    paired = {c: {k: paired_uncertainty(per_mesh["final"][k], per_mesh[c][k]) for k in ("rel_l2_p", "rel_l2_velocity")}
              for c in CANDIDATES[1:]}

    f, p, v = table["final"], table["pressure"], table["velocity"]
    if p["epoch"] == f["epoch"]:
        h1 = "UNRESOLVED"
    elif p["test_pressure"] >= f["test_pressure"]:
        h1 = "CONTRADICTED"
    else:
        h1 = "SUPPORTED" if p["test_over_train_pressure"] < ORIGINAL_RATIO else "UNRESOLVED"
    h2 = "SUPPORTED" if all(table[c]["test_inlet_correlation"] < 0.5 for c in CANDIDATES) else "CONTRADICTED"
    rel = abs(v["test_velocity"] - f["test_velocity"]) / f["test_velocity"]
    h3 = "SUPPORTED" if rel <= 0.10 else "CONTRADICTED"
    out = {
        "rules": __doc__.split("Predictions")[1].split("Uncertainty:")[0].strip(),
        "run": args.run, "test_meshes": len(numbers["final"]), "checkpoints": table, "paired_against_final_on_test": paired,
        "H1_pressure_overfitting": {"verdict": h1, "pressure_selected_epoch": p["epoch"],
                                    "test_pressure_selected": p["test_pressure"], "test_pressure_final": f["test_pressure"],
                                    "ratio_selected": p["test_over_train_pressure"], "ratio_threshold": ORIGINAL_RATIO},
        "H2_receptive_field": {"verdict": h2, "test_inlet_correlation": {c: table[c]["test_inlet_correlation"] for c in CANDIDATES}},
        "H3_velocity_retention": {"verdict": h3, "velocity_selected_epoch": v["epoch"], "relative_difference": rel,
                                  "test_velocity_selected": v["test_velocity"], "test_velocity_final": f["test_velocity"],
                                  "original_full_run_velocity": ORIGINAL_VELOCITY},
    }
    write_json(ROOT / "results" / "validation" / "selection_decision.json", out)
    for c in CANDIDATES:
        t = table[c]
        print(f"{c:9s} epoch {t['epoch']:3d} test vel {t['test_velocity']:.3f} p {t['test_pressure']:.3f} train p {t['train_pressure']:.3f} "
              f"ratio {t['test_over_train_pressure']:.2f} inlet r {t['test_inlet_correlation']:.2f} slope {t['test_inlet_slope']:.2f} "
              f"rms {t['test_inlet_rms']:.3f} grad {t['test_pressure_gradient']:.2f} gauge-free {t['test_gauge_free_pressure']:.3f}")
    print("H1", h1, "| H2", h2, "| H3", h3)


if __name__ == "__main__":
    main()
