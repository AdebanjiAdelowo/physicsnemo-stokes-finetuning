"""Verdicts on the pre-registered predictions of the checkpoint-selection experiment (G13).

    python scripts/selection_decision.py [--run full_select]

Committed before the selected run produced any result. Inputs: the outputs of
``scripts/pressure_diagnostics.py`` for the final checkpoint (``checkpoints/data_driven``, epoch 500)
and for ``checkpoints/selected/{pressure,velocity,joint}`` of the run, each on train:500,
validation:10 and test:100. Reads and writes in ``--results-dir`` (default ``results/validation``).

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

Uncertainty: seeded 95 % percentile intervals from 10,000 whole-mesh bootstrap replicates.
Test contrasts share mesh draws across checkpoints. Pressure gaps resample train and test separately,
keeping checkpoint pairing within each split. Field-error contrasts also report sign tests.
The intervals describe variation across geometries for one trained network.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT
from stokes_ft.evaluate import paired_uncertainty
from stokes_ft.provenance import write_json

ORIGINAL_RATIO = 3.4
ORIGINAL_VELOCITY = 0.121
CANDIDATES = ("final", "pressure", "velocity", "joint")
TEST_ERRORS = ("rel_l2_p", "rel_l2_velocity", "pred_l2_p_gauge_free", "pred_grad_p",
               "weak_momentum_h1dual", "weak_continuity_l2", "pressure_gradient_norm_ratio",
               "recovered_l2_p")


def load(folder: Path, run: str, candidate: str) -> dict:
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
        "test_minus_train_pressure": m(t, "rel_l2_p") - m(tr, "rel_l2_p"),
        "test_over_train_pressure": m(t, "rel_l2_p") / m(tr, "rel_l2_p"),
        "test_gauge_free_pressure": m(t, "pred_l2_p_gauge_free"), "test_pressure_gradient": m(t, "pred_grad_p"),
        "test_upstream_share": m(t, "pred_share_upstream"),
        "test_inlet_slope": t["pred_inlet_level"]["slope"], "test_inlet_correlation": t["pred_inlet_level"]["correlation"],
        "test_inlet_rms": t["pred_inlet_level"]["rms_error"],
        "train_inlet_correlation": tr["pred_inlet_level"]["correlation"],
        "test_recovered_pressure": m(t, "recovered_l2_p"),
        "test_recovered_inlet_level": t["recovered_inlet_level"],
        "test_lubrication_correlation": t["correlation_with_lubrication_estimate"],
        "test_error_correlation": t["per_mesh_error_correlation"],
    } | {
        key: m(t, key) for key in ("weak_momentum_h1dual", "weak_continuity_l2", "pressure_gradient_norm_ratio")
        if key in t
    }


def ordered_rows(data, split, numbers):
    part = data["splits"][split]
    rows = {row["number"]: row for row in part["per_mesh"]}
    if (len(rows) != len(part["per_mesh"]) or len(part["numbers"]) != len(numbers)
            or set(rows) != set(numbers) or set(part["numbers"]) != set(numbers)):
        raise SystemExit(f"{data['run']} epoch {data['epoch']} has different or duplicate {split} meshes")
    return [rows[number] for number in numbers]


def column(rows, name):
    values = np.array([row[name] for row in rows], dtype=float)
    if not np.isfinite(values).all():
        raise SystemExit(f"nonfinite {name} values")
    return values


def interval(values):
    return np.percentile(values, [2.5, 97.5]).tolist()


def inlet_metrics(ref, pred):
    """Vectorised RMS, OLS slope and Pearson r along the mesh dimension."""
    x = ref - ref.mean(axis=-1, keepdims=True)
    y = pred - pred.mean(axis=-1, keepdims=True)
    xx = (x * x).sum(axis=-1)
    yy = (y * y).sum(axis=-1)
    xy = (x * y).sum(axis=-1)
    return {"rms": np.sqrt(((pred - ref) ** 2).mean(axis=-1)), "slope": xy / xx,
            "correlation": xy / np.sqrt(xx * yy)}


def compare_test(before, after, index):
    paired = {}
    for key in TEST_ERRORS:
        if key not in before[0] or key not in after[0]:
            continue
        x, y = column(before, key), column(after, key)
        change = y - x
        paired[key] = paired_uncertainty(x, y) | {
            "absolute_change_mean": float(change.mean()),
            "absolute_change_mean_ci95": interval(change[index].mean(axis=1)),
            "ratio_of_means_change_ci95": interval(y[index].mean(axis=1) / x[index].mean(axis=1) - 1),
            "improved_meshes": int((change < 0).sum()),
        }
    ref = column(before, "ref_p_inlet")
    if not np.array_equal(ref, column(after, "ref_p_inlet")):
        raise SystemExit("reference inlet pressures differ across matched test meshes")
    x, y = column(before, "pred_p_inlet"), column(after, "pred_p_inlet")
    observed_x, observed_y = inlet_metrics(ref, x), inlet_metrics(ref, y)
    sampled_x, sampled_y = inlet_metrics(ref[index], x[index]), inlet_metrics(ref[index], y[index])
    paired["inlet_level"] = {
        key: {"change": float(observed_y[key] - observed_x[key]),
              "change_ci95": interval(sampled_y[key] - sampled_x[key])}
        for key in observed_x
    }
    return paired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="full_select")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "validation")
    args = parser.parse_args()
    data = {c: load(args.results_dir, args.run, c) for c in CANDIDATES}
    original = json.loads((ROOT / "results/validation/pressure_diag_full_e500.json").read_text())
    data["original_full_reference"] = original
    counts = {"test": 100, "train": 500, "validation": 10}
    numbers = {split: data["final"]["splits"][split]["numbers"] for split in counts}
    if any(len(set(numbers[split])) != count for split, count in counts.items()):
        raise SystemExit("expected 100 test, 500 train and 10 validation geometries")
    if len({n for group in numbers.values() for n in group}) != sum(counts.values()):
        raise SystemExit("the evaluation splits overlap")
    rows = {c: {split: ordered_rows(d, split, ids) for split, ids in numbers.items()} for c, d in data.items()}
    table = {c: metrics(d) for c, d in data.items()}
    rng = np.random.default_rng(0)
    test_index = rng.integers(0, counts["test"], (10000, counts["test"]))
    train_index = rng.integers(0, counts["train"], (10000, counts["train"]))
    paired = {c: compare_test(rows["final"]["test"], rows[c]["test"], test_index) for c in CANDIDATES[1:]}
    gaps = {}
    for c, row in table.items():
        test = column(rows[c]["test"], "rel_l2_p")[test_index].mean(axis=1)
        train = column(rows[c]["train"], "rel_l2_p")[train_index].mean(axis=1)
        gaps[c] = test - train
        ref_inlet = column(rows[c]["test"], "ref_p_inlet")[test_index]
        pred_inlet = column(rows[c]["test"], "pred_p_inlet")[test_index]
        row["test_inlet_ci95"] = {k: interval(v) for k, v in inlet_metrics(ref_inlet, pred_inlet).items()}
        row["test_minus_train_pressure_ci95"] = interval(gaps[c])
        row["test_over_train_pressure_ci95"] = interval(test / train)
    paired_gaps = {
        c: {"absolute_change": table[c]["test_minus_train_pressure"] - table["final"]["test_minus_train_pressure"],
            "absolute_change_ci95": interval(gaps[c] - gaps["final"])}
        for c in CANDIDATES[1:]
    }

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
        "run": args.run, "test_meshes": counts["test"],
        "checkpoints": {c: table[c] for c in CANDIDATES},
        "original_full_reference": table["original_full_reference"],
        "paired_against_final_on_test": paired,
        "pressure_gap_change_against_final": paired_gaps,
        "H1_pressure_overfitting": {"verdict": h1, "pressure_selected_epoch": p["epoch"],
                                    "test_pressure_selected": p["test_pressure"], "test_pressure_final": f["test_pressure"],
                                    "ratio_selected": p["test_over_train_pressure"], "ratio_threshold": ORIGINAL_RATIO},
        "H2_receptive_field": {"verdict": h2, "test_inlet_correlation": {c: table[c]["test_inlet_correlation"] for c in CANDIDATES}},
        "H3_velocity_retention": {"verdict": h3, "velocity_selected_epoch": v["epoch"], "relative_difference": rel,
                                  "test_velocity_selected": v["test_velocity"], "test_velocity_final": f["test_velocity"],
                                  "original_full_run_velocity": ORIGINAL_VELOCITY},
    }
    write_json(args.results_dir / "selection_decision.json", out)
    for c in CANDIDATES:
        t = table[c]
        print(f"{c:9s} epoch {t['epoch']:3d} test vel {t['test_velocity']:.3f} p {t['test_pressure']:.3f} train p {t['train_pressure']:.3f} "
              f"ratio {t['test_over_train_pressure']:.2f} inlet r {t['test_inlet_correlation']:.2f} slope {t['test_inlet_slope']:.2f} "
              f"rms {t['test_inlet_rms']:.3f} grad {t['test_pressure_gradient']:.2f} gauge-free {t['test_gauge_free_pressure']:.3f}")
    print("H1", h1, "| H2", h2, "| H3", h3)


if __name__ == "__main__":
    main()
