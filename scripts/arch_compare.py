"""Compare the receptive-field arms (G12) with the rule fixed in docs/VALIDATION.md section 3.3
before any arm was trained. Written to results/validation/arch_comparison.json.

    python scripts/arch_compare.py

Inputs: results/validation/surrogate_arch_<arm>_s<seed>_test100.json from
``scripts/evaluate_surrogate.py`` (first 100 test meshes, identical for every run).

Rule. Primary endpoints: the inlet-level slope against the reference (higher is better, 1 is
ideal) and the inlet-level RMS error (lower is better). Secondary: pressure and velocity relative
L2, gauge-free pressure error, velocity H1 error, boundary mismatches. For each metric, an arm is
**better** than the baseline only if the difference between its mean over seeds and the
baseline's mean over seeds exceeds the baseline's seed range (max - min over its seeds), in the
favourable direction; **worse** if it exceeds it in the other direction; otherwise
**indistinguishable**. A claim that an arm improves pressure prediction additionally requires
"better" on pressure relative L2, not only on the inlet-level correlation or slope. Costs are
reported, not ruled on. An arm with fewer than 3 seeds is reported as incomplete.
"""

import json

import numpy as np

from _bootstrap import ROOT
from stokes_ft.provenance import write_json

ARMS = ("baseline", "global_latent", "shared_depth")
SEEDS = (0, 1, 2)
# metric: (source, key, higher_is_better)
METRICS = {
    "inlet_slope": ("level", "slope_vs_reference", True),
    "inlet_rms_error": ("level", "rms_error", False),
    "inlet_correlation": ("level", "correlation", True),
    "rel_l2_p": ("summary", "rel_l2_p", False),
    "rel_l2_velocity": ("summary", "rel_l2_velocity", False),
    "l2_p_gauge_free": ("summary", "l2_p_gauge_free", False),
    "h1semi_velocity": ("summary", "h1semi_velocity", False),
    "bc_polygon": ("summary", "bc_polygon", False),
    "bc_wall": ("summary", "bc_wall", False),
    "bc_inflow": ("summary", "bc_inflow", False),
}
PRIMARY = ("inlet_slope", "inlet_rms_error")


def value(result, source, key):
    return result["inlet_pressure_level"][key] if source == "level" else result["summary"][key]["mean"]


def verdict(arm_values, base_values, higher_is_better):
    diff = float(np.mean(arm_values) - np.mean(base_values))
    spread = float(np.max(base_values) - np.min(base_values))
    favourable = diff if higher_is_better else -diff
    label = "better" if favourable > spread else "worse" if -favourable > spread else "indistinguishable"
    return {"difference_of_means": diff, "baseline_seed_range": spread, "verdict": label}


def main() -> None:
    results = {}
    for arm in ARMS:
        for seed in SEEDS:
            path = ROOT / "results" / "validation" / f"surrogate_arch_{arm}_s{seed}_test100.json"
            if path.exists():
                results[(arm, seed)] = json.loads(path.read_text())
    table = {}
    for arm in ARMS:
        seeds = [s for s in SEEDS if (arm, s) in results]
        entry = {"seeds": seeds, "complete": seeds == list(SEEDS)}
        for name, (source, key, _) in METRICS.items():
            entry[name] = [value(results[(arm, s)], source, key) for s in seeds]
        runs = [results[(arm, s)] for s in seeds]
        entry["parameters"] = sorted({r["parameters"] for r in runs})
        entry["receptive_field_hops"] = sorted({str(r["receptive_field_hops"]) for r in runs})
        entry["train_seconds_mps"] = [r["training_record"]["train_seconds"] for r in runs if r.get("training_record")]
        entry["inference_seconds_mps"] = [r["accelerator_inference"]["inference_seconds_per_mesh"] for r in runs if r.get("accelerator_inference")]
        entry["inference_seconds_cpu_mean"] = [r["summary"]["inference_seconds_cpu"]["mean"] for r in runs]
        entry["test_numbers_identical"] = len({tuple(r["test_numbers"]) for r in runs}) <= 1
        table[arm] = entry
    numbers = {tuple(r["test_numbers"]) for r in results.values()}
    comparisons = {}
    base = table["baseline"]
    for arm in ARMS[1:]:
        if not (table[arm]["complete"] and base["complete"]):
            comparisons[arm] = "INCOMPLETE: fewer than 3 seeds in an arm; no verdict"
            continue
        comparisons[arm] = {name: verdict(table[arm][name], base[name], hib) for name, (_, _, hib) in METRICS.items()}
        c = comparisons[arm]
        comparisons[arm]["primary_endpoints_better"] = all(c[m]["verdict"] == "better" for m in PRIMARY)
        comparisons[arm]["improves_pressure_prediction"] = c["rel_l2_p"]["verdict"] == "better"
    out = {"rule": __doc__.split("Rule.")[1].strip(), "same_test_meshes_in_every_run": len(numbers) <= 1,
           "arms": table, "comparisons_to_baseline": comparisons}
    write_json(ROOT / "results" / "validation" / "arch_comparison.json", out)
    for arm in ARMS:
        t = table[arm]
        if t["seeds"]:
            print(f"{arm:<14} seeds {t['seeds']} params {t['parameters']} slope {np.round(t['inlet_slope'], 3)} "
                  f"inlet rms {np.round(t['inlet_rms_error'], 3)} p {np.round(t['rel_l2_p'], 3)} vel {np.round(t['rel_l2_velocity'], 3)} "
                  f"train s {np.round(t['train_seconds_mps'])}")
    print(json.dumps(comparisons, indent=1)[:4000])


if __name__ == "__main__":
    main()
