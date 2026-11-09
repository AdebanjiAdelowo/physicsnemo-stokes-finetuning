"""Plot the checkpoint-selection decision and per-mesh inlet results."""

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT
from selection_decision import load
from stokes_ft.plotting import COLORS, GRID, INK, _save, plt

ORDER = ("pressure", "velocity", "joint", "final")
LABEL = {"pressure": "Pressure", "velocity": "Velocity", "joint": "Joint", "final": "Final"}
TRAIN = COLORS["data_driven"]
TEST = COLORS["physics_finetuned"]


def error_comparison(decision, out):
    table = decision["checkpoints"]
    baseline = decision["original_full_reference"]
    paired = decision["paired_against_final_on_test"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.8, 4.1), layout="constrained")
    candidates = [*ORDER, "original"]
    rows = [table[c] for c in ORDER] + [baseline]
    x = np.arange(len(rows))
    for delta, metric, color, label in ((-0.18, "train_pressure", TRAIN, "Train · 500 meshes"),
                                        (0.18, "test_pressure", TEST, "Test · 100 meshes")):
        left.bar(x + delta, [r[metric] for r in rows], width=0.34, color=color, label=label)
    left.axvline(3.5, color=GRID, linewidth=1)
    left.set_xticks(x, [f"{LABEL.get(c, 'Original')}\nepoch {r['epoch']}" for c, r in zip(candidates, rows)])
    left.set_ylabel("Mean pressure relative $L^2$")
    left.set_title("A  |  Training and held-out errors", loc="left")
    left.legend(loc="upper left", fontsize=8)
    left.set_ylim(0, max(r["test_pressure"] for r in rows) * 1.27)

    y = np.arange(3)
    for delta, metric, color, label in ((-0.13, "rel_l2_p", TRAIN, "Pressure"),
                                        (0.13, "rel_l2_velocity", TEST, "Velocity")):
        change = [paired[c][metric] for c in ORDER[:-1]]
        mean = np.array([r["absolute_change_mean"] for r in change])
        lo, hi = np.array([r["absolute_change_mean_ci95"] for r in change]).T
        right.errorbar(mean, y + delta, xerr=[mean - lo, hi - mean], fmt="o", color=color,
                       markersize=4, capsize=3, label=label)
    right.axvline(0, color=INK, linewidth=1, linestyle=":")
    right.set_yticks(y, [f"{LABEL[c]} · {table[c]['epoch']}" for c in ORDER[:-1]])
    right.set_ylim(2.7, -0.55)
    right.set_xlabel("Change in mean relative $L^2$ vs. epoch 500")
    right.set_title("B  |  Paired test differences (95% CI)", loc="left")
    right.legend(loc="best", fontsize=8)
    fig.suptitle("MeshGraphNet checkpoint selection", fontweight="bold")
    _save(fig, out, "checkpoint_errors.png")


def inlet_comparison(results, decision, out):
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7), sharex=True, sharey=True, layout="constrained")
    data = {c: load(results, decision["run"], c)["splits"]["test"]["per_mesh"] for c in ORDER}
    limit = max(max(r[k] for r in rows for k in ("ref_p_inlet", "pred_p_inlet", "recovered_p_inlet"))
                for rows in data.values()) * 1.05
    for ax, c in zip(axes.flat, ORDER):
        rows, table = data[c], decision["checkpoints"][c]
        ref = [r["ref_p_inlet"] for r in rows]
        ax.plot([0, limit], [0, limit], color=INK, lw=1, linestyle=":", label="Reference")
        ax.scatter(ref, [r["pred_p_inlet"] for r in rows], s=18, alpha=0.65, color=TRAIN, linewidths=0,
                   label=f"MeshGraphNet, $r$ = {table['test_inlet_correlation']:.2f}")
        ax.scatter(ref, [r["recovered_p_inlet"] for r in rows], s=17, alpha=0.62, color=TEST, linewidths=0,
                   marker="^", label=f"FEM recovery, $r$ = {table['test_recovered_inlet_level']['correlation']:.2f}")
        ax.set_title(f"{LABEL[c]} · epoch {table['epoch']}", loc="left")
        ax.set(xlim=(0, limit), ylim=(0, limit))
        ax.legend(loc="upper left", fontsize=7.5, frameon=True)
    fig.supxlabel("Reference inlet pressure")
    fig.supylabel("Estimated inlet pressure")
    fig.suptitle("Inlet pressure on 100 held-out geometries", fontweight="bold")
    _save(fig, out, "inlet_checkpoint_comparison.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/checkpoint_selection")
    parser.add_argument("--out", type=Path, default=ROOT / "figures/checkpoint_selection")
    args = parser.parse_args()
    decision = json.loads((args.results_dir / "selection_decision.json").read_text())
    error_comparison(decision, args.out)
    inlet_comparison(args.results_dir, decision, args.out)


if __name__ == "__main__":
    main()
