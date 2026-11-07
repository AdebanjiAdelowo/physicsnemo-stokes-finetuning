"""Figures of the validation study, drawn from results/validation/*.json and runs/local.

    python scripts/validation_figures.py [--out figures/validation]
"""

import argparse
import json

import numpy as np

from _bootstrap import ROOT
from stokes_ft.data import read_sample
from stokes_ft.plotting import COLORS, GRID, INK, MUTED, SEQUENTIAL, _field_axis, _save, plt
from stokes_ft.variants import load_variant_fields
from matplotlib.tri import Triangulation

# categorical slots 1-4 of the validated palette: identity follows the method, never the rank
SERIES = {"data_driven": ("MeshGraphNet", COLORS["data_driven"], "o"),
          "physics_finetuned": ("Physics-informed fit", COLORS["physics_finetuned"], "s"),
          "control_data_only": ("Data-only control", "#1baf7a", "^"),
          "control_data_bc": ("Data + BC penalties", "#eda100", "D")}
RECOVERED = "#1baf7a"


def load(name):
    path = ROOT / "results" / "validation" / name
    return json.loads(path.read_text()) if path.exists() else None


def inlet_level(out):
    geo, sur = load("geometry_pressure.json"), load("surrogate_local_test100.json")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    ax = axes[0]
    rows = geo["per_file"]
    x = np.array([r["dp_lubrication"] for r in rows])
    y = np.array([r["p_inlet"] for r in rows])
    ax.scatter(x, y, s=10, color=COLORS["data_driven"], alpha=0.55, linewidths=0)
    f = geo["fitted_on_train"]
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, f["slope"] * xs + f["intercept"], color=INK, lw=1.2)
    ax.set_xlabel("Lubrication estimate of the pressure drop (no fitting)")
    ax.set_ylabel("Reference inlet pressure level")
    ax.set_title(f"All {geo['files']} files: r = {geo['zero_parameter_estimate']['correlation']:.2f}")
    ax.text(0.03, 0.95, f"line fitted on the 800 training files\nheld-out RMSE {f['held_out_rmse']:.3f} "
            f"(blind predictor {f['blind_predictor_held_out_rmse']:.3f})", transform=ax.transAxes, va="top", color=MUTED, fontsize=8)

    ax = axes[1]
    m = sur["per_mesh"]
    ref = np.array([r["ref_p_inlet"] for r in m])
    lim = (0, max(ref.max(), 1.0) * 1.05)
    ax.plot(lim, lim, color=GRID, lw=1.2, zorder=0)
    ax.scatter(ref, [r["pred_p_inlet"] for r in m], s=16, color=COLORS["data_driven"], marker="o",
               edgecolors="white", linewidths=0.6, label="MeshGraphNet prediction")
    ax.scatter(ref, [r["recovered_p_inlet"] for r in m], s=18, color=RECOVERED, marker="^",
               edgecolors="white", linewidths=0.6, label="Recovered from its velocity (FEM)")
    lv = sur["inlet_pressure_level"]
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Reference inlet pressure level")
    ax.set_ylabel("Inlet pressure level")
    ax.set_title(f"{sur['test_meshes']} test meshes: slope {lv['slope_vs_reference']:.1e} vs {lv['recovered_slope_vs_reference']:.2f}")
    ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("The inlet pressure level is a global functional of the obstacle", fontweight="bold", fontsize=10)
    fig.tight_layout()
    return _save(fig, out, "inlet_pressure_level.png")


def controls(out):
    ps = load("physics_stage.json")
    rows = ps["per_mesh"]
    metrics = [("rel_l2_velocity", "Velocity relative L2"), ("rel_l2_p", "Pressure relative L2"),
               ("bc_polygon", "No-slip RMS on the polygon")]
    meshes = sorted({r["index"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (key, label) in zip(axes, metrics):
        for k, (stage, (name, color, marker)) in enumerate(SERIES.items()):
            pts = [(r["index"], r[key]) for r in rows if r["stage"] == stage]
            if not pts:
                continue
            i, v = zip(*pts)
            ax.scatter(np.array(i) + (k - 1.5) * 0.12, v, s=26, color=color, marker=marker, edgecolors="white",
                       linewidths=0.6, label=name, zorder=3)
        ax.set_xticks(meshes)
        ax.set_xlabel("Test mesh")
        ax.set_title(label)
        ax.set_ylim(bottom=0)
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle("Physics-informed fit against controls with the same network, seed and iterations (local study)",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    return _save(fig, out, "physics_stage_controls.png")


def fields(out, index=0):
    run = ROOT / "runs" / "local"
    s = read_sample(run / "predictions" / f"graph_{index}.vtp")
    tri = Triangulation(*s["coords"].T, s["triangles"])
    stages = [("Reference", {k: s[f"ref_{k}"] for k in "uvp"}), ("MeshGraphNet", {k: s[f"gnn_{k}"] for k in "uvp"})]
    d = load_variant_fields(run, "data_only", index)
    if d:
        stages.append(("Data-only control", d))
    if "filtered_u" in s:
        stages.append(("Physics-informed fit", {k: s[f"filtered_{k}"] for k in "uvp"}))
    ref = stages[0][1]
    fig, axes = plt.subplots(len(stages), 3, figsize=(12, 1.55 * len(stages) + 0.4))
    speed_ref = np.hypot(ref["u"], ref["v"]).ravel()
    vmax_s, (pmin, pmax) = speed_ref.max(), (ref["p"].min(), ref["p"].max())
    for row, (name, f) in enumerate(stages):
        speed = np.hypot(f["u"], f["v"]).ravel()
        a = axes[row, 0].tripcolor(tri, speed, cmap=SEQUENTIAL, vmin=0, vmax=vmax_s, shading="gouraud")
        b = axes[row, 1].tripcolor(tri, f["p"].ravel(), cmap=SEQUENTIAL, vmin=pmin, vmax=pmax, shading="gouraud")
        err = np.abs(f["p"] - ref["p"]).ravel()
        c = axes[row, 2].tripcolor(tri, err, cmap="Greys", vmin=0, vmax=0.4, shading="gouraud")
        _field_axis(axes[row, 0], f"{name}: speed")
        _field_axis(axes[row, 1], f"{name}: pressure")
        _field_axis(axes[row, 2], f"{name}: |p - p_ref|")
    for ax, m in zip(axes[0], (a, b, c)):
        fig.colorbar(m, ax=axes[:, list(axes[0]).index(ax)], shrink=0.6, pad=0.01)
    fig.suptitle(f"Test mesh {index} (local study): the pressure error is a level error upstream of the obstacle",
                 fontweight="bold", fontsize=10)
    return _save(fig, out, f"fields_{index}.png")


def costs(out):
    c = load("costs.json")
    per = c["per_new_geometry_seconds"]
    items = [("Taylor-Hood FEM solve (CPU, 1 thread)", per["taylor_hood_fem_cpu_1_thread"]["value"]),
             ("MeshGraphNet inference (CPU)", per["meshgraphnet_inference_cpu"]["value"]),
             ("Data-only fit, 10k iterations (CPU)", per["data_only_fit_cpu"]["value"]),
             ("Physics-informed fit, 10k iterations (MPS)", per["physics_informed_fit_mps"]["value"]),
             ("Physics-informed fit, 10k iterations (CPU, extrapolated)", per["physics_informed_fit_cpu"]["value"])]
    fig, ax = plt.subplots(figsize=(8.2, 2.8))
    y = np.arange(len(items))[::-1]
    ax.barh(y, [v for _, v in items], height=0.55, color=COLORS["data_driven"])
    for yi, (_, v) in zip(y, items):
        ax.text(v * 1.15, yi, f"{v:,.0f} s" if v >= 10 else f"{v:.2f} s", va="center", color=INK, fontsize=8)
    ax.set_yticks(y, [n for n, _ in items])
    ax.set_xscale("log")
    ax.set_xlim(right=max(v for _, v in items) * 8)
    ax.set_xlabel("Seconds per new geometry (log scale); MeshGraphNet training (2,308 s MPS) excluded")
    ax.grid(axis="y", visible=False)
    ax.set_title("Cost per new geometry on this machine (shared; planning figures, not benchmarks)")
    fig.tight_layout()
    return _save(fig, out, "costs.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="figures/validation")
    out = ROOT / parser.parse_args().out
    for fn in (inlet_level, controls, fields, costs):
        try:
            print(fn(out))
        except (TypeError, FileNotFoundError, KeyError) as error:
            print(f"{fn.__name__}: skipped ({error!r})")


if __name__ == "__main__":
    main()
