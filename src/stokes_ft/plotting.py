"""Figures of a study, drawn from its result files only."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.tri import Triangulation

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
COLORS = {"reference": "#52514e", "data_driven": "#2a78d6", "physics_finetuned": "#eb6834"}
LABELS = {"reference": "Reference (FEniCS)", "data_driven": "Data-driven (MeshGraphNet)",
          "physics_finetuned": "Physics fine-tuned"}
STAGES = list(LABELS)
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlesize": 9.5,
    "axes.titleweight": "bold", "legend.frameon": False,
})


def _rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def _field_axis(ax, title=None):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title)


def _sample(fields, index):
    tri = Triangulation(*fields[f"s{index}_coords"].T, fields[f"s{index}_triangles"])
    get = lambda stage, key: fields[f"s{index}_{stage}_{key}"].astype(float)
    return tri, get


def plot_fields(study_dir: Path, out_dir: Path, index: int, tag: str) -> Path:
    """Velocity magnitude and pressure of the three fields, and the absolute errors of the two models."""
    fields = np.load(study_dir / "fields.npz")
    tri, get = _sample(fields, index)
    speed = {s: np.hypot(get(s, "u"), get(s, "v")) for s in STAGES}
    pressure = {s: get(s, "p") for s in STAGES}
    models = STAGES[1:]
    speed_err = {s: np.hypot(get(s, "u") - get("reference", "u"), get(s, "v") - get("reference", "v")) for s in models}
    pressure_err = {s: np.abs(pressure[s] - pressure["reference"]) for s in models}

    fig, axes = plt.subplots(4, 3, figsize=(12.5, 6.6), constrained_layout=True)
    rows = [
        ("Velocity magnitude", speed, STAGES),
        ("Absolute velocity error", speed_err, models),
        ("Pressure", pressure, STAGES),
        ("Absolute pressure error", pressure_err, models),
    ]
    for r, (name, data, stages) in enumerate(rows):
        lo = min(0.0, min(v.min() for v in data.values()))
        hi = max(v.max() for v in data.values())
        for c, stage in enumerate(STAGES):
            ax = axes[r, c]
            _field_axis(ax, LABELS[stage] if r == 0 else None)
            if stage not in stages:
                marker = fields[f"s{index}_marker"]
                if r == 1:
                    ax.triplot(tri, color="#c3c2b7", linewidth=0.25)
                    for value, label, color in ((1, "inflow", "#2a78d6"), (2, "outflow", "#1baf7a"), (3, "wall", "#52514e"), (4, "polygon", "#eb6834")):
                        pick = marker == value
                        ax.plot(tri.x[pick], tri.y[pick], ".", ms=2.5, color=color, label=label)
                    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=7.5, handletextpad=0.1, columnspacing=0.8, markerscale=3)
                    ax.set_ylabel("Mesh and markers", color=MUTED)
                else:
                    ax.axis("off")
                continue
            im = ax.tripcolor(tri, data[stage], shading="gouraud", cmap=SEQUENTIAL, vmin=lo, vmax=hi, rasterized=True)
            if c == (0 if "reference" in stages else 1):
                ax.set_ylabel(name, color=MUTED)
        fig.colorbar(im, ax=axes[r, :], shrink=0.9, pad=0.01, aspect=12)
    fig.suptitle(f"Test mesh {index}: reference, data-driven prediction and physics fine-tuned field (same colour scale per row)", fontsize=10.5, x=0.01, ha="left")
    return _save(fig, out_dir, f"{tag}_fields_{index}.png")


def plot_residuals(study_dir: Path, out_dir: Path, index: int, tag: str) -> Path:
    """Nodal Stokes residuals (least-squares mesh derivatives) of the three fields, logarithmic scale."""
    fields = np.load(study_dir / "fields.npz")
    tri, get = _sample(fields, index)
    rows = [
        ("Momentum residual", {s: np.hypot(get(s, "momentum_x"), get(s, "momentum_y")) for s in STAGES}),
        ("Divergence residual", {s: np.abs(get(s, "continuity")) for s in STAGES}),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 3.5), constrained_layout=True)
    for r, (name, data) in enumerate(rows):
        hi = max(np.percentile(v, 99.5) for v in data.values())
        norm = LogNorm(vmin=hi * 1e-3, vmax=hi, clip=True)
        for c, stage in enumerate(STAGES):
            ax = axes[r, c]
            _field_axis(ax, LABELS[stage] if r == 0 else None)
            im = ax.tripcolor(tri, np.maximum(data[stage], hi * 1e-3), shading="gouraud", cmap=SEQUENTIAL, norm=norm, rasterized=True)
            if c == 0:
                ax.set_ylabel(name, color=MUTED)
        fig.colorbar(im, ax=axes[r, :], shrink=0.9, pad=0.01, aspect=10)
    fig.suptitle(f"Test mesh {index}: magnitude of the nodal Stokes residuals. The reference panel is the floor of the mesh derivative operator", fontsize=10.5, x=0.01, ha="left")
    return _save(fig, out_dir, f"{tag}_residuals_{index}.png")


def plot_comparison(study_dir: Path, out_dir: Path, tag: str) -> Path:
    """Before and after: per-sample values and means of the prediction errors and the residuals."""
    rows = _rows(study_dir / "per_sample.csv")
    summary = json.loads((study_dir / "summary.json").read_text())
    panels = [
        ("Relative $L^2$ error against the reference", [("rel_l2_u", "u"), ("rel_l2_v", "v"), ("rel_l2_velocity", "velocity"), ("rel_l2_p", "pressure")], False),
        ("RMS residual (nodal, least-squares derivatives)", [("res_momentum", "momentum"), ("res_continuity", "divergence"), ("bc_boundary", "boundary\n(inflow, no-slip)")], True),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.6), constrained_layout=True)
    for ax, (title, metrics, with_reference) in zip(axes, panels):
        for y, (key, label) in enumerate(metrics):
            stages = STAGES if with_reference and key != "bc_boundary" else STAGES[1:]
            for stage in stages:
                values = np.array([float(r[key]) for r in rows if r["stage"] == stage])
                offset = {"reference": 0.26, "data_driven": 0.0, "physics_finetuned": -0.26}[stage]
                ax.plot(values, np.full(len(values), y + offset), "o", ms=3.5, color=COLORS[stage], alpha=0.35, mec="none")
                ax.plot(values.mean(), y + offset, "D", ms=7.5, color=COLORS[stage], mec=SURFACE, mew=1.2,
                        label=LABELS[stage] if y == 0 else None)
                ax.annotate(f"{values.mean():.3g}", (values.max(), y + offset), xytext=(7, 0), textcoords="offset points",
                            va="center", fontsize=7.5, color=MUTED)
        ax.set_yticks(range(len(metrics)), [m[1] for m in metrics])
        ax.set_ylim(len(metrics) - 0.45, -0.55)
        ax.set_xscale("log")
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left")
        ax.margins(x=0.12)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8.5)
    fig.suptitle(f"Before and after physics-informed fine-tuning: {summary['samples']} test mesh(es), study '{summary['study']}' "
                 f"[{summary['result_status']}]. Diamonds are means, dots are single meshes", fontsize=10.5, x=0.01, ha="left")
    return _save(fig, out_dir, f"{tag}_comparison.png")


def plot_finetune_history(study_dir: Path, out_dir: Path, tag: str) -> Path:
    """Error of the fine-tuning network against the reference during its iterations."""
    rows = _rows(study_dir / "per_sample.csv")
    histories = {int(p.stem.split("_")[1]): _rows(p) for p in sorted((study_dir / "finetune").glob("history_*.csv"))}
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.1), constrained_layout=True, sharex=True)
    for ax, key in zip(axes, ("u", "v", "p")):
        curves = []
        for index, history in histories.items():
            it = np.array([int(r["iteration"]) for r in history])
            err = np.array([float(r[f"rel_l2_{key}"]) for r in history])
            ax.plot(it, err, color=COLORS["physics_finetuned"], lw=0.8, alpha=0.3)
            curves.append(err)
        ax.plot(it, np.mean(curves, axis=0), color=COLORS["physics_finetuned"], lw=2, label="Fine-tuning network, mean over meshes")
        base = np.mean([float(r[f"rel_l2_{key}"]) for r in rows if r["stage"] == "data_driven"])
        ax.axhline(base, color=COLORS["data_driven"], lw=2, ls=(0, (4, 2)), label="MeshGraphNet prediction (its target), mean")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title({"u": "u", "v": "v", "p": "pressure"}[key], loc="left")
        ax.set_xlabel("fine-tuning iteration")
    axes[0].set_ylabel("relative $L^2$ error against the reference")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8.5)
    return _save(fig, out_dir, f"{tag}_finetune_history.png")


def plot_training_history(study_dir: Path, out_dir: Path, tag: str) -> Path:
    history = _rows(study_dir / "history.csv")
    epoch = [int(r["epoch"]) for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.0), constrained_layout=True)
    axes[0].plot(epoch, [float(r["loss"]) for r in history], color=COLORS["data_driven"], lw=2)
    axes[0].set_title("Training loss (MSE of normalised u, v, p)", loc="left")
    for key, color in (("u", "#2a78d6"), ("v", "#1baf7a"), ("p", "#4a3aa7")):
        values = [float(r[f"val_rel_l2_{key}"]) for r in history]
        axes[1].plot(epoch, values, color=color, lw=2)
        axes[1].annotate({"u": "u", "v": "v", "p": "pressure"}[key], (epoch[-1], values[-1]), xytext=(5, 0),
                         textcoords="offset points", va="center", fontsize=8.5, color=MUTED)
    axes[1].set_title("Validation relative $L^2$ error (normalised fields)", loc="left")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
    return _save(fig, out_dir, f"{tag}_training_history.png")


def plot_study(study_dir: Path, out_dir: Path, tag: str | None = None) -> list[Path]:
    study_dir = Path(study_dir)
    summary = json.loads((study_dir / "summary.json").read_text())
    tag = tag or summary["study"]
    fields = np.load(study_dir / "fields.npz")
    indices = sorted({int(k.split("_")[0][1:]) for k in fields.files})
    paths = [plot_comparison(study_dir, out_dir, tag)]
    for index in indices:
        paths += [plot_fields(study_dir, out_dir, index, tag), plot_residuals(study_dir, out_dir, index, tag)]
    if any((study_dir / "finetune").glob("history_*.csv")):
        paths.append(plot_finetune_history(study_dir, out_dir, tag))
    if (study_dir / "history.csv").exists():
        paths.append(plot_training_history(study_dir, out_dir, tag))
    return paths
