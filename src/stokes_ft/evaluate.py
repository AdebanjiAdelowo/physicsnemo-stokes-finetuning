"""Identical evaluation of the data-driven and the physics-fine-tuned fields.

Both are read from the same ``predictions/graph_<i>.vtp`` file, which also holds the mesh and
the reference solution, so the two stages are compared on the same nodes of the same test mesh.
The reference solution goes through the same residual operators and gives their floor.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .data import boundary_masks, read_sample
from .finetune import build_finetune_network
from .metrics import field_errors
from .physics import autodiff_residuals, boundary_norms, nodal_residuals, residual_norms
from .provenance import write_json

STAGES = {"reference": "ref", "data_driven": "gnn", "physics_finetuned": "filtered"}
STATUS = {"smoke": "SMOKE", "local": "LOCAL/REDUCED", "full": "CUDA/FULL", "official": "CUDA/FULL"}

# metric key, label, unit of the summary table
TABLE = [
    ("rel_l2_velocity", "Velocity relative L2 error", "fraction"),
    ("rel_l2_u", "u relative L2 error", "fraction"),
    ("rel_l2_v", "v relative L2 error", "fraction"),
    ("rel_l2_p", "Pressure relative L2 error", "fraction"),
    ("rel_l2_total", "Total (u, v, p) relative L2 error", "fraction"),
    ("res_momentum", "Momentum residual, RMS", "field units"),
    ("res_continuity", "Divergence residual, RMS", "field units"),
    ("bc_boundary", "Boundary residual (inflow and no-slip), RMS", "field units"),
    ("bc_inflow", "Inflow residual, RMS", "field units"),
    ("bc_noslip", "No-slip residual, RMS", "field units"),
    ("bc_outflow_pressure", "Outlet pressure, RMS (not in the loss)", "field units"),
]


def stage_fields(sample: dict, prefix: str) -> dict | None:
    if f"{prefix}_u" not in sample:
        return None
    return {key: sample[f"{prefix}_{key}"] for key in ("u", "v", "p")}


def evaluate_fields(sample: dict, fields: dict, nu: float, u_max: float, reference: dict | None) -> tuple[dict, dict]:
    """Metrics of one field set on one mesh, and its nodal residual arrays."""
    interior = boundary_masks(sample["marker"])["interior"]
    residuals = nodal_residuals(sample["coords"], sample["triangles"], fields, nu)
    row = {}
    if reference is not None:
        row.update({f"rel_l2_{k}": v for k, v in field_errors(fields, reference).items()})
    row.update({f"res_{k}": v for k, v in residual_norms(residuals, interior).items()})
    row.update({f"bc_{k}": v for k, v in boundary_norms(sample["coords"], sample["marker"], fields, u_max).items()})
    return row, residuals


def evaluate_study(cfg, run_dir: Path, samples: list[int] | None = None, log=print) -> dict:
    """Evaluate the fine-tuned test samples of a run directory and write the result files.

    ``samples`` restricts the evaluation to a subset of ``cfg.finetune.samples``, for a run whose
    physics-informed stage is incomplete; the summary then records it as partial.
    """
    run_dir = Path(run_dir)
    samples = list(cfg.finetune.samples) if samples is None else list(samples)
    if not set(samples) <= set(cfg.finetune.samples):
        raise ValueError(f"samples {samples} are not all in finetune.samples of the configuration")
    nu, u_max = cfg.physics.nu, cfg.physics.u_max
    rows, plot_fields = [], {}
    for index in samples:
        path = run_dir / "predictions" / f"graph_{index}.vtp"
        sample = read_sample(path)
        reference = stage_fields(sample, "ref")
        interior = boundary_masks(sample["marker"])["interior"]
        if index in cfg.evaluation.plot_samples:
            plot_fields.update({f"s{index}_{k}": sample[k] for k in ("coords", "triangles", "marker")})
        for stage, prefix in STAGES.items():
            fields = stage_fields(sample, prefix)
            if fields is None:
                raise RuntimeError(f"{path} has no '{stage}' fields; run the missing stage first.")
            row, residuals = evaluate_fields(sample, fields, nu, u_max, None if stage == "reference" else reference)
            if stage == "physics_finetuned":
                state = run_dir / "checkpoints" / "finetune" / f"fourier_dnn_{index}.pt"
                if state.exists():
                    model = build_finetune_network("cpu")
                    model.load_state_dict(torch.load(state, map_location="cpu"))
                    exact = residual_norms(autodiff_residuals(model, sample["coords"], nu), interior)
                    row.update({f"autodiff_{k}": v for k, v in exact.items()})
            rows.append({"index": index, "file": path.name, "stage": stage, "nodes": len(sample["coords"]), **row})
            if index in cfg.evaluation.plot_samples:
                for key in ("u", "v", "p"):
                    plot_fields[f"s{index}_{stage}_{key}"] = fields[key].reshape(-1).astype(np.float32)
                for key, value in residuals.items():
                    plot_fields[f"s{index}_{stage}_{key}"] = value.astype(np.float32)

    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(run_dir / "per_sample.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(run_dir / "fields.npz", **plot_fields)

    summary = summarise(cfg, run_dir, rows)
    write_json(run_dir / "summary.json", summary)
    with open(run_dir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "unit", "reference", "data_driven", "physics_finetuned", "change", "relative_change", "samples_improved"])
        for key, label, unit in TABLE:
            c = summary["comparison"][key]
            writer.writerow([label, unit, c["reference"], c["data_driven"], c["physics_finetuned"], c["change"], c["relative_change"], f"{c['samples_improved']}/{summary['samples']}"])
    log(format_table(summary))
    return summary


def summarise(cfg, run_dir: Path, rows: list[dict]) -> dict:
    """Means over the samples per stage, and the paired change from data-driven to fine-tuned."""
    metrics = sorted({k for row in rows for k in row if k.split("_")[0] in ("rel", "res", "bc", "autodiff")})
    by_stage = {stage: [row for row in rows if row["stage"] == stage] for stage in STAGES}
    stages = {}
    for stage, stage_rows in by_stage.items():
        stages[stage] = {}
        for key in metrics:
            values = np.array([row[key] for row in stage_rows if key in row], dtype=float)
            if values.size:
                stages[stage][key] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else None,
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
    comparison = {}
    for key, _, _ in TABLE:
        before = np.array([row[key] for row in by_stage["data_driven"]])
        after = np.array([row[key] for row in by_stage["physics_finetuned"]])
        comparison[key] = {
            "reference": stages["reference"].get(key, {}).get("mean"),
            "data_driven": float(before.mean()),
            "physics_finetuned": float(after.mean()),
            "change": float(after.mean() - before.mean()),
            "relative_change": float(after.mean() / before.mean() - 1.0),
            "samples_improved": int((after < before).sum()),
        }

    def load(name):
        path = run_dir / name
        return json.loads(path.read_text()) if path.exists() else None

    finetune = [json.loads(p.read_text()) for p in sorted((run_dir / "finetune").glob("finetune_*.json"))]
    evaluated = [row["index"] for row in by_stage["data_driven"]]
    finetune = [r for r in finetune if r["index"] in evaluated]
    train, predict = load("train_metrics.json"), load("predict_metrics.json")
    compute = {
        "data_driven_parameters": train and train["parameter_count"],
        "data_driven_train_seconds": train and train["train_seconds"],
        "data_driven_inference_seconds_per_mesh": predict and predict["inference_seconds_per_mesh"],
        "data_driven_peak_cuda_memory_mb": train and train["peak_cuda_memory_mb"],
        "finetune_parameters_per_mesh": finetune[0]["parameter_count"] if finetune else None,
        "finetune_seconds_per_mesh": float(np.mean([r["seconds"] for r in finetune])) if finetune else None,
        "finetune_seconds_total": float(np.sum([r["seconds"] for r in finetune])) if finetune else None,
        "finetune_peak_cuda_memory_mb": max((r["peak_cuda_memory_mb"] or 0.0) for r in finetune) or None if finetune else None,
    }
    return {
        "study": cfg.name,
        "result_status": STATUS.get(cfg.name, "UNLABELLED"),
        "samples": len(evaluated),
        "sample_indices": evaluated,
        "samples_configured": len(cfg.finetune.samples),
        "partial": len(evaluated) < len(cfg.finetune.samples),
        "comparison": comparison,
        "stages": stages,
        "compute": compute,
    }


def format_table(summary: dict) -> str:
    lines = [
        f"study '{summary['study']}' [{summary['result_status']}], {summary['samples']} of "
        f"{summary['samples_configured']} configured test sample(s)",
        f"{'metric':<46}{'reference':>11}{'data-driven':>13}{'fine-tuned':>13}{'change':>9}{'improved':>10}",
    ]
    for key, label, _ in TABLE:
        c = summary["comparison"][key]
        ref = "" if c["reference"] is None else f"{c['reference']:.3e}"
        lines.append(
            f"{label:<46}{ref:>11}{c['data_driven']:>13.3e}{c['physics_finetuned']:>13.3e}"
            f"{100 * c['relative_change']:>+8.1f}%{c['samples_improved']:>7}/{summary['samples']}"
        )
    return "\n".join(lines)
