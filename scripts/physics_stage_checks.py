"""Controlled comparison of the physics-informed stage and its variants, written to
results/validation/physics_stage.json.

    python scripts/physics_stage_checks.py [--run runs/local] [--config local] [--out results/validation/physics_stage.json]

For every test mesh of the run and every field available on it (reference, MeshGraphNet,
upstream physics-informed fit, and each completed variant in ``controls/``), three groups of
quantities are kept apart:

* solution errors against the verified reference (nodal and mesh-weighted, H1 seminorm,
  gauge-free pressure, inlet pressure level);
* residuals: the nodal least-squares residual used so far and the FEM-consistent weak residual,
  each next to the reference's own value (the floor of a vertex-only field);
* boundary mismatches by boundary type.

Paired comparisons (variant against MeshGraphNet, and physics against data-only on the meshes
where both exist) use ``evaluate.paired_uncertainty``. Variants are listed only for the meshes
on which they ran; nothing is extrapolated.
"""

import argparse
import json

import numpy as np

from _bootstrap import ROOT
from stokes_ft.config import load_config
from stokes_ft.data import boundary_masks, read_sample
from stokes_ft.evaluate import STATUS, paired_uncertainty
from stokes_ft.metrics import field_errors
from stokes_ft.physics import nodal_residuals, residual_norms
from stokes_ft.provenance import collect_metadata, write_json
from stokes_ft.variants import load_variant_fields
from stokes_ft.weak import WeakDiagnostics, boundary_breakdown, pressure_levels

import torch

METRICS = ["rel_l2_velocity", "rel_l2_p", "h1semi_velocity", "l2_p_gauge_free", "p_inlet_abs_error",
           "weak_momentum_h1dual", "weak_continuity_l2", "nodal_momentum", "nodal_continuity",
           "bc_inflow", "bc_wall", "bc_polygon", "recovered_p_rel_l2"]


def describe(sample, fields, ref, diag, cfg):
    interior = boundary_masks(sample["marker"])["interior"]
    row = {f"rel_l2_{k}": v for k, v in field_errors(fields, ref).items()}
    row |= diag.errors(fields, ref) | diag.residuals(fields)
    nodal = residual_norms(nodal_residuals(sample["coords"], sample["triangles"], fields, cfg.physics.nu), interior)
    row |= {"nodal_momentum": nodal["momentum"], "nodal_continuity": nodal["continuity"]}
    row |= boundary_breakdown(sample["coords"], sample["marker"], fields, cfg.physics.u_max)
    levels = pressure_levels(sample["coords"], sample["marker"], fields["p"])
    ref_levels = pressure_levels(sample["coords"], sample["marker"], ref["p"])
    row |= {"p_inlet": levels["p_inlet"], "p_inlet_abs_error": abs(levels["p_inlet"] - ref_levels["p_inlet"])}
    recovered = diag.recover_pressure(fields["u"], fields["v"])
    row["recovered_p_rel_l2"] = diag.errors(fields | {"p": recovered}, ref)["l2_p"]
    row["recovered_p_inlet"] = float(recovered[sample["marker"] == 1].mean())
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="runs/local")
    parser.add_argument("--config", default="local")
    parser.add_argument("--out", default="results/validation/physics_stage.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_dir = ROOT / args.run
    controls = sorted(p.name for p in (run_dir / "controls").iterdir()
                      if p.is_dir() and not p.name.endswith("_probe")) if (run_dir / "controls").exists() else []
    rows = []
    for path in sorted((run_dir / "predictions").glob("graph_*.vtp"), key=lambda p: int(p.stem.split("_")[1])):
        index = int(path.stem.split("_")[1])
        s = read_sample(path)
        ref = {k: s[f"ref_{k}"] for k in "uvp"}
        diag = WeakDiagnostics(s["coords"], s["triangles"], s["marker"], cfg.physics.nu, cfg.physics.u_max)
        candidates = {"reference": ref, "data_driven": {k: s[f"gnn_{k}"] for k in "uvp"}}
        if "filtered_u" in s:
            candidates["physics_finetuned"] = {k: s[f"filtered_{k}"] for k in "uvp"}
        for name in controls:
            fields = load_variant_fields(run_dir, name, index)
            if fields is not None:
                candidates[f"control_{name}"] = {k: fields[k] for k in "uvp"}
        for stage, fields in candidates.items():
            rows.append({"index": index, "stage": stage} | describe(s, fields, ref, diag, cfg))

    stages = list(dict.fromkeys(r["stage"] for r in rows))
    by = {(r["index"], r["stage"]): r for r in rows}

    def paired(a, b):
        """Paired change from stage a to stage b on the meshes where both exist."""
        meshes = sorted(i for (i, s) in by if s == b and (i, a) in by)
        out = {"meshes": meshes}
        if len(meshes) < 2:
            out["note"] = "fewer than 2 meshes: per-mesh values only"
        for m in METRICS:
            x = np.array([by[(i, a)][m] for i in meshes])
            y = np.array([by[(i, b)][m] for i in meshes])
            entry = {a: x.tolist(), b: y.tolist(), "improved": int((y < x).sum())}
            if len(meshes) >= 2 and np.all(x > 0):
                entry["uncertainty"] = paired_uncertainty(x, y)
            out[m] = entry
        return out

    comparisons = {f"data_driven->{s}": paired("data_driven", s) for s in stages if s not in ("reference", "data_driven")}
    for s in stages:
        if s.startswith("control_") and "physics_finetuned" in stages:
            comparisons[f"{s}->physics_finetuned"] = paired(s, "physics_finetuned")
    means = {s: {m: float(np.mean([r[m] for r in rows if r["stage"] == s])) for m in METRICS}
             | {"meshes": [r["index"] for r in rows if r["stage"] == s]} for s in stages}
    records = {}
    for name in controls:
        rec = [json.loads(p.read_text()) for p in sorted((run_dir / "controls" / name).glob("record_*.json"))]
        records[name] = {"meshes": [r["index"] for r in rec], "iterations": sorted({r["iterations"] for r in rec}),
                         "device": sorted({r["device"] for r in rec}),
                         "seconds_per_mesh_mean": float(np.mean([r["seconds"] for r in rec])) if rec else None}
    out = {"environment": collect_metadata(torch.device("cpu")), "run": args.run,
           "status": STATUS.get(cfg.name, "UNLABELLED") + f": MeshGraphNet of configs/{args.config}.yaml "
                     f"({cfg.num_training_samples} training meshes, {cfg.epochs} epochs)",
           "stages": stages, "means_over_available_meshes": means, "comparisons": comparisons,
           "control_runs": records, "per_mesh": rows}
    write_json(ROOT / args.out, out)
    for s in stages:
        m = means[s]
        print(f"{s:<28} n={len(m['meshes']):>2} vel {m['rel_l2_velocity']:.3f} p {m['rel_l2_p']:.3f} H1 {m['h1semi_velocity']:.3f} "
              f"pgf {m['l2_p_gauge_free']:.3f} inlet {m['p_inlet_abs_error']:.3f} weakmom {m['weak_momentum_h1dual']:.2e} "
              f"div {m['weak_continuity_l2']:.3f} nodal {m['nodal_momentum']:.3f} wall {m['bc_wall']:.4f} poly {m['bc_polygon']:.4f} in {m['bc_inflow']:.4f}")


if __name__ == "__main__":
    main()
