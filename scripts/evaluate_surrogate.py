"""Evaluate a trained MeshGraphNet on many held-out meshes, with the FEM-consistent diagnostics.

    python scripts/evaluate_surrogate.py --run runs/local --config local --count 100
    python scripts/evaluate_surrogate.py --run runs/arch_global_latent_s0 --config arch --count 100 \
        model_variant=global_latent

Reads the checkpoint and the training normalisation of ``--run``; never writes into it. Test files
are taken from the disk when resident and otherwise from the downloaded archive after its SHA-256
is checked (an evicted file can block on read), into a temporary directory that is removed after.
Writes ``results/validation/surrogate_<run name>_test<count>.json`` with per-mesh errors,
boundary mismatches, inlet-level statistics, the CPU inference time, and the training and
accelerator-inference records of the run. Runs on the CPU. The model sees only the node and edge
inputs (positions, boundary markers, edge geometry); the reference enters only the metrics.
"""

import argparse
import json
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from physicsnemo.utils import load_checkpoint

from _bootstrap import ROOT
from stokes_ft.config import load_config
from stokes_ft.data import DATASET_ARCHIVE, DATASET_SHA256, INFLOW, make_dataset, prepare_split, read_sample, sha256_of
from stokes_ft.metrics import field_errors
from stokes_ft.model import build_meshgraphnet, count_parameters, receptive_field_hops
from stokes_ft.provenance import collect_metadata, write_json
from stokes_ft.weak import WeakDiagnostics, boundary_breakdown, pressure_levels

SF_DATALESS = 0x40000000


def stage_test_files(cfg, numbers, target: Path) -> dict:
    raw = ROOT / cfg.data.raw_dir
    archive = raw.parent / DATASET_ARCHIVE
    zf = None
    origins = {"disk": 0, "archive": 0}
    target.mkdir(parents=True)
    for n in numbers:
        src = raw / f"res_{n}.vtp"
        if not (os.stat(src).st_flags & SF_DATALESS):
            shutil.copy(src, target / src.name)
            origins["disk"] += 1
            continue
        if zf is None:
            if os.stat(archive).st_flags & SF_DATALESS or sha256_of(archive) != DATASET_SHA256:
                raise RuntimeError("an evicted test file needs the archive, which is missing or does not match its checksum")
            zf = zipfile.ZipFile(archive)
            members = {Path(m).name: m for m in zf.namelist()}
        (target / src.name).write_bytes(zf.read(members[src.name]))
        origins["archive"] += 1
    if zf:
        zf.close()
    return origins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cfg = load_config(args.config, args.overrides + [f"num_test_samples={args.count}"])
    run_dir = (ROOT / args.run).resolve()
    numbers = prepare_split(cfg)["test"][: args.count]

    model = build_meshgraphnet(cfg)
    epoch = load_checkpoint(str(run_dir / "checkpoints" / "data_driven"), models=model, device="cpu")
    model.eval()
    rows = []
    with tempfile.TemporaryDirectory(prefix="stokes-eval-") as tmp:
        tmp = Path(tmp)
        for name in ("node_stats.json", "edge_stats.json"):
            shutil.copy(run_dir / name, tmp / name)
        origins = stage_test_files(cfg, numbers, tmp / "data" / "dataset" / "test")
        local = load_config(args.config, args.overrides + [f"num_test_samples={args.count}", "data.split_dir=data/dataset"])
        dataset = make_dataset(local, "test", tmp, root=tmp)
        stats = dataset.node_stats
        for i, graph in enumerate(dataset):
            with torch.no_grad():
                start = time.perf_counter()
                pred = model(graph.x, graph.edge_attr, graph)
                seconds = time.perf_counter() - start
            s = read_sample(dataset.data_list[i])
            fields = {k: dataset.denormalize(pred[:, j:j + 1], stats[f"{k}_mean"], stats[f"{k}_std"]).numpy().astype(np.float64)
                      for j, k in enumerate("uvp")}
            ref = {k: s[f"ref_{k}"] for k in "uvp"}
            diag = WeakDiagnostics(s["coords"], s["triangles"], s["marker"], cfg.physics.nu, cfg.physics.u_max)
            row = {"index": i, "number": numbers[i], "nodes": len(s["coords"]), "inference_seconds_cpu": seconds}
            row |= {f"rel_l2_{k}": v for k, v in field_errors(fields, ref).items()}
            row |= diag.errors(fields, ref) | diag.residuals(fields)
            row |= boundary_breakdown(s["coords"], s["marker"], fields, cfg.physics.u_max)
            row |= {f"pred_{k}": v for k, v in pressure_levels(s["coords"], s["marker"], fields["p"]).items()}
            row |= {f"ref_{k}": v for k, v in pressure_levels(s["coords"], s["marker"], ref["p"]).items()}
            recovered = diag.recover_pressure(fields["u"], fields["v"])
            row["recovered_p_rel_l2"] = diag.errors(fields | {"p": recovered}, ref)["l2_p"]
            row["recovered_p_inlet"] = float(recovered[s["marker"] == INFLOW].mean())
            rows.append(row)

    def col(key):
        return np.array([r[key] for r in rows])

    x, y, rec = col("ref_p_inlet"), col("pred_p_inlet"), col("recovered_p_inlet")
    summary = {k: {"mean": float(col(k).mean()), "median": float(np.median(col(k))), "std": float(col(k).std(ddof=1))}
               for k in ("rel_l2_velocity", "rel_l2_p", "rel_l2_total", "h1semi_velocity", "l2_p_gauge_free",
                         "recovered_p_rel_l2", "weak_momentum_h1dual", "weak_continuity_l2",
                         "bc_inflow", "bc_wall", "bc_polygon", "outlet_p_rms", "inference_seconds_cpu")}
    level = {
        "slope_vs_reference": float(np.polyfit(x, y, 1)[0]), "correlation": float(np.corrcoef(x, y)[0, 1]),
        "rms_error": float(np.sqrt(np.mean((y - x) ** 2))), "pred_std": float(y.std(ddof=1)), "ref_std": float(x.std(ddof=1)),
        "recovered_slope_vs_reference": float(np.polyfit(x, rec, 1)[0]),
        "recovered_correlation": float(np.corrcoef(x, rec)[0, 1]),
        "recovered_rms_error": float(np.sqrt(np.mean((rec - x) ** 2))),
    }
    out = {
        "environment": collect_metadata(torch.device("cpu")), "run": str(run_dir.relative_to(ROOT)),
        "model_variant": cfg.get("model_variant", "baseline"), "parameters": count_parameters(model),
        "receptive_field_hops": receptive_field_hops(model), "checkpoint_epoch": epoch,
        "test_meshes": len(rows), "test_numbers": numbers, "files_read_from": origins,
        "summary": summary, "inlet_pressure_level": level,
        "training_record": json.loads((run_dir / "train_metrics.json").read_text()) if (run_dir / "train_metrics.json").exists() else None,
        "accelerator_inference": json.loads((run_dir / "predict_metrics.json").read_text()) if (run_dir / "predict_metrics.json").exists() else None,
        "per_mesh": rows,
    }
    name = run_dir.name
    path = ROOT / "results" / "validation" / f"surrogate_{name}_test{len(rows)}.json"
    write_json(path, out)
    print(json.dumps({k: out[k] for k in ("run", "model_variant", "parameters", "receptive_field_hops", "test_meshes",
                                          "files_read_from", "summary", "inlet_pressure_level")}, indent=1))


if __name__ == "__main__":
    main()
