"""Numerical checks of the scientific audit, written to results/audit_checks.json.

    python scripts/audit_checks.py [--samples 10] [--run runs/local]

1. Reference solver. A Taylor-Hood P2-P1 solve on each test mesh, with either natural outlet
   form, is compared with the FEniCS reference at the vertices; its time is the classical baseline.
2. Split integrity. The train, validation and test simulation numbers are disjoint, no polygon
   geometry occurs in more than one split, and the distance from each held-out polygon to the
   nearest training polygon. Evicted files are read from the checksummed archive.
3. Receptive field. Hop distances from the polygon to the inlet, the outlet and all nodes,
   against the number of message-passing layers of the MeshGraphNet.
4. Stored local study (needs the untracked run directory). For every test mesh: the
   data-driven inlet pressure level against the reference; for the fine-tuned meshes: the
   pressure offset of both stages and the mesh-weighted errors. With the checkpoint, the
   change of the prediction beyond the receptive field when the polygon inputs are perturbed.
"""

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

from _bootstrap import ROOT
from stokes_ft.audit import hop_distance, lumped_mass, nearest_geometry, p1_divergence_rms, pressure_offset, weighted_relative_l2
from stokes_ft.config import load_config
from stokes_ft.data import DATASET_ARCHIVE, DATASET_SHA256, INFLOW, OUTFLOW, POLYGON, SPLITS, prepare_split, raw_files, read_sample, sample_number, sha256_of
from stokes_ft.fem import FORMS, taylor_hood_solve
from stokes_ft.metrics import field_errors
from stokes_ft.model import build_meshgraphnet
from stokes_ft.provenance import collect_metadata, write_json

SF_DATALESS = 0x40000000  # macOS st_flags bit of a file whose content is not on the disk


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())}


def fields(sample, prefix):
    return {k: sample[f"{prefix}_{k}"] for k in ("u", "v", "p")}


def reference_solver(cfg, numbers):
    out = {}
    for form in FORMS:
        rows = []
        for n in numbers:
            s = read_sample(ROOT / cfg.data.raw_dir / f"res_{n}.vtp")
            fem = taylor_hood_solve(s["coords"], s["triangles"], cfg.physics.nu, cfg.physics.u_max, form)
            errors = field_errors(fem, fields(s, "ref"))
            rows.append({"number": n, "seconds": fem["seconds"], "dofs": fem["dofs"],
                         **{f"rel_l2_{k}": v for k, v in errors.items()}})
        out[form] = {"rel_l2_total": stats([r["rel_l2_total"] for r in rows]),
                     "rel_l2_v": stats([r["rel_l2_v"] for r in rows]),
                     "seconds": stats([r["seconds"] for r in rows]), "per_sample": rows}
    return out


def polygon_points(path) -> np.ndarray:
    s = read_sample(path)
    polygon = np.round(s["coords"][s["marker"] == POLYGON], 9)
    return polygon[np.lexsort(polygon.T)]


def dataset_sources(cfg):
    """Yield ``(number, path, origin)`` for every raw file, read from the disk when it is resident.

    A file evicted by a cloud provider (reading it can block) is taken from the downloaded archive
    instead, but only when that archive has the published checksum. A resident file is compared
    byte for byte with its archive member. Files with neither source are skipped and counted.
    """
    raw_dir = ROOT / cfg.data.raw_dir
    archive = raw_dir.parent / DATASET_ARCHIVE
    usable = archive.exists() and not (os.stat(archive).st_flags & SF_DATALESS) and sha256_of(archive) == DATASET_SHA256
    zf = zipfile.ZipFile(archive) if usable else None
    members = {Path(name).name: name for name in zf.namelist()} if zf else {}
    with tempfile.TemporaryDirectory() as tmp:
        for path in raw_files(raw_dir):
            n = sample_number(path)
            if not getattr(os.stat(path), "st_flags", 0) & SF_DATALESS:
                same = zf.read(members[path.name]) == path.read_bytes() if path.name in members else None
                yield n, path, "disk", same
            elif path.name in members:
                extracted = Path(tmp) / path.name
                extracted.write_bytes(zf.read(members[path.name]))
                yield n, extracted, "archive", None
                extracted.unlink()
            else:
                yield n, None, "missing", None
    if zf:
        zf.close()


def split_integrity(cfg, manifest):
    parts = {split: set(manifest[split]) for split in SPLITS}
    disjoint = all(not (parts[a] & parts[b]) for i, a in enumerate(SPLITS) for b in SPLITS[i + 1 :])
    owner = {n: split for split, numbers in parts.items() for n in numbers}
    polygons, origins, mismatched, skipped = {}, {"disk": 0, "archive": 0}, [], []
    for n, path, origin, same in dataset_sources(cfg):
        if path is None:
            skipped.append(n)
            continue
        origins[origin] += 1
        if same is False:
            mismatched.append(n)
        polygons[n] = polygon_points(path)
    geometries = {}
    for n, points in polygons.items():
        geometries.setdefault(hashlib.sha256(points.tobytes()).hexdigest(), []).append(n)
    repeated = [numbers for numbers in geometries.values() if len(numbers) > 1]
    across = [numbers for numbers in repeated if len({owner[n] for n in numbers}) > 1]
    scanned = set(polygons)
    out = {"splits_disjoint": disjoint, "files": len(owner),
           "geometry_scan": {"scanned": len(scanned), "skipped_not_resident": len(skipped), "read_from": origins,
                             "complete": not skipped and len(scanned) == len(owner),
                             "resident_files_differing_from_archive": mismatched,
                             "scanned_per_split": {split: len(parts[split] & scanned) for split in SPLITS}},
           "distinct_polygons": len(geometries), "repeated_polygon_groups": len(repeated), "groups_across_splits": across}
    train = {n: polygons[n] for n in parts["train"] if n in polygons}
    if train:
        # near duplicates: distance from each held-out polygon to the closest training polygon,
        # against the same distance between training polygons (the spread of the generator)
        held_out = {n: polygons[n] for split in ("validation", "test") for n in parts[split] if n in polygons}
        to_train = nearest_geometry(held_out, train)
        within_train = nearest_geometry(train, train)
        closest = sorted(to_train.items(), key=lambda item: item[1][1])[:5]
        out["nearest_training_polygon"] = {
            "metric": "symmetric Hausdorff distance between polygon node sets, domain units (channel height 0.4)",
            "held_out_to_train": stats([d for _, d in to_train.values()]),
            "train_to_other_train": stats([d for _, d in within_train.values()]),
            "closest_pairs": [{"held_out": n, "train": m, "distance": d} for n, (m, d) in closest],
        }
        used_train = {n: polygons[n] for n in manifest["train"][: cfg.num_training_samples] if n in polygons}
        used_test = {n: polygons[n] for n in manifest["test"][: cfg.num_test_samples] if n in polygons}
        if used_train and used_test:
            # the training files this configuration actually used, against the test meshes it evaluates
            used = nearest_geometry(used_test, used_train)
            out["nearest_training_polygon"]["configured_test_to_used_train"] = {
                "train_files": len(used_train),
                "per_test_mesh": [{"index": i, "number": n, "train": used[n][0], "distance": used[n][1]}
                                  for i, n in enumerate(used_test)],
                "distance": stats([d for _, d in used.values()]),
            }
    return out


def receptive_field(cfg, numbers):
    depth = len(build_meshgraphnet(cfg).processor.processor_layers) // 2
    rows = []
    for n in numbers:
        s = read_sample(ROOT / cfg.data.raw_dir / f"res_{n}.vtp")
        hops = hop_distance(len(s["coords"]), s["triangles"], s["marker"] == POLYGON)
        rows.append({"number": n, "inlet_min_hops": int(hops[s["marker"] == INFLOW].min()),
                     "outlet_min_hops": int(hops[s["marker"] == OUTFLOW].min()),
                     "share_of_nodes_beyond_depth": float((hops > depth).mean())})
    return {"message_passing_layers": depth, "per_sample": rows,
            **{key: stats([r[key] for r in rows]) for key in ("inlet_min_hops", "outlet_min_hops", "share_of_nodes_beyond_depth")}}


def stored_local_study(cfg, run_dir: Path):
    files = sorted((run_dir / "predictions").glob("graph_*.vtp"), key=lambda p: int(p.stem.split("_")[1]))
    inlet, evaluated = [], []
    for path in files:
        index = int(path.stem.split("_")[1])
        s = read_sample(path)
        inflow = s["marker"] == INFLOW
        mass = lumped_mass(s["coords"], s["triangles"])
        ref = fields(s, "ref")
        row = {"index": index, "ref": float(ref["p"][inflow].mean()), "data_driven": float(s["gnn_p"][inflow].mean())}
        if "filtered_p" in s:
            row["physics_finetuned"] = float(s["filtered_p"][inflow].mean())
        row.update({f"data_driven_rel_l2_{k}": v for k, v in field_errors(fields(s, "gnn"), ref).items()})
        row["reference_p1_divergence_rms"] = p1_divergence_rms(s["coords"], s["triangles"], ref["u"], ref["v"])
        inlet.append(row)
        if "filtered_u" in s:
            entry = {"index": index}
            for stage, prefix in (("data_driven", "gnn"), ("physics_finetuned", "filtered")):
                f = fields(s, prefix)
                entry[stage] = {
                    "p1_divergence_rms": p1_divergence_rms(s["coords"], s["triangles"], f["u"], f["v"]),
                    "pressure": pressure_offset(f["p"], ref["p"], mass),
                    "weighted_rel_l2_velocity": weighted_relative_l2(np.hstack([f["u"], f["v"]]), np.hstack([ref["u"], ref["v"]]), mass),
                    "weighted_rel_l2_p": weighted_relative_l2(f["p"], ref["p"], mass),
                    "nodal_rel_l2_velocity": field_errors(f, ref)["velocity"],
                }
            evaluated.append(entry)
    ref_level, gnn_level = np.array([r["ref"] for r in inlet]), np.array([r["data_driven"] for r in inlet])
    sensitivity = {}
    for stage in ("data_driven", "physics_finetuned"):
        tuned = [r for r in inlet if stage in r]
        if len(tuned) > 2:
            x, y = np.array([r["ref"] for r in tuned]), np.array([r[stage] for r in tuned])
            # slope 1: the level follows the obstacle as the reference does; slope 0: blind to it
            sensitivity[stage] = {"meshes": len(tuned), "slope_vs_reference": float(np.polyfit(x, y, 1)[0]),
                                  "correlation": float(np.corrcoef(x, y)[0, 1]), "std": float(y.std(ddof=1)),
                                  "rms_error": float(np.sqrt(np.mean((y - x) ** 2)))}
    out = {
        "run_dir": str(run_dir.relative_to(ROOT)),
        "inlet_pressure_level": {
            "per_sample": inlet,
            "across_mesh_std_ref": float(ref_level.std(ddof=1)),
            "across_mesh_std_data_driven": float(gnn_level.std(ddof=1)),
            "rms_error": float(np.sqrt(np.mean((gnn_level - ref_level) ** 2))),
            "sensitivity_to_geometry": sensitivity,
        },
        "data_driven_all_test_meshes": {k: stats([r[f"data_driven_rel_l2_{k}"] for r in inlet]) for k in ("velocity", "v", "p", "total")},
        "finetuned_meshes": evaluated,
    }
    records = [json.loads(p.read_text()) for p in sorted((run_dir / "finetune").glob("finetune_*.json"))]
    out["finetune_seconds"] = {r["index"]: r["seconds"] for r in records}
    ckpt = run_dir / "checkpoints" / "data_driven"
    if ckpt.exists():
        out["perturbation"] = perturbation_check(cfg, run_dir)
    return out


def perturbation_check(cfg, run_dir: Path):
    from physicsnemo.utils import load_checkpoint
    from stokes_ft.data import make_dataset

    dataset = make_dataset(cfg, "test", run_dir)
    model = build_meshgraphnet(cfg)
    load_checkpoint(str(run_dir / "checkpoints" / "data_driven"), models=model, device="cpu")
    model.eval()
    depth = len(model.processor.processor_layers) // 2
    torch.manual_seed(0)
    beyond, within = [], []
    for graph in dataset:
        marker = graph.marker.argmax(1).numpy()
        polygon = torch.as_tensor(marker == POLYGON)
        hops = _graph_hops(graph, marker == POLYGON)
        x = graph.x.clone()
        x[polygon, 2:] = x[polygon, 2:].roll(1, dims=1)  # relabel the polygon nodes
        x[polygon, :2] += 0.01 * torch.randn(int(polygon.sum()), 2)
        with torch.no_grad():
            change = (model(x, graph.edge_attr, graph) - model(graph.x, graph.edge_attr, graph)).abs().max(1).values.numpy()
        beyond.append(float(change[hops > depth].max()))
        within.append(float(change[hops <= depth].max()))
    return {"max_change_beyond_depth": max(beyond), "max_change_within_depth": stats(within)}


def _graph_hops(graph, sources):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import shortest_path

    n, (row, col) = graph.num_nodes, graph.edge_index.numpy()
    adjacency = coo_matrix((np.ones(len(row)), (row, col)), shape=(n, n)).tocsr()
    return shortest_path(adjacency, unweighted=True, directed=False, indices=np.flatnonzero(sources)).min(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=10, help="number of test meshes for checks 1 and 3")
    parser.add_argument("--run", default="runs/local", help="run directory of the stored study (check 4)")
    parser.add_argument("--config", default="local", help="configuration of that run")
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifest = prepare_split(cfg)
    numbers = manifest["test"][: args.samples]
    run_dir = (ROOT / args.run).resolve()
    out = {
        "environment": collect_metadata(torch.device("cpu")),
        "test_numbers": numbers,
        "reference_solver": reference_solver(cfg, numbers),
        "split_integrity": split_integrity(cfg, manifest),
        "receptive_field": receptive_field(cfg, numbers),
        "stored_local_study": stored_local_study(cfg, run_dir) if (run_dir / "predictions").exists() else "SKIPPED: run directory not present",
    }
    if isinstance(out["stored_local_study"], dict):
        fem = {r["number"]: r["seconds"] for r in out["reference_solver"]["grad"]["per_sample"]}
        seconds = out["stored_local_study"]["finetune_seconds"]
        ratios = [seconds[i] / fem[n] for i, n in enumerate(numbers) if i in seconds]
        if ratios:
            out["runtime_ratio_finetune_to_taylor_hood"] = {
                "note": "fine-tuning on MPS (machine shared with other jobs) against one direct solve on the CPU",
                "per_sample": {i: seconds[i] / fem[n] for i, n in enumerate(numbers) if i in seconds}, **stats(ratios)}
    write_json(ROOT / "results" / "audit_checks.json", out)
    rs = out["reference_solver"]
    for form in FORMS:
        print(f"Taylor-Hood '{form}': relative L2 to the reference {rs[form]['rel_l2_total']}, seconds {rs[form]['seconds']}")
    print("split integrity", out["split_integrity"])
    print("receptive field", out["receptive_field"])
    if isinstance(out["stored_local_study"], dict):
        s = out["stored_local_study"]
        print("inlet pressure level", {k: v for k, v in s["inlet_pressure_level"].items() if k != "per_sample"})
        print("data-driven, all test meshes", s["data_driven_all_test_meshes"])
        print("perturbation", s.get("perturbation"))


if __name__ == "__main__":
    main()
