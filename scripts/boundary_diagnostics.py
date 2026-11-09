"""Where and why the physics-informed fit loses the no-slip condition, from existing outputs only.

    python scripts/boundary_diagnostics.py --run runs/full --config full
    python scripts/boundary_diagnostics.py --run runs/local --config local

For every fine-tuned test mesh of the run:

* **sampling**: node counts per boundary type, and the share of interior collocation nodes within
  ``NEAR`` (= 3 median polygon edge lengths) of the polygon, where the mesh is refined;
* **loss balance**: the weighted loss terms of the upstream objective (data 1, inlet and no-slip 10,
  momentum 1, continuity 10) at the first and last logged iteration, from ``finetune/history_<i>.csv``;
* **location**: polygon corners are polygon boundary nodes whose two polygon boundary edges turn by
  more than ``CORNER_DEGREES`` (30); for the MeshGraphNet, the physics-informed fit and every
  completed variant in ``controls/``, the RMS velocity on polygon nodes within ``NEAR`` of a corner
  and on the others, and the share of the polygon's squared velocity error near corners.

Writes ``results/validation/boundary_diag_<run>.json``. The thresholds are fixed here, before the
results were computed.
"""

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

from _bootstrap import ROOT
from stokes_ft.config import load_config
from stokes_ft.data import INFLOW, OUTFLOW, POLYGON, WALL, mesh_edges, read_sample
from stokes_ft.provenance import write_json
from stokes_ft.variants import load_variant_fields

CORNER_DEGREES = 30.0
NEAR_EDGES = 3.0
WEIGHTS = {"u": 1, "v": 1, "p": 1, "u_in": 10, "v_in": 10, "u_noslip": 10, "v_noslip": 10, "mom_u": 1, "mom_v": 1, "cont": 10}


def polygon_corners(coords, triangles, marker):
    """Polygon boundary nodes where the boundary turns by more than CORNER_DEGREES; median edge length."""
    edges = np.sort(mesh_edges(triangles), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    poly = boundary[(marker[boundary[:, 0]] == POLYGON) & (marker[boundary[:, 1]] == POLYGON)]
    nbrs = defaultdict(list)
    for a, b in poly:
        nbrs[a].append(b), nbrs[b].append(a)
    corners = []
    for node, ns in nbrs.items():
        if len(ns) != 2:
            corners.append(node)
            continue
        d1, d2 = coords[node] - coords[ns[0]], coords[ns[1]] - coords[node]
        cos = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))
        if np.degrees(np.arccos(np.clip(cos, -1, 1))) > CORNER_DEGREES:
            corners.append(node)
    lengths = np.linalg.norm(coords[poly[:, 0]] - coords[poly[:, 1]], axis=1)
    return np.array(sorted(corners)), float(np.median(lengths))


def corner_split(coords, marker, corners, near, u, v):
    poly = np.where(marker == POLYGON)[0]
    dist = np.min(np.linalg.norm(coords[poly][:, None, :] - coords[corners][None, :, :], axis=2), axis=1)
    speed2 = (np.asarray(u).reshape(-1)[poly] ** 2 + np.asarray(v).reshape(-1)[poly] ** 2)
    close = dist <= near
    rms = lambda m: float(np.sqrt(speed2[m].mean())) if m.any() else float("nan")  # noqa: E731
    return {"rms_near_corners": rms(close), "rms_elsewhere": rms(~close), "rms_all": rms(np.ones_like(close)),
            "share_of_squared_error_near_corners": float(speed2[close].sum() / max(speed2.sum(), 1e-300)),
            "share_of_nodes_near_corners": float(close.mean())}


def weighted_terms(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    pick = lambda r: {k: WEIGHTS[k] * float(r[f"loss_{k}"]) for k in WEIGHTS}  # noqa: E731
    first, last = pick(rows[0]), pick(rows[-1])
    group = lambda d: {"data": d["u"] + d["v"] + d["p"], "inlet": d["u_in"] + d["v_in"],  # noqa: E731
                       "noslip": d["u_noslip"] + d["v_noslip"], "pde": d["mom_u"] + d["mom_v"] + d["cont"]}
    return {"first_iteration": int(rows[0]["iteration"]), "last_iteration": int(rows[-1]["iteration"]),
            "first": group(first), "last": group(last), "terms_last": last}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run = ROOT / args.run
    variants = sorted(p.name for p in (run / "controls").iterdir() if p.is_dir() and not p.name.endswith("probe")) \
        if (run / "controls").exists() else []
    rows = []
    for index in cfg.finetune.samples:
        s = read_sample(run / "predictions" / f"graph_{index}.vtp")
        coords, marker, tri = s["coords"], s["marker"], s["triangles"]
        corners, edge = polygon_corners(coords, tri, marker)
        near = NEAR_EDGES * edge
        poly = coords[marker == POLYGON]
        interior = coords[marker == 0]
        d_int = np.min(np.linalg.norm(interior[:, None, :] - poly[None, :, :], axis=2), axis=1)
        row = {"index": index, "nodes": len(coords), "interior": int((marker == 0).sum()),
               "polygon": int((marker == POLYGON).sum()), "wall": int((marker == WALL).sum()),
               "inlet": int((marker == INFLOW).sum()), "outlet": int((marker == OUTFLOW).sum()),
               "corners": len(corners), "median_polygon_edge": edge, "near_radius": near,
               "interior_share_near_polygon": float((d_int <= near).mean()),
               "location": {"meshgraphnet": corner_split(coords, marker, corners, near, s["gnn_u"], s["gnn_v"])}}
        if "filtered_u" in s:
            row["location"]["physics_finetuned"] = corner_split(coords, marker, corners, near, s["filtered_u"], s["filtered_v"])
        for name in variants:
            f = load_variant_fields(run, name, index)
            if f:
                row["location"][name] = corner_split(coords, marker, corners, near, f["u"], f["v"])
        hist = run / "finetune" / f"history_{index}.csv"
        if hist.exists():
            row["loss_balance"] = weighted_terms(hist)
        rows.append(row)

    def mean(get):
        vals = [get(r) for r in rows]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else None

    stages = sorted({k for r in rows for k in r["location"]})
    summary = {"interior_share_near_polygon": mean(lambda r: r["interior_share_near_polygon"]),
               "polygon_nodes": mean(lambda r: r["polygon"]), "interior_nodes": mean(lambda r: r["interior"]),
               "location": {st: {k: mean(lambda r, st=st, k=k: r["location"].get(st, {}).get(k))
                                 for k in ("rms_near_corners", "rms_elsewhere", "rms_all", "share_of_squared_error_near_corners",
                                           "share_of_nodes_near_corners")} for st in stages}}
    if all("loss_balance" in r for r in rows):
        summary["loss_balance"] = {when: {g: mean(lambda r, when=when, g=g: r["loss_balance"][when][g])
                                          for g in ("data", "inlet", "noslip", "pde")} for when in ("first", "last")}
    out = {"run": args.run, "config": args.config, "corner_degrees": CORNER_DEGREES, "near_edges": NEAR_EDGES,
           "weights": WEIGHTS, "summary": summary, "per_mesh": rows}
    write_json(ROOT / "results" / "validation" / f"boundary_diag_{run.name}.json", out)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
