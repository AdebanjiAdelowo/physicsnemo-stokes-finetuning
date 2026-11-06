"""Is the inlet pressure level a global functional of the obstacle? Written to
results/validation/geometry_pressure.json.

    python scripts/geometry_pressure.py

For every dataset file (evicted files are read from the checksummed archive, as in
audit_checks.py), the reference inlet pressure level is compared with a zero-parameter
lubrication estimate of the channel's pressure drop:

    dp_est = Q * 12 nu * integral over x of dx / (h_bottom(x)^3 + h_top(x)^3),   Q = (2/3) U H

where h_bottom, h_top are the open gaps between the walls and the obstacle on the vertical line
at x (h_bottom = H, h_top = 0 where there is no obstacle) and Q is the flow rate of the inlet
profile. Plane Poiseuille flow in a gap of height h has dp/dx = 12 nu q / h^3; two gaps in
parallel add conductances. The estimate uses the obstacle outline only. No fitting.

It also fits, on the 800 training files only, a linear model of the inlet level on this one
feature, and reports its error on the 200 held-out files. A predictor that cannot see the
obstacle can do no better than the training mean, the comparison of interest for the
15-layer MeshGraphNet, whose inlet nodes are 18 to 23 hops from the polygon.
"""

import json

import numpy as np

from _bootstrap import ROOT
from audit_checks import dataset_sources
from stokes_ft.audit import hop_distance
from stokes_ft.config import load_config
from stokes_ft.data import INFLOW, OUTFLOW, POLYGON, prepare_split, read_sample
from stokes_ft.provenance import write_json
from stokes_ft.variants import polygon_segments

H, U, NU = 0.4, 0.3, 0.01


def gaps(segments: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bottom and top open gaps on vertical lines; the full channel where the line misses the obstacle."""
    a, b = segments[:, 0], segments[:, 1]
    lo, hi = np.minimum(a[:, 0], b[:, 0]), np.maximum(a[:, 0], b[:, 0])
    bottom, top = np.full(len(xs), H), np.zeros(len(xs))
    for i, x in enumerate(xs):
        hit = (lo <= x) & (x <= hi) & (hi > lo)
        if not hit.any():
            continue
        t = (x - a[hit, 0]) / (b[hit, 0] - a[hit, 0])
        y = a[hit, 1] + t * (b[hit, 1] - a[hit, 1])
        bottom[i], top[i] = y.min(), H - y.max()
    return bottom, top


def lubrication_drop(segments: np.ndarray, length: float = 1.5, n: int = 3001) -> float:
    xs = np.linspace(0.0, length, n)
    bottom, top = gaps(segments, xs)
    conductance = bottom**3 + top**3
    q = 2.0 / 3.0 * U * H
    return float(q * 12 * NU * np.trapezoid(1.0 / conductance, xs))


def main() -> None:
    cfg = load_config("local")
    manifest = prepare_split(cfg)
    owner = {n: split for split in ("train", "validation", "test") for n in manifest[split]}
    rows = []
    for n, path, origin, _ in dataset_sources(cfg):
        if path is None:
            continue
        s = read_sample(path)
        m = s["marker"]
        segs = polygon_segments(s["coords"], s["triangles"], m)
        hops = hop_distance(len(m), s["triangles"], m == POLYGON)
        p = s["ref_p"].ravel()
        rows.append({"number": n, "split": owner[n], "source": origin,
                     "p_inlet": float(p[m == INFLOW].mean()), "p_outlet": float(p[m == OUTFLOW].mean()),
                     "dp_lubrication": lubrication_drop(segs), "inlet_hops": int(hops[m == INFLOW].min()),
                     "polygon_area_proxy": float(np.ptp(segs[..., 0]) * np.ptp(segs[..., 1]))})
    train = [r for r in rows if r["split"] == "train"]
    held = [r for r in rows if r["split"] != "train"]

    def arr(rs, key):
        return np.array([r[key] for r in rs])

    x_tr, y_tr = arr(train, "dp_lubrication"), arr(train, "p_inlet")
    x_te, y_te = arr(held, "dp_lubrication"), arr(held, "p_inlet")
    slope, intercept = np.polyfit(x_tr, y_tr, 1)
    pred = slope * x_te + intercept
    blind = np.full_like(y_te, y_tr.mean())
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))  # noqa: E731
    ss_tot = float(((y_te - y_te.mean()) ** 2).sum())
    out = {
        "files": len(rows), "train": len(train), "held_out": len(held),
        "read_from": {k: sum(r["source"] == k for r in rows) for k in ("disk", "archive")},
        "inlet_level_all": {"mean": float(arr(rows, "p_inlet").mean()), "std": float(arr(rows, "p_inlet").std(ddof=1)),
                            "min": float(arr(rows, "p_inlet").min()), "max": float(arr(rows, "p_inlet").max())},
        "outlet_level_rms_all": float(np.sqrt(np.mean(arr(rows, "p_outlet") ** 2))),
        "zero_parameter_estimate": {
            "note": "dp_lubrication against the reference inlet level, all files, no fitting",
            "correlation": float(np.corrcoef(arr(rows, "dp_lubrication"), arr(rows, "p_inlet"))[0, 1]),
            "ratio_median": float(np.median(arr(rows, "p_inlet") / arr(rows, "dp_lubrication"))),
            "rmse": rmse(arr(rows, "dp_lubrication"), arr(rows, "p_inlet")),
        },
        "fitted_on_train": {
            "slope": float(slope), "intercept": float(intercept),
            "held_out_rmse": rmse(pred, y_te), "held_out_r2": 1 - float(((pred - y_te) ** 2).sum()) / ss_tot,
            "blind_predictor_held_out_rmse": rmse(blind, y_te),
        },
        "inlet_hops": {"min": int(arr(rows, "inlet_hops").min()), "max": int(arr(rows, "inlet_hops").max()),
                       "share_above_15": float((arr(rows, "inlet_hops") > 15).mean())},
        "per_file": rows,
    }
    write_json(ROOT / "results" / "validation" / "geometry_pressure.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "per_file"}, indent=1))


if __name__ == "__main__":
    main()
