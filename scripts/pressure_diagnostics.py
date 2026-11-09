"""Pressure diagnostics of a trained MeshGraphNet checkpoint, on the CPU, from existing weights only.

    python scripts/pressure_diagnostics.py --run runs/full --config full --epoch 500 \
        --splits test:100,validation:10,train:500
    python scripts/pressure_diagnostics.py --run runs/local --config local --epoch 40 --splits test:100,train:100

For every mesh of the requested splits: relative L2 errors of u, v, velocity, p; the pressure-error
decomposition of ``stokes_ft.pressure_diagnostics`` for the network's pressure and for the pressure
recovered from the network's velocity (Taylor-Hood momentum equation and outlet condition, no
reference input), and for the pressure recovered in the same way from the reference vertex velocity,
which is the floor of these vertex-based measures; inlet levels; the lubrication estimate of the geometry. ``--gradients N`` adds the
norms and cosine similarities of the per-component loss gradients over the first N training meshes,
by parameter group. Writes ``results/validation/pressure_diag_<run>_e<epoch>.json``; never writes
into the run. Files are staged as in ``evaluate_surrogate.py`` (from the verified archive when
evicted), all as the ``test`` split of a temporary directory, so that the normalisation
statistics of the run are read and never recomputed.
"""

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from physicsnemo.utils import load_checkpoint

from _bootstrap import ROOT
from evaluate_surrogate import stage_test_files
from stokes_ft.config import load_config
from stokes_ft.data import INFLOW, make_dataset, prepare_split, read_sample, sample_number
from stokes_ft.metrics import field_errors
from stokes_ft.model import build_meshgraphnet
from stokes_ft.pressure_diagnostics import decompose, pressure_stiffness
from stokes_ft.provenance import collect_metadata, write_json
from stokes_ft.weak import WeakDiagnostics


def evaluate_split(cfg, args, run_dir, model, split, numbers, lubrication):
    rows = []
    with tempfile.TemporaryDirectory(prefix=f"stokes-pdiag-{split}-") as tmp:
        tmp = Path(tmp)
        for name in ("node_stats.json", "edge_stats.json"):
            shutil.copy(run_dir / name, tmp / name)
        stage_test_files(cfg, numbers, tmp / "data" / "dataset" / "test")
        local = load_config(args.config, args.overrides + [f"num_test_samples={len(numbers)}", "data.split_dir=data/dataset"])
        dataset = make_dataset(local, "test", tmp, root=tmp)
        stats = dataset.node_stats
        order = [sample_number(p) for p in dataset.data_list]
        assert sorted(order) == sorted(numbers), "staged files differ from the requested split"
        for i, graph in enumerate(dataset):
            with torch.no_grad():
                pred = model(graph.x, graph.edge_attr, graph)
            s = read_sample(dataset.data_list[i])
            f = {k: dataset.denormalize(pred[:, j:j + 1], stats[f"{k}_mean"], stats[f"{k}_std"]).numpy().astype(np.float64)
                 for j, k in enumerate("uvp")}
            ref = {k: s[f"ref_{k}"] for k in "uvp"}
            diag = WeakDiagnostics(s["coords"], s["triangles"], s["marker"], cfg.physics.nu, cfg.physics.u_max)
            K = pressure_stiffness(diag)
            recovered = diag.recover_pressure(f["u"], f["v"])
            inlet = s["marker"] == INFLOW
            row = {"number": order[i], "nodes": len(s["coords"])}
            row |= {f"rel_l2_{k}": v for k, v in field_errors(f, ref).items()}
            row |= {f"pred_{k}": v for k, v in decompose(diag, f["p"], ref["p"], K).items()}
            row |= {f"recovered_{k}": v for k, v in decompose(diag, recovered, ref["p"], K).items()}
            # floor of the vertex-based measures: pressure recovered from the reference vertex velocity
            floor = diag.recover_pressure(ref["u"], ref["v"])
            row |= {f"floor_{k}": v for k, v in decompose(diag, floor, ref["p"], K).items()}
            row |= {"ref_p_inlet": float(ref["p"][inlet].mean()), "pred_p_inlet": float(f["p"][inlet].mean()),
                    "recovered_p_inlet": float(recovered[inlet].mean()), "dp_lubrication": lubrication.get(order[i])}
            rows.append(row)
    return rows


def summarise(rows):
    keys = [k for k in rows[0] if k not in ("number", "nodes", "dp_lubrication") and isinstance(rows[0][k], float)]
    out = {k: {"mean": float(np.mean([r[k] for r in rows])), "median": float(np.median([r[k] for r in rows]))} for k in keys}
    ref = np.array([r["ref_p_inlet"] for r in rows])
    for src in ("pred", "recovered"):
        y = np.array([r[f"{src}_p_inlet"] for r in rows])
        out[f"{src}_inlet_level"] = {"slope": float(np.polyfit(ref, y, 1)[0]), "correlation": float(np.corrcoef(ref, y)[0, 1]),
                                     "rms_error": float(np.sqrt(np.mean((y - ref) ** 2))), "std": float(y.std(ddof=1))}
    lub = np.array([r["dp_lubrication"] for r in rows], float)
    if np.isfinite(lub).all():
        out["correlation_with_lubrication_estimate"] = {
            src: float(np.corrcoef(lub, [r[f"{src}_p_inlet"] for r in rows])[0, 1]) for src in ("ref", "pred", "recovered")}
    out["per_mesh_error_correlation"] = {
        "velocity_vs_pressure": float(np.corrcoef([r["rel_l2_velocity"] for r in rows], [r["rel_l2_p"] for r in rows])[0, 1]),
        "velocity_vs_recovered_pressure": float(np.corrcoef([r["rel_l2_velocity"] for r in rows],
                                                            [r["recovered_l2_p"] for r in rows])[0, 1])}
    return out


def gradient_diagnostics(cfg, args, run_dir, model, numbers):
    """Per-component MSE loss gradients over training meshes: norms by parameter group and cosines."""
    groups = {"encoders": ("node_encoder", "edge_encoder"), "processor": ("processor",), "decoder": ("node_decoder",)}
    flat = {c: None for c in "uvp"}
    with tempfile.TemporaryDirectory(prefix="stokes-pdiag-grad-") as tmp:
        tmp = Path(tmp)
        for name in ("node_stats.json", "edge_stats.json"):
            shutil.copy(run_dir / name, tmp / name)
        stage_test_files(cfg, numbers, tmp / "data" / "dataset" / "test")
        local = load_config(args.config, args.overrides + [f"num_test_samples={len(numbers)}", "data.split_dir=data/dataset"])
        dataset = make_dataset(local, "test", tmp, root=tmp)
        model.train()
        for graph in dataset:
            pred = model(graph.x, graph.edge_attr, graph)
            for j, c in enumerate("uvp"):
                model.zero_grad()
                torch.mean((pred[:, j] - graph.y[:, j]) ** 2).backward(retain_graph=j < 2)
                g = torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])
                flat[c] = g.clone() if flat[c] is None else flat[c] + g
        model.eval()
    n = len(numbers)
    names = [name for name, p in model.named_parameters()]
    sizes = [p.numel() for p in model.parameters()]
    out = {"meshes": n, "norm_of_mean_gradient": {}, "cosine": {}}
    for c in "uvp":
        g = flat[c] / n
        parts = torch.split(g, sizes)
        out["norm_of_mean_gradient"][c] = {"all": float(g.norm())} | {
            grp: float(torch.sqrt(sum((part ** 2).sum() for name, part in zip(names, parts) if name.startswith(prefixes))))
            for grp, prefixes in groups.items()}
    for a, b in (("u", "v"), ("u", "p"), ("v", "p")):
        out["cosine"][f"{a}{b}"] = float(torch.nn.functional.cosine_similarity(flat[a], flat[b], dim=0))
    return out


def perturbation(cfg, args, run_dir, model, numbers):
    """Receptive-field check on the loaded weights: relabel and displace the polygon nodes' inputs and
    record the largest output change beyond the processor depth and at the inlet nodes."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import shortest_path

    from stokes_ft.data import POLYGON

    depth = len(model.processor.processor_layers) // 2
    torch.manual_seed(0)
    rows = []
    with tempfile.TemporaryDirectory(prefix="stokes-pdiag-pert-") as tmp:
        tmp = Path(tmp)
        for name in ("node_stats.json", "edge_stats.json"):
            shutil.copy(run_dir / name, tmp / name)
        stage_test_files(cfg, numbers, tmp / "data" / "dataset" / "test")
        local = load_config(args.config, args.overrides + [f"num_test_samples={len(numbers)}", "data.split_dir=data/dataset"])
        dataset = make_dataset(local, "test", tmp, root=tmp)
        for graph in dataset:
            marker = graph.marker.argmax(1).numpy()
            polygon = torch.as_tensor(marker == POLYGON)
            n, (r, c) = graph.num_nodes, graph.edge_index.numpy()
            adj = coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)).tocsr()
            hops = shortest_path(adj, unweighted=True, directed=False, indices=np.flatnonzero(marker == POLYGON)).min(0)
            x = graph.x.clone()
            x[polygon, 2:] = x[polygon, 2:].roll(1, dims=1)
            x[polygon, :2] += 0.01 * torch.randn(int(polygon.sum()), 2)
            with torch.no_grad():
                change = (model(x, graph.edge_attr, graph) - model(graph.x, graph.edge_attr, graph)).abs().max(1).values.numpy()
            inlet = marker == INFLOW
            rows.append({"beyond_depth": float(change[hops > depth].max()), "within_depth": float(change[hops <= depth].max()),
                         "inlet": float(change[inlet].max()), "inlet_min_hops": float(hops[inlet].min())})
    return {"depth": depth, "meshes": len(rows), "max_change_beyond_depth": max(r["beyond_depth"] for r in rows),
            "max_change_at_inlet": max(r["inlet"] for r in rows),
            "median_change_within_depth": float(np.median([r["within_depth"] for r in rows])), "per_mesh": rows}


def finetuned_comparison(cfg, run_dir):
    """The pressures available on the fine-tuned test meshes, decomposed against the reference:
    MeshGraphNet; recovered from its velocity; data-only control; physics-informed fit; recovered from
    the fit's velocity; floor (recovered from the reference velocity)."""
    from stokes_ft.variants import load_variant_fields

    rows = []
    for index in cfg.finetune.samples:
        s = read_sample(run_dir / "predictions" / f"graph_{index}.vtp")
        if "filtered_p" not in s:
            continue
        diag = WeakDiagnostics(s["coords"], s["triangles"], s["marker"], cfg.physics.nu, cfg.physics.u_max)
        K = pressure_stiffness(diag)
        inlet = s["marker"] == INFLOW
        sources = {"meshgraphnet": s["gnn_p"], "recovered_meshgraphnet": diag.recover_pressure(s["gnn_u"], s["gnn_v"]),
                   "physics_fit": s["filtered_p"], "recovered_physics_fit": diag.recover_pressure(s["filtered_u"], s["filtered_v"]),
                   "floor": diag.recover_pressure(s["ref_u"], s["ref_v"])}
        control = load_variant_fields(run_dir, "data_only", index)
        if control:
            sources["data_only_control"] = control["p"]
        row = {"index": index, "ref_p_inlet": float(s["ref_p"][inlet].mean())}
        for name, pp in sources.items():
            row[name] = decompose(diag, np.asarray(pp, float).reshape(-1, 1), s["ref_p"], K) | {
                "p_inlet": float(np.asarray(pp).reshape(-1)[inlet].mean())}
        rows.append(row)
    names = [k for k in rows[0] if k not in ("index", "ref_p_inlet")]
    summary = {n: {k: float(np.mean([r[n][k] for r in rows])) for k in rows[0][n]} for n in names}
    ref = np.array([r["ref_p_inlet"] for r in rows])
    for n in names:
        y = np.array([r[n]["p_inlet"] for r in rows])
        summary[n]["inlet_slope"] = float(np.polyfit(ref, y, 1)[0])
        summary[n]["inlet_correlation"] = float(np.corrcoef(ref, y)[0, 1])
    return {"meshes": len(rows), "summary": summary, "per_mesh": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--splits", default="test:100")
    parser.add_argument("--gradients", type=int, default=0)
    parser.add_argument("--perturbation", type=int, default=0, help="receptive-field check on the first N test meshes")
    parser.add_argument("--finetuned", action="store_true", help="decompose the pressures of the fine-tuned test meshes")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cfg = load_config(args.config, args.overrides)
    run_dir = (ROOT / args.run).resolve()
    split = prepare_split(cfg)
    used = {"train": split["train"][: cfg.num_training_samples], "validation": split["validation"][: cfg.num_validation_samples],
            "test": split["test"]}
    assert not set(used["train"]) & set(split["test"]) and not set(used["validation"]) & set(split["test"])
    geo = json.loads((ROOT / "results" / "validation" / "geometry_pressure.json").read_text())
    lubrication = {r["number"]: r["dp_lubrication"] for r in geo["per_file"]}

    model = build_meshgraphnet(cfg)
    epoch = load_checkpoint(str(run_dir / "checkpoints" / "data_driven"), models=model, epoch=args.epoch, device="cpu")
    if epoch != args.epoch:
        raise SystemExit(f"checkpoint of epoch {args.epoch} not found (loaded {epoch})")
    model.eval()
    out = {"environment": collect_metadata(torch.device("cpu")), "run": str(run_dir.relative_to(ROOT)), "config": args.config,
           "epoch": epoch, "splits": {}}
    for item in filter(None, args.splits.split(",")):
        name, count = item.split(":")
        numbers = used[name][: int(count)]
        start = time.perf_counter()
        rows = evaluate_split(cfg, args, run_dir, model, name, numbers, lubrication)
        out["splits"][name] = {"meshes": len(rows), "numbers": numbers, "seconds": time.perf_counter() - start,
                               "summary": summarise(rows), "per_mesh": rows}
        sm = out["splits"][name]["summary"]
        print(f"{name:10s} n={len(rows)} vel {sm['rel_l2_velocity']['mean']:.3f} p {sm['rel_l2_p']['mean']:.3f} "
              f"gauge-free {sm['pred_l2_p_gauge_free']['mean']:.3f} grad {sm['pred_grad_p']['mean']:.3f} "
              f"upstream {sm['pred_share_upstream']['mean']:.2f} | recovered p {sm['recovered_l2_p']['mean']:.3f} "
              f"grad {sm['recovered_grad_p']['mean']:.3f}", flush=True)
    if args.gradients:
        out["gradients"] = gradient_diagnostics(cfg, args, run_dir, model, used["train"][: args.gradients])
        print(json.dumps(out["gradients"], indent=1))
    if args.perturbation:
        out["perturbation"] = perturbation(cfg, args, run_dir, model, used["test"][: args.perturbation])
        print({k: v for k, v in out["perturbation"].items() if k != "per_mesh"})
    if args.finetuned:
        out["finetuned"] = finetuned_comparison(cfg, run_dir)
        for n, v in out["finetuned"]["summary"].items():
            print(f"{n:24s} l2 {v['l2_p']:.3f} gauge-free {v['l2_p_gauge_free']:.3f} grad {v['grad_p']:.3f} "
                  f"upstream {v['share_upstream']:.2f} inlet err {v['inlet_error']:+.3f} slope {v['inlet_slope']:.2f} r {v['inlet_correlation']:.2f}")
    suffix = "" if out["splits"] else "_perturbation" if args.perturbation else "_finetuned"
    path = ROOT / "results" / "validation" / f"pressure_diag_{run_dir.name}_e{epoch}{suffix}.json"
    write_json(path, out)
    print(path)


if __name__ == "__main__":
    main()
