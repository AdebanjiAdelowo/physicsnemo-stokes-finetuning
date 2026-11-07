"""Cost of each method per new geometry, on this machine's CPU, written to results/validation/costs.json.

    python scripts/benchmark_costs.py [--threads 3]

Each task runs in its own process so that its peak resident memory is its own; the memory
reported is the peak minus the resident size after imports (``ru_maxrss``). Times are wall-clock
medians on a machine shared with other work (the load average is recorded); they are planning
figures for this machine, not hardware benchmarks.

Tasks, on the first 10 test meshes of the local study:

* ``fem``: Taylor-Hood P2-P1 assembly and direct solve (single thread), 3 repeats per mesh.
* ``mgn_inference``: one MeshGraphNet forward pass (local checkpoint), 5 repeats per mesh.
* ``pinn_iteration``: iterations of the upstream physics-informed fit (second derivatives).
* ``data_only_iteration``: iterations of the data-only control (no derivatives of the network).

Physics-informed fits are timed per iteration and multiplied by the 10,000 iterations of the
configuration; that product is marked as an extrapolation. Stored measurements of complete runs
(MeshGraphNet training on MPS, 10,000-iteration fits on MPS and CPU) are copied with their source.
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # bytes on macOS


def child(task: str, threads: int) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    import torch

    torch.set_num_threads(threads)
    from stokes_ft.config import load_config
    from stokes_ft.data import read_sample

    cfg = load_config("local")
    run_dir = ROOT / "runs" / "local"
    samples = [read_sample(run_dir / "predictions" / f"graph_{i}.vtp") for i in range(10)]
    base = rss_mb()
    seconds = []
    if task == "fem":
        from stokes_ft.fem import taylor_hood_solve

        for s in samples:
            t = []
            for _ in range(3):
                start = time.perf_counter()
                taylor_hood_solve(s["coords"], s["triangles"], cfg.physics.nu, cfg.physics.u_max)
                t.append(time.perf_counter() - start)
            seconds.append(float(np.median(t)))
        per = {"seconds_per_mesh": seconds}
    elif task == "mgn_inference":
        from physicsnemo.utils import load_checkpoint

        from stokes_ft.data import make_dataset
        from stokes_ft.model import build_meshgraphnet

        model = build_meshgraphnet(cfg)
        load_checkpoint(str(run_dir / "checkpoints" / "data_driven"), models=model, device="cpu")
        model.eval()
        base = rss_mb()
        dataset = make_dataset(cfg, "test", run_dir)
        with torch.no_grad():
            for graph in dataset:
                model(graph.x, graph.edge_attr, graph)
                t = []
                for _ in range(5):
                    start = time.perf_counter()
                    model(graph.x, graph.edge_attr, graph)
                    t.append(time.perf_counter() - start)
                seconds.append(float(np.median(t)))
        per = {"seconds_per_mesh": seconds}
    else:
        from stokes_ft.variants import VARIANTS, VariantFineTuner

        name, iters = ("upstream", 40) if task == "pinn_iteration" else ("data_only", 300)
        for s in samples[:3]:
            torch.manual_seed(0)
            tuner = VariantFineTuner(VARIANTS[name], torch.device("cpu"), s, cfg.physics.nu, cfg.physics.u_max)
            for _ in range(5):
                tuner.step()
            start = time.perf_counter()
            for _ in range(iters):
                tuner.step()
            seconds.append((time.perf_counter() - start) / iters)
        per = {"seconds_per_iteration": seconds, "meshes": 3, "iterations_timed": iters}
    return {"task": task, "threads": threads, "rss_after_imports_mb": base, "peak_rss_mb": rss_mb(),
            "peak_minus_baseline_mb": rss_mb() - base} | per


ACCURACY_NOTE = ("Taylor-Hood reproduces the reference to relative L2 below 1e-9 (audit_checks.json); the MeshGraphNet "
                 "velocity error is 0.21 on 100 test meshes; the physics-informed fit 0.136 on 10 meshes (0.233 before).")


def physics_fit_mps() -> dict:
    """Median over the stored per-mesh fits; the median, not the mean, because some runs shared the accelerator."""
    records = [json.loads(p.read_text()) for p in sorted((ROOT / "results" / "local" / "finetune").glob("finetune_*.json"))]
    seconds = [r["seconds"] for r in records]
    return {"value": float(np.median(seconds)), "per_mesh": seconds, "meshes": len(seconds),
            "kind": "stored, 10,000 iterations, median over meshes; runs above the median overlapped other jobs "
                    "on the shared accelerator or a system sleep"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--child", default=None)
    parser.add_argument("--stored-only", action="store_true",
                        help="keep the CPU measurements of the existing costs.json; refresh only the stored entries")
    args = parser.parse_args()
    if args.child:
        print("RESULT " + json.dumps(child(args.child, args.threads)))
        return

    sys.path.insert(0, str(ROOT / "src"))
    import torch

    from stokes_ft.provenance import collect_metadata, write_json

    if args.stored_only:
        out = json.loads((ROOT / "results" / "validation" / "costs.json").read_text())
        out["per_new_geometry_seconds"]["physics_informed_fit_mps"] = physics_fit_mps()
        out["accuracy_note"] = ACCURACY_NOTE
        fem = out["per_new_geometry_seconds"]["taylor_hood_fem_cpu_1_thread"]["value"]
        out["ratios"]["physics_fit_mps_over_fem"] = out["per_new_geometry_seconds"]["physics_informed_fit_mps"]["value"] / fem
        out["stored_entries_refreshed"] = collect_metadata(torch.device("cpu"))
        write_json(ROOT / "results" / "validation" / "costs.json", out)
        print(json.dumps(out["per_new_geometry_seconds"]["physics_informed_fit_mps"], indent=1), out["ratios"])
        return

    results = {}
    for task, threads in (("fem", 1), ("mgn_inference", args.threads), ("pinn_iteration", args.threads),
                          ("data_only_iteration", args.threads)):
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
        load_before = os.getloadavg()
        out = subprocess.run([sys.executable, __file__, "--child", task, "--threads", str(threads)], env=env,
                             capture_output=True, text=True, check=True).stdout
        line = next(l for l in out.splitlines() if l.startswith("RESULT "))
        results[task] = json.loads(line[7:]) | {"load_average_before": load_before}
        print(task, {k: v for k, v in results[task].items() if not isinstance(v, list)})

    stored = json.loads((ROOT / "results" / "local" / "summary.json").read_text())["compute"]
    fem = float(np.median(results["fem"]["seconds_per_mesh"]))
    mgn = float(np.median(results["mgn_inference"]["seconds_per_mesh"]))
    pinn_it = float(np.median(results["pinn_iteration"]["seconds_per_iteration"]))
    data_it = float(np.median(results["data_only_iteration"]["seconds_per_iteration"]))
    controls = sorted((ROOT / "runs" / "local" / "controls" / "data_only").glob("record_*.json"))
    data_only_measured = [json.loads(p.read_text())["seconds"] for p in controls]
    out = {
        "environment": collect_metadata(torch.device("cpu")),
        "measured_cpu": results,
        "per_new_geometry_seconds": {
            "taylor_hood_fem_cpu_1_thread": {"value": fem, "kind": "measured, median of 10 meshes"},
            "meshgraphnet_inference_cpu": {"value": mgn, "kind": "measured, median of 10 meshes", "threads": args.threads},
            "meshgraphnet_inference_mps": {"value": stored["data_driven_inference_seconds_per_mesh"], "kind": "stored (results/local/summary.json)"},
            "physics_informed_fit_mps": physics_fit_mps(),
            "physics_informed_fit_cpu": {"value": pinn_it * 10000, "kind": "EXTRAPOLATED from the per-iteration time x 10,000", "threads": args.threads},
            "data_only_fit_cpu": {"value": float(np.mean(data_only_measured)) if data_only_measured else data_it * 10000,
                                  "kind": "measured, complete 10,000-iteration runs" if data_only_measured else "EXTRAPOLATED",
                                  "runs": len(data_only_measured), "threads": 3},
        },
        "one_off_costs_seconds": {
            "meshgraphnet_training_mps": {"value": stored["data_driven_train_seconds"], "kind": "stored, 100 meshes x 40 epochs"},
            "training_data_fem_cpu": {"value": fem * 100, "kind": "derived: 100 Taylor-Hood solves at the measured median"},
        },
        "accuracy_note": ACCURACY_NOTE,
    }
    ratio = {"physics_fit_mps_over_fem": out["per_new_geometry_seconds"]["physics_informed_fit_mps"]["value"] / fem,
             "mgn_inference_cpu_over_fem": mgn / fem}
    out["ratios"] = ratio
    write_json(ROOT / "results" / "validation" / "costs.json", out)
    print(json.dumps(out["per_new_geometry_seconds"], indent=1), ratio)


if __name__ == "__main__":
    main()
