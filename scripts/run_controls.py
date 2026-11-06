"""Run controlled variants of the physics-informed stage on stored test predictions.

    python scripts/run_controls.py --config local --variant data_only --samples 0,1 --device cpu --threads 3
    python scripts/run_controls.py --config local --variant hard_bc --samples 0,1 --device mps
    python scripts/run_controls.py --config local --variant data_only --samples 2 --iterations 200 --probe

The variants are defined in ``stokes_ft.variants``. Each reads ``runs/<name>/predictions`` (the
MeshGraphNet prediction is the data target) and writes ``runs/<name>/controls/<variant>/``.
An accelerator run waits for the machine-wide accelerator lock. ``--probe`` writes to
``controls/<variant>_probe`` so that a short timing run never stands in for a result.
A mesh whose record already exists with the requested iteration count is skipped.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import torch

import _bootstrap  # noqa: F401
from stokes_ft.config import REPO_ROOT, load_config
from stokes_ft.gpulock import gpu_lock
from stokes_ft.provenance import collect_metadata, resolve_device, write_json
from stokes_ft.variants import VARIANTS, run_variant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="local")
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--samples", required=True, help="comma-separated test indices")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads (keep low on a shared machine)")
    parser.add_argument("--iterations", type=int, default=None, help="default: pi_iters of the configuration")
    parser.add_argument("--probe", action="store_true", help="timing probe: separate output directory, never a result")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = load_config(args.config, args.overrides)
    run_dir = REPO_ROOT / "runs" / cfg.name
    device = resolve_device(args.device)
    variant = VARIANTS[args.variant]
    if args.probe:
        variant = dataclasses.replace(variant, name=f"{variant.name}_probe")
    iterations = args.iterations or cfg.pi_iters
    if not args.probe and iterations != cfg.pi_iters:
        raise SystemExit("a shortened run must be marked --probe")

    out_dir = run_dir / "controls" / variant.name
    meta_path = out_dir / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"sessions": []}
    session = collect_metadata(device) | {"variant": variant.name, "samples": args.samples, "iterations": iterations,
                                           "torch_threads": torch.get_num_threads()}
    meta["sessions"].append(session)
    write_json(meta_path, meta)

    with gpu_lock(device.type):
        for index in (int(i) for i in args.samples.split(",")):
            record = out_dir / f"record_{index}.json"
            if record.exists() and json.loads(record.read_text())["iterations"] == iterations:
                print(f"{variant.name} mesh {index} already done")
                continue
            run_variant(cfg, run_dir, variant, index, device, iterations=iterations)


if __name__ == "__main__":
    main()
