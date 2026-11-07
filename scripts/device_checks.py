"""Pipeline checks on one device before a long study: resume after an interruption, repeatability,
checkpoint round trip, finite gradients (``stokes_ft.device_checks``).

    python scripts/device_checks.py --config smoke device=cuda

Writes ``runs/device_checks/device_checks.json`` and exits with status 1 if a check fails. The
numbers verify the pipeline; they are not results.
"""

import argparse
import json
import sys

from _bootstrap import ROOT
from stokes_ft.config import load_config
from stokes_ft.device_checks import run_device_checks
from stokes_ft.provenance import collect_metadata, resolve_device, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="smoke")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    device = resolve_device(cfg.device)
    out_dir = ROOT / "runs" / "device_checks"
    result = run_device_checks(cfg, out_dir / "runs", device)
    result["environment"] = collect_metadata(device)
    write_json(out_dir / "device_checks.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("losses", "environment")}, indent=1))
    if not result["passed"]:
        print("DEVICE CHECKS FAILED")
        sys.exit(1)
    print("DEVICE CHECKS PASSED")


if __name__ == "__main__":
    main()
