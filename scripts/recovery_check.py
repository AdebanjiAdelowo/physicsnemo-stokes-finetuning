"""Cross-session recovery test (``stokes_ft.recovery_check``), run in two separate sessions.

    python scripts/recovery_check.py first  [--config recovery_check] [overrides...]
    python scripts/session_state.py pack runs/recovery_resumed   <out>/state-recovery_resumed.tar
    python scripts/session_state.py pack runs/recovery_reference <out>/state-recovery_reference.tar
    ---- the session ends; a new session receives only the two tar files ----
    python scripts/session_state.py unpack <in>/state-recovery_resumed.tar   runs
    python scripts/session_state.py unpack <in>/state-recovery_reference.tar runs
    python scripts/recovery_check.py second [--config recovery_check] [overrides...]

The second call writes ``runs/recovery_check.json`` and exits with status 1 if the test fails.
``--deterministic`` (both sessions) forces deterministic CUDA algorithms for the test and makes
criterion version 3 (bit-identity) the decision; it is not part of the configuration.
"""

import argparse
import json
import os
import sys

if "--deterministic" in sys.argv:  # must precede the first cuBLAS call
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from _bootstrap import ROOT
from stokes_ft.config import load_config
from stokes_ft.provenance import collect_metadata, resolve_device, write_json
from stokes_ft.recovery_check import first_session, second_session

PAUSE_EPOCH = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", choices=("first", "second"))
    parser.add_argument("--config", default="recovery_check")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    if args.deterministic:
        import torch

        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    # run names without digits: the dataset loader orders files by the first digits in their path
    names = ("resumed", "reference", "repeat", "repeatb", "repeatc")
    cfg = {name: load_config(args.config, args.overrides + [f"name=recovery_{name}"]) for name in names}
    runs = ROOT / "runs"
    if args.session == "first":
        print(json.dumps(first_session(cfg["resumed"], cfg["reference"], runs, PAUSE_EPOCH), indent=1))
        return
    device = resolve_device(cfg["resumed"].device)
    out = second_session(cfg["resumed"], cfg["reference"], [cfg[n] for n in names[2:]], runs, PAUSE_EPOCH, device,
                         deterministic=args.deterministic)
    out["environment"] = collect_metadata(device)
    write_json(runs / "recovery_check.json", out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("histories", "environment")}, indent=1))
    print(f"criterion v1 (reported): {'passed' if out['criterion_v1']['passed'] else 'FAILED'}; "
          f"criterion v2: {'passed' if out['criterion_v2']['passed'] else 'FAILED'}; "
          f"criterion v3 (bit-identity): {out['criterion_v3']['verdict']}; decision: {out['decision']}")
    print("RECOVERY CHECK PASSED" if out["passed"] else "RECOVERY CHECK FAILED")
    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
