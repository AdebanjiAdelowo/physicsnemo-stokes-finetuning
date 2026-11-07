"""Carry a run directory between sessions (``stokes_ft.session_state``).

    python scripts/session_state.py pack runs/full /kaggle/working/state-full.tar
    python scripts/session_state.py unpack /kaggle/input/<previous kernel>/state-full.tar runs

``pack`` verifies the checkpoints and writes the tar, ``<tar>.sha256`` and ``<tar>.json``.
``unpack`` verifies the tar against its sidecar, refuses to overwrite an existing run, extracts it
and verifies the checkpoints again. Exits with status 1 on any failed check.
"""

import argparse
import json
import sys

from _bootstrap import ROOT
from stokes_ft.session_state import pack, unpack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("pack", "unpack"))
    parser.add_argument("source")
    parser.add_argument("target")
    args = parser.parse_args()
    try:
        result = pack(ROOT / args.source, ROOT / args.target) if args.action == "pack" \
            else unpack(ROOT / args.source, ROOT / args.target)
    except (RuntimeError, OSError) as error:
        print(f"SESSION STATE {args.action.upper()} FAILED: {error}")
        sys.exit(1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
