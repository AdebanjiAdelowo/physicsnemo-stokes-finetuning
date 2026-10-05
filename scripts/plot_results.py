"""Draw the figures of a study from its result files.

    python scripts/plot_results.py --study results/local --out figures
    python scripts/plot_results.py --study runs/smoke --out runs/smoke/figures
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from stokes_ft.plotting import plot_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for path in plot_study(args.study, args.out):
        print(path)


if __name__ == "__main__":
    main()
