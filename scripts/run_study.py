"""Run a study: data-driven training, prediction, physics-informed fine-tuning, evaluation.

    python scripts/run_study.py --config smoke
    python scripts/run_study.py --config local
    python scripts/run_study.py --config full --resume
    python scripts/run_study.py --config local --stages evaluate
    python scripts/run_study.py --config full name=probe epochs=1 pi_iters=200

Trailing arguments are Hydra overrides. Results go to ``runs/<name>``.
"""

import argparse

import _bootstrap  # noqa: F401
from stokes_ft.config import load_config
from stokes_ft.study import ALL_STAGES, run_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="smoke, local, full or official")
    parser.add_argument("--stages", default=",".join(ALL_STAGES), help="comma-separated subset of " + ",".join(ALL_STAGES))
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run of the same configuration")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    print(run_study(cfg, stages=args.stages.split(","), resume=args.resume))


if __name__ == "__main__":
    main()
