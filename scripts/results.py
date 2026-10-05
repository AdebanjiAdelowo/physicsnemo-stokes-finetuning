"""Move study results between a run directory, an archive and the tracked ``results/``.

    python scripts/results.py package runs/full        # on the GPU machine: runs/full.zip
    python scripts/results.py bundle runs/full         # on the GPU machine: runs/full_checkpoints.zip
    python scripts/results.py publish runs/full.zip    # locally: results/full/
    python scripts/results.py publish runs/local       # from a local run directory

``package`` holds the small records that are tracked in ``results/``. ``bundle`` holds what is
not tracked: the MeshGraphNet checkpoint (the model before the physics-informed stage), the
fine-tuning networks (after it) and the ``.vtp`` files with both sets of fields. ``publish``
refuses to overwrite an existing ``results/<name>/`` unless ``--force`` is given.
"""

import argparse
import shutil
import zipfile
from pathlib import Path

from _bootstrap import ROOT

KEEP_SUFFIXES = {".json", ".csv", ".npz"}
UNTRACKED_DIRS = {"checkpoints", "predictions", "figures"}


def tracked_files(study_dir: Path) -> list[Path]:
    if not (study_dir / "summary.json").exists():
        raise SystemExit(f"{study_dir} has no summary.json; run the evaluate stage first.")
    return [p for p in sorted(study_dir.rglob("*"))
            if p.is_file() and p.suffix in KEEP_SUFFIXES and not UNTRACKED_DIRS & set(p.relative_to(study_dir).parts)]


def package(study_dir: Path) -> Path:
    archive = study_dir.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in tracked_files(study_dir):
            z.write(path, Path(study_dir.name) / path.relative_to(study_dir))
    return archive


def bundle(study_dir: Path) -> Path:
    files = [p for d in ("checkpoints", "predictions") for p in sorted((study_dir / d).rglob("*")) if p.is_file()]
    if not any(p.suffix == ".mdlus" for p in files) or not any(p.name.startswith("fourier_dnn_") for p in files):
        raise SystemExit(f"{study_dir} does not hold the checkpoints of both stages")
    archive = study_dir.parent / f"{study_dir.name}_checkpoints.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
        for path in files:
            z.write(path, Path(study_dir.name) / path.relative_to(study_dir))
    return archive


def publish(source: Path, force: bool) -> Path:
    name = source.stem if source.suffix == ".zip" else source.name
    target = ROOT / "results" / name
    if target.exists():
        if not force:
            raise SystemExit(f"{target} exists; pass --force to replace it.")
        shutil.rmtree(target)
    if source.suffix == ".zip":
        with zipfile.ZipFile(source) as z:
            members = [m for m in z.namelist() if m.startswith(f"{name}/") and Path(m).suffix in KEEP_SUFFIXES]
            z.extractall(ROOT / "results", members)
    else:
        for path in tracked_files(source):
            dest = target / path.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("package", "zip the small result files of a run directory"),
                       ("bundle", "zip the checkpoints of both stages and the prediction files")):
        sub.add_parser(name, help=text).add_argument("study", type=Path)
    p = sub.add_parser("publish", help="copy a run directory or a package archive into results/")
    p.add_argument("source", type=Path)
    p.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "package":
        print(package(args.study.resolve()))
    elif args.command == "bundle":
        print(bundle(args.study.resolve()))
    else:
        print(publish(args.source.resolve(), args.force))


if __name__ == "__main__":
    main()
