"""Download the Stokes dataset from NGC, verify it and write the seeded split.

    python scripts/prepare_data.py            # download (172 MB), verify, unzip (222 MB), split
    python scripts/prepare_data.py --check    # verify what is on disk, download nothing

The archive holds 1000 FEniCS solutions as ``results/res_<n>.vtp``. Nothing under ``data/`` is tracked.
"""

import argparse
import urllib.request
import zipfile

from _bootstrap import ROOT
from stokes_ft.config import load_config
from stokes_ft.data import DATASET_ARCHIVE, DATASET_BYTES, DATASET_FILES, DATASET_SHA256, DATASET_URL, prepare_split, raw_files, sha256_of


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    cfg = load_config("official")
    raw_dir = ROOT / cfg.data.raw_dir
    archive = raw_dir.parent / DATASET_ARCHIVE

    if not archive.exists() and not raw_dir.exists():
        if args.check:
            raise SystemExit(f"no dataset at {raw_dir}")
        raw_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {DATASET_BYTES / 2**20:.0f} MB to {archive}", flush=True)
        urllib.request.urlretrieve(DATASET_URL, archive)
    if archive.exists():
        found = sha256_of(archive)
        if found != DATASET_SHA256:
            raise SystemExit(f"{archive}: SHA-256 {found}, expected {DATASET_SHA256}")
        print("archive SHA-256 verified")
        if not raw_dir.exists():
            with zipfile.ZipFile(archive) as z:
                z.extractall(raw_dir.parent)
    else:
        print("archive not present: the SHA-256 check is skipped")

    files = raw_files(raw_dir)
    if len(files) != DATASET_FILES:
        raise SystemExit(f"{raw_dir}: {len(files)} files, expected {DATASET_FILES}")
    manifest = prepare_split(cfg)
    print(f"{len(files)} files; split seed {manifest['split_seed']}: "
          + ", ".join(f"{k} {len(manifest[k])}" for k in ("train", "validation", "test")))
    print("first test files:", manifest["test"][:10])


if __name__ == "__main__":
    main()
