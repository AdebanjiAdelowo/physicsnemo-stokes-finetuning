# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from examples/cfd/stokes_mgn/preprocess.py and examples/cfd/stokes_mgn/utils.py
# (get_dataset) of NVIDIA PhysicsNeMo, commit b45a5c810c741e6b41f8515be24c51121f8fc21f.
# Modified by Adebanji Adelowo (2026): the split is seeded, taken over the files in numeric
# order, written as links with a manifest, and checked on reuse; the .vtp reader returns a
# dictionary with the triangles and reads the stored predictions only when they are present;
# StokesDataset is constructed inside a run directory, where it keeps its statistics files.

"""Dataset download check, seeded split, PhysicsNeMo dataset construction and .vtp access."""

import contextlib
import hashlib
import json
import os
import random
import re
import shutil
from pathlib import Path

import numpy as np
import pyvista as pv
from physicsnemo.datapipes.gnn.stokes_dataset import StokesDataset

from .config import REPO_ROOT

DATASET_URL = (
    "https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/physicsnemo/"
    "physicsnemo_datasets_stokes_flow/0.0.1/files?redirect=true&path=results_polygon.zip"
)
DATASET_ARCHIVE = "results_polygon.zip"
DATASET_BYTES = 172398657
DATASET_SHA256 = "5aec06af33158c71be9b0cd6be01392a1e91794070f0fde3a72f796004d696d1"
DATASET_FILES = 1000

# point marker values of the dataset
INTERIOR, INFLOW, OUTFLOW, WALL, POLYGON = 0, 1, 2, 3, 4
SPLITS = ("train", "validation", "test")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_number(path) -> int:
    """Simulation number of a dataset file ``res_<n>.vtp``."""
    match = re.fullmatch(r"res_(\d+)\.vtp", Path(path).name)
    if match is None:
        raise ValueError(f"not a dataset file name: {path}")
    return int(match.group(1))


def raw_files(raw_dir: Path) -> list[Path]:
    files = sorted(Path(raw_dir).glob("res_*.vtp"), key=sample_number)
    if not files:
        raise FileNotFoundError(f"no res_*.vtp files in {raw_dir}; run scripts/prepare_data.py")
    return files


def split_numbers(numbers: list[int], seed: int, train_fraction: float, validation_fraction: float) -> dict:
    """Seeded shuffle of the simulation numbers and the 80/10/10 partition of upstream preprocess.py."""
    numbers = sorted(numbers)
    random.Random(seed).shuffle(numbers)
    train_size = int(len(numbers) * train_fraction)
    valid_size = int(len(numbers) * validation_fraction)
    return {
        "train": sorted(numbers[:train_size]),
        "validation": sorted(numbers[train_size : train_size + valid_size]),
        "test": sorted(numbers[train_size + valid_size :]),
    }


def prepare_split(cfg, root: Path = REPO_ROOT) -> dict:
    """Create ``<split_dir>/{train,validation,test}`` and ``split.json``, or verify an existing split."""
    raw_dir = (root / cfg.data.raw_dir).resolve()
    split_dir = root / cfg.data.split_dir
    files = {sample_number(p): p for p in raw_files(raw_dir)}
    parts = split_numbers(list(files), cfg.data.split_seed, cfg.data.train_fraction, cfg.data.validation_fraction)
    manifest = {
        "source": DATASET_URL,
        "files_total": len(files),
        "split_seed": cfg.data.split_seed,
        "train_fraction": cfg.data.train_fraction,
        "validation_fraction": cfg.data.validation_fraction,
        **parts,
    }
    manifest_path = split_dir / "split.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError(f"{manifest_path} was made with other settings; remove {split_dir} to rebuild it.")
    else:
        for split in SPLITS:
            (split_dir / split).mkdir(parents=True, exist_ok=True)
            for n in parts[split]:
                link = split_dir / split / files[n].name
                if not link.exists():
                    try:
                        os.symlink(files[n], link)
                    except OSError:
                        shutil.copy(files[n], link)
        manifest_path.write_text(json.dumps(manifest) + "\n")
    for split in SPLITS:
        present = sorted(sample_number(p) for p in (split_dir / split).glob("res_*.vtp"))
        if present != parts[split]:
            raise RuntimeError(f"{split_dir / split} does not match split.json")
    return manifest


@contextlib.contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def make_dataset(cfg, split: str, run_dir: Path, root: Path = REPO_ROOT) -> StokesDataset:
    """Construct the PhysicsNeMo ``StokesDataset`` of one split.

    ``StokesDataset`` writes ``node_stats.json`` and ``edge_stats.json`` to the current directory
    for the training split and reads them from there for the other two, so it is built inside
    ``run_dir`` and the training split has to be built first. It orders the files by the first
    run of digits in their path, which is the simulation number only if no directory name holds
    a digit; the data directory is therefore passed relative to ``run_dir`` and the order is checked.
    """
    num_samples = {
        "train": cfg.num_training_samples,
        "validation": cfg.num_validation_samples,
        "test": cfg.num_test_samples,
    }[split]
    run_dir = Path(run_dir).resolve()
    relative = os.path.relpath((root / cfg.data.split_dir).resolve(), run_dir)
    with working_directory(run_dir):
        dataset = StokesDataset(name=f"stokes_{split}", data_dir=relative, split=split, num_samples=num_samples)
        dataset.data_list = [str(Path(p).resolve()) for p in dataset.data_list]
    numbers = [sample_number(p) for p in dataset.data_list]
    if numbers != sorted(numbers):
        raise RuntimeError(
            f"StokesDataset did not order the {split} files by simulation number. A directory name in "
            f"'{relative}' probably contains a digit; move the data or the run directory."
        )
    return dataset


def read_sample(path) -> dict:
    """Mesh, boundary markers and fields of one ``.vtp`` file.

    Returns ``coords`` (N, 2), ``triangles`` (M, 3), ``marker`` (N,), the reference fields
    ``ref_u``, ``ref_v``, ``ref_p`` (N, 1) and, when the file holds them, the MeshGraphNet
    predictions ``gnn_*`` and the fine-tuned fields ``filtered_*``.
    """
    pv_mesh = pv.read(str(path))
    faces = np.asarray(pv_mesh.faces).reshape(-1, 4)
    if not (faces[:, 0] == 3).all():
        raise ValueError(f"{path} is not a triangle mesh")
    sample = {
        "coords": np.array(pv_mesh.points[:, 0:2]),
        "triangles": faces[:, 1:].astype(np.int64),
        "marker": np.asarray(pv_mesh.point_data["marker"]).astype(np.int64),
    }
    for prefix, stored in (("ref", ""), ("gnn", "pred_"), ("filtered", "filtered_")):
        for key in ("u", "v", "p"):
            if stored + key in pv_mesh.point_data:
                sample[f"{prefix}_{key}"] = np.array(pv_mesh.point_data[stored + key], dtype=np.float64).reshape(-1, 1)
    return sample


def boundary_masks(marker: np.ndarray) -> dict:
    return {
        "interior": marker == INTERIOR,
        "inflow": marker == INFLOW,
        "outflow": marker == OUTFLOW,
        "noslip": (marker == WALL) | (marker == POLYGON),
    }


def mesh_edges(triangles: np.ndarray) -> np.ndarray:
    """Edges of the triangles, (3M, 2), as upstream utils.get_dataset lists them."""
    return np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
