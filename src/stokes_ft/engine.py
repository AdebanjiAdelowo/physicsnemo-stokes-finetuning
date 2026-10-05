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
# Adapted from examples/cfd/stokes_mgn/train.py and examples/cfd/stokes_mgn/inference.py of
# NVIDIA PhysicsNeMo, commit b45a5c810c741e6b41f8515be24c51121f8fc21f. Modified by Adebanji
# Adelowo (2026): single process (no DistributedManager, DistributedSampler or
# DistributedDataParallel) and no Weights & Biases; explicit device and seed, with the sample
# order drawn per epoch from a seeded generator; a checkpoint every `training.checkpoint_every`
# epochs holding the number of completed epochs, older ones removed; loss, learning rate,
# validation errors and time written to history.csv; inference timed and returned.

"""Data-driven stage: MeshGraphNet training and prediction on the test meshes."""

import csv
import json
import math
import time
from pathlib import Path

import pyvista as pv
import torch
from physicsnemo.utils import load_checkpoint, save_checkpoint
from torch.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader as PyGDataLoader

from .data import make_dataset
from .model import build_meshgraphnet, count_parameters
from .provenance import device_synchronize, write_json

HISTORY_FIELDS = ["epoch", "loss", "lr", "val_rel_l2_u", "val_rel_l2_v", "val_rel_l2_p", "seconds"]


def relative_lp_error(pred, y, p=2):
    """Relative L2 error norm, as a fraction (upstream utils.relative_lp_error returns percent)."""
    return (torch.norm(pred - y, p=p) / torch.norm(y, p=p)).item()


def lr_decay_rate(cfg) -> float:
    """Per-epoch decay factor: the configured one, or the one that ends at 1 % of ``lr``."""
    if cfg.lr_decay_rate is not None:
        return cfg.lr_decay_rate
    final_lr_multiplier = 0.01
    return math.pow(final_lr_multiplier, 1.0 / cfg.epochs)


def _prune_checkpoints(ckpt_dir: Path, keep_epoch: int) -> None:
    for path in ckpt_dir.iterdir():
        parts = path.name.split(".")
        if len(parts) >= 3 and parts[-2].isdigit() and int(parts[-2]) != keep_epoch:
            path.unlink()


def _read_history(path: Path, upto_epoch: int) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [row for row in csv.DictReader(f) if int(row["epoch"]) <= upto_epoch]


def _write_history(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def validate(model, dataloader, device) -> dict:
    """Mean relative L2 error of the normalised u, v, p over the validation meshes."""
    errors = {"u": 0.0, "v": 0.0, "p": 0.0}
    for graph in dataloader:
        graph = graph.to(device)
        pred = model(graph.x, graph.edge_attr, graph)
        for index, key in enumerate(errors):
            errors[key] += relative_lp_error(pred[:, index : index + 1], graph.y[:, index : index + 1])
    return {key: value / len(dataloader) for key, value in errors.items()}


def train_data_driven(cfg, run_dir: Path, device: torch.device, resume: bool = False, log=print) -> dict:
    """Train the MeshGraphNet on the mean squared error of the normalised (u, v, p).

    Returns the training record, also written to ``<run_dir>/train_metrics.json``.
    """
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints" / "data_driven"
    metrics_path = run_dir / "train_metrics.json"
    if resume and metrics_path.exists():
        record = json.loads(metrics_path.read_text())
        if record["epochs"] == cfg.epochs:
            log(f"data-driven stage already complete ({cfg.epochs} epochs)")
            return record
    if not resume and ckpt_dir.exists() and any(ckpt_dir.iterdir()):
        raise RuntimeError(f"{ckpt_dir} holds checkpoints; pass --resume or use another run name.")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    dataset = make_dataset(cfg, "train", run_dir)
    validation_dataset = make_dataset(cfg, "validation", run_dir)
    order = torch.Generator()
    dataloader = PyGDataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, generator=order,
        pin_memory=device.type == "cuda",
    )
    validation_dataloader = PyGDataLoader(
        validation_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=True,
        pin_memory=device.type == "cuda",
    )

    model = build_meshgraphnet(cfg).to(device)
    if cfg.jit:
        model = torch.compile(model)
    model.train()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, fused=device.type == "cuda")
    # StepLR is stepped every iteration with a step size of one epoch: the rate decays per epoch.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(dataloader), gamma=lr_decay_rate(cfg))
    scaler = GradScaler(device.type) if cfg.amp else None

    epoch_init = load_checkpoint(
        str(ckpt_dir), models=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, device=device
    )
    history = _read_history(run_dir / "history.csv", epoch_init)
    if len(history) != epoch_init:
        raise RuntimeError(f"history.csv has {len(history)} epochs but the checkpoint is at epoch {epoch_init}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    log(f"data-driven stage: {count_parameters(model)} parameters, {len(dataloader)} iterations per epoch, "
        f"epochs {epoch_init} to {cfg.epochs} on {device}")
    for epoch in range(epoch_init, cfg.epochs):
        order.manual_seed(cfg.seed * 100003 + epoch)
        device_synchronize(device)
        start = time.perf_counter()
        loss_agg = 0.0
        for graph in dataloader:
            graph = graph.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=cfg.amp):
                pred = model(graph.x, graph.edge_attr, graph)
                loss = criterion(pred, graph.y)
            if cfg.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()
            loss_agg += loss.detach().item()
        device_synchronize(device)
        seconds = time.perf_counter() - start
        loss_agg /= len(dataloader)

        model.eval()
        errors = validate(model, validation_dataloader, device)
        model.train()
        history.append({
            "epoch": epoch + 1, "loss": loss_agg, "lr": optimizer.param_groups[0]["lr"],
            "val_rel_l2_u": errors["u"], "val_rel_l2_v": errors["v"], "val_rel_l2_p": errors["p"],
            "seconds": seconds,
        })
        log(f"epoch {epoch + 1}/{cfg.epochs} loss {loss_agg:.3e} val u {errors['u']:.3f} v {errors['v']:.3f} "
            f"p {errors['p']:.3f} ({seconds:.1f} s)")

        if (epoch + 1) % cfg.training.checkpoint_every == 0 or epoch + 1 == cfg.epochs:
            save_checkpoint(
                str(ckpt_dir), models=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch + 1
            )
            _prune_checkpoints(ckpt_dir, epoch + 1)
            _write_history(run_dir / "history.csv", history)

    train_seconds = sum(float(row["seconds"]) for row in history)
    iterations = cfg.epochs * len(dataloader)
    record = {
        "stage": "data_driven",
        "parameter_count": count_parameters(model),
        "epochs": cfg.epochs,
        "iterations": iterations,
        "train_seconds": train_seconds,
        "seconds_per_iteration": train_seconds / iterations,
        "final_train_loss": float(history[-1]["loss"]),
        "final_val_rel_l2": {key: float(history[-1][f"val_rel_l2_{key}"]) for key in ("u", "v", "p")},
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
        "resumed": epoch_init > 0,
    }
    write_json(metrics_path, record)
    return record


@torch.no_grad()
def predict_test(cfg, run_dir: Path, device: torch.device, log=print) -> dict:
    """Write the MeshGraphNet prediction of every test mesh to ``predictions/graph_<i>.vtp``.

    Each file is a copy of the dataset file with the arrays ``pred_u``, ``pred_v``, ``pred_p``
    in physical units, as upstream inference.py writes them. Returns the inference record.
    """
    run_dir = Path(run_dir)
    out_dir = run_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = make_dataset(cfg, "test", run_dir)
    dataloader = PyGDataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    model = build_meshgraphnet(cfg).to(device)
    model.eval()
    epoch = load_checkpoint(str(run_dir / "checkpoints" / "data_driven"), models=model, device=device)
    if epoch != cfg.epochs:
        raise RuntimeError(f"the data-driven checkpoint is at epoch {epoch}, expected {cfg.epochs}")
    stats = {key: value.to(device) for key, value in dataset.node_stats.items()}

    files, seconds = [], []
    for i, graph in enumerate(dataloader):
        graph = graph.to(device)
        pred = model(graph.x, graph.edge_attr, graph).detach()
        polydata = pv.read(dataset.data_list[i])
        for key_index, key in enumerate(["u", "v", "p"]):
            pred_val = pred[:, key_index : key_index + 1]
            pred_val = dataset.denormalize(pred_val, stats[f"{key}_mean"], stats[f"{key}_std"])
            polydata[f"pred_{key}"] = pred_val.detach().cpu().numpy()
        polydata.save(str(out_dir / f"graph_{i}.vtp"))
        files.append(Path(dataset.data_list[i]).name)

        repeats = []
        for _ in range(cfg.evaluation.timing_repeats + 1):
            device_synchronize(device)
            start = time.perf_counter()
            model(graph.x, graph.edge_attr, graph)
            device_synchronize(device)
            repeats.append(time.perf_counter() - start)
        seconds.append(sorted(repeats[1:])[len(repeats[1:]) // 2])  # median, first call discarded

    record = {
        "stage": "data_driven",
        "checkpoint_epoch": epoch,
        "test_files": files,
        "inference_seconds_per_mesh": sum(seconds) / len(seconds),
        "timing_repeats": cfg.evaluation.timing_repeats,
    }
    write_json(run_dir / "predict_metrics.json", record)
    log(f"predictions written for {len(files)} test meshes, {1e3 * record['inference_seconds_per_mesh']:.1f} ms per mesh")
    return record
