"""Bounded baseline and global-latent feasibility probe on train/validation only."""

import argparse
import csv
import json
import math
import statistics
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import torch
from physicsnemo.utils import load_checkpoint

from _bootstrap import ROOT
from stokes_ft import checkpoint, engine
from stokes_ft.config import load_config
from stokes_ft.data import DATASET_ARCHIVE, DATASET_SHA256, sha256_of
from stokes_ft.model import build_meshgraphnet, count_parameters
from stokes_ft.provenance import collect_metadata, device_synchronize, resolve_device, write_json
from stokes_ft.selection import CRITERIA
from stokes_ft.session_state import pack, unpack
from stokes_ft.study import run_study

ARMS = ("baseline", "global_latent")
PROBE_EPOCHS = 3
PROBE_TRAIN_MESHES = 16
PROBE_VALIDATION_MESHES = 2


def inference_seconds(model, graphs, device):
    elapsed = []
    model.eval()
    with torch.inference_mode():
        for graph in graphs:
            graph = graph.to(device)
            model(graph.x, graph.edge_attr, graph)
            timings = []
            for _ in range(20):
                device_synchronize(device)
                start = time.perf_counter()
                out = model(graph.x, graph.edge_attr, graph)
                device_synchronize(device)
                timings.append(time.perf_counter() - start)
            assert torch.isfinite(out).all().item()
            elapsed.append(statistics.median(timings))
    return statistics.mean(elapsed)


def probe_arm(cfg, work: Path, device):
    start = time.perf_counter()
    before, after = work / "initial", work / "restored"
    archive = work / f"{cfg.name}.tar"
    splits = []
    make_dataset = engine.make_dataset

    def checked_dataset(config, split, *args, **kwargs):
        if split not in {"train", "validation"}:
            raise RuntimeError(f"feasibility probe cannot open the {split} split")
        splits.append(split)
        return make_dataset(config, split, *args, **kwargs)

    with patch.object(engine, "make_dataset", checked_dataset):
        initial = run_study(cfg, stages=("train",), runs_root=before, stop_after_epoch=1)
        first_peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        paused = pack(initial, archive)
        assert paused["status"]["state"] == "paused"
        assert paused["status"]["epochs_completed"] == 1
        restored = unpack(archive, after)
        assert restored["sha256"] == paused["sha256"]
        assert len(restored["checkpoints"]) == 1 and restored["checkpoints"][0]["ok"]
        run = run_study(cfg, stages=("train",), runs_root=after, resume=True)
        metrics = json.loads((run / "train_metrics.json").read_text())
        assert metrics["status"] == "complete" and metrics["resumed_from_epoch"] == 1
        assert metrics["resumed_from_verified_checkpoint"] and metrics["iterations"] == cfg.epochs * cfg.num_training_samples
        assert set(splits) == {"train", "validation"}

    with (run / "history.csv").open() as file:
        history = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(file)]
    assert [row["epoch"] for row in history] == list(range(1, cfg.epochs + 1))
    assert all(math.isfinite(value) for row in history for value in row.values())
    assert all(row["seconds"] > 0 for row in history)
    model = build_meshgraphnet(cfg).to(device)
    selected = {}
    for criterion, score in CRITERIA.items():
        directory = run / "checkpoints" / "selected" / criterion
        choice = json.loads((directory / "selection.json").read_text())
        scores = [score({k: row[f"val_rel_l2_{k}"] for k in "uvp"}) for row in history]
        assert choice["epoch"] == scores.index(min(scores)) + 1
        assert math.isclose(choice["score"], min(scores), rel_tol=0, abs_tol=1e-12)
        assert load_checkpoint(str(directory), models=model, epoch=choice["epoch"], device=device) == choice["epoch"]
        selected[criterion] = choice
    regular = run / "checkpoints" / "data_driven"
    manifests = checkpoint.manifests(regular)
    assert manifests and all(checkpoint.verify(path, None)[0] for path in manifests)
    assert load_checkpoint(str(regular), models=model, epoch=cfg.epochs, device=device) == cfg.epochs
    if cfg.model_variant == "global_latent":
        assert torch.count_nonzero(model.global_mlp[-1].weight).item() > 0, "global weights did not update"
    assert metrics["parameter_count"] == count_parameters(model)

    graphs = checked_dataset(cfg, "validation", run)
    timing = inference_seconds(model, graphs, device)
    seconds_per_update = statistics.mean(row["seconds"] for row in history[1:]) / cfg.num_training_samples
    return {
        "config": {"model_variant": cfg.model_variant, "epochs": cfg.epochs, "seed": cfg.seed,
                   "train_meshes": cfg.num_training_samples, "validation_meshes": cfg.num_validation_samples},
        "train_metrics": metrics, "selected": selected, "opened_splits": sorted(set(splits)),
        "paused_archive_sha256": paused["sha256"], "paused_archive_bytes": paused["bytes"],
        "validation_inference_seconds_per_mesh": timing, "seconds_per_update_after_first_epoch": seconds_per_update,
        "peak_allocated_cuda_mib": max(first_peak, metrics["peak_cuda_memory_mb"]) if first_peak is not None else None,
        "wall_seconds": time.perf_counter() - start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = resolve_device("cuda")
    if torch.cuda.get_device_name(device) != "Tesla T4":
        raise SystemExit("the matched feasibility probe requires a Tesla T4")
    full = load_config("full_arch")
    archive = (ROOT / full.data.raw_dir).parent / DATASET_ARCHIVE
    if sha256_of(archive) != DATASET_SHA256:
        raise SystemExit("the NGC dataset archive does not match the expected SHA-256")
    with tempfile.TemporaryDirectory(prefix="stokes-architecture-probe-") as tmp:
        arms = {}
        for arm in ARMS:
            cfg = load_config("full_arch", [
                f"model_variant={arm}", f"name=arch_probe_{arm}", f"epochs={PROBE_EPOCHS}",
                f"num_training_samples={PROBE_TRAIN_MESHES}", f"num_validation_samples={PROBE_VALIDATION_MESHES}",
            ])
            arms[arm] = probe_arm(cfg, Path(tmp), device)
    projected = {arm: record["seconds_per_update_after_first_epoch"] * full.num_training_samples * full.epochs / 3600
                 for arm, record in arms.items()}
    report = {"environment": collect_metadata(device), "dataset_sha256": DATASET_SHA256,
              "arms": arms, "estimated_update_gpu_hours_per_run": projected,
              "estimated_update_gpu_hours_for_six": 3 * sum(projected.values())}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
