"""Strict deterministic two-session recovery check for both production-width architecture arms."""

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
from omegaconf import OmegaConf
from physicsnemo.utils import load_checkpoint

from _bootstrap import ROOT
from full_arch_probe import ARMS
from stokes_ft import checkpoint, engine
from stokes_ft.config import load_config
from stokes_ft.data import DATASET_ARCHIVE, DATASET_SHA256, prepare_split, sha256_of
from stokes_ft.model import build_meshgraphnet
from stokes_ft.provenance import collect_metadata, resolve_device, write_json
from stokes_ft.selection import CRITERIA
from stokes_ft.session_state import pack, unpack
from stokes_ft.study import run_study

PAUSE = 1
EPOCHS = 2
MESHES = 8
ENV_KEYS = ("git_commit", "device", "gpu_name", "cuda_version", "python_version", "torch_version",
            "physicsnemo", "torch_geometric_version", "torch_scatter_version", "pyvista_version", "numpy_version", "driver_version", "compute_capability")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text())


def compare(a, b):
    """Strict recursive comparison, with max errors on all corresponding numeric leaves."""
    result = {"identical": True, "max_abs": 0.0, "max_rel": 0.0, "tensor_leaves": 0}

    def visit(x, y):
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            result["tensor_leaves"] += 1
            x, y = x.detach().cpu(), y.detach().cpu()
            if x.shape != y.shape or x.dtype != y.dtype:
                result["identical"] = False
                return
            # Compare the bit patterns, including the sign of floating-point zero.
            same = torch.equal(x.reshape(-1).contiguous().view(torch.uint8),
                               y.reshape(-1).contiguous().view(torch.uint8))
            result["identical"] &= same
            if not same and x.numel():
                visit(x.numpy(), y.numpy())
        elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            if x.shape != y.shape or x.dtype != y.dtype:
                result["identical"] = False
                return
            result["identical"] &= x.tobytes() == y.tobytes()
            if x.size:
                delta = np.abs(x.astype(float) - y.astype(float))
                result["max_abs"] = max(result["max_abs"], float(delta.max()))
                result["max_rel"] = max(result["max_rel"], float((delta / np.maximum(np.abs(x.astype(float)), 1e-30)).max()))
        elif isinstance(x, dict) and isinstance(y, dict):
            result["identical"] &= x.keys() == y.keys()
            for key in x.keys() & y.keys():
                visit(x[key], y[key])
        elif isinstance(x, (list, tuple)) and type(x) is type(y):
            result["identical"] &= len(x) == len(y)
            for first, second in zip(x, y):
                visit(first, second)
        elif isinstance(x, float) and isinstance(y, float):
            visit(np.asarray(x), np.asarray(y))
        else:
            result["identical"] &= type(x) is type(y) and x == y

    visit(a, b)
    return result


def inspect_state(run, cfg, split, dataset_hash, epoch, cuda_devices):
    """Refuse fallback or fresh training if any part of the requested boundary is absent."""
    meta = read(run / "study_metadata.json")
    require(meta["dataset"]["archive_sha256_found"] == dataset_hash, "dataset digest differs")
    require(meta["config"] == OmegaConf.to_container(cfg, resolve=True), "run configuration differs")
    status = read(run / "status.json")
    expected_status = "complete" if epoch == cfg.epochs else "paused"
    require(status["state"] == expected_status and status["stages_requested"] == ["train"], "run status differs")
    if expected_status == "paused":
        require(status["epochs_completed"] == epoch, "status epoch differs")
    regular = run / "checkpoints/data_driven"
    manifests = checkpoint.manifests(regular)
    require(manifests and manifests[0].name == f"manifest.0.{epoch}.json", "expected checkpoint epoch missing")
    manifest = read(manifests[0])
    model = build_meshgraphnet(cfg)
    fp = checkpoint.fingerprint(cfg, run, model, split["train"], split["validation"])
    require(manifest["epoch"] == epoch, "manifest epoch differs")
    model_file = f"{type(model).__name__}.0.{epoch}.mdlus"
    required = {model_file, f"checkpoint.0.{epoch}.pt", f"rng.0.{epoch}.pt", f"history.0.{epoch}.csv"}
    ok, reason = checkpoint.verify(manifests[0], fp)
    require(ok, reason)
    require(required <= manifest["files"].keys(), "checkpoint file inventory incomplete")
    payload = torch.load(regular / f"checkpoint.0.{epoch}.pt", map_location="cpu", weights_only=False)
    updates = epoch * cfg.num_training_samples
    require(cfg.batch_size == 1, "recovery check expects one update per mesh")
    require(payload["epoch"] == epoch, "training payload epoch differs")
    optimizer = payload["optimizer_state_dict"]
    steps = [item["step"].item() for item in optimizer["state"].values()]
    require(len(steps) == len(list(model.parameters())) and all(step == updates for step in steps),
            "optimizer update counter differs")
    scheduler = payload["scheduler_state_dict"]
    require(scheduler["last_epoch"] == updates and scheduler["_step_count"] == updates + 1
            and scheduler["step_size"] == cfg.num_training_samples, "scheduler update counter differs")
    rng = torch.load(regular / f"rng.0.{epoch}.pt", map_location="cpu", weights_only=False)
    require({"python", "numpy", "torch"} <= rng.keys() and len(rng.get("cuda", [])) == cuda_devices, "RNG state incomplete")
    history_path = regular / f"history.0.{epoch}.csv"
    require(history_path.read_bytes() == (run / "history.csv").read_bytes(), "checkpoint history differs")
    with history_path.open() as file:
        rows = [{k: float(v) for k, v in row.items() if k != "seconds"} for row in csv.DictReader(file)]
    require([r["epoch"] for r in rows] == list(range(1, epoch + 1))
            and all(math.isfinite(v) for row in rows for v in row.values()), "history epochs or values invalid")
    selections = {}
    for name, score in CRITERIA.items():
        selected = run / "checkpoints/selected" / name
        record = read(selected / "selection.json")
        scores = [score({k: row[f"val_rel_l2_{k}"] for k in "uvp"}) for row in rows]
        require(record["epoch"] == scores.index(min(scores)) + 1 and record["score"] == min(scores),
                f"{name} selection differs from history")
        require(load_checkpoint(str(selected), models=model, epoch=record["epoch"], device="cpu") == record["epoch"],
                f"{name} checkpoint failed to load")
        selections[name] = {"record": record, "model": {k: v.clone() for k, v in model.state_dict().items()}}
    require(load_checkpoint(str(regular), models=model, epoch=epoch, device="cpu") == epoch, "model checkpoint failed to load")
    return {"epoch": epoch, "updates": updates, "fingerprint": fp, "manifest": manifest,
            "model": model.state_dict(), "optimizer": optimizer, "scheduler": scheduler,
            "rng": rng, "history": rows, "selections": selections}


def rejection(action, expected):
    try:
        action()
    except RuntimeError as exc:
        require(expected in str(exc), f"wrong rejection for {expected}: {exc}")
        return str(exc)
    raise RuntimeError(f"invalid checkpoint accepted: {expected}")


def drills(run, cfg, split, digest, archive, work, cuda_devices):
    checks = {}
    def inspect(folder, config=cfg, dataset_hash=digest, epoch=PAUSE):
        return inspect_state(folder, config, split, dataset_hash, epoch, cuda_devices)

    wrong = work / "bad-sidecar.tar"
    os.symlink(archive.resolve(), wrong)
    Path(f"{wrong}.sha256").write_text("0" * 64 + "\n")
    checks["archive_checksum"] = rejection(lambda: unpack(wrong, work / "bad-output"), "does not match")
    incomplete = shutil.copytree(run, work / "incomplete")
    (incomplete / "checkpoints/data_driven" / f"rng.0.{PAUSE}.pt").unlink()
    checks["incomplete"] = rejection(lambda: inspect(incomplete), "missing rng")
    checks["dataset"] = rejection(lambda: inspect(run, dataset_hash="0" * 64), "dataset digest differs")
    checks["epoch"] = rejection(lambda: inspect(run, epoch=PAUSE + 1), "run status differs")
    counter = shutil.copytree(run, work / "counter") / "checkpoints/data_driven"
    payload_path = counter / f"checkpoint.0.{PAUSE}.pt"
    payload = torch.load(payload_path, weights_only=False, map_location="cpu")
    payload["scheduler_state_dict"]["last_epoch"] += 1
    torch.save(payload, payload_path)
    manifest_path = counter / f"manifest.0.{PAUSE}.json"
    manifest = read(manifest_path)
    manifest["files"][payload_path.name] = sha256_of(payload_path)
    write_json(manifest_path, manifest)
    checks["update"] = rejection(lambda: inspect(counter.parent.parent), "scheduler update counter differs")
    for field, value in (("model_variant", next(arm for arm in ARMS if arm != cfg.model_variant)), ("lr", cfg.lr * 2)):
        altered = cfg.copy()
        altered[field] = value
        altered_run = shutil.copytree(run, work / field)
        altered_meta = read(altered_run / "study_metadata.json")
        altered_meta["config"] = OmegaConf.to_container(altered, resolve=True)
        write_json(altered_run / "study_metadata.json", altered_meta)
        field_digest = "architecture_sha256" if field == "model_variant" else "config_sha256"
        checks[field] = rejection(lambda: inspect(altered_run, altered), field_digest)
    return checks


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(cfg, work, **kwargs):
    original = engine.make_dataset
    def dataset(config, split, *args, **options):
        require(split in {"train", "validation"}, f"unexpected split {split}")
        return original(config, split, *args, **options)
    with patch.object(engine, "make_dataset", dataset):
        return run_study(cfg, stages=["train"], runs_root=work, **kwargs)


def replay(paused, cfg, work):
    restored = {}
    load = checkpoint.load_latest
    def verified_load(*args, **kwargs):
        result = load(*args, **kwargs)
        require(result["verified"] and result["epoch"] == PAUSE and not result["skipped"], "unexpected restore boundary")
        _, _, model, opt, sched, *_ = args
        state = {"model": model.state_dict(), "optimizer": opt.state_dict(),
                 "scheduler": sched.state_dict(), "rng": checkpoint._rng_state()}
        for key, value in state.items():
            restored[key] = compare(paused[key], value)
        require(all(row["identical"] for row in restored.values()), "restored state differs before first update")
        return result
    seed_all(cfg.seed + 1)  # exercise restoration, rather than inheriting the reference RNG state
    with patch.object(checkpoint, "load_latest", verified_load):
        run = train(cfg, work, resume=True)
    require(len(restored) == 4, "checkpoint loader did not run")
    return run, restored


def config(arm, path):
    return load_config("full_arch", [f"model_variant={arm}", f"name=recovery_{arm}_{path}",
                                     f"epochs={EPOCHS}", f"num_training_samples={MESHES}",
                                     "num_validation_samples=2", "seed=0"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("export", "restore"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    device = resolve_device("cuda")
    require(torch.cuda.device_count() == 1 and torch.cuda.get_device_name(0) == "Tesla T4", "expected one visible Tesla T4")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    env = collect_metadata(device)
    env["driver_version"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", "0"], text=True).strip()
    env["compute_capability"] = list(torch.cuda.get_device_capability(device))
    cfg = config("baseline", "resumed")
    archive = (ROOT / cfg.data.raw_dir).parent / DATASET_ARCHIVE
    require(sha256_of(archive) == DATASET_SHA256, "dataset archive digest differs")
    split = prepare_split(cfg)
    out = {"stage": args.stage, "environment": env, "dataset_sha256": DATASET_SHA256,
           "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
           "pause_epoch": PAUSE, "updates_per_epoch": MESHES, "total_updates_per_arm": EPOCHS * MESHES, "arms": {}}
    args.output.mkdir(parents=True, exist_ok=True)
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cross-session-", dir=runs) as tmp:
        work = Path(tmp)
        if args.stage == "restore":
            source = read(args.source / "recovery-export.json")
            require(source["dataset_sha256"] == DATASET_SHA256 and source["deterministic_algorithms"], "export provenance differs")
            require(all(env[k] == source["environment"][k] for k in ENV_KEYS), "export environment or commit differs")
        for arm in ARMS:
            states, archives, restored_dirs = {}, {}, {}
            for path, epoch in (("reference", EPOCHS), ("resumed", PAUSE)):
                cfg = config(arm, path)
                name = f"state-{cfg.name}.tar"
                if args.stage == "export":
                    seed_all(cfg.seed)
                    run = train(cfg, work, stop_after_epoch=epoch)
                    archives[path] = pack(run, args.output / name)
                else:
                    exported = source["arms"][arm][path]["archive"]
                    tar = args.source / name
                    require(sha256_of(tar) == exported["sha256"], f"{name} differs from export manifest")
                    verified = unpack(tar, work)
                    require(Path(verified["run_dir"]).name == cfg.name and all(c["ok"] for c in verified["checkpoints"]),
                            f"{name} contains invalid checkpoint(s)")
                    archives[path] = exported
                restored_dirs[path] = work / cfg.name
                states[path] = inspect_state(restored_dirs[path], cfg, split, DATASET_SHA256, epoch, 1)
            if args.stage == "export":
                out["arms"][arm] = {path: {"archive": archives[path], "epoch": state["epoch"], "updates": state["updates"],
                                          "manifest": state["manifest"]} for path, state in states.items()}
            else:
                cfg = config(arm, "resumed")
                drill_root = work / f"drills-{arm}"
                drill_root.mkdir()
                failures = drills(restored_dirs["resumed"], cfg, split, DATASET_SHA256,
                                  args.source / archives["resumed"]["tar"], drill_root, 1)
                final, loaded = replay(states["resumed"], cfg, work)
                resumed = inspect_state(final, cfg, split, DATASET_SHA256, EPOCHS, 1)
                record = read(final / "train_metrics.json")
                require(record["resumed_from_verified_checkpoint"] and record["resumed_from_epoch"] == PAUSE
                        and record["iterations"] == EPOCHS * MESHES, "completion counters differ")
                reference = states["reference"]
                equivalence = {key: compare(reference[key], resumed[key]) for key in
                               ("model", "optimizer", "scheduler", "rng", "history", "selections")}
                out["arms"][arm] = {"rejections": failures, "restored_state": loaded, "final_equivalence": equivalence,
                                    "epochs": resumed["epoch"], "updates": resumed["updates"],
                                    "selected_epochs": {k: v["record"]["epoch"] for k, v in resumed["selections"].items()},
                                    "archives": archives, "passed": all(x["identical"] for x in equivalence.values())}
    out["wall_seconds"] = time.perf_counter() - start
    if args.stage == "restore":
        out["passed"] = all(record["passed"] for record in out["arms"].values())
    write_json(args.output / f"recovery-{args.stage}.json", out)
    print(json.dumps(out, indent=2))
    require(out.get("passed", True), "deterministic recovery comparison failed")


if __name__ == "__main__":
    main()
