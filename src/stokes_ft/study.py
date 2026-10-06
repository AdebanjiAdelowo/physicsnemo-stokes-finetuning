"""The study: data-driven training, prediction, physics-informed fine-tuning, evaluation."""

import json
from pathlib import Path

from omegaconf import OmegaConf

from .config import REPO_ROOT
from .data import DATASET_ARCHIVE, DATASET_SHA256, DATASET_URL, prepare_split, sha256_of
from .engine import predict_test, train_data_driven
from .evaluate import STATUS, evaluate_study
from .finetune import finetune_sample
from .provenance import collect_metadata, resolve_device, write_json

ALL_STAGES = ("train", "predict", "finetune", "evaluate")


def log(message: str) -> None:
    print(message, flush=True)


def run_study(cfg, stages=ALL_STAGES, resume: bool = False, runs_root: Path | None = None,
              evaluate_samples: list[int] | None = None) -> Path:
    """Run the requested stages of one configuration in ``runs/<cfg.name>``.

    ``evaluate_samples`` evaluates a subset of the configured test samples (a partial result).
    """
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        raise ValueError(f"unknown stages: {sorted(unknown)}")
    device = resolve_device(cfg.device)
    run_dir = Path(runs_root or REPO_ROOT / "runs") / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = prepare_split(cfg)
    archive = (REPO_ROOT / cfg.data.raw_dir).parent / DATASET_ARCHIVE
    session = collect_metadata(device)
    session["stages"] = list(stages)
    session["evaluate_samples"] = evaluate_samples
    meta_path = run_dir / "study_metadata.json"
    meta = json.loads(meta_path.read_text()) if resume and meta_path.exists() else {"sessions": []}
    config = OmegaConf.to_container(cfg, resolve=True)
    if meta["sessions"] and meta["config"] != config:
        raise RuntimeError(f"{run_dir} was started with another configuration; it cannot be resumed with this one.")
    meta.update({
        "study": cfg.name,
        "result_status": STATUS.get(cfg.name, "UNLABELLED"),
        "seed": cfg.seed,
        "config": config,
        "dataset": {
            "source": DATASET_URL,
            "archive_sha256_expected": DATASET_SHA256,
            "archive_sha256_found": sha256_of(archive) if archive.exists() else None,
            "files_total": manifest["files_total"],
            "split_seed": manifest["split_seed"],
            "split_sizes": {k: len(manifest[k]) for k in ("train", "validation", "test")},
            "train_numbers_used": manifest["train"][: cfg.num_training_samples],
            "validation_numbers_used": manifest["validation"][: cfg.num_validation_samples],
            "test_numbers_used": manifest["test"][: cfg.num_test_samples],
        },
    })
    meta["sessions"].append(session)
    write_json(meta_path, meta)
    log(f"study '{cfg.name}' on {device}, commit {session['git_commit']}, dirty: {session['git_dirty']}")

    if "train" in stages:
        train_data_driven(cfg, run_dir, device, resume=resume, log=log)
    if "predict" in stages:
        done = run_dir / "predict_metrics.json"
        files = [run_dir / "predictions" / f"graph_{i}.vtp" for i in range(cfg.num_test_samples)]
        if resume and done.exists() and all(p.exists() for p in files):
            log("predictions already written")
        else:
            predict_test(cfg, run_dir, device, log=log)
    if "finetune" in stages:
        out_dir = run_dir / "finetune"
        for index in cfg.finetune.samples:
            record_path = out_dir / f"finetune_{index}.json"
            if resume and record_path.exists() and json.loads(record_path.read_text())["iterations"] == cfg.pi_iters:
                log(f"sample {index} already fine-tuned")
                continue
            record = finetune_sample(cfg, run_dir / "predictions" / f"graph_{index}.vtp", index, device, out_dir, log=log)
            write_json(record_path, record)
    if "evaluate" in stages:
        evaluate_study(cfg, run_dir, samples=evaluate_samples, log=log)
    return run_dir
