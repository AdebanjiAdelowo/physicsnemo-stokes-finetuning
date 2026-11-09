"""Verify the recovered full_select run before evaluating its checkpoints."""

import argparse
import csv
import json
import tarfile
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pyvista as pv
from omegaconf import OmegaConf

from _bootstrap import ROOT
from evaluate_surrogate import SF_DATALESS, stage_test_files
from stokes_ft import checkpoint
from stokes_ft.config import load_config
from stokes_ft.data import DATASET_ARCHIVE, DATASET_SHA256, prepare_split, read_sample, sample_number, sha256_of
from stokes_ft.metrics import field_errors
from stokes_ft.model import build_meshgraphnet, receptive_field_hops
from stokes_ft.provenance import write_json
from stokes_ft.selection import CRITERIA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-diagnostics", type=Path,
                        help="compare the ten saved predictions with the offline epoch-500 error measurements")
    args = parser.parse_args()
    run = ROOT / "runs/full_select"
    cfg = load_config("full_select")
    meta = json.loads((run / "study_metadata.json").read_text())
    assert meta["config"] == OmegaConf.to_container(cfg, resolve=True)
    assert meta["seed"] == cfg.seed
    split = prepare_split(cfg)
    train = split["train"][:cfg.num_training_samples]
    validation = split["validation"][:cfg.num_validation_samples]
    test = split["test"]
    assert len(train) == 500 and len(validation) == 10 and len(test) == 100
    assert len(set(train + validation + test)) == 610
    for name, numbers in (("train", train), ("validation", validation), ("test", test[:cfg.num_test_samples])):
        assert meta["dataset"][f"{name}_numbers_used"] == numbers

    raw = ROOT / cfg.data.raw_dir
    dataset_sha = sha256_of(raw.parent / DATASET_ARCHIVE)
    assert dataset_sha == DATASET_SHA256 == meta["dataset"]["archive_sha256_found"]
    archive_sha = sha256_of(args.archive)
    assert archive_sha == Path(f"{args.archive}.sha256").read_text().split()[0]
    with tarfile.open(args.archive) as tar:
        members = {Path(member.name).relative_to(run.name): member for member in tar if member.isfile()}
        state_file_count = len(members)
        assert set(members) == {p.relative_to(run) for p in run.rglob("*") if p.is_file()}
        for relative, member in members.items():
            with tar.extractfile(member) as source:
                assert source.read() == (run / relative).read_bytes(), relative
    origins = {"verified_disk": 0, "verified_archive": 0}
    with zipfile.ZipFile(raw.parent / DATASET_ARCHIVE) as archive:
        members = {Path(name).name: name for name in archive.namelist()}
        for number in train + validation + test:
            name = f"res_{number}.vtp"
            archived = archive.read(members[name])
            if getattr((raw / name).stat(), "st_flags", 0) & SF_DATALESS:
                origins["verified_archive"] += 1
            else:
                assert archived == (raw / name).read_bytes(), name
                origins["verified_disk"] += 1

    model = build_meshgraphnet(cfg)
    # The dataset fingerprints its full file lists before applying the configured limits.
    fp = checkpoint.fingerprint(cfg, run, model, split["train"], split["validation"])
    checks = {}
    for manifest in checkpoint.manifests(run / "checkpoints/data_driven"):
        ok, reason = checkpoint.verify(manifest, fp)
        assert ok, (manifest, reason)
        checks[manifest.name] = json.loads(manifest.read_text())["files"]
    assert (run / "checkpoints/data_driven/history.0.500.csv").read_bytes() == (run / "history.csv").read_bytes()
    with (run / "history.csv").open() as history_file:
        history = list(csv.DictReader(history_file))
    assert [int(row["epoch"]) for row in history] == list(range(1, cfg.epochs + 1))
    train_record = json.loads((run / "train_metrics.json").read_text())
    assert train_record["epochs"] == cfg.epochs
    assert train_record["iterations"] == cfg.epochs * len(train)
    assert train_record["parameter_count"] == fp["parameter_count"]
    assert train_record["status"] == "complete"
    status = json.loads((run / "status.json").read_text())
    assert status["state"] == "complete" and status["stages_requested"] == ["train", "predict"]

    selected = {}
    for name, criterion in CRITERIA.items():
        directory = run / "checkpoints/selected" / name
        record = json.loads((directory / "selection.json").read_text())
        epoch, errors = min(
            ((int(row["epoch"]), {k: float(row[f"val_rel_l2_{k}"]) for k in "uvp"}) for row in history),
            key=lambda entry: criterion(entry[1]),
        )
        assert record["epoch"] == epoch and record["tracked_since"] == 0
        assert record["errors"] == errors and record["score"] == criterion(errors)
        selected[name] = record | {"files": {
            path.name: sha256_of(path) for path in
            (directory / f"MeshGraphNet.0.{epoch}.mdlus", directory / f"checkpoint.0.{epoch}.pt")
        }}

    predictions = json.loads((run / "predict_metrics.json").read_text())
    assert predictions["checkpoint_epoch"] == cfg.epochs
    files = predictions["test_files"]
    assert list(map(sample_number, files)) == test[:cfg.num_test_samples]
    assert len(list((run / "predictions").glob("graph_*.vtp"))) == len(files)
    cpu_rows = {}
    if args.final_diagnostics:
        diagnostic = json.loads(args.final_diagnostics.read_text())
        assert diagnostic["run"] == str(run.relative_to(ROOT)) and diagnostic["epoch"] == cfg.epochs
        cpu_rows = {row["number"]: row for row in diagnostic["splits"]["test"]["per_mesh"]}
        assert set(cpu_rows) == set(test)
    differences = []
    with tempfile.TemporaryDirectory(prefix="stokes-verify-") as tmp:
        reference = Path(tmp) / "reference"
        stage_test_files(cfg, list(map(sample_number, files)), reference)
        for index, name in enumerate(files):
            path = run / "predictions" / f"graph_{index}.vtp"
            predicted = pv.read(path)
            original = pv.read(reference / name)
            assert np.array_equal(predicted.points, original.points)
            assert np.array_equal(predicted.faces, original.faces)
            assert all(np.array_equal(predicted[key], original[key]) for key in original.point_data)
            assert all(np.isfinite(predicted[f"pred_{k}"]).all() for k in "uvp")
            if cpu_rows:
                sample = read_sample(path)
                errors = field_errors({k: sample[f"gnn_{k}"] for k in "uvp"}, {k: sample[f"ref_{k}"] for k in "uvp"})
                number = sample_number(name)
                differences.append({"number": number} | {
                    k: abs(errors[k] - cpu_rows[number][f"rel_l2_{k}"]) for k in ("p", "velocity")
                })

    out = {
        "training": meta["sessions"][0], "config": meta["config"], "fingerprint": fp,
        "receptive_field_hops": receptive_field_hops(model), "dataset_archive_sha256": dataset_sha,
        "state_archive_sha256": archive_sha, "archive_files_verified": state_file_count,
        "evaluation_split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "evaluation_mesh_sources": origins,
        "regular_checkpoint_manifests": checks, "validation_selections": selected,
        "recovered_status": status, "train_metrics": train_record, "recovered_predictions": predictions,
    }
    if differences:
        out["cpu_cuda_error_agreement"] = {
            "absolute_relative_l2_difference_per_mesh": differences,
            "max": {k: max(row[k] for row in differences) for k in ("p", "velocity")},
        }
    write_json(args.output, out)
    print(f"Verified archive, checkpoints, dataset and {len(files)} epoch-{cfg.epochs} prediction meshes.")


if __name__ == "__main__":
    main()
