"""Result archives remain importable after download renames their filename."""

import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("result_commands", SCRIPTS / "results.py")
results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(results)


def test_publish_renamed_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "ROOT", tmp_path)
    archive = tmp_path / "physicsnemo-stokes-finetuning-full.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("full/summary.json", '{"samples": 10}')
        z.writestr("full/finetune/finetune_0.json", '{}')
        z.writestr("full/checkpoints/metadata.json", '{}')
    target = results.publish(archive, force=False)
    assert target == tmp_path / "results" / "full"
    assert (target / "summary.json").read_text() == '{"samples": 10}'
    assert (target / "finetune" / "finetune_0.json").exists()
    assert not (target / "checkpoints").exists()
    with pytest.raises(SystemExit, match="exists"):
        results.publish(archive, force=False)


@pytest.mark.parametrize("entries", [
    {"../summary.json": "{}"},
    {"a/summary.json": "{}", "b/summary.json": "{}"},
    {"full/history.csv": "epoch"},
])
def test_invalid_archive_is_rejected_before_replacing_results(tmp_path, monkeypatch, entries):
    monkeypatch.setattr(results, "ROOT", tmp_path)
    existing = tmp_path / "results" / "full"
    existing.mkdir(parents=True)
    (existing / "summary.json").write_text("preserve")
    archive = tmp_path / "full.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name, content in entries.items():
            z.writestr(name, content)
    with pytest.raises(SystemExit):
        results.publish(archive, force=True)
    assert (existing / "summary.json").read_text() == "preserve"
