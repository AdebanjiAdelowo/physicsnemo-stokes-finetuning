"""Static checks of the Kaggle launcher: it cannot be executed without a CUDA session."""

import ast
import json
import os
import re
import shlex
import shutil
import tarfile
import zipfile

import pytest

from stokes_ft.config import CONFIG_DIR, REPO_ROOT, load_config

NOTEBOOK = REPO_ROOT / "kaggle" / "run_cuda.ipynb"


def _code():
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["nbformat"] == 4
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_cells_parse_and_have_no_stored_output():
    nb = json.loads(NOTEBOOK.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == []


def test_full_study_is_off_by_default_and_resumable():
    code = _code()
    assert re.search(r"^RUN_FULL = False\b", code[0], re.M)
    assert re.search(r"^RESUME = False\b", code[0], re.M)
    assert re.search(r'^CONFIG = "full"', code[0], re.M)
    assert re.search(r'^STAGES = "train,predict,finetune,evaluate"', code[0], re.M)
    cells = [c for c in code if "--config {CONFIG} --resume" in c]
    assert len(cells) == 1
    cell = cells[0]
    # the step starts only behind `is not True`; the session limit pauses the run, it never reduces the config
    assert "if RUN_FULL is not True:" in cell
    assert cell.index("if RUN_FULL is not True:") < cell.index("--config {CONFIG} --resume")
    assert "--max-hours {budget:.2f}" in cell and "was not reduced" in cell
    # the state is packed in a finally block, also when the study fails
    assert cell.index("finally:") < cell.index("session_state.py pack runs/{CONFIG}")
    restore = [c for c in code if "session_state.py unpack" in c]
    assert len(restore) == 1 and restore[0].startswith("if RESUME:")
    assert "assert len(found) == 1" in restore[0]


def test_order_of_steps():
    text = "\n".join(_code())
    order = ['"checkout", "--quiet", REF', "assert head == REF", "pip install -q -r requirements.txt", "prepare_data.py",
             "pytest", "session_state.py unpack", "--config smoke device=cuda", "device_checks.py",
             "selection_smoke.py", "name=full_probe", "--config {CONFIG} --resume", "session_state.py pack runs/{CONFIG}",
             "results.py package runs/{CONFIG}", "results.py bundle runs/{CONFIG}"]
    positions = [text.index(item) for item in order]
    assert positions == sorted(positions)
    assert text.count("tree_is_clean()") >= 3


@pytest.mark.parametrize(("stages", "output"), [
    ("train", "train_metrics.json"),
    ("predict", "predictions/graph_0.vtp"),
    ("train,predict", "predict_metrics.json"),
    ("evaluate", "summary.json"),
    ("evaluate", None),
])
def test_completed_subset_packages_only_its_outputs(tmp_path, monkeypatch, stages, output):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "scripts"))
    from results import package
    from stokes_ft.session_state import pack

    run_dir = tmp_path / "runs" / "full_select"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(json.dumps({
        "state": "complete", "stages_requested": stages.split(","),
    }))
    if output:
        path = run_dir / output
        path.parent.mkdir(exist_ok=True)
        path.write_text("{}")

    def sh(command, cwd):
        args = shlex.split(command)
        if args[1] == "scripts/run_study.py":
            return
        if args[1] == "scripts/session_state.py":
            pack(cwd / args[3], cwd / args[4])
        elif args[1] == "scripts/results.py":
            package(cwd / args[3])
        else:
            assert args[0] == "cp", command
            shutil.copyfile(cwd / args[1], cwd / args[2])

    scope = {
        "RUN_FULL": True, "STAGES": stages, "CONFIG": run_dir.name,
        "REPO": tmp_path, "WORK": tmp_path, "full_projection_h": 1.0,
        "hours_left": lambda: 2.0, "sh": sh,
        "os": os, "json": json,
    }
    cell = next(cell for cell in _code() if "session_state.py pack" in cell)
    if output is None:
        with pytest.raises(SystemExit, match="has no summary.json; run the evaluate stage first"):
            exec(cell, scope)
        assert (tmp_path / "state-full_select.tar").is_file()
        assert not scope["full_done"]
        return
    exec(cell, scope)

    assert scope["full_done"]
    with tarfile.open(tmp_path / "state-full_select.tar") as archive:
        assert f"full_select/{output}" in archive.getnames()
    result = tmp_path / "physicsnemo-stokes-finetuning-full_select.zip"
    if stages == "evaluate":
        with zipfile.ZipFile(result) as archive:
            assert "full_select/summary.json" in archive.namelist()
    else:
        assert not result.exists()


def test_ref_must_be_a_full_sha():
    assert 'fullmatch(r"[0-9a-f]{40}", REF)' in _code()[0]


def test_referenced_scripts_and_configs_exist():
    text = "\n".join(_code())
    for script in set(re.findall(r"scripts/\w+\.py", text)):
        assert (REPO_ROOT / script).exists(), script
    for name in set(re.findall(r"--config (\w+)", text)):
        assert (CONFIG_DIR / f"{name}.yaml").exists(), name


def test_projection_matches_the_full_config():
    cfg = load_config("full")
    text = "\n".join(_code())
    assert f"* {cfg.epochs} * {cfg.num_training_samples} / 3600" in text
    assert f"* {cfg.pi_iters} * {len(cfg.finetune.samples)} / 3600" in text
    assert cfg.batch_size == 1  # one iteration per training mesh


def test_archive_names():
    text = "\n".join(_code())
    for name in ("{CONFIG}", "{CONFIG}-checkpoints", "{CONFIG}-figures"):
        assert f"physicsnemo-stokes-finetuning-{name}.zip" in text


RECOVERY = REPO_ROOT / "kaggle" / "recovery_test.ipynb"


def _recovery_code():
    nb = json.loads(RECOVERY.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == []
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_recovery_notebook_uses_the_same_setup_and_two_stages():
    code = _recovery_code()
    launcher = _code()
    for marker in ('"checkout", "--quiet", REF', "pip install -q -r requirements.txt", "RAPIDS_IMPORTS", "prepare_data.py"):
        assert sum(marker in c for c in code) == 1
        assert next(c for c in code if marker in c) == next(c for c in launcher if marker in c)
    assert re.search(r'^STAGE = "first"', code[0], re.M)
    assert 'fullmatch(r"[0-9a-f]{40}", REF)' in code[0]
    stage = code[-1]
    assert stage.index('STAGE == "first"') < stage.index("recovery_check.py first") < stage.index("session_state.py pack")
    assert stage.index('STAGE == "second"') < stage.index("session_state.py unpack") < stage.index("recovery_check.py second")
    for script in set(re.findall(r"scripts/\w+\.py", "\n".join(code))):
        assert (REPO_ROOT / script).exists(), script


def test_both_notebooks_check_out_the_same_commit():
    ref = re.compile(r'^REF = "([0-9a-f]{40})"', re.M)
    assert ref.search(_code()[0]).group(1) == ref.search(_recovery_code()[0]).group(1)


def test_recovery_notebook_passes_the_deterministic_flag_to_both_sessions():
    code = _recovery_code()
    assert re.search(r"^DETERMINISTIC = True\b", code[0], re.M)
    stage = code[-1]
    assert 'FLAG = " --deterministic" if DETERMINISTIC else ""' in stage
    assert '"python scripts/recovery_check.py first" + FLAG' in stage
    assert '"python scripts/recovery_check.py second" + FLAG' in stage


def test_architecture_probe_notebook_has_only_the_private_bounded_entry_point():
    folder = REPO_ROOT / "kaggle" / "full_arch_probe"
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    assert meta["is_private"] and meta["enable_gpu"] and meta["enable_internet"]
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert all(not meta[k] for k in ("dataset_sources", "competition_sources", "kernel_sources", "model_sources"))
    nb = json.loads((folder / meta["code_file"]).read_text())
    code = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            ast.parse(source)
            assert not cell["outputs"]
            code.append(source)
    assert re.search(r'^REF = "[0-9a-f]{40}"$', code[0], re.M)
    assert "head == REF and tree_is_clean()" in code[0]
    assert next(c for c in _code() if "RAPIDS_IMPORTS" in c) == code[1]
    assert code[-1].index("prepare_data.py") < code[-1].index("full_arch_probe.py")
    assert "run_study.py" not in "\n".join(code)
