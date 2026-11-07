"""Static checks of the Kaggle launcher: it cannot be executed without a CUDA session."""

import ast
import json
import re

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
    cells = [c for c in code if "--config full --resume" in c]
    assert len(cells) == 1
    cell = cells[0]
    # the step starts only behind `is not True`; the session limit pauses the run, it never reduces the config
    assert "if RUN_FULL is not True:" in cell
    assert cell.index("if RUN_FULL is not True:") < cell.index("--config full --resume")
    assert "--max-hours {budget:.2f}" in cell and "was not reduced" in cell
    # the state is packed in a finally block, also when the study fails
    assert cell.index("finally:") < cell.index("session_state.py pack runs/full")
    restore = [c for c in code if "session_state.py unpack" in c]
    assert len(restore) == 1 and restore[0].startswith("if RESUME:")
    assert "assert len(found) == 1" in restore[0]


def test_order_of_steps():
    text = "\n".join(_code())
    order = ['"checkout", "--quiet", REF', "assert head == REF", "pip install -q -r requirements.txt", "prepare_data.py",
             "pytest", "session_state.py unpack", "--config smoke device=cuda", "device_checks.py",
             "name=full_probe", "--config full --resume", "session_state.py pack runs/full",
             "results.py package runs/full", "results.py bundle runs/full"]
    positions = [text.index(item) for item in order]
    assert positions == sorted(positions)
    assert text.count("tree_is_clean()") >= 3


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
    for name in ("full", "full-checkpoints", "full-figures"):
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
