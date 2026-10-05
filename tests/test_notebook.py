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


def test_full_study_is_off_by_default_and_guarded():
    code = _code()
    assert re.search(r"^RUN_FULL = False\b", code[0], re.M)
    cells = [c for c in code if "--config full --resume" in c]
    assert len(cells) == 1
    # the step starts only behind `is not True` and a budget comparison, never by reducing the config
    assert "if RUN_FULL is not True:" in cells[0]
    assert "> hours_left()" in cells[0] and "was not reduced" in cells[0]
    assert cells[0].index("if RUN_FULL is not True:") < cells[0].index("--config full --resume")


def test_order_of_steps():
    text = "\n".join(_code())
    order = ['"checkout", "--quiet", REF', "assert head == REF", "pip install -q -r requirements.txt", "prepare_data.py",
             "pytest", "--config smoke device=cuda", "name=full_probe", "--config full --resume",
             "results.py package", "results.py bundle"]
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
