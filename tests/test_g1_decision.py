"""The pre-registered G1 decision rule (scripts/g1_decision.py) on synthetic outcomes."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from g1_decision import decide, falsification  # noqa: E402

BEFORE = np.linspace(0.1, 0.38, 10)


def test_ten_of_ten_clear_improvement_generalises():
    out = decide(BEFORE, 0.6 * BEFORE)
    assert out["improved"] == 10 and out["conclusion"].startswith("GENERALISES")


def test_nine_of_ten_with_interval_below_zero_generalises():
    after = 0.6 * BEFORE
    after[3] = BEFORE[3] * 1.05
    out = decide(BEFORE, after)
    assert out["improved"] == 9 and abs(out["uncertainty"]["sign_test_p"] - 0.0215) < 1e-3
    assert out["conclusion"].startswith("GENERALISES")


def test_eight_of_ten_is_not_established():
    after = 0.6 * BEFORE
    after[[2, 5]] = BEFORE[[2, 5]] * 1.05
    assert decide(BEFORE, after)["conclusion"].startswith("GENERALISATION NOT")


def test_nine_of_ten_with_interval_reaching_zero_is_not_established():
    after = 0.99 * BEFORE
    after[0] = BEFORE[0] * 3.0  # one large deterioration pulls the mean change above zero
    out = decide(BEFORE, after)
    assert out["improved"] == 9 and out["uncertainty"]["relative_change_mean_ci95"][1] >= 0
    assert out["conclusion"].startswith("GENERALISATION NOT")


def test_falsification_of_the_receptive_field_explanation():
    ref = np.linspace(0.4, 1.6, 10)
    assert falsification(ref, ref)["contradicted"]
    assert not falsification(ref, 0.695 + 0.001 * np.sin(ref))["contradicted"]
    assert not falsification(ref, 0.3 * ref)["contradicted"]  # follows the geometry but too weakly
