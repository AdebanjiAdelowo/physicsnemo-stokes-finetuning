"""The pre-registered rule of scripts/arch_compare.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from arch_compare import verdict  # noqa: E402


def test_better_only_beyond_the_baseline_seed_range():
    base = [0.30, 0.32, 0.34]  # range 0.04
    assert verdict([0.20, 0.21, 0.22], base, higher_is_better=False)["verdict"] == "better"
    assert verdict([0.30, 0.31, 0.30], base, higher_is_better=False)["verdict"] == "indistinguishable"
    assert verdict([0.40, 0.41, 0.42], base, higher_is_better=False)["verdict"] == "worse"


def test_direction_for_higher_is_better_metrics():
    base = [0.0, 0.01, 0.02]
    assert verdict([0.5, 0.6, 0.55], base, higher_is_better=True)["verdict"] == "better"
    assert verdict([-0.5, -0.4, -0.45], base, higher_is_better=True)["verdict"] == "worse"
