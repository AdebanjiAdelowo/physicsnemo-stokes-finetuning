import numpy as np
import pytest

from stokes_ft.config import REPO_ROOT


def test_paired_rows_follow_simulation_numbers(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "scripts"))
    from selection_decision import ordered_rows

    rows = [{"number": 8}, {"number": 2}, {"number": 5}]
    data = {"run": "example", "epoch": 1, "splits": {"test": {"numbers": [2, 5, 8], "per_mesh": rows}}}
    assert [r["number"] for r in ordered_rows(data, "test", [2, 5, 8])] == [2, 5, 8]
    rows[0]["number"] = 5
    with pytest.raises(SystemExit, match="different or duplicate test meshes"):
        ordered_rows(data, "test", [2, 5, 8])


def test_paired_inlet_statistics_resample_whole_meshes(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "scripts"))
    from selection_decision import compare_test

    ref = np.arange(1.0, 5.0)
    final = [{"ref_p_inlet": x, "pred_p_inlet": 2 * x, "rel_l2_p": x} for x in ref]
    selected = [{"ref_p_inlet": x, "pred_p_inlet": x, "rel_l2_p": x - 0.5} for x in ref]
    resamples = np.array([[0, 0, 1, 3], [2, 1, 2, 1]])
    result = compare_test(final, selected, resamples)
    assert result["rel_l2_p"]["absolute_change_mean_ci95"] == [-0.5, -0.5]
    assert result["rel_l2_p"]["improved_meshes"] == len(ref)
    assert result["inlet_level"]["slope"]["change_ci95"] == [-1.0, -1.0]
    assert result["inlet_level"]["correlation"]["change"] == 0.0
