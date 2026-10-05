import numpy as np

from stokes_ft.metrics import field_errors, relative_l2, rms


def test_relative_l2_and_rms():
    y = np.array([[3.0], [4.0]])
    assert relative_l2(y, y) == 0.0
    assert np.isclose(relative_l2(1.1 * y, y), 0.1)
    assert np.isclose(rms(np.array([3.0, 4.0])), np.sqrt(12.5))


def test_field_errors():
    ref = {"u": np.array([[1.0], [0.0]]), "v": np.array([[0.0], [2.0]]), "p": np.array([[2.0], [1.0]])}
    pred = {"u": ref["u"] + 0.1, "v": ref["v"].copy(), "p": 2 * ref["p"]}
    out = field_errors(pred, ref)
    assert np.isclose(out["u"], np.sqrt(0.02)) and out["v"] == 0.0 and np.isclose(out["p"], 1.0)
    assert np.isclose(out["velocity"], np.sqrt(0.02) / np.sqrt(5.0))
    assert np.isclose(out["total"], np.sqrt(0.02 + 5.0) / np.sqrt(10.0))
