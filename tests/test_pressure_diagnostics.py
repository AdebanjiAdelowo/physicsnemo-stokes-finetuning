"""Pressure-error decomposition on a channel mesh with the exact Poiseuille solution."""

import numpy as np
import pytest

from conftest import channel_mesh
from stokes_ft.pressure_diagnostics import decompose, pressure_stiffness
from stokes_ft.weak import WeakDiagnostics


@pytest.fixture(scope="module")
def setup():
    mesh = channel_mesh(nx=16, ny=8, jitter=0.2, seed=3)
    coords = np.asarray(mesh.points[:, :2], float)
    tri = mesh.faces.reshape(-1, 4)[:, 1:]
    marker = np.asarray(mesh.point_data["marker"]).copy()
    # mark a fake "obstacle" band of interior vertices between x = 0.6 and 0.9 as polygon nodes
    band = (coords[:, 0] > 0.6) & (coords[:, 0] < 0.9) & (marker == 0)
    marker[band] = 4
    diag = WeakDiagnostics(coords, tri, marker, 0.01, 0.3)
    p_ref = np.asarray(mesh.point_data["p"], float).reshape(-1, 1)
    return diag, coords, p_ref, pressure_stiffness(diag)


def test_exact_pressure_has_zero_error(setup):
    diag, _, p_ref, K = setup
    d = decompose(diag, p_ref, p_ref, K)
    assert d["l2_p"] == 0 and d["grad_p"] == 0 and d["offset"] == 0


def test_a_constant_shift_is_a_pure_level_error(setup):
    diag, _, p_ref, K = setup
    d = decompose(diag, p_ref + 0.05, p_ref, K)
    assert d["grad_p"] < 1e-6  # the gradient is unchanged (up to rounding)
    assert d["offset"] == pytest.approx(0.05)
    assert d["l2_p_gauge_free"] < 1e-6 and d["offset_share"] == pytest.approx(1.0)
    assert d["inlet_error"] == pytest.approx(0.05) and d["outlet_error"] == pytest.approx(0.05)


def test_an_upstream_step_is_located_upstream_and_not_removed_by_a_constant(setup):
    diag, coords, p_ref, K = setup
    step = np.where(coords[:, 0] < 0.6, 0.1, 0.0).reshape(-1, 1)
    d = decompose(diag, p_ref + step, p_ref, K)
    assert d["share_upstream"] > 0.9
    assert 0 < d["offset_share"] < 1 and d["l2_p_gauge_free"] > 0
    assert d["grad_p"] > 0  # the step's transition changes the gradient locally
    assert d["inlet_error"] == pytest.approx(0.1) and d["outlet_error"] == pytest.approx(0.0)


def test_shares_sum_to_one(setup):
    diag, coords, p_ref, K = setup
    rng = np.random.default_rng(0)
    d = decompose(diag, p_ref + 0.01 * rng.standard_normal(p_ref.shape), p_ref, K)
    assert d["share_upstream"] + d["share_obstacle"] + d["share_downstream"] == pytest.approx(1.0)
