"""Validation of the FEM-consistent diagnostics on fields with known answers.

Poiseuille flow u = 4 U y (0.4 - y) / 0.16, v = 0, p = 0.15 (1.5 - x) solves the dataset problem
exactly, outlet condition included (p = 0 and du/dx = 0 at x = 1.5). Couette-type flow (y, 0, 0)
solves the Stokes equations with the natural outlet condition and is piecewise linear, so its
vertex interpolant must have zero weak momentum residual.
"""

import numpy as np
import pytest

from conftest import channel_mesh
from stokes_ft.fem import taylor_hood_solve
from stokes_ft.weak import WeakDiagnostics, boundary_breakdown, pressure_levels

NU, U_MAX = 0.01, 0.3


def setup(nx, ny, jitter=0.2):
    mesh = channel_mesh(nx, ny, jitter=jitter)
    coords = np.array(mesh.points[:, :2])
    triangles = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    marker = np.asarray(mesh.point_data["marker"])
    exact = {k: np.asarray(mesh.point_data[k]).reshape(-1, 1) for k in ("u", "v", "p")}
    return WeakDiagnostics(coords, triangles, marker, NU, U_MAX), coords, triangles, exact


def test_exact_piecewise_linear_stokes_field_has_zero_weak_residual():
    d, coords, _, _ = setup(13, 7)
    y = coords[:, 1:2]
    couette = {"u": y, "v": 0 * y, "p": 0 * y}
    assert d.residuals(couette)["weak_momentum_h1dual"] < 1e-13


def test_weak_residual_sees_the_outlet_condition():
    # interior equations hold for both fields; only the outlet traction differs
    d, coords, _, _ = setup(13, 7)
    x, y = coords[:, 0:1], coords[:, 1:2]
    shifted = {"u": y, "v": 0 * y, "p": 0 * y + 0.1}            # constant pressure: -p n != 0 at the outlet
    stagnation = {"u": x, "v": -y, "p": 0 * y}                   # nu du/dx = nu at the outlet
    assert d.residuals(shifted)["weak_momentum_h1dual"] > 1e-3
    assert d.residuals(stagnation)["weak_momentum_h1dual"] > 1e-4
    assert d.residuals(stagnation)["weak_continuity_l2"] < 1e-12  # divergence free


def test_taylor_hood_solution_is_the_zero_of_the_weak_residual():
    d, coords, triangles, _ = setup(13, 7)
    fem = taylor_hood_solve(coords, triangles, NU, U_MAX)
    p = d.system.pressure_from_vertices(fem["p"])
    r = d.momentum_residual(fem["velocity_p2"], p)
    assert np.abs(r).max() < 1e-12
    # its vertex values alone are also exact here, because Poiseuille flow is quadratic along y only
    # on a jittered mesh the linear interpolant is not exact: the floor is the interpolation error
    assert d.residuals({k: fem[k] for k in "uvp"})["weak_momentum_h1dual"] > 1e-6


def test_weak_residual_of_the_interpolated_exact_solution_converges_at_first_order():
    values, sizes = [], []
    for n in (9, 17, 33):
        d, _, _, exact = setup(2 * n - 1, n, jitter=0.0)
        values.append(d.residuals(exact)["weak_momentum_h1dual"])
        sizes.append(0.4 / (n - 1))
    rates = np.diff(np.log(values)) / np.diff(np.log(sizes))
    assert values[-1] < values[0]
    assert np.all(rates > 0.8), rates  # O(h): the H1 interpolation error of a quadratic


def test_weak_residual_grows_with_the_distance_from_the_solution():
    d, coords, _, exact = setup(25, 9)
    rng = np.random.default_rng(0)
    bump = np.sin(np.pi * coords[:, 0:1] / 1.5) * np.sin(np.pi * coords[:, 1:2] / 0.4)
    smooth_noise = bump * rng.standard_normal()
    previous = d.residuals(exact)["weak_momentum_h1dual"]
    for eps in (0.01, 0.03, 0.1):
        perturbed = {"u": exact["u"] + eps * smooth_noise, "v": exact["v"], "p": exact["p"]}
        current = d.residuals(perturbed)["weak_momentum_h1dual"]
        assert current > previous
        previous = current


def test_pressure_recovery_from_the_exact_velocity():
    # with the full Taylor-Hood velocity the exact pressure has zero residual, so it is recovered
    d, coords, triangles, exact = setup(13, 7)
    fem = taylor_hood_solve(coords, triangles, NU, U_MAX)
    np.testing.assert_allclose(d.recover_pressure(velocity_p2=fem["velocity_p2"]), exact["p"], atol=1e-10)
    # from vertex values only, the lost mid-edge information leaves an error of a few percent
    vertex_only = d.recover_pressure(fem["u"], fem["v"])
    assert 1e-4 < np.abs(vertex_only - exact["p"]).max() < 0.05


def test_pressure_recovery_converges_and_fixes_the_level_without_a_gauge():
    errors = []
    for n in (9, 17, 33):
        d, _, _, exact = setup(2 * n - 1, n, jitter=0.0)
        p = d.recover_pressure(exact["u"], exact["v"])
        errors.append(np.sqrt(np.mean((p - exact["p"]) ** 2)) / np.sqrt(np.mean(exact["p"] ** 2)))
    assert errors[-1] < errors[0] and errors[-1] < 0.02


def test_mesh_errors_and_gauge_split():
    d, _, _, exact = setup(13, 7)
    assert d.errors(exact, exact)["l2_velocity"] == 0.0
    shifted = dict(exact, p=exact["p"] + 0.05)
    e = d.errors(shifted, exact)
    assert e["p_offset"] == pytest.approx(0.05)
    assert e["l2_p_gauge_free"] < 1e-12 and e["l2_p"] > 0
    scaled = dict(exact, u=1.1 * exact["u"])
    assert d.errors(scaled, exact)["l2_velocity"] == pytest.approx(0.1)
    assert d.errors(scaled, exact)["h1semi_velocity"] == pytest.approx(0.1)


def test_boundary_breakdown_and_levels():
    d, coords, _, exact = setup(13, 7)
    marker = d.marker
    b = boundary_breakdown(coords, marker, exact, U_MAX)
    assert b["bc_inflow"] < 1e-12 and b["bc_wall"] < 1e-12 and np.isnan(b["bc_polygon"])
    assert b["outlet_p_rms"] < 1e-12
    levels = pressure_levels(coords, marker, exact["p"])
    assert levels["p_drop"] == pytest.approx(0.15 * 1.5)
