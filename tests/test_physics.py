import numpy as np
import torch

from conftest import channel_mesh
from stokes_ft.data import boundary_masks
from stokes_ft.finetune import Stokes
from stokes_ft.physics import autodiff_residuals, boundary_norms, nodal_residuals, residual_norms

NU = 0.01


class Analytic(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, d):
        return dict(zip(("u", "v", "p"), self.fn(d["x"], d["y"])))


def mesh_arrays(nx=31, ny=17):
    mesh = channel_mesh(nx, ny)
    coords = np.array(mesh.points[:, :2])
    triangles = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    return coords, triangles, np.asarray(mesh.point_data["marker"]).astype(int)


def test_equations_are_the_stokes_system():
    eq = {k: str(v) for k, v in Stokes(nu=NU, dim=2).equations.items()}
    assert set(eq) == {"continuity", "momentum_x", "momentum_y"}
    pde = Stokes(nu=NU, dim=2).equations
    import sympy

    x, y = sympy.symbols("x y")
    u, v, p = (sympy.Function(n)(x, y) for n in "uvp")
    assert sympy.simplify(pde["continuity"] - (u.diff(x) + v.diff(y))) == 0
    assert sympy.simplify(pde["momentum_x"] - (p.diff(x) - NU * (u.diff(x, 2) + u.diff(y, 2)))) == 0
    assert sympy.simplify(pde["momentum_y"] - (p.diff(y) - NU * (v.diff(x, 2) + v.diff(y, 2)))) == 0


def test_autodiff_residual_vanishes_for_poiseuille_flow():
    coords, _, _ = mesh_arrays()
    model = Analytic(lambda x, y: (4 * 0.3 * y * (0.4 - y) / 0.16, 0 * x * y, 0.15 * (1.5 - x) + 0 * y))
    res = autodiff_residuals(model, coords, NU)
    for name, values in res.items():
        assert np.abs(values).max() < 1e-5, name


def test_autodiff_residual_matches_a_manufactured_field():
    coords, _, _ = mesh_arrays()
    model = Analytic(lambda x, y: (torch.sin(2 * x) * torch.cos(3 * y), x**2 * y, x * y**2))
    res = autodiff_residuals(model, coords, NU)
    x, y = coords[:, 0], coords[:, 1]
    u = np.sin(2 * x) * np.cos(3 * y)
    np.testing.assert_allclose(res["continuity"], 2 * np.cos(2 * x) * np.cos(3 * y) + x**2, atol=1e-5)
    np.testing.assert_allclose(res["momentum_x"], y**2 + NU * 13 * u, atol=1e-5)  # sign and viscosity factor
    np.testing.assert_allclose(res["momentum_y"], 2 * x * y - NU * 2 * y, atol=1e-5)


def test_nodal_residual_of_linear_fields():
    coords, triangles, _ = mesh_arrays()
    x, y = coords[:, :1], coords[:, 1:]
    res = nodal_residuals(coords, triangles, {"u": 2 * x - 3 * y, "v": x + 5 * y, "p": 0.7 * x - 0.2 * y}, NU)
    np.testing.assert_allclose(res["continuity"], 7.0, atol=1e-3)
    np.testing.assert_allclose(res["momentum_x"], 0.7, atol=1e-3)
    np.testing.assert_allclose(res["momentum_y"], -0.2, atol=1e-3)


def test_nodal_residual_viscous_term_and_poiseuille():
    coords, triangles, marker = mesh_arrays()
    x, y = coords[:, :1], coords[:, 1:]
    # nodes at least two cells away from the boundary: the gradient stencil is applied twice
    core = (x[:, 0] > 0.11) & (x[:, 0] < 1.39) & (y[:, 0] > 0.051) & (y[:, 0] < 0.349)
    res = nodal_residuals(coords, triangles, {"u": y**2, "v": 0 * x, "p": 0 * x}, NU)
    np.testing.assert_allclose(res["momentum_x"][core], -2 * NU, rtol=0.05)
    poiseuille = {"u": 4 * 0.3 * y * (0.4 - y) / 0.16, "v": 0 * x, "p": 0.15 * (1.5 - x)}
    res = nodal_residuals(coords, triangles, poiseuille, NU)
    assert np.abs(res["momentum_x"][core]).max() < 0.05 * 0.15  # against the pressure gradient 0.15
    assert np.abs(res["continuity"][core]).max() < 1e-3
    norms = residual_norms(res, boundary_masks(marker)["interior"])
    assert set(norms) == {"continuity", "momentum_x", "momentum_y", "momentum"}
    assert norms["momentum"] >= norms["momentum_x"]


def test_boundary_norms():
    coords, _, marker = mesh_arrays()
    x, y = coords[:, :1], coords[:, 1:]
    exact = {"u": 4 * 0.3 * y * (0.4 - y) / 0.16, "v": 0 * x, "p": 0.15 * (1.5 - x)}
    norms = boundary_norms(coords, marker, exact, 0.3)
    assert max(norms.values()) < 1e-12
    shifted = {"u": exact["u"] + 0.03, "v": exact["v"] + 0.04, "p": exact["p"] + 0.5}
    norms = boundary_norms(coords, marker, shifted, 0.3)
    for key in ("inflow", "noslip", "boundary"):
        assert np.isclose(norms[key], 0.05)
    assert np.isclose(norms["outflow_pressure"], 0.5)
