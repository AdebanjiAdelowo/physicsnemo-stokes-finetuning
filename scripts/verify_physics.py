"""Checks of the residual formulation against the dataset, written to results/physics_checks.json.

    python scripts/verify_physics.py [--samples 20]

1. Viscosity and sign. The FEniCS reference fields are inserted into the P1 Galerkin weak form
   of ``grad p - nu lap u = 0`` on the given triangles, and nu is fitted by least squares over
   the interior nodes away from the boundaries. The inlet pressure gradient is compared with
   the Poiseuille value ``-8 nu U / H^2``. Neither uses PhysicsNeMo.
2. Boundary data of the reference: inlet profile, no-slip walls, outlet pressure.
3. Automatic-differentiation residual of the fine-tuning script on analytic fields.
4. Accuracy of the least-squares mesh derivative operator on analytic fields, on a dataset mesh.
5. Residual of the reference solution under that operator: the floor of the nodal residual metrics.
"""

import argparse

import numpy as np
import torch

from _bootstrap import ROOT
from stokes_ft.config import load_config
from stokes_ft.data import boundary_masks, prepare_split, read_sample
from stokes_ft.metrics import rms
from stokes_ft.physics import autodiff_residuals, boundary_norms, nodal_residuals, parabolic_inflow, residual_norms
from stokes_ft.provenance import collect_metadata, write_json


def p1_weak_terms(coords, triangles, u, v, p):
    """Per node i: a_i = int grad(w) . grad(phi_i) for w = u, v, and b_i = int p d(phi_i)/dx, dy."""
    x = coords[triangles]
    det = (x[:, 1, 0] - x[:, 0, 0]) * (x[:, 2, 1] - x[:, 0, 1]) - (x[:, 1, 1] - x[:, 0, 1]) * (x[:, 2, 0] - x[:, 0, 0])
    area = np.abs(det) / 2
    grad = np.zeros((len(triangles), 3, 2))
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        grad[:, a, 0] = (x[:, b, 1] - x[:, c, 1]) / det
        grad[:, a, 1] = (x[:, c, 0] - x[:, b, 0]) / det
    grad_u = np.einsum("tad,ta->td", grad, u[triangles])
    grad_v = np.einsum("tad,ta->td", grad, v[triangles])
    p_mean = p[triangles].mean(1)
    a_terms, b_terms = np.zeros((2, len(coords))), np.zeros((2, len(coords)))
    for a in range(3):
        np.add.at(a_terms[0], triangles[:, a], area * (grad_u * grad[:, a]).sum(1))
        np.add.at(a_terms[1], triangles[:, a], area * (grad_v * grad[:, a]).sum(1))
        np.add.at(b_terms[0], triangles[:, a], area * p_mean * grad[:, a, 0])
        np.add.at(b_terms[1], triangles[:, a], area * p_mean * grad[:, a, 1])
    return a_terms, b_terms


def distance_to_boundary(coords, marker):
    boundary = coords[marker > 0]
    return np.sqrt(((coords[:, None, :] - boundary[None, :, :]) ** 2).sum(-1)).min(1)


class Analytic(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, d):
        return dict(zip(("u", "v", "p"), self.fn(d["x"], d["y"])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=20, help="number of test meshes to use")
    args = parser.parse_args()
    cfg = load_config("official")
    nu, u_max = cfg.physics.nu, cfg.physics.u_max
    manifest = prepare_split(cfg)
    numbers = manifest["test"][: args.samples]

    per_sample, floors, bcs, sizes = [], [], [], []
    for n in numbers:
        s = read_sample(ROOT / cfg.data.raw_dir / f"res_{n}.vtp")
        coords, marker = s["coords"], s["marker"]
        u, v, p = (s[f"ref_{k}"][:, 0] for k in ("u", "v", "p"))
        a, b = p1_weak_terms(coords, s["triangles"], u, v, p)
        core = (marker == 0) & (distance_to_boundary(coords, marker) > 0.03)
        a_core, b_core = a[:, core].ravel(), b[:, core].ravel()
        strip = (coords[:, 0] > 0.02) & (coords[:, 0] < 0.15)
        per_sample.append({
            "number": n,
            "nodes": len(coords),
            "nu_fit": float((a_core * b_core).sum() / (a_core * a_core).sum()),
            "inlet_pressure_gradient": float(np.polyfit(coords[strip, 0], p[strip], 1)[0]),
        })
        fields = {k: s[f"ref_{k}"] for k in ("u", "v", "p")}
        floors.append(residual_norms(nodal_residuals(coords, s["triangles"], fields, nu), boundary_masks(marker)["interior"]))
        bcs.append(boundary_norms(coords, marker, fields, u_max))
        sizes.append([coords[:, 0].min(), coords[:, 0].max(), coords[:, 1].min(), coords[:, 1].max()])

    # analytic checks on the mesh of the first sample
    s = read_sample(ROOT / cfg.data.raw_dir / f"res_{numbers[0]}.vtp")
    coords, triangles, interior = s["coords"], s["triangles"], boundary_masks(s["marker"])["interior"]
    x, y = coords[:, :1], coords[:, 1:]
    poiseuille = Analytic(lambda x, y: (4 * u_max * y * (0.4 - y) / 0.16, 0 * x * y, 8 * nu * u_max / 0.16 * (1.5 - x) + 0 * y))
    manufactured = Analytic(lambda x, y: (torch.sin(2 * x) * torch.cos(3 * y), x**2 * y, x * y**2))
    res = autodiff_residuals(manufactured, coords, nu)
    um = np.sin(2 * x[:, 0]) * np.cos(3 * y[:, 0])
    expected = {
        "continuity": 2 * np.cos(2 * x[:, 0]) * np.cos(3 * y[:, 0]) + x[:, 0] ** 2,
        "momentum_x": y[:, 0] ** 2 + nu * 13 * um,
        "momentum_y": 2 * x[:, 0] * y[:, 0] - nu * 2 * y[:, 0],
    }
    autodiff = {
        "poiseuille_max_abs_residual": {k: float(np.abs(v).max()) for k, v in autodiff_residuals(poiseuille, coords, nu).items()},
        "manufactured_max_abs_deviation_from_analytic": {k: float(np.abs(res[k] - expected[k]).max()) for k in expected},
        "manufactured_max_abs_analytic": {k: float(np.abs(expected[k]).max()) for k in expected},
    }

    # least-squares operator: momentum_x of (u, 0, 0) is -nu lap u, continuity is u_x
    boundary = ~interior
    lsq = {}
    for name, u, u_x, lap in (
        ("poiseuille_u", parabolic_inflow(y, u_max), 0 * x, np.full_like(x, -8 * u_max / 0.16)),
        ("sin4x_cos5y", np.sin(4 * x) * np.cos(5 * y), 4 * np.cos(4 * x) * np.cos(5 * y), -41 * np.sin(4 * x) * np.cos(5 * y)),
    ):
        r = nodal_residuals(coords, triangles, {"u": u, "v": 0 * x, "p": 0 * x}, nu)
        lap_h, ux_h = -r["momentum_x"] / nu, r["continuity"]
        lsq[name] = {
            "first_derivative_rms_error_interior": rms((ux_h - u_x[:, 0])[interior]),
            "first_derivative_rms_error_boundary": rms((ux_h - u_x[:, 0])[boundary]),
            "first_derivative_rms_scale": rms(np.hypot(u_x[:, 0], 0.0)) or None,
            "laplacian_relative_error_interior": rms((lap_h - lap[:, 0])[interior]) / rms(lap[interior, 0]),
            "laplacian_relative_error_boundary": rms((lap_h - lap[:, 0])[boundary]) / rms(lap[boundary, 0]),
        }

    def stats(values):
        values = np.asarray(values, dtype=float)
        return {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())}

    sizes = np.array(sizes)
    out = {
        "environment": collect_metadata(torch.device("cpu")),
        "nu_in_code": nu,
        "u_max_in_code": u_max,
        "test_numbers": numbers,
        "domain": {"x_min": float(sizes[:, 0].min()), "x_max": float(sizes[:, 1].max()),
                   "y_min": float(sizes[:, 2].min()), "y_max": float(sizes[:, 3].max()),
                   "nodes": stats([r["nodes"] for r in per_sample])},
        "viscosity_fit_p1_weak_form": stats([r["nu_fit"] for r in per_sample]),
        "inlet_pressure_gradient": {**stats([r["inlet_pressure_gradient"] for r in per_sample]),
                                    "poiseuille_value": -8 * nu * u_max / 0.16},
        "reference_boundary_rms": {k: stats([b[k] for b in bcs]) for k in bcs[0]},
        "autodiff_residual": autodiff,
        "least_squares_operator": lsq,
        "reference_residual_floor_rms": {k: stats([f[k] for f in floors]) for k in floors[0]},
        "per_sample": per_sample,
    }
    write_json(ROOT / "results" / "physics_checks.json", out)
    for key in ("domain", "viscosity_fit_p1_weak_form", "inlet_pressure_gradient", "reference_boundary_rms",
                "autodiff_residual", "least_squares_operator", "reference_residual_floor_rms"):
        print(key, out[key])


if __name__ == "__main__":
    main()
