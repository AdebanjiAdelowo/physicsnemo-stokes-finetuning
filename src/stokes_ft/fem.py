"""Taylor-Hood finite-element solve of the Stokes problem on a dataset mesh (scikit-fem).

Used to identify the discretisation of the FEniCS reference and as the classical baseline of the
study: P2 velocity, P1 pressure on the given triangles, the parabolic inlet profile, no slip on
every other boundary except the outlet, and one of two natural outlet conditions:

* ``"grad"``: weak form ``nu grad u : grad w``, outlet condition ``nu du/dn - p n = 0``
* ``"sym"``: weak form ``2 nu eps(u) : eps(w)``, outlet condition ``2 nu eps(u) n - p n = 0``

Inlet and outlet are the boundary facets at the smallest and largest x of the mesh.
"""

import time

import numpy as np
from scipy.sparse import bmat
from skfem import Basis, BilinearForm, ElementTriP1, ElementTriP2, ElementVector, MeshTri, asm, condense, solve
from skfem.helpers import ddot, div, grad, sym_grad

from .physics import parabolic_inflow

FORMS = ("grad", "sym")


def taylor_hood_solve(coords: np.ndarray, triangles: np.ndarray, nu: float, u_max: float, form: str = "grad") -> dict:
    """Solve on the mesh and return ``u``, ``v``, ``p`` at the mesh vertices as (N, 1) arrays,
    with ``seconds`` (assembly and direct solve) and ``dofs``."""
    if form not in FORMS:
        raise ValueError(f"form must be one of {FORMS}")

    @BilinearForm
    def viscous(u, w, _):
        return nu * ddot(grad(u), grad(w)) if form == "grad" else 2 * nu * ddot(sym_grad(u), sym_grad(w))

    @BilinearForm
    def divergence(u, q, _):
        return div(u) * q

    mesh = MeshTri(np.ascontiguousarray(coords[:, :2].T, dtype=float), np.ascontiguousarray(triangles.T))
    velocity = Basis(mesh, ElementVector(ElementTriP2()))
    pressure = velocity.with_element(ElementTriP1())
    start = time.perf_counter()
    B = asm(divergence, velocity, pressure)
    K = bmat([[asm(viscous, velocity), -B.T], [-B, None]], "csr")

    facet_x = mesh.p[0, mesh.facets]
    x_min, x_max = mesh.p[0].min(), mesh.p[0].max()
    tol = 1e-9 * (x_max - x_min)
    boundary = mesh.boundary_facets()
    inlet = boundary[(np.abs(facet_x[:, boundary] - x_min) < tol).all(0)]
    outlet = boundary[(np.abs(facet_x[:, boundary] - x_max) < tol).all(0)]
    noslip = np.setdiff1d(boundary, np.concatenate([inlet, outlet]))

    solution = np.zeros(K.shape[0])
    inlet_u = velocity.get_dofs(inlet).all("u^1")
    solution[inlet_u] = parabolic_inflow(velocity.doflocs[1, inlet_u], u_max)
    noslip_dofs = velocity.get_dofs(noslip).all()
    solution[noslip_dofs] = 0.0  # the inlet corners belong to the walls; the profile vanishes there
    fixed = np.unique(np.concatenate([velocity.get_dofs(inlet).all(), noslip_dofs]))
    solution = solve(*condense(K, np.zeros(K.shape[0]), x=solution, D=fixed))
    seconds = time.perf_counter() - start

    vel, p = solution[: velocity.N], solution[velocity.N :]
    vertex = velocity.nodal_dofs
    return {
        "u": vel[vertex[0]].reshape(-1, 1),
        "v": vel[vertex[1]].reshape(-1, 1),
        "p": p[pressure.nodal_dofs[0]].reshape(-1, 1),
        "seconds": seconds,
        "dofs": int(K.shape[0]),
    }
