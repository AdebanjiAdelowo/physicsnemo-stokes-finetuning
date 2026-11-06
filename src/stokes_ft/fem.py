"""Taylor-Hood finite-element solve of the Stokes problem on a dataset mesh (scikit-fem).

Used to identify the discretisation of the FEniCS reference and as the classical baseline of the
study: P2 velocity, P1 pressure on the given triangles, the parabolic inlet profile, no slip on
every other boundary except the outlet, and one of two natural outlet conditions:

* ``"grad"``: weak form ``nu grad u : grad w``, outlet condition ``nu du/dn - p n = 0``
* ``"sym"``: weak form ``2 nu eps(u) : eps(w)``, outlet condition ``2 nu eps(u) n - p n = 0``

Inlet and outlet are the boundary facets at the smallest and largest x of the mesh.
"""

import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse import bmat, csr_matrix
from skfem import Basis, BilinearForm, ElementTriP1, ElementTriP2, ElementVector, MeshTri, asm, condense, solve
from skfem.helpers import ddot, div, grad, sym_grad

from .physics import parabolic_inflow

FORMS = ("grad", "sym")


@dataclass
class TaylorHoodSystem:
    """The assembled Taylor-Hood saddle-point system ``K = [[A, -B^T], [-B, 0]]`` on one mesh.

    ``dirichlet`` holds the velocity dofs fixed by the inlet and no-slip conditions; the outlet
    dofs are free, so the natural outlet condition is part of the momentum rows.
    """

    mesh: MeshTri
    velocity: Basis
    pressure: Basis
    A: csr_matrix
    B: csr_matrix
    K: csr_matrix
    dirichlet: np.ndarray
    lifting: np.ndarray  # full-length vector with the inlet profile on the Dirichlet dofs, zero elsewhere

    @property
    def free_velocity(self) -> np.ndarray:
        return np.setdiff1d(np.arange(self.velocity.N), self.dirichlet)

    def velocity_from_vertices(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """P2 coefficient vector of the piecewise-linear interpolant of vertex values.

        A P1 function is exactly a P2 function whose mid-edge value is the mean of the two
        vertex values, so a vertex field is a member of the Taylor-Hood velocity space.
        """
        u, v = np.asarray(u, float).reshape(-1), np.asarray(v, float).reshape(-1)
        out = np.zeros(self.velocity.N)
        nodal, facet = self.velocity.nodal_dofs, self.velocity.facet_dofs
        for comp, values in enumerate((u, v)):
            out[nodal[comp]] = values
            out[facet[comp]] = values[self.mesh.facets].mean(0)
        return out

    def pressure_from_vertices(self, p: np.ndarray) -> np.ndarray:
        out = np.zeros(self.pressure.N)
        out[self.pressure.nodal_dofs[0]] = np.asarray(p, float).reshape(-1)
        return out


def taylor_hood_system(coords: np.ndarray, triangles: np.ndarray, nu: float, u_max: float,
                       form: str = "grad", intorder: int | None = None) -> TaylorHoodSystem:
    """Assemble the Taylor-Hood system of the dataset problem on the given triangles."""
    if form not in FORMS:
        raise ValueError(f"form must be one of {FORMS}")

    @BilinearForm
    def viscous(u, w, _):
        return nu * ddot(grad(u), grad(w)) if form == "grad" else 2 * nu * ddot(sym_grad(u), sym_grad(w))

    @BilinearForm
    def divergence(u, q, _):
        return div(u) * q

    mesh = MeshTri(np.ascontiguousarray(coords[:, :2].T, dtype=float), np.ascontiguousarray(triangles.T))
    velocity = Basis(mesh, ElementVector(ElementTriP2()), intorder=intorder)
    pressure = velocity.with_element(ElementTriP1())
    A = asm(viscous, velocity)
    B = asm(divergence, velocity, pressure)
    K = bmat([[A, -B.T], [-B, None]], "csr")

    facet_x = mesh.p[0, mesh.facets]
    x_min, x_max = mesh.p[0].min(), mesh.p[0].max()
    tol = 1e-9 * (x_max - x_min)
    boundary = mesh.boundary_facets()
    inlet = boundary[(np.abs(facet_x[:, boundary] - x_min) < tol).all(0)]
    outlet = boundary[(np.abs(facet_x[:, boundary] - x_max) < tol).all(0)]
    noslip = np.setdiff1d(boundary, np.concatenate([inlet, outlet]))

    lifting = np.zeros(K.shape[0])
    inlet_u = velocity.get_dofs(inlet).all("u^1")
    lifting[inlet_u] = parabolic_inflow(velocity.doflocs[1, inlet_u], u_max)
    noslip_dofs = velocity.get_dofs(noslip).all()
    lifting[noslip_dofs] = 0.0  # the inlet corners belong to the walls; the profile vanishes there
    fixed = np.unique(np.concatenate([velocity.get_dofs(inlet).all(), noslip_dofs]))
    return TaylorHoodSystem(mesh, velocity, pressure, A.tocsr(), B.tocsr(), K, fixed, lifting)


def taylor_hood_solve(coords: np.ndarray, triangles: np.ndarray, nu: float, u_max: float, form: str = "grad") -> dict:
    """Solve on the mesh and return ``u``, ``v``, ``p`` at the mesh vertices as (N, 1) arrays,
    with ``seconds`` (basis construction, assembly and direct solve; the audit timed assembly
    and solve only) and ``dofs``."""
    start = time.perf_counter()
    system = taylor_hood_system(coords, triangles, nu, u_max, form)
    K = system.K
    solution = solve(*condense(K, np.zeros(K.shape[0]), x=system.lifting.copy(), D=system.dirichlet))
    seconds = time.perf_counter() - start

    velocity, pressure = system.velocity, system.pressure
    vel, p = solution[: velocity.N], solution[velocity.N :]
    vertex = velocity.nodal_dofs
    return {
        "u": vel[vertex[0]].reshape(-1, 1),
        "v": vel[vertex[1]].reshape(-1, 1),
        "p": p[pressure.nodal_dofs[0]].reshape(-1, 1),
        "seconds": seconds,
        "dofs": int(K.shape[0]),
        "velocity_p2": vel,  # full P2 coefficient vector (vertex and mid-edge values)
    }
