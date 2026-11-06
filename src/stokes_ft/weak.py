"""FEM-consistent diagnostics of nodal fields: weak Stokes residuals, mesh norms, pressure recovery.

Why. Vertex-only operators (the least-squares nodal residual of ``physics.py``) give the exact
Taylor-Hood reference a nonzero residual, so a small nodal residual does not show that a field is
closer to the Stokes solution. The diagnostics here use the Taylor-Hood space itself:

* A vertex field (u, v, p), read as piecewise linear, is an exact member of the P2-P1 space (its
  mid-edge values are the means of the vertex values). Its **discrete weak residual** is therefore
  the residual of the Taylor-Hood equations, whose only zero with the dataset boundary data is the
  Taylor-Hood solution. The momentum rows are tested with every velocity test function that the
  inlet and no-slip conditions leave free, so the natural outlet condition ``nu du/dn - p n = 0``
  is included; the strong-form nodal residual cannot see it.
* Momentum residuals are measured in the discrete dual norm of H1, ``sqrt(r^T G^-1 r)`` with
  ``G`` the H1 Gram matrix of the free P2 dofs: the norm in which Stokes stability bounds the
  velocity error. Continuity is the exact L2 norm of ``div u`` of the piecewise-linear field.
* The vertex values of the reference are not the Taylor-Hood solution (its mid-edge values are
  lost), so the reference also has a nonzero weak residual. That floor is the interpolation error
  of any vertex-only representation; it is reported alongside every candidate.
* **Pressure recovery**: for a given velocity, the P1 pressure that minimises the momentum
  residual. The outlet rows fix its level (no gauge condition is needed). It answers what
  pressure the momentum equation and the outlet condition imply for that velocity, independent
  of the pressure the model predicted.

Three kinds of quantity are kept apart throughout: solution errors against the verified
reference, discrete (weak) residuals, and boundary-condition mismatches.
"""

import numpy as np
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import splu
from skfem import BilinearForm, asm
from skfem.helpers import ddot, dot, grad

from .data import INFLOW, OUTFLOW, POLYGON, WALL
from .fem import TaylorHoodSystem, taylor_hood_system
from .physics import parabolic_inflow


@BilinearForm
def _h1(u, w, _):
    return ddot(grad(u), grad(w)) + dot(u, w)


@BilinearForm
def _h1_semi(u, w, _):
    return ddot(grad(u), grad(w))


@BilinearForm
def _mass_vec(u, w, _):
    return dot(u, w)


@BilinearForm
def _mass_scalar(p, q, _):
    return p * q


class WeakDiagnostics:
    """Diagnostics of nodal fields on one dataset mesh. Assemble once, evaluate many fields."""

    def __init__(self, coords: np.ndarray, triangles: np.ndarray, marker: np.ndarray, nu: float, u_max: float):
        self.coords, self.triangles, self.marker = coords, triangles, np.asarray(marker).reshape(-1)
        self.nu, self.u_max = nu, u_max
        self.system: TaylorHoodSystem = taylor_hood_system(coords, triangles, nu, u_max, "grad")
        s = self.system
        self.free = s.free_velocity
        self.G = asm(_h1, s.velocity).tocsr()
        self._G_free = splu(self.G[self.free][:, self.free].tocsc())
        self.M_u = asm(_mass_vec, s.velocity).tocsr()
        self.S_u = asm(_h1_semi, s.velocity).tocsr()
        self.M_p = asm(_mass_scalar, s.pressure).tocsr()
        self._M_p_lu = splu(self.M_p.tocsc())
        self.area = float(self.M_p.sum())

    # ---- representation -------------------------------------------------------------------
    def coefficients(self, fields: dict) -> tuple[np.ndarray, np.ndarray]:
        s = self.system
        return s.velocity_from_vertices(fields["u"], fields["v"]), s.pressure_from_vertices(fields["p"])

    # ---- weak residuals --------------------------------------------------------------------
    def momentum_residual(self, vel: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Momentum rows ``A u - B^T p`` at the free velocity dofs (outlet included)."""
        s = self.system
        return (s.A @ vel - s.B.T @ p)[self.free]

    def residuals(self, fields: dict) -> dict:
        vel, p = self.coefficients(fields)
        r = self.momentum_residual(vel, p)
        momentum_dual = float(np.sqrt(max(r @ self._G_free.solve(r), 0.0)))
        rc = self.system.B @ vel  # (div u, q) for every P1 q
        continuity_dual = float(np.sqrt(max(rc @ self._M_p_lu.solve(rc), 0.0)))
        return {
            "weak_momentum_h1dual": momentum_dual,
            # continuity: L2 projection of div u onto P1 (dual norm in the pressure space), per unit area
            "weak_continuity_l2": continuity_dual / np.sqrt(self.area),
        }

    # ---- solution errors -------------------------------------------------------------------
    def errors(self, fields: dict, reference: dict) -> dict:
        """Mesh-consistent errors of the piecewise-linear fields against the reference.

        ``l2_velocity`` and ``h1semi_velocity`` are relative norms of the velocity difference
        (consistent mass and stiffness matrices). For pressure, ``l2_p`` is the relative error,
        ``l2_p_gauge_free`` the relative error after the best constant shift, and ``p_offset``
        that constant (the area-weighted mean pressure error).
        """
        vel, p = self.coefficients(fields)
        vel_r, p_r = self.coefficients(reference)
        dv, dp = vel - vel_r, p - p_r

        def rel(M, d, ref):
            return float(np.sqrt(d @ (M @ d)) / np.sqrt(ref @ (M @ ref)))

        ones = np.ones_like(p)
        offset = float((ones @ (self.M_p @ dp)) / self.area)
        dp0 = dp - offset
        return {
            "l2_velocity": rel(self.M_u, dv, vel_r),
            "h1semi_velocity": rel(self.S_u, dv, vel_r),
            "l2_p": rel(self.M_p, dp, p_r),
            "l2_p_gauge_free": float(np.sqrt(dp0 @ (self.M_p @ dp0)) / np.sqrt(p_r @ (self.M_p @ p_r))),
            "p_offset": offset,
        }

    # ---- pressure recovery -----------------------------------------------------------------
    def recover_pressure(self, u: np.ndarray | None = None, v: np.ndarray | None = None,
                         velocity_p2: np.ndarray | None = None) -> np.ndarray:
        """Vertex pressure minimising ``||A u - B^T p||`` in the dual H1 norm, for a given velocity.

        Solves the saddle system ``[[G, B^T], [B, 0]] [z; p] = [A u; 0]`` on the free dofs; the
        outlet rows of ``B^T`` make the pressure level unique. ``velocity_p2`` passes a full P2
        coefficient vector instead of vertex values.
        """
        s = self.system
        vel = s.velocity_from_vertices(u, v) if velocity_p2 is None else velocity_p2
        Gf = self.G[self.free][:, self.free]
        Bt = s.B.T.tocsr()[self.free]
        rhs = np.concatenate([(s.A @ vel)[self.free], np.zeros(s.pressure.N)])
        saddle = bmat([[Gf, Bt], [Bt.T, None]], "csc")
        solution = splu(saddle).solve(rhs)
        p = solution[len(self.free):]
        out = np.zeros(len(self.coords))
        out[:] = p[s.pressure.nodal_dofs[0]]
        return out.reshape(-1, 1)


def boundary_breakdown(coords: np.ndarray, marker: np.ndarray, fields: dict, u_max: float) -> dict:
    """RMS boundary mismatches split by boundary type: inlet, channel walls, polygon, outlet pressure."""
    marker = np.asarray(marker).reshape(-1)
    u, v, p = (np.asarray(fields[k], float).reshape(-1) for k in ("u", "v", "p"))
    target_u = np.where(marker == INFLOW, parabolic_inflow(coords[:, 1], u_max), 0.0)
    mismatch = np.hypot(u - target_u, v)

    def rms(mask, values):
        return float(np.sqrt(np.mean(values[mask] ** 2))) if mask.any() else float("nan")

    return {
        "bc_inflow": rms(marker == INFLOW, mismatch),
        "bc_wall": rms(marker == WALL, mismatch),
        "bc_polygon": rms(marker == POLYGON, mismatch),
        "outlet_p_rms": rms(marker == OUTFLOW, p),
    }


def pressure_levels(coords: np.ndarray, marker: np.ndarray, p: np.ndarray) -> dict:
    """Mean inlet and outlet pressure and the inlet-to-outlet drop."""
    marker = np.asarray(marker).reshape(-1)
    p = np.asarray(p, float).reshape(-1)
    inlet, outlet = float(p[marker == INFLOW].mean()), float(p[marker == OUTFLOW].mean())
    return {"p_inlet": inlet, "p_outlet": outlet, "p_drop": inlet - outlet}
