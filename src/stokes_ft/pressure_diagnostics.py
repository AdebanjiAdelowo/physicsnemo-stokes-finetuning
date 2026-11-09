"""Decomposition of a pressure error into a level part, a spatial part and its location.

For a predicted vertex pressure ``p`` and the reference ``p_ref`` on the same mesh (P1, consistent
mass matrix ``M`` and stiffness matrix ``K`` of the Taylor-Hood pressure space):

* ``l2_p``: relative L2 error ``||e|| / ||p_ref||`` with ``e = p - p_ref``;
* ``l2_p_gauge_free``: the same after removing the best constant ``c = mean(e)`` (area-weighted);
  ``offset_share = 1 - (gauge-free / l2_p)^2`` is the fraction of the squared error that one
  global constant explains;
* ``grad_p``: relative error of the pressure gradient ``||grad e|| / ||grad p_ref||``. The momentum
  equation determines the gradient; the level additionally needs the outlet condition and a
  path from the outlet to the point, so a field can have an accurate gradient and a wrong level;
* ``share_upstream``, ``share_obstacle``, ``share_downstream``: fractions of ``e^T M e`` on the
  vertices left of, within, and right of the obstacle's x-range (lumped vertex areas);
* ``inlet_error``, ``outlet_error``: mean error on the inlet and outlet vertices.

Only fields and the mesh enter; nothing is fitted to the reference beyond the reported constant.
"""

import numpy as np
from skfem import asm
from skfem.models.poisson import laplace

from .data import INFLOW, OUTFLOW, POLYGON


def pressure_stiffness(diag):
    """P1 stiffness matrix of the Taylor-Hood pressure space of a ``WeakDiagnostics``."""
    return asm(laplace, diag.system.pressure).tocsr()


def decompose(diag, p: np.ndarray, p_ref: np.ndarray, K=None) -> dict:
    """Error decomposition of the vertex pressure ``p`` against ``p_ref`` (see the module docstring)."""
    s = diag.system
    K = pressure_stiffness(diag) if K is None else K
    x = s.pressure_from_vertices(p)
    xr = s.pressure_from_vertices(p_ref)
    e = x - xr
    M = diag.M_p
    norm = lambda d, A: float(np.sqrt(max(d @ (A @ d), 0.0)))  # noqa: E731
    ref_l2, ref_grad = norm(xr, M), norm(xr, K)
    offset = float(np.ones_like(e) @ (M @ e)) / diag.area
    l2 = norm(e, M) / ref_l2
    gauge_free = norm(e - offset, M) / ref_l2

    marker = diag.marker
    nodal = s.pressure.nodal_dofs[0]
    ev = e[nodal]
    lumped = np.asarray(M.sum(axis=1)).ravel()[nodal]
    xs = diag.coords[:, 0]
    poly = marker == POLYGON
    lo, hi = (xs[poly].min(), xs[poly].max()) if poly.any() else (np.inf, -np.inf)
    sq = lumped * ev ** 2
    total = sq.sum() if sq.sum() > 0 else 1.0
    return {
        "l2_p": l2,
        "l2_p_gauge_free": gauge_free,
        "offset": offset,
        "offset_share": float(1 - (gauge_free / l2) ** 2) if l2 > 0 else 0.0,
        "grad_p": norm(e, K) / ref_grad,
        "share_upstream": float(sq[xs < lo].sum() / total),
        "share_obstacle": float(sq[(xs >= lo) & (xs <= hi)].sum() / total),
        "share_downstream": float(sq[xs > hi].sum() / total),
        "inlet_error": float(ev[marker == INFLOW].mean()),
        "outlet_error": float(ev[marker == OUTFLOW].mean()),
    }
