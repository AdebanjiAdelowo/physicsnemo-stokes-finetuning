"""Quantities of the scientific audit (docs/SCIENTIFIC_AUDIT.md): mesh-weighted errors, the pressure
offset, and graph distances that bound what the MeshGraphNet can see."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

from .data import mesh_edges


def lumped_mass(coords: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Area associated with each node: a third of the area of every adjacent triangle."""
    a, b, c = (coords[triangles[:, k], :2] for k in range(3))
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    mass = np.zeros(len(coords))
    for k in range(3):
        np.add.at(mass, triangles[:, k], area / 3)
    return mass


def p1_divergence_rms(coords: np.ndarray, triangles: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
    """Area-weighted RMS of the divergence of the piecewise-linear interpolant of nodal (u, v).

    An operator independent of PhysicsNeMo; like any vertex-only operator it does not give the
    P2-P1 reference a zero divergence.
    """
    u, v = np.asarray(u, float).reshape(-1), np.asarray(v, float).reshape(-1)
    x = coords[triangles, :2]
    det = (x[:, 1, 0] - x[:, 0, 0]) * (x[:, 2, 1] - x[:, 0, 1]) - (x[:, 1, 1] - x[:, 0, 1]) * (x[:, 2, 0] - x[:, 0, 0])
    divergence = np.zeros(len(triangles))
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        divergence += u[triangles[:, a]] * (x[:, b, 1] - x[:, c, 1]) / det + v[triangles[:, a]] * (x[:, c, 0] - x[:, b, 0]) / det
    area = np.abs(det) / 2
    return float(np.sqrt((area * divergence**2).sum() / area.sum()))


def weighted_relative_l2(pred: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    """Relative L2 error with nodal quadrature weights, an approximation of the integral norm."""
    w = weights.reshape(-1, *([1] * (np.ndim(target) - 1)))
    return float(np.sqrt((w * (pred - target) ** 2).sum() / (w * target**2).sum()))


def pressure_offset(pred_p: np.ndarray, ref_p: np.ndarray, weights: np.ndarray) -> dict:
    """Split the pressure error into its weighted mean (a constant offset) and the remainder.

    ``offset_share`` is the fraction of the weighted squared error that the constant explains.
    """
    error = np.asarray(pred_p, float).reshape(-1) - np.asarray(ref_p, float).reshape(-1)
    offset = float((weights * error).sum() / weights.sum())
    total = float((weights * error**2).sum())
    remainder = float((weights * (error - offset) ** 2).sum())
    return {"offset": offset, "offset_share": 1.0 - remainder / total if total > 0 else 0.0}


def hop_distance(n_nodes: int, triangles: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Smallest number of mesh edges from any node in ``sources`` to every node."""
    edges = mesh_edges(triangles)
    adjacency = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n_nodes, n_nodes)).tocsr()
    return shortest_path(adjacency, unweighted=True, directed=False, indices=np.flatnonzero(sources)).min(0)
