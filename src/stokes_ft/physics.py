"""Stokes residuals and boundary-condition residuals used for evaluation.

Two evaluators of the same residual definition (the ``Stokes`` class of the fine-tuning script):

* ``nodal_residuals`` works on nodal values and the mesh, with the weighted least-squares
  gradient of PhysicsNeMo (``grad_method="least_squares"``, the operator of upstream
  ``pi_fine_tuning_gnn.py``). It applies to any field given on the nodes: the reference
  solution, the MeshGraphNet prediction and the fine-tuned field are treated identically.
* ``autodiff_residuals`` differentiates a coordinate network exactly. It applies only to the
  fine-tuning network and is the quantity that stage minimises.
"""

import numpy as np
import torch
from physicsnemo.sym.eq.gradients import compute_connectivity_tensor
from physicsnemo.sym.eq.phy_informer import PhysicsInformer

from .data import boundary_masks, mesh_edges
from .finetune import Stokes
from .metrics import rms

RESIDUAL_NAMES = ["continuity", "momentum_x", "momentum_y"]


def _informer(nu: float, grad_method: str, device="cpu") -> PhysicsInformer:
    return PhysicsInformer(
        required_outputs=RESIDUAL_NAMES,
        equations=Stokes(nu=nu, dim=2),
        grad_method=grad_method,
        device=device,
        compute_connectivity=False,
    )


def mesh_connectivity(n_nodes: int, triangles: np.ndarray):
    """CSR adjacency of the mesh nodes, built as in upstream pi_fine_tuning_gnn.py."""
    return compute_connectivity_tensor(torch.arange(n_nodes), torch.as_tensor(mesh_edges(triangles)))


def nodal_residuals(coords: np.ndarray, triangles: np.ndarray, fields: dict, nu: float) -> dict:
    """Residuals of nodal fields, single precision on the CPU.

    ``fields`` maps u, v, p to (N, 1) arrays. Returns (N,) arrays for ``continuity``
    (u_x + v_y), ``momentum_x`` (p_x - nu lap u) and ``momentum_y`` (p_y - nu lap v).
    """
    inputs = {key: torch.as_tensor(fields[key], dtype=torch.float32).reshape(-1, 1) for key in ("u", "v", "p")}
    inputs["coordinates"] = torch.as_tensor(coords, dtype=torch.float32)
    inputs["connectivity_tensor"] = mesh_connectivity(len(coords), triangles)
    out = _informer(nu, "least_squares").forward(inputs)
    return {name: out[name].detach().numpy().reshape(-1).astype(np.float64) for name in RESIDUAL_NAMES}


def autodiff_residuals(model, coords: np.ndarray, nu: float, device="cpu") -> dict:
    """Residuals of a coordinate network ``model({"x", "y"}) -> {"u", "v", "p"}`` at ``coords``."""
    points = torch.as_tensor(coords, dtype=torch.float32, device=device).requires_grad_(True)
    out = model({"x": points[:, 0:1], "y": points[:, 1:2]})
    res = _informer(nu, "autodiff", device).forward({"coordinates": points, **{k: out[k] for k in ("u", "v", "p")}})
    return {name: res[name].detach().cpu().numpy().reshape(-1).astype(np.float64) for name in RESIDUAL_NAMES}


def parabolic_inflow(y: np.ndarray, u_max: float) -> np.ndarray:
    return 4 * u_max * y * (0.4 - y) / (0.4**2)


def residual_norms(residuals: dict, interior: np.ndarray) -> dict:
    """Root mean square of each residual over the interior nodes."""
    out = {name: rms(residuals[name][interior]) for name in RESIDUAL_NAMES}
    out["momentum"] = rms(np.hypot(residuals["momentum_x"], residuals["momentum_y"])[interior])
    return out


def boundary_norms(coords: np.ndarray, marker: np.ndarray, fields: dict, u_max: float) -> dict:
    """Root mean square boundary mismatches.

    ``inflow``: |(u, v) - (u_in, 0)| on the inlet nodes. ``noslip``: |(u, v)| on the channel
    walls and the polygon. ``boundary``: both sets together, the conditions in the fine-tuning
    loss. ``outflow_pressure``: |p| on the outlet nodes, which no loss term constrains.
    """
    masks = boundary_masks(marker)
    u, v, p = (np.asarray(fields[key]).reshape(-1) for key in ("u", "v", "p"))
    target_u = np.where(masks["inflow"], parabolic_inflow(coords[:, 1], u_max), 0.0)
    mismatch = np.hypot(u - target_u, v)
    return {
        "inflow": rms(mismatch[masks["inflow"]]),
        "noslip": rms(mismatch[masks["noslip"]]),
        "boundary": rms(mismatch[masks["inflow"] | masks["noslip"]]),
        "outflow_pressure": rms(p[masks["outflow"]]),
    }
