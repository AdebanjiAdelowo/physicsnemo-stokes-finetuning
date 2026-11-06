"""Controlled variants of the physics-informed stage (experiments G4, G5, G6).

Every variant fits the same coordinate network (``build_finetune_network``, the same seed rule, so
the same Fourier matrix and initial weights as the upstream run of that mesh) to the same frozen
MeshGraphNet prediction, with the same optimiser, learning rate and number of iterations. Only the
loss changes. None of them updates the MeshGraphNet; each is a separate per-mesh fit.

=================  =========================================================================
``upstream``       the loss of ``PhysicsInformedFineTuner`` (reproduced exactly; a test checks it)
``data_only``      the three data terms only: the smoothing effect of refitting (G4)
``data_bc``        data terms and the soft inlet and no-slip penalties, no PDE terms
``outlet``         upstream plus the outlet condition ``nu du/dx - p = 0, nu dv/dx = 0`` (weight 10)
``outlet_free_p``  as ``outlet`` with the pressure data weight 0: the pressure is set by the
                   momentum equation and the outlet condition only (G5)
``hard_bc``        upstream PDE and data terms; inlet and no-slip imposed exactly by a
                   distance-function ansatz instead of penalties (G6)
=================  =========================================================================

Outputs go to ``<run_dir>/controls/<variant>/``; the prediction files of the run are never
modified, so the historical results stay as they are.
"""

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from physicsnemo.sym.eq.phy_informer import PhysicsInformer

from .data import INFLOW, OUTFLOW, POLYGON, WALL, read_sample
from .finetune import LOSS_NAMES, LOSS_WEIGHTS, Stokes, build_finetune_network, finetune_seed
from .provenance import atomic_path, device_synchronize, write_json

TERMS = LOSS_NAMES + ["outlet_u", "outlet_v"]
UPSTREAM = dict(zip(LOSS_NAMES, LOSS_WEIGHTS)) | {"outlet_u": 0, "outlet_v": 0}
PDE_TERMS = ("mom_u", "mom_v", "cont")
BC_TERMS = ("u_in", "v_in", "u_noslip", "v_noslip")


@dataclass(frozen=True)
class Variant:
    name: str
    weights: dict = field(default_factory=lambda: dict(UPSTREAM))
    hard_bc: bool = False

    @property
    def needs_pde(self) -> bool:
        return any(self.weights[k] for k in PDE_TERMS)

    @property
    def needs_outlet(self) -> bool:
        return bool(self.weights["outlet_u"] or self.weights["outlet_v"])


def _w(**changes) -> dict:
    return dict(UPSTREAM) | changes


VARIANTS = {
    v.name: v
    for v in (
        Variant("upstream"),
        Variant("data_only", _w(**{k: 0 for k in PDE_TERMS + BC_TERMS})),
        Variant("data_bc", _w(**{k: 0 for k in PDE_TERMS})),
        Variant("outlet", _w(outlet_u=10, outlet_v=10)),
        Variant("outlet_free_p", _w(p=0, outlet_u=10, outlet_v=10)),
        Variant("hard_bc", _w(**{k: 0 for k in BC_TERMS}), hard_bc=True),
    )
}


# ---- distance function for exact boundary conditions ------------------------------------
def polygon_segments(coords: np.ndarray, triangles: np.ndarray, marker: np.ndarray) -> np.ndarray:
    """Boundary edges of the mesh with both ends on the polygon, (S, 2, 2)."""
    from .data import mesh_edges

    edges = np.sort(mesh_edges(triangles), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]  # an edge of one triangle only
    on_polygon = (marker[boundary[:, 0]] == POLYGON) & (marker[boundary[:, 1]] == POLYGON)
    return coords[boundary[on_polygon]][:, :, :2]


def segment_distance(points: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
    """Smooth approximate distance to a set of segments, exactly zero on them (Sukumar and
    Srivastava, 2022, eq. for trimmed line segments, joined by the R-equivalence with m = 1).

    ``points`` (N, 2), ``segments`` (S, 2, 2); returns (N, 1). Its derivatives are smooth away from
    the segments, so it may be differentiated twice at points not on the polygon.
    """
    a, b = segments[:, 0], segments[:, 1]
    length = torch.linalg.norm(b - a, dim=1)
    centre = 0.5 * (a + b)
    d = points[:, None, :]
    f = ((d[..., 0] - a[:, 0]) * (b[:, 1] - a[:, 1]) - (d[..., 1] - a[:, 1]) * (b[:, 0] - a[:, 0])) / length
    t = ((length / 2) ** 2 - ((d - centre) ** 2).sum(-1)) / length
    varphi = torch.sqrt(t**2 + f**4)
    phi = torch.sqrt(f**2 + ((varphi - t) / 2) ** 2)
    return 1.0 / (1.0 / phi).sum(1, keepdim=True)


class HardBCNetwork(torch.nn.Module):
    """``u = g + D N_u``, ``v = D N_v``, ``p = N_p`` with ``D`` zero on the inlet, the walls and the
    polygon, and ``g`` equal to the inlet profile at x = 0 and zero on the walls and the polygon:

        D = (x / L) (y (H - y) / (H/2)^2) (phi / phi_scale),   g = u_in(y) phi / (phi + x)

    ``N`` is the network of the upstream variant, so the trainable parameters are identical.
    """

    def __init__(self, base, segments: np.ndarray, u_max: float, length: float = 1.5, height: float = 0.4):
        super().__init__()
        self.base = base
        self.register_buffer("segments", torch.as_tensor(segments, dtype=torch.float32))
        self.u_max, self.length, self.height = u_max, length, height
        self.phi_scale = 0.1

    def forward(self, inputs: dict) -> dict:
        x, y = inputs["x"], inputs["y"]
        out = self.base(inputs)
        phi = segment_distance(torch.cat([x, y], 1), self.segments)
        walls = y * (self.height - y) / (self.height / 2) ** 2
        D = (x / self.length) * walls * (phi / self.phi_scale)
        u_in = 4 * self.u_max * y * (self.height - y) / self.height**2
        g = u_in * phi / (phi + x)
        return {"u": g + D * out["u"], "v": D * out["v"], "p": out["p"]}


# ---- the fit -------------------------------------------------------------------------------
class VariantFineTuner:
    """Per-mesh coordinate-network fit with the loss of one ``Variant``."""

    def __init__(self, variant: Variant, device, sample: dict, nu: float, u_max: float, lr: float = 1e-3):
        self.variant, self.device, self.nu, self.u_max = variant, device, nu, u_max
        marker = sample["marker"]
        coords = sample["coords"]

        def t(a, grad=False):
            return torch.tensor(np.asarray(a), requires_grad=grad).float().to(device)

        self.gnn = {k: t(sample[f"gnn_{k}"]) for k in "uvp"}
        self.ref = {k: t(sample[f"ref_{k}"]) for k in "uvp"}
        self.coords = t(coords, True)
        self.coords_inflow = t(coords[marker == INFLOW], True)
        self.coords_noslip = t(coords[(marker == WALL) | (marker == POLYGON)], True)
        self.coords_outlet = t(coords[marker == OUTFLOW], True)
        base = build_finetune_network(device)
        if variant.hard_bc:
            self.model = HardBCNetwork(base, polygon_segments(coords, sample["triangles"], marker), u_max).to(device)
            # the distance function is singular on the polygon (where the ansatz fixes u exactly):
            # the PDE is collocated at the other nodes, on a separate tensor so that no derivative
            # is ever taken on the polygon (a NaN there would survive masking in the backward pass)
            self.coords_pde = t(coords[marker != POLYGON], True)
        else:
            self.model = base
            self.coords_pde = None
        if variant.needs_pde:
            self.informer = PhysicsInformer(required_outputs=["continuity", "momentum_x", "momentum_y"],
                                            equations=Stokes(nu=nu, dim=2), grad_method="autodiff", device=device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, fused=device.type == "cuda")

    def parameters_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def _eval(self, points):
        return self.model({"x": points[:, 0:1], "y": points[:, 1:2]})

    def loss(self) -> dict:
        w = self.variant.weights
        zero = torch.zeros((), device=self.device)
        terms = dict.fromkeys(TERMS, zero)
        if any(w[k] for k in BC_TERMS):
            y_in = self.coords_inflow[:, 1:2]
            u_in = 4 * self.u_max * y_in * (0.4 - y_in) / (0.4**2)
            inflow, noslip = self._eval(self.coords_inflow), self._eval(self.coords_noslip)
            terms["u_in"] = torch.mean((u_in - inflow["u"]) ** 2)
            terms["v_in"] = torch.mean(inflow["v"] ** 2)
            terms["u_noslip"] = torch.mean(noslip["u"] ** 2)
            terms["v_noslip"] = torch.mean(noslip["v"] ** 2)
        hard = self.coords_pde is not None
        out = self._eval(self.coords.detach() if hard else self.coords)
        if self.variant.needs_pde:
            pts = self.coords_pde if hard else self.coords
            o = self._eval(pts) if hard else out
            res = self.informer.forward({"coordinates": pts, "u": o["u"], "v": o["v"], "p": o["p"]})
            for key, name in (("mom_u", "momentum_x"), ("mom_v", "momentum_y"), ("cont", "continuity")):
                terms[key] = torch.mean(res[name] ** 2)
        for k in "uvp":
            terms[k] = torch.mean((self.gnn[k] - out[k]) ** 2)
        if self.variant.needs_outlet:
            terms.update(self.outlet_terms())
        return terms

    def outlet_terms(self) -> dict:
        """Mean squares of the outlet condition ``nu du/dn - p n = 0`` with ``n = (1, 0)``."""
        pts = self.coords_outlet
        o = self._eval(pts)
        du = torch.autograd.grad(o["u"].sum(), pts, create_graph=True)[0][:, 0:1]
        dv = torch.autograd.grad(o["v"].sum(), pts, create_graph=True)[0][:, 0:1]
        return {"outlet_u": torch.mean((self.nu * du - o["p"]) ** 2), "outlet_v": torch.mean((self.nu * dv) ** 2)}

    def total(self, terms: dict) -> torch.Tensor:
        return sum(self.variant.weights[k] * terms[k] for k in TERMS if self.variant.weights[k])

    def step(self) -> dict:
        terms = self.loss()
        loss = self.total(terms)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return terms

    @torch.no_grad()
    def errors(self) -> dict:
        out = self._eval(self.coords)
        return {f"rel_l2_{k}": (torch.linalg.norm(self.ref[k] - out[k]) / torch.linalg.norm(self.ref[k])).item() for k in "uvp"}

    def fields(self) -> dict:
        with torch.no_grad():
            out = self._eval(self.coords)
        return {k: out[k].detach().cpu().numpy().astype(np.float64) for k in "uvp"}


def run_variant(cfg, run_dir: Path, variant: Variant, index: int, device, iterations: int | None = None,
                log_every: int | None = None, log=print) -> dict:
    """Fit one variant on test mesh ``index`` of ``run_dir`` and write its outputs."""
    run_dir = Path(run_dir)
    iterations = cfg.pi_iters if iterations is None else iterations
    log_every = cfg.finetune.log_every if log_every is None else log_every
    out_dir = run_dir / "controls" / variant.name
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = read_sample(run_dir / "predictions" / f"graph_{index}.vtp")
    torch.manual_seed(finetune_seed(cfg, index))  # same draw as the upstream run of this mesh
    tuner = VariantFineTuner(variant, device, sample, cfg.physics.nu, cfg.physics.u_max, lr=cfg.pi_lr)
    rows = []
    device_synchronize(device)
    start = time.perf_counter()
    for it in range(iterations):
        terms = tuner.step()
        if it % log_every == 0 or it == iterations - 1:
            device_synchronize(device)
            row = {"iteration": it + 1, "seconds": time.perf_counter() - start}
            row.update({f"loss_{k}": float(v) for k, v in terms.items()})
            row["loss_total"] = float(tuner.total(terms))
            row.update(tuner.errors())
            rows.append(row)
            if not np.isfinite(row["loss_total"]):
                raise FloatingPointError(f"{variant.name} mesh {index}: non-finite loss at iteration {it + 1}")
    device_synchronize(device)
    seconds = time.perf_counter() - start

    with atomic_path(out_dir / f"history_{index}.csv") as tmp, open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fields = tuner.fields()
    with atomic_path(out_dir / f"fields_{index}.npz") as tmp:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **fields)
    with atomic_path(out_dir / f"network_{index}.pt") as tmp:
        torch.save(tuner.model.state_dict(), tmp)
    record = {
        "variant": variant.name, "weights": variant.weights, "hard_bc": variant.hard_bc, "index": index,
        "seed": finetune_seed(cfg, index), "device": str(device), "torch_threads": torch.get_num_threads(),
        "parameter_count": tuner.parameters_count(), "iterations": iterations, "seconds": seconds,
        "seconds_per_iteration": seconds / iterations, "final": rows[-1],
    }
    write_json(out_dir / f"record_{index}.json", record)
    log(f"{variant.name} mesh {index}: {iterations} iterations in {seconds:.1f} s, "
        f"rel L2 u {rows[-1]['rel_l2_u']:.3f} v {rows[-1]['rel_l2_v']:.3f} p {rows[-1]['rel_l2_p']:.3f}")
    return record


def load_variant_fields(run_dir: Path, name: str, index: int) -> dict | None:
    path = Path(run_dir) / "controls" / name / f"fields_{index}.npz"
    record = Path(run_dir) / "controls" / name / f"record_{index}.json"
    if not path.exists():
        return None
    data = np.load(path)
    return {k: data[k].reshape(-1, 1) for k in "uvp"} | {"record": json.loads(record.read_text())}
