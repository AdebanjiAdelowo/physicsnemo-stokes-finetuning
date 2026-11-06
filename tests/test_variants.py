import numpy as np
import pytest
import torch

from conftest import channel_mesh
from stokes_ft.data import INFLOW, OUTFLOW, POLYGON, WALL
from stokes_ft.finetune import LOSS_NAMES, PhysicsInformedFineTuner
from stokes_ft.variants import VARIANTS, HardBCNetwork, VariantFineTuner, polygon_segments, segment_distance

NU, U_MAX = 0.01, 0.3


def sample_from_mesh(mesh, offset=0.0):
    coords = np.array(mesh.points[:, :2])
    s = {"coords": coords, "triangles": np.asarray(mesh.faces).reshape(-1, 4)[:, 1:],
         "marker": np.asarray(mesh.point_data["marker"]).astype(int)}
    for k in "uvp":
        s[f"ref_{k}"] = np.asarray(mesh.point_data[k]).reshape(-1, 1)
        s[f"gnn_{k}"] = s[f"ref_{k}"] + offset
    return s


def channel_with_hole(nx=31, ny=9, box=(0.5, 0.8, 0.1, 0.3)):
    """Structured channel mesh with the triangles inside ``box`` removed and the hole marked as polygon."""
    import pyvista as pv

    mesh = channel_mesh(nx, ny)
    pts = np.array(mesh.points)
    tri = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    centroid = pts[tri, :2].mean(1)
    x0, x1, y0, y1 = box
    inside = (centroid[:, 0] > x0) & (centroid[:, 0] < x1) & (centroid[:, 1] > y0) & (centroid[:, 1] < y1)
    tri = tri[~inside]
    used = np.unique(tri)
    remap = -np.ones(len(pts), int)
    remap[used] = np.arange(len(used))
    pts, tri = pts[used], remap[tri]
    marker = np.asarray(mesh.point_data["marker"])[used].copy()
    on_hole = (np.isclose(pts[:, 0], x0) | np.isclose(pts[:, 0], x1) | np.isclose(pts[:, 1], y0) | np.isclose(pts[:, 1], y1)) & \
              (pts[:, 0] >= x0 - 1e-9) & (pts[:, 0] <= x1 + 1e-9) & (pts[:, 1] >= y0 - 1e-9) & (pts[:, 1] <= y1 + 1e-9)
    marker[on_hole] = POLYGON
    out = pv.PolyData(pts, np.hstack([np.full((len(tri), 1), 3), tri]).ravel())
    for k in "uvp":
        out.point_data[k] = np.asarray(mesh.point_data[k])[used]
    out.point_data["marker"] = marker
    return out


def test_upstream_variant_reproduces_the_upstream_loss_and_steps():
    s = sample_from_mesh(channel_mesh(9, 5, v_amp=0.05), offset=0.01)
    m = s["marker"]
    torch.manual_seed(7)
    ref = PhysicsInformedFineTuner(torch.device("cpu"), s["gnn_u"], s["gnn_v"], s["gnn_p"], s["coords"],
                                   s["coords"][m == INFLOW], s["coords"][(m == WALL) | (m == POLYGON)], NU,
                                   s["ref_u"], s["ref_v"], s["ref_p"], u_max=U_MAX)
    torch.manual_seed(7)
    var = VariantFineTuner(VARIANTS["upstream"], torch.device("cpu"), s, NU, U_MAX)
    for _ in range(3):
        a = ref.train()
        b = var.step()
        for name, value in zip(LOSS_NAMES, a):
            assert torch.allclose(value, b[name], rtol=1e-6, atol=1e-10), name
    for p, q in zip(ref.model.parameters(), var.model.parameters()):
        assert torch.allclose(p, q, atol=1e-7)


def test_data_only_control_has_no_physics_or_boundary_terms_and_ignores_the_reference():
    v = VARIANTS["data_only"]
    assert not v.needs_pde and not v.needs_outlet and all(v.weights[k] == 0 for k in ("u_in", "u_noslip", "cont"))
    mesh = channel_mesh(9, 5, v_amp=0.05)
    a = sample_from_mesh(mesh, 0.0)
    b = dict(a, ref_u=a["ref_u"] + 1.0)  # a different reference must not change the loss
    torch.manual_seed(0)
    la = VariantFineTuner(v, torch.device("cpu"), a, NU, U_MAX).loss()
    torch.manual_seed(0)
    lb = VariantFineTuner(v, torch.device("cpu"), b, NU, U_MAX).loss()
    assert all(torch.equal(la[k], lb[k]) for k in la)
    assert float(la["mom_u"]) == 0.0 and float(la["u_noslip"]) == 0.0


def test_all_variants_share_initialisation_and_parameter_count():
    s = sample_from_mesh(channel_with_hole())
    counts, firsts = set(), []
    for v in VARIANTS.values():
        torch.manual_seed(3)
        t = VariantFineTuner(v, torch.device("cpu"), s, NU, U_MAX)
        counts.add(t.parameters_count())
        firsts.append(next(t.model.parameters()).detach().clone())
    assert counts == {49923}
    assert all(torch.equal(firsts[0], f) for f in firsts)


def test_segment_distance_vanishes_exactly_on_segments_only():
    segs = torch.tensor([[[0.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 1.0]]])
    on = torch.tensor([[0.3, 0.0], [1.0, 0.5], [1.0, 0.0]])
    off = torch.tensor([[0.3, 0.2], [2.0, 2.0], [0.5, -0.1]])
    assert torch.all(segment_distance(on, segs) < 1e-7)
    assert torch.all(segment_distance(off, segs) > 1e-3)
    # first-order agreement with the true distance close to a segment
    near = torch.tensor([[0.5, 1e-3]])
    assert segment_distance(near, segs[:1]).item() == pytest.approx(1e-3, rel=1e-2)


def test_hard_bc_network_imposes_inlet_and_no_slip_exactly_for_any_weights():
    mesh = channel_with_hole()
    s = sample_from_mesh(mesh)
    m = s["marker"]
    segs = polygon_segments(s["coords"], s["triangles"], m)
    assert len(segs) == 2 * (6 + 4)  # the box perimeter in mesh edges (0.3 / 0.05 and 0.2 / 0.05 per side)
    torch.manual_seed(0)
    from stokes_ft.finetune import build_finetune_network

    net = HardBCNetwork(build_finetune_network("cpu"), segs, U_MAX)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p))  # arbitrary weights
        pts = torch.tensor(s["coords"], dtype=torch.float32)
        out = net({"x": pts[:, 0:1], "y": pts[:, 1:2]})
        scale = 1.0 + max(net.base({"x": pts[:, 0:1], "y": pts[:, 1:2]})[k].abs().max().item() for k in "uv")
    u, v = out["u"].numpy().ravel(), out["v"].numpy().ravel()
    y = s["coords"][:, 1]
    np.testing.assert_allclose(u[m == INFLOW], 4 * U_MAX * y[m == INFLOW] * (0.4 - y[m == INFLOW]) / 0.16, atol=1e-6)
    noslip = (m == WALL) | (m == POLYGON)
    # exact up to single precision: the distance function rounds to about 1e-8 on the polygon,
    # multiplied by the network output (made large here by the perturbed weights)
    assert np.abs(u[noslip]).max() < 1e-7 * scale and np.abs(v[noslip]).max() < 1e-7 * scale
    assert np.abs(u[m == WALL]).max() < 1e-6 and np.abs(v[m == WALL]).max() < 1e-6
    assert np.abs(u[m == 0]).max() > 1e-3  # not trivially zero inside


def test_hard_bc_variant_takes_finite_steps_with_second_derivatives():
    s = sample_from_mesh(channel_with_hole(), offset=0.01)
    torch.manual_seed(0)
    t = VariantFineTuner(VARIANTS["hard_bc"], torch.device("cpu"), s, NU, U_MAX)
    for _ in range(3):
        terms = t.step()
    assert all(torch.isfinite(v) for v in terms.values())
    assert float(terms["mom_u"]) > 0


class _Analytic(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def forward(self, d):
        x, y = d["x"], d["y"]
        if self.kind == "poiseuille":
            return {"u": 4 * U_MAX * y * (0.4 - y) / 0.16 + 0 * x, "v": 0 * x, "p": 0.15 * (1.5 - x)}
        return {"u": x + 0 * y, "v": -y + 0 * x, "p": 0 * x}


def test_outlet_term_vanishes_for_poiseuille_and_not_for_stagnation_flow():
    s = sample_from_mesh(channel_mesh(13, 7))
    t = VariantFineTuner(VARIANTS["outlet"], torch.device("cpu"), s, NU, U_MAX)
    t.model = _Analytic("poiseuille")
    terms = t.outlet_terms()
    assert float(terms["outlet_u"]) < 1e-12 and float(terms["outlet_v"]) < 1e-12
    t.model = _Analytic("stagnation")
    assert float(t.outlet_terms()["outlet_u"]) == pytest.approx(NU**2, rel=1e-4)


def test_outlet_free_p_has_no_pressure_data_term():
    v = VARIANTS["outlet_free_p"]
    assert v.weights["p"] == 0 and v.weights["u"] == 1 and v.needs_outlet and v.needs_pde
