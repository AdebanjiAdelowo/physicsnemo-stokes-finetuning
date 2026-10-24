import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from conftest import channel_mesh
from stokes_ft.audit import hop_distance, lumped_mass, p1_divergence_rms, pressure_offset, weighted_relative_l2
from stokes_ft.config import load_config
from stokes_ft.data import mesh_edges
from stokes_ft.fem import taylor_hood_solve
from stokes_ft.model import build_meshgraphnet

NU, U_MAX = 0.01, 0.3


def mesh_arrays(nx, ny, jitter=0.0):
    mesh = channel_mesh(nx, ny, jitter=jitter)
    coords = np.array(mesh.points[:, :2])
    triangles = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    return coords, triangles, mesh


def test_taylor_hood_reproduces_poiseuille_only_with_the_gradient_outlet_form():
    # Poiseuille flow satisfies nu du/dn - p n = 0 at the outlet but not 2 nu eps(u) n - p n = 0
    coords, triangles, mesh = mesh_arrays(16, 9, jitter=0.2)
    exact = {k: np.asarray(mesh.point_data[k]).reshape(-1, 1) for k in ("u", "v", "p")}
    grad_form = taylor_hood_solve(coords, triangles, NU, U_MAX, "grad")
    for key in ("u", "p"):
        np.testing.assert_allclose(grad_form[key], exact[key], atol=1e-10)
    np.testing.assert_allclose(grad_form["v"], 0.0, atol=1e-10)
    sym_form = taylor_hood_solve(coords, triangles, NU, U_MAX, "sym")
    assert np.abs(sym_form["v"]).max() > 1e-3


def test_lumped_mass_and_weighted_error():
    coords, triangles, _ = mesh_arrays(13, 7, jitter=0.2)
    mass = lumped_mass(coords, triangles)
    assert np.isclose(mass.sum(), 1.5 * 0.4)
    target = np.ones((len(coords), 2))
    assert np.isclose(weighted_relative_l2(1.1 * target, target, mass), 0.1)


def test_p1_divergence_is_exact_for_linear_fields():
    coords, triangles, _ = mesh_arrays(13, 7, jitter=0.2)
    x, y = coords[:, 0], coords[:, 1]
    assert np.isclose(p1_divergence_rms(coords, triangles, 2 * x - 3 * y, x + 5 * y), 7.0)
    assert p1_divergence_rms(coords, triangles, 0.4 - y, x) < 1e-12


def test_pressure_offset_separates_a_constant():
    coords, triangles, mesh = mesh_arrays(13, 7)
    mass = lumped_mass(coords, triangles)
    p = np.asarray(mesh.point_data["p"])
    shifted = pressure_offset(p + 0.2, p, mass)
    assert np.isclose(shifted["offset"], 0.2) and np.isclose(shifted["offset_share"], 1.0)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(len(p))
    noise -= (mass * noise).sum() / mass.sum()
    assert abs(pressure_offset(p + noise, p, mass)["offset_share"]) < 1e-12


def test_meshgraphnet_output_is_blind_beyond_its_message_passing_depth():
    """The prediction at a node cannot depend on inputs more than processor_size hops away."""
    coords, triangles, _ = mesh_arrays(41, 4)
    n = len(coords)
    cfg = load_config("smoke")
    model = build_meshgraphnet(cfg).eval()
    depth = len(model.processor.processor_layers) // 2
    edges = to_undirected(torch.as_tensor(mesh_edges(triangles)).t())
    graph = Data(edge_index=edges, num_nodes=n)
    torch.manual_seed(0)
    x = torch.randn(n, cfg.input_dim_nodes)
    edge_attr = torch.randn(edges.shape[1], cfg.input_dim_edges)
    source = np.zeros(n, bool)
    source[0] = True
    hops = hop_distance(n, triangles, source)
    x_perturbed = x.clone()
    x_perturbed[0] += 1.0
    with torch.no_grad():
        change = (model(x_perturbed, edge_attr, graph) - model(x, edge_attr, graph)).abs().max(1).values.numpy()
    assert hops.max() > depth + 5
    assert change[hops > depth].max() == 0.0
    assert change[hops <= depth].max() > 0.0
