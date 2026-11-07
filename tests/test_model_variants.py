"""Receptive-field variants of the MeshGraphNet (stokes_ft.model)."""

import numpy as np
import pytest
import torch
from physicsnemo.utils import load_checkpoint, save_checkpoint
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from conftest import channel_mesh
from stokes_ft.audit import hop_distance
from stokes_ft.config import load_config
from stokes_ft.data import mesh_edges
from stokes_ft.model import (GlobalLatentMeshGraphNet, SharedDepthMeshGraphNet, build_meshgraphnet, count_parameters,
                             receptive_field_hops)


def line_graph(nx):
    mesh = channel_mesh(nx, 3)
    triangles = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    n = mesh.n_points
    edges = to_undirected(torch.as_tensor(mesh_edges(triangles)).t())
    return Data(edge_index=edges, num_nodes=n), triangles, n


def cfg_for(variant, **extra):
    overrides = [f"+model_variant={variant}"] + [f"+{k}={v}" for k, v in extra.items()]
    return load_config("smoke", overrides)


def perturbation_change(model, nx=81, seed=0):
    graph, triangles, n = line_graph(nx)
    cfg = load_config("smoke")
    torch.manual_seed(seed)
    x = torch.randn(n, cfg.input_dim_nodes)
    e = torch.randn(graph.edge_index.shape[1], cfg.input_dim_edges)
    source = np.zeros(n, bool)
    source[0] = True
    hops = hop_distance(n, triangles, source)
    xp = x.clone()
    xp[0] += 1.0
    with torch.no_grad():
        change = (model(xp, e, graph) - model(x, e, graph)).abs().max(1).values.numpy()
    return change, hops


def test_default_is_the_upstream_meshgraphnet():
    model = build_meshgraphnet(load_config("smoke"))
    assert type(model).__name__ == "MeshGraphNet" and receptive_field_hops(model) == 15


@pytest.mark.parametrize("variant", ["baseline", "global_latent", "shared_depth"])
def test_output_shape(variant):
    graph, _, n = line_graph(11)
    cfg = cfg_for(variant)
    model = build_meshgraphnet(cfg).eval()
    out = model(torch.randn(n, cfg.input_dim_nodes), torch.randn(graph.edge_index.shape[1], cfg.input_dim_edges), graph)
    assert out.shape == (n, cfg.output_dim)


def test_parameter_counts_are_matched():
    base = count_parameters(build_meshgraphnet(cfg_for("baseline")))
    assert count_parameters(build_meshgraphnet(cfg_for("shared_depth"))) == base
    assert count_parameters(build_meshgraphnet(cfg_for("global_latent"))) == base + 2 * (128 * 128 + 128)


def test_global_latent_starts_as_the_baseline():
    torch.manual_seed(0)
    base = build_meshgraphnet(cfg_for("baseline")).eval()
    gl = build_meshgraphnet(cfg_for("global_latent")).eval()
    missing, unexpected = gl.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("global_mlp") for k in missing)
    graph, _, n = line_graph(11)
    x, e = torch.randn(n, 7), torch.randn(graph.edge_index.shape[1], 3)
    with torch.no_grad():
        assert torch.equal(base(x, e, graph), gl(x, e, graph))


def test_receptive_fields_are_as_designed():
    torch.manual_seed(0)
    base = build_meshgraphnet(cfg_for("baseline")).eval()
    change, hops = perturbation_change(base)
    assert change[hops > 15].max() == 0.0 and change[hops <= 15].max() > 0

    shared = build_meshgraphnet(cfg_for("shared_depth", processor_repeats=2)).eval()
    assert receptive_field_hops(shared) == 30
    change, hops = perturbation_change(shared)
    assert hops.max() > 35
    assert change[hops > 30].max() == 0.0
    assert change[(hops > 15) & (hops <= 30)].max() > 0  # sees beyond the baseline

    gl = build_meshgraphnet(cfg_for("global_latent")).eval()
    with torch.no_grad():
        for p in gl.global_mlp.parameters():
            p.normal_(0, 0.1)  # a trained, nonzero global path
    assert receptive_field_hops(gl) is None
    change, hops = perturbation_change(gl)
    assert change[hops > 30].min() > 0  # every node, however far, responds


@pytest.mark.parametrize("variant", ["global_latent", "shared_depth"])
def test_checkpoint_round_trip(variant, tmp_path):
    torch.manual_seed(0)
    cfg = cfg_for(variant)
    model = build_meshgraphnet(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01 * torch.randn_like(p))
    save_checkpoint(str(tmp_path), models=model, epoch=3)
    fresh = build_meshgraphnet(cfg).eval()
    assert load_checkpoint(str(tmp_path), models=fresh, device="cpu") == 3
    graph, _, n = line_graph(11)
    x, e = torch.randn(n, 7), torch.randn(graph.edge_index.shape[1], 3)
    with torch.no_grad():
        assert torch.equal(model(x, e, graph), fresh(x, e, graph))
    assert isinstance(fresh, (GlobalLatentMeshGraphNet, SharedDepthMeshGraphNet))


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        build_meshgraphnet(cfg_for("deeper"))


@pytest.mark.parametrize("variant", ["baseline", "global_latent", "shared_depth"])
def test_no_reference_information_reaches_the_prediction(variant):
    """The forward pass uses node and edge inputs only: changing the targets graph.y changes nothing."""
    graph, _, n = line_graph(11)
    cfg = cfg_for(variant)
    torch.manual_seed(0)
    model = build_meshgraphnet(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01 * torch.randn_like(p))
    x, e = torch.randn(n, 7), torch.randn(graph.edge_index.shape[1], 3)
    graph.y = torch.zeros(n, 3)
    with torch.no_grad():
        a = model(x, e, graph)
        graph.y = torch.randn(n, 3) * 100
        b = model(x, e, graph)
    assert torch.equal(a, b)
