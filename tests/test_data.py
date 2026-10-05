import json

import numpy as np
import pytest

from stokes_ft.data import (
    boundary_masks, make_dataset, mesh_edges, prepare_split, read_sample, sample_number, split_numbers,
)


def test_split_is_seeded_disjoint_and_80_10_10():
    a = split_numbers(list(range(1000)), 0, 0.8, 0.1)
    assert a == split_numbers(list(reversed(range(1000))), 0, 0.8, 0.1)
    assert [len(a[k]) for k in ("train", "validation", "test")] == [800, 100, 100]
    assert sorted(a["train"] + a["validation"] + a["test"]) == list(range(1000))
    assert a != split_numbers(list(range(1000)), 1, 0.8, 0.1)


def test_sample_number():
    assert sample_number("some/dir_3/res_412.vtp") == 412
    with pytest.raises(ValueError):
        sample_number("graph_7.vtp")


def test_prepare_split_writes_and_verifies(tiny_cfg, tiny_root):
    root = tiny_root[0]
    manifest = prepare_split(tiny_cfg)
    assert manifest == prepare_split(tiny_cfg)  # second call verifies the existing split
    assert json.loads((root / "data" / "dataset" / "split.json").read_text()) == manifest
    for split in ("train", "validation", "test"):
        names = sorted(p.name for p in (root / "data" / "dataset" / split).iterdir())
        assert names == sorted(f"res_{n}.vtp" for n in manifest[split])
    other = tiny_cfg.copy()
    other.data.split_seed = 5
    with pytest.raises(RuntimeError, match="other settings"):
        prepare_split(other)


def test_dataset_order_features_and_statistics(tiny_cfg, tiny_root, tmp_path_factory):
    manifest = prepare_split(tiny_cfg)
    run_dir = tiny_root[0] / "runs" / "dataset_check"
    run_dir.mkdir(parents=True, exist_ok=True)
    train = make_dataset(tiny_cfg, "train", run_dir)
    test = make_dataset(tiny_cfg, "test", run_dir)
    assert (run_dir / "node_stats.json").exists() and (run_dir / "edge_stats.json").exists()
    assert [sample_number(p) for p in test.data_list] == manifest["test"][: tiny_cfg.num_test_samples]
    graph = train[0]
    assert graph.x.shape[1] == tiny_cfg.input_dim_nodes       # 2 coordinates and 5 marker classes
    assert graph.edge_attr.shape[1] == tiny_cfg.input_dim_edges
    assert graph.y.shape == (graph.x.shape[0], tiny_cfg.output_dim)
    # the test split is normalised with the training statistics
    sample = read_sample(test.data_list[0])
    u = test[0].y[:, 0].numpy() * float(train.node_stats["u_std"]) + float(train.node_stats["u_mean"])
    np.testing.assert_allclose(u, sample["ref_u"][:, 0], atol=1e-5)


def test_read_sample_and_masks(tiny_root):
    sample = read_sample(tiny_root[0] / "data" / "raw" / "results" / "res_0.vtp")
    n = len(sample["coords"])
    assert sample["coords"].shape == (n, 2) and sample["ref_u"].shape == (n, 1)
    assert "gnn_u" not in sample and "filtered_u" not in sample
    masks = boundary_masks(sample["marker"])
    assert sum(m.sum() for m in masks.values()) == n
    assert np.all(sample["coords"][masks["inflow"], 0] == 0.0)
    assert mesh_edges(sample["triangles"]).shape == (3 * len(sample["triangles"]), 2)
