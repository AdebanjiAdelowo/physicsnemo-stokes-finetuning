import torch

from stokes_ft.config import load_config
from stokes_ft.data import make_dataset, prepare_split
from stokes_ft.model import build_meshgraphnet, count_parameters


def test_official_model_size():
    model = build_meshgraphnet(load_config("official"))
    assert count_parameters(model) == 2531587
    assert len(model.processor.processor_layers) == 30  # 15 edge blocks and 15 node blocks


def test_forward_shape(tiny_cfg, tiny_root):
    prepare_split(tiny_cfg)
    run_dir = tiny_root[0] / "runs" / "model_check"
    run_dir.mkdir(parents=True, exist_ok=True)
    graph = make_dataset(tiny_cfg, "train", run_dir)[0]
    model = build_meshgraphnet(tiny_cfg)
    out = model(graph.x, graph.edge_attr, graph)
    assert out.shape == (graph.x.shape[0], 3) and torch.isfinite(out).all()
