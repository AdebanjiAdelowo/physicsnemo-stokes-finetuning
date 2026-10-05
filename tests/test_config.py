from omegaconf import OmegaConf

from stokes_ft.config import load_config

# examples/cfd/stokes_mgn/conf/config.yaml at the pinned upstream commit
UPSTREAM = {
    "input_dim_nodes": 7, "input_dim_edges": 3, "output_dim": 3,
    "hidden_dim_node_encoder": 256, "hidden_dim_edge_encoder": 256, "hidden_dim_node_decoder": 256,
    "aggregation": "sum", "batch_size": 1, "epochs": 500,
    "num_training_samples": 500, "num_validation_samples": 10, "num_test_samples": 10,
    "lr": 1e-4, "lr_decay_rate": None, "amp": False, "jit": False, "wandb_mode": "disabled",
    "graph_path": "graph_7.vtp", "mlp_hidden_dim": 256, "mlp_num_layers": 6, "mlp_input_dim": 2,
    "mlp_output_dim": 3, "pi_iters": 10000, "pi_lr": 0.001,
}


def flat(cfg):
    out = {}

    def walk(node, prefix):
        for key, value in node.items():
            if isinstance(value, dict):
                walk(value, f"{prefix}{key}.")
            else:
                out[f"{prefix}{key}"] = value

    walk(OmegaConf.to_container(cfg, resolve=True), "")
    return out


def differing_keys(a, b):
    fa, fb = flat(load_config(a)), flat(load_config(b))
    assert fa.keys() == fb.keys()
    return {key for key in fa if fa[key] != fb[key]}


def test_official_holds_the_upstream_values():
    cfg = load_config("official")
    for key, value in UPSTREAM.items():
        assert cfg[key] == value, key
    assert cfg.finetune.samples == [7] and cfg.graph_path == "graph_7.vtp"
    assert cfg.physics.nu == 0.01 and cfg.physics.u_max == 0.3
    assert cfg.data.train_fraction == 0.8 and cfg.data.validation_fraction == 0.1


def test_full_differs_from_official_only_in_the_fine_tuned_samples():
    assert differing_keys("official", "full") == {"name", "finetune.samples", "evaluation.plot_samples"}
    assert load_config("full").finetune.samples == list(range(10))


def test_local_reduces_only_the_data_driven_stage():
    assert differing_keys("official", "local") == {
        "name", "device", "epochs", "num_training_samples", "training.checkpoint_every",
        "finetune.samples", "evaluation.plot_samples",
    }
    assert load_config("local").pi_iters == UPSTREAM["pi_iters"]


def test_smoke_is_small():
    cfg = load_config("smoke")
    assert cfg.device == "cpu" and cfg.epochs <= 2 and cfg.pi_iters <= 20 and cfg.num_training_samples <= 8


def test_overrides_apply():
    assert load_config("smoke", ["epochs=3", "finetune.samples=[1]"]).finetune.samples == [1]
