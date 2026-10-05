import numpy as np
import torch

from conftest import channel_mesh
from stokes_ft.finetune import LOSS_NAMES, LOSS_WEIGHTS, PhysicsInformedFineTuner, build_finetune_network


def make_tuner(seed=0, offset=0.0):
    mesh = channel_mesh(9, 5, v_amp=0.05)
    coords = np.array(mesh.points[:, :2])
    marker = np.asarray(mesh.point_data["marker"])
    ref = [np.asarray(mesh.point_data[k]).reshape(-1, 1) for k in "uvp"]
    gnn = [r + offset for r in ref]
    torch.manual_seed(seed)
    return PhysicsInformedFineTuner(
        torch.device("cpu"), *gnn, coords, coords[marker == 1], coords[marker == 3], 0.01, *ref
    )


def test_network_is_new_and_fully_trainable():
    model = build_finetune_network("cpu")
    assert sum(p.numel() for p in model.parameters()) == 49923
    assert all(p.requires_grad for p in model.parameters())
    assert not model.mdls_model.B.requires_grad and model.mdls_model.B.shape == (2, 64)
    out = model({"x": torch.zeros(5, 1), "y": torch.zeros(5, 1)})
    assert set(out) == {"u", "v", "p"} and out["u"].shape == (5, 1)


def test_loss_terms_and_one_step():
    tuner = make_tuner()
    losses = tuner.loss()
    assert len(losses) == len(LOSS_NAMES) == len(LOSS_WEIGHTS) == 10
    assert all(torch.isfinite(term) and term >= 0 for term in losses)
    before = [p.detach().clone() for p in tuner.model.parameters()]
    tuner.train()
    assert all(not torch.equal(a, b) for a, b in zip(before, tuner.model.parameters()))
    errors = tuner.validation()
    assert len(errors) == 3 and all(torch.isfinite(e) for e in errors)


def test_data_terms_target_the_gnn_prediction_not_the_reference():
    # the reference enters only the monitored error: shifting the GNN fields changes the data terms
    exact, shifted = make_tuner(offset=0.0), make_tuner(offset=0.5)
    a, b = exact.loss(), shifted.loss()
    for i in range(3):
        assert not torch.isclose(a[i], b[i])
    for i in range(3, 10):
        assert torch.isclose(a[i], b[i])  # boundary and PDE terms do not see the GNN fields


def test_seed_fixes_the_network():
    a, b, c = make_tuner(seed=3), make_tuner(seed=3), make_tuner(seed=4)
    assert torch.equal(a.model.mdls_model.B, b.model.mdls_model.B)
    assert not torch.equal(a.model.mdls_model.B, c.model.mdls_model.B)
    assert all(torch.equal(p, q) for p, q in zip(a.model.parameters(), b.model.parameters()))
