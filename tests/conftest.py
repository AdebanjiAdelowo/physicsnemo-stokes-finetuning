import numpy as np
import pytest
import pyvista as pv

from stokes_ft.config import load_config


def channel_mesh(nx=13, ny=7, jitter=0.0, seed=0, v_amp=0.0):
    """Triangulated channel [0, 1.5] x [0, 0.4] with the dataset's markers and Poiseuille flow.

    Poiseuille flow with nu = 0.01 and U = 0.3 is an exact solution of the Stokes problem of the
    example: u = 4 U y (0.4 - y) / 0.4^2, v = 0, p = 0.15 (1.5 - x). ``v_amp`` adds a transverse
    velocity that vanishes on the boundary, for tests that need a relative error of v.
    """
    x, y = np.meshgrid(np.linspace(0, 1.5, nx), np.linspace(0, 0.4, ny), indexing="ij")
    interior = np.zeros((nx, ny), bool)
    interior[1:-1, 1:-1] = True
    rng = np.random.default_rng(seed)
    x = x + interior * jitter * rng.uniform(-1, 1, x.shape) * 1.5 / (nx - 1)
    y = y + interior * jitter * rng.uniform(-1, 1, y.shape) * 0.4 / (ny - 1)
    marker = np.zeros((nx, ny))
    marker[0, :] = 1
    marker[-1, :] = 2
    marker[:, 0] = 3
    marker[:, -1] = 3
    idx = np.arange(nx * ny).reshape(nx, ny)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    triangles = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    points = np.stack([x.ravel(), y.ravel(), np.zeros(nx * ny)], 1)
    mesh = pv.PolyData(points, np.hstack([np.full((len(triangles), 1), 3), triangles]).ravel())
    yy, xx = points[:, 1], points[:, 0]
    mesh.point_data["u"] = 4 * 0.3 * yy * (0.4 - yy) / 0.16
    mesh.point_data["v"] = v_amp * np.sin(np.pi * xx / 1.5) * np.sin(np.pi * yy / 0.4)
    mesh.point_data["p"] = 0.15 * (1.5 - xx)
    mesh.point_data["marker"] = marker.ravel()
    return mesh


@pytest.fixture(scope="session")
def tiny_root(tmp_path_factory):
    """A 20-file dataset with the layout of the real one, and the overrides that point a config at it."""
    root = tmp_path_factory.mktemp("stokes")
    raw = root / "data" / "raw" / "results"
    raw.mkdir(parents=True)
    for n in range(20):
        channel_mesh(jitter=0.2, seed=n, v_amp=0.05).save(str(raw / f"res_{n}.vtp"))
    overrides = [
        f"data.raw_dir={raw}", f"data.split_dir={root / 'data' / 'dataset'}",
        "num_training_samples=4", "num_validation_samples=2", "num_test_samples=2",
        "hidden_dim_node_encoder=8", "hidden_dim_edge_encoder=8", "hidden_dim_node_decoder=8",
        "epochs=2", "pi_iters=4", "finetune.log_every=2",
    ]
    return root, overrides


@pytest.fixture()
def tiny_cfg(tiny_root):
    return load_config("smoke", tiny_root[1])
