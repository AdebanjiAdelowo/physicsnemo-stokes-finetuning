# Upstream comparison

Comparison of this repository with the NVIDIA PhysicsNeMo Stokes-flow example it adapts. Checked on 2026-10-24 against a local clone of the upstream repository at the pinned commit.

## Provenance

| Item | Value |
|---|---|
| Upstream repository | https://github.com/NVIDIA/physicsnemo |
| Pinned commit | `b45a5c810c741e6b41f8515be24c51121f8fc21f` (2026-10-02, "mesh.fields: FieldSchema replaces the free-function rank-spec API (#2030)"), version 2.3.0a0 |
| Example | [`examples/cfd/stokes_mgn`](https://github.com/NVIDIA/physicsnemo/tree/b45a5c810c741e6b41f8515be24c51121f8fc21f/examples/cfd/stokes_mgn) |
| Last upstream commit touching the example before the pin | `12a852e85` (2026-05-26, "Add missing dependencies for examples, replace FusedAdam with Adam (#1642)") |
| Library modules used, not copied | `physicsnemo/datapipes/gnn/stokes_dataset.py` (`StokesDataset`), `physicsnemo/models/meshgraphnet/meshgraphnet.py` (`MeshGraphNet`), `physicsnemo/sym/eq/phy_informer.py` (`PhysicsInformer`), `physicsnemo/sym/eq/gradients.py` (autodiff and least-squares gradients, `compute_connectivity_tensor`), `physicsnemo/utils` (checkpoints) |
| Installed package | `nvidia-physicsnemo` from git at the pinned commit (`direct_url.json` of the environment records `commit_id` `b45a5c8...`); the four library files above are byte-identical to the clone |
| Dataset | NGC resource `nvidia/physicsnemo/physicsnemo_datasets_stokes_flow` version 0.0.1, `results_polygon.zip`, SHA-256 `5aec06af...d696d1` (fetched by upstream `raw_dataset/download_dataset.sh`) |

The project is **directly derived from one upstream example**. It does not combine several examples. The two-stage structure (MeshGraphNet, then a physics-informed step on its prediction) is NVIDIA's. Upstream also ships a second variant, `pi_fine_tuning_gnn.py`. That variant is not run here, but its least-squares mesh derivative is reused for evaluation.

## File mapping

| Upstream file | Here | Relationship |
|---|---|---|
| `conf/config.yaml` | `configs/official.yaml` | Same keys and values; the Hydra and path keys are replaced and new keys are added below a marker |
| `preprocess.py` | `src/stokes_ft/data.py` (`split_numbers`, `prepare_split`) | Same 80/10/10 partition; seeded, with a manifest (upstream calls `random.shuffle` with no seed over `os.listdir` order, so its split cannot be reproduced) |
| `utils.py` (`get_dataset`, `parabolic_inflow`) | `src/stokes_ft/data.py` (`read_sample`), `src/stokes_ft/physics.py` | Same arrays and shapes `(N, 1)`; returns a dictionary |
| `train.py` | `src/stokes_ft/model.py`, `src/stokes_ft/engine.py` | Same model arguments, loss, optimiser and `StepLR` rule; single process, seeded, no W&B |
| `inference.py` | `src/stokes_ft/engine.py` (`predict_test`) | Same denormalisation and `pred_*` arrays |
| `pi_fine_tuning.py` | `src/stokes_ft/finetune.py` | `Stokes`, `DNN`, `FourierDNN` unchanged; `PhysicsInformedFineTuner` changed only as listed below |
| `pi_fine_tuning_gnn.py` | `src/stokes_ft/physics.py` (`nodal_residuals`) | Only its least-squares derivative set-up is reused, as an evaluation operator |

## Line-level differences in the physics-informed step

A `diff` of `class Stokes` through the end of `PhysicsInformedFineTuner` against upstream gives only these changes:

1. `u_max=0.3` and `lr=0.001` become constructor arguments with the upstream values as defaults.
2. `fused=True if torch.cuda.is_available() else False` becomes `fused=self.device.type == "cuda"`. This matters only when CUDA exists but another device is chosen.
3. The `wandb.log` call in `validation` is removed.

The ten loss terms, their weights (1, 1, 1, 10, 10, 10, 10, 1, 1, 10), the network (`[2, 128, 128, 128, 128, 3]`, 64 Fourier features with `B ~ 10 N(0, 1)`, 49,923 parameters), Adam with a constant rate of 1e-3, and the 10,000 iterations are identical.

## Upstream behaviour verified in the code

| Topic | Upstream behaviour | Verified by |
|---|---|---|
| Graph | Triangle edges, made bidirected and de-duplicated by `torch_geometric.utils.to_undirected`; edge features are the relative displacement and its norm | Reading `StokesDataset._create_pyg_graph`, `add_edge_features` |
| Node inputs | Position (2) and one-hot marker (5), **not normalised**; only `u, v, p` and the edge features are standardised with training statistics | `normalize_keys=["u","v","p"]` default |
| Normalisation leakage | Statistics are computed on the training split and loaded from `node_stats.json`/`edge_stats.json` for validation and test | `StokesDataset.__init__`; here the dataset is built inside the run directory |
| File order | Sorted by the first digit run in the **full path**, so a digit in a directory name breaks the order | Upstream code; guarded here by a relative path and an order check (`data.make_dataset`) |
| MeshGraphNet | Encoder and decoder width 256 (config); processor width 128 and **15 message-passing blocks** (library defaults); sum aggregation; 2,531,587 parameters | `tests/test_model.py` |
| Learning rate | `StepLR` stepped every iteration with step size of one epoch and factor `0.01^(1/epochs)`, so it ends at 1e-6 | `train.py` |
| Validation metric | Relative L2 of the **standardised** fields | `train.py`; kept as `val_rel_l2_*` |
| Data loss shapes | `gnn_*`, `ref_*` and network outputs are all `(N, 1)`, so there is no silent `(N, N)` broadcasting | `utils.get_dataset` reshapes with `(-1, 1)` |
| PDE loss nodes | Momentum and continuity residuals at **all** mesh nodes, including wall, polygon and inlet nodes | `loss()` evaluates `self.coords` |
| Boundary losses | Inlet profile and no slip as penalties of weight 10; **no outlet term** | `loss()` |
| Least-squares connectivity | `compute_connectivity_tensor` adds reversed edges and removes duplicates, so the stencil is symmetric even though boundary edges appear once in the triangle list | `physicsnemo/sym/eq/gradients.py:451` |
| `model.eval()` in `validation` | Never switched back to `train()`; the network has no dropout or normalisation layers, so this has no effect | Reading `DNN` |

## Upstream README against upstream code and data

These discrepancies are listed in the project README and were re-checked here:

| Upstream README | Code and data at the pin |
|---|---|
| Outlet at `x = 2.2` | All 1000 meshes span `x in [0, 1.5]` |
| Hidden width 256 in encoder, processor and decoder | Processor width not passed; library default 128 |
| Dataset not publicly available | `download_dataset.sh` fetches it from NGC without credentials |
| Outflow condition stated with the problem | No outlet loss term in either fine-tuning script |
| `mlp_hidden_dim: 256`, `mlp_num_layers: 6` | Not read by `pi_fine_tuning.py` |

New in this audit: the reference solution is reproduced to a relative error of 1e-10 or smaller by a P2-P1 Taylor-Hood solve with the outlet condition `nu du/dn - p n = 0` (see `docs/SCIENTIFIC_AUDIT.md`, section 2). The upstream README does not state the discretisation. The README of this repository states this boundary condition, and it is now verified rather than assumed.

## Reproducibility differences

| Upstream | Here |
|---|---|
| No seed; unseeded split | `seed`, `data.split_seed`, per-mesh fine-tuning seed `seed + 1000 + index` |
| `DistributedSampler`, DDP, W&B | Single process with a seeded sample order per epoch; no W&B |
| A checkpoint each epoch, all kept | Every `training.checkpoint_every` epochs, older ones removed |
| Hard-coded `cuda` or `cpu` | Explicit device; an unavailable device is an error |
| Fine-tunes the single file `graph_7.vtp` | A configurable list of test indices (`finetune.samples`) |
