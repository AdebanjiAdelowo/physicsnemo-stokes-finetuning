# Results

Tracked result files. Each study directory is a copy of a run directory without its checkpoints and `.vtp` files, written by `python scripts/results.py publish <run directory or archive>`.

| Path | Content |
|---|---|
| `audit_checks.json` | output of `scripts/audit_checks.py`: Taylor-Hood reproduction of the reference and its solve time, split integrity, receptive field of the MeshGraphNet, and from the local run directory the inlet pressure level, pressure offset, mesh-weighted errors and the perturbation test (see `docs/SCIENTIFIC_AUDIT.md`) |
| `physics_checks.json` | output of `scripts/verify_physics.py`: viscosity fit, boundary data of the reference, automatic-differentiation and least-squares residual checks, residual floor of the reference |
| `validation/geometry_pressure.json` | `scripts/geometry_pressure.py`: reference inlet and outlet pressure, lubrication estimate of the pressure drop and inlet hop distance for all 1000 files; fit on the training split, held-out error |
| `validation/surrogate_local_test100.json` | `scripts/evaluate_surrogate.py`: the local MeshGraphNet on the first 100 test meshes (CPU), errors, weak residuals, inlet level and the pressure recovered from its velocity |
| `validation/physics_stage.json` | `scripts/physics_stage_checks.py`: reference, MeshGraphNet, physics-informed fit and every control on the ten local test meshes; solution errors, nodal and weak residuals, per-boundary mismatches, paired comparisons |
| `validation/costs.json` | `scripts/benchmark_costs.py`: time and peak memory per new geometry of the Taylor-Hood solve, MeshGraphNet inference and the coordinate-network fits, with measured and extrapolated values marked |
| `<study>/study_metadata.json` | per session: timestamp, git commit and dirty state, device, GPU, package versions; resolved configuration, seed, dataset checksum and simulation numbers used |
| `<study>/summary.json` | means over the test meshes per stage, paired change from data-driven to fine-tuned, compute record |
| `<study>/summary.csv` | the comparison table |
| `<study>/per_sample.csv` | every metric for every test mesh and stage (`reference`, `data_driven`, `physics_finetuned`) |
| `<study>/fields.npz` | mesh, fields and nodal residuals of the plotted test meshes |
| `<study>/history.csv`, `train_metrics.json` | data-driven stage: loss, learning rate, validation errors and seconds per epoch; parameters, time, memory |
| `<study>/predict_metrics.json` | test files in order, inference time per mesh |
| `<study>/finetune/history_<i>.csv`, `finetune_<i>.json` | physics-informed stage on test mesh `i`: the ten loss terms and the error against the reference during the iterations; seed, time, memory |
| `<study>/node_stats.json`, `edge_stats.json` | normalisation statistics of the training split, written by `StokesDataset` |

Studies:

| Study | Configuration | Status |
|---|---|---|
| `local` | `configs/local.yaml`: 100 training meshes, 40 epochs, Apple MPS; fine-tuning as upstream, completed and evaluated on all 10 configured test meshes (`summary.json`: `partial: false`); `controls/` holds the fields and records of the `data_only` and `data_bc` controls (`docs/VALIDATION.md`, section 4) | LOCAL/REDUCED |
| `local_2mesh` | the `local` study as first published, with test meshes 0 and 1 only; kept for comparison | LOCAL/REDUCED |
| `full` | `configs/full.yaml`: upstream settings (500 training meshes, 500 epochs), fine-tuning on 10 test meshes; one Kaggle Tesla T4 session, code `ee57d64`; `controls/data_only` holds the data-only control (CPU); `state-full.tar.json` and `kaggle_log_excerpt.txt` record the session | CUDA/FULL |
| `cuda_probe_t4`, `cuda_recovery_t4` | CUDA smoke test, timing probe and the cross-session recovery tests (two failed criteria, then a passing deterministic test) | pipeline checks, not results |
| `official` | `configs/official.yaml`: upstream settings, test mesh 7 only | NOT RUN (it is sample 7 of `full`) |

Units: `rel_l2_*` are relative $L^2$ errors against the FEniCS reference in physical units, as fractions. `res_*` are root mean squares over interior nodes of the Stokes residuals computed with least-squares mesh derivatives; `autodiff_*` are the same residuals of the fine-tuning network by automatic differentiation. `bc_*` are root mean square boundary mismatches. `val_rel_l2_*` in `history.csv` refer to the standardised fields, as in the upstream training script. Times are wall-clock seconds with the device synchronised. Memory fields are CUDA peak allocations and are empty on other devices.
