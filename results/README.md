# Results

Tracked result files. Each study directory is a copy of a run directory without its checkpoints and `.vtp` files, written by `python scripts/results.py publish <run directory or archive>`.

| Path | Content |
|---|---|
| `physics_checks.json` | output of `scripts/verify_physics.py`: viscosity fit, boundary data of the reference, automatic-differentiation and least-squares residual checks, residual floor of the reference |
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
| `local` | `configs/local.yaml`: 100 training meshes, 40 epochs, Apple MPS; fine-tuning as upstream, completed and evaluated on test meshes 0 and 1 of the 10 configured (`summary.json`: `partial: true`) | LOCAL/REDUCED |
| `full` | `configs/full.yaml`: upstream settings, 10 test meshes, CUDA | PENDING |
| `official` | `configs/official.yaml`: upstream settings, test mesh 7 only | NOT RUN (it is sample 7 of `full`) |

Units: `rel_l2_*` are relative $L^2$ errors against the FEniCS reference in physical units, as fractions. `res_*` are root mean squares over interior nodes of the Stokes residuals computed with least-squares mesh derivatives; `autodiff_*` are the same residuals of the fine-tuning network by automatic differentiation. `bc_*` are root mean square boundary mismatches. `val_rel_l2_*` in `history.csv` refer to the standardised fields, as in the upstream training script. Times are wall-clock seconds with the device synchronised. Memory fields are CUDA peak allocations and are empty on other devices.
