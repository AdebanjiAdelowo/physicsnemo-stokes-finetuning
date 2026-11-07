# CUDA smoke test and timing probe (Kaggle, Tesla T4)

Private Kaggle kernel `stokes-mgn-cuda-probe-71e7c79`, version 3, run on 2026-11-07, notebook `kaggle/run_cuda.ipynb` at `ad564c8` checking out code commit `71e7c79268006ad1396c9effb6f12dd0acae67a8` (clean tree, recorded in `full_probe_study_metadata.json`). Steps A to C ran; step D (the full study) did not.

These are pipeline checks and timing measurements. **None of the numbers here is a scientific result**: the probe trains the MeshGraphNet for one epoch.

| Item | Value |
|---|---|
| GPU | Tesla T4, 15,360 MiB (2 in the session; the study uses one) |
| Software | torch 2.15.0+cu132, CUDA 13.2, PhysicsNeMo 2.3.0a0 (`b45a5c8`), torch_geometric 2.8.1, torch_scatter 2.1.2, NumPy 2.5.4, Python 3.13 (local validated environment: torch 2.14.1, Python 3.12) |
| Tests | 96 passed (the suite of `71e7c79`) |
| Dataset | downloaded and SHA-256 verified by `scripts/prepare_data.py` |
| Device checks (`device_checks.json`) | resume after an interruption vs uninterrupted run: max relative difference 4.3e-4; same-seed repeat: 1.2e-4 (tolerance 1e-3); checkpoint weights bit-identical; outputs of two loads agree to 1.3e-6 (one model called twice: 1.2e-6); gradients finite; passed |
| Data-driven stage, full configuration | 93.3 ms per iteration (46.7 s for one epoch of 500 meshes), peak 1,153 MB |
| Inference | 36.5 ms per test mesh |
| Physics-informed stage | 32.4 ms per iteration (6.5 s for 200), peak 289 MB |
| Checkpoints kept | 29.5 MB (latest MeshGraphNet checkpoint with optimiser state, one fine-tuned network) |

Projection for `configs/full.yaml` from these rates: 250,000 training iterations 6.5 h, validation (10 meshes per epoch) under 0.1 h, 10 x 10,000 fine-tuning iterations 0.9 h, prediction and evaluation a few minutes, set-up (install, data, tests, smoke, checks) 0.25 h: about 7.7 h. The single-epoch rate includes warm-up and tends to overestimate.

The two earlier versions of the kernel stopped in set-up: version 1 at the dataset step (preinstalled RAPIDS imported numba, which rejects NumPy 2.5; fixed in `f675cb5`), version 2 at the device checks (a bit-identity criterion that scatter-add cannot meet on CUDA; corrected in `ad564c8`).
