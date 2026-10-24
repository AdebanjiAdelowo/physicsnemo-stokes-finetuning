# Experiment gaps

What the stored evidence does not yet support, and the runs that would close each gap. Status labels: **DONE** (evidence in `results/`), **PENDING** (defined, not run), **NOT DEFINED** (needs code or a design decision first). Times come from measurements on an Apple M3 Pro shared with other jobs. They are planning figures, not benchmarks.

## Measured cost per unit of work

| Work | CPU | MPS |
|---|---|---|
| Physics-informed iteration, one test mesh (about 3,300 nodes) | 0.54 s (8 threads) to 0.76 s (4 threads) | 0.22 s (measured 2026-10-24); 0.17 s in the stored local run |
| One physics-informed run (10,000 iterations) | 1.5 to 2.1 h | 0.5 to 0.6 h |
| MeshGraphNet training, local config (4,000 iterations) | not measured | 2,308 s |
| Taylor-Hood P2-P1 solve, one test mesh (about 29,000 unknowns, direct solver) | 0.5 to 0.9 s | n/a |

## Gaps

| # | Gap | Why it matters | Status | Exact command | Estimated cost |
|---|---|---|---|---|---|
| G1 | Fine-tuning on test meshes 2 to 9 of the local study | The before/after table rests on 2 meshes. The data-driven error of meshes 0 and 1 is below the 10-mesh mean (velocity 0.187 against 0.233), so they are not representative | PENDING | `python scripts/run_study.py --config local --resume` then `python scripts/results.py publish runs/local` | about 4.5 h MPS, idle machine |
| G2 | NVIDIA training schedule (500 meshes, 500 epochs) and 10 fine-tuned meshes | The local MeshGraphNet saw 1.6 % of the upstream iterations. The effect of the physics step on a well-trained surrogate is unknown | PENDING | `kaggle/run_cuda.ipynb` with `REF` set to the commit to run and `RUN_FULL = True` (the launcher measures and refuses a run that does not fit the session) | 250,000 graph updates plus 10 x 10,000 coordinate-network updates; CUDA probe not yet run |
| G3 | Fine-tuning seed sensitivity | The Fourier matrix and initialisation are random; one seed per mesh | NOT DEFINED | needs a seed override for `finetune_seed`; then 3 seeds x meshes 0 and 1 | about 3.5 h MPS |
| G4 | Data-only control: the same coordinate network fitted to the MeshGraphNet prediction with all physics and boundary weights set to 0 | Separates the effect of the PDE terms from the smoothing effect of refitting with a Fourier-feature MLP | NOT DEFINED | needs a loss-weight override in `PhysicsInformedFineTuner.train` | about 1 h MPS for meshes 0 and 1 (no second derivatives) |
| G5 | Outlet condition in the loss (`nu du/dn - p n = 0` at outlet nodes), or pressure data weight reduced | Tests the explanation of the pressure result: the inherited pressure offset (audit section 5) | NOT DEFINED | new loss term; same cost as a fine-tuning run | about 1.2 h MPS for meshes 0 and 1 |
| G6 | Hard no-slip and inlet constraints (distance-function ansatz) | No-slip worsens from 0.003 to 0.016 under the soft penalty | NOT DEFINED | new network wrapper | as G5 |
| G7 | Upstream `pi_fine_tuning_gnn.py` variant | The second upstream physics-informed method keeps the boundary markers as input; not run | NOT DEFINED | adaptation of the script to the run layout | similar to one fine-tuning run per mesh |
| G8 | Classical baseline in the comparison table | A Taylor-Hood solve reproduces the reference exactly in under 1 s per mesh | **DONE** (`results/audit_checks.json`, `reference_solver`) | `python scripts/audit_checks.py` | about 10 s for 10 meshes, plus the 1000-file geometry scan |
| G9 | Receptive field and inlet pressure level of the data-driven model | Explains the data-driven pressure error | **DONE** for the local checkpoint (`stored_local_study`) | `python scripts/audit_checks.py` (needs `runs/local`, untracked) | minutes on CPU |
| G10 | Mesh-consistent physics metric | Vertex-only operators give the reference a nonzero residual (floor 0.3). A Taylor-Hood residual needs mid-edge velocity values, which the MeshGraphNet does not produce | NOT DEFINED | for example, interpolate candidate fields to P2 and report the discrete residual norm, or report distance to the Taylor-Hood solution with the candidate's own boundary data | design decision |
| G11 | Uncertainty over meshes | With 2 meshes no interval is meaningful; with 10, report a paired bootstrap or sign test per metric | NOT DEFINED for G1/G2 outputs | extend `evaluate.summarise` | negligible |
| G12 | Architecture with a receptive field covering the domain (more message-passing steps, a multiscale graph, or a global latent) | 60 to 69 % of the nodes, including the whole inlet, are beyond the 15-hop receptive field | NOT DEFINED | new model configuration | comparable to G2 |

## Resource policy for the pending runs

Run G1 only on an otherwise idle machine. At the time of this audit the load average was 9 to 18 because other sessions were running. Check `df -h ~` and `memory_pressure` first: the local run directory holds 30 MB of checkpoints, and each fine-tuned network is 0.2 MB. Free disk was 44 GiB.
