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
| G1 | Fine-tuning on test meshes 2 to 9 of the local study | The before/after table rests on 2 meshes. The data-driven error of meshes 0 and 1 is below the 10-mesh mean (velocity 0.187 against 0.233), so they are not representative | PENDING, awaiting explicit approval; decision rules fixed in advance below | `python scripts/run_study.py --config local --resume` then `python scripts/results.py publish runs/local` | about 4.5 h MPS, idle machine |
| G2 | NVIDIA training schedule (500 meshes, 500 epochs) and 10 fine-tuned meshes | The local MeshGraphNet saw 1.6 % of the upstream iterations. The effect of the physics step on a well-trained surrogate is unknown | PENDING | `kaggle/run_cuda.ipynb` with `REF` set to the commit to run and `RUN_FULL = True` (the launcher measures and refuses a run that does not fit the session) | 250,000 graph updates plus 10 x 10,000 coordinate-network updates; CUDA probe not yet run |
| G3 | Fine-tuning seed sensitivity | The Fourier matrix and initialisation are random; one seed per mesh | NOT DEFINED | needs a seed override for `finetune_seed`; then 3 seeds x meshes 0 and 1 | about 3.5 h MPS |
| G4 | Data-only control: the same coordinate network fitted to the MeshGraphNet prediction with all physics and boundary weights set to 0 | Separates the effect of the PDE terms from the smoothing effect of refitting with a Fourier-feature MLP | NOT DEFINED | needs a loss-weight override in `PhysicsInformedFineTuner.train` | about 1 h MPS for meshes 0 and 1 (no second derivatives) |
| G5 | Outlet condition in the loss (`nu du/dn - p n = 0` at outlet nodes), or pressure data weight reduced | Tests the explanation of the pressure result: the inherited pressure offset (audit section 5) | NOT DEFINED | new loss term; same cost as a fine-tuning run | about 1.2 h MPS for meshes 0 and 1 |
| G6 | Hard no-slip and inlet constraints (distance-function ansatz) | No-slip worsens from 0.003 to 0.016 under the soft penalty | NOT DEFINED | new network wrapper | as G5 |
| G7 | Upstream `pi_fine_tuning_gnn.py` variant | The second upstream physics-informed method keeps the boundary markers as input; not run | NOT DEFINED | adaptation of the script to the run layout | similar to one fine-tuning run per mesh |
| G8 | Classical baseline in the comparison table | A Taylor-Hood solve reproduces the reference exactly in under 1 s per mesh | **DONE** (`results/audit_checks.json`, `reference_solver`) | `python scripts/audit_checks.py` | about 10 s for 10 meshes, plus the 1000-file geometry scan |
| G9 | Receptive field and inlet pressure level of the data-driven model | Explains the data-driven pressure error | **DONE** for the local checkpoint (`stored_local_study`) | `python scripts/audit_checks.py` (needs `runs/local`, untracked) | minutes on CPU |
| G10 | Mesh-consistent physics metric | Vertex-only operators give the reference a nonzero residual (floor 0.3). A Taylor-Hood residual needs mid-edge velocity values, which the MeshGraphNet does not produce | NOT DEFINED | for example, interpolate candidate fields to P2 and report the discrete residual norm, or report distance to the Taylor-Hood solution with the candidate's own boundary data | design decision |
| G11 | Uncertainty over meshes | With 2 meshes no interval is meaningful; with 10, report a paired bootstrap or sign test per metric | Implemented (`evaluate.paired_uncertainty`: seeded bootstrap interval, sign test, Wilcoxon); applies to G1/G2 outputs once they exist | extend `evaluate.summarise` | negligible |
| G12 | Architecture with a receptive field covering the domain (more message-passing steps, a multiscale graph, or a global latent) | 60 to 69 % of the nodes, including the whole inlet, are beyond the 15-hop receptive field | NOT DEFINED | new model configuration | comparable to G2 |

## G1: necessity and decision rules, fixed before any result exists

**Status.** A resumable G1 run was queued on 2026-10-24 (commit `2aa7509`) behind the shared accelerator lock. It was stopped on 2026-10-25 before acquiring the lock, pending explicit approval. No mesh 2 to 9 was fine-tuned. The completed mesh 0 and 1 outputs were compared byte for byte with a copy taken before the launch and are unchanged. The launch appended one session entry to the untracked `runs/local/study_metadata.json`.

**Necessity: optional.** G1 is the only test of whether the velocity improvement holds beyond meshes 0 and 1, so it is not redundant. With 2 of 2 meshes improved, the smallest attainable two-sided sign-test p-value is 0.5. The two meshes are also easier than average. The main conclusions do not depend on G1, because each already holds on all 10 test meshes without any fine-tuning:

* the receptive-field blindness of the inlet pressure (slope of the data-driven inlet level against the reference: -0.0002);
* the Taylor-Hood baseline (exact, 0.4 to 0.6 s per mesh);
* the dataset integrity (section 7 of the audit).

Without G1, the velocity claim stays limited to "2 meshes of an undertrained model", as the README states. G1 also cannot attribute any improvement to the PDE terms. That needs the data-only control G4.

**Fixed design.**

* All 8 meshes, `configs/local.yaml` unchanged, 10,000 iterations each, and the existing seed rule (`seed + 1000 + index`).
* No early stopping, no per-mesh tuning, and no mesh excluded after its result is seen.
* Analysis is over all 10 meshes, including 0 and 1.

**Primary endpoint.** The velocity relative L2 error, fine-tuned against data-driven, paired per mesh.

| Outcome over 10 meshes | Conclusion |
|---|---|
| At least 9 of 10 improved (two-sided sign test p at most 0.021) and the bootstrap 95 % interval of the mean per-mesh relative change entirely below 0 | The velocity improvement generalises across this test set (local, undertrained surrogate only) |
| 8 of 10 or fewer improved (p at least 0.109), or the interval includes 0 | Generalisation not established; report the per-mesh results and keep the two-mesh scoping |

**Secondary measures (descriptive only, no claim attached).**

* Pressure and total relative L2 error.
* No-slip and inflow residuals.
* Outlet pressure.
* Residuals reported against the floor set by the reference field (audit section 2). A residual below the reference's own is not evidence of a better solution.
* Fine-tuning time against the Taylor-Hood time.

**Falsification test of the receptive-field explanation.** The audit predicts that the fine-tuned inlet pressure level stays anchored to the data-driven one. The explanation is contradicted if the fine-tuned level has a slope of at least 0.5 against the reference level, with a correlation of at least 0.8, over the 10 meshes.

**Abort rules.**

* Stop on a non-finite loss or a crash.
* Atomic outputs make a restart with `--resume` safe.
* Never run concurrently with other accelerator work; the runner waits for the shared lock.

**Suggested companion run.** Run G4 on the same 10 meshes. G4 needs a loss-weight override, and its iterations are cheaper because they have no second derivatives.

## Resource policy for the pending runs

Run G1 only on an otherwise idle machine. At the time of this audit the load average was 9 to 18 because other sessions were running. Check `df -h ~` and `memory_pressure` first: the local run directory holds 30 MB of checkpoints, and each fine-tuned network is 0.2 MB. Free disk was 44 GiB.
