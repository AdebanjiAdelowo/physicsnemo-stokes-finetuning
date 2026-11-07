# Validation study (November 2026)

Continuation of the [scientific audit](SCIENTIFIC_AUDIT.md). It asks when a learned graph surrogate of this Stokes problem is accurate, physically consistent and worth its cost compared with the finite-element method, and it separates what was measured from what remains proposed.

Branch `research/validation-2026-11`, based on `57926cb`. Every number here comes from a file in [`results/validation/`](../results/validation/), produced by a script named next to it. Historical results (`results/local/`, `results/audit_checks.json`, `results/physics_checks.json`) were not modified.

## Status labels

| Label | Meaning |
|---|---|
| **VERIFIED** | Exact or analytic check, independent of training (FEM identity, manufactured solutions, dataset scans) |
| **LOCAL/REDUCED** | Measured with the MeshGraphNet of `configs/local.yaml` (100 training meshes, 40 epochs: 1.6 % of the upstream iterations) |
| **FULL** | Measured with `configs/full.yaml` (500 meshes, 500 epochs, CUDA). **No result in this repository has this label** |
| **PENDING** | Defined, implemented and tested, not executed; the command and cost are given |

## 1. Problem and reference (VERIFIED)

$$-\nu\Delta\mathbf u+\nabla p=0,\qquad \nabla\cdot\mathbf u=0\quad\text{in }\Omega=[0,1.5]\times[0,0.4]\setminus P,\qquad \nu=0.01,$$

parabolic inlet $u=4Uy(0.4-y)/0.4^2$, $U=0.3$; no slip on the walls and the polygon $P$; natural outlet $\nu\,\partial_n\mathbf u-p\,\mathbf n=0$.

The reference fields are the vertex values of a Taylor-Hood P2-P1 solution with this outlet condition: `src/stokes_ft/fem.py` reproduces them to relative $L^2$ error at most $3.2\times10^{-10}$ on all ten evaluated test meshes, and the symmetric-stress outlet form is excluded (1.0 to 2.4 % error) (audit, section 2). The solver was refactored in this study so that the assembled system can be reused; its result is unchanged (the Poiseuille exactness test and outlet-form test pass).

New consequence, over **all 1000 dataset files** (`scripts/geometry_pressure.py`): the outlet pressure is zero to RMS $6\times10^{-10}$, so the natural outlet condition fixes the pressure level exactly, and the **inlet pressure level ranges from 0.30 to 3.53** (mean 0.74, std 0.38).

## 2. Diagnostics: errors, residuals and boundary mismatches

### 2.1 Why a new residual

The audit showed that the nodal least-squares residual gives the exact reference a nonzero value (momentum 0.32 to 0.37), so a field can score below the reference without being closer to a Stokes solution. `src/stokes_ft/weak.py` adds diagnostics built on the Taylor-Hood space:

* A vertex field, read as piecewise linear, **is a member** of the P2-P1 space (its mid-edge values are the means of the vertex values). Its discrete weak residual is therefore the residual of the Taylor-Hood equations, whose only zero with the dataset boundary data is the Taylor-Hood solution.
* The momentum rows are tested with every P2 test function that the inlet and no-slip conditions leave free, so **the natural outlet condition is part of the residual**; the strong-form residual cannot see it.
* Momentum is measured in the discrete $H^{-1}$ (dual $H^1$) norm $\sqrt{r^\top G^{-1}r}$, the norm in which Stokes stability bounds the velocity error; continuity by the $L^2$ projection of $\nabla\cdot\mathbf u$.
* Errors use consistent mass and stiffness matrices: $L^2$ and $H^1$-seminorm of the velocity, $L^2$ of the pressure, and the **gauge-free** pressure error (after the best constant), with that constant reported separately.
* Boundary mismatches are split by boundary: inlet, channel walls, polygon, outlet pressure.
* **Pressure recovery**: for a given velocity, the P1 pressure that minimises the weak momentum residual. The outlet rows fix its level without any gauge condition. It gives the pressure that the momentum equation and the outlet condition imply for that velocity, independent of any predicted pressure.

### 2.2 Validation of the diagnostics (VERIFIED, `tests/test_weak.py`)

| Check | Result |
|---|---|
| Piecewise-linear exact Stokes field (Couette, $p=0$) | weak momentum residual below $10^{-13}$ |
| Same velocity with $p=0.1$ (violates only the outlet condition) | residual above $10^{-3}$: the outlet condition is seen |
| Stagnation flow $(x,-y)$ (divergence-free, harmonic, outlet traction $\nu$) | momentum residual nonzero, continuity below $10^{-12}$ |
| Full P2 Taylor-Hood solution | momentum rows below $10^{-12}$; it is the zero of the residual |
| Vertex interpolant of Poiseuille flow, $h$ halved twice | residual decreases at first order ($O(h)$, the $H^1$ interpolation error) |
| Residual of increasingly perturbed fields | strictly increasing |
| Pressure recovery from the full P2 velocity | exact pressure to $10^{-10}$ |
| Pressure recovery from vertex velocities | converges under refinement; level fixed without a gauge |
| Gauge split, mesh norms | constant shift fully attributed to the offset; 10 % velocity scaling gives exactly 0.10 in $L^2$ and $H^1$ |

### 2.3 What the diagnostics show on the dataset (LOCAL/REDUCED)

On the ten evaluated test meshes the reference floor of the weak momentum residual is $2.0$ to $7.8\times10^{-3}$, against $1.3$ to $2.3\times10^{-2}$ for the MeshGraphNet: a factor 2.5 to 10, a better separation than the nodal residual gives (mean 0.96 against 0.32, a factor of 3).

**The weak residual still cannot certify accuracy for vertex-only fields.** On mesh 0 the physics-informed field reaches $3.2\times10^{-3}$, below the reference's $5.1\times10^{-3}$, while its $H^1$ velocity error is still 25 %. Every vertex-only representation has an interpolation floor, and a smooth field fitted to the strong form can fall beneath it. The residual is a useful relative measure, reported always next to the reference floor; **the error against the verified FEM reference is the only quantity in this study that establishes accuracy.**

## 3. The receptive-field limitation

### 3.1 The missing information is a global scalar (VERIFIED)

`scripts/geometry_pressure.py`, all 1000 files, no training involved ([`geometry_pressure.json`](../results/validation/geometry_pressure.json)):

| Quantity | Value |
|---|---|
| Files whose inlet is more than 15 hops from the polygon | 99.4 % (15 to 26 hops) |
| Correlation of the inlet level with a zero-parameter lubrication estimate of the channel resistance, $\Delta p\approx 12\nu Q\int dx/(h_b^3+h_t^3)$ | **0.89** |
| Same feature, line fitted on the 800 training files, held-out RMSE (200 files) | **0.121** ($R^2=0.77$) |
| Geometry-blind predictor (training mean), held-out RMSE | 0.265 |

The inlet level is, to a good approximation, one global functional of the obstacle outline (its hydraulic resistance). This is the physical reason a local message-passing model needs global information, and it indicates which intervention is cheapest.

### 3.2 The MeshGraphNet is blind to it on 100 test meshes (LOCAL/REDUCED)

`scripts/evaluate_surrogate.py`, the local checkpoint on the first **100** test meshes on the CPU (it reproduces the stored MPS predictions of the ten shared meshes to six digits; [`surrogate_local_test100.json`](../results/validation/surrogate_local_test100.json)):

| Quantity, 100 test meshes | MeshGraphNet | Pressure recovered from its velocity |
|---|---:|---:|
| Inlet level: slope against the reference | **0.00006** | 0.37 |
| Inlet level: correlation with the reference | 0.03 | **0.94** |
| Inlet level: RMS error | 0.237 | 0.198 |
| Pressure relative $L^2$, mean (median) | 0.337 (0.298) | 0.182 (0.152) |
| Meshes where the recovered pressure is better | | 60 of 100 (Wilcoxon $p=3\times10^{-5}$, sign test $p=0.057$) |
| Velocity relative $L^2$, mean | 0.210 | |
| Velocity $H^1$-seminorm error, mean | 0.508 | |

The predicted inlet level has a standard deviation of 0.0005 across 100 geometries, against 0.235 in the reference. **Its velocity field nonetheless carries the obstacle information**: an elliptic solve of the momentum equation with the outlet condition turns it into an inlet level that correlates 0.94 with the reference. The recovered pressure inherits the velocity error (it underestimates the drop, slope 0.37) and is worse than the predicted one on 40 meshes, those whose level happens to be near the training mean. This is evidence about mechanism, not a better predictor: global coupling through the physics, which the 15-hop model lacks, is what supplies the level.

The ten-mesh subset of the earlier studies is slightly harder than these 100 meshes (velocity 0.233 against 0.210); meshes 0 and 1 are easier (0.188).

### 3.3 Controlled architecture experiment (PENDING)

Design (`configs/arch.yaml`, `src/stokes_ft/model.py`, `tests/test_model_variants.py`). Three processors, identical in data, split, seed set, epochs, optimiser and learning-rate schedule:

| Arm | Receptive field | Parameters | Compute per forward pass |
|---|---|---:|---|
| `baseline` (upstream) | 15 hops | 2,531,587 | 1x |
| `global_latent`: mean-pooled node latents through one MLP shared by all blocks, added to every node after each block; zero-initialised, so the untrained model equals the baseline | whole graph | +33,024 (+1.3 %) | about 1x |
| `shared_depth`: the same 15 blocks applied twice | 30 hops (covers the 26-hop maximum inlet distance) | identical | about 2x |

The two variants separate the receptive field from the parameter count; compute is measured, not matched. Tests check the measured receptive fields (exactly zero response beyond 15 or 30 hops for the bounded arms, nonzero response at every node for the global latent), parameter counts, equality of the zero-initialised global latent with the baseline, checkpoint round trips and resumption.

**Why the global latent is the primary candidate.** Section 3.1 shows that the missing quantity is mostly a single global scalar. A global latent supplies global information in one block at 1.3 % extra parameters; depth reaches the inlet only beyond 26 hops and grows in cost with the domain. A multiscale graph (MultiScale MeshGraphNets, Fortunato et al. 2022) is the stronger general answer for elliptic problems but needs coarsened graphs and more design freedom; it is the next step if the global latent fails on the gauge-free part of the error.

**Endpoints, fixed before any run.** Primary: inlet-level slope and RMS error on the first 100 test meshes. Secondary: pressure and velocity relative $L^2$, gauge-free pressure error, $H^1$ velocity error. Three seeds per arm (0, 1, 2); an arm is better only if its 100-mesh mean improves by more than the seed range of the baseline. The local configuration is used deliberately so that all arms are equally undertrained; the result applies to that regime.

**Commands** (MPS, behind the shared accelerator lock; `seed=0` shown, repeat with 1 and 2):

```bash
python scripts/run_study.py --config arch --stages train,predict model_variant=baseline      seed=0 name=arch_baseline_s0
python scripts/run_study.py --config arch --stages train,predict model_variant=global_latent seed=0 name=arch_global_latent_s0
python scripts/run_study.py --config arch --stages train,predict model_variant=shared_depth  seed=0 name=arch_shared_depth_s0
python scripts/evaluate_surrogate.py --run runs/arch_global_latent_s0 --config arch --count 100 model_variant=global_latent
```

**Estimated cost**: baseline 2,308 s (measured), global latent about 2,400 s, shared depth about 4,600 s per seed on MPS, so about 7 h for three seeds of all three arms. Disk: 30 MB of checkpoints per run.

## 4. Physics-informed stage

The physics-informed step **does not update the MeshGraphNet**. For each test mesh separately it fits a new coordinate network $(x,y)\mapsto(u,v,p)$ (49,923 parameters) to the frozen MeshGraphNet prediction plus PDE and boundary penalties, for 10,000 Adam iterations. Each variant below uses the same network, seed rule (hence the same Fourier matrix and initial weights as the upstream run of that mesh), optimiser, learning rate and iteration count; only the loss changes (`src/stokes_ft/variants.py`). A test asserts that the `upstream` variant reproduces the loss and parameter updates of `PhysicsInformedFineTuner` exactly.

| Variant | Loss | Question | Status |
|---|---|---|---|
| `upstream` | data + soft inlet/no-slip (weight 10) + strong-form PDE | | 2 meshes (stored, MPS) |
| `data_only` | the three data terms only | C: more than smoothing? | **LOCAL/REDUCED, 10 meshes** (CPU) |
| `data_bc` | data + soft inlet/no-slip, no PDE | D: boundary penalties without the PDE | **LOCAL/REDUCED, 10 meshes** (CPU) |
| `hard_bc` | upstream PDE and data; inlet and no-slip exact by a distance-function ansatz | D | PENDING |
| `outlet` | upstream + outlet condition (weight 10) | E | PENDING |
| `outlet_free_p` | as `outlet` with pressure data weight 0 | E | PENDING |

Source: `scripts/physics_stage_checks.py`, [`physics_stage.json`](../results/validation/physics_stage.json); figure [`physics_stage_controls.png`](../figures/validation/physics_stage_controls.png). The upstream fit exists on meshes 0 and 1 only (G1 is pending); the two controls ran on all ten meshes, 10,000 iterations each, on the CPU.

### 4.1 Controls on ten meshes (LOCAL/REDUCED)

Paired change from the MeshGraphNet prediction, ten meshes; interval: bootstrap 95 % of the mean per-mesh relative change; p: two-sided sign test.

| Metric | Data-only: change, meshes improved | Data + BC penalties: change, meshes improved |
|---|---|---|
| Velocity relative $L^2$ | +0.1 % [−0.4, +0.7], 6/10, p = 0.75 | −0.7 % [−1.3, −0.3], 9/10, p = 0.02 |
| Pressure relative $L^2$ | +0.04 % [−0.6, +0.8], 6/10, p = 0.75 | −0.2 % [−0.9, +0.4], 7/10, p = 0.34 |
| Velocity $H^1$-seminorm error | −6.9 % [−8.7, −5.0], 10/10 | −10.6 % [−14.2, −7.2], 10/10 |
| No-slip RMS, polygon | +39 % [+27, +52], 0/10 | **−80 %** [−86, −73], 10/10 |
| No-slip RMS, channel walls | +25 % [+2, +52], 3/10 | −67 % [−80, −53], 10/10 |
| Inlet RMS | −13 % [−17, −10], 10/10 | −86 % [−93, −78], 10/10 |
| Inlet pressure level, mean absolute error | 0.233 (MeshGraphNet 0.233) | 0.233 |

### 4.2 The upstream fit against the controls, meshes 0 and 1 (LOCAL/REDUCED, n = 2)

| Metric | Reference | MeshGraphNet | Data-only | Data + BC | Physics-informed |
|---|---|---|---|---|---|
| Velocity rel. $L^2$, mesh 0 / 1 | | 0.160 / 0.215 | 0.161 / 0.214 | 0.159 / 0.214 | **0.120 / 0.099** |
| $v$ rel. $L^2$, mesh 0 / 1 | | 0.444 / 0.626 | 0.443 / 0.624 | 0.443 / 0.628 | **0.236 / 0.234** |
| Velocity $H^1$ error, mesh 0 / 1 | | 0.385 / 0.482 | 0.356 / 0.455 | 0.338 / 0.447 | **0.254 / 0.251** |
| Pressure rel. $L^2$, mesh 0 / 1 | | 0.217 / 0.100 | 0.224 / 0.100 | 0.216 / 0.102 | 0.203 / 0.104 |
| Gauge-free pressure error, mesh 0 / 1 | | 0.187 / 0.094 | 0.190 / 0.094 | 0.185 / 0.094 | 0.170 / 0.095 |
| Inlet level (reference 0.889 / 0.752) | | 0.695 / 0.694 | 0.690 / 0.693 | 0.695 / 0.692 | 0.711 / 0.685 |
| Weak momentum residual ($10^{-3}$) | 5.1 / 5.3 | 13.0 / 14.5 | 12.1 / 13.8 | 11.7 / 13.6 | 3.2 / 6.0 |
| Weak continuity residual | 0.21 / 0.20 | 0.60 / 0.80 | 0.56 / 0.77 | 0.55 / 0.76 | 0.09 / 0.11 |
| No-slip RMS, polygon ($10^{-3}$) | 0 | 7.1 / 5.6 | 8.7 / 6.3 | **0.65 / 1.5** | 41.5 / 39.9 |
| No-slip RMS, walls ($10^{-3}$) | 0 | 1.8 / 1.8 | 2.1 / 1.6 | 0.24 / 1.4 | 8.3 / 6.7 |
| Inlet RMS ($10^{-3}$) | 0 | 7.1 / 5.6 | 6.0 / 4.9 | 0.30 / 1.3 | 1.2 / 1.5 |

### 4.3 Answers to the five questions

**A. Accuracy.** On meshes 0 and 1 the velocity error falls from 0.160 and 0.215 to 0.120 and 0.099 (relative $L^2$), and the $H^1$ error from 0.385 and 0.482 to 0.254 and 0.251. Pressure changes by −6.6 % and +3.8 %. These are two meshes of an undertrained surrogate; G1 tests whether they generalise, with decision rules fixed in advance.

**B. Physical consistency.** Interior residuals fall, by the nodal and the weak measures alike, and the continuity residual falls below the reference's own floor on both meshes (0.09 and 0.11 against 0.21 and 0.20). That is below the interpolation floor of a vertex-only field, so it is not evidence that the field is closer to the solution (section 2.3). Boundary conditions get worse: no-slip on the polygon rises sixfold, from 0.007 to 0.041 (14 % of the inlet velocity scale 0.3).

**C. More than smoothing?** Yes for velocity, on the two meshes where it can be tested. Refitting the same network with the same seed and iterations to the MeshGraphNet prediction alone leaves the velocity $L^2$ error unchanged on all ten meshes (mean change +0.1 %, interval −0.4 to +0.7 %), and the soft boundary penalties alone change it by −0.7 %. The data-only fit does lower the $H^1$ error by 7 %: it removes nodal noise in the gradients, not error. The physics-informed fit lowers the $L^2$ error by 25 % and 54 % and the $H^1$ error by 34 % and 48 %. The difference is attributable to the PDE terms. With two meshes it is a measured effect on those meshes, not a population estimate.

**D. Why no-slip deteriorates.** Not because of the soft penalty. With the same penalties (weight 10) and no PDE terms, the no-slip error falls by 80 % on the polygon and 67 % on the walls, on ten of ten meshes. It deteriorates only when the strong-form PDE terms are added. Two mechanisms are consistent with this: the PDE terms are node means over a mesh refined around the polygon, so they dominate there; and at the polygon corners the exact pressure and velocity gradients are singular, so the strong-form residual cannot be satisfied near them and trades against the boundary values. The hard-constraint variant (G6), which removes the trade-off by construction, is implemented and tested but not run.

**E. Why pressure barely moves.** The inlet level is unchanged by every variant run so far (0.711 and 0.685 against 0.889 and 0.752 in the reference). The pressure-recovery experiment (section 3.2) shows that the momentum equation and the outlet condition, solved globally, would move the level in the right direction from the MeshGraphNet velocity (correlation 0.94 over 100 meshes), while the pressure data term holds it at the MeshGraphNet value. The prediction is falsifiable with `outlet_free_p` (G5): with the pressure data weight at 0 and the outlet condition imposed, the inlet level should move toward the reference. One caveat applies before running it: the strong-form momentum term is a local constraint on $\nabla p$, and a coordinate network may propagate the outlet level upstream much more slowly than the global FEM solve does.

## 5. Cost against the finite-element method

Source: `scripts/benchmark_costs.py`, [`costs.json`](../results/validation/costs.json); figure [`costs.png`](../figures/validation/costs.png). Ten test meshes (about 3,300 nodes; the Taylor-Hood system has about 29,000 unknowns). Machine: Apple M3 Pro, shared, load average 5 to 10 during the measurements; times are planning figures for this machine, not hardware benchmarks.

| Method, per new geometry | Time | Peak memory above imports | Accuracy against the reference | Kind |
|---|---:|---:|---|---|
| Taylor-Hood P2-P1 solve, CPU, 1 thread | **0.31 s** (median) | 569 MB | exact (relative $L^2$ below $10^{-9}$) | measured |
| MeshGraphNet inference, CPU, 3 threads | 0.40 s | 126 MB | velocity 0.21, pressure 0.34 (100 meshes) | measured |
| MeshGraphNet inference, MPS | 0.19 s | not measured | as above | stored |
| Data-only fit, 10,000 iterations, CPU, 3 threads | 120 s | 87 MB | unchanged from the MeshGraphNet | measured, 10 runs |
| Physics-informed fit, 10,000 iterations, MPS | 1,690 s | not measured | velocity 0.11 (2 meshes) | stored |
| Physics-informed fit, CPU, 3 threads | about 2,960 s | 486 MB | | extrapolated from 40 timed iterations |

One-off costs of the surrogate: 2,308 s of MeshGraphNet training on MPS (local configuration) plus the 100 training solutions (31 s at the measured solve time).

**Amortised surrogate inference against a one-off solve.** At this mesh size the solver is not the bottleneck it is assumed to be: one direct solve on one CPU thread is faster than MeshGraphNet inference on three, and within a factor of 1.6 of inference on the GPU of this laptop. On the CPU the surrogate never recovers its training cost. On the laptop GPU it saves 0.11 s per geometry, so the 2,308 s of training (plus 31 s of training data) is recovered after about 20,000 geometries, each predicted with a velocity error of about 0.21 where the solve is exact. The physics-informed fit costs about 5,500 solves per geometry and reaches a velocity error of 0.11 where the solve is exact.

**Where a learned surrogate could still be worthwhile** (not tested here): much larger 3-D meshes, where direct solves scale superlinearly in time and memory; many-query settings that need gradients with respect to geometry; or problems without a cheap, trusted solver. This 2-D Stokes problem with a direct solver is none of these.

## 6. Pending runs and their commands

| Run | Blocker on 2026-11-06 | Command | Estimated cost |
|---|---|---|---|
| G1: upstream physics stage on test meshes 2 to 9 (decision rules fixed in [`EXPERIMENT_GAPS.md`](EXPERIMENT_GAPS.md)) | Shared MPS lock held by another project's training sweep; load average 6 to 8 on 11 cores; disk 98 % full (12 GiB free, ample for this run) | `python scripts/run_study.py --config local --resume` then `python scripts/results.py publish runs/local`, `python scripts/physics_stage_checks.py` | about 4.5 h MPS on an idle machine; 12 to 17 h on the CPU |
| G4 completion against the physics stage | needs G1 for meshes 2 to 9 | none beyond G1 | none |
| G5 `outlet`, `outlet_free_p`, meshes 0 and 1 | MPS lock | `python scripts/run_controls.py --config local --variant outlet_free_p --samples 0,1 --device mps` (and `--variant outlet`) | about 0.5 h per mesh and variant on MPS |
| G6 `hard_bc`, meshes 0 and 1 | MPS lock | `python scripts/run_controls.py --config local --variant hard_bc --samples 0,1 --device mps` | about 0.6 h per mesh on MPS |
| G12 architecture experiment | MPS lock | section 3.3 | about 7 h MPS |
| G2 full CUDA study | no CUDA device here; the Kaggle launcher can only check out a commit that exists on GitHub, and this branch is not pushed | `kaggle/run_cuda.ipynb` with `REF` set to the pushed commit; steps B and C (smoke test, timing and memory probe) run before any full study, and step D refuses to start if the projection exceeds the session budget | no probe exists yet; a projection from the MPS rates is not a substitute and is not given |

The CUDA smoke test and probe were **not run**: no CUDA device is available on this machine. No GPU timing, memory figure or convergence curve in this repository is measured on CUDA.

## 7. Reproduction

Environment: the project's uv virtual environment (`.venv`, Python 3.12.2, torch 2.14.1, PhysicsNeMo at `b45a5c8`); `requirements.txt` lists the direct dependencies and `requirements.lock.txt` the exact resolved environment (`uv pip freeze`, 113 packages; `torch-scatter` is built from source as `requirements.txt` describes). Tests run on the CPU without network or dataset:

```bash
.venv/bin/python -m pytest -q
```

Analyses, all on the CPU, in this order:

```bash
python scripts/geometry_pressure.py                                    # 1000 files, about 1 min
python scripts/evaluate_surrogate.py --run runs/local --config local --count 100
python scripts/run_controls.py --config local --variant data_only --samples 0,1,2,3,4,5,6,7,8,9 --threads 3
python scripts/run_controls.py --config local --variant data_bc   --samples 0,1,2,3,4,5,6,7,8,9 --threads 3
python scripts/physics_stage_checks.py
python scripts/benchmark_costs.py --threads 3
python scripts/validation_figures.py
```

Seeds: MeshGraphNet `seed: 0`; coordinate networks `seed + 1000 + index` for every variant; bootstrap seed 0. Each output JSON records the commit, dirty state and package versions.

## 8. Limitations

* **Undertrained surrogate.** Every learned result uses the local MeshGraphNet (1.6 % of the upstream iterations). The receptive-field blindness is architectural and holds at any training length (it is exact: zero response beyond 15 hops); the size of the other errors, and the effect of the physics-informed step on a well-trained surrogate, are unknown until G2.
* **Two meshes for the upstream physics-informed fit.** All its paired results have n = 2. The controls have n = 10, but their comparison with the physics fit is limited to meshes 0 and 1.
* **One seed per mesh** for every coordinate-network fit (G3). The controls share the seed of the upstream run, which removes initialisation as a confounder but does not measure seed variance.
* **Vertex-only evaluation.** No residual computed from vertex values certifies accuracy (section 2.3). Errors against the Taylor-Hood reference are the primary evidence throughout.
* **Hard-BC variant collocation.** `hard_bc` collocates the PDE at all nodes except the polygon, where the distance function is singular and the ansatz fixes the velocity; the upstream loss also uses the polygon nodes. This difference will have to be stated with its result.
* **Timings** come from a machine shared with other sessions and differ between devices and thread counts; ratios of order 10 or more are robust, factors of 2 are not.
* **No CUDA result** of any kind exists in this repository.

## 9. Remaining research questions

1. Does the PDE-driven velocity improvement hold across the ten meshes (G1), and on a well-trained surrogate (G2)?
2. Does a global latent recover the inlet level at 1.3 % extra parameters, and by how much more than seed noise (G12)?
3. Does removing the pressure data term let the outlet condition fix the level inside the coordinate network (G5), as the FEM pressure recovery suggests it should?
4. Does exact boundary imposition (G6) keep the velocity gain while removing the no-slip loss?
5. At what mesh size, if any, does surrogate inference become cheaper than the direct solve on this hardware?
