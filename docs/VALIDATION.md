# Validation study (November 2026)

Continuation of the [scientific audit](SCIENTIFIC_AUDIT.md). It asks when a learned graph surrogate of this Stokes problem is accurate, physically consistent and worth its cost compared with the finite-element method, and it separates what was measured from what remains proposed.

Branch `research/validation-2026-11`, based on `57926cb`. Every number here comes from a file in [`results/validation/`](../results/validation/), produced by a script named next to it. Historical results (`results/audit_checks.json`, `results/physics_checks.json`) were not modified. `results/local/` was republished once, after G1 added meshes 2 to 9 to the same run; its mesh 0 and 1 rows are unchanged and the two-mesh version is kept in `results/local_2mesh/`.

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

### 3.3 Controlled architecture experiment (LOCAL/REDUCED, 3 arms x 3 seeds, completed 2026-11-07)

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

The queue (`scripts/mps_queue.sh arch`) ran the nine runs with `--stages train,predict --resume`; each was then evaluated on the CPU by `scripts/evaluate_surrogate.py` and the fixed rule applied by `scripts/arch_compare.py` (committed at `b3921e1`, before any arm was trained). Every run recorded a clean working tree. The model sees only node and edge inputs; a test checks that changing the reference fields leaves every arm's output unchanged.

**Results** on the same first 100 test meshes ([`arch_comparison.json`](../results/validation/arch_comparison.json), [`surrogate_arch_*_test100.json`](../results/validation/); figure [`architecture.png`](../figures/validation/architecture.png)). Values are means over seeds 0, 1, 2, with the seed range in brackets.

| Metric | Baseline | Global latent | Shared depth | Verdict, global latent | Verdict, shared depth |
|---|---|---|---|---|---|
| Inlet-level slope (ideal 1) | 0.0001 [0.0000, 0.0002] | 0.0155 [0.0133, 0.0193] | 0.0002 [0.0001, 0.0003] | better | indistinguishable |
| Inlet-level RMS error | 0.2373 [0.2368, 0.2376] | 0.2334 [0.2325, 0.2342] | 0.2371 [0.2367, 0.2375] | better | indistinguishable |
| Inlet-level correlation | 0.06 [0.01, 0.11] | **0.81** [0.80, 0.82] | 0.07 [0.04, 0.11] | better | indistinguishable |
| Pressure rel. $L^2$ | 0.3352 [0.3340, 0.3367] | 0.3310 [0.3279, 0.3348] | 0.3359 [0.3333, 0.3377] | better | indistinguishable |
| Velocity rel. $L^2$ | 0.2098 [0.2089, 0.2103] | 0.2084 [0.2079, 0.2092] | 0.2108 [0.2106, 0.2109] | better (margin 0.0014 against a range of 0.0014) | indistinguishable |
| Gauge-free pressure error | 0.2881 | 0.2880 | 0.2884 | indistinguishable | indistinguishable |
| Velocity $H^1$ error | 0.5108 | 0.5061 | 0.5102 | indistinguishable | indistinguishable |
| No-slip RMS, walls | 0.0016 | 0.0021 | 0.0020 | worse | worse |
| Parameters | 2,531,587 | 2,564,611 | 2,531,587 | | |
| Training, 40 epochs, MPS (median) | 745 s | 993 s | 1,391 s | | |
| Inference per mesh, MPS / CPU 3 threads (median) | 0.074 / 0.29 s | 0.120 / 0.34 s | 0.139 / 0.58 s | | |

Reproducibility check: `arch_baseline_s0` has the configuration of `runs/local` with the new keys at their defaults, and reproduces its 100-mesh errors (velocity 0.210, pressure 0.337 in both).

**Reading.** Under the pre-registered rule the global latent is **better** than the baseline on both primary endpoints and on pressure relative $L^2$, so it meets the rule's condition for a pressure claim. The sizes matter more than the verdict:

* The global latent carries the missing information: its predicted inlet level correlates 0.81 with the reference on every seed, against 0.06 for the baseline. This is the direct test of section 3.1's explanation, and it supports it: the information reaches the inlet once the receptive field covers the domain.
* It barely uses it: the slope is 0.016, so the predicted level moves by under 2 % of the required amount, and the pressure error falls by 1.3 % (0.335 to 0.331). The gauge-free pressure error, which excludes the level, is unchanged, as expected if the arm acts only on the level.
* Shared depth, at 30 hops, is indistinguishable from the baseline on every accuracy metric while costing 1.9 times the training and inference time. Thirty hops reach the inlet only from the nearest obstacles; the global level needs the whole obstacle.
* The rule compares against the range of three baseline seeds, which is a lenient threshold. The velocity verdict sits exactly at that threshold and should not be read as an improvement.

Why the effect is small is not established. Two explanations fit and were not separated: the zero-initialised global path and 40 epochs leave too little training to learn a large output scale for one pooled feature; or the mean-pooled latent of a 15-hop encoder is a weak summary of the channel resistance, which section 3.1 shows depends on the gap widths. A longer schedule (G2) or a pooled feature with an explicit scale (for example, attention pooling) would separate them. The multiscale graph is not needed to show that the information is reachable, but nothing here shows the global latent is sufficient.

## 4. Physics-informed stage

The physics-informed step **does not update the MeshGraphNet**. For each test mesh separately it fits a new coordinate network $(x,y)\mapsto(u,v,p)$ (49,923 parameters) to the frozen MeshGraphNet prediction plus PDE and boundary penalties, for 10,000 Adam iterations. Each variant below uses the same network, seed rule (hence the same Fourier matrix and initial weights as the upstream run of that mesh), optimiser, learning rate and iteration count; only the loss changes (`src/stokes_ft/variants.py`). A test asserts that the `upstream` variant reproduces the loss and parameter updates of `PhysicsInformedFineTuner` exactly.

| Variant | Loss | Question | Status |
|---|---|---|---|
| `upstream` | data + soft inlet/no-slip (weight 10) + strong-form PDE | A, B | **LOCAL/REDUCED, 10 meshes** (MPS; G1 completed 2026-11-07) |
| `data_only` | the three data terms only | C: more than smoothing? | **LOCAL/REDUCED, 10 meshes** (CPU) |
| `data_bc` | data + soft inlet/no-slip, no PDE | D: boundary penalties without the PDE | **LOCAL/REDUCED, 10 meshes** (CPU) |
| `hard_bc` | upstream PDE and data; inlet and no-slip exact by a distance-function ansatz | D | **LOCAL/REDUCED, meshes 0 and 1** (MPS) |
| `outlet` | upstream + outlet condition (weight 10) | E | **LOCAL/REDUCED, meshes 0 and 1** (MPS) |
| `outlet_free_p` | as `outlet` with pressure data weight 0 | E | **LOCAL/REDUCED, meshes 0 and 1** (MPS) |

Source: `scripts/physics_stage_checks.py`, [`physics_stage.json`](../results/validation/physics_stage.json); figure [`physics_stage_controls.png`](../figures/validation/physics_stage_controls.png). The upstream fit and the two controls ran on all ten meshes, 10,000 iterations each (upstream on MPS, controls on the CPU). Meshes 0 and 1 of the upstream fit were run in October; meshes 2 to 9 on 2026-11-07 by resuming the same study (identical configuration, checkpoint and seed rule; the mesh 0 and 1 artifacts were checked bit-identical by SHA-256 afterwards). The two-mesh results as first published are kept in [`results/local_2mesh`](../results/local_2mesh).

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

### 4.2 G1: the upstream fit on ten meshes, pre-registered decision (LOCAL/REDUCED, n = 10)

Source: `scripts/g1_decision.py` (committed at `ce18d14`, before any mesh 2 to 9 was fine-tuned, applying the rules fixed at `57926cb` in [`EXPERIMENT_GAPS.md`](EXPERIMENT_GAPS.md)), [`g1_decision.json`](../results/validation/g1_decision.json).

| Pre-registered test | Result | Outcome |
|---|---|---|
| Primary: velocity relative $L^2$, physics fit against MeshGraphNet, paired over the 10 meshes; generalises if at least 9 of 10 improve and the bootstrap 95 % interval of the mean relative change lies below 0 | 9 of 10 improve (sign test p = 0.021, Wilcoxon p = 0.010); mean change −37.7 % [−54.8, −19.3]; median −52.8 % | **Generalises across this test set** (local, undertrained surrogate only) |
| Falsification of the receptive-field explanation: contradicted if the fitted inlet level has slope ≥ 0.5 and correlation ≥ 0.8 against the reference level | slope 0.13, correlation 0.81 (MeshGraphNet: slope −0.0002, correlation −0.10) | **Not contradicted** (slope criterion fails) |

Per-mesh velocity change: −25, −54, −75, −57, −64, −57, −1, **+11**, −51, −4 % (meshes 0 to 9). Mesh 7 got worse and meshes 6 and 9 barely moved; the earlier two-mesh figures (−25 % and −54 %) were typical, not best cases.

Means over the ten meshes (paired changes in the next table):

| Metric | Reference | MeshGraphNet | Data-only | Data + BC | Physics-informed |
|---|---|---|---|---|---|
| Velocity rel. $L^2$ | | 0.233 | 0.233 | 0.232 | **0.136** |
| $v$ rel. $L^2$ | | 0.665 | 0.665 | 0.665 | **0.307** |
| Velocity $H^1$ error | | 0.518 | 0.485 | 0.467 | **0.299** |
| Pressure rel. $L^2$ | | 0.334 | 0.335 | 0.334 | **0.257** |
| Gauge-free pressure error | | 0.286 | 0.286 | 0.287 | **0.181** |
| Inlet level, mean abs. error (reference mean 0.741) | | 0.233 | 0.233 | 0.233 | 0.195 |
| Weak momentum residual ($10^{-3}$) | 4.5 | 17.3 | 16.6 | 16.3 | 6.1 |
| Weak continuity residual | 0.174 | 0.764 | 0.731 | 0.713 | 0.079 |
| No-slip RMS, polygon ($10^{-3}$) | 0 | 7.6 | 10.5 | **1.4** | 45.8 |
| No-slip RMS, walls ($10^{-3}$) | 0 | 1.8 | 2.2 | 0.6 | 8.0 |
| Inlet RMS ($10^{-3}$) | 0 | 6.0 | 5.2 | 0.8 | 1.7 |

Paired change of the physics-informed fit, ten meshes (bootstrap 95 % interval of the mean per-mesh relative change; two-sided sign test):

| Metric | Against MeshGraphNet | Against the data-only control |
|---|---|---|
| Velocity rel. $L^2$ | −37.7 % [−54.8, −19.3], 9/10, p = 0.021 | −37.8 % [−54.8, −19.8], 9/10, p = 0.021 |
| Velocity $H^1$ error | −42.5 % [−51.2, −31.8], 10/10, p = 0.002 | −38.4 % [−47.5, −28.0], 10/10, p = 0.002 |
| Pressure rel. $L^2$ | −15.6 % [−26.5, −4.9], 9/10, p = 0.021 | −15.6 % [−26.6, −4.8], 7/10, p = 0.34 |
| Gauge-free pressure error | −24.8 % [−42.7, −7.5], 8/10, p = 0.11 | −24.8 % [−42.6, −7.5], 8/10, p = 0.11 |
| Inlet level abs. error | +16.9 % [−28, +87], 6/10, p = 0.75 (means 0.233 → 0.195) | +17.1 % [−28, +89], 6/10, p = 0.75 |
| No-slip RMS, polygon | **+550 %** [+442, +668], 0/10 | +378 % [+292, +466], 0/10 |
| No-slip RMS, walls | +342 % [+220, +519], 0/10 | +269 % [+170, +386], 0/10 |
| Inlet RMS | −71.5 % [−81.2, −59.7], 10/10 | −67.7 % [−77.7, −55.7], 10/10 |

The inlet-level mean of relative changes is dominated by mesh 4, where the MeshGraphNet level error is 0.008; the mean absolute error falls from 0.233 to 0.195, but the improvement is not consistent across meshes (6 of 10).

### 4.3 Answers to the five questions

**A. Accuracy.** Over ten meshes the velocity error falls by 38 % on average (0.233 to 0.136), on 9 of 10 meshes, and the $H^1$ error by 43 % on all ten. This meets the pre-registered criterion for generalisation across this test set. Pressure improves by 16 % (9 of 10 meshes), mostly in shape (gauge-free error −25 %) rather than level (inlet level error improved on 6 of 10). The effect is measured on an undertrained surrogate; its size on a well-trained one (G2) is unknown and could be smaller, since there is less error to remove.

**B. Physical consistency.** Interior residuals fall on all ten meshes, by the nodal and the weak measures alike (weak momentum −66 %, weak continuity −89 %). The continuity residual falls below the reference's own floor (mean 0.079 against 0.174), which is below the interpolation floor of a vertex-only field, so it is not evidence that the field is closer to the solution (section 2.3). Boundary conditions get worse on every mesh: no-slip on the polygon rises about sixfold, from 0.0076 to 0.046 (15 % of the inlet velocity scale 0.3), and on the walls about fourfold. Only the inlet condition improves.

**C. More than smoothing?** Yes, for velocity and the $H^1$ error, on ten meshes. Refitting the same network with the same seed and iterations to the MeshGraphNet prediction alone leaves the velocity $L^2$ error unchanged (mean change +0.1 %, interval −0.4 to +0.7 %), and the soft boundary penalties alone change it by −0.7 %. The data-only fit lowers the $H^1$ error by 7 %: it removes nodal noise in the gradients, not error. Against this control the physics-informed fit lowers the velocity $L^2$ error by 38 % [−55, −20] on 9 of 10 meshes and the $H^1$ error by 38 % on 10 of 10. The difference is attributable to the PDE terms. Against the same control the pressure gain (−16 %) is less consistent (7 of 10 meshes, sign test p = 0.34).

**D. Why no-slip deteriorates.** Not because of the soft penalty. With the same penalties (weight 10) and no PDE terms, the no-slip error falls by 80 % on the polygon and 67 % on the walls, on ten of ten meshes. It deteriorates only when the strong-form PDE terms are added. Two mechanisms are consistent with this: the PDE terms are node means over a mesh refined around the polygon, so they dominate there; and at the polygon corners the exact pressure and velocity gradients are singular, so the strong-form residual cannot be satisfied near them and trades against the boundary values. The hard-constraint variant (G6) removes the trade-off by construction; on meshes 0 and 1 (section 4.4) it meets every boundary condition to single precision, but it does not keep the velocity gain on both meshes: the error rises above the upstream fit on mesh 0 (0.139 against 0.120) and above the MeshGraphNet itself on mesh 1 (0.246 against 0.215), and the interior continuity residual is four times that of the upstream fit. Exact boundary values move the conflict into the interior next to the boundary rather than removing it.

**E. Why pressure barely moves.** The inlet level barely moves. Over ten meshes the fitted level has correlation 0.81 with the reference but slope 0.13 (the MeshGraphNet: correlation −0.10, slope 0.0002): the PDE terms move it in the right direction by about an eighth of the required amount, and the pre-registered falsification threshold (slope ≥ 0.5) is not reached. The controls leave it unchanged. The pressure-recovery experiment (section 3.2) shows that the momentum equation and the outlet condition, solved globally, would move the level in the right direction from the MeshGraphNet velocity (correlation 0.94 over 100 meshes), while the pressure data term holds it at the MeshGraphNet value. The prediction tested with `outlet_free_p` (G5) was that, with the pressure data weight at 0 and the outlet condition imposed, the inlet level would move toward the reference. **It did not**: on meshes 0 and 1 the level moved away from the reference (0.675 and 0.605 against 0.889 and 0.752; upstream fit 0.711 and 0.685) and the pressure error rose (0.244 and 0.199 against 0.203 and 0.104), although the outlet pressure is closer to zero. The caveat stated before the run applies: the strong-form momentum term constrains $\nabla p$ locally, and the coordinate network does not propagate the outlet level upstream the way a global solve does. Adding the outlet term while keeping the pressure data (`outlet`) changes almost nothing: the outlet pressure RMS falls from 0.003 and 0.012 to 0.0006 and 0.0009, the inlet level is unchanged to three decimals, and velocity and pressure errors change by 1 to 4 %. Conclusion on two meshes: the outlet condition is not the missing constraint for this loss; the level needs a global mechanism (section 3.3) or a global solve.

### 4.4 Hard-BC and outlet variants, meshes 0 and 1 (LOCAL/REDUCED, n = 2)

Same network, seed, optimiser and 10,000 iterations as the upstream fit of each mesh; MPS. Values for mesh 0 / mesh 1. With two meshes these are descriptive; no interval is given.

| Metric | MeshGraphNet | Upstream fit | `hard_bc` | `outlet` | `outlet_free_p` |
|---|---|---|---|---|---|
| Velocity rel. $L^2$ | 0.160 / 0.215 | 0.120 / 0.099 | 0.139 / 0.246 | **0.117 / 0.097** | 0.141 / 0.126 |
| Velocity $H^1$ error | 0.385 / 0.482 | 0.254 / 0.251 | **0.236** / 0.369 | 0.254 / 0.257 | 0.306 / 0.302 |
| Pressure rel. $L^2$ | 0.217 / 0.100 | 0.203 / 0.104 | **0.175** / 0.196 | 0.202 / 0.100 | 0.244 / 0.199 |
| Gauge-free pressure error | 0.187 / 0.094 | 0.170 / 0.095 | **0.136** / 0.191 | 0.167 / 0.087 | 0.200 / 0.169 |
| Inlet level (reference 0.889 / 0.752) | 0.695 / 0.694 | 0.711 / 0.685 | 0.736 / 0.624 | 0.712 / 0.685 | 0.675 / 0.605 |
| Weak momentum residual ($10^{-3}$) | 13.0 / 14.5 | 3.2 / 6.0 | 6.0 / 13.1 | 3.3 / 5.6 | 2.9 / 3.8 |
| Weak continuity residual | 0.60 / 0.80 | 0.09 / 0.11 | 0.41 / 0.42 | 0.09 / 0.10 | 0.08 / 0.08 |
| No-slip RMS, polygon | 0.0071 / 0.0056 | 0.042 / 0.040 | **5e-8 / 7e-8** | 0.042 / 0.042 | 0.054 / 0.048 |
| No-slip RMS, walls | 0.0018 / 0.0017 | 0.0083 / 0.0067 | **0 / 0** | 0.0082 / 0.0066 | 0.0090 / 0.0096 |
| Inlet RMS | 0.0071 / 0.0056 | 0.0012 / 0.0015 | **3e-8 / 3e-8** | 0.0013 / 0.0017 | 0.0023 / 0.0027 |
| Outlet pressure RMS | 0.0062 / 0.0062 | 0.0028 / 0.0116 | 0.0012 / 0.060 | **0.0006 / 0.0009** | 0.0017 / 0.0030 |
| Time, 10,000 iterations, MPS | | 336 s (median, 10 meshes) | 510 / 526 s | 328 / 336 s | 327 / 335 s |

The hard-BC constraints were checked on the trained networks: on every polygon node, corners included, the velocity is zero to float32 precision (RMS $5\times10^{-8}$ and $7\times10^{-8}$), the channel walls are exactly zero, and the inlet profile is exact to $3\times10^{-8}$. The PDE is collocated at all nodes except the polygon (section 8). On mesh 0 `hard_bc` gives the best pressure and $H^1$ errors of all variants; on mesh 1 it is worse than the MeshGraphNet on velocity and pressure, with a large outlet pressure (0.060). Two meshes cannot say which is typical.

## 5. Cost against the finite-element method

Source: `scripts/benchmark_costs.py`, [`costs.json`](../results/validation/costs.json); figure [`costs.png`](../figures/validation/costs.png). Ten test meshes (about 3,300 nodes; the Taylor-Hood system has about 29,000 unknowns). Machine: Apple M3 Pro, shared, load average 5 to 10 during the measurements; times are planning figures for this machine, not hardware benchmarks.

| Method, per new geometry | Time | Peak memory above imports | Accuracy against the reference | Kind |
|---|---:|---:|---|---|
| Taylor-Hood P2-P1 solve, CPU, 1 thread | **0.31 s** (median) | 569 MB | exact (relative $L^2$ below $10^{-9}$) | measured |
| MeshGraphNet inference, CPU, 3 threads | 0.40 s | 126 MB | velocity 0.21, pressure 0.34 (100 meshes) | measured |
| MeshGraphNet inference, MPS | 0.19 s | not measured | as above | stored |
| Data-only fit, 10,000 iterations, CPU, 3 threads | 120 s | 87 MB | unchanged from the MeshGraphNet | measured, 10 runs |
| Physics-informed fit, 10,000 iterations, MPS | **336 s** (median of 10 meshes; 326 to 1,942 s) | not measured | velocity 0.136, pressure 0.257 (10 meshes) | stored |
| Physics-informed fit, CPU, 3 threads | about 2,960 s | 486 MB | | extrapolated from 40 timed iterations |

One-off costs of the surrogate: 2,308 s of MeshGraphNet training on MPS (local configuration) plus the 100 training solutions (31 s at the measured solve time).

**Amortised surrogate inference against a one-off solve.** At this mesh size the solver is not the bottleneck it is assumed to be: one direct solve on one CPU thread is faster than MeshGraphNet inference on three, and within a factor of 1.6 of inference on the GPU of this laptop. On the CPU the surrogate never recovers its training cost. On the laptop GPU it saves 0.11 s per geometry, so the 2,308 s of training (plus 31 s of training data) is recovered after about 20,000 geometries, each predicted with a velocity error of about 0.21 where the solve is exact. The physics-informed fit costs about 1,100 solves per geometry (336 s on MPS against 0.31 s on one CPU thread; different hardware, so the ratio is a cost on this machine, not a like-for-like speed comparison) and reaches a velocity error of 0.136 where the solve is exact. The median is used because four of the ten runs shared the accelerator with other jobs or spanned a system sleep (meshes 0, 1, 4, 5: 1,942, 1,437, 1,437 and 980 s); the six others took 326 to 341 s. The earlier figure of 1,690 s was the mean of meshes 0 and 1.

**Where a learned surrogate could still be worthwhile** (not tested here): much larger 3-D meshes, where direct solves scale superlinearly in time and memory; many-query settings that need gradients with respect to geometry; or problems without a cheap, trusted solver. This 2-D Stokes problem with a direct solver is none of these.

## 6. Completed and pending runs

Completed on 2026-11-07, on MPS behind the shared accelerator lock (`scripts/mps_queue.sh all`; logs in `runs/queue_logs/`, untracked): G1 (section 4.2), the nine architecture runs (section 3.3), and `hard_bc`, `outlet`, `outlet_free_p` on meshes 0 and 1 (section 4.4).

| Pending run | Blocker | Command | Estimated cost |
|---|---|---|---|
| G2 full CUDA study | no CUDA device here; the Kaggle launcher can only check out a commit that exists on GitHub, and this branch is not pushed | `kaggle/run_cuda.ipynb` with `REF` set to the pushed commit; steps B and C (smoke test, timing and memory probe) run before any full study, and step D refuses to start if the projection exceeds the session budget | no probe exists yet; a projection from the MPS rates is not a substitute and is not given |
| G5/G6 variants on meshes 2 to 9 (optional; would turn section 4.4 into a ten-mesh paired comparison) | none; not run in this session | `python scripts/run_controls.py --config local --variant hard_bc --samples 2,3,4,5,6,7,8,9 --device mps` (and `outlet`, `outlet_free_p`), then `python scripts/physics_stage_checks.py` | about 70 min (`hard_bc`) and 45 min (each outlet variant) on an idle MPS |
| G3 fine-tuning seed sensitivity | needs a `finetune_seed` override (not implemented) | none yet | about 6 min per mesh and seed on MPS |

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
python scripts/arch_compare.py                                         # after the nine arch runs and their evaluate_surrogate.py calls
python scripts/g1_decision.py
python scripts/validation_figures.py
```

Accelerator runs (MPS, behind the shared lock, in this order): `bash scripts/mps_queue.sh g1`, `bash scripts/mps_queue.sh arch`, `bash scripts/mps_queue.sh variants`. Each architecture run is evaluated with `python scripts/evaluate_surrogate.py --run runs/arch_<arm>_s<seed> --config arch --count 100 model_variant=<arm> seed=<seed> name=arch_<arm>_s<seed>`.

Seeds: MeshGraphNet `seed: 0`; coordinate networks `seed + 1000 + index` for every variant; bootstrap seed 0. Each output JSON records the commit, dirty state and package versions.

## 8. Limitations

* **Undertrained surrogate.** Every learned result uses the local MeshGraphNet (1.6 % of the upstream iterations). The receptive-field blindness is architectural and holds at any training length (it is exact: zero response beyond 15 hops); the size of the other errors, and the effect of the physics-informed step on a well-trained surrogate, are unknown until G2.
* **Ten test meshes.** G1 and the controls have n = 10; the intervals are wide (velocity −55 % to −19 %) and a sign test on ten meshes cannot reach p < 0.002. The 100-mesh evaluation (section 3.2) covers the MeshGraphNet only.
* **One seed per mesh** for every coordinate-network fit (G3). The controls share the seed of the upstream run, which removes initialisation as a confounder but does not measure seed variance.
* **Vertex-only evaluation.** No residual computed from vertex values certifies accuracy (section 2.3). Errors against the Taylor-Hood reference are the primary evidence throughout.
* **Hard-BC variant collocation.** `hard_bc` collocates the PDE at all nodes except the polygon, where the distance function is singular and the ansatz fixes the velocity; the upstream loss also uses the polygon nodes. Its results (section 4.4) therefore differ from the upstream fit in two ways, the constraint and the collocation set.
* **Two meshes for the hard-BC and outlet variants**, and three seeds per architecture arm. The architecture rule compares against the range of three baseline seeds, a lenient threshold.
* **Timings** come from a machine shared with other sessions and differ between devices and thread counts; ratios of order 10 or more are robust, factors of 2 are not.
* **No CUDA result** of any kind exists in this repository.

## 9. Remaining research questions

1. Does the PDE-driven velocity improvement, established on ten meshes of the local surrogate (G1), hold on a well-trained surrogate (G2)?
2. The global latent ranks the inlet level correctly (correlation 0.81) but moves it by under 2 % of the required amount. Is that a training-budget effect or a limit of mean pooling (G2, or an attention-pooled variant)?
3. Removing the pressure data term did not let the outlet condition fix the level in the coordinate network (G5, two meshes). Does a weak-form or global pressure solve inside the fit do so, as the FEM pressure recovery suggests?
4. Exact boundary imposition (G6) removed the no-slip loss but kept the velocity gain on only one of two meshes. Does that hold on ten meshes, and does a distance function with a smoother corner behaviour help?
5. At what mesh size, if any, does surrogate inference become cheaper than the direct solve on this hardware?
