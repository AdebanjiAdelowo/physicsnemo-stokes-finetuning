# Why the full MeshGraphNet predicts velocity better than pressure (November 2026)

Follow-up to [`VALIDATION.md`](VALIDATION.md), section 10. Every analysis here uses existing weights and outputs on the CPU; no model was trained. Labels: **COMPLETED** (computed and committed), **PROPOSED** (designed, not run), **BLOCKED** (cannot be done with what exists).

Sources: `scripts/pressure_diagnostics.py` and `scripts/boundary_diagnostics.py`; results in `results/validation/pressure_diag_*.json` and `boundary_diag_*.json`.

## 1. Formulation

Steady Stokes flow, $-\nu\Delta\mathbf u + \nabla p = 0$, $\nabla\cdot\mathbf u = 0$, with a parabolic inlet, no slip on the walls and the obstacle, and the natural outlet condition $\nu\,\partial_n\mathbf u - p\,\mathbf n = 0$ (section 1 of the validation report). Two facts organise the analysis:

* **The momentum equation fixes the pressure gradient by the velocity**, $\nabla p = \nu\Delta\mathbf u$, and the outlet condition fixes the level. Given an exact velocity, the pressure is fully determined. At a point upstream of the obstacle the level is the integral of $\nu\Delta u$ along a path to the outlet, which crosses the obstacle; in this dataset it ranges from 0.30 to 3.53 and follows the obstacle's hydraulic resistance (correlation 0.89 with a lubrication estimate over 1000 files).
* **The MeshGraphNet is local.** Its 15 message-passing layers give each node a 15-hop receptive field; 99.4 % of the inlets lie further than 15 hops from the obstacle. The network predicts u, v and p from one shared node latent through one shared decoder MLP (2 hidden layers of 256), trained with an equally weighted mean squared error of the three standardised fields (global mean and standard deviation per field).

## 2. Checkpoint inventory (COMPLETED)

| Run | Data-driven checkpoints that exist | Fine-tuned networks |
|---|---|---|
| Full (`runs/full`, from the Kaggle checkpoint archive) | epochs **450, 475, 500** (the newest three are kept; all verify against their manifests) | 10 |
| Local (`runs/local`) | epoch 40 | 10 |

The validation pressure error of the full run was lowest at epoch 68 and the validation velocity errors at epoch 172 (`results/full/history.csv`). **Those weights do not exist.** Evaluating early or validation-selected checkpoints is **BLOCKED** without retraining; nothing here reconstructs or infers them. Epochs 450, 475 and 500 differ by less than 1 % on every measure below, so selecting among them is uninformative.

## 3. Pressure-error decomposition (COMPLETED)

For a predicted pressure $p$ with error $e = p - p_{\mathrm{ref}}$ on the Taylor-Hood pressure space (`src/stokes_ft/pressure_diagnostics.py`, tested on exact Poiseuille fields): relative $L^2$ error; gauge-free error after the best constant and the share of the squared error that constant explains; relative gradient error $\|\nabla e\|/\|\nabla p_{\mathrm{ref}}\|$; shares of $\|e\|^2$ upstream of, within and downstream of the obstacle's $x$-range; inlet and outlet errors. Because vertex fields cannot represent the reference exactly, every measure is also computed for the pressure recovered from the **reference** velocity: the floor (relative $L^2$ 0.020, gradient 0.46 on the test meshes).

Means over the same 100 test meshes:

| | Local, epoch 40 | Full, epoch 500 |
|---|---|---|
| Velocity relative $L^2$ | 0.210 | 0.120 |
| Pressure relative $L^2$ | 0.337 | 0.369 |
| Gauge-free | 0.288 | 0.329 |
| Share explained by one constant | 22 % | **6 %** |
| Share of squared error upstream / at / downstream of the obstacle | 77 / 20 / 2 % | **94 / 6 / 0 %** |
| Mean inlet error | +0.042 | +0.076 |
| Gradient error (floor 0.46) | 1.32 | **5.33** |

The full model's pressure error is almost entirely a **regional level error upstream of the obstacle** (94 % of the squared error), which a global constant cannot remove: the outlet and the downstream field are right (outlet RMS 0.001), the upstream plateau is at the wrong height. Superposed on it is **node-scale noise**: the gradient of its pressure error is 5.3 times the reference gradient, four times the local model's, and 2.7 even on its training meshes.

## 4. Overfitting is memorisation of the pressure level (COMPLETED)

The full model evaluated on its own training meshes, on the validation meshes and on the test meshes (epoch 500):

| | Train (500) | Validation (10) | Test (100) |
|---|---|---|---|
| Velocity relative $L^2$ | 0.042 | 0.127 | 0.121 |
| Pressure relative $L^2$ | **0.108** | 0.356 | **0.369** |
| Inlet level: slope, correlation with the reference | **0.92, 0.99** | 0.04, 0.13 | **0.10, 0.20** |
| Inlet level: correlation with the lubrication estimate (reference: 0.90, 0.92, 0.89) | 0.89 | 0.01 | 0.24 |

The local model, for comparison, has inlet-level correlation 0.12 on its own training meshes and 0.03 on the test meshes: it never learned the level anywhere.

**Receptive-field check on the full weights** (`pressure_diag_full_e500_perturbation.json`): relabelling and displacing the obstacle's input features on 20 test meshes changes the output by **exactly 0.0** at every node beyond 15 hops and at every inlet node (median change within reach 3.6).

Together these establish the mechanism. On the training meshes the full model predicts the geometry-dependent inlet level almost perfectly, yet its inlet nodes provably cannot see the obstacle. The only information available to them is the mesh within 15 hops of the inlet (node positions, edge geometry, boundary markers), which differs between meshes because each geometry was meshed separately. Over 250,000 iterations the network learns to recognise training meshes from these local signatures and to recall their levels. That mapping does not carry over to new geometries (test correlation 0.20), and its errors on them are larger than those of the local model's constant prediction (inlet RMS error 0.252 against 0.237). The rise of the validation pressure error after epoch 68 is the signature of this memorisation; the velocity, which depends mostly on local geometry, generalises much better (train 0.042, test 0.121).

## 5. Hypotheses (COMPLETED unless marked)

| Hypothesis for the pressure failure | Evidence | Verdict |
|---|---|---|
| Pressure-specific overfitting | Train/test pressure ratio 3.4 with inlet-level correlation 0.99 on training and 0.20 on test meshes; validation pressure minimum at epoch 68 | **Supported**: memorisation of the level (section 4) |
| Insufficient global geometric information | Exact zero response of every inlet node to the obstacle on the full weights; level error concentrated upstream (94 %) | **Supported** (it is what makes the memorisation possible and the generalisation impossible) |
| Inadequate pressure representation (decoder capacity) | The same decoder fits the training pressure to 0.108 and the training inlet levels to correlation 0.99 | **Rejected** as the cause of the test error |
| Optimisation imbalance between velocity and pressure | At epoch 500, on 50 training meshes, the norm of the mean pressure-loss gradient is the largest (0.32, against 0.25 for v and 0.10 for u) and nearly orthogonal to the velocity gradients (cosines 0.11 and 0.03); pressure is not starved or opposed at the end of training | **Not supported at the final checkpoint**; earlier epochs **BLOCKED** (no weights) |
| Pressure normalisation or gauge | One constant explains 6 % of the squared error; the outlet level is right (RMS 0.001); the normalisation statistics match the checkpoint fingerprint | **Rejected** |
| Train/test geometry shift | Test inlet levels, lubrication estimates, inlet hop distances and obstacle sizes lie within the training range; two-sample KS p = 0.22, 0.41, 0.85, 0.11; no test mesh above the training maximum or 99th percentile | **Rejected** |
| Pressure errors follow velocity errors | Per-mesh correlation of velocity and pressure error on the test meshes: −0.28 (full), −0.30 (local) | **Rejected**: the direct pressure error is not inherited from the velocity |

## 6. Pressure reconstruction from the predicted velocity (COMPLETED)

**Problem.** `WeakDiagnostics.recover_pressure` takes the vertex velocity $(u, v)$, represents it exactly in the Taylor-Hood P2 space (a P1 field is a P2 field with mid-edge means), and finds the P1 pressure that minimises the Taylor-Hood momentum residual $\|A\mathbf u - B^{\top}p\|$ in the dual $H^1$ norm, by the saddle system $[[G, B^{\top}],[B, 0]]$ on the velocity degrees of freedom that the inlet and no-slip conditions leave free. The outlet rows of $B^{\top}$ carry the natural outlet condition and make the level unique; there is no separate gauge. **Inputs**: mesh, boundary markers, $\nu$, the prescribed inlet profile, and the predicted velocity. **No reference field enters** (checked in the code; the floor row is the only place the reference velocity is used, and it is labelled as such).

**Why it works.** It does exactly the global integration the MeshGraphNet cannot: the pressure at the inlet is obtained from $\nu\Delta\mathbf u$ everywhere between the outlet and the inlet, so the obstacle's resistance, encoded in the predicted velocity around it, sets the upstream level.

Means over the ten fine-tuned meshes of the full study (`pressure_diag_full_e500_finetuned.json`):

| Pressure | Relative $L^2$ | Gauge-free | Gradient error | Upstream share | Inlet slope, correlation |
|---|---|---|---|---|---|
| MeshGraphNet | 0.338 | 0.324 | 5.15 | 0.92 | 0.19, 0.57 |
| Data-only refit of it | 0.336 | 0.323 | 4.94 | 0.92 | 0.18, 0.55 |
| **Recovered from the MeshGraphNet velocity** | **0.091** | 0.080 | 1.08 | 0.67 | **0.66, 0.98** |
| Physics-informed fit | 0.139 | 0.101 | 0.87 | 0.61 | 0.47, 0.96 |
| Recovered from the physics-informed velocity | 0.121 | 0.101 | 0.57 | 0.72 | 0.50, 0.96 |
| Floor: recovered from the reference velocity | 0.023 | 0.020 | 0.49 | 0.65 | 0.96, 1.00 |
| Exact (reference) | 0 | 0 | 0 | | 1, 1 |

On the 100 test meshes the recovered pressure has relative error 0.074 (inlet slope 0.68, correlation 0.96); its error is correlated with the velocity error (0.63), unlike the network's own pressure (−0.28).

**What it corrects.** Both: the level (inlet correlation 0.57 to 0.98; the upstream share of the error falls from 0.92 to 0.67) and the spatial structure (gauge-free error 0.324 to 0.080; gradient error 5.15 to 1.08, against a floor of 0.49). The remaining error is mostly inherited from the velocity error.

**A link to the boundary problem (supported, not proven).** The physics-informed velocity is more accurate than the MeshGraphNet's (0.084 against 0.137), yet the pressure recovered from it is worse (0.121 against 0.091), with a lower inlet level (mean inlet error −0.107 against −0.059). This is the expected sign if the fit's no-slip violation (section 7) lets fluid slip past the obstacle and lowers its apparent resistance.

**Cost.** One recovery is one Taylor-Hood saddle solve, comparable in cost to solving the Stokes problem directly (0.31 s on one CPU thread), which is exact. Reconstruction shows where the information is; it does not make the surrogate computationally competitive with the solver.

## 7. Boundary deterioration under the physics-informed fit (COMPLETED)

`boundary_diag_full.json`, `boundary_diag_local.json`. Polygon corners: polygon boundary nodes where the boundary turns by more than 30°; "near": within three median polygon edge lengths (about 70 % of the polygon nodes, since the obstacles are polygons with many short sides). The thresholds were fixed in the script before it ran.

| No-slip RMS on the obstacle (mean over meshes) | Near corners | Elsewhere | Share of squared error near corners |
|---|---|---|---|
| Full MeshGraphNet | 0.0047 | 0.0045 | 0.73 |
| Full, data-only control | 0.0081 | 0.0052 | 0.83 |
| **Full, physics-informed fit** | **0.054** | **0.020** | **0.94** |
| Local, data + boundary penalties (no PDE) | 0.0015 | 0.0010 | 0.82 |
| Local, physics-informed fit | 0.053 | 0.027 | 0.89 |
| Local, physics fit + outlet condition | 0.054 | 0.020 | 0.91 |
| Local, outlet condition without pressure data | 0.065 | 0.020 | 0.93 |
| Local, hard constraints (`hard_bc`, meshes 0 and 1) | 0 | 0 | |

* **Competition with the PDE terms: supported.** With the same boundary penalties and no PDE terms the no-slip error is 0.0015 near corners; every variant with the strong-form PDE terms ends at 0.05 to 0.065 there. At the last iteration of the full study the weighted no-slip term (0.0029) is larger than the weighted PDE terms (0.0020) and the inlet term (0.00002): no-slip is the constraint the optimiser leaves unresolved.
* **Corner singularities: supported.** The deterioration is about 11 times near corners and 4 times along edges; 94 % of the squared error is near corners. The exact Stokes solution has singular velocity gradients and pressure at reentrant corners, where the strong-form residual cannot be satisfied by a smooth network and trades against the boundary values.
* **Inadequate boundary sampling: not supported as stated before.** Only 5 % of interior collocation nodes lie within three edge lengths of the obstacle (33 polygon nodes against about 2,940 interior nodes per mesh); the earlier explanation that local mesh refinement makes the PDE terms dominate near the obstacle (VALIDATION section 4.3, D) is weakened.
* **Hard constraints** remove the boundary error by construction but, on the two local meshes where they ran, kept the velocity gain on one mesh only (VALIDATION section 4.4). They do not resolve the corner conflict; they move it into the interior next to the boundary.
* **Boundary-loss scaling, representation and optimisation dynamics**: not separated by the existing runs (one weight setting, one network, one optimiser schedule).

## 7a. Statistical reliability

All uncertainty reported so far is **across test geometries**, for one trained network per budget: bootstrap intervals and sign tests over 10 or 100 meshes describe how a fixed model varies between geometries, not how results vary between independent training runs. The 100 test meshes are not 100 seeds. The only seed-level evidence is the local architecture comparison (3 seeds per arm; baseline pressure error range 0.334 to 0.337).

**Proposed multi-seed design.** Three seeds (0, 1, 2) of the full schedule with checkpoint selection (`configs/full_select.yaml`), training and prediction only; evaluation by `pressure_diagnostics.py` on the train, validation and the same 100 test meshes for each of the four candidate checkpoints. Report each metric as the mean over seeds with the seed range, and treat a difference between selection rules as real only if it exceeds the seed range (the rule of the architecture comparison). Cost from the measured T4 rate (103 ms per iteration, 7.16 h per run): about 7.4 h of T4 per seed including set-up, 22 h for three seeds; three sessions of the validated pause-and-continue mechanism or three parallel kernels, against a weekly quota of about 30 h.

## 8. Proposed experiments, ranked (PROPOSED)

One change per experiment; matched split, seeds, budget and evaluation (`scripts/pressure_diagnostics.py` on train, validation and the same 100 test meshes).

| Rank | Experiment | Question it isolates | Cost (T4, measured rates) | Expected value |
|---|---|---|---|---|
| 1 | **Validation-based checkpoint selection** (`configs/full_select.yaml`; criteria fixed in `src/stokes_ft/selection.py`), seed 0, training and prediction | Does stopping before memorisation (validation pressure minimum near epoch 68) lower the test pressure error, and what does it cost in velocity? Training is unchanged, so the only difference from the existing run is which weights are evaluated | one session, about 7.4 h | **High**: directly tests the overfitting mechanism of section 4; cheapest; the existing run is its matched control |
| 2 | Global latent at the full budget, with the same selection | Does global information let the level generalise instead of being memorised? | about 9.7 h (1.33 times the baseline epoch time measured locally); two sessions | High, if rank 1 confirms memorisation; compare against rank 1, not against the existing run |
| 3 | Multi-seed replication of rank 1 (section 7a) | Is the selection effect larger than seed variation? | about 22 h | Required before any claim from rank 1 is generalised |
| 4 | Physics-informed fit with PDE collocation excluded within three edge lengths of polygon corners (on the existing full MeshGraphNet) | Is the no-slip deterioration caused by the corner singularity? | 10 meshes × about 5.3 min = 0.9 h on a T4, or about 1 h on the local MPS | Medium; cheap; one change; directly tests section 7 |
| 5 | Pressure-gradient supervision (loss on $\nabla p$ through the mesh operator) | Does it remove the node-scale noise (gradient error 5.3)? It does not address the level | about 7.4 h | Medium-low |
| 6 | Pressure-loss weighting or a separate pressure decoder | Capacity or balance | about 7.4 h each | **Low**: section 5 rejects capacity and finds no imbalance at the end of training |

Not proposed as an experiment because it is already measured: the hybrid of MeshGraphNet velocity and one finite-element pressure solve (section 6).

## 9. Recommended next experiment

**Rank 1**: the full schedule with checkpoint selection, seed 0, on one Tesla T4:

```text
kaggle/run_cuda.ipynb with REF set to the commit containing configs/full_select.yaml, and step D running
python scripts/run_study.py --config full_select --resume --stages train,predict --max-hours <budget>
```

then, on the CPU, for each of `checkpoints/selected/{velocity,pressure,joint}` and the final epoch, `scripts/pressure_diagnostics.py` on train:500, validation:10 and test:100. The notebook currently runs `--config full`; running rank 1 needs a parameter for the configuration name and an approved push, so it is **not started**.

Predictions fixed now, before it runs:

* if memorisation drives the test pressure error, the pressure-selected checkpoint has a lower test pressure error than the final one **and** a smaller train/test pressure ratio than 3.4;
* if the receptive field is the binding limit, its test inlet-level correlation stays below 0.5 whatever the selection;
* the velocity-selected checkpoint has a test velocity error within 10 % of the final one's (0.121).

## 10. Remaining uncertainties

* One training run per budget; no statement here is about variation between training runs.
* Pre-epoch-450 behaviour (when memorisation sets in, whether the gradient balance changed) is inferred from the logged validation curve only; no weights exist.
* The memorisation mechanism is established by exclusion (exact zero receptive-field response plus near-perfect training-set levels); which local mesh features carry the signature was not identified.
* The link between the no-slip violation and the lower recovered pressure (section 6) is a supported hypothesis.
