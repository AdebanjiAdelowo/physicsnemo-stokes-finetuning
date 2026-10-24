# Scientific audit

Independent audit of the formulation, implementation, evaluation and claims of this repository. Date: 2026-10-24. Repository state audited: commit `56b536a` (`main`, equal to `origin/main`). Upstream reference: NVIDIA PhysicsNeMo at `b45a5c810c741e6b41f8515be24c51121f8fc21f` (see [`UPSTREAM_COMPARISON.md`](UPSTREAM_COMPARISON.md)). Open experiments are listed in [`EXPERIMENT_GAPS.md`](EXPERIMENT_GAPS.md).

New numbers in this document come from `scripts/audit_checks.py`, whose output is [`results/audit_checks.json`](../results/audit_checks.json). The script recomputes from the stored local run directory (`runs/local`, untracked: 10 prediction files, the epoch-40 MeshGraphNet checkpoint, and the two fine-tuned networks). No historical result file was changed.

## Summary

| Question | Finding |
|---|---|
| Is the PDE right? | Yes. `grad p - nu lap u = 0`, `div u = 0`, `nu = 0.01`, no forcing, in the units of the data |
| Are the boundary conditions right? | Yes, and now identified exactly: a Taylor-Hood P2-P1 solve with the parabolic inlet (`U = 0.3`), no slip, and the outlet condition `nu du/dn - p n = 0` reproduces the FEniCS reference to a relative error between 1.6e-13 and 3.2e-10 on all 10 test meshes. The symmetric-stress outlet form is ruled out (1.0 % to 2.4 % error) |
| Pressure gauge | Fixed in the reference by the natural outlet condition (p is about 0 there). Not fixed by any term of the physics-informed loss; the pressure data term sets it |
| Does the code match NVIDIA's? | Yes. The physics-informed step differs from upstream only in argument plumbing and logging; the MeshGraphNet, graph, loss and schedule are upstream's |
| Implementation errors found | None that change a reported number. One failing test (stale launcher-order assertion) was fixed |
| Leakage | None found. Splits are disjoint by simulation number; normalisation statistics come from training files only; no polygon geometry repeats among the 42 files whose content is on the disk (see section 7) |
| Recomputed metrics | Match the stored `per_sample.csv` (boundary and error metrics recomputed with independent code) |
| Main new result | The 15-layer MeshGraphNet cannot see the obstacle from the inlet (18 to 23 hops away). It predicts the same inlet pressure, 0.694 to 0.696, for every test geometry, while the reference ranges from 0.41 to 1.60. This is the main source of its pressure error, and the physics-informed step does not remove it |
| Missing baseline | The classical solver: 0.56 s per mesh on one CPU core for the exact reference, against 1,690 s per mesh for the physics-informed step at a velocity error of 0.11 |

## 1. Mathematical formulation

Steady incompressible Stokes flow in `[0, 1.5] x [0, 0.4]` minus one polygon:

```
-nu lap u + grad p = 0,   div u = 0         in Omega,   nu = 0.01
u = (4 U y (0.4 - y) / 0.16, 0),  U = 0.3     at x = 0
u = 0                                         on y = 0, y = 0.4 and the polygon
nu du/dn - p n = 0                            at x = 1.5
```

Checks:

* **Equation and sign.** `Stokes(PDE)` in `finetune.py` gives `momentum_x = p_x - nu (u_xx + u_yy)` and `continuity = u_x + v_y`, verified symbolically (`tests/test_physics.py`) and against a manufactured field by automatic differentiation (`results/physics_checks.json`, maximum deviation 3e-7).
* **Viscosity.** Fitted from the P1 weak form of the reference: 0.01005 (range 0.00996 to 0.01021). The exact Taylor-Hood reproduction with `nu = 0.01` (section 2) settles it.
* **Forcing.** None, in both the data and the code.
* **Normalisation.** Coordinates are not normalised in either stage, so autodiff derivatives are in physical units. MeshGraphNet targets are standardised with training statistics and de-standardised before the physics-informed step, which works in physical units. Residual weights therefore act on physical quantities whose scales differ: inlet velocity 0.3, inlet pressure about 0.4 to 1.6, momentum residual of order 0.1 to 1.
* **Units.** None stated by the dataset. The README now says this.

## 2. Identification of the reference discretisation

`src/stokes_ft/fem.py` solves the problem on the dataset triangles with P2 velocity and P1 pressure (scikit-fem), using one of two natural outlet forms.

| Outlet form | Relative L2 to the reference, (u, v, p) stacked, 10 test meshes | Time per mesh |
|---|---|---|
| `nu grad u : grad w` (`nu du/dn - p n = 0`) | mean 3.3e-11, max 3.2e-10 | 0.50 to 0.65 s |
| `2 nu eps(u) : eps(w)` (`2 nu eps(u) n - p n = 0`) | 0.010 to 0.024 | 0.47 to 0.78 s |

The reference fields are therefore the **vertex values of a Taylor-Hood P2-P1 solution** with the do-nothing outflow condition, solved to near machine precision. `tests/test_audit.py` asserts the distinguishing property: Poiseuille flow is reproduced exactly with the gradient form and not with the symmetric form.

Consequences:

1. The README's problem statement is verified rather than inferred.
2. **The residual floor is explained.** The reference velocity is quadratic on each triangle, but the files store only its vertex values. Any operator that sees only vertex values gives the exact discrete solution a nonzero residual. The least-squares operator gives a momentum residual of 0.30. An independent P1 element divergence (`audit.p1_divergence_rms`) gives the reference 0.17 to 0.57 over the 10 test meshes. A field can therefore score below the reference on these metrics without being closer to a Stokes solution. On meshes 0 and 1 the fine-tuned field has a P1 divergence of 0.25 and 0.21, against 0.44 and 0.40 for the reference (data-driven: 0.76 and 0.93).

## 3. MeshGraphNet stage

Graph construction, inputs, targets, model and schedule are upstream's; the details are in [`UPSTREAM_COMPARISON.md`](UPSTREAM_COMPARISON.md). Points of scientific relevance:

* **Graph.** The mesh edges are bidirected. Node inputs are position and a one-hot boundary marker. Edge inputs are the relative displacement and its length.
* **Message passing.** 15 processor blocks, each one hop, with sum aggregation. The prediction at a node is a function of the inputs within 15 hops only.
* **Target and loss.** MSE of the standardised `(u, v, p)` at all nodes. Boundary values are not imposed; the boundary marker is an input.
* **Validation metric during training.** Relative error of the standardised fields, as upstream. Test errors are reported in physical units.

### 3.1 Receptive field against the domain

| Quantity (10 test meshes) | Value |
|---|---|
| Message-passing layers | 15 |
| Hops from the polygon to the nearest inlet node | 18 to 23 |
| Hops from the polygon to the nearest outlet node | 61 to 70 |
| Share of nodes farther than 15 hops from the polygon | 60 % to 69 % |
| Change of the trained model's output beyond 15 hops when the polygon-node inputs are perturbed | exactly 0.0 on all 10 meshes |
| Same change within 15 hops (standardised units) | 3.2 to 3.7 |

In Stokes flow the pressure is coupled globally: the pressure level upstream of the obstacle equals the outlet value plus the pressure drop across the whole channel, and that drop depends on the obstacle. The inlet nodes cannot see the obstacle, so the model can only predict an average level:

| | Reference | Data-driven |
|---|---|---|
| Mean inlet pressure, 10 test meshes | 0.41 to 1.60 (std 0.349) | 0.694 to 0.696 (std 0.00056) |
| RMS error of the inlet level | | 0.334 |

Downstream of the obstacle the prediction is accurate. On mesh 0, the mean pressure for `x > 0.8` is 0.0537 against 0.0527, and the outlet RMS is 0.006. The pressure error is a level error **upstream** of the polygon. A domain-wide constant explains only 26 % (mesh 0) and 5 % (mesh 1) of the squared pressure error, so this is not a gauge offset in the usual sense.

This limitation is architectural and would persist under NVIDIA's full training schedule. It is a known weakness of single-scale message passing for elliptic problems; multiscale graph networks (for example MultiScale MeshGraphNets, Fortunato et al., 2022) are one response. The regression test `test_meshgraphnet_output_is_blind_beyond_its_message_passing_depth` pins the property.

## 4. What "physics-informed fine-tuning" means here

It is **not** continued training of the MeshGraphNet. For each test mesh separately, the step:

1. reads the frozen MeshGraphNet prediction `(u_G, v_G, p_G)` at the mesh nodes;
2. initialises a new coordinate network `(x, y) -> (u, v, p)` (64 random Fourier features with scale 10, three hidden layers of width 128, 49,923 parameters);
3. minimises, over 10,000 full-batch Adam steps at a learning rate of 1e-3,

```
mse(u - u_G) + mse(v - v_G) + mse(p - p_G)                 data, all nodes
+ 10 [mse_in(u - u_in) + mse_in(v)]                         inlet penalty
+ 10 [mse_0(u) + mse_0(v)]                                   no-slip penalty
+ mse(r_x) + mse(r_y) + 10 mse(r_c)                          strong-form residual, all nodes, autodiff
```

4. returns the network's values at the nodes.

The reference never enters the loss (verified in code). The step is a per-geometry, test-time regularisation of one prediction by a PINN. It is not a surrogate, and it shares no parameters across meshes.

## 5. Consistency of the physics-informed objective with the mesh and the reference

| Aspect | Assessment |
|---|---|
| Strong form at vertices against P2-P1 reference data | Consistent as a continuous objective: the exact solution makes the residual terms zero. It is inconsistent with the reference **as a discrete object**. The residual is enforced pointwise on a smooth network, while the reference satisfies the Galerkin equations of another space. Agreement in residual therefore does not imply agreement with the reference |
| Residual at boundary nodes | The PDE terms include wall, inlet and polygon nodes. At polygon corners that are re-entrant as seen from the fluid (every corner, if the polygon is convex), the exact pressure and velocity gradients are singular, so the strong-form residual of the exact solution is undefined there. The network is penalised for not matching a field that has no pointwise residual at those nodes |
| Weighting by nodes, not by area | All `mse` terms are node means. Node areas vary by a factor of 15 (mesh 0), so the loss concentrates on the refined region around the polygon. Mesh-weighted and nodal velocity errors differ by at most 1.6 % relative, so this does not change any reported conclusion |
| Pressure | The momentum equation constrains only `grad p`. In the reference the outlet condition fixes the level; in the loss nothing does except `mse(p - p_G)`. Because the data-driven upstream level is wrong (section 3.1), the data term anchors the fine-tuned pressure to the wrong level. The fine-tuned upstream level moves by 0.017 (mesh 0) and -0.009 (mesh 1), against errors of -0.19 and -0.06 |
| Boundary conditions | Soft penalties of weight 10. The MeshGraphNet's no-slip error (0.003) becomes 0.016 after the step; the inlet error falls from 0.006 to 0.001 |
| Viscosity, profile, geometry constants | Agree with the data (0.4 channel height, `U = 0.3`, `nu = 0.01`) |

**Correction to the project's explanation of the pressure result.** The README attributed the unchanged pressure to the missing outlet term. The outlet pressure of the data-driven prediction is already within 0.006 of the reference, so an outlet term alone would supply little new information. What prevents the fix is the pressure data term at weight 1, combined with a data-driven upstream level that is structurally uninformed. The README was edited accordingly. The outlet-term experiment remains a useful test (gap G5), and the prediction above makes it falsifiable.

## 6. Recomputed metrics and fair comparison

### 6.1 Recomputation

Errors, boundary residuals and the outlet pressure of `results/local/per_sample.csv` were recomputed from the `.vtp` files with independent NumPy code and agree to all printed digits. Example, mesh 0: velocity relative L2 is 0.1599 for the data-driven field and 0.1197 for the fine-tuned field; no-slip RMS is 0.00299 and 0.01651.

### 6.2 How representative are meshes 0 and 1?

Data-driven errors on all 10 configured test meshes (only 0 and 1 were fine-tuned):

| Metric | Meshes 0 and 1 | All 10 test meshes, mean (range) |
|---|---|---|
| Velocity relative L2 | 0.188 | 0.233 (0.100 to 0.381) |
| `v` relative L2 | 0.535 | 0.665 (0.417 to 0.923) |
| Pressure relative L2 | 0.159 | 0.334 (0.057 to 0.705) |

Meshes 0 and 1 are easier than average, especially for pressure. Test mesh 7, the single mesh of the upstream example, has the largest data-driven pressure error (0.705).

### 6.3 Comparison on one protocol

Same two meshes, same nodes, errors against the reference:

| Method | Velocity rel. L2 | Pressure rel. L2 | No-slip RMS | Cost per new geometry |
|---|---|---|---|---|
| MeshGraphNet (local config) | 0.188 | 0.159 | 0.0028 | 0.19 s inference (after 2,308 s of training) |
| MeshGraphNet + physics-informed step | 0.109 | 0.154 | 0.0163 | 1,690 s per mesh (MPS, shared machine) |
| Taylor-Hood P2-P1 solve on the same mesh | below 1e-12 | below 1e-12 | 0 (imposed) | 0.56 s (CPU, one direct solve) |

The physics-informed step needs exactly the information that the classical solver needs: the mesh, the PDE, the viscosity and the boundary data. On this problem it is about 3,000 times slower than the solver, and the solver's result is the reference. The physics-informed step is therefore not competitive as a solver here. Its value has to be argued as a correction mechanism for settings without a cheap solver, and this toy problem cannot test that setting.

The comparison also lacks a control that separates the PDE terms from the refitting itself (gap G4).

## 7. Data, leakage, reproducibility and provenance

* **Split.** 800/100/100, seeded (`split_seed = 0`), recorded in `data/dataset/split.json`, disjoint by construction and re-verified on every run. The local study uses the first 100 training, 10 validation and 10 test files in numeric order.
* **Normalisation.** Statistics from the training split only, written to and read from the run directory (`node_stats.json`, `edge_stats.json`).
* **Geometry duplicates.** On this machine 958 of the 1000 raw files have been evicted by the cloud file provider (`SF_DATALESS`), and reading them blocks, so the scan covered the 42 resident files (30 train, 2 validation, 10 test). These hold 42 distinct polygons. A full scan is one command once the files are local (`python scripts/audit_checks.py`).
* **Reference solver as leakage.** The physics-informed step uses only the boundary data and the PDE. The reference enters only the monitoring printout, as upstream.
* **Seeds.** One seed per stage; sensitivity not measured (gap G3).
* **Provenance.** Every run records the commit, dirty state, PhysicsNeMo commit, package versions, resolved configuration, dataset checksum and simulation numbers. The local study was run at commit `910170b`/`f18a24c` (`results/local/study_metadata.json`). The installed PhysicsNeMo is the pinned commit (`direct_url.json`).
* **Environment.** The project uses a pip virtual environment from `requirements.txt` without a lockfile. `scikit-fem==11.0.0` was added for the audit solver.

## 8. Tests and checks run

| Check | Result |
|---|---|
| `python -m pytest -q` before the audit | 41 passed, 1 failed (`test_order_of_steps`) |
| Cause of the failure | Commit `1d57b7a` added probe packaging (`results.py package runs/{study}`) before the full run; the test located the first occurrence of `results.py package`. The launcher was correct |
| `python -m pytest -q` after the audit | 47 passed (including 5 new tests in `tests/test_audit.py`) |
| `python scripts/audit_checks.py` | `results/audit_checks.json` |
| Fine-tuning timing probe (100 iterations, mesh 2) | CPU 0.54 to 0.76 s per iteration, MPS 0.22 s per iteration |

## 9. Claims

| Claim (README or portfolio) | Verdict |
|---|---|
| Physics-informed fine-tuning reduces velocity error and interior residuals | **Supported for 2 meshes of an undertrained model**. Not established for the NVIDIA schedule or the other 8 meshes |
| "The fine-tuned field ends closer to the reference than the prediction it was fitted to" | Supported (both meshes) |
| Residuals fall "to the resolution of the metric" | Supported; the floor is now explained by the P2-P1 reference (section 2) |
| Pressure changes little because no outlet term sets the level | **Corrected**: the data-driven outlet pressure is already right; the upstream level is wrong because of the receptive field, and the data term preserves it |
| The Stokes problem, boundary conditions and `nu` as stated | Verified exactly |
| The equations and constants in NVIDIA's code agree with the data | Verified |
| Any statement that physics-informed fine-tuning makes the surrogate "physically consistent" or a better solver | **Unsupported**. No-slip worsens, pressure is unchanged, and a direct solve is exact and about 3,000 times faster |
| Population-level effect sizes | **Unsupported** with n = 2 |

## 10. Changes made in this audit

| File | Change |
|---|---|
| `tests/test_notebook.py` | The order test now looks for the full-study packaging steps (`results.py package runs/full`, `results.py bundle runs/full`) |
| `src/stokes_ft/fem.py` | New: Taylor-Hood P2-P1 solver with both outlet forms |
| `src/stokes_ft/audit.py` | New: lumped-mass weights, mesh-weighted error, P1 divergence, pressure-offset split, hop distance |
| `scripts/audit_checks.py` | New: the checks of sections 2, 3.1, 6 and 7, written to `results/audit_checks.json` |
| `tests/test_audit.py` | New regression tests: Taylor-Hood exactness and outlet-form discrimination on Poiseuille flow; lumped mass; P1 divergence; offset split; zero dependence of the MeshGraphNet beyond its message-passing depth |
| `requirements.txt` | `scikit-fem==11.0.0` |
| `README.md`, `results/README.md` | Verified discretisation, classical baseline, receptive-field result, refined pressure explanation, new files |
