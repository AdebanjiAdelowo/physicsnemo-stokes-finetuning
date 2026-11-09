# Full-budget global-latent MeshGraphNet experiment

Pre-registered on 2026-11-09 before the feasibility probe or any new held-out evaluation.

## Matched comparison

Compare the original 15-block MeshGraphNet with the existing global-latent variant. The variant adds a shared mean-pooled node MLP after each processor block. Its final linear layer starts at zero, so it initially reproduces the baseline given identical shared weights. The treatment includes 33,024 additional parameters (1.30%). The comparison estimates the effect of adding this global path as implemented, including its small increase in capacity.

Run both arms with model seeds 0, 1 and 2 on the same single-T4 setup and pinned repository/dependency versions. Use the original split seed 0 and NGC archive digest. Use the first 500 training meshes, first 10 validation meshes and all 100 held-out meshes in simulation-number order. Each matched seed uses the same shared weight initialisation and per-epoch shuffle order across arms. Use 500 epochs, batch size one, Adam with the existing learning-rate decay, identical input features and training-set normalisation, and mean squared error on normalised velocity and pressure. Record versions, checkpoint fingerprints, per-session wall time and peak allocated GPU memory, cumulative update time, parameters, and synchronized warmed inference latency on the same fixed validation graphs. Report the maximum across sessions for memory. Training and recovery sessions use the train stage only.

Retain the pressure, velocity and joint validation minima (first epoch on ties) and epoch 500. The scores are validation pressure relative L2, mean of the two velocity-component relative L2 errors, and the equal-weight mean of those two scores. Freeze the six run manifests and all selected epoch IDs before computing any held-out result. Verify the selected scores against the saved histories and retain the per-mesh diagnostic rows keyed by simulation number. Record any resumes and use the existing state archive and fingerprint checks.

## Endpoints and decision

The primary comparison pairs the **pressure-selected** global and baseline checkpoints within each seed. Pressure and velocity use the arithmetic mean of per-mesh nodal relative L2 errors in physical units. Fit the inlet slope by OLS with an intercept across the 100 geometry-level inlet means. The ideal slope is 1. Compute inlet RMS across the same 100 predicted minus reference inlet means.

Declare success only if all six runs complete and **all** of the following hold:

1. Mean held-out pressure error decreases by at least 5% relative to the matched baseline mean across the three seeds.
2. Mean inlet RMS error decreases by at least 5% relative to the matched baseline mean across the three seeds.
3. Each global-latent seed has a slope in [0.5, 1.5], and mean absolute slope error from 1 decreases relative to the baseline.
4. Every matched seed improves in held-out pressure error, inlet RMS error and absolute slope error from 1.

If all six runs complete and any criterion fails, report that the joint criterion was not met and identify which endpoints improved. An incomplete run matrix leaves the primary comparison unresolved. Inlet correlation is secondary and cannot satisfy the slope criterion.

Report inlet correlation, velocity relative L2, FEM pressure-gradient and gauge-free pressure error, weak momentum and continuity residuals, parameters, time and memory as secondary endpoints. Also report training-set pressure error and the absolute test-minus-train gap. Evaluate the velocity-selected, joint-selected and final checkpoints on the same meshes and with the same metrics as descriptive comparisons. Compare inlet and spatial pressure diagnostics to distinguish level calibration from changes in spatial structure. Treat historical seed-0 results (pressure 0.2694 at epoch 68, velocity 0.1106 at epoch 135) as references outside the matched comparison.

## Statistical reporting

Report the arm values and signed global-minus-baseline effects for each seed, then the mean, range and descriptive 95% paired Student t interval for the three seed effects (two degrees of freedom). The interval is sensitive to its assumptions at this sample size. Separately compute 10,000 paired whole-mesh bootstrap replicates with seed 0, drawing the same test simulation IDs across both arms and all seeds in each replicate. Recompute slope and RMS inside each replicate before averaging seed effects. These geometry intervals condition on the three trained pairs. Training/test gap intervals resample the two splits separately while keeping the matched model pairing. Use pointwise intervals. Meshes and spatial nodes do not contribute independent training replicates. The joint practical criterion is descriptive and supports a conclusion for the three studied seeds.

## Approval gate

Before scheduling any full run, execute the bounded CUDA probe on both arms at production width and verify the dataset checksum, finite training, checkpoint selection, archive recovery and device measurements. The probe uses training and validation meshes only. Record its measured cost and project the six-run budget. A full-budget launch requires separate explicit authorization.
