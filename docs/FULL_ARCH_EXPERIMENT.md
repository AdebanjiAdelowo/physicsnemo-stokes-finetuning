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

## Feasibility preparation

The CPU audit used validation mesh 16, with the inlet at least 22 hops from the obstacle. Changing obstacle markers to interior markers while holding positions and edges fixed produced zero change in baseline inlet pressure and a maximum change of 0.00518 in the active global model, in normalised units. For this mechanism check, the final global weight was deliberately set nonzero. Training keeps its original zero initialization. Changing the graph targets left both outputs identical. Tests verify shared initialization for all three seeds, isolation of differently sized graphs in a batch, permutation equivariance of predictions, and verified archive-based pause and resume with validation minima on both architectures. [CPU audit record](../results/full_arch_feasibility/cpu_audit.json)

The [private Kaggle probe](../kaggle/full_arch_probe/full_arch_probe.ipynb) checks out commit `d91ed9d59c61bd1ca4732c4d93956b5dc0b7d282` and requires a clean checkout at that exact SHA. Its three epochs per arm use 16 training meshes and two validation meshes at production width. It pauses after epoch one, verifies a packed and restored checkpoint, then resumes and checks finite forward and backward training, the three selected epochs, final load, and nonzero learned global weights. It measures timed updates after epoch one, validation-only inference, peak allocation and wall time. The clone, data and checkpoint scratch space are outside the Kaggle output directory. The metadata has no attached sources. The CPU audit and tests are complete. CUDA timings, memory and the new per-seed endpoints are pending.

The probe passes if the pinned checkout and public dataset hash verify, both arms complete and resume from verified state, checkpoint minima match their histories, and both produce finite timed outputs. Before authorizing full jobs, review measured memory headroom and projected time including environment setup, validation, serialization and any continuation sessions. The probe uses small subsets and its update-rate extrapolation is approximate.

### Commands after authorization

The GitHub branch currently points to `6dd0e7f405ef68be80692ceb5dce1f21e0762bae`. Make the pinned probe commit reachable first. The following push advances the branch only through the pinned probe code and tests. The Kaggle command submits the private feasibility notebook with a one-hour timeout. [Kaggle CLI notebook commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md#kaggle-kernels-push)

```bash
probe_ref=d91ed9d59c61bd1ca4732c4d93956b5dc0b7d282
git push origin "$probe_ref:refs/heads/research/validation-2026-11"
remote_tip=$(git ls-remote --heads origin research/validation-2026-11 | cut -f1)
test "$remote_tip" = "$probe_ref" &&
  kaggle kernels push -p kaggle/full_arch_probe --accelerator NvidiaTeslaT4 -t 3600
kaggle kernels status adebanjiadelowo/stokes-global-latent-cuda-probe
kaggle kernels output adebanjiadelowo/stokes-global-latent-cuda-probe \
  --file-pattern '^full_arch_probe\.json$' -p runs/full_arch_feasibility
```

For the separately approved full phase, use one job per arm and seed. From a clean checkout with the accepted CUDA stack and verified public dataset, set `arm` to `baseline` or `global_latent`, and `seed` to 0, 1 or 2. Set the session hours from the confirmed platform limit with a margin for packaging. Retain each session's archive and sidecars. For a continuation, restore the previous verified archive into the fresh job first and repeat the same command with the same arm and seed.

```bash
run="full_arch_${arm}_s${seed}"
python scripts/prepare_data.py
# Continuation only:
# python scripts/session_state.py unpack "$previous_state" runs
python scripts/run_study.py --config full_arch --stages train --resume \
  --max-hours "$session_hours" "model_variant=$arm" "seed=$seed" "name=$run"
python scripts/session_state.py pack "runs/$run" "$output_dir/state-$run.tar"
```

After freezing all six completed run manifests, evaluate each checkpoint with the existing diagnostic script. Use `data_driven` and epoch 500 for the final candidate. Read each selected epoch from its corresponding `checkpoints/selected/<criterion>/selection.json` and use `selected/<criterion>` for `checkpoint_dir`.

```bash
python scripts/pressure_diagnostics.py --config full_arch --run "runs/$run" \
  --checkpoint "$checkpoint_dir" --epoch "$epoch" --splits test:100,validation:10,train:500 \
  --selection-evaluation --output-dir results/full_arch \
  "model_variant=$arm" "seed=$seed" "name=$run"
```

### Planning estimate

The historical T4 baseline used 7.35 hours of training updates and peaked at 1,153 MiB allocated. Applying the roughly 1.29× reduced-budget MPS update-time ratio gives an indicative 9.46 update-hours for one global run and about **50.4 T4 update-hours for six runs**, before setup, validation and recovery overhead. This transfers a ratio across hardware and training budgets. Replace it with the T4 probe estimates before scheduling. A global run may need more than one session under a nine-hour allocation.
