# Cross-session recovery test on CUDA (Kaggle, Tesla T4): FAILED its numerical criterion

Two separate private Kaggle executions of `kaggle/recovery_test.ipynb` at code commit `f90639531c129e03d03ec1c538d90766d66ff8a6` (clean tree, 109 tests passed in each), on 2026-11-08. Session two received session one's state only through session one's persisted output, attached as a kernel source (`/kaggle/input/notebooks/adebanjiadelowo/stokes-recovery-a-f906395/`). Configuration `configs/recovery_check.yaml`: the upstream architecture and optimiser, 50 training meshes, 6 epochs, checkpoint every 2 epochs, pause after epoch 3. These numbers verify continuation; they are not results.

## Outcome under the criteria fixed in advance (`src/stokes_ft/recovery_check.py`)

| Criterion | Result |
|---|---|
| State restored from the persisted output, tar and checkpoint checksums verified | yes (`state-recovery_*.tar.json`, `.sha256`) |
| Corruption drill: damaged newest checkpoint detected, older one used | yes: "checksum mismatch in checkpoint.0.3.pt", resumed from epoch 2 |
| Incompatibility drill: different learning rate refused | yes: "fingerprint differs in ['config_sha256']" |
| Resumed from the verified checkpoint of the pause epoch | yes, epoch 3 |
| Largest relative difference from the uninterrupted reference below 1e-2, or at most twice that of an uninterrupted repeat | **no: 2.31 % against a repeat spread of 0.76 %** |
| **Overall** | **FAILED** |

## Where the difference arises (recorded deviation)

Relative difference from the reference, largest over the four series (training loss, validation relative L2 of u, v, p):

| Epochs | Resumed run | Uninterrupted repeat |
|---|---|---|
| 1 to 3 (before the pause; both runs uninterrupted, same session as the reference) | **2.31 %** (validation u error, epoch 3) | 0.65 % |
| 4 to 6 (after the resume in session two) | 0.77 % | 0.76 % |

The 2.31 % arises at epoch 3, before the interruption, in a stretch where the "resumed" run was ordinary uninterrupted training in the same session as the reference: it is run-to-run variation of a single validation number under non-deterministic CUDA scatter-add, not an effect of recovery. After the resume the resumed run is as close to the reference as an independent repeat. The criterion was nevertheless not met as written: it included the epochs before the pause and estimated the run-to-run spread from one repeat. It is reported as a failure and was not changed after the fact.

## Retest with criterion version 2: also FAILED

Criterion version 2 was fixed and pushed (`e0c57f7`) before its first execution: only the epochs after the pause, run-to-run spread from three uninterrupted repeats, pass if the resumed run's largest relative difference from the reference is below 1e-2 or at most 1.5 times the largest repeat difference. Two further separate executions (`stokes-recovery2-a/b-e0c57f7`, 109 tests passed in each; `recovery_check_v2.json`). The mechanics passed again (restore from persisted output, corruption fallback to epoch 2, incompatible configuration refused, resume from the verified epoch-3 checkpoint).

| Epochs 4 to 6, largest relative difference from the reference | Value |
|---|---|
| Resumed run | **5.32 %** (validation u error, epoch 5) |
| Repeats 1, 2, 3 | 2.46 %, 1.46 %, 1.03 % (all also at the validation u error, epoch 5) |
| Threshold | 3.69 % |
| **Outcome** | **FAILED** |

Training loss after the resume differs from the reference by +0.33 %, +0.06 %, +0.01 % (epochs 4 to 6), inside the repeats' range (up to +0.45 %). The validation u error is the volatile series in every run: the two independent session-one references of the two tests differ from each other by 2.4 % at epoch 5. Whether the 5.32 % is a rare excursion of CUDA non-determinism or a small effect of resumption cannot be decided from these runs.

On the CPU with one thread, where training is deterministic, a run paused after an epoch and resumed is bit-identical to the uninterrupted run in every history value and every final weight (`tests/test_checkpoint.py::test_resumed_run_is_bit_identical_on_a_deterministic_device`). That establishes that the checkpoint restores everything the training trajectory depends on; it does not by itself settle the CUDA question, which a test with deterministic CUDA algorithms would. (With two CPU threads two identical runs already differ at about 1e-8 in the first epoch: multi-threaded reductions are not bit-reproducible either.)

## Deterministic retest with criterion version 3: PASSED

Criterion version 3 was fixed and pushed (`ee57d64`) before its first execution. Both sessions ran `scripts/recovery_check.py --deterministic` (deterministic CUDA algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic cuDNN) in two separate private Kaggle executions on Tesla T4s (`stokes-recovery3-a/b-ee57d64`, 2026-11-08; 111 tests passed in each; no unsupported-operation error). Session two received session one's state only through its persisted output (`/kaggle/input/notebooks/adebanjiadelowo/stokes-recovery3-a-ee57d64/`). Evidence: `recovery_check_v3_deterministic.json`, `v3_state/`, `v3_session_*_log_excerpt.txt`.

| Criterion version 3 | Result |
|---|---|
| Restored from the persisted output; tar and checkpoint checksums verified | yes (checkpoints of epochs 3 and 2) |
| Corruption drill | detected ("checksum mismatch in checkpoint.0.3.pt"), resumed from epoch 2 |
| Incompatibility drill | refused ("fingerprint differs in ['config_sha256']") |
| Resumed from the verified checkpoint of epoch 3 | yes |
| Resumed run bit-identical to the reference: every stored history value except wall-clock seconds, every final weight | **yes** |
| Three repeats bit-identical to the reference | yes (all three) |
| **Verdict** | **PASSED** |

Criteria versions 1 and 2, computed on the same runs, also pass (all differences 0). Together with the earlier non-deterministic results, this shows that the 2.31 % and 5.32 % differences of the first two tests came from CUDA non-determinism, not from the checkpoint: once non-determinism is removed, resumption across sessions reproduces the uninterrupted run exactly. The full study itself runs with the default (non-deterministic) algorithms of the upstream configuration; deterministic mode was used for this test only.
