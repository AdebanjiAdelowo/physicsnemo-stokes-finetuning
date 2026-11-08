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
