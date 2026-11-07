# Running the full study across Kaggle sessions

A Kaggle GPU session has a time limit, and its files survive only if the session ends normally: what is in `/kaggle/working` then becomes the output of that notebook version. A later session can read that output, read-only, by attaching the version as a kernel source (it appears under `/kaggle/input`). The study uses this as its only persistence mechanism; a checkpoint that exists only inside an interrupted session is lost with it.

## What is saved

Every checkpoint of the data-driven stage (`src/stokes_ft/checkpoint.py`) consists of PhysicsNeMo's model and training-state files (model, Adam state, learning-rate scheduler, AMP scaler, epoch), the Python, NumPy, torch and CUDA random-number-generator states, the per-epoch history up to that epoch, and a manifest with the SHA-256 of each file and a fingerprint of the run. Files are written to a temporary directory and moved into place; the manifest is moved last and is the commit point, so a checkpoint without a manifest is never loaded. The newest three checkpoints are kept (`training.checkpoint_every` is 25 epochs for the full study).

The fingerprint holds the SHA-256 of the resolved configuration (without local paths, device and run name), the seed, the simulation numbers of the training and validation files, the SHA-256 of the normalisation statistics, and the names and shapes of all model parameters.

Fine-tuning has no intermediate checkpoints: each test mesh takes about 5.5 min on a T4, and each finished mesh writes its record and network, which resumption skips.

## How a session ends

`scripts/run_study.py --max-hours H` pauses at the next safe point after `H` hours: after an epoch (writing a verified checkpoint of that epoch, whatever the schedule) or before a test mesh whose fine-tuning might not finish (1.5 times the duration of the previous mesh). `runs/<name>/status.json` records `paused` or `complete`. The launcher (`kaggle/run_cuda.ipynb`, step D) sets `H` to what is left of `SESSION_BUDGET_HOURS` minus 0.5 h, then always runs

```bash
python scripts/session_state.py pack runs/full /kaggle/working/state-full.tar
```

which verifies the newest checkpoint and writes the whole run directory to `state-full.tar`, with `state-full.tar.sha256` and a summary `state-full.tar.json`. This happens in a `finally` block, so a failure later in the session still leaves the last state in the output.

## How the next session continues

1. Create a new notebook version or kernel from `kaggle/run_cuda.ipynb` with the same `REF`, `RUN_FULL = True` and `RESUME = True`, and attach the previous version's output as a kernel source.
2. Step A2 finds exactly one `state-full.tar` under `/kaggle/input` and runs `scripts/session_state.py unpack`, which
   * checks the tar against its SHA-256 sidecar,
   * refuses to overwrite an existing `runs/full`,
   * extracts with tarfile's `data` filter (no absolute paths, links or path escapes),
   * verifies the checksums of every checkpoint manifest.
3. Step D runs `run_study.py --config full --resume`. The study refuses a configuration different from the one stored in `study_metadata.json`; training then rebuilds the datasets, recomputes the fingerprint and loads the newest checkpoint whose checksums verify, falling back to an older one if the newest is damaged, and **refusing** (`IncompatibleCheckpoint`) if the fingerprint differs. Completed stages and fine-tuned meshes are skipped.

A session killed by the platform before it ends normally leaves no output; the next session then continues from the state of the last session that did end normally.

## Verification

* CPU tests (`tests/test_checkpoint.py`, `tests/test_recovery_check.py`): a paused and resumed run matches an uninterrupted one; corrupted and incomplete checkpoints are skipped; a different configuration or different normalisation statistics are refused; RNG states round-trip; the tar round trip refuses tampering and overwriting; two simulated sessions that share only the packed state pass the recovery check.
* CUDA, two separate Kaggle executions (`kaggle/recovery_test.ipynb`, `scripts/recovery_check.py`): see [`VALIDATION.md`](VALIDATION.md), section 6, and `results/cuda_recovery_t4/`.
