#!/usr/bin/env bash
# Accelerator queue of the November 2026 validation study, in priority order.
# Every step acquires the machine-wide accelerator lock itself (stokes_ft.gpulock: an flock that
# waits, never forces, and is released by the kernel when the process exits, also on a crash).
# The chain stops at the first failure so that a failed step is inspected before anything else runs.
# Each step is resumable: rerunning this script skips completed work.
#   bash scripts/mps_queue.sh            # all steps
#   bash scripts/mps_queue.sh g1         # one group: g1, arch, variants
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
LOGS=../runs/queue_logs
mkdir -p "$LOGS"
step() { local name=$1; shift; echo "=== $(date '+%F %T') start $name"; "$PY" -u "$@" > "$LOGS/$name.log" 2>&1; echo "=== $(date '+%F %T') done $name"; }

group=${1:-all}
if [[ $group == all || $group == g1 ]]; then
  # G1: physics-informed stage on test meshes 2 to 9; meshes 0 and 1, training and predictions are skipped
  step g1 run_study.py --config local --resume
fi
if [[ $group == all || $group == arch ]]; then
  # G12: three arms x three seeds, seeds interleaved so that partial results stay balanced
  for seed in 0 1 2; do
    for arm in baseline global_latent shared_depth; do
      step "arch_${arm}_s${seed}" run_study.py --config arch --stages train,predict --resume \
        model_variant=$arm seed=$seed name=arch_${arm}_s${seed}
    done
  done
fi
if [[ $group == all || $group == variants ]]; then
  # G5, G6 on meshes 0 and 1, on MPS like the stored upstream fits of those meshes
  for v in hard_bc outlet_free_p outlet; do
    step "variant_$v" run_controls.py --config local --variant $v --samples 0,1 --device mps
  done
fi
echo "=== $(date '+%F %T') queue finished"
