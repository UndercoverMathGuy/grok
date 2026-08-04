#!/bin/bash
# Phase-2 farm: selection pressure on top of the flat (orthogonal W_E) start.
# Arm N: SAM-lite weight noise sigma=0.3 annealed to 0 at epoch 6000.
# Arm T: tilted ERM t=5.
# Cell 2034, init seeds paired 1:1 with the phase-1 orthWE runs.
set -e
cd "$(dirname "$0")/.."

run_one() {
  arm=$1; is=$2; shift 2
  d="runs/phase2-$arm/p-113/seed2034/seed$is"
  if [ -f "$d/metrics.json" ]; then echo "skip $d (done)"; return; fi
  echo "=== phase2-$arm init $is ($(date +%H:%M:%S)) ==="
  uv run python -m grok.train --run-name "phase2-$arm/p-113/seed2034/seed$is" \
    --p 113 --data-seed 2034 --init-seed "$is" --embed-init orthogonal \
    --num-epochs 20000 --save-every 1000 --spectra-every 100 "$@"
}

for is in 11285 33428 777 4242; do
  run_one noise $is --grad-noise 0.3 --grad-noise-until 6000
done
for is in 11285 33428 777 4242; do
  run_one tilt $is --loss-tilt 5.0
done
echo PHASE2-DONE
