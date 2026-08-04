#!/bin/bash
# Follow-up to the noise-arm wash: sustained pressure. sigma=0.2 annealed to 0
# at epoch 16000 (past committee consolidation, ~5-17k) instead of 6000.
# Tests whether the null was a schedule artifact. 4 paired seeds, cell 2034.
set -e
cd "$(dirname "$0")/.."

while pgrep -f "farm_phase2.sh|farm_epsfloor.sh" > /dev/null; do sleep 60; done

for is in 11285 33428 777 4242; do
  d="runs/phase2-noise2/p-113/seed2034/seed$is"
  if [ -f "$d/metrics.json" ]; then echo "skip $d (done)"; continue; fi
  echo "=== phase2-noise2 sigma0.2-until16k init $is ($(date +%H:%M:%S)) ==="
  uv run python -m grok.train --run-name "phase2-noise2/p-113/seed2034/seed$is" \
    --p 113 --data-seed 2034 --init-seed "$is" --embed-init orthogonal \
    --grad-noise 0.2 --grad-noise-until 16000 \
    --num-epochs 20000 --save-every 1000 --spectra-every 100
done
echo NOISE2-DONE
