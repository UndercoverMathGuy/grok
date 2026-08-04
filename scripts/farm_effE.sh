#!/bin/bash
# Arm E: the composition — noise erases unearned early leads (sigma 0.2 -> 0
# at 4k), tilt 5 supplies the merit signal, wd 1 keeps the audition/laundering
# balance that won every sweep. Orthogonal W_E, cell 2034, 4 paired seeds.
set -e
cd "$(dirname "$0")/.."
for is in 11285 33428 777 4242; do
  d="runs/eff-E/p-113/seed2034/seed$is"
  if [ -f "$d/metrics.json" ]; then echo "skip $d (done)"; continue; fi
  echo "=== eff-E noise0.2@4k+tilt5 init $is ($(date +%H:%M:%S)) ==="
  uv run python -m grok.train --run-name "eff-E/p-113/seed2034/seed$is" \
    --p 113 --data-seed 2034 --init-seed "$is" --embed-init orthogonal \
    --loss-tilt 5.0 --grad-noise 0.2 --grad-noise-until 4000 \
    --num-epochs 20000 --save-every 1000 --spectra-every 100
done
echo EFFE-DONE
