#!/bin/bash
# The combined recipe: orthogonal W_E (no lottery) + tilted ERM t=5 (margin
# selector) + adam_eps 1e-11 (unpinned CE floor). 4 paired seeds, cell 2034.
set -e
cd "$(dirname "$0")/.."
for is in 11285 33428 777 4242; do
  d="runs/combined/p-113/seed2034/seed$is"
  if [ -f "$d/metrics.json" ]; then echo "skip $d (done)"; continue; fi
  echo "=== combined init $is ($(date +%H:%M:%S)) ==="
  uv run python -m grok.train --run-name "combined/p-113/seed2034/seed$is" \
    --p 113 --data-seed 2034 --init-seed "$is" --embed-init orthogonal \
    --loss-tilt 5.0 --adam-eps 1e-11 \
    --num-epochs 20000 --save-every 1000 --spectra-every 100
done
echo COMBINED-DONE
