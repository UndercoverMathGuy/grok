#!/bin/bash
# Arm G: CVaR-5% loss (mean CE over the worst 5% of train examples) replacing
# tilt — cohort-voting worst-case pressure, no single-example thrash, pressure
# doesn't sharpen as training converges. Orth W_E, wd 1, cell 2034, paired seeds.
set -e
cd "$(dirname "$0")/.."
for is in 11285 33428 777 4242; do
  d="runs/eff-G/p-113/seed2034/seed$is"
  if [ -f "$d/metrics.json" ]; then echo "skip $d (done)"; continue; fi
  echo "=== eff-G cvar0.05 init $is ($(date +%H:%M:%S)) ==="
  uv run python -m grok.train --run-name "eff-G/p-113/seed2034/seed$is" \
    --p 113 --data-seed 2034 --init-seed "$is" --embed-init orthogonal \
    --loss-cvar 0.05 \
    --num-epochs 20000 --save-every 1000 --spectra-every 100
done
echo EFFG-DONE
