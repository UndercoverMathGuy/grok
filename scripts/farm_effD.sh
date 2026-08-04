#!/bin/bash
# Arm D: tilt 5 + wd 0.4 — stretch the audition window (wd is the audition
# clock per the eff-B post-mortem) so tilt has more epochs to steer the vote.
# Queued behind the efficiency farm (arm A). Longer audition may delay
# grokking, so 30k epochs instead of 20k.
set -e
cd "$(dirname "$0")/.."
while pgrep -f "farm_efficiency.sh" > /dev/null; do sleep 60; done
for is in 11285 33428 777 4242; do
  d="runs/eff-D/p-113/seed2034/seed$is"
  if [ -f "$d/metrics.json" ]; then echo "skip $d (done)"; continue; fi
  echo "=== eff-D tilt5 wd0.4 init $is ($(date +%H:%M:%S)) ==="
  uv run python -m grok.train --run-name "eff-D/p-113/seed2034/seed$is" \
    --p 113 --data-seed 2034 --init-seed "$is" --embed-init orthogonal \
    --loss-tilt 5.0 --weight-decay 0.4 \
    --num-epochs 30000 --save-every 1000 --spectra-every 100
done
echo EFFD-DONE
