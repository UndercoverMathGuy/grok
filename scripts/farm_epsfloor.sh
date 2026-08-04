#!/bin/bash
# CE-floor discriminating experiment: does the ~1e-7 train-CE equilibrium move
# when Adam's eps drops 1e-8 -> 1e-10? (H6 aftermath: wd anneal was a null, so
# the floor is pinned by something else — eps is suspect #1.)
# Mainline config (normal W_E, wd 1.0), p=113, cell 2034, 8 init seeds.
# Waits for the phase-2 farm to finish before touching the GPU.
set -e
cd "$(dirname "$0")/.."

while pgrep -f "farm_phase2.sh" > /dev/null; do sleep 60; done

for is in 11285 33428 777 4242 1001 1002 1003 1004; do
  d="runs/epsfloor/p-113/seed2034/seed$is"
  if [ -f "$d/metrics.json" ]; then echo "skip $d (done)"; continue; fi
  echo "=== epsfloor eps=1e-10 init $is ($(date +%H:%M:%S)) ==="
  uv run python -m grok.train --run-name "epsfloor/p-113/seed2034/seed$is" \
    --p 113 --data-seed 2034 --init-seed "$is" --adam-eps 1e-10 \
    --num-epochs 20000 --save-every 1000
done
echo EPSFLOOR-DONE
