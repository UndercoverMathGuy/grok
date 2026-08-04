#!/bin/bash
# Worker for the orthWE farm: args are "dataseed:initseed" pairs, run sequentially.
set -e
cd "$(dirname "$0")/.."
for spec in "$@"; do
  ds="${spec%%:*}"; is="${spec##*:}"
  d="runs/orthWE/p-113/seed$ds/seed$is"
  if [ -f "$d/metrics.json" ]; then echo "skip $d (done)"; continue; fi
  echo "=== orthWE p-113 data $ds init $is ($(date +%H:%M:%S)) ==="
  uv run python -m grok.train --run-name "orthWE/p-113/seed$ds/seed$is" \
    --p 113 --data-seed "$ds" --init-seed "$is" --embed-init orthogonal \
    --num-epochs "${EPOCHS:-30000}" --save-every 1000 --spectra-every "${SPECTRA:-50}"
done
echo WORKER-DONE
