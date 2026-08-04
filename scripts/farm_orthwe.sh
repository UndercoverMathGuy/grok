#!/bin/bash
# Farm the semi-orthogonal-W_E experiment: p=113, two baseline data cells,
# init seeds paired with runs/p-113 where they exist (11285/33428, 54735/66433)
# plus two fresh seeds per cell. Resumable: completed runs are skipped.
set -e
cd "$(dirname "$0")/.."
mkdir -p runs/orthWE

run_one() {
  ds=$1; is=$2
  d="runs/orthWE/p-113/seed$ds/seed$is"
  if [ -f "$d/metrics.json" ]; then echo "skip $d (done)"; return; fi
  echo "=== orthWE p-113 data $ds init $is ($(date +%H:%M:%S)) ==="
  uv run python -m grok.train --run-name "orthWE/p-113/seed$ds/seed$is" \
    --p 113 --data-seed "$ds" --init-seed "$is" --embed-init orthogonal \
    --num-epochs 30000 --save-every 1000 --spectra-every 50
}

for is in 11285 33428 777 4242; do run_one 2034 $is; done
for is in 54735 66433 777 4242; do run_one 3604 $is; done
echo FARM-DONE
