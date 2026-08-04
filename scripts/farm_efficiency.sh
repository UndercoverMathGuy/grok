#!/bin/bash
# Efficiency pivot (all eps=1e-8, orthogonal W_E, cell 2034, 4 paired seeds):
#   probe: tilt 15 stability (6k epochs)
#   arm A: tilt 15            — cranked worst-case pressure
#   arm B: tilt 5  + wd 2.5   — tax the amplitude-compensation channel
#   arm C: tilt 15 + wd 2.5   — both
# A and C are skipped if the probe fails to memorize (final train CE > 1e-3).
set -e
cd "$(dirname "$0")/.."

run_one() {
  name=$1; is=$2; shift 2
  d="runs/$name/p-113/seed2034/seed$is"
  if [ -f "$d/metrics.json" ]; then echo "skip $d (done)"; return; fi
  echo "=== $name init $is ($(date +%H:%M:%S)) ==="
  uv run python -m grok.train --run-name "$name/p-113/seed2034/seed$is" \
    --p 113 --data-seed 2034 --init-seed "$is" --embed-init orthogonal \
    --num-epochs 20000 --save-every 1000 --spectra-every 100 "$@"
}

if [ ! -f runs/phase2-probes/tilt15-e8/metrics.json ]; then
  uv run python -m grok.train --run-name phase2-probes/tilt15-e8 --p 113 \
    --data-seed 2034 --init-seed 11285 --embed-init orthogonal \
    --loss-tilt 15.0 --num-epochs 6000 --save-every 1000
fi
STABLE=$(uv run python -c "
import json
tr = json.load(open('runs/phase2-probes/tilt15-e8/metrics.json'))['train_losses']
print('yes' if tr[-1] < 1e-3 else 'no')")
echo "tilt15 probe stable: $STABLE (final train CE gate 1e-3)"

for is in 11285 33428 777 4242; do run_one eff-B $is --loss-tilt 5.0 --weight-decay 2.5; done
if [ "$STABLE" = "yes" ]; then
  for is in 11285 33428 777 4242; do run_one eff-A $is --loss-tilt 15.0; done
  for is in 11285 33428 777 4242; do run_one eff-C $is --loss-tilt 15.0 --weight-decay 2.5; done
else
  echo "SKIPPING arms A and C (tilt 15 unstable)"
fi
echo EFFICIENCY-FARM-DONE
