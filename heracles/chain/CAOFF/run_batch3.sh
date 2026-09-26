#!/bin/bash
cd "$(dirname "$0")"
while pgrep -f run_batch2.sh > /dev/null; do sleep 5; done
run() { uv run --with numpy --with torch python ascent.py "$@" >> batch3.log 2>&1; }
par() { while [ "$(jobs -r | wc -l)" -ge 5 ]; do sleep 1; done; }
for p in 5 7 11; do N=$(( 3 * (p - 1) / 2 + 3 ))
  for xj in "0 0.1" "0 0.25" "0.1 0.1" "0.1 0.25" "0.3 0.1" "0.3 0.3" "0.05 0"; do set -- $xj
    for st in blockmix random; do for seed in 0 1; do
      run $p $N u P $st $seed 2000 0.1 $1 $2 & par
    done; done
  done
done
wait; echo DONE >> batch3.log
