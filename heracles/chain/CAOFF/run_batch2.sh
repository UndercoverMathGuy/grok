#!/bin/bash
cd "$(dirname "$0")"
run() { uv run --with numpy --with torch python ascent.py "$@" >> batch2.log 2>&1; }
par() { while [ "$(jobs -r | wc -l)" -ge 5 ]; do sleep 1; done; }
# (b) forced X / junk at p = 5, 7
for p in 5 7; do N=$([ $p = 5 ] && echo 9 || echo 12)
  for xj in "0.05 0" "0.15 0" "0.3 0" "0.05 0.1" "0.15 0.2" "0 0.2" "0 0.4"; do set -- $xj
    for st in blockmix lmax random; do for seed in 0 1; do
      run $p $N u diag $st $seed 1500 0.1 $1 $2 & par
    done; done
  done
done
# (a) larger p, unforced
for p in 7 11 13; do N=$(( 3 * (p - 1) / 2 + 3 ))
  for bud in diag P; do for st in blockmix single random; do for seed in 0 1; do
    run $p $N u $bud $st $seed 1500 0.1 & par
  done; done; done
done
wait; echo DONE >> batch2.log
