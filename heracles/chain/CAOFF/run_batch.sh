#!/bin/bash
# stage 1: p = 5, all variants, several seed types, 4 in parallel
cd "$(dirname "$0")"
run() { uv run --with numpy --with torch python ascent.py "$@" >> batch1.log 2>&1; }
for var in u s f; do
  for st in blockmix lmax single random; do
    for seed in 0 1 2; do
      for sig in 0.05 0.2; do
        [ "$st" = random ] && [ "$sig" = 0.2 ] && continue
        run 5 9 $var diag $st $seed 1500 $sig &
        while [ "$(jobs -r | wc -l)" -ge 4 ]; do sleep 1; done
      done
    done
  done
done
wait
echo DONE >> batch1.log
