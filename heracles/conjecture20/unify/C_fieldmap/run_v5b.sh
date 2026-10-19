#!/bin/bash
# directional-price searches, chain B (p=7,11,13)
for cfg in "7 3 2.0 16" "11 3 2.0 12" "7 4 2.0 12" "13 3 2.0 10" "7 3 2.5 12" "7 5 2.0 8"; do
  set -- $cfg
  OMP_NUM_THREADS=1 timeout 1500 uv run --with numpy --with scipy python v5_directional.py $1 $2 $3 $4 2 2>&1 | grep -v warning >> v5_directional_B.out
done
