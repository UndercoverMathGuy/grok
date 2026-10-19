#!/bin/bash
# directional-price searches, chain A (p=5)
for cfg in "5 3 2.0 16" "5 4 2.0 16" "5 3 2.5 12" "5 5 2.0 12" "5 4 3.0 12" "5 6 2.0 8"; do
  set -- $cfg
  OMP_NUM_THREADS=1 timeout 1500 uv run --with numpy --with scipy python v5_directional.py $1 $2 $3 $4 1 2>&1 | grep -v warning >> v5_directional_A.out
done
