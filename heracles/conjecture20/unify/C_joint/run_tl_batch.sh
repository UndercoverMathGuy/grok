#!/bin/bash
cd /home/user/grok/heracles/conjecture20/unify/C_joint
for a in "17 7 4.0 40" "19 7 4.0 40" "23 7 4.0 40" "19 15 8.0 40" "23 15 8.0 40" "19 3 2.0 40" "23 3 2.0 40" "19 5 3.0 40" "23 5 3.0 40" "23 2 1.5 40" "29 5 3.0 30" "31 3 2.0 30" "41 5 3.0 20" "43 3 2.0 20"; do
  OMP_NUM_THREADS=1 timeout 850 uv run --with numpy --with scipy --with mpmath python tl_search2.py $a 2>&1 | grep -v -i warning
done
