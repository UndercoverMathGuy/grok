#!/bin/bash
cd /home/user/grok/heracles/conjecture20/unify/global
for args in "11 2 2.0 3,5 6 avg 5,4" "13 2 2.0 3,5 6 avg 6,5" "17 2 2.0 3,5 5 avg 8,7,6" "19 2 2.0 3,5 4 avg 9,7" "23 2 2.0 3 4 avg 11,9" "5 3 2.0 4,6 6 avg 2,1" "7 3 2.0 4,6 5 avg 3,2" "11 3 2.0 4 4 avg 5,4" "5 4 2.0 5,7 4 avg 2,1" "13 2 2.5 3,5 5 avg 6"; do
  OMP_NUM_THREADS=1 timeout 2400 uv run --with numpy --with scipy --with mpmath python direction2.py $args 2>&1 | grep -v UV_NATIVE
done
