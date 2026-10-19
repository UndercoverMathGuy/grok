#!/bin/bash
cd /home/user/grok/heracles/conjecture20/unify/global
for args in "11 2 2.0 3,5 6 5,4,3,1" "13 2 2.0 3,5 5 6,5,4,1" "5 3 2.0 4,6 6 1,2" "7 3 2.0 4,6 5 3,2,1" "5 4 2.0 5,7 4 1,2" "7 2 2.5 3,5 5 3,1" "5 3 3.0 4,6 5 1,2"; do
  OMP_NUM_THREADS=1 timeout 1500 uv run --with numpy --with scipy --with mpmath python direction.py $args 2>&1 | grep -v UV_NATIVE
done
