#!/bin/bash
cd /home/user/grok/heracles/conjecture20/unify/global
run(){ OMP_NUM_THREADS=1 timeout 2400 uv run --with numpy --with scipy --with mpmath python w1_compensation.py "$@" 2>&1 | grep --line-buffered -v UV_NATIVE; }
run 5 2 2.0,2.1,2.2,2.3 1 4 12
run 5 2 2.0,2.2 1 8 12
run 7 2 2.0,2.2,2.5 1,2 8 12
run 7 2 2.0,2.5 1,2 11 8
run 5 3 2.0,2.15 1 6 10
