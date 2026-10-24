#!/bin/bash
# near-transition searches: p=11 n=2 (b_c = 2.0141, 2.2298), p=5 n=3 (b_c = 2.1561), p=7 n=3 (b_c = 2.5848)
cd /home/user/grok/heracles/conjecture20/unify/global
run(){ OMP_NUM_THREADS=1 timeout 3000 uv run --with numpy --with scipy --with mpmath python gsearch.py "$@" 2>&1 | grep --line-buffered -v UV_NATIVE; }
for b in 2.0 2.009 2.019 2.10 2.225 2.235 2.3; do run 11 2 $b 16 24 rand,IF2,IF3,IF5,IF4 1; done
for b in 2.10 2.151 2.161 2.25; do run 5 3 $b 10 24 rand,IF1,IF2 2; done
for b in 2.50 2.58 2.59 2.70; do run 7 3 $b 13 18 rand,IF2,IF3 3; done
