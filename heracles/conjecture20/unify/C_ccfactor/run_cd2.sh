#!/bin/bash
# CD searches with width < nu per feature (every feature non-IF, so cheapness and compensation are forced)
cd /home/user/grok/heracles/conjecture20/unify/C_ccfactor
run() { OMP_NUM_THREADS=1 timeout 900 uv run --with numpy --with scipy --with torch python c3_search.py "$@" 2>&1 | grep -v UV_NATIVE > out/$(echo "$@" | tr ' ,' '__').out; }
run cd 7 2 2.0 1,2,3 2 40 11 &
run cd 11 2 2.0 1,2,3,4,5 2 30 12 &
run cd 13 2 2.0 1,2,3,4,5,6 2 24 13 &
wait
run cd 11 2 2.5 1,2,3,4,5 2 30 14 &
run cd 7 3 2.0 1,2,3 3 20 15 &
run cd 5 3 2.0 1,2 3 40 16 &
wait
