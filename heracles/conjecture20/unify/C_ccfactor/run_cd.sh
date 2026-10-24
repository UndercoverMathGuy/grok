#!/bin/bash
# CD adversarial searches (class C). Each run < 12 min.
cd /home/user/grok/heracles/conjecture20/unify/C_ccfactor
run() { OMP_NUM_THREADS=1 timeout 900 uv run --with numpy --with scipy --with torch python c3_search.py "$@" 2>&1 | grep -v UV_NATIVE > out/$(echo "$@" | tr ' ,' '__').out; }
run cd 7 2 2.0 1,2,3 4 30 1 &
run cd 11 2 2.0 1,2,3,4,5 3 20 2 &
run cd 11 2 2.5 1,2,4 4 30 3 &
wait
run cd 13 2 2.2 1,2,3,4,5,6 3 16 4 &
run cd 11 2 2.0 1,3,4 4 30 5 &
run cd 5 3 2.0 1,2 5 30 6 &
wait
run cd 7 3 2.0 1,2,3 4 16 7 &
run cd 13 2 2.0 1,2,5 4 30 8 &
run cd 7 2 2.5 1,3 5 30 9 &
wait
