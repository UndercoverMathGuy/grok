#!/bin/bash
# usage: run_jobs.sh jobsfile nparallel ; each line "script args..." -> out/<script>_<args joined>.out
cd /home/user/grok/heracles/conjecture20/unify/C_hunt
cat "$1" | xargs -P "$2" -I{} bash -c 'line="{}"; name=$(echo "$line" | sed "s/\.py//; s/ /_/g"); OMP_NUM_THREADS=1 timeout 900 uv run --with numpy --with scipy --with mpmath python $line > out/$name.out 2>&1'
