#!/bin/bash
# usage: run_jobs2.sh jobsfile nparallel ; each line "script args..." -> out2/<script>_<args joined>.out
cd /home/user/grok/heracles/conjecture20/unify/C_hunt
cat "$1" | xargs -P "$2" -I{} bash -c 'line="{}"; name=$(echo "$line" | sed "s/\.py//; s/ /_/g"); OMP_NUM_THREADS=1 timeout 900 uv run --with numpy --with scipy --with mpmath python -u $line 2>&1 | grep --line-buffered -v UV_NATIVE > out2/$name.out'
