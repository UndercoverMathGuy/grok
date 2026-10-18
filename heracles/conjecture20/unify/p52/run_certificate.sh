#!/bin/bash
# Official re-runnable certificate for the (5,2) single-feature bound (s = 3/2, TARGET 0.995).
# Runs the branch-and-bound in four phi_pay chunks; each log ends with a DONE line (fail must be 0).
cd "$(dirname "$0")"
for L in "0,7" "1,6" "2,5" "3,4"; do
  OMP_NUM_THREADS=1 uv run --with numpy --with scipy --with cvxpy python bb52.py 0.995 cert_l${L/,/_}.log l=$L > cert_l${L/,/_}.out 2>&1 &
done
wait
grep -h DONE cert_l*.log
