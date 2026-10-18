#!/bin/bash
export OMP_NUM_THREADS=1
R="uv run --with numpy --with scipy --with torch python adv_dq.py"
for b in 2.25 2.75; do
timeout 1500 $R 5 2 $b 6 8 2>&1 | grep -v UV_NATIVE > adv_dq_p5_n2_b${b}_m6.out
done
for b in 2.75 3.0; do
timeout 1500 $R 7 2 $b 9 6 2>&1 | grep -v UV_NATIVE > adv_dq_p7_n2_b${b}_m9.out
done
echo done > adv2_done.flag
