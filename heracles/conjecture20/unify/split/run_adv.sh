#!/bin/bash
export OMP_NUM_THREADS=1
R="uv run --with numpy --with scipy --with torch python adv_dq.py"
timeout 1500 $R 5 2 2.5 6 8 2>&1 | grep -v UV_NATIVE > adv_dq_p5_n2_b2.5_m6.out
timeout 1500 $R 5 2 3.0 6 8 2>&1 | grep -v UV_NATIVE > adv_dq_p5_n2_b3.0_m6.out
timeout 1500 $R 7 2 2.0 9 6 2>&1 | grep -v UV_NATIVE > adv_dq_p7_n2_b2.0_m9.out
timeout 1500 $R 7 2 2.5 9 6 2>&1 | grep -v UV_NATIVE > adv_dq_p7_n2_b2.5_m9.out
timeout 2400 $R 5 3 2.0 8 6 2>&1 | grep -v UV_NATIVE > adv_dq_p5_n3_b2.0_m8.out
echo done > adv_done.flag
