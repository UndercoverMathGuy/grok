#!/bin/bash
# E1 grid: random mixed networks and IF+noise starts (sequential, one CPU)
export OMP_NUM_THREADS=1
R="uv run --with numpy --with scipy --with torch python exp_random.py"
for b in 2.0 2.5 3.0; do
  timeout 900 $R 5 2 $b 4,6,8,10 6 rand 2>&1 | grep -v UV_NATIVE > e1_p5_n2_b${b}_rand.out
  timeout 900 $R 5 2 $b 6,9 4 ifnoise 0.3 2>&1 | grep -v UV_NATIVE > e1_p5_n2_b${b}_ifnoise03.out
  timeout 900 $R 5 2 $b 6,9 4 ifnoise 1.0 2>&1 | grep -v UV_NATIVE > e1_p5_n2_b${b}_ifnoise10.out
  timeout 900 $R 7 2 $b 6,9,12 4 rand 2>&1 | grep -v UV_NATIVE > e1_p7_n2_b${b}_rand.out
  timeout 900 $R 7 2 $b 9,12 3 ifnoise 0.3 2>&1 | grep -v UV_NATIVE > e1_p7_n2_b${b}_ifnoise03.out
done
for b in 2.0 3.0 4.0; do
  timeout 1200 $R 5 3 $b 6,8,12 4 rand 2>&1 | grep -v UV_NATIVE > e1_p5_n3_b${b}_rand.out
  timeout 1200 $R 5 3 $b 8,12 3 ifnoise 0.3 2>&1 | grep -v UV_NATIVE > e1_p5_n3_b${b}_ifnoise03.out
done
echo done > e1_done.flag
