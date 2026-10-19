#!/bin/sh
# Re-run the N2MIX certificates (each log must end with "fails=0").
cd "$(dirname "$0")"
R="uv run --with numpy --with scipy --with mpmath --with cvxpy --with clarabel python"
export OMP_NUM_THREADS=1
# (7,2), general networks, every s in [1, 1.2725]  (b >= 2.3576)
$R rig72.py 1.2725 0.99995 pat 34   > out/cert72_pat_s1.2725.log 2>&1
# (7,2), L-pure networks (no DC input, no frequency 3 anywhere), every s in [1, 1.5]  (all b in [2,3])
$R rig72.py 1.5 0.996 nopaid 32     > out/cert72_nopaid_s1.5.log 2>&1
# (11,2), networks with features in {2,3,4,5}, every s in [1, s1]  (dense-regime count)
$R rigp.py 11 1 1.10 0.99 l2 40     > out/cert11_l2_s1.10.log 2>&1
tail -n 1 out/cert72_pat_s1.2725.log out/cert72_nopaid_s1.5.log out/cert11_l2_s1.10.log
