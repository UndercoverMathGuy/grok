#!/usr/bin/env bash
# q sweep on the x^2 two-operand net at p 71 (penalty exponent dial, program exponent e = 3/q). Run ON the pod:
#   (setsid nohup bash heracles/qsweep_pod.sh > qsweep.log 2>&1 < /dev/null &)
# Phase 1: q in 1.5 1.2 1.0 2.5 3 2, seeds 0..3, lam 1e-4, Morwani recipe (init 1, lr 0.05 doubling to 12.8, 40k steps, m 500)
set -uo pipefail
cd /workspace
export NSUM_DEV=cuda
PY="python3 -u heracles/nsum.py"
SEEDS=${SEEDS:-"0 1 2 3"}; LAM=${LAM:-1e-4}; STEPS=${STEPS:-40000}
for q in ${QS:-1.5 1.2 1.0 2.5 3 2}; do
  $PY --n 2 --p 71 --penalty l2q --q $q --seeds $SEEDS --steps $STEPS --init_scale 1 --lrmax 12.8 --m 500 --lam $LAM
done
echo "QSWEEP DONE"
