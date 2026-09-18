#!/usr/bin/env bash
# Gated-chain (product) family on a pod: h_i = prod_j e_ij(a_j), nu = n + 1, exact for every n and every q in [2, nu].
# Run ON the pod from /workspace:  (setsid nohup bash heracles/prod_pod.sh PHASE > prod_PHASE.log 2>&1 < /dev/null &)
#   PR   escape probe for n 3 (init x steps 10k, warm-up on)                      ~10 min
#   P2   n 2 p 71 untied, 4 seeds, Morwani recipe (nu 3, K* 5)                      ~2 min
#   P3   n 3 p 71 untied, 4 seeds, 120k (nu 4, K* 4); recipe from the 09-17 probe: init 14, warm-up, lr2 3.2, LAM 1e-5   ~3.5 h
#        (lam 1e-4 under-fits: acc 0.966, margin -14 at 120k -- the product term lacks the multinomial 6 of x^3, i.e. 2.4x stronger decay)
#   PQ   n 3 p 71 q sweep: q 8/3 (e 1.5 -> 5) and q 4 (e 1 -> dense), 2 seeds each  ~2 h
#   P4   n 4 p 59 tied, 2 seeds, 120k (nu 5, K* 3)                                  ~2 h
set -uo pipefail
cd /workspace; export NSUM_DEV=cuda
PY="python3 -u heracles/nsum.py --act prod"
INIT3=${INIT3:-14}; LR2=${LR2:-3.2}; INIT4=${INIT4:-14}; STEPS=${STEPS:-120000}; LAM=${LAM:-1e-5}
PY="$PY --lam $LAM"
case $1 in
  PR) for init in 1 4 14; do $PY --n 3 --p 71 --seeds 0 --steps 10000 --init_scale $init --lrmax 12.8 --clip 1 --wd_warmup 20000 --escape_frac 0.8 --lrmax2 3.2; done; echo "PR DONE" ;;
  P2) $PY --n 2 --p 71 --seeds 0 1 2 3 --steps 40000 --init_scale 1 --lrmax 12.8; echo "P2 DONE" ;;
  P3) $PY --n 3 --p 71 --seeds 0 1 2 3 --steps $STEPS --init_scale $INIT3 --lrmax 12.8 --clip 1 --wd_warmup 20000 --escape_frac 0.8 --lrmax2 $LR2; echo "P3 DONE" ;;
  PQ) for q in 2.6667 4; do $PY --n 3 --p 71 --penalty l2q --q $q --seeds 0 1 --steps $STEPS --init_scale $INIT3 --lrmax 12.8 --clip 1 --wd_warmup 20000 --escape_frac 0.8 --lrmax2 $LR2; done; echo "PQ DONE" ;;
  P4) $PY --n 4 --p 59 --tied --seeds 0 1 --steps $STEPS --init_scale $INIT4 --lrmax 12.8 --clip 1 --wd_warmup 20000 --escape_frac 0.8 --lrmax2 $LR2; echo "P4 DONE" ;;
  *) echo "unknown phase"; exit 1 ;;
esac
