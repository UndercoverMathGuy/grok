#!/usr/bin/env bash
# FVF pod programme (2026-09-19): does the fixed-support ladder survive a RANDOM start, and can free training
# be driven to the ceiling, at the degree that carries the RESULTS_NU4 verdict (nu = 4, gated chain, p = 71)?
# Run ON the pod from /workspace:  (setsid nohup bash heracles/fvf_pod.sh PHASE > fvf_PHASE.log 2>&1 < /dev/null &)
set -uo pipefail
cd /workspace; export NSUM_DEV=cuda
PH=$1; shift
OUT=heracles/FINAL_RUNS/FVF
FX="python3 -u heracles/fvf_fixed.py --out $OUT"
FR="python3 -u heracles/fvf_free.py --out $OUT"
case $PH in
  # ---- A. random init INSIDE the projected span: does the ladder survive losing the construction? (nu 4)
  R4)  for K in 2 3 4 5 6 7 8 10 16; do
         $FX --act prod --n 3 --p 71 --start rand --Ks $K --seeds 0 1 2 --steps 6000 --eta 0.01 --width $((12*K)) --tag _lad
       done ;;
  # ---- A'. the same at nu 3 on the paper's prime (chain n 2, p 113) and for Morwani's x^2 net
  R3)  for K in 2 3 4 5 6 7 8 10 16; do
         $FX --act prod --n 2 --p 113 --start rand --Ks $K --seeds 0 1 2 --steps 6000 --eta 0.01 --width $((6*K)) --tag _lad
       done
       for K in 2 3 4 5 6 7 8 10; do
         $FX --act pow --n 2 --p 113 --start rand --Ks $K --seeds 0 1 2 --steps 6000 --eta 0.01 --width $((12*K)) --pop cube --tag _lad
       done ;;
  # ---- B. free support AND random init, same optimiser: is the projection what does the work?
  RF4) for W in 64 250 500; do
         $FX --act prod --n 3 --p 71 --start randfull --Ks 35 --seeds 0 1 2 --steps 6000 --eta 0.01 --width $W --tag _rf$W
       done
       for W in 30 150 500; do
         $FX --act prod --n 2 --p 113 --start randfull --Ks 56 --seeds 0 1 2 --steps 6000 --eta 0.01 --width $W --tag _rf$W
       done ;;
  # ---- C. routes out of the junk basin under ORDINARY training (nu 4 chain, the archived recipe's cell)
  FR4) C="--n 3 --p 71 --m 500 --seeds 0 1 --polish --report 20000 --init 14 --lrmax 12.8 --lrmax2 3.2 --clip 1 --lam 1e-5 --lam_q 1e-6"
       $FR --mode l2      $C --steps 60000
       $FR --mode qnu     $C --steps 40000 --then_l2 30000
       $FR --mode qanneal $C --steps 40000 --then_l2 30000
       $FR --mode junk    $C --steps 40000 --then_l2 30000 --mu 3.0
       $FR --mode ascent  $C --steps 20000 ;;
  *) echo "unknown phase"; exit 1 ;;
esac
echo "$PH DONE"
