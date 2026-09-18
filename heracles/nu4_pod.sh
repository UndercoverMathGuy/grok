#!/usr/bin/env bash
# NU4 programme (2026-09-18): fixed-support ladders (T1), free-GD recipe variants (T2), seeds (T3).  RESULTS_NU4.md
# Run ON the pod from /workspace:  (setsid nohup bash heracles/nu4_pod.sh PHASE [seeds...] > nu4_PHASE.log 2>&1 < /dev/null &)
set -uo pipefail
cd /workspace; export NSUM_DEV=cuda
PH=$1; shift; SEEDS="${*:-0 1 2 3}"
OUT=heracles/FINAL_RUNS/NU4
NS="python3 -u heracles/nsum.py --grid --out $OUT --trace_every 20000"
FX="python3 -u heracles/nu4_fixed.py --out $OUT"
R3="--n 3 --p 71 --init_scale 14 --lrmax 12.8 --clip 1 --wd_warmup 20000 --escape_frac 0.8"          # nu 4 recipe (RESULTS_PROD)
BASE3=$OUT/n3_prod_p71_l2_lam1e-05_lrmax12.8_init14_m500_st120000_clip1_wdw20000_esc0.8_lr23.2
case $PH in
  F3prod) $FX --act prod --n 3 --p 71 ;;                                         # T1 nu 4 chain
  F3pow)  $FX --act pow --n 3 --p 71 --Ks $SEEDS ;;                              # T1 nu 4 x^3 (args = K list)
  F2)     $FX --act prod --n 2 --p 71 --Ks 2 3 4 5 6 7 8 10 12; $FX --act pow --n 2 --p 71 --Ks 2 3 4 5 6 7 8 10 12 ;;   # nu 3 control
  F4)     $FX --act prod --n 4 --p 59 --tied --Ks 2 3 4 5 6 8 ;;                  # T1 nu 5 (tied, as RESULTS_PROD)
  S3prod) $NS --act prod $R3 --lrmax2 3.2 --lam 1e-5 --steps 120000 --seeds $SEEDS ;;   # T3 nu 4 chain, baseline recipe
  S3pow)  $NS --act pow $R3 --lrmax2 3.2 --lam 1e-4 --steps 120000 --seeds $SEEDS ;;    # T3 nu 4 x^3, RESULTS_NS recipe
  S4)     $NS --act prod --n 4 --p 59 --tied --init_scale 14 --lrmax 12.8 --clip 1 --wd_warmup 20000 --escape_frac 0.8 --lrmax2 3.2 --lam 1e-5 --steps 120000 --seeds $SEEDS ;;  # T3 nu 5
  V_m)    for m in 125 1000; do $NS --act prod $R3 --lrmax2 3.2 --lam 1e-5 --steps 120000 --m $m --seeds $SEEDS; done ;;
  V_lr)   $NS --act prod $R3 --lrmax2 0.8 --lam 1e-5 --steps 120000 --seeds $SEEDS ;;
  V_lam6) $NS --act prod $R3 --lrmax2 3.2 --lam 1e-6 --steps 120000 --seeds $SEEDS ;;
  V_init) $NS --act prod --n 3 --p 71 --init_scale 8 --lrmax 12.8 --clip 1 --wd_warmup 60000 --escape_frac 0.8 --lrmax2 3.2 --lam 1e-5 --steps 120000 --seeds $SEEDS ;;
  V_long) $NS --act prod --n 3 --p 71 --init_scale 14 --lrmax 12.8 --clip 1 --lrmax2 3.2 --lam 1e-5 --steps 240000 --resume $BASE3 --seeds $SEEDS ;;   # same lam, +240k
  V_ann)  $NS --act prod --n 3 --p 71 --init_scale 14 --lrmax 12.8 --clip 1 --lrmax2 3.2 --lam 1e-6 --steps 120000 --resume $BASE3 --seeds $SEEDS ;;   # lam 1e-5 -> 1e-6
  POL)    $FX --act prod --n 3 --p 71 --ckpts $BASE3/seed*/final.safetensors --lams 1e-5 1e-6 1e-7 --steps 6000 ;;   # margin polish of free endpoints
  *) echo "unknown phase"; exit 1 ;;
esac
echo "$PH DONE"
