#!/usr/bin/env bash
# PAX pod cohort.  Run ON the pod from /workspace:
#   (setsid nohup bash heracles/pax_pod.sh A > pax_A.log 2>&1 < /dev/null &)
# A = RECIPE SELECTION. Does the quality of the basin depend on the learning rate AT ESCAPE?
#     Same decay-free warm-up (lambda = 0 until CE < 0.8 ln p) at three escape learning rates, 16 seeds,
#     at the prime where the archived recipe is mediocre (113) and the one where it fails outright (157).
# B = PRIME AXIS with the winning recipe (ESCLR set from A).
set -uo pipefail
cd /workspace
export PAX_OUT=/workspace/pax_runs PAX_DEV=cuda
PY="python3 -u heracles/pax_run.py --act pow --m 500 --lam 1e-4 --init_scale 4 --clip 1 --escape_frac 0.8 --report 1000 --nosave"
SEEDS="100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115"
STEPS=${STEPS:-12000}
phase=$1; shift
case $phase in
  A)
    for P in 113 157; do
      for LRE in 12.8 1.6 0.4; do
        WDW=$(python3 -c "print(int(6000*12.8/$LRE))")
        $PY --p $P --seeds $SEEDS --steps $STEPS --lr_const $LRE --lr_after 12.8 --wd_warmup $WDW \
            --tag paxA_p${P}_esclr${LRE}
      done
    done
    echo PHASE_A_DONE ;;
  B)
    ESCLR=${ESCLR:-1.6}
    for P in "$@"; do
      WDW=$(python3 -c "print(int(6000*12.8/$ESCLR))")
      S=$SEEDS; [ "$P" -ge 211 ] && S="100 101 102 103 104 105 106 107"
      $PY --p $P --seeds $S --steps $STEPS --lr_const $ESCLR --lr_after 12.8 --wd_warmup $WDW \
          --tag paxB_p${P}_esclr${ESCLR}
    done
    echo PHASE_B_DONE ;;
esac

# --- appended: phases BC (prime axis at two lambdas) and CTL (no-warm-up control) -----------------
