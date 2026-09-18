#!/usr/bin/env bash
# nsum ladder on a pod. Run ON the pod from /workspace:
#   (setsid nohup env INIT4=8 bash heracles/nsum_pod.sh PHASE > nsum_PHASE.log 2>&1 < /dev/null &)
# Phases:
#   A2  init/lr sweep for x^4 (n4 p59 tied) and x^5 (n5 p31 tied), 10k steps, 1 seed         ~50 min on a 4090
#   B3  three-operand column at p 71: x^3 untied (4 seeds), x^3 tied (4), ReLU nu2 anchor (2), L_{2,4} foil (2)   ~4 h
#   B4  x^4 p59 tied (4 seeds) + x^5 p31 tied (4 seeds)                                       ~4 h
#   C   x^4 p71 tied, seeds $2.. (73 min/seed at m 500)
#   D   x^3 p113 untied, seeds $2.. (93 min/seed)
set -uo pipefail
cd /workspace
export NSUM_DEV=cuda
PY="python3 -u heracles/nsum.py"
INIT3=${INIT3:-14}; INIT4=${INIT4:-10}; INIT5=${INIT5:-10}; LRMAX=${LRMAX:-12.8}; STEPS=${STEPS:-40000}; M=${M:-500}; CLIP=${CLIP:-1}; WDW=${WDW:-0}
phase=$1; shift
case $phase in
  A2)
    for init in 4 8 16; do $PY --n 4 --p 59 --tied --seeds 0 --steps 10000 --init_scale $init --lrmax 12.8; done
    $PY --n 4 --p 59 --tied --seeds 0 --steps 10000 --init_scale 8 --lrmax 3.2
    for init in 4 16 64; do $PY --n 5 --p 31 --tied --seeds 0 --steps 10000 --init_scale $init --lrmax 12.8; done
    echo "PHASE A2 DONE" ;;
  B3)
    $PY --n 3 --p 71 --penalty l2   --seeds 0 1 2 3 --steps $STEPS --init_scale $INIT3 --lrmax $LRMAX --m $M --clip $CLIP --wd_warmup $WDW
    $PY --n 3 --p 71 --penalty l2   --seeds 0 1 2 3 --steps $STEPS --init_scale $INIT3 --lrmax $LRMAX --m $M --clip $CLIP --wd_warmup $WDW --tied
    $PY --n 3 --act relu --p 71 --penalty l2 --seeds 0 1 --steps $STEPS --init_scale 1 --lrmax $LRMAX --m $M --clip $CLIP --wd_warmup $WDW
    $PY --n 3 --p 71 --penalty l2nu --seeds 0 1     --steps $STEPS --init_scale 1      --lrmax $LRMAX --m $M --clip $CLIP --wd_warmup $WDW
    echo "PHASE B3 DONE" ;;
  B4)
    $PY --n 4 --p 59 --tied --penalty l2 --seeds 0 1 2 3 --steps $STEPS --init_scale $INIT4 --lrmax $LRMAX --m $M --clip $CLIP --wd_warmup $WDW
    $PY --n 5 --p 31 --tied --penalty l2 --seeds 0 1 2 3 --steps $STEPS --init_scale $INIT5 --lrmax $LRMAX --m $M --clip $CLIP --wd_warmup $WDW
    echo "PHASE B4 DONE" ;;
  C)
    $PY --n 4 --p 71 --tied --penalty l2 --seeds "$@" --steps $STEPS --init_scale $INIT4 --lrmax $LRMAX --m $M --clip $CLIP --wd_warmup $WDW
    echo "PHASE C DONE" ;;
  D)
    $PY --n 3 --p 113 --penalty l2 --seeds "$@" --steps $STEPS --init_scale $INIT3 --lrmax $LRMAX --m $M --clip $CLIP --wd_warmup $WDW
    echo "PHASE D DONE" ;;
  *) echo "unknown phase $phase"; exit 1 ;;
esac
