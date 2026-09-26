#!/usr/bin/env bash
# LADDER Part 2 -- the collapse test on the gated chain (--act prod), free training, TRENDS ONLY.
# Run ON the pod from /workspace:  (nohup bash heracles/ladder_pod.sh PHASE > lad_PHASE.log 2>&1 < /dev/null &)
#
# Recipe (PRIME_AXIS sec 5.2 "the fixed recipe" + RESULTS_PROD note 4):
#   penalty OFF until train CE < 0.8 ln p (--wd_warmup 20000 --escape_frac 0.8), constant lr from step 1,
#   grad-norm clip 1, init 4 (n = 2) / 14 (n >= 3), lr cap 3.2 after the warm-up at n >= 3.
#   lam: 1e-4 and 1e-5 at n 2 q 2; 1e-5 for plain l2 at n >= 3; 1e-6 for every q > 2
#        (1e-5 collapses q = 4 to zero weights, RESULTS_PROD).
#
# Cells.  (nu, q) pairs, nu = n+1, programme exponent s = nu/q:
#   A  matched s = 3/2 : (3,2) n2 | (4,8/3) n3 | (5,10/3) n4   -- same predicted share structure
#   B  dense endpoint q = nu : (3,3) n2 | (4,4) n3             -- all frequencies
#   C  degree dial at fixed q = 3 : nu 3 (=dense) | nu 4 | nu 5 -- sparser by degree alone
set -uo pipefail
cd /workspace; export NSUM_DEV=cuda
PY="python3 -u heracles/nsum.py --act prod --m 500 --out heracles/runs/LAD"
W="--clip 1 --wd_warmup 20000 --escape_frac 0.8"
N2="--n 2 --init_scale 4 --lrmax 12.8 $W --steps ${S2:-40000}"
N3="--n 3 --init_scale 14 --lrmax 12.8 --lrmax2 3.2 $W --steps ${S3:-60000}"
N4="--n 4 --tied --m 250 --init_scale 14 --lrmax 12.8 --lrmax2 3.2 $W --steps ${S4:-60000}"
SEEDS="--seeds 0 1 2 3"
case $1 in
  # ---- A0: nu 3 arm of the matched-s triple + the nu 3 dense endpoint (both primes; minutes)
  A0) for p in 59 71; do
        $PY $N2 --p $p $SEEDS --penalty l2 --lam 1e-4
        $PY $N2 --p $p $SEEDS --penalty l2 --lam 1e-5
        $PY $N2 --p $p $SEEDS --penalty l2q --q 3 --lam 1e-6
      done ;;
  # ---- A3: nu 4 arm of the matched-s triple (q = 8/3), p 59
  A3) $PY $N3 --p 59 $SEEDS --penalty l2q --q 2.6666666667 --lam 1e-6 ;;
  # ---- B3/C3: nu 4 dense endpoint (q = 4) and the q = 3 rung of the degree dial, p 59
  B3) $PY $N3 --p 59 $SEEDS --penalty l2q --q 4 --lam 1e-6 ;;
  C3) $PY $N3 --p 59 $SEEDS --penalty l2q --q 3 --lam 1e-6 ;;
  # ---- A4: nu 5 arm of the matched-s triple (q = 10/3), p 59 tied
  A4) $PY $N4 --p 59 $SEEDS --penalty l2q --q 3.3333333333 --lam 1e-6 ;;
  # ---- C4: nu 5 at q = 3 (degree dial), p 59 tied
  C4) [ -f /workspace/RUN_C4 ] || { echo "C4 SKIPPED (budget cap)"; exit 0; }; $PY ${N4/--steps 60000/--steps 30000} --p 59 $SEEDS --penalty l2q --q 3 --lam 1e-6 ;;
  # ---- D3: OPTIONAL second prime for the nu 4 matched-s arm (only if budget allows)
  D3) $PY $N3 --p 71 $SEEDS --penalty l2q --q 2.6666666667 --lam 1e-6 ;;
  # ---- score everything pulled so far
  SCORE) python3 -u heracles/ladder_abl.py --runs heracles/runs/LAD/* \
           --ceil heracles/FINAL_RUNS/LADDER/json/ceilings.json \
           --out heracles/FINAL_RUNS/LADDER/train/scored.json ;;
  VALID) python3 -u heracles/ladder_abl.py --validate 59 \
           --out heracles/FINAL_RUNS/LADDER/train/scored.json ;;
  *) echo "unknown phase"; exit 1 ;;
esac
echo "$1 DONE"
