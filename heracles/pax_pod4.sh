#!/usr/bin/env bash
# Phase C (revised): the lambda ladder on the prime axis.  p 211 at lambda 1e-4 sits exactly on the separation
# boundary (median margin -0.009), so gamma is undefined above p ~ 200 at that lambda -- the ladder is the
# experiment, not a continuation.  p 509 is run here (lambda 1e-5) rather than at 1e-4, where it cannot separate.
set -uo pipefail
cd /workspace
export PAX_OUT=/workspace/pax_runs PAX_DEV=cuda
PY="python3 -u heracles/pax_run.py --act pow --m 500 --init_scale 4 --clip 1 --lr_const 12.8 \
    --wd_warmup 6000 --escape_frac 0.8 --report 12500 --nosave --steps 25000"
S16="100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115"
S8="100 101 102 103 104 105 106 107"
for P in 59 71 89 97 113 127 157; do $PY --p $P --seeds $S16 --lam 1e-5 --tag paxC_p${P}_lam1e-5; done
$PY --p 211 --seeds $S16 --lam 1e-5 --achunk 64 --tag paxC_p211_lam1e-5
$PY --p 307 --seeds $S8  --lam 1e-5 --achunk 64 --tag paxC_p307_lam1e-5
echo PHASE_C_DONE
$PY --p 157 --seeds $S8 --lam 1e-6 --tag paxD3_p157_lam1e-6
python3 -u heracles/pax_run.py --act pow --m 500 --init_scale 4 --clip 1 --lr_const 12.8 --wd_warmup 6000 \
    --escape_frac 0.8 --report 25000 --nosave --steps 100000 --p 157 --seeds $S8 --lam 1e-4 --tag paxD1_p157_long100k
$PY --p 509 --seeds $S8 --lam 1e-5 --achunk 16 --tag paxE_p509_lam1e-5
echo PHASE_D_DONE
