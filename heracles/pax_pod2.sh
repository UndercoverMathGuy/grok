#!/usr/bin/env bash
# PAX pod phases B / C / CTL.   Run ON the pod:  (setsid nohup bash heracles/pax_pod2.sh > pax_BC.log 2>&1 &)
set -uo pipefail
cd /workspace
export PAX_OUT=/workspace/pax_runs PAX_DEV=cuda
PY="python3 -u heracles/pax_run.py --act pow --m 500 --init_scale 4 --clip 1 --lr_const 12.8 \
    --wd_warmup 6000 --escape_frac 0.8 --report 5000 --nosave --steps 25000"
S16="100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115"
S8="100 101 102 103 104 105 106 107"
# CTL: identical recipe but the penalty ON from step 1 -> isolates the warm-up from the lr schedule
$PY --p 113 --seeds $S16 --lam 1e-4 --wd_warmup 0 --tag paxCTL_p113_nowu
# B: prime axis, lambda 1e-4
for P in 59 71 89 97 113 127 157; do $PY --p $P --seeds $S16 --lam 1e-4 --tag paxB_p${P}_lam1e-4; done
for P in 211 307;             do $PY --p $P --seeds $S16 --lam 1e-4 --achunk 64 --tag paxB_p${P}_lam1e-4; done
$PY --p 509 --seeds $S8 --lam 1e-4 --achunk 32 --tag paxB_p509_lam1e-4
echo PHASE_B_DONE
# C: same axis at lambda 1e-5 (does the residual quality gap close as lambda falls?)
for P in 59 71 89 97 113 127 157; do $PY --p $P --seeds $S16 --lam 1e-5 --tag paxC_p${P}_lam1e-5; done
for P in 211 307;             do $PY --p $P --seeds $S16 --lam 1e-5 --achunk 64 --tag paxC_p${P}_lam1e-5; done
echo PHASE_C_DONE
