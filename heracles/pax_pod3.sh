#!/usr/bin/env bash
# PAX phase D -- what IS the residual quality decay with p?  Two probes at p 157 (quality 0.18 at 25k/m500/lam1e-4):
#   D1 long   : 100k steps          -> is it just the step budget?
#   D2 wide   : m 2000, 25k steps   -> is width 500 the limit once n_f = 78?
#   D3 lam1e-6: 25k steps           -> does it keep improving as lambda falls?
set -uo pipefail
cd /workspace
export PAX_OUT=/workspace/pax_runs PAX_DEV=cuda
PY="python3 -u heracles/pax_run.py --act pow --init_scale 4 --clip 1 --lr_const 12.8 --wd_warmup 6000 \
    --escape_frac 0.8 --report 10000 --nosave --p 157"
S8="100 101 102 103 104 105 106 107"
$PY --m 500  --seeds $S8 --lam 1e-4 --steps 100000 --tag paxD1_p157_long100k
$PY --m 2000 --seeds $S8 --lam 1e-4 --steps 25000  --tag paxD2_p157_m2000
$PY --m 500  --seeds $S8 --lam 1e-6 --steps 25000  --tag paxD3_p157_lam1e-6
echo PHASE_D_DONE
