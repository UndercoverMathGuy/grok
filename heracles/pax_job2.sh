#!/bin/zsh
# JOB 2 -- the proposed p-aware recipe:  DECAY-FREE WARM-UP + constant lr + clip, then lambda on.
# lambda = 0 until CE < 0.8 ln p removes the escape race entirely (no pre-escape decay), so escape no longer
# depends on p; lambda = 1e-4 is then switched on for the whole remaining budget, exactly as in the archived runs.
set -e
cd /Users/ruhaanrajadhyaksha/projects/grok
R="uv run python heracles/pax_run.py --lam 1e-4 --init_scale 4 --lr_const 12.8 --clip 1 \
   --wd_warmup 3000 --escape_frac 0.8 --report 500 --threads 3"
${=R} --p 157 --steps 10000 --seeds 100 101 --tag pax_fix_p157_i4_wu
${=R} --p 211 --steps 10000 --seeds 100     --tag pax_fix_p211_i4_wu
${=R} --p 113 --steps 10000 --seeds 100 101 --tag pax_fix_p113_i4_wu
echo JOB2_DONE
