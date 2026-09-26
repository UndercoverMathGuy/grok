#!/bin/zsh
# JOB 3 -- is the residual quality gap set by HOW VIOLENT the escape is (the lr at escape), rather than by how
# much norm survives it?  Same warm-up (lambda = 0 until CE < 0.8 ln p) but escape at lr 1.6 instead of 12.8,
# then lr 12.8 for the rest.  Archived cells that escaped at lr 1.6 reach quality 0.35-0.50; those that escaped
# at lr 12.8 reach 0.13-0.26, and the pax_fix p157 run (escape at lr 12.8) reached 0.21.
set -e
cd /Users/ruhaanrajadhyaksha/projects/grok
uv run python heracles/pax_run.py --lam 1e-4 --init_scale 4 --lr_const 1.6 --lr_after 12.8 --clip 1 \
  --wd_warmup 6000 --escape_frac 0.8 --report 500 --threads 3 \
  --p 157 --steps 8000 --seeds 100 101 --tag pax_slow_p157_i4_wu_lr1.6
echo JOB3_DONE
