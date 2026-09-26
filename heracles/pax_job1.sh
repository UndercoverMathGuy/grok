#!/bin/zsh
# JOB 1 -- measure the ESCAPE PROGRESS TIME tau*(p) with the penalty switched OFF (lambda = 0).
# Theory (PRIME_AXIS sec 2): with lambda = 0 the norm is constant, so tau(t) = lr * ||theta_0|| * t and the step at
# which CE first falls below 0.8 ln p measures tau* directly.  tau* must be independent of lr and of init scale.
set -e
cd /Users/ruhaanrajadhyaksha/projects/grok
R="uv run python heracles/pax_run.py --lam 0 --steps 6000 --stop_ce 0.8 --report 100 --nosave --threads 3"
# prime axis, init 4, lr 12.8
for P in 59 71 89 113 157; do
  ${=R} --p $P --init_scale 4 --lr_const 12.8 --seeds 100 101 --tag pax_tau_p${P}_i4_lr12.8
done
for P in 211 307; do
  ${=R} --p $P --init_scale 4 --lr_const 12.8 --seeds 100 --tag pax_tau_p${P}_i4_lr12.8
done
# controls at p 71: tau* must not move with lr or with init scale
${=R} --p 71 --init_scale 4 --lr_const 3.2 --seeds 100 101 --tag pax_tau_p71_i4_lr3.2
${=R} --p 71 --init_scale 2 --lr_const 12.8 --seeds 100 101 --tag pax_tau_p71_i2_lr12.8
${=R} --p 71 --init_scale 1 --lr_const 12.8 --seeds 100 101 --tag pax_tau_p71_i1_lr12.8
${=R} --p 113 --init_scale 2 --lr_const 12.8 --seeds 100 --tag pax_tau_p113_i2_lr12.8
echo JOB1_DONE
