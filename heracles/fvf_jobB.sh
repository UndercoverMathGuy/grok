#!/bin/zsh
# Job B: FREE support, free junk, random init, but with the GOOD optimiser (normalised GD + exact line search on
# the global scale, lam annealed 1e-5 -> 1e-7).  This separates "the construction does the work" from
# "the Fourier projection does the work".  Chain n=2, p=71, nu=3.
cd /Users/ruhaanrajadhyaksha/projects/grok
export OMP_NUM_THREADS=3
for W in 30 60 150; do
  nice -n 10 uv run python heracles/fvf_fixed.py --n 2 --p 71 --start randfull --Ks 35 --seeds 0 1 2 \
      --steps 6000 --eta 0.01 --width $W --tag _rf$W
done
echo JOB_B_DONE
