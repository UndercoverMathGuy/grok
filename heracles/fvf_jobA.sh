#!/bin/zsh
# Job A: random-init-inside-the-span fixed-support ladder (the de-circularising control) + the
# "release the construction" control (constructed start, NO projection).  Chain n=2, p=71, nu=3.
# Width 4K (>= 18) = about twice the (n+1)K neurons the analytic construction uses; the K=5 width sweep
# showed 3K is width-limited and 6K, 12K give the same answer as 6K.
cd /Users/ruhaanrajadhyaksha/projects/grok
export OMP_NUM_THREADS=3
for K in 2 3 4 5 6 8 12; do
  W=$((4 * K)); [[ $W -lt 18 ]] && W=18
  nice -n 10 uv run python heracles/fvf_fixed.py --n 2 --p 71 --start rand --Ks $K --seeds 0 1 \
      --steps 5000 --eta 0.01 --width $W --tag _lad
done
nice -n 10 uv run python heracles/fvf_fixed.py --n 2 --p 71 --start con --Ks 3 5 8 12 --seeds 0 1 \
    --steps 5000 --eta 0.01 --no_proj --tag _rel
echo JOB_A_DONE
