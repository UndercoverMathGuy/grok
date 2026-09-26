#!/bin/zsh
# Job C: free-training routes out of the junk basin.  Chain n=2, p=71, nu=3, width 100, one seed each
# (a second seed is added only if the first shows something).  Every run ends with the same polish optimiser
# (normalised GD + exact line search on the scale, lam 1e-5 -> 1e-7, NO projection) so gamma is comparable.
cd /Users/ruhaanrajadhyaksha/projects/grok
export OMP_NUM_THREADS=3
COMMON="--n 2 --p 71 --m 100 --seeds 0 --polish --report 4000"
nice -n 10 uv run python heracles/fvf_free.py --mode l2      $=COMMON --steps 20000
nice -n 10 uv run python heracles/fvf_free.py --mode qnu     $=COMMON --steps 12000 --then_l2 8000
nice -n 10 uv run python heracles/fvf_free.py --mode qanneal $=COMMON --steps 12000 --then_l2 8000
nice -n 10 uv run python heracles/fvf_free.py --mode junk    $=COMMON --steps 12000 --then_l2 8000 --mu 3.0
nice -n 10 uv run python heracles/fvf_free.py --mode ascent  $=COMMON --steps 12000
echo JOB_C_DONE
