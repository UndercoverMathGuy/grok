#!/bin/zsh
# O52 scaling: N_delta(p) for C0 in {1,2} ReLU Adam, plus GD at C0=4; CPU (MPS busy with quad queue)
cd /Users/ruhaanrajadhyaksha/projects/grok
for p in 11 13 17 31; do for c in 1 2; do
  uv run python mais/selection_law.py --p $p --m $((c*p)) --wd 1e-4 --seeds 64 --steps 10000 --device cpu
done; done
for p in 11 17 31; do
  uv run python mais/selection_law.py --p $p --wd 1e-4 --seeds 32 --steps 10000 --opt gd --lr 1.0 --device cpu
done
echo O52 SWEEP DONE
