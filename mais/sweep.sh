#!/bin/zsh
cd /Users/ruhaanrajadhyaksha/projects/grok
for p in 11 13 17 23; do for wd in 1e-4 1e-3; do
  uv run python mais/selection_law.py --p $p --wd $wd --seeds 128 --steps 10000 --device mps
done; done
echo MAIS SWEEP DONE
