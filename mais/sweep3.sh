#!/bin/zsh
cd /Users/ruhaanrajadhyaksha/projects/grok
until grep -q "MAIS SWEEP2 DONE" mais.log; do sleep 20; done
for m in 23 46; do uv run python mais/selection_law.py --p 23 --m $m --wd 1e-4 --seeds 64 --steps 10000 --device mps; done
uv run python mais/selection_law.py --p 23 --wd 3e-3 --seeds 64 --steps 10000 --device mps
uv run python mais/selection_law.py --p 23 --m 23 --wd 3e-3 --seeds 64 --steps 10000 --device mps
echo MAIS SWEEP3 DONE
