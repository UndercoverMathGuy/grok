#!/bin/zsh
cd /Users/ruhaanrajadhyaksha/projects/grok
until grep -q "MAIS SWEEP5 DONE" mais.log; do sleep 20; done
uv run python mais/selection_law.py --p 23 --m 184 --wd 1e-4 --seeds 64 --steps 10000 --act quad --device mps
for p in 13 17 31; do uv run python mais/selection_law.py --p $p --wd 1e-4 --seeds 128 --steps 10000 --act quad --device mps; done
echo MAIS SWEEP6 DONE
