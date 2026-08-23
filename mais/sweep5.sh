#!/bin/zsh
cd /Users/ruhaanrajadhyaksha/projects/grok
until grep -q "MAIS SWEEP4 DONE" mais.log; do sleep 20; done
uv run python mais/selection_law.py --p 23 --wd 1e-4 --seeds 256 --steps 10000 --act quad --device mps
for m in 23 46; do uv run python mais/selection_law.py --p 23 --m $m --wd 1e-4 --seeds 128 --steps 10000 --act quad --device mps; done
uv run python mais/selection_law.py --p 47 --wd 1e-4 --seeds 32 --steps 10000 --act quad --device mps
uv run python mais/selection_law.py --p 23 --wd 1e-4 --seeds 128 --steps 10000 --act quad --opt gd --lr 1.0 --device mps
echo MAIS SWEEP5 DONE
