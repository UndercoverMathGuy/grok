#!/bin/zsh
cd /Users/ruhaanrajadhyaksha/projects/grok
until grep -q "MAIS SWEEP DONE" mais.log; do sleep 20; done
uv run python mais/selection_law.py --p 23 --wd 1e-4 --seeds 64 --steps 10000 --opt gd --lr 1.0 --device mps
uv run python mais/selection_law.py --p 47 --wd 1e-4 --seeds 16 --steps 10000 --device mps
uv run python mais/selection_law.py --p 23 --wd 1e-4 --seeds 64 --steps 10000 --act quad --device mps
echo MAIS SWEEP2 DONE
