#!/bin/zsh
cd /Users/ruhaanrajadhyaksha/projects/grok
until grep -q "MAIS SWEEP6 DONE" mais.log; do sleep 20; done
uv run python mais/selection_law.py --p 23 --wd 1e-4 --seeds 64 --steps 30000 --act quad --device mps --out notes/mais/long
uv run python mais/selection_law.py --p 47 --wd 1e-4 --seeds 32 --steps 30000 --act quad --device mps --out notes/mais/long
echo MAIS SWEEP7 DONE
