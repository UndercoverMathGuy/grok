#!/bin/zsh
cd /Users/ruhaanrajadhyaksha/projects/grok
until grep -q "MAIS SWEEP3 DONE" mais.log; do sleep 20; done
for wd in 0.1 1; do uv run python mais/selection_law.py --p 23 --wd $wd --decoupled --seeds 64 --steps 10000 --device mps; done
echo MAIS SWEEP4 DONE
