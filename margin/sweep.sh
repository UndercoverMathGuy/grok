#!/bin/zsh
cd /Users/ruhaanrajadhyaksha/projects/grok
for p in 7 11 13 17 23; do
  uv run python margin/search.py --p $p --steps 4000 --restarts 96 --fourier 16 --act relu
done
uv run python margin/search.py --p 11 --steps 4000 --restarts 96 --fourier 16 --act quad
for p in 11 13; do   # wider nets: more room for non-sparse solutions
  uv run python margin/search.py --p $p --m $((12*p)) --steps 4000 --restarts 64 --fourier 16 --act relu
done
echo SWEEP DONE
