#!/usr/bin/env bash
# Pod-side scoring loop (full interface-ablation instrument, CPU float64, runs alongside training).
# A cell is complete when every seed*/ dir under it has spectra.npz. Scores final weights to
# heracles_scores/<tag>/<cell>.json and every checkpoint ckpt_XXXXXX.safetensors present in the first seed dir to
# heracles_scores/<tag>/<cell>@ckpt_XXXXXX.json. Exits with SCORE DONE once the training log says ALL DONE and
# nothing is left to score. Usage: score_loop.sh <train log> [runs dir] [scores dir]
LOG=$1; RUNS=${2:-/workspace/heracles_runs}; OUT=${3:-/workspace/heracles_scores}
cd /workspace
while true; do
  did=0
  for cell in "$RUNS"/*/*/; do
    [ -d "$cell" ] || continue
    tag=$(basename "$(dirname "$cell")"); n=$(basename "$cell")
    incomplete=0; for sd in "$cell"seed*/; do [ -f "$sd/spectra.npz" ] || incomplete=1; done
    [ $incomplete = 1 ] && continue
    mkdir -p "$OUT/$tag"
    if [ ! -f "$OUT/$tag/$n.json" ]; then
      echo "[$(date +%H:%M)] score $tag/$n"; python3 -u heracles/score.py --runs "$cell" --out "$OUT/$tag/$n.json" > "$OUT/$tag/$n.log" 2>&1; did=1
    fi
    first=$(ls -d "$cell"seed*/ | head -1)
    for ck in "$first"ckpt_*.safetensors; do
      [ -f "$ck" ] || continue; b=$(basename "$ck" .safetensors)
      if [ ! -f "$OUT/$tag/$n@$b.json" ]; then
        echo "[$(date +%H:%M)] score $tag/$n@$b"; python3 -u heracles/score.py --runs "$cell" --ckpt "$b.safetensors" --out "$OUT/$tag/$n@$b.json" > "$OUT/$tag/$n@$b.log" 2>&1; did=1
      fi
    done
  done
  if [ $did = 0 ] && grep -q "ALL DONE" "$LOG" 2>/dev/null; then echo "SCORE DONE"; break; fi
  [ $did = 0 ] && sleep 300
done
