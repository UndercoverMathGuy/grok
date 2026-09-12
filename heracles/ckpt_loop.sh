#!/usr/bin/env bash
# Pod-side checkpoint scorer for long batches: scores ckpt_XXXXXX.safetensors as soon as EVERY seed dir of a cell has
# it (before the batch finishes). Output heracles_scores/<tag>/<cell>@ckpt_XXXXXX.json (same names as score_loop.sh,
# which skips files that exist). Usage: ckpt_loop.sh <train log> [runs dir] [scores dir]
LOG=$1; RUNS=${2:-/workspace/heracles_runs}; OUT=${3:-/workspace/heracles_scores}
cd /workspace
while true; do
  did=0
  for cell in "$RUNS"/*/*/; do
    [ -d "$cell" ] || continue
    tag=$(basename "$(dirname "$cell")"); n=$(basename "$cell"); mkdir -p "$OUT/$tag"
    first=$(ls -d "$cell"seed*/ 2>/dev/null | head -1); [ -n "$first" ] || continue
    for ck in "$first"ckpt_*.safetensors; do
      [ -f "$ck" ] || continue; b=$(basename "$ck" .safetensors)
      [ -f "$OUT/$tag/$n@$b.json" ] && continue
      missing=0; for sd in "$cell"seed*/; do [ -f "$sd$b.safetensors" ] || missing=1; done
      [ $missing = 1 ] && continue
      echo "[$(date -u +%H:%M)] ckpt score $tag/$n@$b"; python3 -u heracles/score.py --runs "$cell" --ckpt "$b.safetensors" --out "$OUT/$tag/$n@$b.json" > "$OUT/$tag/$n@$b.log" 2>&1; did=1
    done
  done
  if [ $did = 0 ] && grep -q "ALL DONE" "$LOG" 2>/dev/null; then echo "CKPT DONE"; break; fi
  [ $did = 0 ] && sleep 120
done
