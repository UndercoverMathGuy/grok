#!/usr/bin/env bash
# twin/run_gpu.sh — the whole twin pipeline, sequentially, on one GPU.
#
#   bash twin/run_gpu.sh              # full suite
#   WIDTH=80 bash twin/run_gpu.sh     # bigger batches (more VRAM, ~0.19 GB/lane)
#   STAGE=1 bash twin/run_gpu.sh      # stop after the bases
#
# WIDTH IS PART OF THE ARITHMETIC.  Gate G3 fails: a run's trajectory depends on
# the vmap batch width in the last bit, and grokking amplifies that. Every
# comparison the analysis makes is therefore within a single batch, and chunk
# widths are held constant inside each cohort. If you resume an interrupted
# cohort you MUST resume with the same WIDTH, or delete that cohort's runs.
set -euo pipefail
cd "$(dirname "$0")/.."

# cuBLAS reduction workspaces are nondeterministic without this; torch raises
# under use_deterministic_algorithms(True) if it is unset.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PY="${PY:-uv run --group convert python -u}"
RUNS="${RUNS:-runs_twin}"
MF="${MF:-twin/manifests}"
WIDTH="${WIDTH:-40}"
STAGE="${STAGE:-2}"

echo "== determinism gates (p=113, 2000 epochs; G3 is informational) =="
$PY twin/gates.py --device cuda --epochs 2000 --runs "$RUNS/_gates" || \
  { echo "HARD GATE FAILED — stop"; exit 1; }

echo "== stage 1: manifests =="
$PY twin/make_twin.py --stage 1 --out "$MF"

echo "== stage 1: BASES-A (8 x 30k) + BASES-B (8 x 15k) =="
$PY twin/train_twin.py --manifest "$MF/stage1.json" --runs "$RUNS" \
    --device cuda --width "$WIDTH"

[ "$STAGE" = "1" ] && { echo "stopping after stage 1"; exit 0; }

echo "== stage 2: manifests (needs base committees + init W_E energies) =="
$PY twin/make_twin.py --stage 2 --out "$MF" --runs "$RUNS"

# FORK first (it is 57% of the compute and answers the lock question on its
# own), then the two init-edit cohorts, then HOLD.
for cohort in fork swap dial hold; do
  echo "== stage 2: $cohort =="
  $PY twin/train_twin.py --manifest "$MF/$cohort.json" --runs "$RUNS" \
      --device cuda --width "$WIDTH"
done

echo "== analysis =="
$PY twin/analyze_twin.py --runs "$RUNS" --out twin/results
echo "TWIN SUITE DONE"
