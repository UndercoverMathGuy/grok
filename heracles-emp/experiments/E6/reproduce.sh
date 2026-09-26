#!/bin/bash
# Regenerate predictions, runs and analysis for E6 from scratch (NORMS B11). Training runs on GPU pods only.
set -e
cd "$(dirname "$0")/../.."
# 1. programme predictions beyond the vertex cache (pod CPU): python experiments/common/predict.py experiments/common/pred p:s ...
#    (the exact list is in experiments/common/logs/pred_*.log); switch points: python experiments/common/switches.py ...
uv run python experiments/common/predictions.py E6
# 2. grid from the frozen recipe (experiments/common/RECIPE.json)
uv run python experiments/common/build.py E6
# 3. launch on pods (edit hosts in the plan file): python experiments/common/launch.py experiments/common/plan_main.json
#    (+ plan_P5_rebalance.json / plan_P6_rebalance.json / plan_P8_rebalance.json and configs/grid_reruns.json for the reruns)
# 4. pull + verify: python experiments/common/pull_fast.py POD HOST PORT E6
# 5. controls (pod CPU): python experiments/common/controls.py E6
# 6. analysis (light):
uv run python experiments/E6/analysis/analyze_E6.py
uv run python experiments/common/write_reports.py

# 7. Amendment A1 (post hoc, light): uv run python experiments/common/amend_A1.py && uv run python experiments/common/write_A1_reports.py
