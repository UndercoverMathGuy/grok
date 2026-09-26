#!/bin/bash
# Regenerate predictions, runs and analysis for E3 from scratch (NORMS B11). Training runs on GPU pods only.
set -e
cd "$(dirname "$0")/../.."
# 1. programme predictions beyond the vertex cache (pod CPU): python experiments/common/predict.py experiments/common/pred p:s ...
#    (the exact list is in experiments/common/logs/pred_*.log); switch points: python experiments/common/switches.py ...
uv run python experiments/common/predictions.py E3
# 2. grid from the frozen recipe (experiments/common/RECIPE.json)
uv run python experiments/common/build.py E3
# 3. launch on pods (edit hosts in the plan file): python experiments/common/launch.py experiments/common/plan_main.json
#    (+ plan_P5_rebalance.json / plan_P6_rebalance.json / plan_P8_rebalance.json and configs/grid_reruns.json for the reruns)
# 4. pull + verify: python experiments/common/pull_fast.py POD HOST PORT E3
# 5. controls (pod CPU): python experiments/common/controls.py E3
# 6. analysis (light):
uv run python experiments/E3/analysis/analyze_E3.py
uv run python experiments/common/write_reports.py
