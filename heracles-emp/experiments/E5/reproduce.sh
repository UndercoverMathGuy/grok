#!/bin/bash
# E5 is analysis-only on E1 networks (run experiments/E1/reproduce.sh first).
set -e
cd "$(dirname "$0")/../.."
uv run python experiments/E5/analysis/analyze_E5.py
uv run python experiments/common/write_reports.py
