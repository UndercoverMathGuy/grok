#!/bin/bash
# E4 is analysis-only on E1 networks (run experiments/E1/reproduce.sh first).
set -e
cd "$(dirname "$0")/../.."
uv run python experiments/E4/analysis/analyze_E4.py
uv run python experiments/common/write_reports.py
