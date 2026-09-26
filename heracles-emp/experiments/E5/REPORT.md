# E5 — bottleneck offsets (analysis only): REPORT

Written 2026-09-25 ~19:05 IST. PREREG.md (stamped before E1 results) frozen; no amendments.
Numbers: `experiments/E5/analysis/analyze_E5.py` -> analysis/summary.json, per_run.csv (dump in REPORT_numbers.md). Data: every E1 run with margin > 0; per-offset worst margins in float64 (metrics.json).

## Result: MISMATCH
Prediction: at a vertex optimum on K frequencies, >= K offsets tie at the worst margin, and they are the tight offsets of the aligned optimal vertex.
| group | n | per-run match (n_near(5%) >= K AND Jaccard(near-tied, tight) >= 0.5) | frac n_near(5%) >= K(thr10) | median Jaccard (5% / 1%) | n_near(1%) = n_tight exactly | mean(n_near(1%) - n_tight) | mean(n_near(5%) - K) |
|---|---|---|---|---|---|---|---|
| ok | 126 | **0.016** | 0.095 | 0.183 / 0.125 | 0.008 | -3.83 | -4.0 |
| not converged | 45 | 0.022 | 0.022 | 0.125 / 0.0 | 0.022 | -3.8 | -6.27 |

- Trained networks have FEWER near-tied worst offsets than the programme's tight set (about 4 fewer on average), and the tied offsets are mostly not the predicted ones (median Jaccard 0.18). Declared support threshold (>= 50% of ok runs matching) is not met.
- Consistent with E1: the trained allocations are not at the predicted vertex, so its tight equations do not bind.

## Controls
Positive: for every nu-block control the near-tied set equals the predicted tight set exactly and has >= K* offsets (summary.json 'positive_controls_all_match': true; for nu-blocks the per-offset margin equals the coverage of y*, an exact identity). Null: random-init networks are in E1/analysis/controls.json (negative margins; not meaningful for ties).
