# E4 — ablation bound (analysis only): REPORT

Written 2026-09-25 ~19:04 IST. PREREG.md (stamped before E1 results) frozen; no amendments.
Numbers: `experiments/E4/analysis/analyze_E4.py` -> analysis/summary.json (full dump in REPORT_numbers.md). Data: every E1 run with margin > 0 (ablation computed in float64 on the pods by core.measure.ablation_margins).

## Result: SUPPORTED (as a measurement on trained networks; theorem on the constructed controls)
Bound: deleting frequency j costs a relative margin loss >= 1 - (1 - w_j)^s (w_j = measured budget share).
| group | runs | (run, j) pairs with w_j >= 0.01 satisfying the bound | all j | permutation null (w shuffled) | by nu (3 / 4 / 5) |
|---|---|---|---|---|---|
| ok | 126 | **765 / 768 = 0.996** | 0.861 | 0.417 | 0.998 / 1.000 / 0.980 |
| not converged | 45 | 357 / 358 = 0.997 | 0.945 | 0.625 | 0.994 / 1.000 / 1.000 |

- Declared criterion (>= 90% of active pairs): met. The bound is loose on trained networks: median excess (rel_loss - bound) over active pairs 0.288 (5th pct 0.060, 1st pct 0.015).
- Violations (listed in full in summary.json): ok runs: E1_nu3_p43_s1.25_seed4_r1 j=19 (w 0.086, loss 0.073 vs bound 0.107); E1_nu5_p13_s1.5_seed1 j=3 (w 0.048, 0.051 vs 0.071); E1_nu5_p13_s2_seed3 j=3 (w 0.079, 0.095 vs 0.152). Not converged: E1_nu3_p113_s1.25_seed1 j=52 (w 0.018, 0.004 vs 0.023).
- 'All j' is lower (0.86) because ablating an essentially unused frequency (w < 0.01) can slightly RAISE the margin (negative loss) while the bound is a small positive number.
- The theorem is stated for the optimal clean allocation; trained networks are not optimal, so satisfaction here is an empirical observation.

## Controls
Positive (nu-block networks at every E1 cell): **185 / 185** support pairs satisfy the bound (E1/analysis/controls.json ablations). Null: the same trained runs with shares permuted across frequencies satisfy it in only 42% (ok) of pairs, so the agreement is specific to the measured shares.
