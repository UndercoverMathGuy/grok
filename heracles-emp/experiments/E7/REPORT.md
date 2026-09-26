# E7 — plateaus in q: REPORT

Written 2026-09-25 ~18:47 IST. PREREG.md (stamped 17:55:06 IST) frozen; no amendments.
All numbers: `experiments/E7/analysis/analyze_E7.py` -> analysis/summary.json, per_run.csv; controls: `experiments/common/controls.py` -> analysis/controls.json. Full tables: REPORT_numbers.md (`experiments/common/write_reports.py`).

## What was run
Product nu = 4, p = 37, 16 values of s = 4/q in [1, 2] (mid-plateau points of the exact switch points, predictions.json), seeds 0, 1: **32 runs, 19 ok, 13 not_converged (margin > 0), 0 diverged/crashed** (RUNS.csv). Recipe FROZEN-2026-09-25-1755 (lr = 0.28 p, lt = 3e-3, 150k steps). Pod mceuqia93fkz9s (RTX 4090).
Exact programme at p = 37 (vertex cache): 11 switch points in [1, 2] (1.097, 1.122, 1.144, 1.180, 1.194, 1.208, 1.244, 1.345, 1.402, 1.627, 1.809); 12 plateaus with K* = 18, 15, 14, 13, 12, 10, 9, 6, 6, 5, 4, 3.

## Result: MISMATCH (the plateau prediction is not borne out by the trained networks)
| group | n | best-matching plateau = predicted plateau | K(thr10) = K* | K within 1 | count R2 (identity) | count R2 (fitted) | Pearson | Spearman | OLS slope | mean signed K error | median TV(w, w*) | median cos(yhat, y*) | median gamma / class-IF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ok | 19 | **2/19 (0.105)** | 0.263 | 0.421 | 0.187 | 0.605 | 0.778 | 0.606 | 0.548 | -0.32 | 0.530 | 0.576 | 0.664 |
| not converged | 13 | 4/13 (0.308) | 0.000 | 0.231 | -0.082 | 0.626 | 0.791 | 0.848 | 0.632 | +2.46 | 0.469 | 0.624 | 0.250 |

- Counts move with s in the predicted direction (Pearson 0.78, Spearman 0.61 on ok runs) but do not sit on K* (exact 26%); at s <= 1.23 trained counts are BELOW K* (e.g. s = 1.05: 15 and 12 vs 18; s = 1.16: 9 and 7 vs 13), at s >= 1.6 seed 0 lands on 7-8 frequencies vs K* = 3-5 (junk-heavy basin).
- Budget shares do not match the predicted vertex: median TV 0.53 (ok runs). Nulls: random allocation on the predicted support size, median TV 0.35-0.62 across s; random-init networks 0.26-0.77. **The trained networks are not closer to y* than the random-allocation null at most s.**
- Within-plateau vs adjacent-plateau spread of the sorted budget-normalised amplitudes: 0.112 vs 0.101 (ok runs): no detectable jump at the predicted switch points; the "piecewise constant with jumps" signature is not visible at this resolution.
- Only s = 1 (q = nu, dense) matches: K = 18 = K* (seed 0, TV 0.087).
- Ceiling (theorem): max gamma / (B_p Gamma_s) over runs = 0.816 <= 1 (no violation).

## Controls
- Positive (nu-block networks at every s): all 16 exact (K = K*, TV < 1e-6, gamma / class optimum = 1 to 1e-6).
- Nulls: listed above (controls.json).

## Scope and caveats
- Claim type: the switch points are exact statements about the programme (theorem-level); whether trained networks follow them is a measurement, and here the measurement is negative for p = 37, nu = 4 under this recipe.
- 13/32 runs not converged under the declared criterion (mostly gamma still drifting); they are reported separately above and do not change the conclusion.
- Two seeds per s: per-s statements are weak; the sweep-level statements use all 19 ok runs.

---
Post-hoc re-scoring (Amendment A1, 2026-09-25 23:48 IST): see REPORT_A1.md and AMENDMENTS.md. This report is unchanged above this line.
