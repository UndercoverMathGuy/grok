# E8 — K across p at fixed s (nu = 3): REPORT

Written 2026-09-25 ~19:06 IST. PREREG.md (stamped 17:55:06 IST) frozen; no amendments.
Numbers: `experiments/E8/analysis/analyze_E8.py` -> analysis/summary.json, per_run.csv (dump in REPORT_numbers.md).

## What was run
48 new runs (p in {11, 13, 17, 19, 29, 37, 47, 71}, s in {1.25, 1.5}, 3 seeds; RUNS.csv) plus the E1 nu = 3 runs (p = 23, 31, 43, 59, 83, 113). K*(p) exact for p <= 37, best-found (not certified) beyond. The asymptotic Theta(log p) claim is NOT testable at these p (stated in the PREREG).

## Result: counts track K* at small p only; the trained K grows much faster than K* at large p
K*/log p over the tested p: 1.60-2.67 (s = 1.25), 0.83-1.38 (s = 1.5): the programme's K* stays small (5-10 and 2-6).
| s | group | p with median K(thr10) = K* | K exact | within +-1 | R2 identity | R2 fitted | Spearman |
|---|---|---|---|---|---|---|---|
| 1.25 | ok | 5/14 (0.357) | 0.346 | 0.615 | 0.300 | 0.398 | 0.613 |
| 1.5 | ok | 3/14 (0.214) | 0.288 | 0.635 | 0.065 | 0.446 | 0.805 |

Per p (s = 1.25, ok: p, K*, median trained K, n): (11, 5, 5, 3), (13, 6, 6, 3), (17, 5, 5, 3), (19, 6, 5, 3), (23, 7, 6, 6), (29, 9, 7, 3), (31, 7, 7, 6), (37, 6, 9, 3), (43, 6, 8, 6), (47, 8, 6, 3), (59, 8, 8, 4), (71, 10, 11, 1), (83, 10, 11, 4), (113, 10, 20, 4).
Per p (s = 1.5, ok): (11, 2, 3), (13, 3, 3), (17, 3, 4), (19, 3, 4), (23, 4, 4), (29, 4, 3.5), (31, 4, 5), (37, 5, 6), (43, 5, 5), (47, 5, 6), (59, 5, 8.5), (71, 5, 9), (83, 6, 9.5), (113, 6, 15).
- Mismatch: at p >= 59 (s = 1.5) and p = 113 (both s) trained networks use 1.5-2.5x K*. Share agreement degrades with p: median TV(w, w*) 0.00-0.13 (s = 1.25) and 0.03-0.26 (s = 1.5) at p <= 19, vs 0.45-0.65 at p >= 37 (both s).
- Not-converged runs (p >= 29) follow the same pattern with larger K (summary.json).

## Controls
Positive: nu-block controls at every cell score exactly (analysis/controls.json). Null: random-init and random-allocation in controls.json.

---
Post-hoc re-scoring (Amendment A1, 2026-09-25 23:48 IST): see REPORT_A1.md and AMENDMENTS.md. This report is unchanged above this line.
