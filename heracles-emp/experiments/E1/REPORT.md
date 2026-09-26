# E1 + E1b — ansatz validity (count and shares vs the share programme): REPORT

Written 2026-09-25 ~19:05 IST. PREREG.md (stamped 17:55:06 IST) frozen; no amendments.
Numbers: `experiments/E1/analysis/analyze_E1.py` -> analysis/summary.json, cells.csv, per_run.csv (every metric, every run); controls `experiments/common/controls.py` -> analysis/controls.json. Complete tables (all cells, every count definition, every metric): **REPORT_numbers.md** (`experiments/common/write_reports.py`).

## What was run
Product networks, frozen recipe FROZEN-2026-09-25-1755; grid per PREREG: nu = 3 (p 23..113, s 1.25, 1.5, 6 seeds), nu = 4 (p 17..59, s 1.25, 1.5, 2, 4 seeds), nu = 5 (p 11, 13, 17, s 1.25..2.5, 4 seeds). 180 run directories: **126 ok, 46 not_converged, 8 diverged** (all 8 at nu = 4, p = 59: the lr = c p rule does not transfer to p = 59).
Incident: 48 nu = 3 runs (p = 23, 31, 43, 83) were trained but LOST when their pod was deleted mid-pull (my error; RUNS.csv status 'invalid', unverified partial files in lost_P4_unverified/, never used); they were rerun under ids `<id>_r1` with the identical config and are the ones analysed.
Predictions: exact (vertex cache) for p <= 37; best-found (not certified) for p = 43 (s 1.25, 1.5), 59, 83, 113; p = 43, s = 2 exact by certified K cap. Runner-up gaps are listed per cell (as small as 2.5e-5 at p = 31, s = 1.25).

## Headline (primary = 126 ok runs; not-converged runs reported separately in REPORT_numbers.md)
**The ansatz is NOT supported at the share level, and only partly at the count level.**
- Declared per-run match (K(thr10) = K* AND TV(w, w*) <= 0.10 AND cos(yhat, y*) >= 0.95): **10.3% of ok runs** (0% of not-converged).
- Declared per-cell support (medians): **1 of 36 cells** with ok runs (nu = 4, p = 17, s = 2: K 3/3, TV 0.04, cos 0.99). 3 cells have no ok run (nu = 4: p = 43 s = 1.5; p = 59 s = 1.25, 2).
- Counts vs K* (all ok runs; every definition in REPORT_numbers.md): thr10 exact 0.341, within +-1 0.659, R2 (identity) 0.385, R2 (fitted) 0.467, Pearson 0.683, Spearman 0.759, Kendall 0.625, OLS slope 1.36 [1.10, 1.62], mean signed error +0.94. Other definitions: thr20 exact 0.333 / R2id 0.634 / slope 1.01 [0.88, 1.15]; pr_amp R2id 0.579; mass_95 exact 0.357.
- Shares (ok runs, median [10th, 90th pct]): TV(w, w*) **0.418** [0.104, 0.615]; cos(yhat, y*) 0.703 [0.512, 0.968]; amp R2 identity -0.02 [-1.23, 0.88]; amp R2 fitted 0.365; amp Pearson 0.596; Spearman 0.437; Jaccard(thr10) 0.50.
- Margin: gamma / class optimum (nu-block value) median 0.716 [0.351, 0.952]; gamma / universal ceiling median 0.417, **max 0.732 <= 1** (no theorem violation).
- Nulls (controls.json): random-allocation null median TV 0.52 (median over cells), random-init networks median TV 0.64. Trained networks (0.42) beat both nulls on shares, but by a modest margin.

## Where it works and where it fails (cells.csv)
- Good agreement only at the smallest p and the largest s: nu = 4, p = 17 (s 1.5: TV 0.21, cos 0.90; s 2: TV 0.04, cos 0.99), nu = 5, p = 11 (s 1.5, 2, 2.5: K = 2 = K*, TV 0.07-0.12), nu = 3, p = 23, s = 1.25 (TV 0.11, cos 0.96, but K 6 vs 7).
- **Systematic mismatch at larger p: trained networks use MORE frequencies than K* and spread the budget**: nu = 3, p = 113: K = 20 vs 10 (s 1.25), 15 vs 6 (s 1.5), TV 0.61-0.65; p = 83: 11 vs 10, 9.5 vs 6; p = 59, s 1.5: 8.5 vs 5. nu = 4, p = 59: K 10-13 vs 4-8 (and 8/12 diverged).
- **At nu = 5, p = 13 trained networks use FEWER frequencies** (K 1-3 vs 2-6).
- Several near-tie cells (runner-up gap < 1e-3) cannot distinguish the optimum from its runner-up; this does not explain the large-p mismatches (K off by 2x).

## E1b (equal s across nu)
No (p, s) meets the declared criterion (within-s TV / between-s TV < 0.5): ratios 0.53-1.15 at the 9 shared (p, s) (p = 17: nu 4/5; p = 23, 31, 43, 59: nu 3/4). No s is shared by all three nu at a common p (nu = 5 was only runnable at p <= 17; nu = 3 starts at p = 23). Partial support: at several points the cross-nu spread equals the seed noise within a cell (p = 31, s = 1.5: 0.093 vs 0.095; p = 31, s = 1.25: 0.104 vs 0.097; p = 17, s = 1.25: 0.113 vs 0.126), while different s at the same (p, nu) differ more (0.15-0.26). Count differences: within-s mean |dK| 0.0-1.8 vs between-s 0.7-2.9.

## Controls
Positive: nu-block networks at every E1 cell's y* score perfectly under the same pipeline (all 39 cells: K exact, TV < 1e-6, gamma / class optimum = 1 to 1e-6). Nulls: above.

## Scope
Ceiling and class optimum are theorems (thm:universal, thm:cyclic) and are respected. That trained, interfering networks follow y* is a measurement; under this recipe it holds only at small p and large s. Not-converged runs (46) show the same pattern with larger errors (thr10 exact 0.04, TV 0.51).

---
Post-hoc re-scoring (Amendment A1, 2026-09-25 23:48 IST): see REPORT_A1.md and AMENDMENTS.md. This report is unchanged above this line.
