# E3 — q = nu dense endpoint: REPORT

Written 2026-09-25 ~18:53 IST. PREREG.md (stamped 17:55:06 IST) frozen; no amendments.
All numbers: `experiments/E3/analysis/analyze_E3.py` -> analysis/summary.json, per_run.csv; controls `experiments/common/controls.py` -> analysis/controls.json. Full dump: REPORT_numbers.md.

## What was run
48 planned runs (12 cells x seeds 0-3) at q = nu (s = 1): product nu = 3 (p = 23, 59, 113), nu = 4 (23, 43, 59), nu = 5 (11, 13, 17); power nu = 3 (23, 59, 113). Final statuses (RUNS.csv, last row): **23 ok, 24 not_converged, 1 diverged (product nu=4 p=59 seed 2), 12 crashed**. The 12 'crashed' ids are three batches that were stopped at a queue boundary BEFORE training (queue rebalancing on slow-CPU pods, see LOG.md) and were rerun in full under new ids `<id>_r1` (all 12 reruns finished). Power nu = 4, 5 were not run (optional; time).

## Result per cell (median over runs; 'ok' first, 'nc' = not converged with margin > 0)
Prediction: all h = (p-1)/2 frequencies, equal amplitudes (cor:dense). Declared per-run match: K(thr10) = h AND min A / max A >= 0.5 AND TV(w, uniform) <= 0.05.

| model | nu | p | grp | n | K(thr10) / h | min/max A | CV(A) | TV to uniform | gamma / ceiling | per-run match | cell |
|---|---|---|---|---|---|---|---|---|---|---|---|
| product | 3 | 23 | ok | 4 | 11 / 11 | 0.772 | 0.068 | 0.0195 | 0.955 | 0.75 | supported |
| product | 3 | 59 | ok | 4 | 29 / 29 | 0.776 | 0.042 | 0.0076 | 0.981 | 0.75 | supported |
| product | 3 | 113 | ok | 3 | 56 / 56 | 0.551 | 0.085 | 0.0158 | 0.956 | 0.67 | supported |
| product | 3 | 113 | nc | 1 | 56 / 56 | 0.545 | 0.113 | 0.0288 | 0.936 | 1.0 | (nc) |
| product | 4 | 23 | ok | 3 | 11 / 11 | 1.000 | 9e-6 | 3.5e-6 | 0.999 | 1.0 | supported |
| product | 4 | 43 | nc | 4 | 20.5 / 21 | 0.173 | 0.344 | 0.146 | 0.726 | 0 | **mismatch (none converged)** |
| product | 4 | 59 | nc | 3 | 26 / 29 | 0.045 | 0.555 | 0.253 | 0.610 | 0 | **mismatch (none converged; 1 diverged)** |
| product | 5 | 11 | ok | 4 | 4.5 / 5 | 0.543 | 0.239 | 0.0996 | 0.877 | 0.5 | **mismatch (median)** |
| product | 5 | 13 | ok | 1 | 6 / 6 | 0.408 | 0.321 | 0.141 | 0.537 | 0 | **mismatch** |
| product | 5 | 17 | ok | 2 | 5.5 / 8 | 9e-6 | 0.899 | 0.409 | 0.422 | 0 | **mismatch** |
| power | 3 | 23 | ok | 2 | 11 / 11 | 0.773 | 0.078 | 0.034 | 0.877 | 1.0 | supported |
| power | 3 | 59 | nc | 4 | 25 / 29 | 4e-7 | 0.528 | 0.207 | 0.683 | 0 | **mismatch (none converged)** |
| power | 3 | 113 | nc | 4 | 18.5 / 56 | 1e-7 | 1.71 | 0.672 | 0.464 | 0 | **mismatch (none converged)** |

(gamma / ceiling: product B_p Gamma_1 = B_p p/(p-1); power 2 B_p p/(p-1). Values <= 1 everywhere, no theorem violation.)

## Reading
- Product nu = 3 (all p up to 113) and nu = 4 at p = 23: dense and near-flat, as predicted, reaching 0.95-1.00 of the ceiling. Power nu = 3 at p = 23 (the attained Morwani case): dense, 0.88 of the ceiling.
- **Mismatches**: product nu = 4 at p = 43, 59 and power nu = 3 at p = 59, 113 did not converge under the frozen recipe, and their spectra are far from flat (TV to uniform 0.15-0.67; power p = 113 uses ~19 of 56 frequencies). Product nu = 5 converged at p = 11-17 but is not flat (TV 0.10-0.41; p = 17 uses ~5.5 of 8). Whether these are optimisation failures or genuine departures cannot be separated with these runs; the non-converged ones are reported as such.
- **Null caution (important)**: the count prediction alone is NOT diagnostic here. Random-init networks already have K(thr10) = h or h-1..2 at every cell (controls.json: e.g. 29/29 at p = 59, 54/56 at p = 113). Flatness separates them: random init has min/max 0.06-0.32 and TV to uniform 0.13-0.25, while the supported cells have TV 0.00-0.03.

## Controls
- Positive: nu-block product networks and the Morwani power network with flat y: all 12 cells exact (K = h, TV ~ 1e-16, gamma / ceiling = 1 to 1e-8).
- Null: random-init networks (above).
