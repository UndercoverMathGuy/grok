# RESULTS — what the empirical programme supports (top level)

Written 2026-09-25 23:50 IST by the orchestrator; revised 2026-09-26 after AUDIT.md, from the per-experiment reports. Every number is copied from the file
named beside it, and those files name the script that made it.

- **REG** is the preregistered analysis (REPORT.md). It is the primary record.
- **A1** is the post-hoc re-scoring (REPORT_A1.md and AMENDMENTS.md). It was defined after REG was seen and is
  labelled as such everywhere.
- AUDIT.md (independent; 2026-09-26 00:31 IST): 3 FAIL, 9 CAVEAT, and no numeric errors. This revision applies its safe wordings; its section 6 is the list of paper-safe statements.

**Theorems vs measurements.** The ceiling, the class optimum, the ablation bound for the optimal allocation, exact
equivariance and the switch points are theorems about the programme and constructed networks. Every statement about
*trained* networks is a measurement under the frozen recipe (full-batch GD with momentum, explicit L_{2,q} penalty,
float32 with TF32 off). Trained networks are not junk-free: clean energy fraction 0.47–1.00 over converged runs, median 0.97 (A1_per_run.csv, column energy_clean).

## Summary

| exp | claim | REG verdict | A1 (post hoc) | usable statement |
|---|---|---|---|---|
| all | the margin never exceeds the ceiling (thm:universal) | holds: max γ/ceiling 0.73 (E1), 0.82 (E7) | - | theorem, respected in every run |
| E1 | trained count follows K*(p, s) | per-run joint match 10%. thr_10: ±1 in 0.66, R² 0.39, slope 1.36, bias +0.94 | **thr_20 (registered): ±1 in 0.70 [0.60, 0.78], R² 0.63, slope 1.01, bias +0.16.** K_2pct: ±1 in 0.70, R² 0.60, slope 0.91, bias +0.02 | within ±1 of K* in 70% of converged runs under the registered thr_20 count (66% under the registered primary thr_10); the best constant predictor gets 50%, a permuted K* 36% (AUDIT). Exact in 33–34%. Choosing thr_20 as the headline is post hoc: it is the best of the 9 registered definitions (range 0.51–0.70). The pooled bias is near 0 only because errors of both signs cancel: it overcounts at ν=3, p ≥ 59 and undercounts at ν=5, p=11/13 and ν=4, p=23, s=1.25 |
| E1 | trained budget shares follow y* | TV to y* 0.42 vs null 0.52: **not supported** | efficiency (programme value of the learned allocation / optimum) **0.87** [0.85, 0.88] vs nulls 0.45 (random allocation) and 0.67 (random init). TV to the 1%-optimal set 0.29 vs null 0.43 | learned allocations reach ~87% of the programme optimum, far above nulls, but are not optimal: only 12% of runs reach ≥ 0.95. Do not claim shares match y* |
| E1b | the same s gives the same behaviour across ν | no cell met the TV criterion | median-count difference across ν at the same (p, s): 0.67 (8/9 pairs within 1) vs 0.94 across s | weak: the counts agree across ν, but the contrast is small at these p |
| E3 | q = ν is dense and flat | supported: product ν=3 (p 23, 59, 113), ν=4 (p 23), power ν=3 (p 23). Mismatch where runs did not converge (ν=4 p 43/59, power p 59/113) and at ν=5 | - | q = ν gives dense, near-flat spectra for product ν=3 (p=23, 59, 113), ν=4 (p=23) and power ν=3 (p=23), at 0.88–1.00 of the ceiling. It does NOT hold at ν=5 as a rule: 2 of 7 converged runs (both p=11) are dense and flat, the other 5 miss 1–3 frequencies or are far from flat (per-cell median TV to uniform 0.10–0.41), and is not established at ν=4 p ≥ 43 or power p ≥ 59 (no run converged). Count alone is not diagnostic; flatness is |
| E4 | ablation bound 1−(1−w_j)^s | 765/768 = 0.996 of active pairs vs permutation null 0.42 | - | supported as a measurement; bound is loose (median excess 0.29) |
| E5 | K offsets tie at the worst margin | **mismatch** (1.6%) | - | not supported on trained networks. Consistent with them sitting at ~0.87 of the optimum, not at a vertex. Theorem only on constructed networks |
| E6 | dilation equivariance, uniform marginals | supported: support map exact in 12/12 pairs, drift ≤ 4.4e-5; marginals p = 0.75 | count ±1 in 0.96; efficiency 0.91 | equivariance supported (12/12 pairs). Marginals: uniformity NOT REJECTED (χ² p=0.75, 30 runs, low power, approximate p-value). Do not write "uniform marginals supported" |
| E7 | plateaus/switch points in q (ν=4, p=37) | **mismatch**: plateau identity 2/19 | still mismatch: K_2pct ±1 in 0.42, bias −2.2; counts move with s (Spearman 0.61–0.94 across the 6 count definitions) but are compressed; efficiency 0.85 | the count moves in the predicted direction with q; exact plateaus are not resolved |
| E8 | K across p (log-p) | "tracks K* only at small p" (thr_10) | per-cell median K_2pct within ±1 of K* in 23/28 ν=3 cells of E1+E8 up to p=113 (p=113: 11 vs 10, 7 vs 6). Misses: overshoot at s=1.5, p=59/71/83 (8, 6.5, 8 vs 5, 5, 6); undershoot at s=1.25, p=29/47 (7, 6 vs 9, 8); efficiency 0.94 (E8 runs, p ≤ 71), dropping to 0.72–0.84 at p=113 | post hoc energy count: per-cell median within ±1 of K* at 23/28 (p, s) cells up to p=113; the best constant per s gets 18/28 (AUDIT). The registered thr_10 count overshoots 1.5–2.5× at p ≥ 59. K* is best-found, not certified, beyond p=37. Optimality degrades with p. Θ(log p) itself is not testable at these p |

## Reading

1. Counts are where the ansatz shows in trained networks, modestly: within ±1 of K* in 70% of runs vs 50% for the best
   constant predictor. The choice of thr_20 (a registered definition) as the headline is post hoc. The registered primary
   (thr_10) is a poor instrument at large p: it counts tail frequencies that carry < 2% of the energy.
2. Shares: GD finds good allocations but not optimal ones (median efficiency 0.85–0.94 by experiment, lower at large
   p and small s). Statements at the share level (exact y*, bottlenecks, plateau identity) do not transfer to
   trained networks under this recipe.
3. The ceiling holds in every run (349/349). The ablation bound holds for 765/768 active pairs of converged runs; on
   trained networks it is a measurement, not a theorem. Relabelled inits give the relabelled support in 12/12 pairs.
   Constructed controls are exact everywhere, and the audit independently rebuilt them at 7 cells.
4. Provenance: the pull-time sha256 lists were not kept (pull_fast.py deleted them; AUDIT FAIL 1). Current hashes are
   frozen in audit/sha256_frozen_20260926.json. Re-measuring 12 sampled runs reproduces metrics to about 1e-15, and
   retraining 6 on a fresh pod reproduces status, counts and supports (γ within 8.6e-6).

## Pointers

- REPORT.md, REPORT_A1.md, AMENDMENTS.md and REPORT_numbers.md in each experiments/E*/.
- A1 scripts: experiments/common/amend_A1.py and experiments/common/write_A1_reports.py.
- Compute ledger: COMPUTE.md. Sweep $7.67 + audit pod $0.38; all pods deleted (list-pods empty 2026-09-26).

## 2026-09-26 — Amendment A2 (post hoc): shape of the count along the sweeps
Script: experiments/common/amend_A2_shape.py -> experiments/common/A2_summary.json; paper figures: paper_figs/make_figs.py -> FIGURE_NUMBERS.json.

**q-sweep at fixed (ν, p)** (count definition thr_20; all 13 definitions in A2_summary.json):
- E1 + E8: the median count moves with K* in 30/36 pairs (4 ties), against a shuffle null of 0.38 (95%: 0.53); p = 0.001.
  - All 13 definitions: 26–34/36, each with p = 0.001.
- E7: 83/96 pairs vs a null of 0.47.
  - All 13 definitions: 64–83/96, p ≤ 0.041.
- The counts are compressed: slope 0.50, and 0.27–0.61 across definitions and the two sweeps.

**ν ladder at q = 2:** 5/7 pairs (p = 0.16; 3–7/7 across definitions). Not established.

**Same s across ν:** median counts differ by 1.12 at the same (p, s), against 1.31 across s. Does not establish "s alone decides".

**Count across p:** per-cell thr_20 within ±1 of K* at 20/28 cells, against 16/28 for the best constant per s.
