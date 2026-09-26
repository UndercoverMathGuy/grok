# AMENDMENTS — E6

## A1 — post-hoc re-scoring (2026-09-25 23:48 IST, by the orchestrator at the author's request)

**Status.** POST HOC. Defined after the registered results were seen. The frozen PREREG.md and REPORT.md are unchanged and stay the primary record. A1 results are reported separately in REPORT_A1.md.

**Reason.** The registered headline count, thr_10 (A_j >= 10% of max A), counts faint tail frequencies at large p. Median p=113 (nu=3): thr_10 gives 20 vs K*=10; K_2pct gives 11.

The registered share test compares against a single vertex y*. Many cells have near-tied optima, with runner-up gaps down to about 1e-5. At p=37, s≈1.2, the 1%-optimal set holds up to 889 vertex orbits. A different optimum can therefore score as a large TV.

**Protocol.**
1. **Energy-share counts** K_2pct, K_5pct, N_05 and N_10. These are the definitions used throughout the earlier programme (heracles/nsum.py `analyse`), reimplemented here.
2. **Programme efficiency**: eff = programme value at the measured allocation / Gamma_s(p). It uses core.programme.programme_value.
3. **Share match to the 1%-optimal vertex set.** Only for vertex-cache primes p <= 37. The alignment rule is the registered one (max cosine), and min-TV is reported as well.
   - Nulls: 1000 random allocations for eff and 50 for the set match (support K*, Dirichlet shares), plus the step-0 random-init network from controls.json.
   - Positive control: the nu-block network at y*.
   - A 5% set was tried and **dropped before reading its null**. It holds up to 62k orbits (about 1M vertices), so matching it is vacuous. Its trained-run numbers were seen; they are not reported.

Every registered metric is still reported beside the A1 metrics, including thr_20, which is registered. Bootstrap CIs are over seeds (core.metrics.bootstrap_over_seeds).

**Scripts.**
- `experiments/common/amend_A1.py` writes analysis/A1_per_run.csv, analysis/A1_summary.json and experiments/common/A1_summary.json.
- `experiments/common/write_A1_reports.py` writes REPORT_A1.md.
- Runtime: about 7 s on the laptop, reading the pulled runs only.
