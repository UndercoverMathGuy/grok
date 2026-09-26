# E6 — dilation equivariance and uniform marginals: REPORT

Written 2026-09-25 ~18:55 IST. PREREG.md (stamped 17:55:06 IST) frozen; no amendments.
All numbers: `experiments/E6/analysis/analyze_E6.py` -> analysis/summary.json. Full dump: REPORT_numbers.md.

## What was run
Product nu = 3, p = 23, q = 2 (s = 1.5; K* = 4, h = 11; exact programme). 50 runs (RUNS.csv): **48 ok, 2 not_converged**, 0 diverged/crashed. Pairs: seeds 0-5 x relabelling u in {1, 2, 5} (init_relabel_u, core/train.py; full-batch GD, so there is no batch order to relabel). Marginals: seeds 100-131 (u = 1). Pod grbou4dlnvjo6f (RTX 4090), float32 TF32 off, 150k steps.

## Equivariance (12 pairs: 6 seeds x u in {2, 5}, all pairs ok/ok)
- Support map: the relabelled run's support (thr10) equals the mapped support {j : fold(u j) in support of the u = 1 run} in **12/12 pairs**.
- Amplitudes: max_j |A'_j - A_{fold(u j)}| / max A: max over pairs **1.4e-6** (10 of 12 pairs below 7e-8).
- Parameter drift ||theta' - relabel(theta)|| / ||theta||: median **1.3e-6**, max **4.4e-5** (seed 4, both u), after 150k float32 GPU steps. Relative difference of gamma: <= 4.0e-7.
- Reading: exact equivariance of GD holds in exact arithmetic (theorem); in float32 the trajectories stay within ~1e-6..4e-5 relative, and the learned frequency map is exact in every pair.

## Marginals (32 seeds with margin > 0; 30 ok)
- P(j active) per frequency j = 1..11 (ok runs): 0.37 0.37 0.50 0.43 0.30 0.33 0.63 0.37 0.53 0.40 0.43; E[K]/h = 0.424.
- Chi-square vs uniform (dof 10): statistic 6.77, **p = 0.75** (ok runs); 6.79, p = 0.75 (all 32). Uniformity is not rejected (with 30 runs the test has low power; this is "no detectable non-uniformity", not a proof).
- Side observation (count, also relevant to E1): K(thr10) distribution over the 30 ok runs: 4 (x14), 5 (x14), 6 (x1), 8 (x1) vs K* = 4.

## Controls
- Positive: nu-block network at y* (controls.json): exact (K = 4, TV ~ 0, gamma / class optimum = 1). Null: random-init network scored in controls.json.
- Equivariance test in the suite: tests/test_train.py::test_init_relabel_equivariance (passed locally and on the pod).

---
Post-hoc re-scoring (Amendment A1, 2026-09-25 23:48 IST): see REPORT_A1.md and AMENDMENTS.md. This report is unchanged above this line.
