# PREREG — E4: ablation bound (analysis only, no new training)

Stamp (`date`): Fri Sep 25 18:02:00 IST 2026
Written BEFORE any E1 evaluation run finished and before any E4 analysis. Frozen from here on.

## Claim and reference
CONSEQUENCES §3 (THEORY_SPEC §3 "Ablation bound"): for an OPTIMAL clean allocation, deleting frequency j leaves margin <= Gamma (1 - w_j)^s, i.e. relative margin loss >= 1 - (1 - w_j)^s (w_j = budget share). Theorem for the clean allocation / constructed networks; a MEASUREMENT on trained (interfering, not exactly optimal) networks.

## Data and predictions
- Trained: every E1 run (experiments/E1/runs, status ok primary; not_converged with margin > 0 separately). Ablation = core.measure.ablate_frequency (projects +-j out of every weight vector), raw margins, computed on the pod at the end of each run (metrics.json measurement.ablation).
- Positive control: nu-block networks at each E1 cell's y* (experiments/E1/analysis/controls.json): the bound must hold for every j in the support (theorem).
- Null: random-init networks (controls.json null_init has no ablation; the null here is the bound evaluated with a random permutation of the measured w_j across frequencies of the same run, 100 permutations).
- Prediction per (run, j): rel_loss_j >= 1 - (1 - w_j)^s, w_j = measured budget share.

## Metrics
- Fraction of (run, j) pairs with w_j >= 0.01 satisfying rel_loss_j >= bound_j - 1e-9 (primary), and for all j; per nu, per s; the distribution of rel_loss_j - bound_j; the same on the positive controls; permutation null fraction.
## Match / mismatch
- Positive controls: must satisfy the bound for every support j (else a bug is reported).
- Trained: supported if >= 90% of (run, j with w_j >= 0.01) pairs satisfy the bound; mismatch otherwise. Every violation is listed.
## Compute
Local light analysis of metrics.json (experiments/E4/analysis/analyze_E4.py).
