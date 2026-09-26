# PREREG — E5: bottlenecks (analysis only, no new training)

Stamp (`date`): Fri Sep 25 18:02:00 IST 2026
Written BEFORE any E1 evaluation run finished and before any E5 analysis. Frozen from here on.

## Claim and reference
CONSEQUENCES §7: at a vertex optimum supported on K frequencies, at least K offsets tie at the minimum coverage, and their (tight) equations fix the amplitude ratios. Theorem for the programme; MEASUREMENT on trained networks.

## Data and predictions
- Trained: every E1 run; per-offset worst margins h_delta (folded, delta = 1..h; metrics.json measurement.per_offset_margin_folded, float64 on the pod).
- Prediction: the tight offsets of the aligned optimal vertex (core.programme.tight_offsets of the y_all row chosen by the declared alignment rule, core.metrics.align); their number n_tight >= K*.
- Positive control: nu-block networks at each cell's y* (controls.json): the measured near-tied set must equal the predicted tight set.
- Null: random-init networks (controls.json null_init) near-tied counts.

## Metrics
- n_near(eps) = #{delta : h_delta <= min h + eps |min h|}, eps in {0.01, 0.05}; compared with K (thr_10), K* and n_tight (signed error, exact fraction); Jaccard(near-tied set, predicted tight set); the amplitude-ratio agreement is the E1 amp.* metrics restricted to the support (reported again here).
## Match / mismatch
- Per run: MATCH if n_near(0.05) >= K (thr_10) and Jaccard(near-tied(0.05), predicted tight set) >= 0.5. Supported if >= 50% of ok runs match; every cell listed.
## Compute
Local light analysis (experiments/E5/analysis/analyze_E5.py).
