# T2-D v3 — locked-model forecaster on a hyperparameter-diverse cohort

Registered 2026-08-20, before any v3 run exists.

**Model:** ridge on 20 forecast-time observables predicting the law's
residual, trained on all 316 featurized add-113 runs, weights LOCKED in
`flight/v3_model.json` (sha256 prefix `51494d321f52f587`). Feature and
prediction code shared byte-identically between training and forecast
via `flight/v3_core.py`. Held-out-cell estimate on training data:
R²(resid) 0.744, median miss ×1.23. Physical floor (replica experiment):
median ×1.18 — the model cannot beat this and is not expected to.

**Cohort:** 102 fresh runs — 17 cells × 6 seeds (families `v3_*`,
META_SEED 20260820), spanning three registered novelty tiers:

- in-dist (5 cells): (f, wd, lr, width) values all seen in training;
- new-value (10 cells): frac 0.35, wd 2, lr 5e-4/2e-3, width 256/1024,
  and novel combinations — inside the training hull, values unseen;
- extrapolation (2 cells): wd 5, frac 0.22 — outside the hull.

**Protocol:** identical to v1/v2 — train to epoch 500, forecast from
segment-1 spectra only with the locked model, commit
`flight/forecasts_v3.json`, warm-resume to 30k. Runs that generalize
before 500 are window violations (excluded, reported). Forecast
"CENSORED" iff predicted > 30k; scored correct iff the run does not
generalize by 30k.

## Bars

- **P-V3a (primary):** ≥ 60% of scoreable runs within ×1.5
  (v1: 27%; v2: 47% on their cohorts; held-out estimate ~80%,
  discounted for novel hyperparams).
- **P-V3b:** median miss ≤ ×1.35 over scoreable runs.
- **P-V3c:** Spearman(forecast, realized) ≥ 0.85 over numeric pairs.
- **P-V3d (tiered):** in-dist tier alone ≥ 65% within ×1.5. New-value
  and extrapolation tiers are reported per-tier with no bar (first
  exposure), except the registered expectation that extrapolation is
  worst — if extrapolation beats in-dist, that itself is reported as a
  surprise.
- Failure reporting: all bars reported as registered; the locked model
  is not retrained or reweighted after seeing any v3 outcome.
