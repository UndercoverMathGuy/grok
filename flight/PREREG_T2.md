# Tier-2 prereg — flight-time law causal & forecast cohorts

Registered 2026-08-20, before any tier-2 run exists. Trainer:
`cloud/train_flight_t2.py` (META_SEED 20260820; seeds drawn by crc32
family, kept whatever comes out). Law under test, constants frozen from
the 455-run evidence sweep (`notes/lambda_evidence.json`):

> t_gen = t_mem + max( log(r_mem / r*) / lambda , floor )
> r* = 10^-2.25 (add113 task median), floor = 132 ep,
> floor branch iff r_mem <= e * r*.

r(t) = 1 − committee share of logit-spectrum energy; committee = largest
log-gap in the energy spectrum at the reference snapshot; t_mem / t_gen =
first sustained (x2 snapshots) train_acc >= 0.99 / test_acc >= 0.9.

All pauses resume WARM (model + Adam moments + step offset via
`save_optim=True`). The only cold restarts in tier-2 are the T2-B
optimizer-reset control arms (registered separately when T2-A/B launch).

## T2-D — per-seed forecasts across the dial (the skill test)

**Forecast arm.** 15 fresh runs: cells (frac, wd) in {(0.2,3), (0.25,1),
(0.3,1), (0.3,3), (0.4,1)} x 3 seeds — chosen to span predicted t_gen
~1.5k–22k and both wd values; every cell's historical min t_gen >= 725.
Protocol: train to epoch 500; compute each run's forecast from segment-1
spectra only (committee at ep 500; lambda fit on [t_mem, 500]); write
`flight/forecasts_t2d.json`; commit; warm-resume to 30k untouched.
The forecast code refuses to overwrite, and segment 2 refuses to start
without the forecasts file.

- **P-F1 (primary):** >= 11/15 forecasts within x1.5 of realized t_gen
  (|log10 miss| <= 0.176). Runs forecast CENSORED score correct iff the
  run does not generalize by 30k; a numeric forecast on a run that
  censors scores wrong (and vice versa).
- **P-F2:** Spearman(forecast, realized) >= 0.8 over the numeric pairs.
- **P-F3 (window):** any run that generalizes before epoch 500 is
  excluded from P-F1/P-F2 and reported as a window violation (expected
  0/15 by design).

**Floor arm (ex-ante band calls, no pause).** 8 fresh runs: (0.6,1),
(0.7,1), (0.9,1), (0.9,3) x 2 seeds, 10k budget.
- **P-F4:** >= 7/8 land on the floor branch: t_gen − t_mem <= 300 ep.

## T2-C — the product law and the Adam knob

36 runs: lr {3e-4, 1e-3, 3e-3} x wd {0.3, 1, 3} x 4 seeds, add113 f0.3,
30k. lambda scored on the full-wait window (as in the evidence sweep).
Cells with lr*wd product <= 3e-4 are expected mostly censored at 30k —
their (partial) lambda is still measured and their censoring is itself a
law prediction.

- **P-C1 (product collapse):** regression log lambda ~ log(lr*wd) over
  generalized runs has slope in [0.7, 1.3]; the exactly matched-product
  pair (lr 1e-3, wd 3) vs (lr 3e-3, wd 1) has median-lambda ratio in
  [1/1.3, 1.3].
- **P-C2 (dissociation carry-over):** t_ident (largest-gap instrument)
  varies by < 2x across the grid while t_gen spans > 20x.
- **P-C3 (eps, exploratory):** eps {1e-6, 1e-10} x 4 seeds at lr 1e-3
  wd 1. No kill bar; direction and size of lambda / t_gen / t_ident
  shifts reported. Motivated by the homeostat (floor ~ eps) and the
  kappa term.

## T2-E — closing the Deff loop (fixes the confounded task table)

24 runs: {add113, add96, sub113} x frac {0.3, 0.6} x wd 1 x 4 seeds,
50k budget (censoring symmetric across tasks, unlike the degrok2
15k/30k mix), save_every=500 for sparse checkpoints.

- **P-E1:** the frac-matched kappa*Deff table preserves the ordering
  add113 > add96 > sub113 at f0.3 (medians).
- **P-E2 (the loop):** an independent per-run efficiency measurement
  (margin-per-norm gap between committee and junk subspaces from the
  epoch-500..2000 checkpoints; analysis script to be committed before
  unblinding) predicts per-run lambda with Spearman >= 0.5 pooled
  within-task. Fail = Deff remains a lump; the treadmill keeps its
  lr*wd evidence but loses the task-fingerprint claim.
- **P-E3:** sub113 f0.3 at 50k resolves its censoring status; the law's
  per-run predictions for those runs are recorded at their t_mem+500
  (same rule as T2-D) and scored.

## Failure reporting

Every P-bar above is reported pass/fail as registered. Known model
limitations carried into this prereg (not escape hatches): lambda is
non-stationary near t_gen (accelerating cleanup) — this biases T2-D
forecasts toward over-prediction, and the x1.5 band was chosen with
that bias known; r* task-dependence means T2-E sub/add96 runs use their
own task r* medians (10^-1.73 / 10^-1.94) for P-E3-type predictions.
