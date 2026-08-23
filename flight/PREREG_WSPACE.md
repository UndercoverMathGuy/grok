# Weight-space mechanism test — deletion vs outcompetition

Registered 2026-08-21, before any `wspace` run exists. Analysis script
`flight/wspace_analysis.py` committed alongside; bars below are scored by
it, unchanged, after the cohort finishes.

## Why

The flight-time law's order parameter r is the off-committee share of the
logits' energy on the cos/sin(ω(a+b)) **diagonal** family. "Junk grows in
74% of waits" is therefore a statement about other task-aligned
frequencies, not about the memorizer; and it is wd-dependent (17% at wd
0.1 → 88% at wd 3). Nanda et al. (ICLR 2023) claim the delay is weight
decay shedding the memorized solution. To contest that we must measure
*their* quantities: memorization content in the full 2D logit spectrum,
Fourier energy of W_E / W_U in weight space, and excluded/restricted loss.

## Cohort

24 runs: add113 × frac {0.3, 0.5} × wd {0.3, 1, 3} × 4 seeds (families
`wspace_f*_wd*`, META_SEED 20260820), lr 1e-3, width 512, 50k budget,
checkpoints every 250 epochs, tap every 25 epochs.

## Observables (all committee-agnostic at record time)

- 2D logit spectrum per output, softmax gauge removed, grouped by input
  frequency pair: `diag2d` (k=k′ blocks, 56), `single2d` (one input
  constant), `off2d` (k≠k′, both nonzero — the memorizer's home),
  `const2d`, `total2d`.
- `wE_energy`, `wU_energy`: Fourier energy over the token axis (57, index
  0 = constant).
- `pnorm`: squared Frobenius norm per parameter matrix (11).
- Post hoc from checkpoints, with the final committee C (same gap rule as
  every other cohort): Nanda excluded loss (train CE with every 2D
  coefficient having k∈C or k′∈C zeroed) and restricted loss (train CE
  keeping only k=k′∈C and the constant).

Wait window = [t_mem, t_gen], t_mem = first 2 consecutive train acc ≥ 0.99,
t_gen = first 2 consecutive test acc ≥ 0.9. Runs not memorizing, or not
generalizing by 50k, are reported and excluded from wait statistics.
Trends are log-slopes over the wait (per kep) and end/start ratios.

## Bars

- **P-W1 (memorizer in logit space):** fraction of waits in which `off2d`
  *increases* (end/start > 1), per wd. Deletion predicts a decrease in a
  majority at every wd; outcompetition predicts increase at wd ≥ 1.
- **P-W2 (memorizer in weight space):** fraction of waits in which
  off-committee `wE_energy` and off-committee `wU_energy` increase, per wd.
  Same predictions.
- **P-W3 (Nanda's metric):** excluded-loss trajectory through the wait.
  Deletion predicts it rises from t_mem onward (memorization being shed
  from early in the wait): the epoch at which excluded loss first exceeds
  2× its t_mem value lies in the first half of the wait for a majority of
  runs. Outcompetition predicts it stays near its t_mem value until the
  committee dominates: that epoch lies in the second half (or never).
- **P-W4 (scale artefact check):** sign agreement, per run, between the
  old diagonal-junk trend and the `off2d` trend. If diagonal junk grows
  while `off2d` falls, the earlier "junk grows" finding is reclassified as
  a re-representation / logit-scale effect, not evidence against deletion.

## Outcome mapping (committed)

The "outcompetition, not deletion" correction is retained as a paper claim
only if **P-W1 and P-W2 show increase in ≥ 50% of waits at both wd 1 and
wd 3, AND P-W3 is in the second half for a majority at wd ≥ 1**. Any other
pattern demotes the mechanism to a descriptive two-regime statement and
the paper's spine stays forecasting + predictability floor + onset. Mixed
results (e.g. logit-space growth, weight-space decay) are reported as such.

## Stage 3 (queued after this cohort, same prereg file)

`wd0`: add113 × frac {0.3, 0.5} × wd {0, 0.01, 0.03, 0.1} × 6 seeds, 150k
budget. **P-W5:** if λ has a nonzero floor, wd = 0 runs generalize within
150k in a majority of seeds at frac 0.5; if the delay diverges as the
competitor laws say, they do not. Per-wd λ estimates are fit and compared
against the intercept model λ = a·wd + λ₀ vs a power law λ ∝ wd^β with no
intercept — AIC decides, reported either way.
