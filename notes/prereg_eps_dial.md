# Prereg: the Adam-eps dial (2026-09-04, written BEFORE any sweep run)

Question: is Adam's eps a lever on committee size K, via the sign(l_inf) -> GD(l2)
regime switch, and can WHEN the switch occurs be predicted forward, with no
constant fitted to any K histogram?

Driver `cloud/sweep_eps.py`, instrument `tensor_tap` in
`cloud/train_semifinal_torch.py` (per-tensor |grad| / sqrt(v) quantiles every
100 epochs), reader `nudial/eps_chronology.py`. Vanilla legacy recipe, p=113,
frac 0.3, lr 1e-3, wd 1.0, full batch, 50k epochs; ONE knob: adam_eps in
{1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4}; 24 seeds per cell, the
SAME 24 seeds as runs_sweep/base (paired). Stable-f32 loss throughout (immune
to the fp32 softmax-absorption confound at 16-17 nats by construction).

## What is already known (measured today on existing runs, before the sweep)

`notes/eps_chronology_torch_p113.json` (8 natural eps=1e-8 runs, 1000-epoch
checkpoints, full-batch gradient Adam sees):

- Train CE reaches 3e-6 by epoch 1000 and 3e-7 by epoch 2000; from epoch 2000
  on, 100% of coordinates of every tensor have |g| < eps. Grokking (5-13k),
  cleanup and committee lock (2.7k-35k) all happen in the GD regime
  (|g| << eps, effective GD rate lr/eps). The sign phase IS the memorization
  phase and nothing else.
- Therefore the "s = (degrees crossed by lock)/2" rule is DEAD before the
  sweep: it gives s=3 -> K=3 at eps=1e-8 for all 8 runs; observed mode 4.
  Recorded as pre-falsified; it is not a prediction of this prereg.
- Reference sign-phase trajectory (`notes/prereg_eps_dial_crossings.json`,
  two seeds at eps=1e-12, 2500 epochs, CPU, tap every 25 epochs): median |g|
  per tensor decays exponentially at 0.42 decades / 100 epochs, identical
  between seeds to the snapshot. Pre-crossing dynamics do not depend on eps
  (Adam's denominator is eps-free while sqrt(v) >> eps), so this ONE
  trajectory predicts the switch epoch for every cell.

## Frozen predictions

P1 (WHEN the switch occurs; forward, zero fitted constants). Epoch at which
the median |g| of each tensor first drops below the cell's eps:

| eps   | W_in/W_out | W_E, W_V, W_O, W_U |
|-------|-----------:|-------------------:|
| 1e-4  |        0   |              150   |
| 1e-5  |      225   |              325-350 |
| 1e-6  |      450   |              575   |
| 1e-7  |    675-700 |              800-825 |
| 1e-8  |      925   |             1025-1050 |
| 1e-9  |     1150   |             1275-1300 |
| 1e-10 |  1375-1400 |             1500-1525 |
| 1e-12 |  1850-1875 |             2000-2025 |

Call: per cell, the across-seed median crossing epoch (tap resolution 100)
lands within +-100 epochs of the table for every signal tensor, and the
order W_in/W_out before W_E/W_V/W_O/W_U (gap 100-150) holds in every cell.
Slope: +235 epochs per decade of 1/eps.

P2 (floor law, zero fitted constants): final train CE per cell scales as
eps^1: geometric-mean floor = (9.4e-8 at 1e-8 base) x (eps/1e-8), log-log
slope 1.00 +- 0.15 across grokked cells. Existing points: 1e-6 -> 1.1e-5,
1e-10 -> 8e-10 (flight), consistent.

P3 (K direction; the falsifier of the whole idea). Among runs that grok
(test acc >= 0.99 at 50k), committee size (committee_from_coeffs on final
spectra, same instrument everywhere) is NON-INCREASING in eps:
  mean K over {1e-12, 1e-10, 1e-9}  >  mean K at 1e-8  >  mean K over
  {1e-7, 1e-6, 1e-5, 1e-4}, one-sided Mann-Whitney low-vs-high pooled
  p < 0.05, effect >= 0.3 in mean K.
NULL to beat (the p-rigid seed story, and the field's "eps ~ 0" consensus):
every cell's histogram equals base {3:4, 4:15, 5:5}; no trend.
This prereg does NOT claim step positions of a K staircase: the derivation
that produced them (crossed-degrees-at-lock) is dead (above). If P3 holds,
the mechanism to develop is sign-phase SEEDING: amplitude-blind growth during
memorization sets how many circuits sit within threshold of the leader at
the switch; cleanup then prunes in the GD regime with barriers. If P3 fails
(flat), eps is not a K lever and the +1 is not an eps-switch effect.

P4 (secondary, no direction called): median grok epoch per grokked cell
within [0.5x, 2x] of base median 7000 (the switch moves by only ~235
epochs per decade; the GD-regime cleanup balance g/eps = wd*w is
eps-invariant since g scales with eps).

P5 (reported, not called): grok success rate per cell. eps=1e-4 runs at an
effective GD rate lr/eps = 10 in the GD regime and may fail or oscillate;
any cell with < 12/24 grokked is reported as no-data for P3.

## Scoring

Paired by seed with runs_sweep/base. P1 from the tap (median over seeds).
P2 from metrics.json final train CE. P3 as stated; also report per-cell
histograms, mean K, and the pooled trend (Spearman of K vs log eps). All
runs kept whatever comes out; no seed dropped by outcome.
