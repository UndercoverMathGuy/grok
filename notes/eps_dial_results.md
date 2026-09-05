# Adam-eps dial: results (2026-09-04, same day as prereg notes/prereg_eps_dial.md)

192 runs, 8 cells x 24 paired seeds (the runs_sweep/base seeds), vanilla
legacy transformer recipe, 50k epochs, one 5090 (~90 min, pod deleted).
Data: runs_eps/<cell>/s<seed>/{spectra.npz (incl. tensor tap), config,
metrics}. Scored by nudial/score_eps_dial.py -> notes/eps_dial_scored.json.
All 192 runs grokked (test acc >= 0.99). No seed dropped.

## Per cell (committee_from_coeffs on final spectra; K_thr = #freqs >= 0.1 max;
## k_eff = participation ratio; switch = first snapshot with median |g| < eps)

| eps   | K hist (n=24)        | mean K | k_eff | med grok | CE floor | switch in/out, E/V/O/U (prereg) | train CE at switch |
|-------|----------------------|-------:|------:|---------:|---------:|--------------------------------|-------------------:|
| 1e-4  | {2:1, 3:8, 4:15}     | 3.58 | 2.43 | 3600 | 1.4e-3  | 0 / 300   (0 / 150)     | 4.8 (none) |
| 1e-5  | {3:6, 4:17, 5:1}     | 3.79 | 2.45 | 4900 | 1.3e-4  | 300 / 400 (225 / 337)   | 3e-3 |
| 1e-6  | {2:1, 3:6, 4:14, 5:3}| 3.79 | 2.28 | 6000 | 1.1e-5  | 500 / 600 (450 / 575)   | 3e-4 |
| 1e-7  | {3:4, 4:18, 5:2}     | 3.92 | 2.56 | 6850 | 1.0e-6  | 700 / 900 (687 / 812)   | 1e-5 |
| 1e-8  | {3:3, 4:17, 5:4}     | 4.04 | 2.56 | 6950 | 9.3e-8  | 1000 / 1100 (925 / 1037)| 3e-6 |
| 1e-9  | {3:2, 4:19, 5:3}     | 4.04 | 2.79 | 7450 | 9.6e-9  | 1200 / 1300 (1150 / 1287)| 2e-7 |
| 1e-10 | {3:2, 4:21, 5:1}     | 3.96 | 2.52 | 7750 | 1.6e-9  | 1400 / 1600 (1387 / 1512)| 9e-9 |
| 1e-12 | {3:2, 4:18, 5:4}     | 4.08 | 2.69 | 7550 | 1.6e-11 | 1900 / 2000 (1862 / 2012)| 1e-10 |
| base 1e-8 (runs_sweep) | {3:4, 4:15, 5:5} | 4.04 | | 7000 | 9.4e-8 | | |

Same-instrument 1e-8 control replicates base: mean K 4.04 vs 4.04, 18/24
seeds at the same K, floor 9.33e-8 vs 9.36e-8.

## Scoring against the prereg

- P1 (switch epoch, forward from one eps=1e-12 reference run, zero fitted
  constants): PASS in all 8 cells, every signal tensor within +-100 epochs
  (tap resolution) of the table; W_in/W_out before E/V/O/U everywhere.
  The reference eps=1e-12 and eps=1e-8 trajectories agree to 0.02 decades
  until the 1e-8 run plateaus at |g| ~ 2e-9 (the GD-regime balance).
- P2 (floor ~ eps^1): PASS, slope 0.992 over 8 cells and 8 decades.
- P3 (K non-increasing in eps; low {1e-12,1e-10,1e-9} > 1e-8 > high
  {1e-7..1e-4}): PASS as registered — low mean 4.03 (n=72) vs high 3.77
  (n=96), delta +0.26, one-sided permutation p = 0.0016; 1e-8 at 4.04.
  BUT the shape is one-sided: flat below 1e-8 (paired dK vs base
  +0.00/-0.08/+0.04), monotone above (-0.12/-0.25/-0.25/-0.46). The pooled
  Spearman over all 192 is only -0.10 because the flat side dominates.
- P4 (grok epoch within [0.5x, 2x] of 7000): PASS (3600..7750); it is
  itself monotone in eps on the high side and saturates below 1e-8.
- P5: 24/24 grokked in every cell, including 1e-4 (effective GD rate
  lr/eps = 10 did not destabilise anything).

## Reading

1. Where K is decided. At eps <= 1e-8 every coordinate is below eps by
   epoch ~1100-2000 with train CE already at its floor; grokking (5-13k),
   cleanup and lock all happen in the eps-dominated regime, where Adam is
   exactly plain GD with rate lr/eps on gradients whose scale is set by eps
   -> the cleanup dynamics is eps-invariant, and so is the committee
   (identity: paired Jaccard vs base 0.90 at 1e-9, 10/24 identical sets at
   1e-10). The sign (l_inf) phase is the memorisation phase and nothing
   else. The "crossed degrees at lock" pricing rule was pre-falsified on
   existing runs (notes/eps_chronology_torch_p113.json) and is dead.
2. What eps does move. K falls only when the switch cuts into memorisation
   itself (train CE at switch >= 1e-5): the tail of memorisation then runs
   as rich-get-richer GD instead of amplitude-blind sign descent, the seed
   narrows, identity reshuffles (Jaccard 0.60 at 1e-4, 7/24 strict subsets
   of the base committee), grokking comes earlier, and the histogram slides
   toward K* = 3 (8/24 threes at 1e-4 vs 3-4/24 at <= 1e-8). This is the
   same direction as the round-3 plain-GD cohorts (at/below K*). So the
   optimizer's contribution to the extra member is sign-descent
   memorisation; the cleanup that fixes K is optimizer-independent.
3. The extra member is lifted at the grok takeover, not carried in. In
   base runs the coherent count (freqs within 0.15 of the leader) is 3 for
   the 2-3k epochs before grok and 4 at grok (18/24 exact at grok+1000);
   the last member joins within ~500 epochs of grok in 2/3 of runs. Its
   identity is NOT coverage-greedy (argmax coverage completion of the core
   hits 1/24, median rank 14/53) — identity is the init ticket, timing is
   the takeover. The sign-phase coherent spectrum is flat (15-30 freqs
   within threshold at epoch 300-800) and does not predict K.
4. Oddities to keep: one eps=1e-4 run (s74361) is a genuine 2-frequency
   committee {27, 3} (amps 1.0/0.46, next 0.016) at full test accuracy;
   one eps=1e-6 K=2 reading is a gap-detector artifact (third member at
   0.145). k_eff rises mildly with 1/eps (2.3 -> 2.7): amplitude sharing
   inside a fixed committee.

## Status for the paper

- Dead: "eps is an implicit-bias dial on committee size" as a headline. eps
  is negligible for structure once memorisation completes under sign
  descent, which agrees with Zhang-Zou rather than contradicting them.
- Solid, small: forward-predicted switch chronology (P1), floor law (P2),
  and the one-sided K response tied to memorisation-under-sign-descent.
  One figure: K and grok epoch vs train CE at the switch.
- The +1 question is now sharper: the count is the halting point of the
  GD+wd cleanup cascade from a memorised, spectrally flat state, with one
  recruitment at takeover; the recruit's identity comes from init. The
  open item is a reduced amplitude-cascade model (GD + wd, coverage table
  M_K, flat seed at measured amplitude, degree nu) that must halt at 4 for
  p=113/nu=6 and give 12/5/4/4 across the MLP ladder — validation data is
  already on disk (runs_nudial, runs_sweep, runs_eps).
