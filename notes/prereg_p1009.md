# Prereg: p = 1009 relu-4 MLP count (2026-09-07, before any run)

Purpose. The only test where the theory's PEAK moves at fixed degree nu:
K*(p, nu=4) = argmax_K M_K(p)/K^2 grows with log p (nudial/kstar_pscale.py,
proven 08-23). At p = 113 the peak is 4 and the trained AdamW mode is 5
(K*+1, 192 finals, all five count instruments agree). At p = 1009 the peak
is 7. Whether the trained count follows the peak, stays p-rigid, or follows
an entropy model is the sharpest open discriminator (registered in
nudial_results round 8.4 and seeded_ensemble_theory sec 5; never run).

Frozen theory (nudial/kstar_pscale.json, equal-share ladder
gamma_hat(K) = M_K/K^2, free-amplitude refinement <= 2.5%):

  K      4      5      6      7      8      9     10     11     12
  M_K  0.711  1.415  2.217  3.029  3.812  4.667  5.522  6.446  7.284
  r(K) 0.719  0.916  0.996  1.000  0.963  0.932  0.893  0.862  0.818

  K* = 7 (6 is a 0.4% near-tie). Plateau sets: {K : r >= 0.74} = {5..12+},
  {K : r >= 0.84} = {5..11}. K = 4 is BELOW the plateau (0.72).

Recipe (identical to cohort A/M at p = 113 except p): relu family, nu = 4,
layout (1,1), d = r = 128, frac 0.5, wd 1, lr 1e-3, AdamW MLX semantics
(betas .9/.98, eps 1e-8, no bias correction, decoupled wd), warmup 10,
10k epochs full batch (exact, via chunked gradient accumulation), seeds =
first 16 of train_nudial.draw_seeds(32) at m = 512 and first 8 at
m = 2048. Trainer nudial/p1009.py, scorer nudial/score_p1009.py.
Grokked = final test acc >= 0.99; counts scored on grokked runs only.
Primary count = K_thr10 (|coeff| >= 0.1 x max); K_gap, K_thr05, k_eff,
K_load90 reported alongside (count_definitions audit: all agree on AdamW
MLP finals at p = 113). If fewer than 50% grok in 10k epochs the cohort is
rerun at 20k before scoring (stated, not a prediction).

Accounts on the record and what each predicts for the m = 512 mode:
  (a) landscape peak-tracking  ........ mode in {6, 7, 8}
  (b) seed law (p-rigid seed ~4 projected onto the plateau) ... mode 5
  (c) "+1 above K*" continuation from p = 113 .................. mode 8
  (d) entropy / multiplicity models (dead at p <= 211) ........ mode >= 11
  (e) count is not a cost quantity at all ..................... mode <= 4

PREDICTIONS
P1  Law 1 (plateau concentration, zero parameters): >= 90% of grokked
    m = 512 runs have K_thr10 in {5, ..., 12}. KILL of the plateau law if
    the mode is <= 4 or >= 13, or if < 75% land in {5..12}.
P2  The peak moves with p (the paper's "cost decides count" claim at fixed
    nu): median K_thr10 at p = 1009 >= 6, i.e. at least one above the
    p = 113 relu-4 median of 5. KILL if median <= 5 (the count is p-rigid;
    the seed account wins and the count claim narrows to "plateau
    membership only").
P3  Width: m = 2048 median K_thr10 within +-1 of the m = 512 median
    (count is width-invariant, as 32x at p = 113).
P4  Exploratory, not a kill: at m = 512 the train set (509k pairs) exceeds
    the parameter count (~450k), so a memorisation phase may not exist;
    at m = 2048 (~1.0M params) it may. Record for each run whether train
    acc reaches 0.99 before test acc does. If the widths differ in that
    and their medians differ by > 1, the memorisation era sets the count
    (seed mechanism); if they agree, it does not.
P5  Margin: grokked nets sit within 26% of the best K's constructed
    ceiling if the construction is available at p = 1009; otherwise report
    gamma in absolute terms and the equal-share r(K_obs) of each run.

Scoring output: notes/p1009_scored.json + a table in notes/nudial_results.md.
