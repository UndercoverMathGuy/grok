# heracles plan (2026-09-11)

Scope ("the cubby"): a + b mod p with p in {59, 71, 97, 113, 127} (the range every grokking mod-add
paper uses; nothing published above 127), the exactly homogeneous ReLU MLP family with the linear-depth
degree dial nu in {3, 4, 5, 6}, cross-entropy, and only optimizers whose geometry and penalty share one
norm for the whole run (THEORY.md sec 4). Count instrument: interface ablation (score.py). Margin
instrument: gamma_q = min margin / ||theta||_q^nu, plus gamma_2 for every run.

Predictions come from heracles/kstar.json (equal-share ladder). The theory predicts a ladder; the
registered statement per cell is (a) the mode equals K*(p, nu, q) and (b) >= 90% of grokked runs land
in the plateau {K : gamma_hat(K) >= 0.74 max}. (a) failing while (b) holds is "plateau, not point";
both failing kills the cell.

## Cohort R — recipe sweep (runs first; not a hypothesis test)

Cell: nu 4 (layout (1,1)), m 512, d = r = 128, p 113, frac 0.5, data_seed 911, 8 seeds, 20k epochs,
spectra every 200. Grids in jobs.py (656 runs). Purpose: for each optimizer class find the recipe that
(1) groks in >= 7/8 seeds within 20k epochs (test acc >= 0.99), (2) has finite weights, (3) among those,
the highest median gamma_2 (q = 2 classes) or gamma_q (others). Ties broken toward smaller wd (closer
to the Wei limit). The chosen recipe per class is frozen in RECIPES.md before cohort H is launched.

Expected from the clean cohort (nudial): gd lr 1 wd 1e-3 groks 8/8 at K = 4 = K*(113, 4, 2), gamma_2
about 3.5e-6 (0.54 of the constructed ceiling 6.39e-6), clean fraction 0.84; heavy-ball lr 0.1 wd 1e-3
groks 8/8, gamma 0.64 of ceiling at 20k and rising. Nothing is known for lq, sign, or sgd at these
settings; the grids are guesses scaled from gradient magnitudes and may need one more pass.

## Cohort H — the main sweep (REGISTERED 2026-09-12, before launch; jobs.py cohort_H)

Context: the R pod died overnight on 09-12 (account credit ran out). scores/R and scores/Rsides survived; R2
(the l3 / l4 re-grid) and every raw R run were lost. Stage 1 below retrains R2 and a provenance slice of R
(the 14 RECIPES.md cells, cohort_Rkeep) before the H cells. Hardware: whatever secure card is in stock
(5090 preferred; L40S 48 GB or RTX PRO 4500 32 GB as fallbacks), one pod, runs on a network volume so a
termination is resumable (jobs.py skips runs with spectra.npz). Scoring runs on the pod alongside training
(score_loop.sh, full instrument, checkpoints included); one pull at the end.

H1  p x nu grid, q = 2:  p {59, 71, 97, 113, 127} x nu {3, 4, 5, 6}, 20k epochs, wd 3e-4, clip 1.
    Registered primary = record lr 3, 8 seeds. lr 1 x 8 seeds rides in the same batch as the fallback:
    if lr 3 groks < 7/8 in a cell, that cell is read from lr 1 (stated in the table). 320 runs.
    Prediction table (K*, plateau74) from kstar.json:
      nu 3: 6, 5, 6, 6, 7   (plateaus 3..12 / 4..12 / 4..12 / 4..12 / 4..12)
      nu 4: 4, 4, 4, 4, 5   (3..7 / 3..7 / 3..8 / 3..8 / 3..8)
      nu 5: 3, 3, 3, 4, 4   (2..5 / 3..5 / 3..5 / 3..6 / 3..6)
      nu 6: 3, 3, 3, 3, 3   (2..4 / 2..4 / 3..4 / 2..5 / 3..5)
    Given R's count-vs-wd finding, the registered statement per cell is: mode in {K*, K* + 1} and >= 90%
    of grokked runs in plateau74. Kill: nu-6 mode outside {3, 4} at >= 2 primes, or plateau < 90% at >= 3 cells.
H2  wd x frac, p 113, nu {4, 6}, record recipe: wd {1e-3, 3e-4} x frac {0.3, 0.7} x 8 seeds, 60k epochs with
    a 20k checkpoint; wd 1e-3 is read at the 20k checkpoint and wd 3e-4 at 60k (the 1/(lr wd) budget rule).
    frac 0.5 is read from H3's 20k / 60k checkpoints. 64 runs. Registered statement: frac does not move the
    mode by >= 1 at fixed wd; plateau >= 90% in every cell; frac moves grok time only.
H3  The Wei limit, q = 2, record recipe, p 113, frac 0.5: wd {1e-3, 3e-4, 1e-4, 3e-5} x nu {4, 6} x 16 seeds,
    600k epochs, checkpoints at 20k / 60k / 200k. Every wd is read at its matched budget (20k / 60k / 200k /
    600k) and at every later checkpoint. 128 runs, the long pole (two batches of 64 at 600k epochs).
    Prediction (Wei et al. 4.1): gamma_2 / ceiling rises monotonically in 1/wd at matched budgets and the clean
    fraction -> 1. Registered fork on the count: (a) mode -> K* = 4 (nu 4) / 3 (nu 6) as wd -> 0, the +1 at
    wd 3e-4 being a finite-lambda effect; or (b) mode stays K* + 1 while the margin keeps rising, i.e. the +1
    belongs to the l2 max-margin landscape reachable by gradient flow. (b) kills every "+1 is optimizer-borne"
    sentence. Also read: whether the count at fixed wd moves between 20k and 600k (drift = not equilibrated).
H4  Geometry dial, p 113, nu {4, 6}: gd (q 2, from H3), lq q3, lq q4, lq q6, sign + box. Launches as stage 2
    once R2 is scored: q3 / q4 recipes = R2 winners by the RECIPES rule; q6 and all nu-6 l_q cells carry a
    3 x 3 (lr, wd) mini-grid with the same rule applied post hoc (recipe selection inside the cell is
    registered here, so it is not a free parameter of the count). Prediction at nu 6: K* = 3, 4, 6, dense,
    dense; at nu 4: 4, plateau 4..12, dense, dense, dense. Kill: the nu-6 mode is not monotone in q.
H5  Optimizer robustness at 60k, wd 3e-4, nu {4, 6}: gdm (lr 0.3, clip 1) and sgd (lr 1, B 1024) x 8 seeds;
    gd comes from H3's 60k checkpoint. 32 runs. Prediction: same mode and plateau as gd; gamma within 20%.
H6  Constructed ceilings per (p, nu): port construct_gamma to give the "of ceiling" denominator (local, later).

Stage 1 = R2 (144) + Rkeep (112) + H1 (320) + H2 (64) + H5 (32) + H3 (128) = 800 runs, 38 batches, in that
order (cheapest first, so a second termination loses the least). Estimated 12-15 GPU hours on an L40S-class
card, about $15 of the $20 on the account; H4 (~250 runs, ~1.5 h) follows from the same volume.

## Out of scope for heracles (recorded so nobody re-pitches them)

p = 1009 (outside the literature range; no memorisation phase at m 512); the log p growth claim;
D_113 (coupled pricing); AdamW / Lion as record optimizers (mixed geometry, kept as labelled controls);
Muon (spectral geometry, separate axis); S5 (needs its own trainer; after H).

## Files

kstar.py / kstar.json   ladder and K*(p, nu, q) predictions
train.py                batched trainer, constant-geometry optimizers
score.py                interface-ablation count + legacy counts + gamma_2 / gamma_q
jobs.py                 cohorts (R, R2, Rkeep, H1, H2, H3, H5, H = stage 1 chain; H4 pending R2)
score_loop.sh           pod-side continuous scoring (finals + checkpoints)
pod.sh                  deploy / launch / pull helper
scores/                 scored JSON per cohort (committed); runs/ is gitignored
