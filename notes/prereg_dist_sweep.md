# Prereg P-N14: the free-energy distribution P(K) as a predictive model (2026-08-29)

Committed BEFORE any sweep run exists. Model, with every number frozen here:

  P(K) ~ C(nf, K) * gamma(K)^beta
  gamma(K) = M_K(p) / K^(nu/2)     best achievable margin at committee size K
  nf = (p-1)/2                     number of available frequencies
  beta = 16.02                     FROZEN, fitted once on the existing 69-run
                                   vanilla cohort (notes/nudial_results round 5)
  M_K(p): notes/mk_tables_sweep.json (computed 2026-08-29, before this sweep)
  nu: 6 for the ReLU transformer (measured per-tensor on Nanda's artifact);
      10 for the Quad transformer (degree calculus: activation input has
      degree 4 = E+OV+W_in, squaring doubles it, +W_out +W_U)

Recipe: vanilla legacy (Nanda) throughout — train_semifinal_torch Config
defaults; ONE knob varies per cell. Driver cloud/sweep_dist.py, SEED_DRAW
20260829. Cells and FROZEN predictions: notes/prereg_dist_sweep.json.

Headline registered calls:
  1. wd-INVARIANCE: wd0.3 and wd3.0 histograms statistically identical to
     base (P(K) has no wd term).
  2. frac-INVARIANCE: f0.2/f0.4/f0.5 identical to base (no frac term).
  3. p97: mode flips DOWN to 3 = K* (the +1 disappears; P = .52/.40 on 3/4).
  4. p127: mode 4 with heavy 5-shoulder (.15/.56/.25 on 3/4/5).
  5. p211: K* moves to 4, mode to 5 (.23/.43/.18 on 4/5/6, broad tail).
  6. quad: committee collapses to K=2 (P(2)=.996), conditional on grokking
     (never trained before; if <5/24 grok the cell is reported as no-data).

Scoring (on grokked runs, final test acc >= 0.99, committee_from_coeffs on
final spectra — the identical instrument used everywhere):
  Primary: per-cell and pooled cross-entropy (mean -log P(K_obs), nats)
  under the frozen P(K), K truncated to 1..12 (instrument cap; P(K>=13)
  recorded per cell in the JSON).
  Baselines: (a) uniform over K=1..12; (b) transfer-null = existing 69-run
  baseline empirical histogram, Laplace-smoothed (the "nothing ever
  changes" model — the frozen model must beat it at p97/p211/quad to claim
  p-scaling); (c) saturated per-cell empirical (floor); (d) secondary
  tier: post-hoc single-beta refit per cell (shape quality).
  Per-cell multinomial G^2 with parametric-bootstrap p (no refit, frozen model).
