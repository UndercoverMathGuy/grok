# Deterministic P(K): Boltzmann on the shaded margin ladder (2026-09-05)

Model (nudial/boltzmann_pk.py): P(K) = exp(beta r_s(K)) / Z over K = 1..12,
r_s(K) = gamma_s(K)/max gamma_s, gamma_s(K) = M_K(p)/K^s. Two constants
(beta, s); the shaded exponent s = nu_sel/2 is needed because the endpoint
nu = 6 ladder peaks at 3 and no monotone weight can then put the mode at 4
(seeded_ensemble_theory.md sec 3). Fit on the p = 113 pooled P-N14 cells
(base + wd0.3 + wd3 + f0.4 + f0.5, n = 120), predicted frozen elsewhere.
Score = mean -log P(K_obs) (nats); null = 69-run vanilla histogram applied
everywhere (Laplace 0.5); sat = per-cell floor.

Fit: beta = 12.7, s = 2.44 (nu_sel = 4.88; the round-8 window was 4.66-4.83).

p      n   CE_model  CE_null  CE_sat  mode_pred  P(3/4/5/6)
97    24    0.923    0.844    0.675      3       .454/.413/.102/.021
113  120    1.022    1.061    0.959      4       .193/.647/.122/.027  (in-sample)
127   24    1.065    1.079    0.918      4       .212/.532/.202/.036
211   16    1.255    1.016    0.900      4       .070/.482/.341/.079
Joint 4-prime fit: beta 13.4, s 2.42; leave-one-prime-out pooled CE 1.071
vs null 1.031. Power-law variant (r^beta): identical picture (LOO 1.056).
p = 1009 frozen prediction of this model: mode 6, P(4/5/6) = .06/.33/.35.

Verdict: the two-constant Boltzmann places the mode correctly at 113, 127,
211 and calls p97 a 3/4 near-tie (obs 18/24 at 4), but it does NOT beat
the p-rigid null out of sample: the ladder flattens toward K = 5 at p211
faster than the histograms do (predicted P(5) = 0.34, observed 2/16). Same
p-rigidity that killed the entropy models in P-N14. Deterministic, honest,
and to be reported as such: the ladder gives support and mode; the
histogram width is not a landscape quantity.
