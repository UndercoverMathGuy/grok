# heracles Z plan — abelian groups Z_120 and Z_11 x Z_11 (REGISTERED 2026-09-13 12:10Z, before any run)

Why: two abelian tasks that stress the two halves of the ladder K* = argmax_K M_K / K^{nu/2} that mod-add cannot.
- Z_120 (cyclic, composite): all 59 two-dim irreps cost the same but their coverage vectors differ — a frequency k with
  gcd(k, 120) = g has zero coverage on the subgroup of order g, so committees must be chosen by coverage. The weighted
  choice is entirely on the M_K side, and the ceiling is CONSTRUCTED exactly (cosine committees, kstar_ab.py), no
  fixed-support caveat.
- Z_11 x Z_11 (torus): every irrep is a direction (j, l) and covers everything off one line; K* >= 2 by coverage alone
  and the ladder's shape differs from a cyclic group's (M_1 = 0). A test of the M_K side with a different geometry.
Cross-check: the same code on Z_113 reproduces kstar.json / construct_q.py exactly (K* 4 / 3, plateaus 3..8 / 2..5,
ceilings 6.39e-6 / 6.2e-8).

## Predictions (kstar_ab.json; equal-share ladder, q = 2; constructed ceilings at m 512, layout (1,1) / (2,2))

| group | nu | K* | plateau74 | gamma_hat(K*+1)/gamma_hat(K*) | S*_K* | constructed ceiling (argmax K) |
|---|---|---|---|---|---|---|
| Z_120 | 4 | 4 | 3..8 | 0.977 | {f2, f15, f22, f36} (all non-faithful: gcd 2, 15, 2, 12) | 5.85e-6 (K 4) |
| Z_120 | 6 | 3 | {3, 4} | 0.904 | {f9, f25, f39} | 5.92e-8 (K 3) |
| Z_11^2 | 4 | 4 | 4..8 | 0.883 | {(1,4), (3,7), (4,3), (4,10)} | 5.83e-6 (K 4) |
| Z_11^2 | 6 | 4 | {3, 4} | 0.707 | same | 5.4e-8 (K 4) |
| Z_113 (check) | 4 / 6 | 4 / 3 | 3..8 / 2..5 | 0.983 / 0.941 | as kstar.json | 6.39e-6 / 6.2e-8 |

M_K (Z_120): 0.001, 0.17, 0.88, 1.88, 2.87, 3.76, 4.88, 6.0, 6.58, 7.63 for K = 1..10.
M_K (Z_11^2): 0.000, 0.16, 0.74, 1.89, 2.60, 3.70, 4.59, 5.72, 6.55, 7.50.
Discriminating predictions: (i) Z_11^2 at nu 6 predicts K* = 4 where every cyclic group of this size predicts 3, with a
sharp top (0.707); (ii) Z_120 at nu 4 predicts a committee of NON-faithful frequencies chosen for complementary coverage
(no coprime frequency in S*_4), i.e. the network should use divisor frequencies freely and choose them by coverage.

## Cohorts (jobs.py; pod with volume; record recipe gd lr 3, wd 3e-4, clip 1; frac 0.5; m 512; data_seed 911)
- ZR  recipe check: lr {1, 3} x 8 seeds x 20k, both groups (32 runs). Selection by the RECIPES rule; expected: lr 3 wins.
- Z1  main: record x nu {4, 6} x 16 seeds x 60k (20k ckpt), both groups (64 runs).
- ZL  fixed-support ladder at S*_K for K in {K*-1, K*, K*+1, K*+2} x nu {4, 6} x 4 seeds x 60k, both groups (~64 runs):
      the measured cross-check of the constructed ceilings, same method as S5's E0.

## Registered statements per (group, nu) cell of Z1 (grokked runs)
(a) mode of K_abl in {K*, K* + 1} (the mod-add H1 reading; H3 says the +1 is the finite-wd tie-break).
(b) gamma_2 median >= 0.5 of the constructed ceiling; the ZL fixed-support best at K* within 0.7-1.3x of the construction.
(c) coverage-optimality of the chosen committee: for every grokked run, M(S_run) >= 0.74 M_K(|S_run|) where S_run is the
    counted set (the network's frequencies are a near-optimal coverage set, not a random one). Kill: < 70% of runs.
(d) Z_120: the counted committees contain non-faithful frequencies (gcd > 1) in >= 50% of runs at nu 4 (prediction (ii)).
    Kill: every counted committee is coprime-only.
(e) Z_11^2 nu 6: mode 4 (prediction (i)); kill: mode 3 with < 25% of runs at 4.
Reading rule as in PLAN.md: (a) failing with (c) holding is "plateau, not point"; (c) failing kills the weighted-choice claim.
