# Stage 2 results (2026-09-12) — H4 geometry dial + constructed l_q ceilings (H6 for the dial)

Scores: scores/H4 (full instrument, pod vs2z7ht3uk9nyj). Recipe per (nu, q) chosen post hoc by the RECIPES rule
(>= 7/8 grok, finite, max median gamma_q, ties -> smaller wd), as registered in PLAN.md H4. Constructed ceilings:
heracles/construct_q.py -> construct_q.json (exact product-grid K-committee nets, K = 1..12 from kstar.json plus
a fully dense 56-frequency net, optimally balanced in each l_q norm, amplitudes polished).

## H4 — count vs geometry exponent q, p 113   [analyze_H4.py]

| nu | q = 2 (gd) | q = 3 | q = 4 | q = 6 | q = inf (sign + box) |
|---|---|---|---|---|---|
| 4 predicted K* | 4 | 8 (plateau 4..12) | 12 / dense | 12 / dense | dense |
| 4 measured mode (hist) | 4-5 | 5 {4:1,5:4,6:3} | 5 {5:5,6:2,7:1} | 5 {5:3,6:3,7:1,8:1} | 6 {6:4,7:2,8:1,10:1} |
| 6 predicted K* | 3 | 4 | 6 | 12 / dense | dense |
| 6 measured mode (hist) | 4 {4:5,5:2,6:1} | 4 {3:2,4:6} | 4 {3:1,4:4,5:2,6:1} | 4 {3:1,4:6,5:1} | (sign nu 6 pending) |

- Registered kill (nu-6 count not monotone in q): FIRED. The nu-6 count is 4 at every q; q = 4 and q = 6 miss
  their predictions (6, dense) outright. At nu 4 the count rises weakly with q (4-5, 5, 5-7, 5-7, 6-7) and
  never approaches the predicted 8..12 / dense at q >= 4.
- Every l_q cell grokked 8/8 at its winner, with train CE 4e-4 (q3) to 7e-8 (q6) at nu 4 and 5e-6 to 2e-9 at
  nu 6, i.e. far more equilibrated than GD (1e-2). Not a budget effect.

## Control 1 — the optimizers reach their own-norm optima relative to each other (cross-norm matrix)

median over 8 seeds of m_min / ||theta||_q^nu, nu 4 (rows = training geometry, columns = evaluation norm):

| trained with | gamma_2 | gamma_3 | gamma_4 | gamma_6 | gamma_inf |
|---|---|---|---|---|---|
| gd (q 2) | **4.27e-6** | 3.9e-3 | 0.093 | 1.7 | 24 |
| lq q 3 | 3.66e-6 | **9.2e-3** | 0.39 | 13 | 1.6e3 |
| lq q 4 | 2.84e-6 | 9.1e-3 | **0.47** | 20 | 4.4e3 |
| lq q 6 | 2.29e-6 | 8.0e-3 | 0.46 | **24** | 1.1e4 |
| sign + box | 1.03e-6 | 3.7e-3 | 0.22 | 13 | **4.5e4** |

Diagonal dominance in every column: each geometry finds the best solution in its own norm. The miss is not the
optimizer failing to follow its geometry.

## Control 2 — constructed K-committee ceilings vs trained (construct_q.py)

| nu | q | constructed argmax_K (gamma_q) | trained gamma_q | trained / constructed |
|---|---|---|---|---|
| 4 | 2 | 4 (6.39e-6) | 4.27e-6 | 0.67 |
| 4 | 3 | 5 (1.43e-3) | 9.23e-3 | 6.5 |
| 4 | 4 | 5 (2.09e-2) | 0.472 | 23 |
| 4 | 6 | 8 (0.314) | 24.1 | 77 |
| 4 | inf | dense (52) | 4.54e4 | 870 |
| 6 | 2 | 3 (6.2e-8) | 3.96e-8 | 0.64 |
| 6 | 3 | 4 (6.0e-5) | 6.07e-3 | 100 |
| 6 | 4 | 4 (1.8e-3) | 2.24 | 1200 |
| 6 | 6 | 5 (5.7e-2) | 682 | 12000 |

- At q = 2 the construction is a genuine ceiling (trained at 0.64-0.67 of it, argmax_K = K* in both nu), as before.
- At every q > 2 the trained nets exceed the best K-committee construction by 1-4 orders of magnitude. The
  equal-share K-committee family, on which the l_q ladder K* = argmax_K M_K K^{-nu/q} rests, is nowhere near the
  l_q optimum for q > 2. Mechanism: for q > 2 the l_q norm of a fixed linear map falls when the map is spread over
  more rows / neurons / columns at smaller magnitude (a split x -> (x/2, x/2) followed by a summing layer costs
  2^{2/q - 1} < 1), so the l_q optimum uses all of d, m and r at near-equal weight magnitude; the ladder's
  per-frequency cost K^{nu/q} assumes each frequency pays for its own neurons and does not describe that regime.
  What then bounds the frequency count is cross-talk between frequencies sharing neurons, which neither the
  ladder nor the construction prices. That is the superposition gap (notes/l2_theorem.md) at work.

## Verdict

- The l_2 ansatz stands: ceiling matched, count within +-1 of K*, ordering right (stage 1).
- The l_q generalisation (THEORY.md sec 3-4) is WITHDRAWN as a quantitative count predictor for q > 2: the
  ladder's absolute values are off by 10-10^4 and its count predictions (8..12 / dense) are not seen. What the data
  supports: the count rises weakly with q at nu 4 (4-5 -> 6-7 from q 2 to inf) and is flat at 4 for nu 6; no
  geometry in {2, 3, 4, 6, inf} produces a dense solution in this family. The "constant-geometry optimizers are
  admissible" scoping stays (each geometry is internally consistent, Control 1); the claim that q sets the count
  through the ladder does not.
- Open: an l_q ceiling that uses full width (d, m, r) with superposed frequencies, to test whether the trained
  l_q nets are at their optimum. Not needed for the l_2 paper spine.
