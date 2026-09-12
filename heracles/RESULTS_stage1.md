# Stage 1 results (2026-09-12) — R2, Rkeep, H1, H2 on pod vs2z7ht3uk9nyj (RTX PRO 4500), full instrument

Scores: scores/Rpod (retrained R recipe cells + R2), scores/H1, scores/H2. Raw runs (spectra, metrics, configs,
final weights, H2 20k checkpoints) pulled to heracles/runs (gitignored; 608 runs, 887 MB). Every cell below
grokked 8/8 unless stated. Counts = interface-ablation K_abl with the read-and-write rule.

## H1 — p x nu grid, q = 2, record recipe (lr 3, wd 3e-4, clip 1), 20k epochs   [analyze_H1.py]

mode of K_abl (8 seeds) vs K*(p, nu, q = 2):

| nu | p 59 | p 71 | p 97 | p 113 | p 127 |
|---|---|---|---|---|---|
| 3  | 5 (K* 6) | 5 (5) | 5 (6) | 5 (6) | 6 (7) |
| 4  | 4 (4) | 4 (4) | 5 (4) | 5 (4) | 4/5 tie (5) |
| 5  | 4 (3) | 4 (3) | 4 (3) | 4 (4) | 4 (4) |
| 6  | 3 (3) | 4 (3) | 4 (3) | 4 (3) | 4 (3) |

- |mode - K*| <= 1 in 20/20 cells. Monotone in nu at every prime (5-6 at nu 3 down to 3-4 at nu 6).
- Registered clause (a) "mode in {K*, K* + 1}": holds in 15/20. The 5 misses are all K* - 1, at nu 3 (4 of 5
  primes, flat ladder, plateau 3..12) and nu 4 p 127 (tie 4/5 vs K* 5). The sharp rows (nu 5, 6) run K* + 1.
- Registered clause (b) ">= 90% of grokked runs in plateau74": fails in 4 cells (59/nu6 0.88, 71/nu5 0.88,
  71/nu6 0.75, 113/nu6 0.88), which triggers the registered kill for that clause (>= 3 cells). The misses are
  1-2 seeds at K* + 2 in plateaus 3 wide. Recorded as registered: (b) KILLED; (a) plateau-not-point at nu 3.
- Reading: the count is predicted to within one feature everywhere with no fitted parameter, and the offset
  has one sign per row (low where the ladder is flat, high where it is sharp). The measured count moves less
  than K* does across nu >= 4 (4-5 almost everywhere while K* goes 3-5). H3 decides whether the K* + 1 at
  the sharp rows is the wd -> 0 limit or a finite-wd offset.
- The lr 1 fallback groks 8/8 in 19/20 cells (7/8 at 59/nu3) with 2-5x lower gamma_2 and wider histograms;
  the record recipe transfers across the whole cubby without retuning.
- nu-6 kill rule (mode outside {3, 4} at >= 2 primes): not triggered.

## H2 — wd x frac, p 113 nu 4, record recipe, 60k epochs with a 20k checkpoint

| frac | wd | mode @20k | mode @60k | histogram @60k | gamma_2 / ceiling | clean |
|---|---|---|---|---|---|---|
| 0.3 | 3e-4 | 5 | 5 | {4:3, 5:4, 7:1} | 0.50 | 0.94 |
| 0.3 | 1e-3 | 4 | 4 | {3:1, 4:5, 5:2} | 0.48 | 0.96 |
| 0.5 | 3e-4 | 5 (Rpod, 20k) | — | {4:3, 5:5} | 0.67 | 1.00 |
| 0.5 | 1e-3 | 4 (Rpod, 20k) | — | {3:1, 4:6, 5:1} | 0.42 | 0.85 |
| 0.7 | 3e-4 | 4 | 4 | {3:1, 4:6, 5:1} | 0.70 | 1.00 |
| 0.7 | 1e-3 | collapse | collapse | 0/8 grok: weights decay to zero (pnorm 0, CE = ln p) | — | — |

- 20k vs 60k: identical modes and margins (within 0.03 of ceiling) in every cell; train CE is flat from 5k
  epochs on (4.9e-2 at wd 1e-3, 1.3e-2 at wd 3e-4). These runs are at their finite-wd fixed point by 5k
  epochs, so the wd dependence of the count is a property of wd, not of the budget. (Matters for H3.)
- Registered statement "frac does not move the mode by >= 1 at fixed wd": at wd 3e-4 the mode is 5 at frac
  0.3 and 0.5 and 4 at frac 0.7 (K* = 4), i.e. moves by 1 at the top of the range; at wd 1e-3 it is 4 at 0.3
  and 0.5. Recorded as a marginal miss. Frac 0.7 wd 3e-4 is the best cell measured so far: mode K*, 0.70 of
  ceiling, clean 1.00.
- Recipe boundary: wd 1e-3 collapses at frac 0.7 (as wd 3e-3 does at frac 0.5). Add to RECIPES.

## Rkeep — the 14 RECIPES cells retrained (scores/Rpod vs scores/Rsides)

Same modes and margins as the lost R runs to within seed noise (gd record 0.67 of ceiling both times, mode 5
both times; gdm/sgd 0.61/0.66; sign+box 0.16; MIXED 0.18). RECIPES.md stands; raw trajectories now exist.
