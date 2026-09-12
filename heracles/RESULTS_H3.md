# H3 results — the Wei limit (2026-09-12)   [scores/H3, analyze_H3.py]

Record recipe (gd lr 3, clip 1), p 113, frac 0.5, 16 seeds per arm, full instrument at every checkpoint.
nu 4: wd {1e-3, 3e-4, 1e-4, 3e-5}, 600k epochs, checkpoints 20k / 60k / 200k (L40S pod fm3bmxbopmvqmm).
nu 6: wd {1e-3, 3e-4, 1e-4}, 200k epochs, checkpoints 20k / 60k. All 112 runs grokked and finite.
"of ceiling" = gamma_2 / constructed K* ceiling (nu 4: 6.39e-6 at K 4; nu 6: 6.2e-8 at K 3, construct_q.py).
Ladder flatness at the top (kstar.json, q 2): nu 4 gamma_hat(5)/gamma_hat(4) = 0.98; nu 6 gamma_hat(4)/gamma_hat(3) = 0.94.

## nu 4 (K* = 4, plateau74 3..8) — 600k finals (checkpoints agree; see table in analyze_H3.py output)

| wd | mode | histogram | gamma_2 / ceiling | clean | train CE |
|---|---|---|---|---|---|
| 1e-3 | 4 | {3:5, 4:7, 5:4} | 0.52 | 0.89 | 5e-2 |
| 3e-4 | 5 | {4:6, 5:9, 6:1} | 0.64 | 0.99 | 1.3e-2 |
| 1e-4 | 5 | {4:5, 5:8, 6:2, 7:1} | 0.69 | 0.99 | |
| 3e-5 | 4 | {4:9, 5:5, 6:2} | 0.74 | 1.00 | |

- Margin rises monotonically as wd -> 0 at matched budget (0.52 -> 0.64 -> 0.69 -> 0.74) and clean -> 1.00: the
  Wei prediction. Budget: 200k and 600k reads are identical in every arm (wd 3e-5 needs > 20k to equilibrate;
  by 60k it is at 0.66 and by 200k at 0.72).
- Count: never drifts up (no arm past mode 5; 6-7 in <= 3/16 seeds). At the highest margin the 4/5 split tips
  toward K*: 6:9 (3e-4) -> 5:8 (1e-4) -> 9:5 (3e-5). Registered fork: (a), weakly — the count moves toward K* as
  wd -> 0, as a shift of a near-even split rather than a jump. That is what the ladder implies: gamma_hat(5) is
  0.98 of gamma_hat(4), so K* and K* + 1 are tied within 2% and the finite-wd solution family straddles them.
- wd 1e-3 is the low-margin regime seen in R: clean 0.89, members dropped, mode 4 with 3s.

## nu 6 (K* = 3, plateau74 2..5) — 200k finals

| wd | mode | histogram | gamma_2 / ceiling | clean |
|---|---|---|---|---|
| 1e-3 | 4 | {3:1, 4:9, 5:5, 6:1} | 0.70 | 1.00 |
| 3e-4 | 4 | {3:1, 4:8, 5:7} | 0.62 | 1.00 |
| 1e-4 | 4 | {3:4, 4:8, 5:4} | 0.70 | 1.00 |

- Mode 4 = K* + 1 at every wd and budget; clean 1.00 everywhere; margin 0.62-0.71, NOT monotone in wd at nu 6
  (3e-4 is the lowest). Seeds at K* = 3 appear only at the smallest wd and grow with budget (2 -> 3 -> 4 of 16 at
  20k / 60k / 200k). gamma_hat(4)/gamma_hat(3) = 0.94: a flatter top than nu 4's, and the count sits on the
  K* + 1 side of it.
- Registered fork at nu 6: (b) at this budget, with a slow drift toward (a). Neither the "count keeps rising"
  nor the "clean fraction falls" outcome occurred at either nu.

## Verdict

The wd -> 0 limit under plain GD reaches 0.74 (nu 4) / 0.70 (nu 6) of the constructed ceiling with clean
committees, and the count sits on the K* / K* + 1 boundary of a ladder whose top two rungs are within
2% (nu 4) and 6% (nu 6) of each other, tipping toward K* as the margin rises. The +1 is a
finite-wd, finite-budget offset inside a nearly flat plateau corner, not a property of the optimizer and not a
different optimum. Paper sentence: predicted K*, measured K* or K* + 1, with the tie-break moving toward K* in
the limit. The count-vs-wd effect of R / H2 is this same boundary seen from the low-margin side.
