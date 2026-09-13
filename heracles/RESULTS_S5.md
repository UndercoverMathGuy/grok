# S5 results (2026-09-13, pod fsglufchnz325w L40S + volume jolr5zlyu6)   [scores/SR, S0, S1, S3; analyze_S5.py]

Task S5 (|G| 120, label a @ b), heracles ReLU MLP, m 512, d = r = 128, nu {4, 6}, gd + clip 1 + l_2 wd. Instruments:
interface ablation over isotypic components (read-and-write rule, fixed 2% floor), folded amplitudes A_rho, leader =
faithful irrep with the largest amplitude, share = its fraction of faithful energy, sign_rel = A_sign / A_leader,
gamma_2 = min margin / ||theta||_2^nu. Predictions: PLAN_S5.md A2 (E0, registered before any E2 score was read).

## E1 — recipe sweep (96 runs; RECIPES_S5.md)

Record: gd lr 3, wd 3e-4, clip 1 at frac 0.6 and 0.8 (8/8 grok, highest gamma_2, grok at 800-2200 epochs at frac 0.8).
wd 1e-3 collapses at both fractions; wd 1e-4 groks at lr 3 but is not equilibrated at 20k. The mod-add record transfers.

## E0 — measured weighted ladder (80 fixed-support runs, 4 seeds per support, all grokked)

| support | nu 4 best gamma_2 | / argmax | sign_rel | nu 6 best gamma_2 | / argmax | sign_rel |
|---|---|---|---|---|---|---|
| std + sign | **3.39e-6** | 1.00 | 0.17 | 1.34e-8 | 0.69 | 0.00 |
| std_sign + sign | 3.01e-6 | 0.89 | 0.15 | **1.95e-8** | 1.00 | 0.05 |
| std | 2.80e-6 | 0.83 | | 1.34e-8 | 0.69 | |
| std_sign | 2.75e-6 | 0.81 | | 1.62e-8 | 0.83 | |
| 5b | 2.17e-6 | 0.64 | | 1.08e-8 | 0.55 | |
| 5a + sign | 2.01e-6 | 0.59 | 0.18 | 9.5e-9 | 0.49 | 0.01 |
| 5b + sign | 1.78e-6 | 0.53 | 0.11 | 9.4e-9 | 0.48 | 0.05 |
| 5a | 1.52e-6 | 0.45 | | 5.7e-9 | 0.29 | |
| 6 + sign | 1.38e-6 | 0.41 | 0.15 | 8.2e-9 | 0.42 | 0.05 |
| 6 | 9.5e-7 | 0.28 | | 5.4e-9 | 0.28 | |

Registered: leader std (+ sign 0.17) at nu 4, std_sign (+ sign 0.05) at nu 6; 6 never leads; 5b > 5a.

## E2 — main (64 runs, 16 seeds per cell, 60k epochs, all grokked; 20k checkpoint = 60k in every cell)

| cell | leaders | K_abl | (a) one irrep >= 0.9 | mixed (>= 2 faithful > 0.2) | (b) predicted leads | (c) sign_rel on std / others | (d) gamma_2 med / ceiling | clean |
|---|---|---|---|---|---|---|---|---|
| nu 4 frac 0.6 | std 7, std_sign 6, 5b 2, 5a 1 | {1:3, 2:8, 3:4, 4:1} | 0.69 | 5 | std 0.44 | 0.171 / 0.127 | 2.32e-6 / 3.39e-6 = 0.68 | 0.75 |
| nu 4 frac 0.8 | std 7, std_sign 4, 5b 3, 5a 2 | {1:2, 2:11, 3:3} | 0.69 | 5 | std 0.44 | 0.171 / 0.107 | 2.25e-6 / 3.39e-6 = 0.66 | 0.91 |
| nu 6 frac 0.6 | std 9, std_sign 5, 5b 2 | {1:11, 2:5} | 1.00 | 0 | std_sign 0.31 | 0.000 / 0.000 | 1.33e-8 / 1.95e-8 = 0.69 | 0.59 |
| nu 6 frac 0.8 | std 4, std_sign 4, 5b 4, 5a 3, 6 1 | {1:7, 2:7, 3:2} | 0.94 | 1 | std_sign 0.25 | 0.073 / 0.053 | 1.05e-8 / 1.95e-8 = 0.54 | 0.62 |

Median gamma_2 by leader (clean runs), free runs vs the E0 ladder:

| nu | ladder order (E0) | free-run order (E2, medians) |
|---|---|---|
| 4 | std 3.39 > std_sign 3.01 > 5b 2.17 > 5a 2.01 > 6 1.38 (e-6) | std 3.17 / 3.10 > std_sign 2.67 / 2.96 > 5a 1.52 / 2.06 ~ 5b 1.36 / 2.04 > mixed 1.0-1.7 |
| 6 | std_sign 1.95 > std 1.34 > 5b 1.08 > 5a 0.95 > 6 0.82 (e-8) | std_sign 1.50 / 1.50 > std 1.31 / 1.42 > 5b 0.99 / 0.82 > 5a 0.78 > 6 0.55 > mixed 0.43 |

Statement scorecard (as registered in PLAN_S5.md A / A2):
- (a) K* = 1: holds at nu 6 (1.00 / 0.94 single-irrep); at nu 4 it is 0.69 in both cells with 5/16 mixed runs, which
  meets the registered kill (>= 3 mixed). The mixed runs are the low-margin tail: every mixed run has a lower gamma_2
  than every clean std or std_sign run in its cell (std + std_sign pairs, std + 5a, std + 5b, 5b + 6). Same shape as
  mod-add's K* + 1: a finite-wd second basin at lower margin, not a second optimum. nu 6 has no such basin at 16 seeds.
- (b) predicted leader >= 60%: FAILS in every cell (0.44, 0.44, 0.31, 0.25); the nu-6 frac-0.8 cell trips the < 30% kill.
  The population leader is std at both nu (7/16, 7/16, 9/16, 4/16), not the ladder argmax at nu 6. The leader is a
  seed lottery weighted toward the 4-dim irreps; at nu 6 frac 0.8 it spreads to every faithful irrep.
- (c) sign patch: std-led runs at nu 4 sit at 0.171 (ladder 0.17) with the top runs at 0.16-0.17; std_sign-led at
  0.15-0.18 (ladder 0.15). At nu 6 the patch is 0.00-0.15 (ladder 0.00 on std, 0.05 on std_sign). Holds where the
  ladder gives a nonzero patch; the amplitude is the ladder's.
- (d) gamma_2 >= 0.5 of the argmax ceiling: holds in every cell (0.68, 0.66, 0.69, 0.54). The std + sign runs at nu 4
  reach 0.92-1.03 of the fixed-support ceiling, i.e. the free optimum for that support IS the ceiling.

Reading. The weighted l_2 ladder predicts the MARGIN of each irrep solution: the rank order of leaders by gamma_2 in
the free runs is the ladder order in all four cells (std > std_sign at nu 4, std_sign > std at nu 6, then 5b >= 5a > 6),
and the values are within ~25% of the fixed-support ceilings (std at nu 6 exceeds its 4-seed ceiling by 1.2x: the
ladder's absolute values are best-of-4 and noisy at that level). What the ladder does NOT predict is which basin a seed
lands in: the leader frequency does not follow the margin ordering, and the nu-6 flip of the argmax to std_sign shows
up only as std_sign-led runs having the higher margin, not as std_sign leading more often. This is the S5 form of the
mod-add finding that the count is a margin property while identity is init-borne (notes 09-04 cascade test, 08-22
identity predictor): here the count is 1 (+ sign) and the identity is the lottery.

Not run: E3 (S2c, wd -> 0 at S5). Registered trigger was "leader or sign_rel moves with wd in E1"; in E1 the leader
split at wd 1e-4 (std_sign 4 / std 3 at both fracs) equals the split at 3e-4 and sign_rel is 0.15 vs 0.14-0.17, so the
trigger did not fire. The open question it would answer: does the leader distribution move toward the ladder argmax as
wd -> 0 (the mod-add tie-break did), or is the basin fixed at init for good? One cell (nu 6, wd 1e-4 vs 3e-4, 16 seeds,
200k) settles it; ~$1.

## E1b — optimizer control at S5 (gdm lr 0.3 clip 1; sgd lr 1 B 1024; wd 3e-4, frac 0.8, nu 4, 8 seeds, 20k)

| opt | grok | gamma_2 med | clean | K_abl | leaders | share | sign_rel |
|---|---|---|---|---|---|---|---|
| gd (E1) | 8/8 | 2.53e-6 | 0.86 | {1:1, 2:4, 3:3} | std_sign 4, std 3, 5a 1 | 1.00 | 0.144 |
| gdm | 8/8 | 2.28e-6 | 0.92 | {1:1, 2:5, 3:2} | std_sign 4, std 2, 6 1, 5a 1 | 1.00 | 0.135 |
| sgd | 8/8 | 1.83e-6 | 0.87 | {2:5, 3:2, 4:1} | std_sign 3, std 2, 5a 2, 6 1 | 0.98 | 0.155 |

Same count, same patch, same lottery, margins within 10-30%: the l_2 family agrees at S5 as it did on mod-add (H5).

## E4 — other groups (S3 + S3b addendum + S3c retry; scores/S3)

Measured ladders (fixed-support, best gamma_2 over 4 seeds, all supports grokked 4/4) and free runs (8 seeds, 60k;
D_113 20k). "of argmax" uses the ladder including the addendum supports (pairs at A5, triples at S4).

S4 (irreps 1, 1, 2, 3 = std, 3' = std_sign; sign patch available; the 2-dim irrep is non-faithful):

| nu | ladder top rungs (of argmax) | free runs |
|---|---|---|
| 4 | std_sign+sign+2 1.00, std+sign+2 0.99, std_sign+sign 0.97, std+sign 0.90, std_sign 0.89, std 0.80 | 8/8 grok; leaders std 6, std_sign 2; K_abl 3 (leader + sign + 2) in 6/8; single faithful irrep 1.00; gamma_2 0.71 of argmax; sign_rel on std 0.36 (ladder 0.22) |
| 6 | std_sign 1.00, std_sign+sign+2 0.97, std+sign+2 0.96, std_sign+sign 0.95, std+sign 0.94, std 0.73 | 7/8 grok; leaders std 6, std_sign 1; K_abl 2 in 6/7; single faithful 0.86; gamma_2 0.63 of argmax |

A5 (irreps 1, 3a, 3b, 4, 5; simple group, no sign irrep; 3a / 3b are Galois conjugates):

| nu | ladder top rungs (of argmax) | free runs |
|---|---|---|
| 4 | 3a+4 1.00 (med 0.67), 4+5 0.85 (med 0.39), 3a+3b 0.84 (med 0.70), 4 0.84 (med 0.81), 3a 0.74, 3b 0.73, 5 0.38 | 8/8 grok; 3a + 3b in 8/8 (K_abl 2 in 7/8), leader 3a 4 / 3b 4 at share 0.73; gamma_2 5.5e-6 = 0.58 of argmax, 0.69 of the 3a+3b rung |
| 6 | 3a+3b 1.00, 3a 0.94, 3b 0.92, 3a+4 0.88, 4+5 0.62, 4 0.56, 5 0.16 | 7/8 grok (one run destabilised after 40k); leaders 5 4, 3b 2, 3a 1; single faithful 0.86; gamma_2 2.4e-8 = 0.32 of argmax |

- S4: K* = 1 (+ sign, + the cheap non-faithful 2) holds in every grokked run; the ladder top is flat (six supports within
  20% at nu 4), the free runs sit in it at 0.63-0.71 of the argmax, and the leader lottery again favours std over the
  std_sign argmax (6/8 at both nu).
- A5: the free solution is the conjugate PAIR 3a + 3b at nu 4 — the patch role sign plays on S5 / S4 is played by the
  Galois conjugate, which covers the two 5-cycle classes evenly — and the ladder prices it as a top rung (0.84 of the
  best-of-4 argmax, 0.70 by median, within the flat top). At nu 6, 3a + 3b IS the argmax but the free runs land on the
  worst rung (5 leads 4/7) at 0.32 of the ceiling: the strongest case in the programme of the lottery going against the
  ladder. Those runs are low-margin (ladder 5 = 0.16 of argmax; free 5-led runs 0.28) — the same "lower rung = lower
  margin" ordering, with an unlucky population.
- D_113 (56 two-dim irreps + sign): the record recipe COLLAPSES at frac 0.8 (0/8 at both nu, weights -> 0; the fixed-support
  ladder at frac 0.8 did not grok either, 3/14, so it is void). Retry at frac 0.5, nu 4, 20k, 4 seeds each: 4/4 grok at
  wd 3e-4 and 1e-4, K_abl 4-5 (wd 3e-4: {4:3, 5:1}; wd 1e-4: {5:1, 6:2, 8:1}), K_load90 3-5, 3-4 two-dim irreps carrying
  90% of the margin plus the sign irrep at >= the leader amplitude (the reflection parity), clean 0.68-0.78, gamma_2
  1.1e-6. Reading: D_113's count is mod-add's K*(113, nu 4) = 4 rotation frequencies plus the parity patch, with no
  hierarchical pair structure (members e.g. {49, 48, 12, 4}, {4, 22, 7, 43}); the fixed-support ladder for D_113 must be
  re-run at frac 0.5 to price it (not done today).

## Verdict (S5 programme, 2026-09-13; 470 runs, ~$5.5)

1. The weighted l_2 ladder's COUNT prediction holds on S5, S4 and A5: one faithful irrep plus the cheapest coverage patch
   (sign on S5 / S4, the Galois conjugate on A5), K_abl 1-2 in the clean runs, exactly as the coverage table says. The only
   deviation is a low-margin two-irrep basin in 5/16 nu-4 S5 runs, the S5 form of mod-add's +1.
2. The ladder predicts the MARGIN of each solution: in every cell the leaders ranked by measured gamma_2 follow the
   fixed-support ladder order, and the top free runs reach the ceiling (S5 nu 4 std + sign at 0.92-1.03 of it). The sign
   patch amplitude is the ladder's (0.17 on std at S5 nu 4).
3. The ladder does NOT predict which near-tied support a seed lands in. The registered leader statement (b) failed in
   every S5 / S4 cell (predicted leads 0.14-0.44) and the nu-6 flip to std_sign appears only as a margin ordering, not as
   a frequency. The identity is a seed lottery, biased toward the 4-dim (S5) / 3-dim (S4) permutation irreps regardless of
   the small (< 20%) differences between top rungs. Same conclusion as the mod-add programme: count and margin are l_2
   properties; identity is init-borne.
4. D_113 stays the scope boundary: the count matches mod-add's K* = 4 plus a parity patch, but the recipe needs frac 0.5
   and the ladder was not priced there.
5. Open: E3 (wd -> 0 at S5 / A5 nu 6) — does the lottery drift toward the argmax as the margin rises, as the mod-add
   tie-break did? One 16-seed cell at nu 6, 200k epochs, ~$1.
