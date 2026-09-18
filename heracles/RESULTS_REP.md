# REP — fresh-seed replication of the modular-addition counts on the exact two-operand family (2026-09-18)
[FINAL_RUNS/REP, nsum_rep.py (batched-seeds copy of nsum.py, n = 2), rep_table.py, rep_pred.json, rep_scored.json]

Networks: x² on a + b (Morwani's quadratic net, ν 3), bilinear e(a)·e(b)·w(c) (gated chain n 2, ν 3), ReLU on a + b
(ν 2 dense control). Full p² table, full-batch GD (NOT AdamW), width 500, no biases, lr 0.05 doubled at 1e3..10e3
capped at 12.8, 40k steps, penalty λ Σ_i ‖θ_i‖^q (q = 2 plain weight decay, q = 3 = Morwani's L_{2,3}), λ 1e-4.
Seeds 100–115 (16 per cell; 100–107 at p ≥ 211). No seed overlaps any earlier run. Trainer = nsum_rep.py: nsum.py's
recipe and instruments verbatim with a leading seeds dimension and the grid evaluated by broadcasting instead of
one-hot GEMMs (checked against nsum.py on CPU: weights agree to 1–4e-7 relative after 1500 steps for pow / prod+warm-up+clip /
relu+L_{2,ν}; on the pod, seeds 0–3 of the existing p 71 x² q 2 cell give K_2pct 5, 6, 4, 7 / N_.05 6, 7, 13, 15 against the
archived 5, 6, 4, 6 / 6, 7, 12, 13 — seeds 0, 1 identical, the two low-margin seeds drift by one frequency).

## 0. Predictions — FROZEN BEFORE ANY REP RUN WAS SCORED (rep_pred.json, from kstar.py `ladder`)

K*(p, ν, q) = argmax_K M_K(p)/K^{ν/q}, ν = 3. plat.95 / plat.74 = rungs within 5% / 26% of the top (the 26% plateau is
nsum_table.py's `pred`). "dense" = the all-frequency rung (n_f + ½) n_f^{−ν/q} is the maximum. M_K is exhaustive while
C(n_f−1, K−1) ≤ 4e6 and local search (a lower bound) above; ladders are computed to K = n_f at p 59, 71, to K = 30 (+ the
exact dense rung) at p 113, to K = 26 at the other primes (so plat.74 upper ends of 26 / 30 mean "at least").
r(K) = ratio of rung K to the top, K = 3..12.

| p | n_f | q | ν/q | K* | plat.95 | plat.74 | r(3..12) |
|---|---|---|---|---|---|---|---|
| 59 | 29 | 2 | 1.50 | 6 | 4..7 | 3..17 | 3:0.78 4:0.95 5:0.98 6:1.00 7:0.97 8:0.94 9:0.92 10:0.90 11:0.87 12:0.84 |
| 59 | 29 | 2.25 | 1.33 | 6 | 5..11 | 4..29 | 3:0.69 4:0.89 5:0.95 6:1.00 7:1.00 8:0.98 9:0.98 10:0.97 11:0.96 12:0.94 |
| 59 | 29 | 2.5 | 1.20 | 10 | 6..29 | 4..29 | 3:0.61 4:0.81 5:0.89 6:0.96 7:0.98 8:0.98 9:0.99 10:1.00 11:1.00 12:0.99 |
| 59 | 29 | 2.75 | 1.09 | dense | 20..29 | 5..29 | 3:0.49 4:0.67 5:0.76 6:0.83 7:0.86 8:0.87 9:0.89 10:0.91 11:0.92 12:0.93 |
| 59 | 29 | 3 | 1.00 | dense | 26..29 | 7..29 | 3:0.40 4:0.56 5:0.64 6:0.72 7:0.76 8:0.78 9:0.80 10:0.83 11:0.84 12:0.85 |
| 71 | 35 | 2 | 1.50 | 5 | 5..7 | 4..16 | 3:0.72 4:0.91 5:1.00 6:0.96 7:0.95 8:0.91 9:0.88 10:0.88 11:0.84 12:0.81 |
| 71 | 35 | 2.25 | 1.33 | 7 | 5..10 | 4..35 | 3:0.66 4:0.86 5:0.99 6:0.98 7:1.00 8:0.97 9:0.96 10:0.98 11:0.95 12:0.94 |
| 71 | 35 | 2.5 | 1.20 | 10 | 7..21 | 4..35 | 3:0.57 4:0.78 5:0.93 6:0.94 7:0.98 8:0.97 9:0.97 10:1.00 11:0.98 12:0.98 |
| 71 | 35 | 2.75 | 1.09 | dense | 21..35 | 5..35 | 3:0.46 4:0.65 5:0.79 6:0.82 7:0.86 8:0.87 9:0.89 10:0.92 11:0.92 12:0.92 |
| 71 | 35 | 3 | 1.00 | dense | 29..35 | 7..35 | 3:0.37 4:0.54 5:0.66 6:0.70 7:0.75 8:0.76 9:0.78 10:0.82 11:0.82 12:0.84 |
| 89 | 44 | 2 | 1.50 | 5 | 5..9 | 4..20 | 3:0.72 4:0.93 5:1.00 6:1.00 7:1.00 8:1.00 9:0.97 10:0.94 11:0.93 12:0.89 |
| 97 | 48 | 2 | 1.50 | 6 | 5..9 | 4..20 | 3:0.69 4:0.89 5:0.98 6:1.00 7:0.98 8:0.99 9:0.96 10:0.94 11:0.90 12:0.88 |
| 113 | 56 | 2 | 1.50 | 6 | 5..8 | 4..20 | 3:0.61 4:0.89 5:0.97 6:1.00 7:0.96 8:0.99 9:0.93 10:0.92 11:0.89 12:0.88 |
| 113 | 56 | 2.25 | 1.33 | 8 | 6..12 | 4..30+ | 3:0.53 4:0.80 5:0.91 6:0.97 7:0.95 8:1.00 9:0.96 10:0.97 11:0.95 12:0.95 |
| 113 | 56 | 2.5 | 1.20 | 16 | 8..28+ | 5..56 | 3:0.46 4:0.72 5:0.85 6:0.92 7:0.93 8:0.99 9:0.97 10:0.99 11:0.99 12:1.00 |
| 113 | 56 | 2.75 | 1.09 | dense | 16..56 | 6..56 | 3:0.36 4:0.59 5:0.71 6:0.79 7:0.81 8:0.88 9:0.87 10:0.90 11:0.90 12:0.92 |
| 113 | 56 | 3 | 1.00 | dense | (31..55 not evaluated)..56 | 10..56 | 3:0.28 4:0.47 5:0.57 6:0.65 7:0.67 8:0.73 9:0.74 10:0.77 11:0.78 12:0.80 |
| 127 | 63 | 2 | 1.50 | 7 | 5..9 | 4..21 | 3:0.60 4:0.85 5:0.97 6:0.98 7:1.00 8:0.99 9:0.96 10:0.94 11:0.91 12:0.90 |
| 157 | 78 | 2 | 1.50 | 6 | 5..10 | 4..24 | 3:0.56 4:0.85 5:0.97 6:1.00 7:1.00 8:1.00 9:0.98 10:0.96 11:0.94 12:0.93 |
| 211 | 105 | 2 | 1.50 | 6 | 5..11 | 4..26+ | 3:0.51 4:0.79 5:0.95 6:1.00 7:1.00 8:1.00 9:1.00 10:0.97 11:0.96 12:0.94 |
| 307 | 153 | 2 | 1.50 | 8 | 6..11 | 5..26+ | 3:0.44 4:0.72 5:0.90 6:0.97 7:0.98 8:1.00 9:0.99 10:0.97 11:0.96 12:0.95 |

ReLU a + b (ν 2): dense at every prime. x² under L_{2,3} (q 3): dense.

What the prediction is and is not. At ν/q = 1.5 the ladder is nearly flat from K = 5 to 9 at every prime (rungs within
3–5% of each other), so the falsifiable content of the prime axis is: (i) the count stays O(5–10) and does NOT grow with
n_f (29 → 153), (ii) counts lie in plat.74, mostly in plat.95, (iii) a slow upward drift of the plateau (4..7 at p 59 to
6..11 at p 307). "K* ± 1" is reported but the argmax itself is not sharply identified by the theory at these primes.

Scoring rules fixed now: three count instruments reported side by side for every cell (N_.05, K_5pct, K_2pct — no
primary picked after the fact); a seed is "fit" iff fp64 accuracy ≥ 0.999; hit rates are over ALL seeds of a cell
(unfit / collapsed / diverged seeds are misses); sparse cells: count ∈ plat.74, count ∈ plat.95, |count − K*| ≤ 1;
dense cells: count ≥ 0.8 n_f. Recipes fixed before launch (from RESULTS_QM / Q / PROD): x² init 1 at p 59, 71 in the
q dial; x² init 4 (uniform) on the prime axis and at p 113; bilinear init 4 + decay-free warm-up until CE < 0.8 ln p
+ clip 1 (lr cap 12.8) at every prime; ReLU init 1 clip 1; λ → 0 continuation = resume at λ/10 for 60k steps, lr 12.8;
bilinear q dial run at λ 1e-4 and 1e-5.

## 1. Results (scored 2026-09-18 after all pulls; 1160 runs, 74 cells; full per-seed table: FINAL_RUNS/REP_logs/rep_table.md, rep_scored.json)

Format: median over fit seeds [range]; hits = seeds (of all seeds in the cell) with count in plat.74 / in plat.95 / within K* +- 1, for N_.05 | K_5pct | K_2pct.

### R1 prime axis, nu 3, q 2, lambda 1e-4, 40k steps, init 4

x^2 (no warm-up):
| p | K* (plat.95) | fit | N_.05 | K_5pct | K_2pct | clean | in plat.74 | in plat.95 | K*+-1 |
|---|---|---|---|---|---|---|---|---|---|
| 59 | 6 (4..7) | 16/16 | 8 [7-11] | 6 [4-11] | 9 [7-12] | 0.66 | 16/16/16 | 4/12/3 | 4/10/3 |
| 71 | 5 (5..7) | 16/16 | 10 [6-12] | 7 [5-9] | 10 [6-13] | 0.59 | 16/16/16 | 2/9/1 | 1/4/1 |
| 89 | 5 (5..9) | 16/16 | 10 [8-21] | 8 [5-10] | 9 [8-12] | 0.55 | 15/16/16 | 7/14/9 | 0/5/0 |
| 97 | 6 (5..9) | 16/16 | 12 [7-25] | 7 [3-9] | 9.5 [7-12] | 0.52 | 15/15/16 | 2/15/8 | 1/9/1 |
| 113 | 6 (5..8) | 16/16 | 18 [10-23] | 7.5 [5-9] | 9 [7-11] | 0.46 | 12/16/16 | 0/14/5 | 0/8/2 |
| 127 | 7 (5..9) | 16/16 | 19.5 [14-27] | 6 [4-8] | 7 [5-9] | - | 13/16/16 | 0/15/16 | 0/10/12 |
| 157 | 6 (5..10) | 0/16 (acc 0.944-0.998, under-trained at 40k) | 13-21 | 1-6 | 3-22 | - | miss | miss | miss |
| 211 | 6 (5..11) | 0/8 (acc <= 0.08: never left zero) | - | - | - | - | miss | miss | miss |
| 307 | 8 (6..11) | 0/8 (acc 0.012: never left zero) | - | - | - | - | miss | miss | miss |
Pooled (128 seeds): fit 96; in plat.74 87/95/96; in plat.95 15/79/42; K*+-1 6/46/19.

Bilinear (init 4, warm-up, clip 1):
| p | K* (plat.95) | fit | N_.05 | K_5pct | K_2pct | in plat.74 | in plat.95 | K*+-1 |
|---|---|---|---|---|---|---|---|---|
| 59 | 6 (4..7) | 16/16 | 7 [5-12] | 6 [4-9] | 7 [5-9] | 16/16/16 | 11/14/13 | 11/13/13 |
| 71 | 5 (5..7) | 16/16 | 8.5 [6-16] | 6.5 [4-8] | 7 [6-8] | 16/16/16 | 5/12/11 | 1/8/3 |
| 89 | 5 (5..9) | 16/16 | 9 [6-18] | 6 [5-8] | 7 [6-9] | 16/16/16 | 9/16/16 | 1/9/7 |
| 97 | 6 (5..9) | 16/16 | 11.5 [7-21] | 6.5 [3-9] | 7 [4-10] | 14/15/16 | 5/14/14 | 2/11/9 |
| 113 | 6 (5..8) | 16/16 | 14 [6-20] | 6 [4-8] | 6.5 [4-8] | 16/16/16 | 1/15/15 | 1/12/12 |
| 127 | 7 (5..9) | 16/16 | 16 [10-20] | 5 [4-8] | 6 [4-8] | 16/16/16 | 0/12/13 | 0/6/9 |
| 157 | 6 (5..10) | 16/16 (margin med -0.6) | 18.5 [11-26] | 5 [2-7] | 6 [4-19] | 15/13/16 | 0/9/11 | 0/9/10 |
| 211 | 6 (5..11) | 4/8 (acc 0.988-0.999, margin -2.2) | 23 [19-25] | 3 [1-3] | 9 [2-25] | 4/0/2 | 0/0/0 | 0/0/0 |
p 307 bilinear not run (budget). Pooled (120 seeds): fit 116; plat.74 113/108/114; plat.95 31/92/93; K*+-1 16/68/63.

### R2 q dial (medians; K_2pct is the RESULTS_Q instrument)

x^2, K_2pct median [range] (N_.05 median) — 40k, then lambda 1e-5 continuation:
| q | pred p59 | p59 init 1 | pred p71 | p71 init 1 | pred p113 | p113 init 4 |
|---|---|---|---|---|---|---|
| 2 | 6 | 7 [3-10] (10) -> 6.5 | 5 | 6 [4-14] (13) -> 5 | 6 | 9 [7-11] (18) -> 9 |
| 2.25 | 6 | 8 [6-11] (11.5) -> 8 | 7 | 6 [4-10] (14) -> 6 | 8 | 9 [7-11] (21.5) -> 9 |
| 2.5 | 10 | 10 [7-13] (13.5) -> 10 | 10 | 7.5 [6-10] (16.5) -> 7.5 | 16 | 9 [7-11] (22.5) -> 9 |
| 2.75 | dense | 15 [11-20] (16) -> 15 | dense | 14 [12-15] (17) -> 13.5 | dense | 12 [10-15] (25) -> 12 |
| 3 | dense | 29 [28-29] (28) -> 29 | dense | 35 [33-35] (33) -> 35 | dense | N_.05 48 [47-50] -> 49 of 56 (K_2pct saturates: 1/56 < 2%) |
Bilinear p 71 (init 4 warm-up), K_2pct median (N_.05): lambda 1e-4: 7 (8.5), 7.5 (9), 9 (10), 11.5 (11), 35 (34);
lambda 1e-5: 8.5 (8.5), 8.5 (8.5), 10 (9.5), 12 (12.5), 35 (34); continuations change no median by more than 1.

### R3 nu 2 dense control / R4 penalty swap
| cell | fit | N_.05 | K_2pct | clean | dense (N_.05 >= 0.8 n_f) |
|---|---|---|---|---|---|
| ReLU p 71 | 16/16 | 33 [32-33] of 35 | 35 | 0.92 | 16/16 |
| ReLU p 113 | 16/16 | 47 [45-49] of 56 | 28 (instrument saturates) | - | 16/16 |
| x^2 L_{2,3} p 71 init 1 | 16/16 | 33 of 35 | 35 | 0.99 | 16/16 |
| x^2 L_{2,3} p 113 init 4 | 16/16 | 48 [47-50] of 56 | - | 0.93 | 16/16 |
| x^2 L_{2,3} p 113 init 1 | 16/16 | 40.5 [38-44] of 56 | - | - | 0/16 (under 45) |
| x^2 plain l2 p 113 init 4 (same net, R4) | 16/16 | 18 [10-23] | 9 [7-11], K_5pct 7.5 | 0.46 | sparse |

### Post-hoc init-scale control (added AFTER the first four R1 cells were scored; exploratory, not pre-registered)
x^2 q 2 K_2pct median / K_5pct median: p 71 init 1 / 2 / 4 = 6 / 9 / 10 and 4 / 8 / 7; p 89 init 1 collapsed (0/16 fit), init 2 / 4 = 7 / 9;
p 97 init 2 / 4 = 7 / 9.5; p 113 init 2 unfit (0/16), init 3 / 4 = 7 / 9. Clean fraction rises with init (p 71: 0.31, 0.46, 0.59).

## 2. Reading
1. Replicates: sparse-vs-dense switch by degree (R3) and by penalty on the same net (R4) — 16/16 everywhere except L_{2,3} p 113 at init 1
   (38-44 of 56, as in RESULTS_QM's 25-29 "under-populated" cell); the q dial is monotone at all three primes and in both nets, endpoints exact;
   lambda -> 0 continuation moves no median by more than 1 (RESULTS_Q reading 2); clean fraction monotone in q, 1.00 at q = nu.
2. Replicates weakly: counts are O(5-10) at every prime while n_f goes 29 -> 78 (K_5pct medians 5-8, K_2pct 6-10); every fit seed is in plat.74 on K_2pct.
   But plat.74 is wide (4..16+), and the sharper tests are mediocre: plat.95 hit rate 15-79% (x^2) / 26-77% (bilinear) depending on instrument,
   K*+-1 5-36% (x^2) / 13-57% (bilinear). Bilinear is consistently closer to the prediction than x^2.
3. Does not replicate as stated: (a) N_.05 grows with p (x^2 8 -> 19.5, bilinear 7 -> 23 over p 59 -> 211), leaving plat.95 entirely for p >= 113 —
   the junk/tail share grows with p even though the >=5%-energy committee does not; (b) the x^2 count depends on init scale (p 71: K_2pct 6 at init 1,
   10 at init 4), so "K* = 5 at p 71" is an init-1 result and the QM "7-9 at p 113" is partly the x4 init; (c) mid-dial at p 113 is flat
   (K_2pct 9, 9, 9, 12 for q 2..2.75 vs predicted 6, 8, 16, dense) — only N_.05 moves (18 -> 25); (d) q 2.75 is predicted dense and measures 12-15 at
   every prime (RESULTS_Q had the same 12.5; plat.74 hit only).
4. Recipe failures (counted as misses): x^2 at init 4 does not escape zero in 40k steps at p 211, 307 and is under-trained at p 157; x^2 p 89 init 1
   and p 113 init 2 collapse/underfit; bilinear p 211 fits 4/8 with negative margins. The prime axis above p 127 needs a warm-up/longer recipe for x^2.
