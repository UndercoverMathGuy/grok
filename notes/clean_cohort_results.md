# Clean cohort 2026-09-09: fresh p=113 sweeps + S5, scored by interface ablation

User decision 09-09: scrap reliance on existing runs, train a fresh p=113
cohort with architecture / weight-decay / fraction / optimizer / width
sweeps plus S5, score everything with the new interface-ablation count
(notes/ablation_count_results.md), and compare against the formula's K*.
Pod: 5090 secure (hcz3q8vp2a8g63), whole cohort trained in ~45 min, pulled
(230 runs, 2362 checkpoints verified), pod deleted. Data: runs_clean/
(gitignored), logs notes/clean_pod_logs/, scored JSON
notes/ablation_count_clean_{mlp,tf,s5}.json, table nudial/clean_summary.py.

Drivers: nudial/train_nudial.py --cohort CLEAN (Lion + ckpt_every added),
cloud/clean_cohort.py (transformer: opt gd/lion, n_pre_lin / n_post_lin
degree dial, S5). Checkpoints every 1000 (MLP) / 2500 (transformer) /
5000 (S5) epochs, so the count can be re-run at any stage.

## Formula (p=113 equal-share ladder, recomputed 09-09, nudial/kstar_table.py)

| degree nu | K* | plateau {r >= 0.74} |
|---|---|---|
| 2 | dense (sparse/dense 0.8x) | 6..12+ |
| 3 | 6 | 4..12 |
| 4 | 4 | 3..8 |
| 5 | 4 | 3..6 |
| 6 | 3 | 2..5 |
| 7 | 3 | 2..4 |
| 8 | 2 | 2..3 |
| 10 | 2 | 2 |
| S5 (nu 6) | 1 faithful irrep, sign patch allowed | |

## Transformer p=113 (1L ReLU, d 128, 4 heads, frac 0.3, wd 1, 25k epochs)

Measured count = interface-ablation K_abl on grokked runs (test acc >= 0.99).
K_abl == legacy K_thr10 in every cell (both shown in clean_summary.py).

| cell | n grokked | K_abl | median | K* |
|---|---|---|---|---|
| base wd 1 frac 0.3 | 8/8 | 4 x8 | 4 | 3 |
| wd 0.4 | 3/3 | 3, 4, 5 | 4 | 3 |
| wd 2.5 | 3/3 | 3, 4, 4 | 4 | 3 |
| frac 0.5 | 3/3 | 4, 5, 5 | 5 | 3 |
| frac 0.7 | 3/3 | 3, 4, 4 | 4 | 3 |
| d_mlp 128 | 3/3 | 3, 4, 4 | 4 | 3 |
| d_mlp 2048 | 3/3 | 3, 4, 4 | 4 | 3 |
| +1 pre-linear (nu 7) | 6/6 | 4 x5, 3 | 4 | 3 |
| +1 pre +1 post (nu 8) | 6/6 | 3 x3, 4 x3 | 3.5 | 2 |
| +2 pre +2 post (nu 10) | 3/4 | 2, 3, 4 | 3 | 2 |
| GD lr 1.0 wd 1e-3 / lr 0.3 wd 3e-4 | 0/6 | diverged / stuck at 0.5-0.6 acc | | |
| Lion lr 1e-4 / 3e-4, wd 1 | 0/6 | test acc 0.93-0.98 at 25k (not grokked) | | |

Reading: invariant to wd, frac, width; falls with the degree dial in the
order the formula gives (4 -> 4 -> 3.5 -> 3 for K* 3 -> 3 -> 2 -> 2);
sits at K* + 1 at every degree. Unfolded functional count (before the
harmonic fold) is 7-14 in every transformer: the extra load-bearing
frequencies are all harmonics / cross terms of the fundamentals.

## MLP p=113 (m 512 unless stated, frac 0.5, 10k epochs)

| cell | n grokked | K_abl histogram | median | K* |
|---|---|---|---|---|
| relu-4 wd 0.3 | 16/16 | {5:1, 6:7, 7:2, 8:4, 9:2} | 6.5 | 4 |
| relu-4 wd 1 | 16/16 | {4:1, 5:6, 6:5, 7:3, 8:1} | 6 | 4 |
| relu-4 wd 3 | 16/16 | {5:10, 6:5, 8:1} | 5 | 4 |
| relu-4 frac 0.3 | 8/8 | {5:1, 6:2, 7:3, 8:2} | 7 | 4 |
| relu-4 frac 0.7 | 8/8 | {4:1, 5:5, 6:2} | 5 | 4 |
| relu-4 m 128 | 8/8 | {4:1, 5:4, 6:3} | 5 | 4 |
| relu-4 m 2048 | 8/8 | {4:1, 5:1, 6:5, 7:1} | 6 | 4 |
| relu-4 GD lr 1.0 wd 1e-3 frac 0.5 | 8/8 | {3:3, 4:2, 5:3} | 4 | 4 |
| relu-4 GD lr 0.3 wd 3e-4 frac 0.5 | 0/8 | acc 0.5 | | |
| relu-4 GD lr 0.3 wd 3e-4 frac 1.0 | 8/8 | {2:1, 3:6, 4:1} | 3 | 4 |
| relu-4 Lion lr 1e-4 / 3e-4 wd 1 | 0/16 | acc 0.5 | | |
| relu-5 | 8/8 | {4:3, 5:5} | 5 | 4 |
| relu-6 | 8/8 | {3:2, 4:5, 5:1} | 4 | 3 |
| relu-3 | 8/8 | {8:1, 9:3, 10:1, 11:3} | 9.5 | 6 |
| relu-2 | 8/8 | dense: all 56 frequencies >= 10% amplitude | dense | dense |
| quad-3 | 8/8 | dense: all 56 >= 10% (clean frac 0.84) | dense | 6 |
| quad-4 | 8/8 | >= 16 (candidate cap), clean frac 0.51 | many | 4 |

Reading: AdamW relu-4 sits at K*+1 to K*+2 and MOVES with wd (0.3 -> 3:
6.5 -> 5) and frac (0.3 -> 0.7: 7 -> 5), which the transformer does not;
width-invariant. Plain GD at frac 0.5 lands at K* exactly (median 4), the
only cell that does; GD full-data at K*-1. relu-5 / relu-6 at K*+1;
relu-3 at K*+3.5; nu = 2 dense as the formula says; the quad family is
dense/many (known: quad breaks the M_K structure, 08-2x). Lion did not
grok in either architecture at these settings (exploratory, untuned).

## S5 (1L ReLU transformer, frac 0.7, wd 1 / 2, 50k epochs)

8/8 grokked. 7/8 use ONE faithful irrep (std x3, std_sign x2, 5a x3 across
the eight) plus the sign patch (sign cut costs 0.23-0.75 of the min
margin, counted in 8/8); 1/8 uses two faithful irreps (5a + std_sign).
Prediction K* = 1 faithful irrep: holds. Old S5 runs (32, 09-04 cohort)
re-scored with the same instrument: 31/32 one faithful irrep.

## Bottom line

With the surgical count, on fresh runs, at one prime: the formula orders
the count correctly across degree (transformer 4/4/3.5/3 vs K* 3/3/2/2;
MLP relu-6/5/4/3 = 4/5/6/9.5 vs 3/4/4/6; nu=2 dense; S5 = 1) and the
trained count sits above K* by +1 (transformer, every degree; relu-5/6)
to +2..+3.5 (relu-4/3 AdamW), with the excess shrinking under stronger
wd and larger train fraction, and vanishing under plain GD (K* exactly).
The excess is optimizer-borne, not a counting artefact: the ablation and
logit counts agree on every grokked run in this cohort.

## Optimizer sweep (same day, two passes, pods opt-sweep / opt-sweep2, ~$1 total)

Question (user): is there an optimizer that keeps l2 geometry (no drift
toward the l_inf / sign regime of Adam and Lion) and still groks and gets
close to the margin optimum, since plain GD "isn't doing that right".
Added to both trainers: gdm = heavy-ball GD (mu 0.9, coupled l2), ngd =
normalized GD (unit gradient step, coupled l2), and large-epsilon Adam
(eps 1 / 10, so the update collapses to momentum GD at lr/eps). Pass 1
was mis-scaled (ngd wd 1e-3 memorises, eps cells with per-step decay 1e-3
collapse, transformer gdm lr 0.1-0.3 diverges); pass 2 scaled every arm to
the working plain-GD cell (lr 1, wd 1e-3 MLP / lr 0.3 transformer).
Scored JSON: notes/ablation_count_opt{,2}_{mlp,tf}.json.

MLP relu-4 m512 frac 0.5 (nu 4; ceiling = constructed K=4 free 6.39e-6):

| optimizer | epochs | grokked | K_abl | gamma med | of ceiling | clean frac |
|---|---|---|---|---|---|---|
| AdamW wd 1 (reference) | 10k | 16/16 | {4:1, 5:6, 6:5, 7:3, 8:1} med 6 | 4.90e-6 | 0.77 | 0.998 |
| AdamW wd 3 | 10k | 16/16 | med 5 | 4.24e-6 | 0.66 | 0.995 |
| plain GD lr 1 wd 1e-3 | 10k / 20k | 8/8 | med 4 | 3.47e-6 / 3.43e-6 | 0.54 | 0.84 |
| momentum GD lr 0.1 wd 1e-3 | 10k | 8/8 | {3:1, 4:6, 5:1} | 3.73e-6 | 0.58 | 0.94 |
| momentum GD lr 0.1 wd 1e-3 | 20k | 8/8 | {3:2, 4:5, 5:1} | 4.09e-6 (max 4.73e-6) | 0.64 | 0.996 |
| momentum GD lr 0.3 wd 1e-3 | 10k | 8/8 | {3:3, 4:3, 5:2} | 3.97e-6 | 0.62 | 0.97 |
| momentum GD lr 0.03 | 20k | 8/8 | med 4 | 2.76e-6 | 0.43 | 0.80 |
| Adam eps 1, lr 1, wd 1e-3 | 10k / 20k | 8/8 | med 3.5 / 3.5 | 3.51e-6 / 3.72e-6 | 0.55 / 0.58 | 0.92 |
| Adam eps 10, lr 10, wd 1e-4 | 10k | 8/8 | {2:1, 3:1, 4:5, 5:1} | 3.76e-6 | 0.59 | 0.93 |
| normalized GD lr 0.1 wd 0.03 | 20k | 6/8 | med 3 | 5.9e-7 | 0.09 | 0.72 |
| normalized GD (wd 1e-3 / 0.1), Lion | | 0 | memorises / collapses | | | |

Transformer p=113 frac 0.3 (nu 6 measured, gamma = m_min / ||theta||^6):

| optimizer | epochs | grokked | K_abl | gamma med |
|---|---|---|---|---|
| AdamW wd 1 (reference) | 25k | 8/8 | 4 x8 | 1.46e-8 |
| momentum GD lr 0.03 wd 1e-3 | 50k | 3/3 | 3, 3, 3 | 1.19e-8 |
| momentum GD lr 0.01 | 50k | 2/3 | 4 (+ one unreadable, m_min 0.24) | |
| Adam eps 1, lr 0.3, wd 1e-3 | 50k | 3/3 | 3, 3, 4 | 1.17e-8 |
| Adam eps 10, lr 10 | 25k | 2/3 | 3, 4 | 8.7e-9 |
| plain GD lr 0.3 / 1.0 | 25k-50k | 0 | diverges | |
| normalized GD, Lion, gd-momentum lr >= 0.1 | | 0 | | |

Reading. Every l2-geometry optimizer that groks lands on K* (MLP mode 4
vs AdamW mode 5-6; transformer 3/3 at exactly 3 vs AdamW 8/8 at 4), so
the +1 is the sign-regime optimizer's, not the task's. Momentum GD is the
usable one: groks 100% in both architectures at lr 0.1 (MLP) / 0.03
(transformer), clean circuits (0.996), count = K*. Its normalized margin
is BELOW AdamW's at these budgets (0.64 vs 0.77 of the MLP ceiling; 0.82x
on the transformer) and still rising with epochs (0.58 -> 0.64 from 10k to
20k, best run 4.73e-6 ~ AdamW median) because GD-type dynamics
equilibrate at train CE ~0.05 instead of the 1e-7 Adam floor. Open: does
momentum GD at 50-100k epochs reach AdamW's margin at K = K*, or is the
K*+1 basin genuinely the higher-margin one (which would mean the
equal-share optimum is not where trained nets can get)? Large-eps Adam is
the same statement with one number changed in the record recipe.
