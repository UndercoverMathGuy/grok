# Heracles: complete results note (runs of 2026-09-11 to 09-12, written 09-13)

One document for everything trained in the heracles line. Per experiment: what it was, what came out (numbers),
what we conclude. Source tables: RESULTS_stage1.md, RESULTS_stage2.md, RESULTS_H3.md, RECIPES.md; scored JSON in
scores/ (committed); raw runs in runs/ (gitignored, 1,104 runs + 288 checkpoints, tarballs runs_stage1_full.tgz,
runs_stage2_full.tgz, runs_h3_full.tgz).

## 0. Setup common to every experiment

- Task: a + b mod p, p in {59, 71, 97, 113, 127}. Train fraction 0.5 unless stated, data_seed 911.
- Model: exactly homogeneous ReLU MLP, m 512 hidden, d = r = 128, degree dial nu in {3, 4, 5, 6} via extra linear
  layers (LAYOUT {3: (1,0), 4: (1,1), 5: (2,1), 6: (2,2)} = (pre, post) linear layers). Cross-entropy. No TF32.
- Optimizers: constant-geometry only. gd / gdm / sgd (l_2, q = 2); lq q in {3, 4, 6} (steepest descent in l_q with
  an l_q penalty); sign / signm + box (q = inf). AdamW and Lion excluded (mixed geometry).
- Count instrument: interface ablation (score.py), read-and-write rule: a frequency counts if ablating it at the
  input interface AND at the output interface each hurts. "clean" = fraction of members that pass both.
- Margin instrument: gamma_2 = min margin over the full grid / ||theta||_2^nu. "of ceiling" = gamma_2 divided by
  the constructed K* committee ceiling (construct_q.py): nu 4: 6.39e-6 at K 4; nu 6: 6.2e-8 at K 3.
- Predictions: kstar.json equal-share ladder, K* = argmax_K M_K / K^{nu/q}; plateau74 = {K : gamma_hat(K) >= 0.74 max}.
- Hardware: RTX PRO 4500 32 GB (EU-RO-1, stage 1 + 2) and L40S 48 GB (US-TX-3, H3), each with a network volume.
  All pods and volumes deleted after verified pulls. Total spend about $12.

## 1. Cohort R + R2 + Rkeep: recipe sweep (656 + 144 + 112 runs)

What: for each optimizer class, find the recipe that groks >= 7/8 seeds in 20k epochs, stays finite, and has the
highest median normalised margin in its own norm; ties to smaller wd. Cell: nu 4, p 113, frac 0.5, 8 seeds.
R ran on the first pod (died out of credit; scores survived, raw runs lost). R2 (lq q3/q4 re-grid) and Rkeep
(the 14 chosen cells) were retrained on the second pod for provenance.

Result: the frozen recipes (RECIPES.md).

| class | recipe | grok | gamma_2 of ceiling | clean | K_abl histogram |
|---|---|---|---|---|---|
| gd (record) | lr 3, wd 3e-4, clip 1 | 8/8 | 0.67 | 0.99 | {4:3, 5:4, 6:1} |
| gd no clip | lr 1, wd 3e-4 | 8/8 | 0.57 | 0.97 | {4:1, 5:5, 6:2} |
| gdm mu 0.9 | lr 0.3, wd 3e-4, clip 1 | 8/8 | 0.60 | 0.98 | {5:6, 6:2} |
| sgd B 1024 | lr 1, wd 3e-4 | 8/8 | 0.60 | 0.98 | {4:2, 5:5, 6:1} |
| lq q 3 | lr 0.03, wd 1e-4 | 8/8 | 0.57 (gamma_3 9.2e-3) | 0.99 | {4:1, 5:4, 6:3} |
| lq q 4 | lr 0.03, wd 1e-4 | 8/8 | 0.44 (gamma_4 0.47) | 0.99 | {5:5, 6:2, 7:1} |
| sign + box | lr 1e-4, box 0.2 | 8/8 | 0.16 (gamma_inf 4.6e4) | 0.98 | {6:5, 7:2, 10:1} |
| signm + box | lr 1e-4, box 0.2 | 8/8 | 0.16 | 0.98 | {5:1, 6:4, 7:2, 8:1} |
| MIXED control (sign + l2 shrink) | lr 3e-4, wd 1 | 8/8 | 0.19 | 0.96 | {5:4, 6:2, 7:1, 8:1} |

Boundaries: wd 3e-3 never groks (weights decay to zero first); wd 1e-4 at 20k is under-equilibrated (0.10 of
ceiling); lr 3 without clip diverges in 1-6/8 seeds; box 0.05 cannot represent the task; lq q3 collapses at
wd >= 0.03; MIXED at lr 1e-4 memorises forever (train CE 1e-34, 0/8 grok).

Rkeep replicated R to seed noise (gd 0.67 of ceiling both times, mode 5 both times; gdm/sgd 0.61/0.66; sign 0.16).

Finding that reshaped the H design: under l_2 GD the count is 5 = K* + 1 in the high-margin regime (wd 3e-4:
15/24 runs at 5, none below 4) and 3-4 in the low-margin regime (wd 1e-3: 0.36-0.42 of ceiling, clean 0.63-0.86,
members dropped). The older "plain GD sits at K* exactly" came from the wd 1e-3 regime. R2 showed the same
direction at q 3 and q 4 (count rises as wd falls at fixed budget).

Conclusion: one record recipe per class; the gd record transfers across the whole cubby without retuning
(H1). The count-vs-wd effect is real and became H3's question.

## 2. H1: p x nu grid, q = 2 (320 runs)

What: p {59, 71, 97, 113, 127} x nu {3, 4, 5, 6}, record recipe (lr 3, wd 3e-4, clip 1), 20k epochs, 8 seeds,
with lr 1 x 8 seeds riding along as the fallback. Registered: mode in {K*, K* + 1} and >= 90% of grokked runs in
plateau74. Kill: nu-6 mode outside {3, 4} at >= 2 primes, or plateau < 90% at >= 3 cells.

Result: mode of K_abl (K* in brackets).

| nu | p 59 | p 71 | p 97 | p 113 | p 127 |
|---|---|---|---|---|---|
| 3 | 5 (6) | 5 (5) | 5 (6) | 5 (6) | 6 (7) |
| 4 | 4 (4) | 4 (4) | 5 (4) | 5 (4) | 4/5 (5) |
| 5 | 4 (3) | 4 (3) | 4 (3) | 4 (4) | 4 (4) |
| 6 | 3 (3) | 4 (3) | 4 (3) | 4 (3) | 4 (3) |

- |mode - K*| <= 1 in 20/20 cells, monotone in nu at every prime, no fitted parameter.
- Clause (a) mode in {K*, K* + 1}: 15/20. All five misses are K* - 1: four at nu 3 (flat ladder, plateau 3..12)
  and the 4/5 tie at p 127 nu 4. The sharp rows (nu 5, 6) sit at K* + 1.
- Clause (b) plateau >= 90%: fails in 4 cells (59/nu6 0.88, 71/nu5 0.88, 71/nu6 0.75, 113/nu6 0.88), 1-2 seeds
  at K* + 2 in plateaus only 3 wide. Registered kill for (b) FIRED; recorded as such.
- lr 1 fallback: groks 8/8 in 19/20 cells with 2-5x lower gamma_2 and wider histograms. Never needed.
- nu-6 kill rule: not triggered.

Conclusion: the l_2 ladder predicts the count to within one feature everywhere in the cubby with zero free
parameters, and the sign of the offset follows the ladder's shape (low where flat, high where sharp). The measured
count moves less than K* (4-5 almost everywhere for nu >= 4 while K* runs 3-5). The plateau clause as registered
is dead; the point prediction +-1 is alive.

## 3. H2: wd x frac (64 runs, nu 4 and nu 6)

What: p 113, record recipe, wd {1e-3, 3e-4} x frac {0.3, 0.7} x 8 seeds, 60k epochs with a 20k checkpoint.
Registered: frac does not move the mode by >= 1 at fixed wd; plateau >= 90%; frac moves grok time only.

Result, nu 4 (K* 4):

| frac | wd | mode 20k | mode 60k | hist 60k | of ceiling | clean |
|---|---|---|---|---|---|---|
| 0.3 | 3e-4 | 5 | 5 | {4:3, 5:4, 7:1} | 0.50 | 0.94 |
| 0.3 | 1e-3 | 4 | 4 | {3:1, 4:5, 5:2} | 0.48 | 0.96 |
| 0.5 | 3e-4 | 5 | (R) | {4:3, 5:5} | 0.67 | 1.00 |
| 0.5 | 1e-3 | 4 | (R) | {3:1, 4:6, 5:1} | 0.42 | 0.85 |
| 0.7 | 3e-4 | 4 | 4 | {3:1, 4:6, 5:1} | 0.70 | 1.00 |
| 0.7 | 1e-3 | collapse | collapse | 0/8 grok, weights -> 0 | - | - |

Result, nu 6 (K* 3): mode 4 in every grokking cell (frac 0.3: {3:2, 4:4, 5:2} at wd 3e-4, {3:3, 4:4, 5:1} at
wd 1e-3; frac 0.7 wd 3e-4: {4:6, 5:2}, 0.72 of ceiling); frac 0.7 wd 1e-3 collapses as at nu 4.

- 20k = 60k in every cell (modes identical, margins within 0.03); train CE flat from 5k epochs. These are
  finite-wd fixed points, so the wd dependence of the count is a property of wd, not budget.
- Frac statement: holds at nu 6 (4 everywhere) and at nu 4 wd 1e-3; marginal miss at nu 4 wd 3e-4 (5 at frac
  0.3/0.5, 4 = K* at frac 0.7). Frac 0.7 wd 3e-4 nu 4 is the best cell in the programme: mode K*, 0.70, clean 1.00.
- New recipe boundary: wd 1e-3 collapses at frac 0.7 (the admissible wd ceiling shrinks with frac).

Conclusion: frac is a margin/budget dial, not a count dial, apart from a one-step shift at the top of the
range that coincides with the highest margin; the count-vs-wd pattern of R is confirmed and is at equilibrium.

## 4. H3: the Wei limit, wd -> 0 (112 runs, 16 seeds per arm)

What: record recipe, p 113, frac 0.5. nu 4: wd {1e-3, 3e-4, 1e-4, 3e-5}, 600k epochs, checkpoints 20k/60k/200k.
nu 6: wd {1e-3, 3e-4, 1e-4}, 200k epochs, checkpoints 20k/60k. Wei et al. Thm 4.1 prediction: normalised margin
rises monotonically in 1/wd at matched budget, clean -> 1. Registered fork on the count: (a) mode -> K* as
wd -> 0 (the +1 is a finite-lambda effect) or (b) mode stays K* + 1 while the margin rises (the +1 belongs to the
l_2 max-margin landscape). Danger outcomes named in advance: count runs away upward; clean fraction falls.

Result, nu 4 (K* 4, plateau 3..8), 600k finals:

| wd | mode | histogram | of ceiling | clean |
|---|---|---|---|---|
| 1e-3 | 4 | {3:5, 4:7, 5:4} | 0.52 | 0.89 |
| 3e-4 | 5 | {4:6, 5:9, 6:1} | 0.64 | 0.99 |
| 1e-4 | 5 | {4:5, 5:8, 6:2, 7:1} | 0.69 | 0.99 |
| 3e-5 | 4 | {4:9, 5:5, 6:2} | 0.74 | 1.00 |

Result, nu 6 (K* 3, plateau 2..5), 200k finals:

| wd | mode | histogram | of ceiling | clean |
|---|---|---|---|---|
| 1e-3 | 4 | {3:1, 4:9, 5:5, 6:1} | 0.70 | 1.00 |
| 3e-4 | 4 | {3:1, 4:8, 5:7} | 0.62 | 1.00 |
| 1e-4 | 4 | {3:4, 4:8, 5:4} | 0.70 | 1.00 |

- Wei holds at nu 4: margin 0.52 -> 0.64 -> 0.69 -> 0.74 monotone in 1/wd, clean -> 1.00. 200k = 600k in every arm
  (wd 3e-5 reaches 0.66 by 60k, 0.72 by 200k, 0.74 at 600k): equilibrated.
- Count never runs away: no arm past mode 5; 6-7 in <= 3/16 seeds; nothing dense; nothing collapsed.
- Tie-break toward K*: the nu 4 4:5 split goes 6:9 -> 5:8 -> 9:5 as margin rises. Fork (a), weakly, as a shift
  of a near-even split. Ladder says why: gamma_hat(5)/gamma_hat(4) = 0.98, the top two rungs are tied within 2%.
- nu 6: mode K* + 1 = 4 at every wd; seeds at K* = 3 appear only at the smallest wd and grow with budget
  (2 -> 3 -> 4 of 16 at 20k/60k/200k). Fork (b) drifting toward (a). gamma_hat(4)/gamma_hat(3) = 0.94.
- Irregularity: nu 6 margin not monotone in wd (0.70, 0.62, 0.70). Does not affect the count reading.
- wd 1e-3 is R's low-margin regime (clean 0.89, members dropped) and is where "GD at K* exactly" came from.

Conclusion: the +1 is a finite-wd, finite-budget offset inside a nearly flat plateau corner, not an optimizer
property and not a second optimum. Paper sentence: predicted K*, measured K* or K* + 1, tie-break moving toward
K* as wd -> 0, margin approaching a constructed ceiling (0.74 of it at nu 4).

## 5. H4: geometry dial q in {2, 3, 4, 6, inf} (320 runs)

What: p 113, nu {4, 6}. Recipe per (nu, q) chosen post hoc by the RECIPES rule from a 3 x 3 grid lr {0.01, 0.03,
0.1} x wd {1e-5, 1e-4, 1e-3} (nu-4 q3/q4 reused from R2); sign + box at nu 6 from lr {1e-4, 3e-4} x box {0.2, 0.4}.
Prediction (l_q ladder K* = argmax_K M_K K^{-nu/q}): nu 4: 4, 8 (plateau 4..12), dense, dense, dense; nu 6: 3, 4,
6, dense, dense. Kill: nu-6 mode not monotone in q.

Result: measured mode (histogram):

| nu | q 2 | q 3 | q 4 | q 6 | q inf |
|---|---|---|---|---|---|
| 4 | 4-5 | 5 {4:1,5:4,6:3} | 5 {5:5,6:2,7:1} | 5 {5:3,6:3,7:1,8:1} | 6 {6:4,7:2,8:1,10:1} |
| 6 | 4 {4:5,5:2,6:1} | 4 {3:2,4:6} | 4 {3:1,4:4,5:2,6:1} | 4 {3:1,4:6,5:1} | 4 {4:3,5:3,7:1} (7/8 grok) |

- Kill FIRED: nu-6 count is 4 at every q; q 4 and q 6 miss 6 / dense outright. nu 4 rises only weakly with q
  (4-5 -> 6-7) and never approaches 8..12 / dense.
- Every l_q cell grokked 8/8 with train CE 4e-4 to 2e-9, far more equilibrated than gd (1e-2). Not a budget effect.

Control 1, cross-norm matrix (nu 4, median gamma_q; rows = training geometry, columns = evaluation norm):

| trained | gamma_2 | gamma_3 | gamma_4 | gamma_6 | gamma_inf |
|---|---|---|---|---|---|
| gd | 4.27e-6 | 3.9e-3 | 0.093 | 1.7 | 24 |
| lq 3 | 3.66e-6 | 9.2e-3 | 0.39 | 13 | 1.6e3 |
| lq 4 | 2.84e-6 | 9.1e-3 | 0.47 | 20 | 4.4e3 |
| lq 6 | 2.29e-6 | 8.0e-3 | 0.46 | 24 | 1.1e4 |
| sign + box | 1.03e-6 | 3.7e-3 | 0.22 | 13 | 4.5e4 |

Diagonal dominance in every column: each optimizer finds the best solution in its own norm. The optimizers are
cleared; the ladder is at fault.

Conclusion: the l_q generalisation of the ladder is WITHDRAWN as a count predictor for q > 2 (THEORY.md status
line). q only weakly raises the count at nu 4 and not at all at nu 6; no geometry in the family produces a dense
solution. The l_2 case is untouched.

## 6. H6 / construct_q: constructed l_q ceilings (local CPU)

What: exact product-grid K-committee nets, K = 1..12 from kstar.json plus a dense 56-frequency net, optimally
balanced in each l_q norm, amplitudes polished; compared to the trained winners.

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

- At q = 2 the construction is a true ceiling (trained at 0.64-0.67 of it, argmax_K = K* at both nu).
- At q > 2 trained nets beat the best committee construction by 1-4 orders of magnitude.

Why (mechanism, worked out after the kill): l_2 is a nuclear-type norm on a linear chain, blind to how a map is
spread across neurons; for q > 2 splitting a map across n rows costs n^(2/q - 1) < 1, so the l_q optimum spreads
every frequency across all of d, m, r at near-equal magnitudes. Per block the ideal carrier is
sign(cos)|cos|^(1/(q-1)) with gain G_q over the cosine (nu 4: 1.07 / 1.17 / 1.35 / 2.63 for q 3 / 4 / 6 / inf);
the remainder is superposition cross-talk, which has no closed form. The committee ladder's per-frequency cost
K^{nu/q} does not describe that regime.

Conclusion: the ladder's K-committee family is the right object only at q = 2. Good news for the paper: l_2 is what
weight decay is.

## 7. H5: optimizer robustness within l_2 (32 runs)

What: gdm (lr 0.3, clip 1) and sgd (lr 1, B 1024) at wd 3e-4, nu {4, 6}, 60k epochs with a 20k checkpoint, 8
seeds; compared with gd. Prediction: same mode and plateau, gamma within 20%.

| nu | gd | gdm | sgd |
|---|---|---|---|
| 4 | mode 5, 0.67 | mode 5 {4:1, 5:7}, 0.61 | mode 5 {4:1, 5:6, 6:1}, 0.63 |
| 6 | mode 4, 0.64 | mode 3 {3:4, 4:2, 5:2}, 0.70 | mode 4 {3:2, 4:3} (5/8 grok), 0.80 |

Conclusion: at nu 4 the statement holds. At nu 6 momentum and minibatch reach a higher margin (0.70-0.80) and a
count one nearer K* = 3, consistent with H3: higher margin fraction, count nearer K*. 20k = 60k in every cell.

## 8. Overall conclusions

1. The l_2 max-margin ladder predicts the feature count to within +-1 at every (p, nu) in the cubby with no
   fitted parameter, and the sign of the offset follows the ladder's shape.
2. The +1 seen at sharp rows is a finite-wd tie-break inside a plateau corner whose top two rungs are within 2-6%
   of each other. Taking wd -> 0 moves the split toward K* and the margin to 0.74 of a constructed ceiling.
3. wd and frac move the count by at most one step, and only through the margin regime; they are not count dials.
4. Every constant-geometry optimizer reaches the optimum of its own norm; gd, gdm, sgd agree within l_2.
5. The l_q ladder is dead for q > 2: width-priced norms spread features across the whole net and the committee
   family is 10-10^4 below the trained optimum. Only l_2 (weight decay) is committee-shaped.
6. The registered plateau clause (>= 90% in plateau74) fails as written and is dropped; the point prediction
   stays. Nothing in 1,104 runs went dense, and nothing collapsed at an admissible recipe.
7. Paper sentence: "predicted K*, measured K* or K* + 1 across five primes and four degrees, the tie breaking
   toward K* in the wd -> 0 limit, with margins reaching 0.74 of a constructed ceiling."

## 9. Registered statements: scorecard

| cell | statement | outcome |
|---|---|---|
| H1 (a) | mode in {K*, K* + 1} | 15/20; misses all K* - 1 (nu 3 flat ladder, p127 nu4 tie) |
| H1 (b) | >= 90% in plateau74 | KILLED (4 cells below 0.90) |
| H1 nu-6 kill | mode outside {3, 4} at >= 2 primes | not triggered |
| H2 | frac moves mode < 1 at fixed wd | holds at nu 6 and nu 4 wd 1e-3; marginal miss nu 4 wd 3e-4 |
| H3 margin | monotone in 1/wd, clean -> 1 | holds at nu 4; nu 6 margin non-monotone, clean 1.00 |
| H3 fork | (a) K* vs (b) K* + 1 as wd -> 0 | (a) weakly at nu 4; (b) drifting to (a) at nu 6 |
| H4 | nu-6 mode monotone in q | KILLED (4 at every q); l_q ladder withdrawn |
| H5 | gdm / sgd = gd | holds at nu 4; nu 6 one step nearer K* with higher margin |

## 10. Infrastructure notes for the next run

- RTX PRO 4500 saturates at about 40 runs per batch (1242 run-epochs/s); nu-6 layout OOMs at 144 on 32 GB, use
  <= 72 with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. L40S handled 64 x 600k at about 1850 run-epochs/s.
- torch.compile mode reduce-overhead crashes on the second compile in one process (Dynamo "ValueError: stoi");
  jobs.py now defaults to mode "default".
- Network volumes exist only in some data centres; MIG 24 GB is US-PA-1 only (no volumes). No mid-run tarring.
- Score on the pod alongside training only when the pod is otherwise idle; under contention it fell to
  45 min per cell, while the Mac scores at about 4 s per run.
- Pods and volumes created and deleted this run: vo9xcls79cik7f (died, out of credit), vs2z7ht3uk9nyj,
  90brisuaex662e, fm3bmxbopmvqmm; volumes avlg2t5txg, 14r5qnlle5, 41kfyuj9bl. lyu-pilot pd4m2buaf5q8b4 untouched.
