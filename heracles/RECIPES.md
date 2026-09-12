# Frozen recipes from cohort R (2026-09-11)

Cell: relu nu 4 (layout (1,1)), m 512, d = r = 128, p 113, frac 0.5, data_seed 911, 8 seeds, 20k epochs.
Selection rule (PLAN.md): >= 7/8 grokked (test acc >= 0.99), all finite, then highest median normalised
margin (gamma_2 for q = 2 classes, gamma_q otherwise), ties -> smaller wd. Counts below are the FULL
interface-ablation instrument (read-and-write rule on; scores/Rsides). "of ceiling" = gamma_2 /
6.39e-6, the constructed K = 4 free-amplitude ceiling for this cell.

| class | recipe | grok | gamma_2 of ceiling | clean | K_abl histogram |
|---|---|---|---|---|---|
| gd (record, q = 2) | lr 3, wd 3e-4, clip 1 | 8/8 | 0.67 | 0.99 | {4:3, 5:4, 6:1} |
| gd, no clip (alt) | lr 1, wd 3e-4 | 8/8 | 0.57 | 0.97 | {4:1, 5:5, 6:2} |
| gdm (mu 0.9) | lr 0.3, wd 3e-4, clip 1 | 8/8 | 0.60 | 0.98 | {5:6, 6:2} |
| sgd (B 1024) | lr 1, wd 3e-4 | 8/8 | 0.60 | 0.98 | {4:2, 5:5, 6:1} |
| lq q = 4 | lr 0.01, wd 0.03 | 8/8 | 0.35 (gamma_4 0.37) | 0.98 | {4:3, 5:3, 6:2} |
| lq q = 3 | pending cohort R2 (R grid collapsed all 72 runs to zero weights) | | | | |
| sign + box (q = inf) | lr 1e-4, box 0.2 | 8/8 | 0.16 (gamma_inf 4.6e4) | 0.98 | {6:5, 7:2, 10:1} |
| signm + box | lr 1e-4, box 0.2 | 8/8 | 0.16 | 0.98 | {5:1, 6:4, 7:2, 8:1} |
| MIXED control (sign + l2 shrink) | lr 3e-4, wd 1 | 8/8 | 0.19 | 0.96 | {5:4, 6:2, 7:1, 8:1} |

Boundaries measured in R (all q = 2 classes unless noted):
- wd 3e-3: 0/8 grok in every cell, every lr (weights decay to zero before signal grows).
- wd 1e-4: partial grok, margin 0.10 of ceiling at 20k epochs — not equilibrated; needs ~5x budget.
- lr 3 without clip: divergence in 1-6 of 8 seeds; clip 1 removes it at no margin cost.
- sign + box 0.05: 0/8 (box too tight to represent the task); box 0.1 groks but at 0.07 of ceiling.
- lq q = 3 at wd >= 0.03: total collapse (penalty-dominated early phase under the |v|^{1/2} map).
- lq q = 4 at wd >= 0.3: collapse; at wd 0.03 the runs sit at train CE ~0.5 (soft-margin regime).
- MIXED at lr 1e-4: 0/8 grok with train CE 1e-34 (memorises, never generalises).

Finding that changes the H design (see PLAN.md H3): under pure l2 GD-family optimizers the count is
5 = K* + 1 in the high-margin, clean regime (wd 3e-4: 15/24 runs at 5, none below 4) and 3-4 in the
low-margin regime (wd 1e-3: margin 0.36-0.42 of ceiling, clean 0.63-0.86, read-only members
dropped). The earlier "plain GD sits at K* exactly" (notes/clean_cohort_results.md) came from the
wd 1e-3 regime. Whether the near-limit count is K* or K* + 1 is now the question H3 answers by
taking wd -> 0 with a budget that scales as 1/(lr wd).
