# S5 recipe (E1 / cohort SR, 2026-09-13, pod fsglufchnz325w L40S)   [scores/SR, analyze_S5.py --cohort SR]

Cell: S5 (|G| 120, label a @ b), relu nu 4 (layout (1,1)), m 512, d = r = 128, data_seed 911, 8 seeds, 20k epochs,
gd + clip 1. Selection rule as in RECIPES.md: >= 7/8 grok, finite, highest median gamma_2, ties -> smaller wd.
Counts = interface-ablation K_abl over irreps (read-and-write rule, fixed 2% floor: S5 has no null pool).
"leader" = faithful irrep with the largest folded amplitude; "share" = its fraction of faithful isotypic energy;
"sign_rel" = A_sign / A_leader.

| frac | lr | wd | grok | gamma_2 med | gamma_2 max | clean | K_abl | leaders | share med | sign_rel med |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.6 | 1 | 1e-4 | 0/8 (memorises by 20k, 1 late grok) | - | - | - | - | - | - | - |
| 0.6 | 1 | 3e-4 | 7/8 | 1.23e-6 | 2.24e-6 | 0.73 | {2:3, 3:3, 6:1} | std_sign 5, std 2 | 1.00 | 0.146 |
| 0.6 | 1 | 1e-3 | 0/8 (collapse) | - | - | - | - | - | - | - |
| 0.6 | 3 | 1e-4 | 8/8 | 1.35e-6 | 2.52e-6 | 0.76 | {2:3, 3:2, 4:2, 5:1} | std_sign 4, std 3, 5a 1 | 0.97 | 0.170 |
| **0.6** | **3** | **3e-4** | **8/8** | **2.22e-6** | 2.92e-6 | 0.76 | {1:1, 2:6, 3:1} | std_sign 5, std 2, 5a 1 | 1.00 | 0.169 |
| 0.6 | 3 | 1e-3 | 3/8 | 1.31e-6 | 1.73e-6 | 0.59 | {1:1, 2:1, 3:1} | std_sign 2, std 1 | 1.00 | 0.140 |
| 0.8 | 1 | 1e-4 | 6/8 | 5.42e-7 | 1.38e-6 | 0.85 | {1:1, 3:2, 4:1, 5:1, 6:1} | std_sign 3, std 2, 5a 1 | 1.00 | 0.141 |
| 0.8 | 1 | 3e-4 | 8/8 | 1.75e-6 | 2.47e-6 | 0.82 | {1:1, 2:3, 3:2, 4:1, 5:1} | std_sign 4, std 2, 6 1, 5a 1 | 0.98 | 0.140 |
| 0.8 | 1 | 1e-3 | 0/8 (collapse) | - | - | - | - | - | - | - |
| 0.8 | 3 | 1e-4 | 8/8 | 2.08e-6 | 2.90e-6 | 0.86 | {2:4, 3:2, 5:1, 6:1} | std_sign 4, std 3, 5a 1 | 0.93 | 0.153 |
| **0.8** | **3** | **3e-4** | **8/8** | **2.53e-6** | 2.88e-6 | 0.86 | {1:1, 2:4, 3:3} | std_sign 4, std 3, 5a 1 | 1.00 | 0.144 |
| 0.8 | 3 | 1e-3 | 0/8 (collapse) | - | - | - | - | - | - | - |

Record (frozen for S0 / S1 / S2 / S3): **gd lr 3, wd 3e-4, clip 1** at both fractions — the mod-add record transfers
unchanged. Grok times at frac 0.8: 800-2200 epochs (frac 0.6: 1800-5000).

Boundaries: wd 1e-3 collapses at S5 at both fractions (the admissible wd ceiling is lower than mod-add's at frac 0.5,
as it was at frac 0.7 there); wd 1e-4 at lr 1 memorises inside 20k (0/8 at frac 0.6); lr 3 wd 1e-4 groks 8/8 but at a
lower margin (not equilibrated in 20k, same as mod-add).

First readings (not the registered test; E2 is): every grokked run in every cell carries ONE faithful irrep (share
0.93-1.00) plus the sign irrep at 0.14-0.17 of the leader's amplitude; the ablation count is 2 (leader + sign) in the
mode, 1 when the sign patch does not clear the 2% floor, 3+ only in the low-margin cells. The leader under plain GD in
the MLP is std_sign in ~55% of runs, std ~30%, 5a ~10%, 6 once, 5b never — the reverse of the AdamW-transformer
prior (std 78%). E0's measured ladder decides which of these the l_2 ceiling favours; E2 tests it at 16 seeds.
