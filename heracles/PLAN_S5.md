# heracles S5 plan (drafted 2026-09-12, not yet registered)

Why S5 next: mod-add is the equal-share case (every frequency is a 2-dim irrep of identical cost). S5's irreps
have dimensions 1, 1, 4, 4, 5, 5, 6, so the l_2 ladder becomes WEIGHTED: each irrep rho has its own representation
cost c_rho and its own coverage vector over the 6 nontrivial conjugacy classes (notes/s5_irrep_count.md). The
prediction is no longer an integer with a plateau but a NAMED irrep (plus a possible cheap patch), which is a
sharper test than mod-add. The l_q dial is out of scope (withdrawn, RESULTS_stage2.md); q = 2 only.

## What is already known (notes/s5_irrep_count.md, 2026-09-04; 1L ReLU transformer, AdamW)
- Predicted K* = 1 faithful irrep (every faithful irrep self-covers every class; l_2 cost exponent 2/nu < 1 makes
  a second irrep a pure loss); sign irrep as a coverage patch at relative amplitude s* = 0.125 on a std leader,
  s* = 0 on 5a / std_sign leaders. Read: 32/32 grokked runs = one faithful irrep + sign at 0.08-0.15 (std-led),
  worst-class margin lifts x1.46 exactly as predicted. Matches Chughtai et al. 2023 and Stander et al. 2024;
  refutes Morwani's dense prediction non-abelian.
- NOT settled: WHICH irrep leads. std led 78% although the hand cost model (cost ~ d^alpha, alpha in {0, 1, 1.2})
  prefers std_sign (alpha 1) or 6 (alpha 0). The note's candidate explanation: std's true cost is below d^alpha
  (permutation matrices minus the trivial block: sparse 0/1 entries, cheap for ReLU products). This is exactly the
  weighting question, and it is the thing the heracles machinery answers by construction rather than by alpha.

## Objects

Architecture: the heracles ReLU MLP family unchanged (E stack -> U, V -> ReLU -> W stack -> logits over 120),
nu in {4, 6}, m 512, d = r = 128; inputs (a, b), label a @ b (cloud/groups.py convention). Same optimizers as
the mod-add cubby (gd record recipe first; gdm / sgd as robustness), same interface-ablation scorer with
frequencies replaced by isotypic components (PROJ from groups.py; ablate irrep rho at the input interface and
at the output interface, read-and-write rule unchanged), gamma_2 = min margin / ||theta||_2^nu as before.

Measured ladder (replaces alpha): for each faithful irrep rho in {std, std_sign, 5a, 5b, 6} and for the sign
patch, build the single-irrep net in the exact architecture (explicit irrep matrices: std = permutation rep
minus trivial; std_sign = sign x std; 5a, 5b, 6 numerically from the regular representation via the isotypic
projectors + a canonical decomposition), balance blocks in closed form, polish, and record c_rho = the
constructed gamma_2 of the single-irrep net at each nu, then the constructed gamma_2 of each (leader + sign)
pair at the predicted s*. That gives the WEIGHTED ladder gamma_hat(support) with no free parameter, hence a
predicted leader per nu and a predicted patch amplitude.

## Cohorts

S-R  recipe sweep (like R): gd lr {1, 3} x wd {1e-3, 3e-4, 1e-4} x clip 1, frac {0.6, 0.8} (S5 needs frac
     >= 0.6 to grok in the transformer; unknown for the MLP), nu 4, 8 seeds, 20k epochs = 96 runs. Selection by
     the RECIPES rule. gdm / sgd at the chosen (lr, wd) x 8 = 16 more.
S-0  constructed weighted ladder (local, CPU): c_rho per irrep per nu, pair ceilings, predicted leader + s*.
     REGISTERED BEFORE S-1 IS SCORED.
S-1  main: record recipe x nu {4, 6} x frac {0.6, 0.8} x 16 seeds = 64 runs. Registered statements per cell:
     (a) exactly one faithful irrep carries >= 90% of the non-sign isotypic energy in >= 90% of grokked runs
         (K* = 1; kill if >= 2 faithful irreps above 0.2 in >= 3/16 runs);
     (b) the leader is the S-0 argmax irrep in >= 60% of runs at each nu (kill if the S-0 argmax leads < 30%);
     (c) sign patch amplitude within [0.08, 0.17] when the leader is std, <= 0.05 otherwise;
     (d) gamma_2 >= 0.5 of the S-0 pair ceiling (the constructed ceiling is a real ceiling, as at mod-add).
S-2  wd -> 0 at S5 (the H3 analogue, only if S-1 shows a leader that depends on wd): wd {1e-3, 3e-4, 1e-4} x 8
     seeds at matched budgets, checkpoints; reads whether the leader identity and the patch amplitude move.
S-3  other groups with the same code path: S4 (irreps 1, 1, 2, 3, 3; K* and leader by the same ladder), A5
     (1, 3, 3, 4, 5), and D_113 as the registered hard case where the closed form failed (notes 09-07): the
     measured ladder replaces the closed form, prediction = argmax of constructed gamma over supports of <= 2
     irreps.

## Budget / order
S-0 is CPU and gates everything (registration). S-R + S-1 + S-2 = ~250 runs of 20k-60k at |G| = 120 (14400
inputs, 2.3x mod-add's 6384): ~4-5 h on an L40S-class card, ~$5. S-3 doubles that. Do S-0 first, in the gap while
H3 finishes, then register S-1 and launch S-R.

## Code to write (heracles/, isolated)
- train.py: `task` field (modadd | s5 | s4 | a5 | d113) -> label table + vocabulary size; snapshots record
  isotypic energies via PROJ instead of Fourier coefficients (same npz keys, `energy` indexed by irrep).
- score.py: irrep adapter (ablate isotypic components at the two interfaces).
- construct_group.py: explicit irrep matrices, single-irrep and leader+patch nets, balanced gamma_2 (port of
  construct_q.py's balance with q = 2).
- kstar_group.py: the weighted ladder from the constructed c_rho and the coverage table.
