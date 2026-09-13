# heracles S5 plan (drafted 2026-09-12, schedule added 09-13; NOT YET REGISTERED, awaiting sign-off)

Why S5 next: mod-add is the equal-share case (every frequency is a 2-dim irrep of identical cost). S5's irreps
have dimensions 1, 1, 4, 4, 5, 5, 6, so the l_2 ladder becomes WEIGHTED: each irrep rho has its own representation
cost c_rho and its own coverage vector over the 6 nontrivial conjugacy classes (notes/s5_irrep_count.md). The
prediction is no longer an integer with a plateau but a NAMED irrep (plus a possible cheap patch), which is a
sharper test than mod-add. The l_q dial is out of scope (withdrawn, RESULTS_stage2.md); q = 2 only.

## A. Experiments list (sign off each line)

| # | name | what | runs | GPU time | gate |
|---|---|---|---|---|---|
| E0 | S-0 constructed weighted ladder | single-irrep nets for std, std_sign, 5a, 5b, 6 and the sign patch in the exact MLP at nu {4, 6}; balanced + polished gamma_2 per irrep = c_rho; pair ceilings (leader + sign at s*); argmax = predicted leader per nu, predicted s* | 0 (CPU) | 0 | must be committed BEFORE E2 is scored |
| E1 | S-R recipe sweep | gd lr {1, 3} x wd {1e-3, 3e-4, 1e-4}, clip 1, frac {0.6, 0.8}, nu 4, 8 seeds, 20k epochs; then gdm / sgd at the winner x 8 | 96 + 16 | ~50 min | RECIPES rule picks one gd recipe; not a hypothesis test |
| E2 | S-1 main | record recipe x nu {4, 6} x frac {0.6, 0.8} x 16 seeds, 60k epochs, ckpt 20k; full irrep-ablation instrument | 64 | ~75 min | statements (a)-(d) below |
| E3 | S-2 Wei limit at S5 | wd {1e-3, 3e-4, 1e-4} x 8 seeds at nu 4, frac 0.8, 200k epochs, ckpts 20k/60k | 24 | ~80 min | CONDITIONAL: runs only if E2 shows the leader or s* moving with wd across the S-R cells |
| E4a | S-3 S4 | record recipe, nu {4, 6}, 8 seeds, 20k; ladder from E0 machinery (irreps 1, 1, 2, 3, 3) | 16 | ~5 min | leader + count predicted before scoring |
| E4b | S-3 A5 | same, A5 (irreps 1, 3, 3, 4, 5) | 16 | ~10 min | same |
| E4c | S-3 D_113 | same, D_113 (|G| 226, 51k inputs); prediction = argmax constructed gamma over supports of <= 2 irreps (replaces the failed closed form of 09-07) | 16 | ~30 min | the registered hard case |

Total: 248 GPU runs, ~4.5 h of L40S-class time at S5 throughput (14,400 inputs vs 6,384 for mod-add, so about
half of yesterday's run-epochs/s), about $5 including the volume. Local CPU work: E0 plus the group tables for E4.

Registered statements for E2 (per cell, grokked runs only):
  (a) exactly one faithful irrep carries >= 90% of the non-sign isotypic energy in >= 90% of runs (K* = 1).
      Kill: >= 2 faithful irreps above 0.2 in >= 3/16 runs.
  (b) the leader is the E0 argmax irrep in >= 60% of runs at each nu. Kill: E0 argmax leads < 30%.
  (c) sign patch amplitude in [0.08, 0.17] when the leader is std; <= 0.05 otherwise.
  (d) gamma_2 >= 0.5 of the E0 pair ceiling (the construction is a real ceiling, as at mod-add: 0.64-0.74).
Prior from the transformer runs (09-04, AdamW): std led 25/32, 5a 5, std_sign 2, sign at 0.08-0.15, 6 never.
The MLP under gd is a fresh reading; (b) is where the weighting question is decided.

## B. Hour-by-hour schedule (T = sign-off; one pod, L40S secure + network volume; everything inside heracles/)

| hour | GPU | local (Mac) |
|---|---|---|
| T+0:00 to 1:00 | - | train.py: `task` field (modadd / s5 / s4 / a5 / d113), label table via cloud/groups.py convention, vocabulary |G|; snapshots write isotypic energies (PROJ) under the same npz keys. jobs.py: cohort_SR. CPU smoke on S4. Commit. |
| T+1:00 | create pod + volume, upload, launch E1 (112 runs, one batch of 112 at nu 4 fits an L40S) | - |
| T+1:00 to 2:30 | E1 training | construct_group.py: explicit irrep matrices (std = perm minus trivial; std_sign = sign x std; 5a, 5b, 6 from the regular rep via isotypic projectors + canonical decomposition), single-irrep and leader + sign nets, balance + polish (port of construct_q.py at q = 2). kstar_group.py: weighted ladder, predicted leader + s* per nu. |
| T+2:30 to 3:30 | E1 finishing / idle | score.py irrep adapter (ablate isotypic components at both interfaces, read-and-write rule unchanged). Pull E1, score, pick the recipe (RECIPES rule). Write E0 numbers + E2 statements into this file, RECIPES_S5.md. COMMIT = registration. |
| T+3:30 | launch E2 (64 runs, 60k, ckpt 20k) | - |
| T+3:30 to 4:45 | E2 training | group tables for S4, A5, D_113 (character tables, class maps, PROJ) in heracles/groups_extra.py; run construct_group.py for each; write E4 predictions into this file; commit. jobs.py: cohort_S3 (+ cohort_S2 ready, unlaunched). |
| T+4:45 to 5:00 | pull E2 | score E2 locally (~4 s/run), read (a)-(d). Decide E3 by the registered condition. |
| T+5:00 | launch E4a + E4b + E4c (48 runs) and, if triggered, E3 (24 runs, 200k) as one chain | write RESULTS_S5.md skeleton with the E2 tables |
| T+5:00 to 6:00 (to 7:30 with E3) | E4 (+E3) training | E4 predictions finalised before any E4 score is read |
| end | pull everything, verify, delete pod + volume | score E4 (+E3), RESULTS_S5.md complete, memory + commit |

Order rationale: E1 does not depend on the scorer or on E0, so the GPU starts at T+1:00 while E0 is written on CPU.
E0 is committed before E2 is scored, which is the only ordering the registration needs. E2 launches the moment the
recipe is chosen. E3 stays conditional as drafted, so the pod is deleted by T+6:00 in the base case.

## C. What is already known (notes/s5_irrep_count.md, 2026-09-04; 1L ReLU transformer, AdamW)
- Predicted K* = 1 faithful irrep (every faithful irrep self-covers every class; l_2 cost exponent 2/nu < 1 makes
  a second irrep a pure loss); sign irrep as a coverage patch at relative amplitude s* = 0.125 on a std leader,
  s* = 0 on 5a / std_sign leaders. Read: 32/32 grokked runs = one faithful irrep + sign at 0.08-0.15 (std-led),
  worst-class margin lifts x1.46 exactly as predicted. Matches Chughtai et al. 2023 and Stander et al. 2024;
  refutes Morwani's dense prediction non-abelian.
- NOT settled: WHICH irrep leads. std led 78% although the hand cost model (cost ~ d^alpha, alpha in {0, 1, 1.2})
  prefers std_sign (alpha 1) or 6 (alpha 0). Candidate explanation: std's true cost is below d^alpha (permutation
  matrices minus the trivial block: sparse 0/1 entries, cheap for ReLU products). This is exactly the weighting
  question, and it is what E0 answers by construction rather than by alpha.

## D. Objects

Architecture: the heracles ReLU MLP family unchanged (E stack -> U, V -> ReLU -> W stack -> logits over |G|),
nu in {4, 6}, m 512, d = r = 128; inputs (a, b), label a @ b (cloud/groups.py convention). Same optimizers as
the mod-add cubby (gd record recipe first; gdm / sgd as robustness), same interface-ablation scorer with
frequencies replaced by isotypic components (PROJ from groups.py; ablate irrep rho at the input interface and
at the output interface, read-and-write rule unchanged), gamma_2 = min margin / ||theta||_2^nu as before.

Measured ladder (replaces alpha): for each faithful irrep rho in {std, std_sign, 5a, 5b, 6} and for the sign
patch, build the single-irrep net in the exact architecture, balance blocks in closed form, polish, and record
c_rho = the constructed gamma_2 of the single-irrep net at each nu, then the constructed gamma_2 of each
(leader + sign) pair at the predicted s*. That gives the WEIGHTED ladder gamma_hat(support) with no free
parameter, hence a predicted leader per nu and a predicted patch amplitude.

## E. Code to write (heracles/, isolated; nothing imported from outside except cloud/groups.py's conventions, copied in)
- train.py: `task` field -> label table + vocabulary size; snapshots record isotypic energies via PROJ instead of
  Fourier coefficients (same npz keys, `energy` indexed by irrep).
- score.py: irrep adapter (ablate isotypic components at the two interfaces).
- construct_group.py: explicit irrep matrices, single-irrep and leader + patch nets, balanced gamma_2.
- kstar_group.py: the weighted ladder from the constructed c_rho and the coverage table.
- groups_extra.py: S4, A5, D_113 character tables, class maps, PROJ (S5 copied from cloud/groups.py).
- jobs.py: cohort_SR, cohort_S1, cohort_S2, cohort_S3.

## F. Out of scope
l_q geometries (withdrawn); AdamW / Lion (mixed geometry, controls only); frac < 0.6 at S5 (memorises, dense,
never groks in the transformer; one 8-seed frac 0.5 gd cell may be added to E1 as a boundary check if time allows).
