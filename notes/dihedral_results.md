# Dihedral D_113: scored against notes/prereg_dihedral.md (2026-09-05, same day)

16 runs (frac 0.5 / 0.7 x wd 1 x 8 seeds, 1L ReLU transformer d128 mlp512,
30k epochs, one 4090, pod deleted). 15/16 grokked (>= 0.99 test; the one
miss sits at 0.986). Data: runs_dihedral/<cell>/s<seed>/{spectra.npz,
config, metrics}; scored by cloud/score_dihedral.py -> notes/
dihedral_scored.json (interim cell-1 score in dihedral_interim.json).

Frozen predictions vs outcome (grokked runs, n = 15):

D1 count (K* = 3 two-dim irreps, mode 4 under AdamW, >= 70% in {3,4,5}):
   FAIL as registered. Gap detector K_gap pooled {1: 4, 2: 3, 3: 1, 4: 4,
   5: 2, 6: 1}: bimodal at 1 and 4, only 47% in {3,4,5}. The threshold
   count K_thr (rho's with A >= 0.1 x leader) is unimodal: [4,4,4,4,5,5,5,
   5,6,6,6,7,7,7,9], median 5. The discrepancy is the amplitude PROFILE:
   D_113 committees are hierarchical, median top-6 relative amplitudes
   [1, 0.49, 0.37, 0.18, 0.15, 0.10], where Z_113 transformer committees
   are democratic ([1, .85, .69, .60, .46]). The gap detector finds the
   largest log-gap and lands on 1 in a steep ladder. Same instrument
   caveat as the quad transformer.
D2 sign irrep absent (< 0.05 of leader in >= 80%; KILL if median >= 0.10):
   FAIL, not killed. Pooled median sign share 0.065; 33% below 0.05, 73%
   below 0.10; max 0.24. Sign is small but present. Hard-margin reading
   (reflections over-covered 3.6x, so sign never binds) is not what CE at
   the floor does: the sign circuit is the cheapest possible irrep (one
   parity bit) and raises the logit gap against ALL 113 opposite-parity
   wrong answers, which cross-entropy pays for even when they are not the
   binding class. Finite-scale CE vs hard margin, as in ce_equilibrium.py.
D3 Morwani-dense (all 57 nontrivial irreps at equal mass) killed if
   K_thr <= 8 in every run: strict rule NOT met (14/15 <= 8, one run at 9).
   Equal mass over 57 is nonetheless excluded by the profile (sixth irrep
   at 0.10 of the leader, 50 irreps below 0.03).
D4 memorised runs dense: 1 non-grokked run, K_thr 10 vs grokked median 5
   (n = 1, direction consistent, no claim).
D5 identity init-driven: PASS. 15 grokked runs have 13 distinct leaders,
   mean pairwise committee Jaccard 0.02.

Reading. Sparse yes (5 of 57), sign small yes, identity init-driven yes;
but the equal-share ladder that fits Z_p and S5 does not describe D_113:
the count by threshold is 5 (one above the Z_113 transformer mode) and the
allocation is steeply hierarchical. The closed-form reduction "D_n = Z_n
program" assumed decoupled pricing with identical cost per two-dim irrep;
in a 1L transformer a two-dim irrep circuit must realise the matrix
product rho(a) rho(b) (non-commuting), not an additive phase, so C_arch
need not decouple across irreps or match the frequency circuit's AM-GM
cost. This is the same scope boundary the framework already declares for
non-decoupled pricing (seeded_ensemble_theory.md sec 1, quad): the count
must then come from the MEASURED constrained sup (fixed-support training
on D_113), not from the closed form. Not run. Until it is, D_113 is a
registered partial failure of the closed form, not of the framework, and
must be reported that way. No group ladder claim goes in the paper on
this evidence.
