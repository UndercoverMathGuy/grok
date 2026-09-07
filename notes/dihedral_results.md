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

# Fixed-support ladder: scored against notes/prereg_fixed_support_dihedral.md (2026-09-07, same day)

84 runs (arms out/both x K 2..7 x 6 seeds, rand4 x 6, free x 6; 20k
epochs, 5090, pod deleted), plus the 16 free 30k runs rescored with the
same instrument. Data runs_dihedral/fs_*/, scores
notes/fixed_support_dihedral_scored.json, log fixed_support_dihedral_pod.log.
gamma = exact full-grid min margin / ||theta||^6, medians over separated
grokked seeds. Ceiling = max_K median gamma, arm out: 4.36e-9 at K = 4.

  arm out (W_U projected, inputs free):
    K        2      3      4      5      6      7
    gamma  3.70   3.94   4.36   4.16   4.03   4.18   (e-9)
    r_meas 0.85   0.91   1.00   0.96   0.93   0.96
    K_thr  2,2,2,2,3 | 2x6 | 1,2,2,2,3,4 | 2,2,3,3,4,5 | 1,2,2,3,4,4 | 2,2,2,3,3,4
  arm both (W_E and W_U projected):
    gamma  2.76   1.06   1.00   2.67   1.63   2.82   (e-9), erratic: 12/31
           grokked seeds end below 1.5e-9 at train acc 1.0 (stuck)
  rand4 (both, random 4-set): 2.40e-9.  free (20k, same code path):
    2.65e-9, K_thr [6,7,7,7,10,13], sign 0.178.  free 30k: 2.63e-9,
    K_thr [4,4,4,5,5,5,5,5,6,6,7,7,8,8,9], sign 0.065.
  profiles (median top-6 rel amplitude): out [1,.55,0,0,0,0];
    both [1,.47,0,0,0,0]; rand4 [1,.76,.05,0,0,0]; free 20k
    [1,.62,.45,.32,.21,.14]; free 30k [1,.49,.37,.18,.15,.10].

Verdicts:
F1 PASS: argmax over allowed support is interior (K = 4).
F2 FAIL: arm-both r_meas at K = 2/4/5 = 0.98/0.36/0.95 vs closed form
   0.75/0.94/0.74; the decoupled AM-GM ladder does not describe D_113.
F3 FAIL (not killed): free gamma / ceiling median 0.604 (>= 0.5), but
   >= 0.56 in only 58% and >= 0.74 in only 37% (n = 19; range 0.02-1.07).
   The plateau law (97.5% within 26% on Z_p) does not hold on D_113.
F4 PASS but VACUOUS: 86% of free K_thr in {K : r_meas >= 0.74} = {2..7},
   because r_meas is flat over ALLOWED support — the networks do not use
   the extra irreps (K_thr median 2 at every K from 2 to 7).
F5 "cost feature": fixed-support finals are MORE hierarchical than free
   runs (4th member 0.00 vs 0.18-0.32). The hierarchy is not the lottery.
F6 PASS: gamma_out >= gamma_both at every K, ratio 1.3-4.4 (> 1.2 at the
   argmax: 4.35). Restricting the INPUT side to the output irreps costs
   margin => the input representation uses irreps outside the output
   support => pricing is coupled across irreps. This is the mechanism of
   the closed-form failure, now measured.
F7 FAIL as registered: rand4 (2.40e-9) > both_K4 (1.00e-9). Driven by 3/5
   stuck both_K4 seeds; rand4 vs out_K4 is 0.55 but that is a cross-arm
   comparison and was not registered.
F8 PASS: sign share medians 0.01-0.10 on fixed supports.
F9: out arm 100% grok at K >= 3; both arm 5/6, 5/6, 4/6, 6/6, 6/6 (K5 at
   67% < 80%, marginal fail). K = 2 grokked 5/6 in both arms.

Post-hoc (not registered): gamma by USED count in arm out — K_thr 2:
4.31e-9 (n 20), 3: 3.42e-9 (n 7), 4: 4.30e-9 (n 5), 5: 5.38e-9 (n 1);
free by K_thr shows no monotone trend (4: 3.55e-9, 5: 2.37e-9, 7:
3.38e-9), the free spread is seeds ending with poor margins, not count.
Free count is NOT converged: the 20k free control has K_thr 6-13 vs 4-9
at 30k (same recipe) — D_113 cleanup is slow and "K_thr median 5" is an
epoch-dependent number.

Reading. On D_113 (1L ReLU transformer) the constrained max-margin
solution uses ~2 hierarchical two-dim irreps and reaches 4.4e-9; free
training over-recruits (5-13 used irreps, shrinking with epochs) and
lands at 0.6 of that ceiling with a 50x spread across seeds. The
decoupled per-irrep pricing that gives the Z_p and S5 closed forms is
wrong here (F2, F6): the 2-dim irrep circuit needs input irreps outside
its output support, so C_arch couples. Scope statement for the paper:
the closed-form count holds where pricing decouples (abelian; S5 with
K* = 1); for D_n the count needs measured cost and even then the
trained count sits above the constrained optimum and below the ceiling.
No D_n count claim; the measured-ladder machinery is the deliverable.
