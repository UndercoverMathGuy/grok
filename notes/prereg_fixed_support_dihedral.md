# Prereg: D_113 fixed-support ladder — the measured constrained optimum (2026-09-07, before any run)

Purpose. The closed-form "D_n = Z_n program" prediction FAILED on D_113
(notes/dihedral_results.md: K_thr median 5, hierarchical profile, sign
present at 0.065). The framework's own fallback (seeded_ensemble_theory
sec 1) is that when pricing does not decouple the count must come from the
MEASURED constrained sup gamma_hat(K) = sup{margin/||theta||^nu : logit
support <= K irreps}. This cohort measures it, the way P-N12 did for
Z_113 (fixed-support training is a lower bound on the constrained sup and
the "stronger maximiser" that beat every free run there).

Frozen closed form (cloud/dihedral.py, nu = 6): K* = 3, ladder r(K)
2: 0.749, 3: 1.000, 4: 0.941, 5: 0.740, 6: 0.578, 7: 0.442.
Free-run facts already in hand (15 grokked): K_thr = [4,4,4,4,5,5,5,5,6,6,
6,7,7,7,9], median 5; top-6 relative amplitudes [1,.49,.37,.18,.15,.10];
sign share median 0.065; 13 distinct leaders in 15 runs.

Design (cloud/fixed_support_dihedral.py; 1L ReLU transformer d128 h4
mlp512, AdamW lr 1e-3 (.9,.98) eps 1e-8 wd 1, frac 0.5, full batch, 20k
epochs — every free run grokked by epoch 6k; 6 seeds per cell, seeds drawn
with default_rng(20260907)). Projected AdamW: the group-token columns of
W_E and/or W_U are re-projected after every step onto the isotypic span of
the support (trivial + sign always included; the '=' token free).
  arm "out"   : W_U projected only, support = optimal Z_113 set S_K mapped
                to rho_j, K in {2..7}  -> true constrained sup over OUTPUT
                support (inputs may use any irrep)
  arm "both"  : W_E and W_U projected, same K  -> decoupled-circuit class
  arm "rand4" : both projected, K = 4 random support (per seed)
  arm "free"  : identity projection through the same code path (control)
Readout (cloud/score_fixed_support_dihedral.py): exact full-grid min
margin / ||theta||_2^6 from the final checkpoint; grokked = test acc >=
0.99; gamma_meas(K) = median over grokked seeds; r_meas(K) = gamma_meas/max;
K_thr, sign share, top-6 profile as in score_dihedral.py; the 16 existing
free runs are scored with the same instrument.

PREDICTIONS
F1  Interior optimum: argmax_K gamma_meas in arm "out" lies in {3, 4, 5}.
    KILL if it is at 2 or 7 (the constrained optimum is not resolved in
    the range, or the count is not a cost quantity on D_113).
F2  Closed-form shape: |r_meas(K) - r(K)| <= 0.15 at K = 2, 4, 5 in arm
    "both". PASS means the closed form was quantitatively fine and the
    09-05 failure was the gap detector on a hierarchical profile; FAIL
    means pricing is coupled (C_arch differs from the decoupled AM-GM
    cost) and the paper must say the D_113 count needs measured cost.
    Not a kill either way; the reading is registered here.
F3  Law 1 on D_113: gamma of every free grokked run (16 + 6) relative to
    max_K gamma_meas (arm "out") is >= 0.56 in 100% and >= 0.74 in >= 90%.
    KILL of the plateau law on D_113 if the median ratio < 0.5.
F4  Count in the measured plateau: >= 80% of free grokked runs have K_thr
    in {K : r_meas(K) >= 0.74} (arm "out"). KILL if < 60%.
F5  Hierarchy is a cost feature: fixed-support finals at K = 4 and 5 (arm
    "both") have a 4th-member relative amplitude < 0.5 (median), like the
    free runs. If >= 0.6 (democratic), the hierarchy is init-borne, not a
    property of the constrained optimum. Both readings registered.
F6  Superset ordering: gamma_out(K) >= gamma_both(K) (median) at every K
    with >= 4 grokked seeds. gamma_out/gamma_both > 1.2 at the argmax
    means the input side recruits irreps outside S (coupling is real).
F7  Coverage term is real: gamma_rand4 < gamma_both(4) by >= 10% (median).
    KILL of "support optimality matters on D_113" if rand4 >= both(4).
F8  Sign present on fixed supports: median sign share in 0.03-0.15.
F9  Grokking: >= 80% of seeds grok at K >= 3 in both arms; K = 2 may not
    (coverage 0.75 of max) — reported, not predicted.

Also reported: per-tensor norms, grok epochs, K_gap for comparability.
Output: notes/fixed_support_dihedral_scored.json + a section in
notes/dihedral_results.md.
