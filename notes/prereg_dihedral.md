# Prereg: dihedral D_113 irrep count (2026-09-05, before any run)

Theory computed first (cloud/dihedral.py), no D_n network has ever been
trained in this repo. Purpose: the S5 result says the count is a
coverage-table quantity (1 irrep because faithful S5 irreps self-cover every
class at >= 0.5); Z_113 says 3-4 because a single frequency covers the
nearest offsets by only 1 - cos(2 pi/113) = 0.0015. D_113 is NON-ABELIAN
(|G| = 226, 58 irreps) but its faithful irreps are two-dimensional with
chi_j(r^k) = 2 cos(2 pi j k/113): they cover rotation classes EXACTLY like
Z_113 frequencies and every reflection by 1. So the same formula gives:

FROZEN PREDICTIONS (1L ReLU transformer, nu = 6, AdamW recipe of the S5 arm)
- D1  count: committee of two-dim irreps with K* = 3, K = 4 near-tie
  (ladder r(K): 2: 0.749, 3: 1.000, 4: 0.941, 5: 0.740, 6: 0.578) => under
  AdamW the observed mode is K*+1 = 4, as for Z_113; >= 70% of grokked runs
  in {3, 4, 5}. KILL if modal K >= 7 or <= 1.
- D2  the sign irrep is ABSENT: reflections are covered by the rho's at
  K x the rotation coverage (3.6x at K*), so the binding class is always a
  rotation and sign (which covers only reflections) never pays. Sign
  amplitude / leader amplitude < 0.05 in >= 80% of grokked runs. (Contrast
  S5, where sign was predicted and found at 0.125 as a coverage patch.)
  KILL if median sign share >= 0.10.
- D3  Morwani L_{2,nu} dense prediction: all 57 nontrivial irreps at equal
  mass. KILL of Morwani-dense if K_thr (irreps >= 0.1 x max) <= 8 in every
  grokked run.
- D4  memorised (non-grokked) runs are dense, as in S5 (>= 20 irreps above
  0.1 x max).
- D5  which rho_j lead is init-driven (no distinguished frequency): the
  committee sets vary across seeds (mean pairwise Jaccard < 0.3), unlike S5
  where std led 78%.

Readout: irrep energies from spectra.npz (coeffs = sqrt(energy) over
[sign, rho_1..rho_56]); committee = committee_from_coeffs gap detector
(the transformer instrument) and K_thr at 0.1 x max; amplitude A_rho =
sqrt(E_rho) * d_rho as in the S5 note.

Design: cloud/sweep_dihedral.py, frac {0.5, 0.7} x wd 1 x 8 seeds, 30k
epochs, spectra every 100. Grokked = final test acc >= 0.99.
