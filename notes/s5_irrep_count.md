# S5: the plain-l2 max-margin irrep count, predicted then read (2026-09-04)

Order of operations (this session): theory computed first from the S5
character table (cloud/groups.py) with no S5 committee read; committees of
the 48 existing S5 runs read afterwards (runs_degrok2/s5120_* 8 cells x 4
seeds, 100k epochs, frac 0.6-0.9, wd 1-2, 1L ReLU transformer d_mlp 512;
runs_tasks/s5_* 4 cells x 4, 25k epochs, frac 0.3/0.5).

## Theory (same object as mod-add, irreps in place of frequencies)

Logit ansatz: L(a,b,c) = sum_rho A_rho chi_rho(a b c^-1)/d_rho. For wrong
answer with offset g != e the margin is m(g) = sum_rho A_rho (1 - chi_rho(g)/d_rho):
coverage of class g by irrep rho is 1 - chi/d (the analogue of 1 - cos).
Cost of a d-dim irrep circuit at amplitude A: nu (A/e_rho)^{2/nu} with
efficiency e_rho ~ d^{-alpha/2}... written as cost = d^alpha A^{2/nu};
alpha = 1 from feature counting (d^2 entries of RMS 1/sqrt d per factor),
alpha in {0, 1.2} as sensitivity bounds. Max-min over all 64 supports of
the six nontrivial irreps (nu = 6 and 4).

Coverage table (classes (12), (12)(34), (123), (123)(45), (1234), (12345)):
  sign     d1  [2, 0, 0, 2, 2, 0]      zero on even classes
  std      d4  [.5, 1, .75, 1.25, 1, 1.25]
  std_sign d4  [1.5, 1, .75, .75, 1, 1.25]
  5a       d5  [.8, .8, 1.2, .8, 1.2, 1]
  5b       d5  [1.2, .8, 1.2, 1.2, .8, 1]
  6        d6  [1, 1.33, 1, 1, 1, .83]

PREDICTION: K* = 1. Every faithful S5 irrep covers every nontrivial class
by itself (worst class 0.5-0.83), unlike a single frequency in Z_p (M_1 =
0.0015), so the sparsity-favouring l2 cost (exponent 2/nu < 1) makes any
second irrep a pure loss. Equal-share ladder at nu=6: gamma(2)/gamma(1) =
0.25-0.27, gamma(6)/gamma(1) = 0.02-0.03 — no plateau at all. Preferred
leader: std_sign at alpha=1 (worst 0.75 at d=4), 6 at alpha=0. The sign
irrep is never a leader (zero coverage on even classes). Morwani et al.
(L_{2,nu} norm) predict all six irreps.

Refinement, computed by hand before the readout of amplitudes: given a
std leader, adding sign at relative amplitude s lifts the worst class
(12) from 0.5 to 0.5 + 2s until it meets the next-worst class (123) at
0.75, i.e. s* = 0.125; beyond that no margin gain. Given a 5a or std_sign
leader the worst class set includes an even class, which sign cannot lift:
s* = 0.

## Readout (energy -> amplitude A_rho = sqrt(E_rho) d_rho; K_thr at 0.1 max)

Grokked: 32/32 of the frac >= 0.6 runs; 0/16 of the frac 0.3/0.5 runs
(those are memorised and DENSE: 6 irreps within 0.2-1.0 — dense is what
memorisation looks like, sparse is what generalisation looks like).

All 32 grokked runs: ONE faithful irrep at full amplitude — std 25, 5a 5,
std_sign 2 — plus the sign irrep at 0.08-0.15 in 26/32; every other irrep
<= 0.03. The 6-dim irrep never appears (max 0.03 in one run).
  K_thr hist {1: 11, 2: 21}; gap-detector identical.
  sign amplitude by leader: std -> 0.10-0.15 (predicted 0.125);
  5a -> 0.00-0.11 (predicted 0); std_sign -> 0.08 (predicted 0).
  Load-bearing check: with sign the worst-class margin rises x1.46 in
  every std-led run (25/25, from 0.53 at (12) to 0.78 at (123), exactly
  the predicted hand-off to the next class) and x1.00 in 5a/std_sign-led
  runs (7/7). Sign is present from epoch 0-125, before grokking (775-14k).

## Against the literature

- Chughtai, Chan, Nanda (ICML 2023, 2302.03025), S5: "1d sign and 4d
  standard most commonly learned; std_sign, 5a, 5b approximately equally
  and less often; 6d never learned"; "systematic bias toward lower-
  dimensional irreps". All three are what the plain-l2 optimum with cost
  ~ d^alpha gives: sparse, low-d preferred, sign as the cheap coverage
  patch, 6 (highest cost, no partner benefit) never. Our leader stats
  (std 78%, 5a 16%, std_sign 6%, 6: 0%) match theirs qualitatively.
- Stander, Yu, Fan, Biderman (ICML 2024, 2312.06581): S5/S6 networks
  contain a "sign circuit" plus coset circuits. The sign circuit is the
  s* = 0.125 patch on the std leader.
- Morwani et al. (ICLR 2024): dense (all irreps) under L_{2,nu}. Refuted
  under weight decay on their own non-abelian testbed: 1 (+ sign patch).

## What this settles and what it does not

Settles: the count is a coverage-table quantity — 3-4 for Z_113 because
single frequencies do not self-cover, 1 for S5 because faithful irreps do;
the +1 in S5 is the cheapest coverage patch and its amplitude is
predicted (0.125 vs 0.10-0.15). In mod-add all frequencies cost the same
and none is a distinguished patch, which is why the mod-add recruit's
identity is init-driven (coverage-greedy hit 1/24) while the S5 recruit's
identity is forced.
Does not settle: WHICH faithful irrep leads. std leads 78% although it has
the worst self-coverage of the faithful irreps and the theory prefers
std_sign (alpha=1) or 6 (alpha=0). Candidate: std's actual representation
cost is below d^alpha (its matrices are permutation matrices minus the
trivial block: sparse 0/1 entries, cheap for ReLU products); measurable
with the fixed-support instrument. Leader identity remains the init/cost
question, as in mod-add.
