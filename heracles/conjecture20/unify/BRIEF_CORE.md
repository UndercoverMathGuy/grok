# Brief CORE: the one statement left (RD) and the upper half (FE)

**HARD RULES FROM THE USER (unchanged)**
- General proofs only, valid for EVERY odd prime p, EVERY n >= 2 and EVERY b in [2, nu] at once.
- No case analysis by (p, n).
- No "p >= p_0", "b near nu" or "large n" results; never ride a parameter to infinity.
- No parameter-specific computer-assisted certificates.
- Numerics only for sanity checks and counterexample searches.

Read first: `unify/E_SYNTHESIS.md` (what is proved, refuted and equivalent), then `unify/BRIEF.md` (notation) and
`unify/BRIEF_E.md` (framework).

## 1. The targets

**(RD_S), restricted decoupling.** For every s in [1, nu/2] and every network theta whose clean coefficients
vanish outside S:
- gamma(theta; s) <= pi_{n,b}(p) B_p Gamma^{(S)}_s.
- Also with feature weights lambda near 1, where the margin uses sum_l lambda_l Re(Phi_l (1 - rho^{-l delta})).

By E_SYNTHESIS, Theorem T, RD for all S and lambda near 1 gives:
- the value;
- the lower half of (C);
- the union-of-supports statement;
- the exact count for generic weights.
Every entropic, dual, split and KKT form of RD is equivalent to RD itself.

**(FE), the upper half.** At every global maximiser theta*:
- H(theta*, s) >= H(nu E(theta*, s)) + H^(1)_min(s), where E_l is the cost-weighted output-energy share at frequency
  l.
- Together with RD it excludes "straddlers", i.e. maximisers whose feature set is a union of >= 2 optimal supports.
- It is false at non-global local maxima, so it must use global optimality.

## 2. A concrete pointwise form of the margin (class C first; verify every step)

Field decomposition (interference_framework.tex, Lemma dec):
- g(x, delta) = sum_l Re(Phi_l(x) (1 - rho^{-l delta})), where Phi_l(x) = (2/p) sum_i w^_i(l) prod_t u_it(x_t) rho^{l x_t}.
- E_x Phi_l = conj(beta_l).

Write Phi_l = A_l + f_l(x), with A_l = E_x Phi_l (real > 0 after rotation) and f_l of mean zero. Pairing delta with
-delta gives, at every x and every delta:

    sum_l (A_l + Re f_l(x)) kappa_l(delta)  -  | sum_l Im f_l(x) sin(2 pi l delta / p) |  >=  h .

In class C the fluctuations have this structure:
- f_l is supported on the characters 2l * 1_T (T nonempty).
- Fluctuations of different features are orthogonal, so their L2 norms add. Compensation is therefore a pointwise
  (flat-bottom) phenomenon, not an L2 one.
- Each neuron's field is a Riesz product: beta_i prod_t (1 + xi_it zeta_l^{x_t}), with zeta_l = rho^{2l} (check the
  conventions).
- The field is circular: E (Phi_l - A_l)^2 = 0, so the real and imaginary fluctuation energies are equal.

Payment and detection:
- **Payment (gain per feature).** E_width W6:
  - with phase-aligned atoms the gain is quadratic in the relative interval interference;
  - atom cancellation (L > |A|) can make it linear, with a constant growing like n^2.
  - The network's total gain over IF is at most the cost-weighted average of the per-feature gains.
- **Detection.** A single test sees either one feature (delta-Fourier isolation) or an L1/L2 aggregate. For K
  features with relative fluctuation eps, the gain is ~ eps^2. Detection by averages is ~ eps/sqrt(K). Generic
  minima over x are ~ eps sqrt(n log p / K).
  - So a general proof must capture the MINIMUM over x (extreme values) or the structure of the fluctuations
    (Riesz products, character sets 2l*1_T, the arithmetic of the binding offsets).
  - Equally, this heuristic says RD could FAIL for many features with flat-bottomed compensated fluctuations.
    Settle it.

## 3. Facts any proposed inequality must respect (all on disk)

- **p = 3 winners.** `exact_certificate.py`, `oned_certified/p3_n*.json`. RD holds there trivially (one frequency)
  while the IF value fails, so any inequality using nu^{1-s} instead of pi must fail at p = 3.
- **(5,2,2) mixed violator.** `unify/split/certify_viol.py`, `viol_p*_n2_b2.npz`. It breaks canonical-split
  decoupling at gamma = 0.79 gamma_IF.
- **E_dual D8/D9 networks.** `unify/E_dual/viol_entropy.py`, `pinned_profile.py`. They break Forster-type and
  profile-class identities.
- **E_width X1/X2.** IF networks break pointwise-in-s entropy inequalities in both signs.
- **Legendre construction** (arith2). It breaks support-only flatness lemmas.
- **(FE) fails at non-global local maxima** (`unify/E_reduce/verify/v4_optima.out`).
- **The (5,6) width-6 single-feature network** (LARGEN R10b).

## 4. General tools already proved (cite, do not re-prove)

From the earlier notes and rounds:
- the Renyi temperature identity (W1);
- convex-end rigidity (E3);
- no cheap feature implies decoupled for b in (2, nu) (E_kkt R4, R5);
- the static gain bound (D7);
- Theorem B / Corollary U / Theorem G / Corollary S;
- spectral Lemma 7 for every n (W6);
- aligned circular single features are never cheap (W8);
- a single neuron has h <= 0 (W9);
- mode budgets (E7);
- delta-Fourier isolation (p52 R12, W4);
- the frequency split and pinching (SPLIT, KM1);
- KM2 rigidity;
- Note Thm q1n2: pi_{2,b}(p) = 3^{1-s} for every p;
- the price-programme theorem.
