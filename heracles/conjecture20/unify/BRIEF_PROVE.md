# Brief PROVE: proving the reduced statements

**HARD RULES FROM THE USER (unchanged)**
- General proofs only, valid for EVERY odd prime p, EVERY n >= 2 and EVERY b in [2, nu] at once.
- No case analysis by (p, n).
- No "p >= p_0", "b near nu" or "large n" results; never ride a parameter to infinity.
- No parameter-specific computer-assisted certificates.
- Numerics only for sanity checks and counterexample searches.
- A dichotomy on a property of the NETWORK or of its ALLOCATION (e.g. efficient vs inefficient) is NOT a parameter
  case split and is allowed.

Read first:
- `unify/E_SYNTHESIS.md`, `unify/C_cc/IDENTITY.md` and `unify/BRIEF_CORE.md`;
- `unify/C_joint/REPORT.md` (round 1), and round 2 in `unify/C_joint/r2/` (joint2_proofs.tex and outputs);
- `unify/C_ccfactor/REPORT.md` and `unify/C_ccstab/REPORT.md` with their VERIFY.md;
- `unify/BRIEF.md` for notation.

## 1. The ladder of statements left to prove

The infinite family of cases has been reduced to the statements below. Proving them proves (C).

**(A) The single-set core RT_n** (C_joint J4: implied by RD in class C; the hardest identified piece).
- r = 1/s, s in [1, nu/2] and lambda = (n/(n+1))^{1-r}, so c_n(s) = 1 - lambda^s = 1 - (n/nu)^{s-1}.
- For every S in F, every alpha, beta in [0, inf)^S and every set of phases phi, the claim is

      min_{delta != 0} [ sum_l (alpha_l + beta_l) kappa_l(delta)  -  max_w ( U(w - delta) - U(w) ) ]
          <=  Gamma^{(S)}_s * ( sum_l alpha_l^r + lambda sum_l beta_l^r )^s ,

  where U(w) = sum_{l in S} beta_l cos(2 pi l w / p + phi_l) and w ranges over Z_p.
- Reading of the terms:
  - beta is "single-set" amplitude, cheaper than IF amplitude alpha by the factor lambda (n neurons instead of nu);
  - U(w - delta) - U(w) is the interference it creates at lag delta.
- This is a theorem about real trigonometric polynomials on Z_p. n enters only through lambda and the range of s.
- Facts any proof must respect:
  - it is an equality when beta = 0 and alpha is a programme maximiser;
  - (Legendre, J6) beta_l = 2/sqrt p on all of F with U = Legendre symbol has lag-depth 2 at EVERY delta, while
    sum_l beta_l kappa_l(delta) = sqrt p. So any detection that ignores the efficiency of the allocation fails. RD
    survives only because a flat allocation on all of F is an inefficient cover for s > 1 + O(1/p);
  - in the window s - 1 ~ p^{-1/2} the margin of safety is only about 3.4 (J11e).
- Partial results:
  - J2: sharp aligned single-set payment;
  - J7: Parseval closure when c_n^2 max_delta sum_{L'} kappa <= 1;
  - J8: p-uniform gain cap;
  - J9: one cheap feature is always caught;
  - N3: polynomial-tilt detection is exact;
  - N6/N7: second-order closure with skewness;
  - N5: third-order interactions need additive triples on a shared slot set;
  - arith2 (n=2): relations case closed for p <= 59, for sum_{L'} kappa < 29.7, and for b > 3 - O(p^{-1/2}).

**(B) Class C reduces to the core.** Multi-set families and non-aligned atoms must reduce to (A) plus per-feature
payments. Relevant facts:
- FR (J1) and PWE_C (N10, conjectured): some field row has programme coverage at most the IF value of the cost.
- CD_C (CC17, conjectured): h <= pi^C B_p M(P^s) for every class-C network, where P_l is the b-cost of feature l.
  It holds numerically at every allocation searched, and equality holds on the whole decoupled manifold.
- Ghost frequencies: neurons whose atoms cancel, giving beta_k = 0 but interference != 0.

**(C) Mixed networks reduce to class C** at the optimum: (MM_opt), through the channel lemma (SPLIT R4) and mode
budgets (E7). Per-network (MM) is false (SPLIT R10), so global optimality or a canonical reallocation is needed.

**(D) Upper half (FE) / non-straddling.** See E_reduce Thm C and Prop O, and the C_upper results when available.

## 2. Structural facts that should be used

- **Inefficient allocations are free.** The network's value is at most tau_n(s) M(a) / (sum a^r)^s, with a = alpha + beta.
  So if the allocation a is inefficient by more than the factor 1/tau_n(s) = 1 - c_n(s), then (A) holds trivially. The
  content is therefore NEAR-EFFICIENT allocations.
- **Near-efficient allocations are structured.** They are close to programme maximisers, whose supports have size
  K* = Theta(log p). Their binding offsets are where all l*delta/p (l in the support) are simultaneously badly
  spread, which is Bohr-set / Dirichlet-approximation arithmetic. Detection should be strongest exactly there.
- **The interference is real and one-dimensional.** U has spectrum in S, magnitudes beta_l, free phases. Its
  lag-differences have L2 norm^2 = sum_l beta_l^2 kappa_l(delta).
