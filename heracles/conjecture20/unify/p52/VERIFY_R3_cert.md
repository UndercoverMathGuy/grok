# Referee verdicts on p52 (cert lens; saved by the orchestrator)

## Overall
Both claims hold. I re-derived every step of each proof, re-ran every certificate, and wrote independent checks. All are in /home/user/grok/heracles/conjecture20/unify/p52/verify/ under the ref3_* names.\n\nR9 ((5,2) single feature, gamma <= 0.995 gamma_IF): CONFIRMED.\n- The chain R2 -> R3 -> R4 -> R5 -> R6 -> R7 -> R8 is complete and correct for arbitrary networks, i.e. any width, DC, multi-frequency inputs and all outputs.\n- My full rerun of chunk l3_4 ends 'DONE: I=146 II=461 fail=0 certs=1066'. Its log is identical line by line to cert_l3_4.log apart from timings.\n- The three earlier reruns (l0_7, l1_6, l2_5) are byte-identical to the official logs once timings are stripped. Totals are I=575, II=1540, certs=1842, fail=0, as reported.\n- The only rigour holes are sub-ulp:\n  - two unrounded float subtractions;\n  - the phi_pay and psi grids end at +-np.pi instead of +-pi.\n- I closed both in patched copies and re-ran all four chunks: every chunk ends fail=0, with box-for-box identical logs.\n\nR14 ((7,2), b in [2.5381, 3]): CONFIRMED.\n- I re-derived R11, R12, R13 and the relabelling.\n- My own differently designed B&B gives fails=0 on every claimed s-range.\n- mpmath 1.3.0's interval power and exp are not strictly outward-rounded, but the error (at most 3.2e-28 relative) is far smaller than the final upward rounding (at least 1.1e-16).\n\nNo network found by random, adversarial or optimised search beats the IF value. The best were 0.78176 at (5,2) and 0.85499 at (7,2), s = 1.182. The certified bounds are tight exactly at the IF configurations.\n\nNeither claim implies anything about the IF value at p=3. R12, on which R14 depends, does hold on the p=3 Conjecture-20 counterexamples.\n\nThe imported theorems are used within their hypotheses:\n- Note Thm c1sharp (computer-assisted; cited, not re-proved here);\n- Part I Thm A;\n- paper Lemmas 2 and 6 and Thm 8;\n- Rotfel'd's trace inequality.

## Salvage
Nothing needs salvaging. Recommended hardening of the code:\n\n(1) In bb52.try_certs, write ph_lo = dn(p_lo - a_hi) and ph_hi = up(p_hi - a_lo). In pay_adapt._cell_lb, wrap slo - p_hi and shi - p_lo in dn/up. Set the outer ends of the phi_pay grid (pg) and the psi grid (sg) to +-3.1415926535897936 so they enclose +-pi. With these changes all four chunks still end fail=0 with identical boxes (verify/ref3_patched/patched_l*.log).\n\n(2) In simple72_rig.py, either compute powers as exp(log) and widen them by a relative 1e-20, or state that mpmath's sub-ulp error is absorbed by the final up(). An independent B&B built this way passes (verify/ref3_r14_bb.out).\n\n(3) cbrt_hi should have a fallback, or an assertion, if the cube still fails to dominate after 6 iterations. This never triggers in practice.

## Verdicts

### R9: CONFIRMED

CLAIM: for every b in [2,3] and every network at (5,2) with beta_2 = 0, gamma <= 0.995 gamma_IF (computer-assisted).

PROOF RE-DERIVED (no gaps found).
- R2 mode budget: Parseval over all k in Z_5 (DC included); |what(1)|^2 + |what(2)|^2 <= (5/2)||w||^2; AM-GM gives total mode mass <= 4e^2; the 4 own modes each have modulus |beta_i|. Correct for arbitrary inputs and outputs.
- R3 Pythagorean cost: Minkowski in l^t with t = 2s >= 2, then (sum rho_i^r)^s >= sum rho_i since r <= 1. Correct.
- R4 margin ledger: q = v^* G v/(25M) is a probability, because 2f = 0 mod 5 forces f = 0 for f in {-2..2}^2.
  - Re-derived: 25 Chat_j(-chi) = (1/M) sum_delta c_j(delta) what_delta(f), with 2f = -chi.
  - Own pairs: f = 0 with conj(A); f = -1_T with conj(beta_T) (paper Lemma 6). The other 21 f for Phi_1 and 24 f for Phi_2 (phi_2(0) = conj(beta_2) = 0) are bounded by Cauchy-Schwarz with ||nonown|| <= R.
  - cert_hs uses exactly this set.
- R5 elimination of R: correct.
- R6 symmetries:
  - Conjugation x -> -x, c -> -c gives g'(x,d) = g(-x,-d), so alpha >= 0.
  - Slot swap.
  - Re A <= 0 gives h <= 0 by paper Lemma 2.
  - Torus equivariance: what'(f) = what(f) e^{-i<f,t>}, so a_T counter-rotates and the moduli and n are unchanged.
  - The payment depends only on (|B|, |C|, |D|, cos phi_pay).
- R7 twisted Rotfel'd payment:
  - Chain entries re-derived, including |X' - alpha^{+-1}X| <= sqrt(2L(L - |A|)).
  - Rotfel'd (PSD, concave f(t) = t^r with f(0) = 0) gives P_1 >= L M_r(nu).
  - dS/dq = r g[mu1,mu2,mu3] > 0 with g'' = (r-1)(r-2)(1+mu)^{r-3} > 0; dS/dP < 0 with h'' = (r-1)(1+mu)^{r-3}(2 + r mu) < 0.
  - S is decreasing along both boundary curves (PSD q = P - 1 and discriminant q = -2(P/3)^{3/2}), which cross at P = 3/4. So S_min sits at the corner (Pbar, max(qlow, lowK)).
  - The circle reduction is valid: two closed half-planes share a ray. The circle parametrisation is exact for all 3 chains; the sign of phi_pay is irrelevant since psi covers the full circle.
  - Tail L >= 2.5 Lmin follows from Q^s >= L.
- R8 s-monotonicity: for s <= 3/2, P_1(s) >= L M_{1/s} >= L M_{2/3} >= F_{3/2}. Also c_1(s) <= c_1(3/2), and Thm A is s-free. Gamma_s and c_s decrease in s. So the s = 3/2 bound dominates every b in [2,3].
- Imports are used within their hypotheses: Note Thm c1sharp (all atoms, 1 <= s <= 3/2) and Part I Thm A (arbitrary unit phases, R^2 = |B|^2 + |C|^2 + |D|^2).

CODE AUDIT.
- iv.py:
  - +, -, *, / and sqrt are widened by 1 ulp each way; cos/sin use Taylor to x^42 or x^43 with a Lagrange remainder |x|^44/44! (resp. 45!) and handle interior extrema; the cube-root enclosures are verified by cubing; the pi bracket and the cos/sin 72/144 constants are correct.
  - My audit (verify/ref3_iv_audit.py, 200-bit mpmath, about 60k points and intervals including +-np.pi, +-2pi, +-6.3 and the true pi): 0 failures.
  - cbrt_hi has no fallback after 6 iterations; this never triggers.
- pay_adapt/pay_rig:
  - S_lb_point: each root bound is verified by the sign of p(t) on the correct monotone branch, with fallbacks sqrt(P/3), -sqrt(P/3) and -2 sqrt(P/3), all valid.
  - LB_Smin takes the max of the boundary candidate and the interior candidate (only when provably feasible), and returns +inf only when qlow > 2(Pbar/3)^{3/2}, which is valid.
  - L-cells share float endpoints, so there are no gaps; the true L >= Lmin = max(1, m_lo).
- bb52 coverage:
  - 476 initial boxes per chunk = 238 m-boxes (only boxes lying wholly in m1 < m2 are skipped) x 2 phi_pay cells; the 4 chunks cover all 8 cells.
  - max m >= 1.4 is covered by F >= 1.4/sqrt3 = 0.8083 > THR_I = 0.785686.
  - Region I is valid. alpha_max is verified by cos_iv.
  - Region II: either the trivial bound or a rigorised certificate on each alpha-cell.
  - Bisection; no time or queue cutoff; reaching the 2e-3 width floor counts as FAIL (it never occurred).
- cert_hs.rigorise: G~ = sum lam_k u_k u_k^* with float lam_k >= 0 is exactly PSD; the sums have error bounds 2n u sum|.|; M is enclosed and the division by M is enclosed.

HOLES FOUND (sub-ulp) AND CLOSED.
- Two float subtractions are not outward-rounded: ph = p - alpha in try_certs, and psi - phi_pay in pay_adapt._cell_lb.
- The phi_pay and psi grids end at +-np.pi, not +-pi.
- Effect: about 1e-15 on the bound, and accepted cells have no slack beyond TARGET.
- I made outward-rounded copies with the grids widened to +-PI.hi (verify/ref3_patched/) and re-ran all four chunks: every chunk ends fail=0, with box-for-box identical logs (I=575, II=1540, certs=1842). So 0.995 is rigorous.

RERUNS.
- My full l3_4 rerun (verify/ref3_chunk_l3_4.log) ends 'DONE: I=146 II=461 fail=0 certs=1066'. It is identical line by line to cert_l3_4.log apart from timings.
- The earlier l0_7, l1_6 and l2_5 reruns are byte-identical to the official logs after stripping timings.

INDEPENDENT CHECKS (all pass).
- Payment (ref3_payment_audit):
  - LB_Smin <= S on 20k random PSD matrices.
  - F <= P_1(s) for s in {1, 1.1, 1.25, 1.5} on single, two-atom, IF, near-IF, random and Nelder-Mead-adversarial atom systems (adversarial min P_1 - F = 4.9e-4).
  - Exact equality cases: F = 0.5773502691896256 < 3^{-1/2}.
- Certificate identity (ref3_cert_endtoend):
  - q built explicitly is >= 0 with sum 1.
  - My own FFT pairings match cert_hs's a_0, a_T and n to 2.2e-16.
  - On 160 networks with DC and beta_2 killed: h <= sum q g; the pairing identity is exact; the CS step holds; non-own mass <= R.
- End-to-end: the certified point bound is >= the directly computed gamma/gamma_IF for random networks and for 4-atom Hadamard realisations of random domain points.
- Tight case (ref3_cert_tight): optimised high-ratio networks reach 0.78176 = kappa_a/Gamma (the IF block); the bound there is 0.78176; no violation at s = 1.5, 1.25, 1.
- SLSQP adversary: the best is 0.78176 at s = 1.5, 0.65738 at s = 1.25 and 0.55279 at s = 1.
- p = 3: R9 is p=5-specific (aliasing, kappa, R2 Parseval at p=5), so it implies nothing about the IF value at p = 3.

### R14: CONFIRMED

CLAIM: (C) at (7,2) for b in [2.5381, 3]. For s in [1, 1.182], every two-feature network has gamma <= 0.9995 gamma_IF (<= 0.999 for s <= 1.175).

PROOF RE-DERIVED.
- R11(i): sum_j kappa_j(delta) = 7/2, so Gamma_s(7) = (7/2) 3^{-s} with the uniform maximiser unique, and K* = 3.
- R11(ii): with one feature j, pick delta with j delta = +-1. Then h <= kappa_min A, and B_7 P^s >= Q_j^s >= L >= A, so ratio <= 3^{2s-1} kappa_min/3.5 <= 0.9682.
- Relabelling a -> 2a permutes the classes 1 -> 2 -> 3 -> 1, so WLOG the features are {1,2}.
- R12 re-derived:
  - sum_delta rho^{j delta} c_l(delta) = -p[l = j], and the conjugate sum vanishes for l, j in F.
  - phi_j(2j 1_T) = conj(beta_{j,T}).
  - mu symmetric gives sum_l K_l Re beta_l.
  - With |w| = min mu/2, q_w >= 0 and has mass 1.
  - Valid for any width, DC, any outputs, every odd p.
- R13 re-derived:
  - Single-neuron budget sum_j |beta_ij|^{2/3} <= e_i^{2/3} (Hoelder + Parseval over F + AM-GM; the constants match exactly).
  - l^r monotonicity for r >= 2/3; then F_j A_j <= 3^{1-s} Q_j^s via Note Thm c1sharp (p-free atom inequality, s in [1, 3/2]) and Q^s >= L.
- The simple bound follows, using K_l >= 0, Re beta_l <= A_l and beta_3 = 0.
- The slab x_j >= XMAX is valid (F_j >= 3^{1-s} XMAX >= 1, and dropping M_j only weakens the bound).
- tau in [0,1] by homogeneity.

CODE AUDIT of simple72_rig.py.
- Coverage of (tau, x1, x2) in [0,1] x ([0,1.75] U slab)^2 and of the s-cells is complete.
- mu is renormalised exactly by an interval Z.
- The final upper end is converted float -> up(); depth > 40 counts as FAIL.
- Defect found: mpmath 1.3.0 interval ** and iv.exp are NOT strictly outward-rounded. I observed 7 of 20,000 boxes with corner values outside the interval, by at most 3.2e-28 relative (sub-ulp at 80 bits). iv.cos and iv.log were clean, and so were basic interval ops on 80k checks.
- This is harmless here: the final up(float(q.b)) adds >= 1.1e-16 relative, 11 orders of magnitude more. The script's claim that 'every quantity is enclosed in mpmath interval arithmetic' is technically false, but the result is rigorous.

INDEPENDENT VERIFICATION (verify/ref3_r14_bb.py).
- A different branch-and-bound:
  - s is a 4th bisection dimension;
  - the slab starts at XMAX = 2;
  - bounds come from explicit monotonicity in tau, x, s, r (numerator max at the tau endpoints, F_j >= max(1 - c1(s1) x_hi, 3^{1-s1} max(1, x_lo)), and ||y||_{1/s} >= ||y||_{1/s0}, a lower bound since s0 <= s);
  - dyadic mu renormalised with Fractions;
  - every transcendental computed as exp(log) and widened by a relative 1e-20.
- Results:
  - [1, 1.175] @ 0.999: 9,496 boxes, fails = 0, worst 0.998995.
  - [1.175, 1.182] @ 0.9995: 10,090 boxes, fails = 0.
  - Merged [1, 1.182] @ 0.9995: 15,946 boxes, fails = 0.
- The earlier referee's reruns of simple72_rig and its extension also gave fails = 0.

NETWORK CHECKS (ref3_r12_r14_networks, ref3_if_networks, ref3_adversary).
- R12 identity: 5,400 checks at p = 7 (DC and all frequencies) with explicit q_w, max relative error 1.2e-15, q_w >= 0; the R12 inequality and the R13 ledger hold.
- R12 also holds on the p = 3 Conjecture-20 counterexamples (n = 11, 13), checked by brute force over all 3^n inputs: identity exact to 1e-15, inequality with slack >= 2.08. exact_certificate.py confirms these networks beat IF (1.054 and 1.109).
- Explicit two-feature IF networks attain the simple bound with equality: 0.78907 (b=3), 0.85499 (b=2.538), 0.86171 (b=2.5), 0.98006 (b=2), at A2/A1 = 2cos(pi/7). This reproduces R16(a).
- SLSQP adversary from IF starts plus extra neurons: best 0.85499 at s = 1.182 and 0.78907 at s = 1, never above IF.

DEPENDENCIES: c1sharp (computer-assisted, Note, cited not re-proved) and paper Lemmas 2 and 6. No Danskin, Rogosinski or Laguerre is used. The claim is p=7-specific and implies nothing at p = 3.
