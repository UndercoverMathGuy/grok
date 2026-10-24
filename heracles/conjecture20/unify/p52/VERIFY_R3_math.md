# Referee verdicts on p52 (math lens; saved by the orchestrator)

## Overall
I re-derived every step of the (5,2) proof chain and of the (7,2) results. I found no mathematical error in any of the 14 claims. All my independent checks pass. Scripts and outputs are in /home/user/grok/heracles/conjecture20/unify/p52/verify/, prefix hr_ / hr1..hr6. They do not import the authors' code, except hr6, which calls the authors' SDP only to generate a certificate and then re-evaluates it with my own code.

**Logical chain for (C) at (5,2).**
- The reduction R1 needs the hypothesis 'every network with beta_2 = 0 has gamma < gamma_IF'.
- R9's certified statement (<= 0.995 gamma_IF for every b in [2,3]) is exactly that hypothesis, so R10 follows.
- R9 rests on:
  - the three ledgers R2 (mode budget), R3 (Pythagorean cost) and R4 (margin), joined by R5 (elimination of R);
  - the WLOG reductions R6;
  - the payment lower bound R7, made s-uniform by R8.
- I proved each of these. Each holds for arbitrary inputs, any width, DC and outputs on every frequency, and for every b in [2,3]. They need beta_2 = 0 exactly; the R1 relabelling (a -> 3a) supplies this.

**s-monotonicity (R8).** The correct form is F_{3/2}(data) <= P_1(s) for every s in [1, 3/2]. Each ingredient satisfies it. P_1(s) itself is NOT monotone in s: K > 3 equal atoms make it increase. The argument never uses monotonicity of P_1, only of the lower bounds.

**Large-interference cut.** 1.4/sqrt3 = 0.80829 >= THR_I = kappa_a/(0.995 Gamma_{3/2}) = 0.78569. If max(m) >= 1.4, the trivial bound gives <= 0.995 at s = 3/2, and by R8 at every s.

**Imported results, all applied within their hypotheses.**
- Rotfel'd: PSD summands, f(t) = t^r concave with f(0) = 0.
- Paper Lemma 6: arbitrary inputs.
- Note Thm c1sharp and Part I Thm A: stated for all atoms, 1 <= s <= 3/2. My adversarial searches found no violation (max normalised violation 4.7e-16 and -2.0e-13).
- Paper Thm 8 / Prop 9: give gamma* >= gamma_IF.

**Independent checks run.**
- hr1: R2, R3, R4, R5, R12 and R13 on random and adversarial networks with DC and all frequencies, at p = 3, 5, 7, 11, 13 and n = 2, 3. R4 was tested with arbitrary probabilities q, not only Hermitian-square ones. All residuals are <= rounding.
- hr2 / hr2b: R7, i.e. the chain entries, the uncertainty disk radius, the Rotfel'd step, (P,q)-monotonicity (0 violations out of 7811), the closed-form S_min against brute force (agreement to 3e-50 at high precision), the disk-to-circle reduction, and payment <= P_1.
- hr3: SLSQP maximisation of gamma/gamma_IF over general networks with beta_2 = 0 imposed exactly. Best values:
  - 0.78176 at b = 2 and 0.63498 at b = 2.5, both the single-feature IF block;
  - at (7,2), b = 2.5381, two features: 0.85498.
- hr4: R12 holds on the p=3 counterexample networks (n = 11, 13). No claim here implies the IF value at p=3.
- hr5: my own branch-and-bound for the R14 bound, using monotonicity bounds and per-box LP mu, with a different cut XMAX = 2:
  - s in [1, 1.175], target 0.999: 167,746 boxes, 0 failures;
  - s in [1.175, 1.182], target 0.9995: 364,634 boxes, 0 failures;
  - pointwise supremum 0.99710 at s = 1.182.
- hr6: a spot-check of R9's certificate pipeline on three of the hardest Region II boxes. My own evaluation of q, a_0, a_T and n matches cert_hs's interval enclosures to 1e-6. The torus-rotated certificate gives the identical bound. The rigorous box payment F lies below my grid estimate of the circle payment. h <= a + R n holds on actual networks.

**Rigour nits (all belong to R9's computation, not to the mathematics).**
- The phi_pay grid and the psi grid cover [-float(pi), float(pi)], which misses a slice of width about 1e-16 at the true +-pi.
- ph_lo and ph_hi are not rounded outward.
- Both can be closed by continuity of gamma in the network parameters. The other referee (ref3) is re-running a patched bb52 that covers the true [-pi, pi] and rounds these bounds outward.

**Dependencies.**
- R10's correctness ultimately depends on the full R9 branch-and-bound run, which another referee audits. My spot-checks and the earlier reruns all end with fail = 0.
- R14 depends on simple72_rig.py, which I re-verified independently (hr5). Two other referees (cc9 and ref3) re-verified it too.

## Salvage
- **R7, .tex wording.** Replace 'the minimum over the uncertainty disk is attained on its boundary circle' with 'for every interior X' there is a circle point with P >= P(X') and q <= q(X'). Hence S(X') >= min over the circle of S_min(Pbar, q)'. The code and REPORT (d) already use this form.
- **R8, precise form.** State it as F_{3/2}(data) <= P_1(s) for all s in [1, 3/2]. Do not say or imply that P_1(s) is monotone; it is not for more than 3 comparable atoms.
- **R6 / R9 code.** Cover the true closed interval [-pi, pi] in the phi_pay and psi grids, and round ph_lo/ph_hi and the cdiff arguments outward. Alternatively, state the continuity argument: gamma is continuous in theta and the set of networks with phi_pay outside the float range has empty interior. ref3's patched rerun implements the first option.
- **R12.** Holds verbatim for every n >= 1.
- **R17.** The 'no two-feature analogue' statement is precisely: at b = 2 no inequality 3^{1-s} B_p P^s >= sqrt(Ledger^2 + c R^2) with c > 0 and Ledger = (sum_j (3^{1-s} Q_j^s)^r)^s (or with F_j A_j) holds. The split neuron is an equality case of the budget with non-own mass 3e^2.

## Verdicts

### R1: CONFIRMED

**(i) Programme value.** For y >= 0 with y_1^r + y_2^r <= 1, the minimum over delta of sum_j kappa_j(delta) y_j is at most the mean of the delta = 1 and delta = 2 values, which is (5/4)(y_1 + y_2). The power mean gives y_1 + y_2 <= 2^{1-s}, so Gamma_s <= (5/2) 2^{-s}. The value is attained at y = 2^{-s}(1,1).
- Uniqueness: if y_1 != y_2, the two delta-values differ by (kappa_a - kappa_b)(y_1 - y_2) != 0, so the minimum is strictly below the mean.
- So the maximiser is unique for every r in (0,1] and K* = 2 on s in [1, 3/2]. The s < 1.855 condition is only needed to rule out one-point supports, and those are already excluded by uniqueness.

**(ii) Relabelling.** Put u'(a) = u(3a), w'(c) = w(3c).
- Then g'(x,delta) = g(3x, 3delta), so the margin and the cost are unchanged.
- uhat'(k) = uhat(2k), because 3^{-1} = 2 mod 5. Hence beta'_1 = beta_2 and beta'_2 = conj(beta_1).
- Numerically (hr1): margin difference 0; beta'_1 = beta_2 and beta'_2 = conj beta_1 to 1e-16.

**Logic.** A maximiser has gamma* >= gamma_IF > 0, so it has at least one feature. If it had exactly one, the relabelled network (when needed) would have beta_2 = 0 and gamma >= gamma_IF, contradicting the hypothesis. With |F| = 2, exactly 2 features follow. No existence of maximisers is needed for the implication.

### R2: CONFIRMED

**Derivation.**
- Parseval: sum_k |uhat(k)|^2 = 5||u||^2.
- Since w is real, |what(1)|^2 + |what(2)|^2 <= (5/2)||w||^2. Output DC is dropped, which is allowed because it cancels in g.
- So the total mode mass is (2/125)^2 (5/2) 25 ||w||^2 ||u_1||^2 ||u_2||^2 = 0.016 ||w||^2 ||u_1||^2 ||u_2||^2.
- AM-GM gives <= 0.016 ||omega||^6/27 = 16/27000 ||omega||^6 = 4 B_5^2 ||omega||^6 = 4e^2.
- The four own modes (1; +-1, +-1) each have modulus |beta(omega)|, because |uhat(-1)| = |uhat(1)| for real u.

Input DC (k_t = 0) and every mixed mode are included, and the bound is b-independent.

**Check (hr1).** A Nelder-Mead adversarial search plus 3000 random neurons (DC included) gives max normalised [non-own - 4(e^2 - |beta|^2)] = 4.3e-15. Equality is attainable, so this is consistent with <= 0. The own-mass identity error is 4.5e-13.

### R3: CONFIRMED

**Derivation.**
- e_i^r = B_5^r ||omega_i||^{3r} = B_5^r ||omega_i||^b, since 3r = b. So (sum_i e_i^r)^s = B_5 P^s, and ||theta||^nu = P^s.
- Write e_i^r = ||(|beta_i|^r, rho_i^r)||_t with t = 2/r >= 2. The triangle inequality in l^t (Minkowski) gives sum_i e_i^r >= ||(Q, sum_i rho_i^r)||_t.
- Raising to the power s gives B_5 P^s >= sqrt(Q^{2s} + (sum rho_i^r)^{2s}).
- (sum rho_i^r)^s >= sum rho_i because r <= 1. Multiplying by 3^{1-s} gives the claim.

**Second claim.** phi_j(chi) = sum_i m_i(j; chi - j1), and the shift chi -> chi - j1 is a bijection. The triangle inequality over neurons with R2 gives ||phi_nonown|| <= sum_i 2 rho_i = R. R is s-independent.

**Check (hr1).** On 400 random networks (with and without DC, widths 1-7) at s = 1, 1.1, 1.25, 1.5:
- max relative [sqrt(P_1^2 + c R^2) - 3^{1-s} B_5 P^s] = 4e-16;
- max [||nonown phi||, computed from the TABLE (not the modes), - R] = -0.04.

### R4: CONFIRMED

**Derivation.**
- h <= sum q g for any probability q.
- g = sum_{j in {1,2}} Re(Phi_j c_j), and q is real, so sum q g = Re sum_j sum_x Phi_j C_j = 25 Re sum_{j,chi} phi_j(chi) Chat_j(-chi). This is exact.
- Own coefficients: phi_1(0) = conj A and phi_1(2 1_T) = conj beta_{1,T}. The input frequencies are chi_t - 1 in {+-1}, and Lemma 6 holds for arbitrary inputs.
- phi_2(0) = conj beta_2 = 0, which is the only place beta_2 = 0 is used.
- Cauchy-Schwarz on the remaining coordinates with R3 gives the bound.
- Characters do not collide, because Phi_1 and Phi_2 pair with different Chat_1 and Chat_2.
- n in cert_hs uses exactly the right index sets: q-frequency f corresponds to psi = 2f, own pairs are f = 0 and f = -1_T, and for j = 2 only f = 0 is excluded.

**Check (hr1).** I used ARBITRARY probabilities q, not only HS, and 300 networks with beta_2 cancelled exactly:
- identity error 1e-15;
- max [sum q g - (a + R n)] = -0.42.

hr6 checks the same with the authors' HS certificates on networks: max -5.9e-11. That case is tight because n is about 0.

### R5: CONFIRMED

**Derivation.**
- a + nR <= (a_+/F) F + (n/sqrt c)(sqrt c R) <= sqrt(a_+^2/F^2 + n^2/c) sqrt(F^2 + cR^2) by Cauchy-Schwarz in R^2.
- Combined with R3 and R4: when a + nR >= 0, (a + nR)/sqrt(P_1^2 + c_s R^2) <= (a + nR)/sqrt(F^2 + c_s R^2), for any 0 < F <= P_1.
- When a + nR < 0, gamma < 0 and the bound is trivial.
- In the B&B, F >= max(1, M)/sqrt3 > 0 always, so F > 0 holds.
- c_{3/2} = 3^{-1}/4 = 1/12, so n^2/c = 12 n^2, matching cert_hs.eval_cells.

**Check (hr1).** The closed form dominates the true chain value in all 300 tested cases.

### R6: CONFIRMED

**q is a probability.**
- q >= 0 on the whole torus, since G_delta is PSD.
- sum_x v^* G v = 25 tr G, because 2f mod 5 is injective on {-2..2}^2. It is in fact bijective onto Z_5^2, so every character is represented.
- Rotating G -> D_t^* G D_t gives v(theta - t)^* G v(theta - t). This is still >= 0 with the same mass, and its f-coefficients are multiplied by e^{-i<f,t>}.
- So a_T gets the phase e^{i<1_T,t>}, a_0 and every |Chat| (hence n) are unchanged, and the pairings Re(conj(beta_T) a_T) are preserved when B, C, D carry the same phases.
- phi_pay = alpha + arg D - arg B - arg C is invariant. The payment depends only on (|A|, |B|, |C|, |D|, phi_pay). I checked that all three chains give +-phi_pay.

**Network symmetries.**
- Slot swap maps (A,B,C,D) to (A,C,B,D).
- x -> -x, c -> -c conjugates all four; margin and cost are preserved, since g'(x,d) = g(-x,-d).
- These combine: conjugate first if alpha < 0, then swap; the torus normalisation then makes B and C real.
- The phi_pay range [-pi, pi] is fully covered, so restricting alpha to [0, pi/2] costs nothing.
- Re A <= 0 gives h <= 0 by Lemma 2.
- Homogeneity, so |A| = 1: a/F is degree 0 because every payment bound is 1-homogeneous (Lmin, rho_L and L(S/3)^s scale; Pbar and q are scale-free).

**Check (hr6).** I rotated the authors' SDP certificates by (+arg B, +arg C). The data are normalised by -arg; my first attempt used the wrong sign and gave different bounds. The rotated certificates evaluated at the ORIGINAL data give bounds identical to the normal-form bounds (e.g. 0.42289 = 0.42289).

**Technicality (affects R9's code, not this lemma).** The B&B grids cover phi_pay in [-float(pi), float(pi)], missing about 1e-16 at the true +-pi. This can be closed by continuity; ref3 has a patched rerun.

### R7: CONFIRMED

**Chain entries.** For u_i = (1, p xi, p^2 xi eta), H = sum varrho_i u_i u_i^*:
- H_21 = B and H_32 = C;
- H_31 = sum beta_i p_i xi_i eta_i, and |H_31 - alpha D| <= sum varrho |p - alpha| <= sqrt(L sum varrho |p - alpha|^2) = sqrt(2L(L - |A|)).

The other two chains are analogous, with X' close to conj(alpha) C or conj(alpha) B. Each chain's q-phase is +-phi_pay.

**(a) Rotfel'd.** tr(sum X_i)^r <= sum tr X_i^r for PSD X_i, with t^r concave and 0 at 0. With tr(varrho u u^*)^r = (3 varrho)^r this gives P_1 >= L M_r(nu(H/L)).

**(b) Monotonicity.**
- The characteristic polynomial is mu^3 - P mu - q.
- dS/dq = r g[mu_1, mu_2, mu_3] with g = (1+mu)^{r-1} and g'' = (r-1)(r-2)(1+mu)^{r-3} >= 0.
- dS/dP = r h[mu_1, mu_2, mu_3] with h'' = (r-1)(1+mu)^{r-3}(r mu + 2) <= 0 for mu > -1.
- Degenerate points follow by continuity; at r = 1, S is constant.
- K is correct: mu_min >= -1 iff q >= P - 1 when P <= 3, together with the discriminant condition.

**(c) Minimum over q, then over P.** For each P the minimum over q is at max(qlow, lowK(P)). The path P -> S(P, q*(P)) is continuous and piecewise non-increasing:
- on q = qlow it is interior monotone;
- on the PSD boundary the eigenvalues are 0 and (3 +- sqrt(4P-3))/2, spreading with fixed sum;
- on the discriminant boundary S = 2(1+t)^r + (1-2t)^r, which is decreasing;
- the two boundary branches switch at P = 3/4.

The feasible P-set is an interval ending at Pbar, because upK is increasing. Clipping Pbar at 3 is correct (LB_Smin does it).

**(d) Disk to circle.** Two closed half-planes through 0 share a nonzero direction d. Moving X' along d increases |X'| (convex quadratic with nonnegative initial slope) and decreases q (linear), so S(true) >= S_min(circle point). The circle parametrisation is exact.

**Wording.** The .tex says the minimum is 'attained on its boundary circle'. What is proved, and what the code uses, is min over the disk >= min over the circle of S_min(Pbar, q). Note also that the minimising spectrum in (c) does not depend on r.

**Checks (hr2, hr2b), all passing.**
- Chain entries: error 4e-15.
- Disk radius: max [|X' - alpha^{+-1} X| - rho_L] = 7e-16.
- L M_r(nu) <= P_1: residue 9e-11, a rounding artefact from tiny eigenvalues raised to r at single atoms.
- Monotonicity: 0/7811 violations.
- Closed form vs brute force: 3e-50 at high precision.
- Disk-to-circle: max -4e-16.
- Full circle payment <= P_1: 5e-6, a float artefact at a double root; at high precision the single-atom payment equals P_1 to 1e-17.
- pay_adapt implements exactly this, with sound interval cells, the tail L >= 2.5 Lmin and the 3-chain maximum. hr6 confirms F <= my grid circle payment on three boxes.

### R8: CONFIRMED

**What is needed.** For each fixed network, R is s-independent, and a and n are s-independent (the certificate does not involve s). Gamma_s = (5/2) 2^{-s} and c_s = 3^{2-2s}/4 decrease in s.

The argument needs F_{3/2}(data) <= P_1(s) for all s in [1, 3/2]. P_1(s) itself is NOT monotone: K > 3 equal atoms give d/ds log P_1 = ln(K/3) > 0. But every ingredient of F satisfies the inequality:
- Twisted: L M_{2/3}(nu) <= L M_r(nu), the power mean increasing in r, for the true L.
- c1sharp: 1 - c_1(3/2) M <= 1 - c_1(s) M, because c_1(s) = 1 - 3^{1-s} increases in s and the theorem holds on [1, 3/2].
- Thm A: s-independent and valid on [1, 3/2].
- Trivial: 3^{-1/2} max <= 3^{1-s} max <= 3^{1-s} L <= P_1(s).
- Tail: 3^{1-s} L.

**Consequence.** Whenever a + nR >= 0, (a + nR)/(Gamma_s sqrt(F_s^2 + c_s R^2)) <= (a + nR)/(Gamma_{3/2} sqrt(F_{3/2}^2 + c_{3/2} R^2)). The trivial branch kappa_a cos(alpha)/(Gamma F) and the Region I threshold are dominated the same way.

**Check (hr3).** At b = 2.5 the best network found is 0.635 = kappa_a/Gamma_{1.2}, as expected.

### R10: CONFIRMED

R10 = R1(ii) + R9 + gamma* >= gamma_IF (paper Thm 8 / Prop 9).

**R9 is exactly what R10 needs.** R9 states gamma(theta) <= 0.995 gamma_IF(b) for every b in [2,3] and every product network with beta_2 = 0. That covers any width, DC, multi-frequency inputs and outputs on all frequencies, including beta_1 = 0. R1(ii) needs strict < gamma_IF for exactly this class.

**R9's mathematical reduction is complete.**
- R2 + R3 + R4 + R5 give the ratio bound.
- R6 gives the normal form.
- R7 gives the payment.
- R8 makes s = 3/2 dominate.
- The cut: max m >= 1.4 gives F >= 1.4/sqrt3 = 0.80829 >= THR_I = 0.78569.
- Region II uses alpha_max, verified rigorously in bb52.

**Code inspection.**
- bb52, pay_adapt, pay_rig, cert_hs and iv implement the formulas correctly:
  - the circle cells are sound;
  - LB_Smin takes valid maxima of lower bounds;
  - pow23 clips at 0, so a negative 1 + lb is handled soundly;
  - the HS certificate G = sum lam u u^* is exactly PSD;
  - the mass normalisation is exact;
  - Bnd0 equals Re(conj A a_0) + sum Re(conj beta_T a_T).

**Independent spot-check (hr6).** I ran the SDP at the centres of three of the hardest Region II boxes. My own grid evaluation of q (q >= 0, sum 1), C_j, the Fourier transforms, a_0, a_T, n and the bound matches cert_hs's enclosures to 1e-6.

**Adversarial search (hr3).** SLSQP over general networks with beta_2 = 0 imposed exactly finds a maximum of 0.78176 = kappa_a/Gamma_{3/2}, far below 0.995.

**Caveats.**
- The full B&B was not re-run by me, as instructed; another referee audits it, and the existing reruns end with fail = 0.
- Ulp-level coverage nits in R9's code (the float-pi endpoints of the phi_pay and psi grids, and ph_lo/ph_hi not rounded outward) are closable by continuity of gamma in the network parameters. They are being patched and re-run by ref3.

### R11: CONFIRMED

**(i) Programme value at p = 7.**
- sum_{j=1}^3 kappa_j(delta) = 3 - sum_j cos(2 pi j delta/7) = 7/2 for every delta != 0.
- The average over delta of sum_j kappa_j(delta) y_j is (7/6) sum_j y_j.
- The power mean gives sum_j y_j <= 3^{1-s}, so Gamma_s <= (7/2) 3^{-s}, with equality at uniform y.
- Uniqueness: for s > 1 it is the strict power mean. At s = 1, K y = const 1 with K = [[a,b,c],[b,c,a],[c,a,b]] and det K = -(a+b+c)(a^2+b^2+c^2-ab-bc-ca) != 0, since a, b, c are distinct.

**(ii) One feature.**
- Take delta with j delta = +-1. Lemma 2 gives h <= kappa_min Re beta_j <= kappa_min |A|.
- The single-neuron budget gives B_7 P^s >= Q^s >= L >= |A|. That budget is valid for r >= 2/3; I re-derived it via AM-GM and Parseval.
- The ratio 3^{2s-1} kappa_min/3.5 increases in s and equals 0.968169 at s = 3/2.

**Checks.** hr3 finds the programme values numerically and the single-feature maximum 0.559 = 3^{2s-1} kappa_min/3.5 at s = 1.5. In fact 3^2 * 0.37651/3.5 at s = 1.5 is 0.968; the searched networks reach the IF single-feature value, which is below this bound, consistent.

### R12: CONFIRMED

**Derivation.**
- Since chi = 2j 1_T != 0 (p odd), sum q_w = 1. q_w >= 0 because |2Re(...)| <= 2|w| = min mu <= mu_delta.
- E_x[g rho^{-<chi,x>}] = (1/2) sum_l [phi_l(chi) c_l(delta) + conj(c_l(delta) phi_l(-chi))].
- sum_{delta != 0} rho^{j delta} c_l(delta) = -1 - (p[l=j] - 1) = -p[l=j], and the conjugate sum is -p[l = -j] = 0 for l, j in F.
- So the w-part equals -p Re(w phi_j(chi)) = -p Re(w conj(beta_{j,T})), because chi - j1 has entries +-j.
- Symmetric mu turns the mean part into sum_l K_l Re beta_l.
- The bound needs no class-C, DC or width hypothesis, and holds for every n >= 1.

**Checks.**
- hr1: random networks with DC at (p,n) in {(3,2), (3,3), (5,2), (5,3), (7,2), (7,3), (11,2), (13,2)}, random complex w, every j and T. Identity relative error 8e-15; aligned test min q = 7e-11 >= 0; max [h - bound] = -0.99.
- hr4, against the p=3 counterexamples (n = 11, 13, gamma/gamma_IF = 1.054 and 1.109): the R12 bound 1.5 Re beta - 0.75|beta_T| exceeds h for every |T|. So there is no conflict with the p=3 value refutation, and R12 is not a value claim.

### R13: CONFIRMED

**Derivation.**
- |beta_ij|^{2/3} = (2p^{-3})^{2/3} (|what(j)|^2 |uhat_1(j)|^2 |uhat_2(j)|^2)^{1/3}, which by AM-GM is at most (2p^{-3})^{2/3} (sum of the three)/3.
- Summing over j in F with sum_{j in F} |uhat(j)|^2 <= (p/2)||u||^2 gives 2^{2/3}/(6p) ||omega||^2 = B_p^{2/3} ||omega||^2 = e^{2/3}.
- For r >= 2/3: sum_j |beta_ij|^r <= (sum_j |beta_ij|^{2/3})^{3r/2} <= e_i^r.
- Then 3^{1-s} B_p P^s >= 3^{1-s} (sum_j Q_j)^s = (sum_j (3^{1-s} Q_j^s)^r)^s. Dropping the features with A_j = 0 only lowers the right side.
- 3^{1-s} Q_j^s >= F_j A_j combines c1sharp (atoms beta_ij with Lemma 6 phases, valid for arbitrary inputs, s in [1, 3/2]) with Q_j^s >= L_j >= max(A_j, M_j).
- The ledger is p-free apart from B_p.

**Dependency.** It rests on the imported computer-assisted Note Thm c1sharp. My adversarial Nelder-Mead search found max normalised violation 4.7e-16, the equality case being a single atom.

**Check (hr1).** p = 5, 7, 11, s in {1, 1.1, 1.182, 1.25, 1.5}, random networks, single-frequency atoms and equal-split neurons: max relative [ledger - cost] = 6.8e-16.

### R14: CONFIRMED

**Reduction.**
- Multiplication by 2 permutes the classes 1 -> 2 -> 3 -> 1. hr1 confirms beta'_1 = conj beta_3, beta'_2 = beta_1, beta'_3 = conj beta_2, with the margin unchanged. Conjugation does not affect A_j, M_j or Re beta.
- So the pairs {1,3} and {2,3} map to {1,2}.

**The bound.**
- R12 with the single (j,T) that maximises M gives h <= sum_j K_j A_j - 3.5 min(mu) max_j M_j, using Re beta_j <= A_j with K_j >= 0.
- The denominator is Gamma_s (sum (F_j A_j)^r)^s by R13.
- The bound is scale-free, so A_1 + A_2 = 1.
- The x = infinity slab is valid: F_j >= 3^{1-s} x >= 3^{-0.182} * 1.75 = 1.433 >= 1. Dropping M_j only weakens the numerator.

**Code inspection (simple72_rig.py).** The mpmath intervals with s as an interval, the exact renormalisation of mu, the max-interval for M, and the interval form of F are all correct.

**Independent re-verification (hr5).** A different bounding method (monotonicity in tau, x and s; F at s_hi; power mean at s_lo; Gamma at s_hi), my own LP for mu, and cut XMAX = 2:
- s in [1, 1.175], target 0.999: 167,746 boxes, 0 failures;
- s in [1.175, 1.182], target 0.9995: 364,634 boxes, 0 failures;
- pointwise supremum 0.98828 at s = 1.175 and 0.99710 at s = 1.182.

hr5 uses float arithmetic with 1e-9 safety, so it is an independent confirmation, not a second certificate. The cc9 and ref3 referees also reproduced the interval run.

**Logic and adversarial search.** With R11(ii) and zero-feature h <= 0, every maximiser has 3 features for s in [1, 1.182], i.e. b in [2.53807, 3]. hr3's SLSQP at b = 2.5381 with beta_3 = 0 finds 0.85498, the two-feature IF value, below 0.9995.

### R17: CONFIRMED

**Exact computation (p = 7, frequencies 1 and 2 split equally, balanced norms).**
- |what(j)|^2 = (p/4)||w||^2, and similarly for the inputs.
- |beta_j| = 2(12p)^{-3/2} ||omega||^3 = 2^{-3/2} e, so sum_j |beta_j|^{2/3} = e^{2/3}.
- The total mode mass is (2/(27p^3)) ||omega||^6 = 4e^2. The own mass is 4|beta_1|^2 + 4|beta_2|^2 = e^2, so the non-own mass is 3e^2.
- For r > 2/3 the same neuron gives sum_j |beta_j|^r = 2^{1-3r/2} e^r < e^r.

hr1 with random phases gives 1.000000, 4.000000 and 3.000000.

**Channel claim.** The non-own modes (1;(2,2)), (1;(2,-1)), (2;(1,1)) and so on have characters (3,3), (3,0), (0,3). These equal -(4,4), -(4,0), -(0,4), i.e. feature 2's sign-flip characters up to conjugation, via 4 = -1 - 2 mod 7.

**Scope of the 'no analogue' consequence.** For this single neuron at b = 2, cost = 3^{-1/2} e equals the multi-feature own ledger (sum_j (F_j A_j)^{2/3})^{3/2} with F_j = 3^{-1/2} (single atoms), while R > 0. So no inequality cost >= sqrt(ledger^2 + c R^2) with c > 0 can hold. That is exactly the Pythagorean form of R3. It does not exclude a channel-resolved ledger, and the report says so.
