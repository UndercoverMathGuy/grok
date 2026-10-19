# Referee verdicts on known/REPORT_R2.md (saved by the orchestrator)

## Overall
Bottom line: the three proved tensor lemmas KM1-KM3 stand, and KM4 is correct. The four 'refuted' ledgers reach the right conclusions but each overstates what was shown.

KM1-KM3: confirmed. I reconstructed each proof line by line, and my own adversarial and independent numerics pass.
- KM1 is the S2 cost lemma; it is sharp at b=2 for pinching but not for additivity.
- KM2 is fully proved. My free-phase check confirms the equal-phases step, which the agent's own scripts had assumed rather than tested.
- KM3 is correct, including the full family of minimisers.

KM4: the inequality is correct and could be labelled PROVED. I re-derived LARGEN R3 (Rogosinski is used within its hypotheses) and checked it adversarially, including at p=3. Two side remarks are inaccurate: the 'equivalently' in H2, and 'strictly stronger than pointwise' for k >= 2.

Refutation ledgers: no conclusion is wrong, but each needs a scope correction.
- L1 calls '(MM) false at p=3' certified. It is certified only per-network; the class-C optimum at p=3 is numerical only.
- L2's counterexample is valid, but it says nothing against arithmetic (Sidon-type) hypotheses.
- L3's R8a counterexample is valid; the 0.28*2^n constant needs its b=2, n>=4 qualifier, and the l_1 part needed a cone-field example, which I supplied.
- L4 lacks an explicit counterexample and wrongly calls the transition identity equivalent to (C).

None of these results bears on (MM) or (D) beyond showing that cost-side theorems are irrelevant. Scripts and outputs are in /home/user/grok/heracles/conjecture20/unify/known/verify/ (v_km1, v_km2, v_km3, v_km4, v_l1, v_l2l3; .py and .out). The agent's own files are in unify/knownmath/, not unify/known/.

## Salvage
Corrected or weakened statements that survive:

- KM1. Keep (a) and (b) as stated. Replace 'sharp threshold b=2' with 'sharp threshold for pinching'. Additivity on direct sums holds at nu=2 for every r; its status for nu >= 3 and b < 2 is not determined here. Simplify (b): N_r(g) >= N_1(g) >= ||g||_F, so N_r of a rank-one tensor is its Frobenius norm.

- KM2. Keep as PROVED. Add:
  - the Craigen violation is |2 - 2e^{2ia}| = 4|sin a|, zero iff a = 0 mod pi;
  - the network uniqueness holds up to slot rotations c_t, the compensating output phase, sign flips of factor pairs, and factor balancing.

- KM3. Keep as PROVED. The minimiser family is verified for random phi_t with sum phi_t = 0 and pair sign flips.

- KM4. Upgrade to PROVED, citing BH (Bayart-Pellegrino-Seoane-Sepulveda 2014) and the re-derived R3. Corrections:
  - drop 'equivalently';
  - add that for k = 1 it gives sum_t |beta_t| <= (4/p)|A-c|/cos(pi/p) with constant 1;
  - for k >= 2 it improves on pointwise detection only in aggregate (l_{2k/(k+1)}), since pointwise control |beta_T| <= (4/p)|A-c|/cos(pi/p)^{|T|} already has constant 1.

- L1. Restate: (i) per-network 'mixing never helps' is certified false at p=3 (and also fails at p=5 for the R10 network), so no per-network or p-free cost/norm theorem implies (MM_opt) or (D); (ii) '(MM_opt) false at p=3' is numerical, because the class-C optimum at p=3 is unproved; (iii) the cost-side theorems (Strassen/Shitov, nuclear additivity, Friedland-Lim, Comon/Shitov) are irrelevant because pinching already holds for b >= 2. Label it as an obstruction or remark, not REFUTED.

- L2. Restate as: 'no lower bound sigma_delta(S) >= c_0 > 0 can hold uniformly over all S of a given size (intervals with Rudin-Shapiro signs give sigma <= ~2.2/sqrt(K)). Hence O1 must use efficiency/coverage or an arithmetic hypothesis that efficient supports satisfy.' Sidon/dissociated hypotheses do give sigma >= c_0.

- L3. Keep as REFUTED for the fixed-radius polydisc payment, for every s > 1, with constant >= (1 - (n/nu)^{s-1}) R^{-n}; this is >= 0.28*2^n at b=2 for n >= 4. For the l_1 currency at R = 1, the failure is on detection: P = A + eps*H_2 (random-sign multi-affine degree 2, with eps*sup|H_2| = (A/2)sin(pi/p)) has c >= A/2 and sum|beta_T| ~ sqrt(n)*A, so the cone margin cannot control sum_T|beta_T| uniformly in n.

- L4. Restate as a remark: R1/R2 hold for every compact network family and cannot locate the programme's switching point s_c (two-network toy with crossing at s_x != s_c). The transition identity is a NECESSARY consequence of (C) near s_c (GLOBAL R7), not equivalent to it.

## Verdicts

### KM1: CONFIRMED

Proof of (a) re-derived and correct: for a decomposition f = sum_i (x)_t v_it, Pi_k f = sum_i (x)_t P_t^(k) v_it, and generalised Hoelder with nu factors gives sum_k prod_t a_kt^r <= prod_t ||a_.t||_{r nu}^r <= prod_t ||a_.t||_2^r = ||T_i||_F^r when r nu >= 2. Equality holds iff the columns are proportional at b=2, which is the S2 equality case. Taking the infimum gives pinching; concatenation gives the reverse inequality, so the quasi-norm is additive on direct sums.

(b) is correct. The Rotfel'd/Schatten detour is unnecessary: for r <= 1, N_r(g) >= N_1(g) >= ||g||_F, so N_r of a rank-one tensor is its Frobenius norm, and the pinched sum is exactly 2^{1-b/2} > 1.

Network reading is correct. AM-GM balancing gives min cost = nu^{b/2} N_r(f)^r; checked.

Independent checks (known/verify/v_km1.py, .out): an adversarial L-BFGS maximisation of the termwise quantity for nu = 2..5, K = 2, 3, 6 and b in {1.8, 1.95, 2, 2.05, 2.5, nu} gives exactly 1 for b >= 2 and exactly K^{1-b/2} for b < 2.

Caveats (wording only):
- The headline 'pinching/additivity with sharp threshold b=2' is sharp for PINCHING only. At nu = 2, additivity on direct sums holds for every r (block-diagonal singular values; verified to 1e-15). Nothing shows additivity failing for b < 2.
- KM1 is the S2 cost lemma restated. It carries no information about the margin.
- The 'r -> 0 limit = Strassen' placement is heuristic. The Kong-Lee-Wang attribution is only via a comment line in Cochrane arXiv:2001.07202, which does exist.

### KM2: CONFIRMED

Every step re-derived and correct.
- (i) H = U D U^* is normal (U unitary from Lemma 7's equality clause). The chain lemma makes it upper triangular with unit diagonal. A normal triangular matrix is diagonal (Frobenius = sum |eig|^2), so H = I and D = I.
- (ii) The generating identity is right.
- (iii) With z_t = -1/xi_{sigma(t),t}, every l != l_0 term has a zero factor, giving prod_t w_{sigma(t),t} = nu for every bijection, so all w_lt != 0.
- (iv) Transposing sigma gives vanishing 2x2 minors, hence rank one with non-zero factors.
- (v) All circles |z - 1/alpha_l| = 1/|alpha_l| pass through 0. 'Not all alpha_l equal' follows because n >= 2 gives two indices A != B other than l_0, and equal alpha would force m_A = m_B, contradicting orthogonality. Two distinct such circles share at most one other point, so gamma_t is constant.
- (vi) Newton's identities with p_1..p_{nu-1} = 0 and zeta_{l_0} = 1 give {zeta} = mu_nu, each root once (z^nu - 1 has simple roots).
- n = 1 is handled separately; the converse is trivial.

The agent's scripts ASSUMED equal phases. My script (v_km2.py) frees the phases d_l and solves (JF) from random starts, for n = 1..5 (727 solutions in total). Every solution has d_l = 1 (to 1e-15 for n = 2, 4; up to 8.5e-6 from slow convergence at n = 1), ratios xi_l/xi_0 that do not depend on the slot (to 2.5e-15), and {zeta} = mu_nu exactly once each. An over-complete search (nu+1 free-modulus terms, r in {0.5, 0.8}) finds a minimum of nu, with one term going to 0.

Small corrections:
- The Craigen violation equals 4|sin a| exactly. It is zero iff a = 0 mod pi, not 'only at a = 0'.
- The network reading should add sign flips of factor pairs and factor balancing to 'rotations of the input slots'.
- The Paper's Lemma 7 statement is correct as written, since (JF) is its standing hypothesis; KM2 strictly sharpens it.

### KM3: CONFIRMED

Re-derived.
- Projecting factors onto the planes is a product of contractions: it fixes C_nu and strictly lowers any term with an out-of-plane component.
- The character expansion of prod_t Re(conj z_t e^{i theta_t}) has coefficient 2^{-nu} beta prod_{T} xi, checked numerically to 7e-15. The conjugation pairing eps <-> -eps lets one fix eps_nu = +1, which gives exactly (JF) with beta = 2^{nu-1}. So Lemma 7 is used within its hypotheses: real decompositions, complex coefficients, any N.
- The lower bound and the count >= nu (e_nu(x) >= 1) follow. The block attains the bound.
- At r = 1, ||C||_sigma = max cos(sum theta) = 1 and ||C||_F^2 = 2^{nu-1}.
- Uniqueness for r < 1 follows from KM2. Translating back (xi = e^{-2i phi}, arg determined mod pi, beta_l > 0 forcing sum_t phi_lt = 0) gives exactly the stated family, up to relabelling l -> -l and pair sign flips.

Independent checks (v_km3.py):
- The claimed family with RANDOM phi_t (sum 0) and random pair flips reproduces C_nu to 1e-14 for nu = 2..6. The agent tested only phi = 0.
- A penalty-continuation search for cheaper exact decompositions (nu = 3, 4; r in {2/nu, 0.6, 0.85, 1}; N = nu..nu+3 terms; a 3-dimensional ambient space allowed) gives a minimum cost/bound of 1.00000, never below.
- The hyperdeterminant of C_3 is -4, consistent with real rank 3. Fits with nu-1 terms stall at squared residual 1.

Literature placement is plausible and was not audited in depth.

### KM4: MINOR_GAP

The main inequality is correct. I re-derived H1 (LARGEN R3) completely:
- Phi(x) = P(z(x)) with z_t = rho^{2j x_t}, which covers all of mu_p^n because 2j is invertible mod p.
- P is multi-affine, so P maps conv(mu_p)^n, which contains the polydisc of radius cos(pi/p), into the convex set c+K.
- The c' < c device puts A - c' and f(D) in the open sector, so f is subordinate to G = c' + M^{2/p}. G is convex univalent because the opening 2pi/p < pi.
- |G'(0)| = (4/p)|beta|^{2/p-1} Re(beta) <= (4/p)|A - c'|. Rogosinski's theorem (|a_k| <= |G'(0)| for subordination to a convex map) applies within its hypotheses.
- The w^k coefficient is R^k H_k(zeta). The maximum principle and polynomial BH applied to the k-homogeneous H_k then give the claim.

H2 is a published theorem (BH constant independent of n; subexponential per Bayart-Pellegrino-Seoane-Sepulveda), so the label could be PROVED.

Adversarial check (v_km4.py): hill-climbing on class-C single features at p in {3, 5, 7}, n in {2, 3}, width m in {n, n+1, n+2}. Maximum LHS/RHS is 0.84 for R3 (k = 1..3), 0.85 for the k = 1 BH bound (BH_1 = 1), and 0.19 for the k = 2 bound with the explicit tetrahedral constant 2^{7/4}.

Inaccuracies (side statements only):
- (H2) 'Equivalently, BH_k <= C_eps(1+eps)^k' is wrong. C^{sqrt(k log k)} implies that bound but is not equivalent to it. The conclusion does not depend on the growth rate.
- 'Strictly stronger than the pointwise (max_T) detection' is false for k >= 2. R3 alone gives |beta_T| <= (4/p)|A-c|/cos(pi/p)^k with constant 1 (average H_k against conj zeta^T), whereas the l_{2k/(k+1)} bound carries BH_k > 1. The new bound is stronger only in aggregate (l_q over the C(n,k) coefficients); only at k = 1 is it a strict improvement (an l_1 bound with constant 1).

### L1-additivity: MINOR_GAP

I re-verified the counterexample ingredients independently (v_l1.py):
- p = 3: gamma/gamma_IF = 1.054399 (n = 11) and 1.109163 (n = 13), exact rationals. The frequency split has negative margin (-109.8 and -199.4 gamma_IF when DC pieces are kept in the cost; SPLIT's -273/-594 drops them).
- (5,2,2) R10 violator: gamma/gamma_IF = 0.790214 and h/RHS(D_Q) = 1.033200, reproduced; split 0.377-0.378 gamma_IF.
- Reason (a) is sound. KM1 shows cost-side pinching/additivity already holds for b >= 2, and sign flips do not preserve h. So rank/norm additivity theorems (Strassen/Shitov, nuclear additivity, Friedland-Lim, Comon/Shitov) cannot be what is missing.

Gaps:
- (1) 'So (MM) is false at p=3' is stated as following from CERTIFIED counterexamples. What is certified is the per-network statement 'mixing raises this network's margin above its split'. For the optimum version (max over all networks = max over class C, i.e. (MM_opt)), the class-C side at p = 3 is numerical only; SPLIT R12 itself says so. Per-network (MM) also fails at p = 5 (the R10 violator has gamma > gamma_split), so 'p >= 5 arithmetic' does not rescue the per-network form either. The p-free argument therefore rigorously refutes only per-network implications.
- (2) The proposition is a literature meta-claim. 'Decoupling is not an instance of any known false conjecture' is a negative search result, not a refutation.

Citations spot-checked: Shitov Acta Math 2019, Friedland-Lim Math. Comp. 2018, Hastings 2009. Cochrane arXiv:2001.07202 exists, but mentions Kong-Lee-Wang only in its comments field.

### L2-flatness: MINOR_GAP

The counterexample is verified (v_l2l3.py). I used the Note's definition: sigma_delta(S) = inf over b and phases of (1/2)osc / sum b_l |sin(pi l delta/p)|.
- At p = 1000003, S = {1..K} with Rudin-Shapiro signs has sup = sqrt(2K) exactly, and sum_l s_l(delta = round(p/2K)) = 0.637K.
- sigma_delta <= 0.670, 0.379, 0.193, 0.098 and 0.049 for K = 8, 32, 128, 512, 2048, decaying like 2.2/sqrt(K).
- So any bound sigma >= c_0 > 0 that is uniform over all sets of a given size is false.

Citations checked: Bedert arXiv:2509.05260 states 1/5 - o(1); Jin-Milojevic-Tomon-Zhang arXiv:2509.03490 states 1/10 - o(1).

Gaps:
- (1) 'Any theorem giving sigma(S) >= c_0 from |S| or arithmetic alone is false' overstates. Arithmetic hypotheses do give it: for Sidon/dissociated S, (1/2)osc >= sup|P| >= c(Sidon constant) sum b_l >= c sum b_l s_l, so sigma >= c_0 > 0. The counterexample only kills bounds that hold for every S of a given size, or for arithmetic classes that contain long intervals. Whether efficient supports fall into a good arithmetic class (they can carry relations) is the open question, and it is not refuted.
- (2) The rest is a literature-relevance survey, plausible but not a refutation. Rudin's |S| = O(S^2 log|G|) was stated from memory; it is correct in form.

### L3-largen-bohr: MINOR_GAP

The core refutation is valid.
- The n atoms beta_i = A/n, xi_it = e^{2 pi i i/n} give P = A(1 + z_1...z_n) exactly; verified to 1e-15 for n = 2..7.
- gain = A(1 - (n/nu)^{s-1}) and sup_{D_R^n}|P - A| = A R^n.
- Unified's (L) quantifies over EVERY finite atom system with no margin condition, so the family is admissible as a counterexample. Note that its own cone margin is negative: cm = -2A cos(2pi/p) at k = 2.
- The required constants reproduce: 4.55, 86.5, 2.39e4, 1.63e9, 7.13e18 at n = 4, 8, 16, 32, 64 (b = 2, R = 1/2).

Gaps:
- (1) '>= 0.28 * 2^n' holds only at b = 2 for n >= 4: gain/A is 0.18 at n = 2 and rises from 0.28 at n = 4 to 0.39. For fixed s > 1 the constant is about (s-1)2^n/n, still exponential.
- (2) 'Any l_1-over-T payment' is refuted by this family only at radius R < 1. At R = 1 the family has sum|beta_T| = A, so the l_1 payment itself is not refuted. The exclusion there comes from detection, and the Bohr-radius argument concerns sup-norms of arbitrary polynomials rather than cone-margin fields.

The gap in (2) can be closed. Take P = A + eps*H_2 with random-sign degree-2 multi-affine H_2, realisable by atoms, and eps chosen so that eps*sup|H_2| = (A/2)sin(pi/p). Then c >= A/2, while sum|beta_T|/|A - c| grows like sqrt(n). My check of the multi-affine degree-2 ratio sum|c_T|/sup gives 1.34, 1.79, 2.47 and 3.37 at n = 4, 8, 16, 32. The agent did not give this.

### L4-temperature-transition: SERIOUS_GAP

The conclusion is right, but the evidence is informal and contains a false statement.

What holds:
- GLOBAL R1/R2 (entropy derivative, Danskin slopes) hold for any compact family of networks.
- The transition identity involves the programme's switching point s_c, which R1/R2 do not see. A two-network toy shows R1/R2 cannot imply it: two networks with fixed cost profiles and gamma_i(s) = h_i (sum_k a_ik^{1/s})^{-s}, with h_i tuned so that the curves cross at any s_x != s_c. R1/R2 hold and the identity fails.

What is wrong:
- (1) The REFUTED label has no explicit counterexample; the check is 'logical'. The toy above is easy, but the agent did not give it.
- (2) 'GLOBAL R7 itself shows it is equivalent to (C) near s_c' is FALSE. R7 proves only (C) on both sides => W_{K<}(s_c) = gamma*(s_c). A single equality at s_c cannot imply the count statement at all s near s_c. This could mislead later rounds into treating the identity as sufficient for (C).
- (3) 'Generic upper envelopes switch branches at crossings that the programme does not predict' is a heuristic, not a theorem about this problem.
