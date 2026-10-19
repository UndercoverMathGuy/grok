# Referee verdicts on E_reduce/REPORT.md (saved by the orchestrator)

## Overall
No claim is FALSE and none has a SERIOUS_GAP. The reduction framework is mathematically sound: R0-R9 and R14 are CONFIRMED, and R10, R11, R12, R13, R15 have MINOR_GAPs. I rebuilt every proof:
- width reduction, compactness and continuity;
- the weighted restricted universal bound;
- the decoupled construction with the per-delta cone lemma;
- elementary Danskin;
- the a.e. Dini lemma;
- the equivalence Theorem A;
- the tax picture with the decoupled envelope;
- the polar-cone argument of Theorem B;
- Corollary U;
- Theorem G(b) semiconvexity.
All are correct and uniform in (p, n, b).

Main defects:
- R13(b), and the remark that 'M singleton implies (FE)', silently need the two-sided equality W = D on [s0, s0+eps). Theorem C's stated hypothesis is one-sided.
- Theorem T at b = nu cites R13, which excludes s0 = 1. The clause is still true unconditionally, via Corollary U at s = 1 (hypothesis from R2c) and uniqueness of the s = 1 programme maximiser (uniform).
- Meta-claims are labelled PROVED but are heuristics: R10 'no value-function argument can...', R12 'removal adds nothing', R15 'minimality'. Separately, R11's 'lambda = 1 is the only obstruction' is literally false: lambda = (1, 0.98, 0.972427) at p = 7, s = 1.8 has two maximisers.
- Endpoint b = 2: '(H3) automatic' and the 'n = 2 differentiable' example do not apply at s = nu/2 without extending pi to b < 2. (H2)_2 may be unsatisfiable if free-gluing maximisers exist.

Substantive caveat on what is achieved: everything about the count is CONDITIONAL. (H1) at lambda = 1, S = F is exactly equivalent to Conjecture 20' (R6a), and the upper half additionally needs the global entropy hypothesis (FE). My search shows (FE) holds at near-global optima (slack ~1e-14) but fails at non-global local maxima (slack down to -0.34). So it is a genuinely global statement that local or KKT reasoning cannot supply.

p = 3: there W = D holds trivially (every network is single-feature). The exact certificates give gamma/gamma_IF = 1.054399 and 1.109163, consistent with the PRICE value. There is no contradiction.

Independent scripts and outputs are in /home/user/grok/heracles/conjecture20/unify/E_reduce/verify/:
- vlib.py
- v1_identities.py, v1_identities.out
- v1b_restricted_bound.py, v1b_restricted_bound.out (adversarial N_S networks with cancelling interference; max ratio 0.9997 <= 1)
- v2_programme.py, v2_programme.out
- v3_decoupled_p3.py, v3_decoupled_p3.out
- v4_optima.py, v4_optima.out, v4b_optima.out
- v5_ties.py, v5_ties.out

## Generality audit
Every proof is uniform in odd prime p, n >= 2 and b in [2, nu]. There is no case split by (p, n), no p >= p0 and no large-n or b-near-nu limit. Each parameter use is explicit:
- b >= 2 is used in the power-mean lower bound P >= D^{1-b/2} (compactness), in the single-neuron budget (sum_k m_k^{b/2} <= (sum m_k)^{b/2}), in the removal inequality and at the b = 2 split.
- sigma >= 1 is used in l1 <= l_{1/sigma}, and 1/sigma <= 1 in concavity for width reduction.
- p prime is used for kappa_l(delta) > 0 at every delta != 0, and for the regular dilation action.

Explicit parameter limits:
1. Theorems B and C (R8, R13) require s0 in (1, nu/2], i.e. they exclude b = nu. Theorem T nevertheless claims its last clause at b = nu through R13: a hidden gap at one parameter value. It can be repaired unconditionally by Corollary U at s = 1 plus M(1) = {uniform}.
2. R13(b) needs the two-sided hypothesis; it does not hold under the one-sided one as stated.
3. At the endpoint b = 2, R3(iii)'s countable-kink criterion and the '(H3) automatic' and 'n = 2 differentiable' remarks see only the left derivative. Extending pi to b slightly below 2 is needed; it is valid, but not stated.
4. (H3) is not vacuous at p = 3, contrary to R15's remark, though it is harmless there.

Hidden special-casing appears only in the numerical checks (small p in {3,5,7,11,13}, n in {2,3}), which are sanity checks only. The proofs themselves contain no computer-assisted or parameter-specific step.

## Salvage
1. Theorem T at b = nu. Replace 'R13 gives the full count' by 'R13 for b < nu; at b = nu, Corollary U with hypothesis R2(c) plus uniqueness of M(1)'. Uniqueness proof: the delta-average of sum_j y_j kappa_j(delta) is p/(p-1) for every y with sum y = 1. A maximiser therefore has sum_j y_j cos(2 pi j delta/p) constant in delta != 0, and Fourier inversion forces y uniform. Hence S* = F and nu E = uniform, unconditionally.
2. R13(b) and the singleton remark. Add the hypothesis W^lambda_{S*} = D^lambda_{S*} on [s0, s0+eps) for lambda near 1. Under (H1) Theorem A supplies it for s0 < nu/2.
3. Endpoint b = 2. Extend pi_{n,b} to b in (2 - eps, 2]: width reduction holds since 1/sigma < 1, and P >= 1 on K_D. Danskin then gives a right derivative at s = nu/2, and the kink criterion covers b = 2. For n = 2 at b = 2, derive H^(1) = log 3 from KM2 (Lemma 7 equality rigidity) rather than from differentiability.
4. R11 wording. Say 'for Lebesgue-a.e. lambda (the exceptional set is null and contains 1, the ray c*1 and tie hypersurfaces through 1)'.
5. R10, R12 and the R15 minimality paragraph should be relabelled as remarks/heuristics, or R10 stated precisely as: 'the first-order relations conv G_W (+C) = conv G_D (+C) at (s, 1) do not exclude a full-support maximiser'.
6. (H2)_2. Reformulate as '(FE) for the class-C split of every global maximiser, and every global maximiser's split is a maximiser' (or (FE)_C plus (MM_opt)), which R14(c) supports. As stated, (H2)_2 is false whenever a free-gluing maximiser exists.
7. (FE) fails at non-global local maxima (verify/v4_optima.out), so any proof of (FE) must use global optimality.

## Verdicts

### R0-setting: CONFIRMED

Re-derived. tau_mu multiplies the frequency-l output by sqrt(mu_l). So Phi_l -> sqrt(mu_l) Phi_l, G_l -> sqrt(mu_l) G_l and beta_l -> sqrt(mu_l) beta_l. With lambda = mu^{-1/2} this gives h(theta) = h_lambda(tau_mu theta); the norm identity is orthogonality of P_0, P_l. Output DC cancels in every gap. Independent check v1_identities.out on random networks with input and output DC, unbalanced, p in {3,5,7}, n in {2,3}: gap = sum_l G_l to 6.8e-14; E_x Phi_l = conj(beta_l) to 1.2e-15; h(theta) - h_lambda(tau theta) to 2.8e-14; cost(tau theta) = cost_mu to 1.1e-15; beta(tau theta) = sqrt(mu) beta to 9.4e-16.

### R1-basic: CONFIRMED

(a) GLOBAL R2 argument re-derived. The polyhedron {mu >= 0 : sum mu_i T^_i = T} lies in the orthant and is pointed. sum mu^{1/sigma} is concave (1/sigma = b/nu <= 1) and coordinatewise non-decreasing along its recession cone, so it is minimised at a vertex with <= p^nu non-zeros. AM-GM then balances. For the tax cost apply the reduction to tau_mu theta and pull back. (b) Correct. Trivial omission: a neuron with a zero factor cannot be 'balanced'; it must be deleted, which also strictly lowers cost. (c) Power mean gives P >= D^{1-b/2} (uses b >= 2), K_D is compact, N_S is closed, and Berge's theorem gives joint continuity in (sigma, lambda). Gamma = max over the simplex of m_lambda(z^sigma). Positivity bounds are correct. General in p, n, b.

### R2-bounds: CONFIRMED

(a) Weighted Lemma 2, then Re beta_l <= |beta_l| <= Q_l^sigma (l1 <= l_{1/sigma}, since sigma >= 1), monotonicity of m_lambda, homogeneity, and the single-neuron budget. I re-derived the budget: C_b = 2^r (2 p nu)^{-b/2}, C_b^s = B_p, using AM-GM and b >= 2. Only l in S enter because Re beta_l = 0 off S; A_l must be defined only on S, and the proof does this. (b) The per-delta cone inequality g_1(x,delta) >= kappa_1(delta) cm(Phi(x)) holds because |cot(pi delta/p)| <= cot(pi/p); it is valid for any sign of c. Relabelling gives g'(x,delta) = g(lx, l delta) and kappa_1(l delta) = kappa_l(delta); gaps add; the cost factorises; shares are z_l w_i. (c) Correct. Independent checks: v1_identities.out (budget ratio <= 0.79). v1b_restricted_bound.out: adversarially optimised N_S networks carrying cancelling interference at j not in S (beta_j = 0 exactly) give max h_lambda/(B_p Gamma^{lambda,(S)} P^s) = 0.9997, tight at b = nu, never above 1. v3_decoupled_p3.out: per-delta cone lemma to 5e-16, gamma_lambda(Theta)/psi - 1 >= -2.2e-16, entropy additivity to 4.4e-16, beta_j off spt A = 0.

### R3-danskin: CONFIRMED

The elementary Danskin proof is correct: MVT plus a compactness subsequence plus uniform continuity of grad f on X x a compact neighbourhood. (i) The temperature identity and joint continuity of H, including zero shares, hold; X = K_D cap N_S cap {h >= eta} contains all maximisers for sigma in a compact interval; enlarging D does not change v. (ii) d/dmu_l ||omega||_mu^b = (b/2)||omega||^{b-2}||P_l w||^2, times -sigma/P, gives -(nu/2)E_l. (iii) Young's theorem: points where both one-sided derivatives exist and differ are countable. Independent finite differences on random networks with DC: |dlog gamma/ds + H| <= 1.0e-10 and |dlog gamma_mu/dmu_l + (nu/2)E_l| <= 1.0e-10 (v1_identities.out). Caveat on (iii): on [1, nu/2] the endpoint s = nu/2 (b = 2) has only a left derivative, so H^(1)_min(nu/2) is not seen and 'Delta^(1) > 0 only at kinks' says nothing at b = 2. The fix is easy: extend pi to b slightly below 2. Width reduction still works (1/sigma < 1), and P >= 1 on K_D for b < 2.

### R4-dini-ae: CONFIRMED

Step 1 (Dini-derivative monotonicity) is correct. Step 2: U open, containing N, with |U| < eps/C. I' = C on U and D_+ I >= 0 elsewhere, so D_+ I >= G >= phi. Then D^+(L - I) <= D^+L - D_+I <= phi - G <= 0, giving L(c') - L(a') <= C|U| < eps. Only finiteness of D^+L and D_+I is used. No measurability of phi is needed. The bound C is genuinely needed (Cantor function). The case C <= 0 is trivial.

### R5-thmA-equivalence: CONFIRMED

(i)=>(ii): L = log W - log D is continuous, >= 0, and L(1) = 0 (R2c). Danskin gives D^+ log W = -H*_min. The touching curve psi (R2b, any optimal theta_1 and y in M) gives D_+ log D >= -H^dec. So D^+L <= phi = H^dec - H*_min <= log(D|S|), and phi <= 0 wherever (E) holds; R4 closes it. (ii)=>(iii): Danskin for pi (single-feature class) and for the Gibbs envelope log Gamma = max over A in the simplex with m >= eta of [log m(A) - s log sum A^{1/s}] gives right derivatives -H^(1)_min and -min_M H(z). Left derivatives give the maximal entropies. (iii)=>(iv)=>(i) are immediate. General: no parameter restriction. The a.e. hypothesis really does self-improve to every s < nu/2.

### R6-value-lower-half: CONFIRMED

Correct as a conditional. (a) Theorem A with S = F and lambda = 1; W_F = gamma*, and the equivalence holds via (ii)<=>(i). (b) gamma* <= W_{S*} = D_{S*} <= pi B_p Gamma_s <= gamma* (price-programme theorem). It needs only W_{S*} = D_{S*} at that s, and the closed interval covers b = 2. p = 3 consistency was checked on the exact certificates (v3_decoupled_p3.out): gamma/gamma_IF = 1.054399 (n = 11) and 1.109163 (n = 13); h = kappa_min min_x cm(Phi_1) to 1e-10; nu E_1 = 1; output DC-free. At p = 3 every network is single-feature, so W = D = kappa_min B_3 pi holds trivially with pi > nu^{1-s}: the PRICE value, not the IF value. There is no conflict with the counterexamples.

### R7-tax-decoupled-family: CONFIRMED

(a) follows from R0/R1 via tau_mu (scale invariance, width reduction in the tax cost via tau_mu). (b) <=: c(theta^(l)) = mu_l^{-1/2} c(tau_{mu_l} theta^(l)) <= mu_l^{-1/2} pi B_p B_l^sigma. This holds for any sign of c. Then use amplitudes A~_l = A_l B_l^sigma, noting that (A~_l)^{1/sigma} = A_l^{1/sigma} B_l. >=: optimal choices. (c) A copy below price strictly lowers every delta-term because kappa_l(delta) > 0 for every delta != 0 (p prime). Optimal single features are balanced and DC-free, so E_l = z_l/nu and the mu-gradient is -z/2. Psi is C^1 in (sigma, mu), and the family is compact on {N >= eta}. The D-side Danskin derivative in t = log lambda matches max_y <z(y), eta> to 1e-6 in 8 random directions at p = 11, s = 1.5 and p = 7, s = 1.8 (v2_programme.out (4)).

### R8-thmB-first-order-rigidity: CONFIRMED

Re-derived. v_W = v_D on {sigma <= s0} near (s0, 1), so the directional derivatives agree on the half-space Pi = {eta_sigma <= 0}. By Danskin, these are the support functions of the compact sets conv G_W and conv G_D. The polar cone is C = {(t,0) : t >= 0}, and sigma_{K+C} equals sigma_K on Pi and +inf off Pi. Hence conv G_W + C = conv G_D + C, both being closed convex sets. Width-independence holds because conv G_W(D') + C is the same for every D'. The mu-coordinates give nu E = sum c_k z_k, and the sigma-coordinate gives H <= sum c_k H(z_k) + h_bar. In the two-sided case C = {0}. s0 > 1 is needed and stated. Independent search (v4_optima.out, optimiser from the refereed GLOBAL lib): the near-global optima found (gamma/D = 1.000000 at (5,2,2.2) and (7,2,2.5); 0.999999 at (11,2,2)) have nu E = z(y) to 1e-10 to 3e-7.

### R9-union-of-supports: CONFIRMED

The mu-only Danskin at fixed s needs only W^lambda_S(s) = D^lambda_S(s) for lambda near 1, and both signs are available, so the mu-projections of conv G_W and conv G_D coincide. R6(b) needs only W_{S*}(s) = D_{S*}(s) plus the unconditional price-programme theorem. The inclusion S* in spt(nu E) holds because beta_j != 0 forces some neuron with P_j w_i != 0 and positive share. Correct at every s in [1, nu/2], including b = 2. At s = 1 the hypothesis is unconditional (R2c); this matters for R15.

### R10-dilation-obstruction: MINOR_GAP

The mathematical content is correct. Z_p^x/{+-1} acts on F as a group on itself, which is regular. m(k.y) = m(y) because kappa_{kj}(delta) = kappa_j(k delta). A proper support is not invariant under a transitive action, so the orbit contains distinct supports with union F. H(z) is constant on an orbit, and the orbit average with h_l = H^(1)_min lies in conv G_D. Verified at (5,1.9), (7,1.8), (11,1.5), (13,1.5) and (13,1.3), the last with 6 maximisers (v2_programme.out (2)). Gap: the conclusion 'no argument that uses only value functions and their one-sided derivatives can exclude...' is a meta-claim with no formal proof. What is proved is only that the first-order envelope relations at (s, 1) do not exclude the point. Two things are not addressed: zeroth-order constraints gamma_lambda(theta*; sigma) <= D^lambda(sigma) at nearby parameters, and second-order envelope constraints (Hess_xi f(theta*) <= Hess v_D in directions where the maximisers' first derivatives tie). Also, 'H(z) + H^(1)_min equals H^dec' needs y taken from a minimal-H(z) orbit.

### R11-thmG-generic-weights: MINOR_GAP

(a) is correct. Under tau_{mu0}, E^{mu0}_l(theta') = E_l(theta)/mu0_l on both sides; I checked the D-side gradient -(1/2) z_l/mu0_l directly. (b) is correct and unconditional: with x = (log B - t)/s, the bracket's t-Hessian is -(1/s)(diag q - qq^T), which is >= -(1/s)I. So v + |t|^2/(2s) is a sup of convex functions, hence differentiable a.e., and at those points Danskin forces a unique z, hence a unique y. Numerically: second differences of v + |t|^2/(2s) >= 0.35 > 0, and 40 random lambda each give a unique maximiser (v2_programme.out (3)). (c) and Corollary S are correct. Gap (wording in the statement): 'the symmetric point lambda = 1 is the only obstruction' is false as written. The exceptional null set contains the ray c*1 and tie hypersurfaces through 1. Explicit point: p = 7, s = 1.8, lambda = (1, 0.98, 0.972427) has two programme maximisers, supports (1,3) and (1,2) (v5_ties.out). Corollary S is a criterion only: it is not shown that every lambda = 1 maximiser is such a limit.

### R12-removal: MINOR_GAP

The inequality is correct: (1-eps)^{b/2} <= 1-eps for b >= 2, and h(theta^{-j}) <= gamma* P(theta^{-j})^s. Checked on random networks with DC: max[P^{-j}/P - (1 - E_j)] = -6.1e-4 (v1_identities.out). Gap: the statement also asserts that removal 'is consistent with every straddler' and that 'finite removal adds nothing beyond Corollary U'. Neither is proved. E_j > 0 for a straddler does not imply h(theta^{-j}) <= h(1 - E_j)^s is automatically satisfied, and the compensation argument is heuristic. 'As E_j -> 0 this is the subsidy derivative of Theorem B' is also unproved. These are remarks, not theorems.

### R13-thmC-upper-half: MINOR_GAP

Main part (a) is correct. Theorem B in N_{S*} gives the representation, R6(b) makes the y_k programme maximisers, and (FE) minus Theorem B gives JS <= h_bar - H^(1)_min <= Delta^(1). JS = 0 then forces equal z_k by strict concavity, z determines y, and S* = spt y. Gaps: (1) Part (b) uses 'H(theta*) minimal over conv G_W = conv G_D'. That needs the TWO-SIDED equality W^lambda_{S*} = D^lambda_{S*} also on [s0, s0+eps), but the stated hypothesis is only Theorem B's one-sided one. With one-sided equality, H*_min(s0) < H^dec is possible, conv G_W can carry a t > 0 cone shift, and (b) does not follow. The remark 'If M(s0) is a singleton and s0 < nu/2, (FE) follows from Theorem B' needs the same extra hypothesis. Both are fine under (H1) via Theorem A for s0 < nu/2. (2) The parenthetical 'everywhere where pi is differentiable, e.g. n = 2' does not cover s0 = nu/2 (b = 2): there only a left derivative exists on [1, nu/2]. At n = 2 the conclusion H^(1) = log 3 at b = 2 needs KM2 rigidity or an extension of pi to b < 2. Independent observation: (FE) holds with slack about 0 at near-global optima but FAILS at non-global local maxima (slack -0.06 to -0.34 at (7,2,2.5), (7,2,2), (5,2,2.2), (11,2,2), (5,3,2.6); v4_optima.out, v4b_optima.out). So (FE) is a genuinely global hypothesis and cannot come from local or KKT reasoning.

### R14-endpoint-b2: CONFIRMED

(a) and (b) follow from Theorem A on the closed interval, Corollary U at s = nu/2, and one-sided Theorems B/C (s0 = nu/2 is allowed). (c) is correct, with DC-free meaning no input and no output DC: ||omega||^2 = sum_k ||omega^(k)||^2, so P and E coincide at b = 2 and H(split) = H + H(k|i). Numerically (v1_identities.out): |P - P_split|/P <= 4e-16, |E - E_split| <= 2e-16, H - H_split <= -0.42, beta identical, and with DC E = (P_split/P)E_split to 1e-16. (d) is correct. The claim that such limits have entropy H*_max(nu/2) is true; a cleaner proof than 'left Danskin' is concavity of log gamma in s plus dH/ds >= 0, which give H(theta*, nu/2) - C delta_k <= H(theta_k, nu/2) for every maximiser theta*. Notes: 'Equivalently, (FE) for class-C maximisers plus (MM_opt)' is a sufficient condition, not an equivalence; with (MM_opt) maximisers are DC-free and their splits are maximisers. Whether free-gluing maximisers exist is open, as stated.

### R15-target-theorem: MINOR_GAP

The assembly is correct for b in [2, nu), with these gaps. (1) b = nu (s = 1): the last clause is attributed to R13, which needs s0 in (1, nu/2], so the cited proof does not cover b = nu. The conclusion is nevertheless true there, unconditionally. Corollary U at s = 1 has its hypothesis supplied by R2(c), and M(1) = {uniform}: every y with sum y = 1 has delta-average p/(p-1), so a maximiser needs sum_j y_j cos(2 pi j delta/p) constant in delta, which Fourier inversion turns into y constant. Hence S* = F and nu E = uniform (checked: v2_programme.out (1), p = 5 to 13). (2) 'At p = 3 all hypotheses are vacuous' is wrong for (H3), which is a real condition at p = 3 but unneeded since (C) is trivial there. (3) (H3)_2 being 'automatic outside the countable kink set' says nothing at the specific endpoint b = 2 (see R3/R13). (4) (H2)_2 as stated (FE for EVERY global maximiser) is false if a free-gluing maximiser exists (R14c), and the alternative 'through the split' route is not among T's hypotheses. (5) The 'minimality' paragraph (each hypothesis needed) is heuristic; Proposition O does not show that (H2) is necessary. The value claim is consistent with p = 3.
