# Referee verdicts on E_width/REPORT.md (saved by the orchestrator)

## Overall
All 9 PROVED claims survive a line-by-line rebuild and independent numerics; both refutations are valid.

The substance is thinner than the labels suggest:
- W1, W3 and W5 are algebraic identities (W1's Renyi rewriting of the temperature identity).
- W3 in particular says nothing about any network. Its slack equals log D_S(s) - log D_S(1) + int H^dec for every theta, so it is just the Dini bound on D_S. Its n=2 check is vacuous because that slack is identically 0 there.
- W2 is a correct and useful negative structural fact. (E_S), (EW_S) and the integrated/Renyi forms are all EQUIVALENT to the restricted value identity W_S = D_S, so the entropic route cannot be easier than proving gamma* = pi B_p Gamma_s.
- X1 and X2 show that no s-local entropy-deficit inequality of either sign holds.
- W4 extends p52 R12 to every non-clean coefficient. W6 and W7 are clean all-n spectral versions of Lemma 7 that hold for every network and are consistent with the p=3 winners.
- W8 is the only genuinely new n-uniform VALUE result: aligned circular single features are never cheap at any p >= 5, n, b. But phase alignment excludes atom cancellation, which is exactly where the p=3 counterexamples (1-|A|/L = 0.22) and the large-n combs live.

So nothing here closes (E) or (C). Independent scripts are in /home/user/grok/heracles/conjecture20/unify/E_width/verify/: vlib.py, v1_w1_w5_w9.py, v2_w4.py, v3_w6_w7.py, v4_w8.py, v5_x1_x2_w3.py.

## Generality audit
- W1, W3, W4, W5, W9: hold for every odd p (including 3), every n and every b in [2,nu], with no case split.
- W2: same. It uses width bounds p^nu, which are finite for each (p,n) but not a limit or a case split.
- W6 and W7: also fully general. The only restriction is in W6(c): the final exp bound needs s <= nu/2, which is stated.
- W4: the corollary is for S=F at s=1 only, as stated.
- W8: NOT parameter-free in p. It assumes cos(2pi/p) > 0, i.e. p >= 5; at p=3 it is explicitly excluded. This is an explicit hypothesis, not a limit, and BRIEF section 4 requires value claims to use p >= 5. Its class is narrow: single features (outputs at one frequency), circular field, positive cone margin, phase-aligned atoms. Within that class it is uniform in n and b, with constant k_p <= sqrt5 = k_5, and no limits are taken.
- W2 and W3 involve pi_{n,b}(p), which is defined for all parameters but known in closed form only at n=2. So the numerical checks of W2 and W3 are only possible, and only done, at n=2. Their proofs do not depend on n=2.
- X1 and X2 are refutations at specific parameters ((11,2), (13,2), (5,2)), which is legitimate. They rely on the cited n=2 price theorem pi = 3^{1-s}.
No hidden special-casing found.

## Salvage
- Treat W2 as the main structural lesson: any entropic width theorem implying (E_S) is the restricted value identity. Future work should target W_S = D_S directly (e.g. via W7 + W4-type detection), not entropy inequalities.
- Relabel W3 as a remark (an identity plus the Dini bound), not a theorem.
- Fix W5's reading to: at b=2 without DC, sharing lowers entropy by exactly the convex-end cost excess; for b>2 or with DC, only the upper bound holds and the sign can flip.
- W8's natural extension is to cancelling atoms. W6(c) already isolates the cost of cancellation as the factor |A|/L against twist errors sqrt(2L(L-|A|)) on the n(n-1)/2 non-prefix entries. That is the precise residual for single features at large n.

## Verdicts

### W1: CONFIRMED

Rebuilt the proof line by line. P_b = sum a_i^{1/s} = P_nu^{1/s} sum m_i^{1/s}, and s log sum m^{1/s} = s(1-1/s)H_{1/s} = (s-1)H_{1/s}. So gamma(s) = gamma(1) e^{-(s-1)H_{1/s}}. The integral form follows because d/ds log gamma = -H. Zero-mass neurons drop out. Nothing else is assumed beyond h > 0 (and the multiplicative form does not even need that). Independent check (verify/v1_w1_w5_w9.py): on random networks at (5,2), (3,4), (7,3) and on the exact p=3 certificates at n=11 and 13, the relative error is at most 8.9e-16.

### W2: CONFIRMED

Every implication rechecked.
- Lipschitz: by width reduction, W_S(s) is the sup over one fixed family (width <= p^nu) for every s. On that family |d/ds log gamma| = H <= log p^nu, so log W_S is Lipschitz. log pi is Lipschitz for the same reason. log Gamma^{(S)} = max over vertices v of -s log sum v^{1/s}.
- Dini bound: f(s') = log e(theta_1;s') + log G_{s'}(y*) satisfies f <= log D_S - log B_p, with equality at s. I recomputed d/ds log G_s(Y) = -H(z). Attainment of pi and Gamma holds by compactness after normalising P_nu = 1.
- Danskin needs a compact family with d/ds gamma = -H gamma jointly continuous. Both hold.
- (a)=>(e) and (b)=>(e): the standard 'Lipschitz, L(1)=0, L>=0, D^+L<=0 a.e. on {L>0}' argument. L_S(1) = 0 follows from Lemma 2 plus the budget at r=1, and pi(1) = 1.
- (c)<=>(e) holds by W1.
- (d)=>(e): W1 plus the Dini bound, integrated by absolute continuity.
- (e)=>(d) and the equality claim: under (e) the decoupled networks are maximisers. At points where log W_S is differentiable (a.e.), the right and left Danskin derivatives coincide, so all finite-width maximisers share one entropy, and it equals H^dec.
- H^dec is l.s.c. because the argmax correspondences are upper hemicontinuous and compact-valued. For s>1 every maximiser has linearly independent tables (strict concavity of t -> sum |1+tc_i|^r), so width <= p^nu and the argmax sets are compact.
The cited tools (GLOBAL R2 width reduction, the price-programme theorem) are used within their hypotheses. Numerically, at n=2 the slack log D_S(s) - log D_S(1) + int H^dec is 0 to 1e-9 for p=7, 11, 13, with S=F and subsets (verify/v5_x1_x2_w3.py). The reading is correct: (E_S) is exactly as strong as the restricted value identity W_S = D_S.

### W3: CONFIRMED

The statement is true, but it says nothing about any particular network. Substituting W1 into it, RHS - log gamma(theta;s) = log D_S(s) - log D_S(1) + int_1^s H^dec_S for every theta. So W3 is exactly the integrated Dini bound for D_S (part of W2) dressed with an identity.
- The 'winner form' and the 'free half' are W1 rearranged, with Delta_S an unknown non-negative number.
- Delta_S >= 0 is correct (Lemma 2 plus the budget at r=1).
- The quadrature check in c6 is vacuous: at n=2 this slack is identically 0 for every network. I reproduced that: slack is 0 to 1e-9 at p=7, 11, 13 for several S.
Calling W3 'the requested aim in its correct integrated form' overstates it. It is a valid inequality that can do no work toward (E).

### W4: CONFIRMED

Re-derived the test.
- g = sum_l Re(Phi_l c_l). Then sum_{delta!=0} rho^{j delta} c_l(delta) = -p[l=j], and sum rho^{j delta} conj(c_l) = -p[j+l=0] = 0, since j+l lies in [2,p-1]. Hence sum_delta rho^{j delta} E[g rho^{-<chi,x>}] = -(p/2) phi_j(chi).
- q_w >= 0 when |w| <= min mu/2, and it has total mass 1 because sum_x rho^{-<chi,x>} = 0 for chi != 0.
- Symmetric mu makes sum_delta mu_delta E g = sum_l K_l Re beta_l.
- Uniform mu gives K_l = p/(p-1) = Gamma_1, with min mu = 1/(p-1); the corollary then follows from sum Re beta <= B_p P_nu.
- Valid for every p (including 3), n and network. The corollary is for S=F and s=1 only, as stated.
Independent check (verify/v2_w4.py) used random NON-uniform symmetric mu on 296 (network, j, chi) triples, including DC and multi-frequency networks: identity error 2.7e-15, max(h - bound) = -0.052, corollary max = -4.8e-4.

### W5: CONFIRMED

The algebra is right: s log P_b = log P_nu + (s-1)H_{1/s} for any positive vector. SPLIT R1 is used within its hypotheses: ||omega||^b >= (sum_k ||omega^{(k)}||^2)^{b/2} >= sum_k ||omega^{(k)}||^b for b >= 2, with equality at b=2 iff there is no DC. The gamma comparison needs h(theta) >= 0, which is implicit.
Independent check (v1): identities to 1.8e-15; split side minus theta side <= 1.8e-15 on 240 cases with and without DC, for s in [1, nu/2].
Caveat on the 'reading' only. At b > 2 sharing can RAISE the Renyi entropy. My DC-free example at (5,2), s=1.2: H(m_theta) = 0.00620 > H(m_split) = 0.00545. So 'sharing lowers entropy' holds only at b=2 without DC. For b>2 the inequality is a one-sided bound on the change, not a sign. The mathematical statement itself is correct.

### W6: CONFIRMED

(a)
- Rotfel'd/McCarthy (the r-triangle inequality for Schatten quasi-norms, valid for arbitrary matrices) and Weyl's majorant (t -> e^{rt} convex and increasing) are used correctly.
- Lambda >= |tr H|/nu = |A|, and H_{1/s} >= H_1.
- IF gives zero interval coefficients below the diagonal (H_ab = beta_{(b,a]} for a > b), so H - AI is strictly upper triangular.
(b)
- Rotfel'd for the PSD rank-one terms, then Schur on the Gram side: diag (nu rho_l) is majorised by the spectrum, so H_alpha(rho/L) >= H_alpha(pi).
(c)
- With the TeX convention u_{l,t} = p_l prod xi (c9 uses it; c1 uses p_l^t, but c9 covers the stated version): K_t0 = beta_[1,t] exactly.
- The twist error, by Cauchy-Schwarz, is sqrt(2L(L-|A|)).
- H_2 = log nu - log(1+V), with diagonal entries L. Then (s-1)2/nu <= 1-2/nu.
- The last exponential bound needs s <= nu/2, which is stated.
Independent check (verify/v3_w6_w7.py): random (non-normal) matrices for Rotfel'd/Weyl, 4800 atom systems for n=1..8 (including real cancelling, aligned and root-of-unity xi), and aligned chains for n=1..9. All excesses are <= 3.8e-15 once round-off eigenvalues of rank-deficient matrices are clipped. Without clipping, spurious excesses of 1e-3 appear, coming from (1e-16)^r terms.

### W7: CONFIRMED

Proof rebuilt.
- Q_j = L_j^r e^{(1-r)H_{1/s}(rho/L)} >= |A_j|^r e^{(1-r)H_{1/s}(rho/L)}, and W6(a) gives Q_j >= |A_j|^r e^{(1-r)H_{1/s}(mu_j)}.
- The budget gives sum_j Q_j <= B_p^r P_b.
- With y_j = |A_j|/(sum|A|^r)^s, which is feasible, Lemma 2 gives Delta' >= 0.
- The algebra and the winner-width corollary follow. H~ is an exponential mean of the H_j, so H~ < log nu forces some H_j < log nu, for each slot order (the feature may depend on the order).
Independent check (v3) on 41 networks with shared (merged) neurons, DC and perturbations at (5,2), (7,2), (5,3), (3,3), (7,3), using the stated H_j = max(H(mu_j), H(rho_j/L_j)): min slack -1.8e-15, Delta' >= 0. Exact p=3 certificates: n=11 gives gamma/gamma_IF = 1.0544, slack 1.177, Delta' 0.821; n=13 gives slack 1.350. So W7 is consistent with the p=3 counterexamples. It is an inequality, not a value claim.

### W8: CONFIRMED

Every step re-derived.
- cm >= c means Phi - c lies in the sector of half-angle pi/p, so Re z^2 >= cos(2 theta)|z|^2.
- Circularity gives E(Phi-c)^2 = (A-c)^2, and Parseval applies to the distinct characters 2j 1_T.
- Algebra: 2u^2 - 2t^2 cos^2 theta <= 2u^2 sin^2 theta because t >= u >= 0, and 2 sin^2 theta / cos 2 theta = sec(2 pi/p) - 1 = k_p.
- c/a = cos psi - u, so u <= x.
- The cone lemma at j delta = +-1 gives h <= kappa_min c.
- The budget and W6(c) apply with L = a, and sum over intervals <= sum over T.
- (s-1)2/nu <= 1 - 2/nu; k_5 = sqrt5 exactly (difference 4.6e-41); k_p decreases in p.
- Elementary lemma: u2 = 0.6624598 and F(u2) = 0.1047656 > 0, checked at 40 digits.
- The circularity claim is valid at p >= 5 for single-frequency inputs with DC: Fourier support in {0, j, 2j}^n, and -j = 2j only when p = 3.
Independent adversarial optimisation (verify/v4_w8.py) over phase-aligned single-frequency + DC networks at (5,3), (5,4), (7,3), (5,2), (7,2), b=2, several widths: best gamma/gamma_IF = 0.99999x, always at x = 0 (IF), never above 1.
The hypotheses are strong: phase alignment (L = |A|) removes exactly the cancellation where the p=3 winners (1 - |A|/L = 0.22) and the large-n Gaussian combs (LARGEN R5/R9) live. Within its stated hypotheses the theorem is correct and n- and b-uniform.

### W9: CONFIRMED

g(x,delta) = P(x)(w(y) - w(y-delta)). If h > 0, then P(x) != 0, and y(x) is the strict argmax of w where P > 0 and the strict argmin where P < 0. So at most two labels occur, but y is onto Z_p with p >= 3. This holds for every n >= 1 and every p >= 3. Random search at (3,2), (3,6), (5,3), (7,2) found a max single-neuron margin of -4.7e-3 (v1).

### X1: CONFIRMED

Valid refutation, recomputed independently (verify/v5_x1_x2_w3.py) with my own vertex-enumeration programme, by building the uniform IF network and computing its margin directly.
- p=11: log gamma - RHS = +0.18157, +0.18078, +0.37777 at s = 1.35, 1.45, 1.49. The closed form log(F^{-s}(p/2)/Gamma_s) - (s-1)(H(z*) - log F) agrees.
- p=13: +0.124, +0.232, +0.232.
- Equality holds at s=1.2.
Inputs: pi_{2,b} = 3^{1-s} is the cited n=2 price theorem. H(theta_1) = log 3 for every optimal single feature follows rigorously from Danskin, because log pi is differentiable at interior s. A smaller H^dec would only enlarge the violation. Since even Delta = 0 fails, no Delta >= 0 rescues the s-local form.

### X2: CONFIRMED

Recomputed. With S=F at (5,2), H^dec = log 6, and the inequality reduces to kappa_min 4^s <= 5, which fails for s > log2(5/kappa_min)/2 = 1.42760. I get log gamma - RHS = +0.0033, +0.0311, +0.0865, +0.1004 at s = 1.43, 1.45, 1.49, 1.5, and -0.0105 at s = 1.42. These are interior s, where H(theta_1) = log 3 holds for all optimal single features by Danskin. The counterexample uses S=F, not S = spt(theta). That is legitimate for the statement as written ('theta in N_S').
