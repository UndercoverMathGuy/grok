# Referee verdicts on E_split/REPORT.md (saved by the orchestrator)

## Overall
I rebuilt all twelve claims line by line and ran independent checks; none is false and none has a serious gap.

- **CONFIRMED (6):** E2, E2b, E3, E5, E6, E9. The two refutations (E2b and E9) stand.
- **MINOR_GAP (6):** E1, E4, E7, E8, E10.
  - E1: boundary and wording defects (the s = 1 equality case in (c), "proportional tables" in (d), h > 0 in (e), the DC block in (f), S non-empty in (h)).
  - E4: the unsupported sentence "the only monotone quantity".
  - E7: a p = 3 exception to a side remark.
  - E8: "(E^C_S)" is never defined.
  - E10: a misleading title.

Mathematically, the batch reduces (E_S) to a local statement. E3 settles the convex end s = 1 rigidly for every p and every n >= 2: maximisers are frequency-pure, IF, aligned, with H* = H^dec, so L'_+(1) = 0. E4 shows (E_S) is equivalent to W_S ≡ D_S, to (EW_S), and to M_S = L_S/(s-1) being non-increasing. E5 shows a failure must bifurcate from a maximiser at a point where L_S = 0 with entropy at most H^dec_S. If that point is s = 1, the maximiser is the decoupled IF block configuration.

**Value claims vs p = 3:**
- E3's convex-end bound holds on the exact certificates: gamma(.;1)/(B_3 Gamma_1) = 0.139 at n = 11 and 0.147 at n = 13.
- E4(c) is n = 2 only and conditional.
- At p = 3, (E_S) holds trivially because every network is a single feature.

So nothing conflicts with the counterexamples.

All scripts and outputs are in /home/user/grok/heracles/conjecture20/unify/E_split/verify/:
- vlib.py
- v1_calculus (E1, E9)
- v2_entropic and v2b_polish (E2, E2b; 40-digit exact counterexample)
- v3_convex_end and v3b_polish_s1 (E3, including the Fourier-coefficient identity and the p = 3 certificates)
- v4_mode_budget (E7)
- v5_kkt (E6)
- v6_monotone (E4, E5)

## Generality audit
I audited each claim for all odd p, every n >= 2, every b in [2, nu] and every S.

**Genuinely general (no case split):** E1 through E6, E8 and E10.
- E2 does not depend on p at all, and holds for every r in (0,1], beyond the b range.
- E3 uses n >= 2 (nu >= 3, so that ||A||_nu = ||A||_2 forces a single entry), p odd (2l 1_T is unique), and p prime (kappa_j(delta) > 0). All hold throughout the allowed range, including p = 3; I checked (3,3) explicitly.
- E4(c) is a conditional corollary stated for n = 2 only. It is an instance, not hidden special-casing.

**Hidden special-casing found:**
1. E7(ii): "No positive constant is possible at b = 2" fails at p = 3. There |F| = 1, the counterexample neuron degenerates (N = 0), and c_b = 1/nu provably works at b = 2. The remark needs p >= 5. The inequalities (i), (ii) and (iii) themselves hold for all p, n, b.
2. E1(c): the header includes s = 1 (b = nu), where "equality iff t uniform" fails, because H_1 = H_Shannon.
3. E1(h) needs S non-empty; E1(e) needs h(sigma theta) > 0.

**No parameter limits:** none of the claims uses p >= p_0, b near nu or large n.

**Refutations:** E2b is shown at n = 2 and E9 at p = 11, n = 2, b = 3. A single instance suffices for a refutation, so no generality is required there.

## Salvage
- **E1:** state (c) for s > 1. In (d), add the balanced-maximiser remark (even sign flips; a negative ratio cancels), so that "proportional tables" is fully covered. In (f), sum lambda_i over all blocks for block splits, including the DC block. In (e) assume h(sigma theta) > 0, and in (h) assume S is non-empty.
- **E2:** add the shorter, stronger frame proof. Aligned weights with the interval IF patterns give sum w_l m_l m_l^* = I_nu, so max w_l <= 1/nu, so H_inf(esc_r(w)) >= log nu for every r <= 1. This proves Renyi-entropy bounds of every order and uses only interval patterns.
- **E4:** delete the sentence "the only monotone quantity that works".
- **E5:** specify the topology: neighbourhoods in the padded, normalised width-D parameter space K_D.
- **E7:** restrict the b = 2 no-constant remark to p >= 5. Note that at p = 3, c = 1/nu works at b = 2. Define N^(l) with output in F.
- **E8:** define (E^C_S). Add one line that E3 and E4 transfer verbatim to class C: width reduction keeps a subset of neurons, class-C Danskin holds, and pi^C(1) = 1 via the block.
- **E10:** retitle it "exact entropy of maximisers under W_S ≡ D_S". It is not the upper half of (C).
- **What remains open after this batch:** (Loc_S) near low-entropy maximisers at departure points s_0 with L_S(s_0) = 0. At s_0 = 1 this is a concrete local question around the decoupled IF block configuration for b slightly below nu. At s_0 > 1 the local structure is unknown.

## Verdicts

### E1: MINOR_GAP

I re-derived every part and all the identities hold for every p, n and b. (a) Use H = log Z - E_w log c and (s-1)b = nu-b. (b) phi(s) + (1-s)phi'(s) = F and d/ds[...] = (1-s)phi''. (c) A copy t^{1/b}omega has table t^s T and sum t^s = e^{(1-s)H_s}; the rate is ~ s/log(1/tau) -> 0. (d) (1+x)^{1/s} < 1 + x^{1/s}; the pair is the replication of (1+lambda^b)^{1/b}omega, and F is scale-invariant. (e) The flips are orthogonal on every factor. (f) c_ik = c_i rho_ik^{b/2} plus the chain rule; the block-group character prod_t S[B_t,t]S[B_0,t] averages to 1 iff all labels agree. (g) The copy's gap is z_j^s g(jx,jdelta). (h) Cancelling pairs.
My independent check is verify/v1_calculus.py -> v1_calculus.out, on (5,2), (7,2), (5,3), (3,2), (3,4), (11,2), (3,3):
- F formulas, tangent intercept, dF/ds and the quad-integral identity agree to 1e-9;
- replication with Dirichlet t agrees to 2e-15;
- the merge with even sign flips is strictly cheaper, and the pair formula holds to 2e-15;
- per-neuron Sigma keeps c_i, beta and beta_T to 7e-15;
- the split chain rule holds to 9e-16;
- the block split equals the explicitly enumerated block-sign-group average and keeps beta and beta_T to 2e-15, without raising the cost;
- the dilation-union bounds hold;
- the cancelling-pair gamma ratio is exact.
Minor defects:
(1) The header includes s = 1, where 'equality iff t uniform' in (c) is false: H_1 = H, so dF = 0 for every t. I observed dF = 4e-16 for non-uniform t at s = 1.
(2) The proof of (d) covers only omega and lambda*omega with lambda > 0, but the conclusion is about 'proportional tables'. It extends because maximisers are balanced: positive-ratio proportional tables then differ by an even number of factor sign flips, which change neither table nor cost. A negative ratio gives cancellation, which is strictly cheaper. This needs to be said.
(3) (e) needs h(sigma theta) > 0 for F to be defined.
(4) 'The same formulas hold for every block split' must sum lambda_i over all blocks, including the block that contains 0. In (f) the sum is over F only.
(5) (h) needs S non-empty.

### E2: CONFIRMED

The proof is correct. Paper Lemma 7 (appendix lem:cost) holds for every r in (0,1] and for arbitrary configurations satisfying (JF). With aligned atoms, |beta| = sum w = 1, so psi(r) >= (1-r)log nu. Then psi(1) = 0, so psi'(1) <= -log nu. psi is convex (log-sum-exp in r), so psi'(r) <= psi'(1), and psi - r psi' >= log nu. The converse follows from d/dr(psi/r) = -H(esc_r)/r^2.
Independent stronger proof: aligned weights make H = sum_l w_l m_l m_l^* Hermitian PSD. It is triangular by the interval IF patterns, so H = I_nu. Then 1 = m_l^* H m_l/nu >= w_l nu, so max w <= 1/nu. This gives max esc_r(w) <= w_max <= 1/nu for every r <= 1, hence H(esc_r) >= H_inf(esc_r) >= log nu, for every Renyi order and using only the interval patterns.
Check: verify/v2_entropic.py -> v2_entropic.out. I adversarially minimised H(esc_r), using a penalty plus an exact least-squares polish with IF residual < 1e-10, for n = 2, 3, 4, m = nu..nu+4 and r in {1, 0.6, 2/nu}. The overall minimum of H(esc_r) - log nu is -2.2e-16. At every exact IF optimum, the frame identity sum w m m^* = I (to 1e-8) and max w <= 1/nu were asserted. The claim is valid for all n, independent of p.

### E2b: CONFIRMED

The refutation stands. The printed 8-digit data have IF residuals of about 4e-9. I polished them by 40-digit mpmath Newton steps (verify/v2b_polish.py -> v2b_polish.out). The result is an exact IF configuration (residual 1e-41) within 5e-9 of the printed point, with all moduli positive. It has H = 0.948612287 < log 3 = 1.098612289 and |beta| = 0.1249. I also realised it as three genuine product neurons at p = 5, n = 2 (u_t = cos(2 pi a/p - xa_lt/2), and w chosen so the atoms match to 2e-16). The recomputed beta_{1,T} vanish to the printed precision. Lemma 7 holds with large room (1.39 >= 0.36 at r = 2/3). Since this is a single explicit counterexample, generality is not at issue.

### E3: CONFIRMED

I rebuilt the proof.
Upper bound chain: h <= A(delta) (Paper Lemma 2, symmetrised over +-delta) <= B(delta). Then min B <= (sum_S sum_i|beta_ij|) Gamma_1^S <= (sum_F ...) Gamma_1^S <= B_p sum a Gamma_1^S.
Budget equality: Hoelder sum_j prod_t A_t(j) <= prod_t ||A_t||_nu <= prod_t ||A_t||_2 <= 1 (Parseval), plus AM-GM. Equality forces: no DC; one non-zero entry per factor (nu >= 3, i.e. n >= 2); the same plane in every factor; equal norms.
At a maximiser every step is tight:
- kappa_j(delta*) > 0 for p prime gives (ii);
- off-S atoms vanish and every neuron is tight, giving (i);
- y attains Gamma_1^S, giving (iv);
- h = A(delta*) forces both E_x g(.,+-delta*) = h, so g(.,delta*) is constant.
Fourier coefficient: the coefficient of g(.,delta) at 2l 1_T in class C is exactly (1/2) conj(beta_{l,T})(1 - rho^{-l delta}), using odd p and l, j in F. I verified this to 8e-16 in verify/v3_convex_end.py (B); this gives (iii). The converse is the clean-table computation.
Entropy: tight neurons give w_i = |beta_i|/sum, and E2 at r = 1 gives H >= log nu + H(y). Optimal single features at s = 1 (Companion definition: outputs in span{1,cos,sin}) are forced to Phi_1 = beta real and constant, i.e. aligned IF, with least entropy log nu. So H* = H^dec at s = 1.
Danskin's right derivative at the endpoint s = 1 is legitimate: the formula is smooth for s > 0, and the argmax set at s = 1 lies in a compact set with h >= eta.
Checks:
- verify/v3_convex_end.py (A): non-block aligned IF networks ('wide' with nu+2 neurons, unequal two-block mixtures) attain B_p Gamma_1^S exactly (ratio 1.000000000000) for (5,2), (7,2), (5,3), (11,2), (3,3) and several S, with H - H^dec >= 0.
- (C): breaking any of (i)-(iii) strictly lowers gamma.
- verify/v3b_polish_s1.py: epigraph SLSQP maximisers (ratio 1.000000) are single-plane to 1e-13 and IF to 1e-14, with H - H^dec in [0, 0.116]. Non-maximisers can have H < H^dec, which is allowed.
- p=3 certified networks evaluated at s = 1: gamma/(B_3 Gamma_1) = 0.139 (n=11) and 0.147 (n=13), both <= 1, so there is no conflict with the counterexamples.
The result holds for all odd p and all n >= 2.

### E4: MINOR_GAP

The equivalence theorem and the derivative formula are correct.
- J = L - (s-1)L'_+, so M'_+ = -J/(s-1)^2.
- (vi)=>(ii): L ≡ 0 gives H*_S = H^dec_S.
- (i)=>(iv): F = log W_S + (s-1)H and W_S >= D_S.
- (iv)=>(v): L is Lipschitz, because log W_S, log pi and log Gamma^S are sups of curves with |slope| <= log(width). So M is locally absolutely continuous with M' <= 0 a.e.
- (v)=>(vi): M >= 0, M is non-increasing and M(1+) = 0 (E3).
- (iii)=>(vi): L'_+ <= 0 on {L > 0}, plus Dini monotonicity.
- The Danskin facts ((log D_S)'_+ = -H^dec_S via pi and Gamma^S separately, then summed) are valid.
Corollaries:
- (a): a union of relabelled single features lies in N_S, so W_S >= B_p Gamma^{P,S} >= D_S.
- (b): Gamma^{{l}} = kappa_min.
- (c): uses Note Thm q1n2 (pi_{2,b} = 3^{1-s} at every prime). It is conditional and n=2 only, consistent with the p=3 counterexamples at n >= 11. At p=3, (E_S) holds trivially, since every network is a single feature, so W_{1} = D_{1}; there is no contradiction.
Check: verify/v6_monotone.py -> .out, on 200 envelopes of genuine temperature curves:
- the Danskin right derivative matches finite differences to 7e-8;
- M'_+ = -J/(s-1)^2 holds to relative error 2e-6.
Gap: the sentence 'the only monotone quantity that works is M_S' is undefined and unproved. Strike it, or label it as an opinion. The mathematics otherwise holds for all p, n, S.

### E5: CONFIRMED

I checked the steps.
- Tangent at s_0: phi_k(s_k) <= phi_k(s_0) - H(theta_k;s_0)(s_k - s_0), with phi_k(s_0) <= log D_S(s_0). Also phi_k(s_k) = log W_S(s_k) > log D_S(s_k) = log D_S(s_0) - H^dec(s_k - s_0) + o. This gives the entropy bound.
- The same tangent inequality, rearranged, gives phi_k(s_0) >= log W_S(s_k) -> log W_S(s_0).
- gamma(.;s_0) and H(.;s_0) are continuous on the compact K_D, because Z >= D^{1-b/2} there. So limits are maximisers with H <= H^dec.
- s_0 = 1 is E3.
- (Loc) <=> (E_S) is correct: the contradicting sequence lands in U for large k.
The 'neighbourhood' must be read in the padded width-D parameter space; this should be stated.
Check: verify/v6_monotone.py (3). At the first departure of 200 envelopes of real temperature curves, winners have H(theta_k, s_0) - H_D(s_0) <= -0.126 over 990 cases. The result holds for all p, n, S.

### E6: CONFIRMED

The directional derivative of log min_a sum lambda_i^nu g_i(a) - s log sum lambda_i^b c_i at lambda = 1 is min_{a in A} nu sum v_i(g_i(a)/h - w_i), using sb = nu. If 0 were not in the convex hull of these finitely many vectors, separation would give a strictly ascending direction. So q exists. The floor follows from |g_i| <= 2 ||w||_inf prod ||u_t||_inf <= 2 nu^{-nu/2} a_i (AM-GM).
Only global maximisers are claimed; an N_S maximiser need not be rescaling-stationary inside N_S, and the claim correctly does not say it is.
Check: verify/v5_kkt.py -> .out.
- IF optima at (5,2,2.5), (7,2,2.2) and (5,3,3.2) (gamma/gamma_IF = 1.00000000): LP residual exactly 0 on argmin sets of size 100, 294 and 500.
- The same networks with one block rescaled by 1.3 give residual 0.04-0.09, so the LP discriminates.
- The share floors hold.
- The p=3 certificate networks are not rescaling-stationary (single active point, LP infeasible). This is no contradiction: E6 assumes a maximiser.
The result holds for all p, n, b.

### E7: MINOR_GAP

Re-derived:
- |beta_j| = Ehat prod_t a_t(j)^{1/2};
- total mode mass (output in F) = 2^n Ehat^2 (1 - a_0(0)), and own mass = 2^n Ehat^2 x, so (i) holds;
- for (ii), sum g_j <= 1, sum g^q <= M^{q-1} and x >= M^nu; eta is decreasing on [0,1] with eta(1) = 0; Ehat <= e and r < 2;
- (iii) follows by Minkowski in l^{2/r} with (sum rho^r)^s >= sum rho, and by Cauchy-Schwarz with ||e||_{2-r} <= ||e||_r.
Check: verify/v4_mode_budget.py -> .out, with brute-force mode enumeration.
- 1280 neurons with p in {3,5,7,11}, n in {2,3}: (i) max excess 1.3e-14, (ii) 6.7e-16.
- The b=2 example is exact: budget 1.000000000000 and 2^{-n}N/e^2 = 1 - sum a^nu.
- Network forms: -3e-6 and 1e-16, i.e. <= 0.
Gap (hidden special-casing): 'No positive constant is possible at b=2' requires |F| >= 2, i.e. p >= 5. At p=3 the example degenerates (N = 0), and in fact c = 1/nu works at b=2. Proof: 1 - g >= (1 - g^nu)/nu, and N = Ehat^2 (1 - a_0(0) - g^nu). Numerically, min [e^r - |beta|^r]/(e^{r-2} 2^{-n} N) = 0.356, 0.293 and 0.222 >= 1/nu at n = 2, 3, 4.
Also, N_i^(l) in (iii) should state explicitly that output-DC modes are excluded. Including them with the natural p^{-nu} normalisation is still harmless.

### E8: MINOR_GAP

(a) Correct in both directions:
- <=: W_S = D^C_S <= D_S <= W_S;
- =>: D^C_S <= W^C_S <= W_S = D_S = D^C_S. Identities must hold on the whole range.
(b) Correct algebra, provided (E^C_S) means W^C_S ≡ D^C_S. (E^C_S) is never defined. If it means the entropy statement inside class C, the step (E^C_S) => W^C_S ≡ D^C_S needs class-C versions of E3 and E4. These hold but are unstated:
- width reduction keeps a subset of the neurons, hence stays in class C;
- class-C Danskin for W^C_S and pi^C;
- pi^C(1) = 1 via the block;
- the class-C convex end is identical to E3.
Consistency at p=3 holds: there every network is a single feature, so W_{1} = pi B kappa_min, W^C_{1} = pi^C B kappa_min, and MM^rel is an equality.
(c) Correct: a plane-k neuron gets the sign (prod_t s_{k,t})^2 = 1, and its split has one non-zero piece. Checked algebraically and via v1_calculus block-split enumeration.

### E9: CONFIRMED

The refutation is valid. I rebuilt both networks independently (verify/v1_calculus.py, E9 section):
- p = 11, n = 2, b = 3, small neuron at norm^2 ratio 0.1 spread equally over 5 planes, plus a pure neuron: H(theta) = 0.1370 -> H(split) = 0.0959. By hand: lambda_1 = 5^{1-b/2} = 0.447 shrinks the mixed neuron's cost weight.
- b = 2 with DC, equal norms, neuron 1 at 90% DC: H = 0.6931 -> 0.3046.
- At b = 2 without DC the same pair gives an increase (0.3046 -> 0.4509), as E1(f) predicts.
It is a single explicit counterexample. At p=3, b > 2 without DC the split is the identity, so the refuted statement happens to be true there, but that does not affect the refutation.

### E10: MINOR_GAP

The content is correct under the stated hypothesis.
- Lower half: as in BRIEF_E.
- Exact entropy: by GLOBAL R2 restricted, D^- log W_S = -max H and D^+ = -min H over maximisers. At points where log D_S = log W_S is differentiable (a.e. by Lipschitz), max H = min H = -(log D_S)' = H^dec_S.
- Variance: Var_w(log w) is bounded on the D-simplex, so phi_theta'' >= -C uniformly, and log W_S is semiconvex. It is twice differentiable a.e. (1-D Alexandrov), and its second derivative equals d/ds of -H^dec_S a.e. Second-order contact of phi_theta* from below gives -Var/s <= -(H^dec_S)'.
Gap: the title 'Upper half of (C)' is misleading. Nothing here proves the upper half, as the claim itself admits ('do NOT exclude extra features'). Retitle it, e.g. 'Exact entropy of maximisers under W_S ≡ D_S'. No numerical check beyond consistency is possible; the logic is pure analysis and holds for all p, n, S.
