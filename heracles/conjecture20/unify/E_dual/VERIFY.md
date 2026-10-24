# Referee verdicts on E_dual/REPORT.md (saved by the orchestrator)

## Overall
No claim is FALSE. The two refutations (D8, D9) stand; I reproduced them independently from exact rational data at (5,2,2), s = 3/2.

Verdicts:
- CONFIRMED (PROVED claims): D1, D1v, D2, D4, D4', D5, D6 (as a correctly hypothesised special case), D7.
- MINOR_GAP: D3a, D3b, D3c, D10.
  - D3a: an undefined Phi^dec, and decoupled margins can exceed the price formula.
  - D3b: 'any proof must be two-level' is a heuristic.
  - D3c: 'the s=1 certificate structure cannot exist at s>1' fails whenever maximiser neurons have equal size, e.g. p=5 for s < 1.855, K*=1 regimes, and the agent's own (7,2,1.3) case.
  - D10: logic correct, but the tolerance table is numerically wrong in places. At p=13, s=2 it is overestimated by about 3x (exact vertex enumeration plus an explicit witness), and at p=11, s=1.5, eps=1e-3 by 2.4x.
- Both refutations carry overreaching side remarks:
  - D8: 'Forster bounds can hold only at gamma = W' is not shown.
  - D9: 'W_F = D_F at (5,2) is the p52 theorem' is a false citation. p52 is a count theorem (one-feature networks <= 0.995 gamma_IF), not the value identity.

Substantively, the PROVED results are exact reformulations, not progress on the target:
- D1 and D3a: the s-dependence is a Renyi entropy of an s-free distribution.
- D4: (E_S) is equivalent to W_S == D_S. So the entropic route has zero slack: proving (E_S) IS proving the restricted price-programme value identity.
- D3b and D3c: obstructions to single-tensor duality.
- D9: the identity is false for profile classes. Feature classes are untouched.

Nothing here contradicts the p=3 counterexamples:
- D7 holds on the exact n=11 and n=13 certificates (gamma/bound = 0.139, 0.147).
- D3b's gap uses p >= 5 exactly where needed.
- D4' is an s=1 statement.

All referee scripts and outputs are in /home/user/grok/heracles/conjecture20/unify/E_dual/verify/. v3_kkt.py's own main loop stalled and was killed; its .out file now holds only a note, and v3b_kkt_main.py runs that check instead.

## Generality audit
General for every odd p, every n >= 2 and every b in [2, nu], with no hidden case split:
- D1, D1v, D2 (needs s > 1, i.e. b < nu, intrinsically), D3a (uses s >= 1), D3c, D4, D5 (any n >= 1, arbitrary atoms), D7, D10(a,b).
- D3b: its strict gap Gamma_s < Gamma_1 needs p >= 5. That is intrinsic (equality at p=3), not a parameter limit; every other part is general.

Restricted in scope:
- D4' holds only at the endpoint s = 1 (b = nu). It is general in p and n but gives only L_F = o(s-1); it carries no information on any open s-range.
- D6 is explicitly special-cased and violates the user's hard rule:
  - aligned atoms, class C, single-frequency single feature;
  - (c) holds for all n and b only when p >= 7 (tau_p <= 1);
  - at p = 5 only when s - 1 <= 0.2236 nu;
  - nothing at p = 3.
  The agent labels it so; it must not be used in the general programme.
- D8 and D9 are single counterexamples at (p,n,b) = (5,2,2). That is legitimate for refutations. D8's decoupled side uses n=2-specific results (Note Thm q1n2, Theorem A for b in [2,3], KM2), which is acceptable inside a counterexample.

Hidden parameter dependence in the numerics: the D10 tolerance values are reweighted-LP upper bounds and some are wrong. Exact values are in verify/v6_d10_tau.out.

No PROVED claim proves (E), the count (C) or any value identity at any (p,n,b).

## Salvage
- D3a: define Phi^dec explicitly as the price-formula value log B_p + log e(theta_1;1) + log M(yhat) (not the actual decoupled margin, which can be larger). Then the sup-convolution and transform-additivity statement is exact. Drop 'convex objective'.
- D3b: keep the theorem. Restate the consequence as 'no bound of the form h <= <E_q g, f> <= ||E_q g||_sigma x cost, with any entropy cap, is tight for s > 1, p >= 5'. Delete 'any proof must be two-level'.
- D3c: keep the identity t_i = h nu^{nu/2} a_i^{r-1} and the F/KL bounds; they are valid at every KKT point. Restrict the negative conclusion to maximisers with unequal neuron sizes. Note that it is silent at p=5 (s < 1.855), in K*=1 regimes, and wherever y* gives equal neuron costs.
- D8: delete 'can hold only at gamma = W'. State: the exact decoupled frontier fails at level 0.790 gamma_IF; levels in (0.790, 1] are untested.
- D9: replace the p52 citation. W_F = D_F at (5,2) is open; p52 proves only the count. Keep 'feature classes are not refuted'.
- D10: replace the floor_tolerance table with the exact vertex-enumeration values in verify/v6_d10_tau.out:
  - p=13, s=2: tau = 1.41e-3, 1.41e-2, 4.38e-2 at eps = 1e-6, 1e-4, 1e-3;
  - p=11, s=1.5, S={1,4,2}: saturates at 4.23e-3 from eps = 1e-3.
  State tau ~ c eps^{1/s} as numerical only. Present (b) as a reformulation, not a reduction.
- D6: keep it as a located obstruction only, as the agent intends. Note that (a) needs no alignment, and (c) holds for any common phase, not only real atoms.

## Verdicts

### D1: CONFIRMED

I re-derived all four parts.
- (i) is pure algebra: s log sum p^{1/s} = (s-1) H_{1/s}(p).
- (ii) is the Gibbs variational principle with x_i = (1/s) log p_i. The objective KL(pi||p) - (s-1)H(pi) = s sum pi log pi - sum pi log p is strictly convex for every s>0, so the minimiser w(s) is unique.
- (iii): psi(beta) = (1+beta) log sum p^{1/(1+beta)} has derivative H(w(1+beta)), which is non-decreasing because dH/ds = Var/s. At eta = psi'(s-1) the conjugate equals (s-1)H(w) - s log sum p^{1/s}, which is KL(w||p) by (ii).
- (iv) uses 1-homogeneity of M and (i) applied to yhat.
Hypotheses are only h>0 and zero neurons dropped.
Independent check (verify/v1_d7_lemma2.out (5), 50 digits, 100 random cost vectors with s in [1,4]):
- (i) error 4e-50;
- the Gibbs equality at w_s has error 2e-50;
- 2000 random pi never beat the maximum.

### D1v: CONFIRMED

Each factor of copy j is scaled by y_j^{1/nu}, so a_(j,i) = y_j a_i, and relabelling by an invertible j (p prime) preserves norms. Hence p_Theta = yhat (x) p_1, and Renyi additivity holds for every order.

Copy j has f'(x)[c] = f(jx)[jc], so its only clean frequency is +-j. This needs theta_1 to have frequency 1 as its only feature. Hence |beta_j(Theta)| = y_j |beta_1|.

The value formula is D_S = pi B_p Gamma^S_s combined with e(theta;s) = e(theta;1) K_r(p)^{1-s} (D1(i), since P_b^s = (sum a) K_r^{s-1}).

Independent check (verify/v8_d1v_product.out) with a NON-IF single feature (Gaussian inputs, DC, frequency-1 output) at (7,2), (11,2), (7,3):
- profile error 4e-16;
- product error 8e-17;
- Renyi additivity at orders 0.4, 1, 2 to 1e-15.
The 'entangled state / multiplicativity' reading is only an analogy.

### D2: CONFIRMED

Each appended pair (u_1,...,u_n,w), (-u_1,u_2,...,w) has tables T and -T, so the table is unchanged. Hence:
- the cost is Z_0 + c, independent of M;
- the dust contributes (c/(Z_0+c)) log(2M(Z_0+c)/c) to H, which tends to infinity;
- gamma' = gamma (1+c/Z_0)^{-s}.
F tends to infinity requires s>1; it is stated for s>1. The claim is valid for all p, n, b and is trivial.
The consequence 'only entropy caps constrain' is a correct reading: sup{gamma : H >= eta} = W for every eta (not attained).

### D3a: MINOR_GAP

The main identity log W_S = sup_eta [Phi_{S,1/s}(eta) - (s-1)eta] is correct.
- <= : take eta = H_{1/s}(p_theta).
- >= : log gamma = log gamma_1 - (s-1)H_r >= log gamma_1 - (s-1)eta when H_r <= eta. This uses s >= 1, which holds on the range.

Gaps:
(1) 'Phi^dec is the sup-convolution ... and its transform is the sum' uses an object that is never defined.
- It is true for the PRICE value log B_p + log e(theta_1;1) + log M(yhat).
- It is not true for actual margins of decoupled unions: h(union) = min of a sum >= sum of mins, so gamma_1(Theta) can exceed the product formula.
(2) Calling the capped s=1 problem 'a convex objective' is misleading: the entropy cap is on the decomposition, not the table, and is non-convex.
The 'direction of the bound' remark (h/P^s = gamma_1 K_r^{1-s}) is correct.

### D3b: MINOR_GAP

The mathematics is correct.
- The capped set contains all unit rank-ones (point masses have H_r = 0) and lies in their convex hull, so the supremum is ||E_q g||_sigma.
- At s=1, minimax is not even needed. The sandwich is nu^{-nu/2}||E_unif g||_sigma <= (p/(p-1))B_p = B_p Gamma_1 = W(1) <= nu^{-nu/2} inf_q ||E_q g||_sigma, using the mean-gap identity and the r=1 budget. I verified the identity independently to 3e-16 on 3000 networks with DC at p = 3..11.
- Strictness Gamma_s < Gamma_1 holds for p >= 5 and s > 1: either K_r(yhat*) > 1, or yhat* is a point mass with M = kappa_min = 1 - cos(2pi/p) < 1 < p/(p-1). At p=3 it is correctly equal (kappa_min = 3/2).

Independent ALS check (verify/v5_d3b.out) at (3,2), (5,2), (7,2), (5,3):
- uniform q gives exactly 1.00000000 x B_p Gamma_1;
- the ALS lower bounds for random q are 1.21-1.39 x B_p Gamma_1.

Gap: 'for s>1 there is no exact dual in terms of a single dual tensor' is proved only for bounds of the literal form h <= <E_q g, f> <= ||E_q g||_sigma x cost. 'Any proof must be two-level' is a heuristic, not a theorem.

### D3c: MINOR_GAP

I re-derived the KKT part.
- A local maximiser of gamma at unit cost is a local maximiser of (t subject to g >= t and cost <= 1).
- MFCQ holds: along (-theta, -nu h - 1), active g - t increase at rate 1 (g is nu-homogeneous), and the cost decreases at rate b.
- Pairing the stationarity equation with each factor gives equal factor norms and E_q g(omega_i) = lambda b ||omega_i||^b / nu. Summing gives lambda = nu h / b.
- Hence t_i = h nu^{nu/2} a_i^{r-1}, and E_w log t = log h + (nu/2) log nu + (s-1)H(w). The F and KL bounds follow from t_i <= ||E_q g||_sigma and D1(iii).

Independent check at GENUINE local maxima from Gaussian starts (own optimiser, softmin plus sequential-LP polish, q by bounded least squares; verify/v3b_kkt_main*.out):
- (5,2,b=2), neuron sizes 0.61-1: stationarity residual 6.5e-5, max|t/pred-1| = 1.7e-5, F identity 3e-9, lambda b/(nu h) = 1.000000;
- (7,2,b=2.6), sizes 0.33-1: 6.8e-4 at residual 3e-3;
- equal-size maxima: about 1e-7;
- poorly converged runs (residual 0.05-0.5) mismatch in proportion to the residual, which is not evidence against the claim.

Gap: 'the s=1 certificate structure cannot exist at s>1 / answers (b) in the negative' holds only at maximisers with UNEQUAL neuron sizes. It says nothing:
- at p=5 for s < 1.855 (y* uniform, so all neurons are equal);
- in K*=1 regimes;
- at the agent's own (7,2,s=1.3) case, where the t spread is 1.000..1.000.
It is also never shown that max_i t_i equals ||E_{q*} g||_sigma.

### D4: CONFIRMED

I re-derived the logic. Step by step:
- W==D implies (EW): trivially.
- (EW) for all s implies W==D:
  - On {L_S > 0} every maximiser has H >= H^dec.
  - D_+ log D_S >= -H^dec holds correctly, via the product lower-bound function s' -> e(theta_1;s') B_p M(yhat*) K_{1/s'}(yhat*)^{1-s'}. This is <= D_S(s'), with equality at s and derivative -(H(theta_1,s) + H(z*)). It is not obtained via gamma(Theta) <= D_S, which would be false.
  - Dini monotonicity with L_S(1) = 0 then closes the argument.
- (E) a.e. implies W==D: phi = H^dec - H* is bounded and Borel. H* is lsc, and H^dec is an inf over widths of lsc functions. D^+L <= phi holds everywhere, so the Vitali-Caratheodory argument applies.
- W==D implies (E) a.e. with equality:
  - log W_S is Lipschitz (bounded Dini derivatives).
  - At points of differentiability all maximisers of any finite width share one entropy (GLOBAL R2).
  - The decoupled network has gamma >= pi B_p Gamma^S = D_S = W_S, so it is a maximiser, and its entropy is H(theta_1,s) + H(z*) for every optimal pair.
The claim is pure logic, valid for all p, n, b and S. At p=3 it is consistent: W = D trivially, and H^dec is the counterexamples' own entropy.

### D4': CONFIRMED

I re-derived the chain h <= E g = (p/(p-1)) sum Re beta_j <= (p/(p-1)) sum|beta_ij| <= (p/(p-1)) B_p sum a = W(1) sum a.
- (1) g == h forces phi_x = h(1_0 - 1/p), hence Phi_l == 2h/p.
- (2) Phase alignment.
- (3) Budget equality: ||.||_nu = ||.||_2 for nu >= 3 forces one frequency per factor; with beta_i != 0 the frequency is common; there is no DC and the factors are balanced.
- (4) D5 (aligned and IF) gives collision <= 1/nu per frequency, and the chain rule gives H >= log F + log nu.
- H^dec_F(1) = log nu + log F, by uniqueness of the Gamma_1 maximiser and min cm = Re beta_1 forcing Phi constant.
- The endpoint Danskin argument is fine, since f is defined for s near 1. Together with gamma_IF <= D_F, it gives D^+L_F(1) = 0.

Independent optimisation (verify/v7_s1_maximisers.out) from Gaussian starts at (5,2) and (3,2), b = nu: every run reaching 1.000000 has beta_j/(2h/p) = 1, max|beta_T|/|beta| <= 3e-7, and H(p) >= log(nu F) (1.92-2.01 vs 1.79; 1.33-1.50 vs 1.10).

Scope: this is a statement at the single endpoint s = 1 (b = nu) for all p and n. It yields only L_F = o(s-1), with no a.e. content for (E).

### D5: CONFIRMED

I re-derived every step:
- ||M||_F^2 >= nu^2 sum|beta_l|^2 (drop the cross terms; ||m_l||^2 = nu);
- entries M_kk = L and M_{k'k} = tilde-beta_I;
- phase bound |tilde-beta_I - e^{-i phi} beta_I| <= sqrt(2L(L - |A|)) by Cauchy-Schwarz with sum |beta_l| cos psi_l = |A|;
- Schur-Horn majorisation via G = V*V;
- equality case: orthogonal chain vectors force N <= nu, then Cauchy-Schwarz forces nu equal atoms;
- Lemma 7 for aligned atoms via K_r >= K_2 >= nu, r <= 1.

Independent check (verify/v2_d5_d6.out, v2b_d5_precision.out):
- 3000 random systems (n = 1..5, general phases): max(lhs - rhs) = 2e-16, and majorisation never fails.
- 150 adversarial Nelder-Mead runs found double-precision 'violations' up to 3.7e-9. Every one is cancellation in sqrt(2(1-|A|/L)): at 60 digits all are negative (-1e-8 to -1e-11).

### D6: CONFIRMED

The proof is correct under its stated hypotheses.
- (a): |arg(Phi - c)| <= pi/p gives Re (Phi-c)^2 >= cos(2pi/p)|Phi-c|^2. Also E(Phi - E Phi)^2 = 0, since the monomials z^T z^T' with T, T' non-empty have mean 0 for odd p. Alignment is NOT needed for (a).
- (b): D5 (aligned) plus Bernoulli, for s >= 1.
- (c): uses 0 <= Re A - c <= |A| - c. This is valid for any common phase, not only real atoms (the agent's check used only real atoms).

Independent check on ACTUAL single-frequency networks with non-zero common atom phase (field from the table; verify/v2_d5_d6.out, v2c_d6_smallphase.out):
- (a), (b) and (c) are never violated;
- adversarial maximisation of c/(nu^{1-s}Q^s) under 2(s-1)tau_p <= nu gives 0.85, 0.61, 0.993, 0.38, 0.95 at (5,2,1.5), (5,3,1.85), (7,2,1.5), (7,3,2), (5,2,1.2).

Generality: explicitly NOT general (aligned, class C, single feature; p >= 7, or p = 5 with s - 1 <= 0.2236 nu; vacuous at p = 3). This violates the hard rule, as the agent itself says, and must not be counted toward a general proof.

### D7: CONFIRMED

Paper Lemma 2 (h <= sum Re beta_j kappa_j(delta) for all delta) gives h <= M(yhat) sum|beta_j|. The r=1 single-neuron budget is a property of each neuron, valid whatever b the network is evaluated at, and gives sum|beta_j| <= B_p sum a. Then D1(i) and D1(iv) give the bound. The nu_pi algebra is correct for s > 1.

Independent check (verify/v1_d7_lemma2.out):
- Lemma 2, the budget and the mean-gap identity hold on 3000 networks: arbitrary inputs, DC, perturbed IF, unequal sizes, p in {3,5,7,11}, n in {2,3};
- the D7 ratio over 989 positive-margin networks x 3 values of s is at most 0.956;
- on the exact p=3 certificates (n = 11, 13; gamma/gamma_IF = 1.0544, 1.1092), gamma/bound = 0.139 and 0.147.
It is consistent with the counterexamples and is not a value claim. It holds for all p, n, b.

### D8: CONFIRMED

The REFUTATION is valid.

I independently recomputed (verify/v4_d8_d9.out: own exact-rational margin and cost from the 12-digit rationalised weights, 60-digit DFT):
- gamma/gamma_IF = 0.790214100482;
- H(w_{3/2}) = 1.559103638;
- R(0) = 0.7817580;
- argmin of R at t = 0.0557 (value 0.7303);
- t0 = 0.2693156166;
- eta^dec = 1.704571115 > H.

The decoupled side is sound at (5,2,b=2):
- GLOBAL R9(i) at l = +-1 gives min cm <= Re beta_1 - cos(pi/5) M. Theorem A (s <= 3/2) gives Re beta_1 <= 3^{1-s}Q^s + M/2. Since cos(pi/5) > 1/2 strictly, an optimal single feature has M = 0.
- Then min cm = Re E Phi forces Phi to be a real constant, i.e. IF.
- Lemma 7 equality and KM2 then give the 3-neuron equal-cost block, so H = log 3.
- H(z(t)) is increasing on [0,1]; t > 1 is the mirror case.

Overreach: 'a Forster-type bound with decoupled extremals can hold only at the optimal level gamma = W' is not shown. Only one level (0.79 gamma_IF) is refuted; levels in (0.79, 1) are untested.

### D9: CONFIRMED

The REFUTATION is valid. I reproduced all three violators independently (exact margins, 60-digit profiles):
- 0.790214 vs D_{yhat} = 0.777321 (excess 1.66%);
- 0.785339 vs 0.771993;
- 0.776161 vs 0.763442.
D_C <= W_C always holds (D10(a)), so this refutes W_C = D_C for pinned profile classes.

Errors in the side remarks:
(1) 'At (5,2), W_F = D_F is the p52 theorem' is FALSE as a citation. p52 proves the COUNT (every one-feature network has gamma <= 0.995 gamma_IF). It does not prove gamma* = gamma_IF. W_F = D_F at (5,2) is still the open value statement.
(2) 'Decoupled networks are extremal only near the optimum' is unproved: near-optimal extremality is exactly the open conjecture.

### D10: MINOR_GAP

The logic is correct.
- (a) Decoupled networks with profile yhat lie in A_C (D1v) and achieve >= pi B_p M K^{1-s}. At s=1, gamma_1 <= B_p M(yhat_theta) is attained by IF networks.
- (b) '<=': a maximiser with feature set S' not in K, where S' strictly contains some K, lies in A_{C(S',eps)} with eps = min yhat_k > 0.
- (b) '=>': attainment follows from width reduction, compactness, and h >= eta, which keeps sum|beta| >= h/2 > 0 so yhat is continuous. tau > 0 because C(S,eps) is compact and contains no programme maximiser.
- Sufficiency is immediate.
(b) is close to a tautological restatement of the upper half. 'Must be proved through this slack form' is a heuristic.

Numerics are wrong in places. Exact vertex enumeration (verify/v6_d10_tau.out) and an explicit floor witness (v6b) show that floor_tolerance.out overestimates tau:
- p=13, s=2, S = {2,3,1}: true values at eps = 1e-6, 1e-4, 1e-3 are 1.41e-3, 1.41e-2, 4.38e-2, not 4.2e-3, 4.2e-2, 0.13;
- p=11, s=1.5, S = {1,4,2}: tau saturates at 4.23e-3 from eps = 1e-3 on, not 1.0e-2.
The scaling tau ~ c eps^{1/s} survives, with c about 1.4 at p=13, s=2.

Note: optimal supports come as dilation orbits (at p=11 my vertex enumeration returned support {1,3} where the agent's code returned {1,4}), so K must be taken as the whole orbit. The statement allows this.
