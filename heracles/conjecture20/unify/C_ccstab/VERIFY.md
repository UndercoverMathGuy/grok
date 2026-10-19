# Referee verdicts on C_ccstab/REPORT.md (saved by the orchestrator)

## Overall
The algebraic core is sound and general:
- S1 and S2 hold for every network (re-derived, and checked independently, including the p = 3, n = 11 counterexample).
- P and H2 are correct.
- Theorem L's proof is correct once h > 0 is assumed. Without it the literal conclusion fails for negative-margin members of N_S (counterexample v9).

The main problem is the stability story built on L.
- The interference-free manifold through the aligned IF equality cases contains non-aligned IF networks, for every n. Moving one Riesz node by eps gives atom phase spread ~eps with interference identically 0.
- Along such curves, d_int is identically 0 and the RD/RD-IF slack is quadratic in the distance (slope 2.000). This holds at:
  - the (5,2,2) decoupled maximiser of D_F;
  - the n = 2 single-feature RD equality case;
  - (7,3), (5,4) and (3,4).
- Hence O's 'RD is conically stable at its interference-free equality cases' is false.
- O-num's 'first order in every direction' reflects sampling: no non-aligned IF-tangent direction was sampled.
- L2's 'near IF never beat gamma_IF whatever the phases' is unproved. With interference eta << eps^2 added, Theorem L has X < 0 and L2 fails, at arbitrarily small distance from the equality case (v4).

Other points:
- H1 omits the s = 1 endpoint.
- H3-num's 'never at s = 1' partly reflects tie-breaking.
- R is valid against KM2 unions, and survives my extension to IF-manifold copies, but remains incomplete for general decoupled networks.

Scripts are in /home/user/grok/heracles/conjecture20/unify/C_ccstab/verify/: vlib.py, v1_identities.py, v2_theoremL.py, v3_ifmanifold.py, v4_near_if_silence.py, v5_frontier.py (and its output v5_frontier.out), v6_ties.py, v7_p3.py, v8_dint_lower.py, v9_negmargin.py, v10_frontier_ifmanifold.py.

## Generality audit
Fully general, for every odd p (including 3), every n >= 2 and every b in [2, nu], with no case split:
- S1 and S2(a), S2(b): any network.
- H1 (apart from the s = 1 omission), H2 (given an optimal pair), P.
- H3's logic. Its strict instance exists only at parameters with programme transitions, numerically p = 11, 13, 17 at n = 2.

Restricted to class C by explicit hypothesis, with no p, n or b limits:
- S2(c), L, L1, L2, O(ii).
- Their constants (2^{n+3}, (2^n-1)K', the n(n+1)/2 intervals) are n-dependent but valid for all n.
- No step uses p >= 5. This is consistent with the p = 3 winners: v7 shows they are non-circular (|E f^2|/E|f|^2 = 0.28) and non-class-C, and their frequency split has h < 0.

Hidden restrictions:
- (1) L, and L1/L2 through it, need h > 0, which N_S does not impose.
- (2) O's 'Consequence' silently drops the class-C hypothesis, so nothing is proved for mixed-mode or DC perturbations of IF networks.
- (3) The 'near IF' readings of L2 and O are really cone statements, valid only in directions where interference is first order. They are not neighbourhood statements, so no uniform local theorem follows.
- (4) H3-num, G and the R frontier are single-parameter or n = 2 numerics, legitimate as evidence or counterexamples only.
- (5) 'H^dec_S not monotone' fails at p = 5, 7, n = 2.

## Salvage
- **L:** add the hypothesis h > 0. Then 1 - X > 0 automatically and the proof is complete; the conclusion 'log(D_S/gamma) >= X' then holds for any sign of X.
- **O:** replace 'conically stable' with the true statement, for class-C theta with h > 0: log(gamma_IF/gamma) >= [sum kappa E/O - sum_S kappa(|A| - a^-)]/sum kappa a, where the detection term is >= c sqrt(E) and the payment term is O(L - |A| + E^I). That is cone-type stability transverse to the IF manifold only.
- **Neighbourhood theorem near IF (L2's intended consequence):** Theorem L must be combined with Lemma 7 on the IF manifold. Compare theta with its nearest IF point (which pays its own cancellation deficit 1 - G >= 0 exactly) rather than with the aligned block, or use a payment bound that charges cancellation against the IF deficit instead of against detection.
- **L1:** already gives a genuine neighbourhood result inside ALIGNED class C, since O -> 0 there; state it as such.
- **H1:** add the s = 1 endpoint case.
- **H3-num:** report ties.
- **R:** either define 'decoupled' as copies of the optimal single feature, or extend the search to arbitrary single features. My IF-manifold extension (v10) and a coarse DE search already support it.
- **G:** correct the two class-C misstatements.

## Verdicts

### S1: CONFIRMED

Re-derived every line.
- E_delta over the p-1 nonzero offsets of c_l(delta) is p/(p-1) = Gamma_1, so E_{x,delta} g = Gamma_1 sum Re beta_l.
- Pairing +-delta gives kappa_l, so h + sbar_sym(delta) = sum Re beta_l kappa_l(delta). Substituting gives d_int N = min sbar_sym and d_prog N = E_delta sbar - min sbar_sym.
- The slack identity Phi_j = (2/p)(h - sum_delta sigma rho^{j delta}) follows from sum_{delta in Z_p} rho^{j delta} = 0. Parseval on Z_p with tau(0) = 0, halved over F, gives the energy identity.

Independent code (verify/v1_identities.py) on 90 networks:
- p = 3 to 13, n = 2 to 4, general, class C, and with strong DC.
- Errors: identity 1.8e-15, d_int 1.4e-14, d_prog 7.5e-15, energy 2.2e-15.
- The identities also hold on the p = 3, n = 11 counterexample (verify/v7_p3.py), as they must, since they are general.

### S2: CONFIRMED

All three parts are correct.
- (a) Variance <= second moment, sigma^2 <= sigma_max sigma, and drop -(sum tau)^2. This gives (2/p) sum_delta E_x sigma^2 <= 2(1-1/p) sigma_max N (d_prog + d_int).
- (b) E_delta c_l conj(c_l') = Gamma_1(1 + [l = l']) and E_delta c_l c_l' = Gamma_1, since l + l' is in [2, p-1]. Then |w|^2 + Re w^2 = 2 (Re w)^2.
- (c) In class C only neurons at frequency l feed Phi_l, and Phi_l^{(i)} = conj(beta_il) prod_t (1 + e_it z_t). So |g| <= 2^{n+1} B_p sum a by the r = 1 budget, and sigma <= osc g <= 2 max|g|.

Independent checks (v1): S2(a) min slack 0.31 N^2, S2(b) error 1.3e-15, S2(c) min slack 6.8 N. On the p = 3 counterexample, (a) and (b) hold and the single-offset diagonal formula fails (5484 vs 6635), consistent with the stated non-class-C compensation.

Cosmetic: the check text says smax/N lies in [0.6, 4.6], but c5 lists viol_p5:3 at 0.001.

### L: MINOR_GAP

The proof is correct for positive-margin networks:
- margin step (+-delta_0 pairing);
- the mean-zero lemma (it is even weaker than Bhatia-Davis);
- the class-C variance lemma (characters 2l 1_T are distinct and never antipodal for p odd, including p = 3);
- W6(c) with the twist bound sqrt(2L(L-|A|)), V <= V' (prefix entries are over-counted harmlessly), Q >= L^r, Lemma-2 programme bound, single-neuron budget, and pi >= nu^{1-s}.
No step uses p >= 5.

Gap: the last step multiplies by (1-X), which needs 1-X >= 0. That holds automatically when h > 0, but N_S (E_reduce) also contains networks with h <= 0. The literal conclusion is then false: v9_negmargin.py at (5,2,2) uses a small IF block plus a ghost pair on feature 2 (beta_2 = 0, E_2 > 0). It gets X = 293.8 >= 0 and gamma - (1-X) gamma_IF = +286 gamma_IF. Fix: assume h > 0. RD is trivial otherwise.

Independent ledger (v2_theoremL.py, 440 networks, 198 with h > 0): p = 3 to 11, n = 2 to 5, non-aligned atoms (independent output phases), ghost features. Every step holds to 5e-14.

The 'local regime' label oversells the theorem: it is silent (X < 0) in every neighbourhood of the aligned IF equality case (see L2 and O).

### L1: CONFIRMED

The derivation is correct.
- Alignment gives L = |A| and tau = 0, so V' = 2E^I/(nu|A|^2) with E^I <= E.
- Bernoulli: (1+V)^{-(s-1)} >= 1-(s-1)V. Then O <= nu Abar/(2(s-1)) is equivalent to the sufficient condition.
- In aligned class C, l not in S forces all atoms of l to be 0, so E_l = 0 and the sums in Abar are consistent.
- O <= max sigma(.,+-delta_0) because sigma >= 0.
- At b = 2 the factor is exactly (n+1)/(n-1), and larger for b > 2.

Unlike L2, the condition holds in a whole aligned-class-C neighbourhood of IF, since O -> 0.

The reported numbers are slightly misstated. Two of the 24 stored networks (viol_p5:3, viol_p7:3) are essentially class C (energy defect 5e-7 and 2e-6), not '2-8% outside the plane'. The '1-62%' figures are absolute split gamma/gamma_IF values, not retained fractions (viol_p7:1 keeps 72%).

### L2: SERIOUS_GAP

The inequality itself is correct: (|beta_I|+tau)^2 <= 2|beta_I|^2 + 2tau^2, #I = n nu/2, and Bernoulli.

The stated consequence does not follow: 'near IF the left side is second order, the right side first order, so class-C networks near IF never beat gamma_IF whatever the phases'. The two sides are controlled by DIFFERENT small parameters:
- L - |A| is second order in the atom phase spread;
- sum kappa E / O is first order in the interference amplitude, which can vanish independently.

The IF manifold through the aligned nu-atom block contains non-aligned IF configurations, for every n. Take nodes a_k, set xi_kt = a_k, and let z be the null vector of [a_k^j], j = 1..n. Moving one node by eps gives:
- phase spread 1.5 eps;
- E identically 0;
- (L-|A|)/L ~ eps^2/4 (v3_ifmanifold.py).

Adding interference eta = 0.05 eps^2 (v4_near_if_silence.py) gives, at parameter distance 5e-3 to 0.28 from the equality case:
- (5,2,2): X in [-2.3e-3, -2.1e-6], L2 lhs about 50 x det;
- (7,3,2): X in [-1.8e-2, -1.7e-5], L2 lhs about 270 x det.

So L and L2 are silent in every neighbourhood of the aligned IF equality case. Those networks do satisfy gamma < gamma_IF, but L2 does not prove it.

### O: SERIOUS_GAP

Parts (i)-(iii) are correct.
- (i) The telescoping chain (AM-GM, DC via Parseval, l_nu <= l_2, Hoelder, phase) is exact. Each gap is nonnegative and vanishes to second order at balanced, DC-free, single-frequency neurons with real positive atom ('aligned' must mean beta_ij > 0, because sigma_i uses Re beta_il unrotated).
- (ii) The chain is right: sbar_sym >= (m_delta + m_{-delta})/2, then the mean-zero and variance lemmas, then O_delta <= 4 sum_T |beta_{l,T}| and Cauchy-Schwarz, with kappa_l(delta) >= kappa_min. It is class C only. v8_dint_lower.py on 175 networks: both inequalities hold to 2e-14.
- (iii) restates W6(c).

The headline consequence 'RD is conically stable at its interference-free equality cases' is FALSE. What is proved is slack >= c sqrt(E) - C(dist^2), and sqrt(E) can be much smaller than dist.

Explicit curves (v3_ifmanifold.py) start at:
- the decoupled b = 2 maximiser of D_F at (5,2,2);
- the single-feature RD equality case at n = 2;
- and at (7,3), (5,4), (3,4).
Along them:
- the networks stay exactly IF with positive margin, so d_int = 0;
- the parameter distance grows linearly in eps;
- log(gamma_IF/gamma) has log-log slope 2.000 against distance (e.g. 1.2e-5 at distance 1.8e-2).
The RD deficit is therefore quadratic in the distance, not conical.

The 'consequence' also drops the class-C qualifier: Theorem L says nothing about mixed-mode or DC perturbations.

### O-num: SERIOUS_GAP

The reported slopes reproduce for the sampled directions. They do not support 'in every direction'.

None of the four direction kinds is tangent to the IF manifold with atom phase spread:
- 'phase-only' and 'classC-aligned' set the output phase to sum(phis), so atoms stay real positive (aligned);
- 'general' directions are random, so they generically carry first-order interference.

Along non-aligned IF-tangent curves (v3_ifmanifold.py), at the same base points (5,2,2) and S = F:
- d_int is identically 0 (no slope 1);
- the RD-IF slack has slope 2.000 against both eps and parameter distance;
- E_m sigma and the Renyi drop are second order.

This curve class is measure-zero for random sampling but structurally essential: it is exactly where detection vanishes. The data therefore cannot support the conical-stability inference in O.

### H1: MINOR_GAP

The core is correct.
- Gamma = (min over the polyhedron of sum v^r)^{-s}. The minimum is at a vertex: sum v^r is concave and coordinatewise increasing along the recession cone v >= 0 (monotonicity, rather than 'tends to infinity', is the right justification; it also covers r = 1).
- phi_v(s) = -s log sum v^{1/s} = -log sum v - (s-1) H_{1/s}(vhat).
- phi_v is concave, as minus the perspective of the convex log-sum-exp.
- phi_v' = -H(esc_{1/s} vhat), escort entropy is non-decreasing in s, and envelope switches drop H weakly.
- S_theta' = H(theta,s) - H^dec_S(s) at differentiability points.

Gaps:
- (a) The hardest-exponent characterisation ('an entropy crossing: on a kink, or at b = 2') omits the endpoint s = 1. Any theta attaining the convex-end bound gamma(theta;1) = B_p Gamma_1 has S(1) = 0. If its entropy exceeds H^dec, s = 1 is its unique hardest exponent: e.g. an IF union with 2nu atoms per feature has S(s) = S_0(s) + (s-1) log 2.
- (b) 'H^dec_S is not monotone' is not general. It is constant on [1, 1.5] at p = 5, 7, n = 2, and the price part of H^dec is not controlled for n >= 3.

### H2: CONFIRMED

Re-derived.
- gamma^dec(sigma) = e(theta_1;sigma) B_p G_sigma(y*) <= pi(sigma) B_p Gamma^{(S)}_sigma = D_S(sigma), with equality at s_1.
- (log gamma^dec)' = -H^pair: W1 for theta_1 and log G_sigma(y) = log M(yhat) - (sigma-1) H_{1/sigma}(yhat).
- log gamma(theta;s) - log gamma(theta;s_1) = integral of H(theta,.).
- Subtracting gives the transport inequality, valid for all p, n, S.

The statement presupposes that an s_1-optimal pair exists (pi attained). It is conditional on that, and E_reduce's bounded-width compactness makes it plausible.

The c5 check (n = 2, quadrature error -1.2e-8) is consistent.

### H3: CONFIRMED

This is logically immediate.
- An s_0-optimal decoupled network attains D(s_0), so S(s_0) = 0.
- 'Not optimal at nu/2' means S(nu/2) > 0 by definition.
- Scaling gamma(theta;1) by e^eps with 0 < eps < S(nu/2) satisfies RD at nu/2 and violates it at s_0.

Scope:
- The no-go is about inference rules on abstract data (gamma_1, m). Whether the perturbed data are realisable is not addressed, so structural arguments are not excluded.
- The strict case exists only where programme transitions occur (numerically p = 11, 13, 17 at n = 2). At p = 5, 7, n = 2 the decoupled optimum is s-independent and S is identically 0.

The c2_hardest.out values 0.0853, 0.1182, 0.1769 reproduce from the file.

### H3-num: MINOR_GAP

The numbers match c2_hardest.out and c2b_vertices.out: 14/15/2/0, 53/31/3/0 and 102/283/6/0, the s = 1.34-1.35 minima at p = 11, and the +0.18 crossing for the p = 11 violators.

Two caveats:
- (a) c2b breaks ties by taking the LAST zero of S. The first-branch optimal vertex (S identically 0 on [1, s_c)) is therefore never counted 'at s = 1', although s = 1 is among its minimisers. v6_ties.py finds exactly one such vertex at p = 11 and at p = 13, and no vertex with a strict unique minimum at s = 1.
- (b) 'Never hardest at s = 1' holds only for the census family (IF unions with exactly nu atoms per block on vertices, plus small perturbations). Networks attaining the convex-end bound with excess entropy are hardest at s = 1 (see H1).

All of this is n = 2 only (pi = 3^{1-s} known exactly).

### P: CONFIRMED

The proof is correct.
- RD_S for all s is equivalent to log gamma(theta;1) - (s-1) H_{1/s}(m) <= log D_S(s) for all s, by W1. N_S is s-free, and h <= 0 is trivial.
- Each bracket is Schur-concave: (s-1) >= 0 and Renyi entropies are Schur-concave and invariant under zero-padding. An infimum of Schur-concave functions is Schur-concave.
- The minimiser coincides with argmin S_theta. It is attained: log pi is a supremum of continuous functions, hence lower semicontinuous, and the infimum is over the compact interval [1, nu/2].
- The far regime is immediate from H >= 0.

The claim is general in p, n, b and S.

### R: MINOR_GAP

The numbers reproduce independently (v5_frontier.py):
- theta_v has H_{2/3}(m) = 1.537697 and gamma_1/B_p = 0.869933, with exact rational margin per split/certify_viol.out;
- the KM2 frontier gives M = 0.818485 at yhat = (0.886, 0.114);
- M(yhat) = 0.691 + 1.118 min(yhat) is exact at p = 5.
So theta_v beats every two-block KM2 union at the same or lower Renyi concentration. That refutes the slice statement for 'decoupled' meaning copies of the optimal single feature.

The statement's formula is confusing ('1 - D' with D a deficit, inside a sup). For the broader E_SYNTHESIS notion (copies of ANY single feature, product masses) the refutation is incomplete, as the caveat admits.

My extensions found nothing better than KM2:
- v10: exact optimisation over copies of NON-ALIGNED IF single features (the IF manifold) returns the KM2 value 0.818484;
- a weak DE over general class-C copies or free components with k <= 3 found at most 0.43 (inconclusive).

'Every slice certificate is now refuted' is an overreach.

### G: MINOR_GAP

Most figures reproduce from c5, c7 and c9: d_prog, E_m sigma, d_int ranges, the Legendre L1 ratio formula 2(n-1) sqrt(p)/(n+1) (1.491 at (5,2)), (L) failing on all Legendre cases, gamma/gamma_IF = 0.13-0.54, and detection 0.53-1.23 vs payment 0.42-0.99.

Misstatements:
- (a) 'None is class C: 1.6-7.9%' is false for viol_p5:3 and viol_p7:3. Their off-plane energy is 5.3e-7 and 1.8e-6, and they are near-IF (gamma/gamma_IF 0.9997 and 0.975).
- (a) '1-62%' is a range of ABSOLUTE split gamma/gamma_IF values, not retained fractions (viol_p7:1 retains 72%), and it excludes the two near-IF entries.
- (c) Detection is also too weak at (7,3) (0.73 vs 0.75) and (7,4), not only at p = 5.

These are numerical descriptions only and do not affect any proof.
