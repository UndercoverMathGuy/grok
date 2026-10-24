# split: report (from the agent's structured output; the harness blocks subagents from writing report files)
## Headline
Proved backbone:
- The frequency split costs nothing extra and keeps every atom (so every feature, cheap feature and master-inequality term except detection and unused budget). It is the average over the sign-flip group Σ, and Σ is the largest such group.
- Every mixed mode is classified by channel and price.
- (C) ⇐ (C_C) + (MM_opt). At (p,n) = (5,2) and (7,2), b ∈ [2,3], this reduces the count exactly to "every global maximiser is interference-free".

Refuted: no all-network inequality can replace (MM). A certified mixed network at (5,2,2) violates canonical-split decoupling by 3.3%. Its split satisfies it, as every class-C network does there.

## Principle
PRINCIPLE (split). The feature count is a property of the Σ-invariant (frequency-diagonal) part θ_split of a network.
- Atoms, features, cheapness, Q_l, a_l, G and the mean gaps are carried by clean blocks. The split keeps them all and never raises the cost.
- Everything else is Σ-odd ("mixed modes"). Mixed modes can affect the count only by HIDING the class-C interference of a non-optimal diagonal core.

Hiding has three sources, each with an exact price:
- DC and dead content: paid, = P − P_live, sits inside U.
- Shared live frequencies: free at b = 2, paid at b > 2 by superadditivity of at least (b/2−1)·M^{b/2−1}·(E−M).

How hiding works:
- It reaches a feature l's sign-flipped character only through 2|F|−1 explicit channels: partner |3l|; foreign |2l±j|; DC only via output |2l|.
- A free channel needs an F2 relation 2l ≡ ±j ± q between two live frequencies of one neuron.
- It acts only through the span of the class-C interference characters.
- It works offset by offset. On a single binding pair it can be total. On the full set of offsets the own mode is isolated in the δ-Fourier domain (this is the slack identity), so at most half can be hidden.
- At p = 3 every character of Z_3^n is a relation of length ≤ 2, so hiding is unconstrained. This is exactly the counterexample mechanism.

UNIFIES:
- From the Note: the channels lemma, the Lorentz/Aczél lemma, (F1)/(F2), Prop. hide and the Chebotarev dichotomy.
- From the companion: the circularity lemma and its Question (b).
- The p = 3 mechanism (DC = partner = relation), and "spurious features need all-slot content".
- It splits the open problems into Σ-even ones (O1 relations, O3 single-feature constants: class-C questions) and Σ-odd ones (O2 compensation, the mixed half of O4: the (MM) question).

CASES REMOVED:
- Separate DC/dead cases: priced exactly in U.
- All mixed modes off the span of the class-C interference. For example, all non-diagonal mixing in the relations case.
- Feature creation by mixing.
- Interference-free cores: mixing never helps them.
- Single-output networks with no "2l-pair" in any slot: they are circular at every odd p, so companion Thm 14 extends to them at p ≥ 5.
- b = ν: every maximiser is in class C.

LIMIT: (MM) is a value-type statement. It is false at p = 3 and cannot be relaxed to all networks (certified).

## Results

### R1 [PROVED]

Frequency split lemma (S2, with equality cases).
(a) Let b>=2 and let omega be a neuron. Then sum_{k in F} ||omega^(k)||^b <= ||omega||^b. Equality holds iff P_0 omega = 0 and (b = 2 or at most one piece is non-zero).
(b) Let b>2, e_k = ||omega^(k)||^2, e_0 = ||P_0 omega||^2, E = sum_F e_k, M = max e_k. Then slack >= (b/2) E^{b/2-1} e_0 + (b/2-1) M^{b/2-1} (E-M). At b=2 the slack equals e_0 exactly. Relative to the live split, P - P_live = DC energy + dead energy.
(c) theta_split is in class C and carries the same atoms (beta_ij, xi_ijt) for every i, j. Hence it has the same beta_j, beta_{j,T}, Q_j, a_j, cheap set, coverage gap G and E_x g(.,delta). Costs satisfy P_live <= P_split <= P.
PROOF.
- Frequency planes are orthogonal in every factor, so ||omega||^2 = sum_{k in {0} u F} e_k.
- t -> t^{b/2} is superadditive (strictly for b>2 when two terms are positive) and increasing. Equality forces e_0 = 0 and, for b>2, a single positive e_k.
- For (b), use convexity three times:
  - (E+e_0)^q - E^q >= q E^{q-1} e_0;
  - E^q >= M^q + q M^{q-1}(E-M);
  - sum e_k^q <= M^q + M^{q-1}(E-M).
- For (c), atoms and xi = conj(u_hat(j))/u_hat(j) use only u_hat(+-j) and w_hat(j), which P_j preserves (paper Lemma 6).

*Where:* Proof in this statement. Check: check_split.py -> check_split.out

*Check:* Random networks at (p,n) = (3,2), (5,2), (5,3), (7,2), (7,3), (11,2), (5,4) and b in {2, 2.5, 3, nu}:
- cost(split) - cost <= 0 in every case;
- |slack_{b=2} - DC energy| <= 3.5e-14;
- beta, beta_T and atoms are preserved to 1e-15;
- equality cases: 0.0 (b=2, no DC); 3.6e-15 (one frequency, b=3); strict otherwise (45.7, 15.4).

### R2 [PROVED]

Sign-flip averaging (S3), extended.
Let Sigma' = {+-1}^{F x [n]} x {+-1}^{[n]} act by
- u_t -> d_t P_0 u_t + sum_k s_{k,t} P_k u_t,
- w -> P_0 w + sum_k (prod_t s_{k,t}) P_k w,
either globally or independently per neuron (Sigma'^m). Then:
(1) Every element preserves the cost and every beta_{j,T}.
(2) A block with kappa_0 = k in F is multiplied by prod_{t: kappa_t = 0} d_t s_{k,t} * prod_{t: kappa_t not in {0,k}} s_{k,t} s_{kappa_t,t}. This sign is trivial iff kappa = (k,...,k).
(3) E_sigma T(sigma omega) = T(omega_split), modulo output-DC blocks, which never enter gaps.
(4) h(theta_split) >= E_sigma h(sigma theta), for both the global and the per-neuron group.
(5) Averaging only over the frequencies in K gives the cost-non-increasing partial split {omega^(k): k in K} u {(I - sum_K P_k) omega}.
PROOF.
- The maps are orthogonal on every factor.
- beta_{j,T} contains each u_hat_t(+-j) once and conj(w_hat(j)) once, so the signs multiply to (prod_t s_{j,t})^2 = 1.
- An unpaired independent sign averages to 0.
- h is a minimum of linear functionals of the table, hence concave, and is blind to tables that are constant in c.

*Where:* Proof in this statement. Check: check_split.py -> check_split.out

*Check:* - Sigma-averaged table = split + DC block, to 2.8e-15.
- With DC flips, average minus split has gaps of 1.3e-14.
- Per-neuron averages agree to 1.8e-15.
- E_sigma h(sigma theta) <= h(split) in every case.
- Cost and own data are invariant to 1.7e-13 and 9e-16.

### R3 [PROVED]

Sigma is maximal, and the torus average is the interference-free (IF) projection.
(a) Consider maps acting identically on all neurons by orthogonal maps that preserve every frequency plane in each slot. Those fixing beta_j and beta_{j,T} of every network are exactly Sigma': plane rotations by 0 or pi with output compensation, plus DC signs.
(b) Averaging over the full rotation torus (with output compensation) keeps exactly the clean modes (j..j;-j). It kills every sign-flipped and every mixed mode.
So enlarging Sigma by per-frequency translations or rotations cannot keep the class-C data. The larger average is the paper's Lemma-2 IF projection, which the split cost cannot attain: paper Lemma 7 charges a factor nu^{1-r}.
PROOF.
- Identify the plane E_k with C via v -> v_hat(k).
- Reflections are antiholomorphic, but beta_k is holomorphic in each u_hat_t(k). So only rotations phi_{t,k} (output psi_k) are possible.
- beta_k fixed gives sum_t phi_{t,k} = psi_k.
- beta_{k,{t}} fixed gives -phi_{t,k} + sum_{t' != t} phi_{t',k} = psi_k.
- Subtracting the two gives 2 phi_{t,k} = 0 mod 2pi.
- In (b), a mode's phase exp(i(sum_t eps_t phi_{|q_t|,t} - sum_t phi_{k,t})) averages to 0 unless all q_t = k.

*Where:* Proof in this statement. Check: check_lemmas.py L1, L2 -> check_lemmas.out

*Check:* - A rotation by 0.7 keeps beta_2 (to 3e-17) and multiplies beta_{2,{1}} by e^{-1.4i}.
- A rotation by pi fixes all own data.
- A reflection changes beta_2 by 0.46.
- The exact average over a 9-point angle grid keeps clean modes (to 1.5e-14) and kills all others (to 2.5e-14).

### R4 [PROVED]

Mode-character bijection and the complete channel list.
(a) For each output frequency j, the coefficient of Phi_j at chi is phi_j(chi) = 2p^{-nu} sum_i w_hat_i(j) prod_t u_hat_it(chi_t - j). So each character carries one mode, q = chi - j*1.
- The mode is clean iff chi in {0, 2j}^n. Then phi_j(0) = conj(beta_j) and phi_j(2j 1_T) = conj(beta_{j,T}).
- So mixed modes have zero clean and zero sign-flipped content, and E_x g_R = 0.
(b) g_hat_delta(chi) = (1/2) sum_j [phi_j(chi) c_j(delta) + conj(phi_j(-chi) c_j(delta))]. At +-chi_{l,T} the mixed contributions are exactly 2|F|-1 channels. Each has on-frequency input j off T, and on T:
- partner: output l, input |3l|;
- direct foreign: output j != l, input |2l - j|;
- conjugate foreign: output j, input |2l + j|.
DC enters only through output j = |2l mod p|. A mode reaches a signed sum Lambda = sum eps 2l 1_T iff q = +-Lambda - j*1, with slot-t input |Lambda_t -+ j|.
(c) At p = 3, Z_3^n = {chi_{1,T} - chi_{1,S}}: every mixed mode is a relation of length <= 2. At p >= 5 the class-C characters and their short signed sums are reachable only through the inputs listed in (b).
PROOF.
- u(a) rho^{ja} = p^{-1} sum_c u_hat(c-j) rho^{ca}.
- Solve q = +-chi - j*1. |2l - j| = j forces l = j; |2l + j| = j is impossible for l, j in F.
- At p = 3, chi = 2*1_T + 1_S = 2*1_T - 2*1_S.

*Where:* Proof in this statement. Check: check_split.py, check_lemmas.py L3

*Check:* - phi_l from the formula vs from the table: 3e-16.
- Clean identifications: 2e-16.
- Mixed part of g_hat at +-2l 1_T vs the predicted channel sum: 5e-16 at (5,2), (7,2), (7,3), (11,2).

### R5 [PROVED]

Cost status of mixed modes, and the F2 condition.
- P - P_live >= 0. Equality holds iff there is no DC, no dead content and, for b>2, no shared live neuron. At b = 2 it equals DC energy + dead energy exactly.
- In the master inequality, U(theta) >= gamma_IF (P^s - P_live^s). So every mixed mode with a DC or dead factor is paid inside U (quadratically in the mode for one unpaid factor, by Note Lemma 'paid compensation is quadratic').
- A mixed mode is free only if all its factors are live and b = 2.
- A free mode reaches +-chi_{l,T} only through a neuron with live j, q and 2l = +-j +- q (F2), including (j,q) = (l,|3l|).
PROOF.
- Per neuron, ||omega||^b - sum_{k in L} ||omega^(k)||^b = [R1 slack] + sum_{k not in L} ||omega^(k)||^b.
- The single-neuron budget on each live piece gives Q_l <= C_b P_l^live.
- Hence U = nu^{1-s}((C_b P)^s - (sum Q)^s) Gamma_s >= nu^{1-s} C_b^s Gamma_s (P^s - P_live^s), and C_b^s = B_p.
- The F2 statement follows from R4(b).

*Where:* Proof in this statement. Check: check_split.out (b=2 identity); energies reported in out/e1*.out

*Check:* - |slack_b2 - DC energy| <= 3.5e-14 on random networks.
- All optimised networks report DC, dead and secondary energies consistent with the theory: at b>2 the optima have secondary energy <= 7e-6.

### R6 [PROVED]

Joint circularity.
(a) E[(Phi_l - E Phi_l)(Phi_k - E Phi_k)] = sum_{chi != 0} phi_l(chi) phi_k(-chi). A non-zero term needs, in every slot, signed inputs q_t (output-l neuron) and q'_t (output-k neuron) with q_t + q'_t = -(l+k), not all (-l,-k).
(b) Class-C joint fields are proper (all pseudo-covariances vanish) at every odd p, p = 3 included.
(c) A 2l-pair is a pair {a != a'} with a +- a' = +-2l, e.g. {0,|2l|}, {l,|3l|}, {j,|2l+-j|}; at p = 3 the pair {0,l} is one. A single-output network has a circular field if no input slot carries a 2l-pair. Conversely, one 2l-pair in a slot plus on-frequency input elsewhere gives a non-zero pseudo-variance for generic weights.
(d) For p >= 5, every 2l-pair-free single-output network satisfies min_x cm(Phi_l) <= Re beta_l - k_p sigma, with sigma^2 >= (1/2) sum_T |beta_{l,T}|^2. This extends companion Thm 14 from single-frequency+DC networks to all 2l-pair-free single-output networks: no gain over the IF value when k_p >= k*(n,s).
PROOF.
- Expand in characters and use E rho^{(chi+chi')x} = [chi + chi' = 0].
- In class C, q in {+-l} and q' in {+-k}, so q + q' = -(l+k) forces chi = 0.
- The companion's proof of (d) needs only the cone lemma, circularity, Parseval with R4(a), and paper Lemma 6.

*Where:* Proof in this statement. Check: check_lemmas.py L4 -> check_lemmas.out; check_r6.py -> check_r6.out

*Check:* Pseudo-variances:
- class C at p = 3, 5, 7, 11: <= 2e-17;
- p = 3 single-frequency + DC: 1.40;
- p = 5 single-frequency + DC: 2e-17;
- p = 5 with one {0,2} pair: 0.37.
In 9 cases at p = 3, 5, 7, 11, 'pair present iff pseudo-variance != 0' matched every time.

### R7 [PROVED]

Span transfer.
For each delta let V_delta = Z_p-span of the characters where g_hat_split(.,delta) != 0 (in class C: span{1_T : some beta_{l,T} != 0}). Let g_R^V be the part of the mixed gap supported on V_delta. Then
  min_x g_theta(.,delta) <= min_x (g_split + g_R^V)(.,delta).
So mixed modes off V_delta never raise a directional minimum.
Corollaries:
(a) If theta is IF, then h(theta) <= h(theta_split), with equality iff g_R = 0 on every binding offset. Mixing never improves an IF core.
(b) If all class-C interference sits on one slot-set T, only modes with a common signed off-frequency on all of T can help. In the relations case these are the diagonal modes only.
(c) Det_mu(theta) >= Det_mu(theta_split + R^V). Every test-based class-C argument with test support Lambda transfers verbatim to networks whose mixed modes vanish on Lambda minus {0}.
PROOF.
- Averaging over cosets of the annihilator of V_delta projects onto V_delta. This fixes g_split and maps g_R to g_R^V, because E g_R = 0.
- The minimum is at most the coset average.

*Where:* Proof in this statement. Check: check_lemmas.py L5, L6

*Check:* - Relations-case core at p = 7 with off-diagonal mixing: max over delta of (min g_theta - min g_split) = 8.9e-16.
- Quantitative form on a generic network: 0.0.
- Glued IF blocks plus DC at (5,2), (7,2), (7,3): h(theta) - h(split) <= -0.81 at most (always <= 0).

### R8 [PROVED]

Universal bound with live split costs.
  h(theta) <= B_p min_delta sum_j (P_j^live)^s kappa_j(delta) <= B_p Gamma_s P_live^s.
Consequence: every global maximiser has P_live/P >= pi_{n,b}(p)^r >= nu^{r-1}.
PROOF.
- Paper Lemma 2 bounds h by the clean amplitudes.
- |beta_j| <= Q_j^s <= (C_b P_j^live)^s = B_p (P_j^live)^s.
- The second inequality is the definition of Gamma_s.
- For the consequence, compare with gamma* >= pi B_p Gamma_s.

*Where:* Proof in this statement. Check: check_lemmas.py L7

*Check:* Over 90 random networks at (5,2), (7,2), (5,3) and b in {2, 2.5, nu}: max of h - RHS = -6.1 (always <= 0).

### R9 [PROVED]

The reduction.
Definitions:
- (C_C): class-C maximisers use |spt y*| features.
- (MM_opt): every global maximiser theta* has gamma(theta*_split) >= gamma(theta*).
- (IF_C): class-C maximisers are IF. This implies (C_C) by paper Thm 8(1).
Claims:
(a) (C_C) + (MM_opt) => (C), and gamma* = gamma*_C.
(b) Under (IF_C), the following are equivalent:
  - (MM_opt);
  - every global maximiser is IF;
  - every global maximiser has no DC, no dead content, P_split = P, and g_theta* = g_split on every binding offset.
  At b > 2 this means every global maximiser lies in class C. At b = 2 it means class C up to free gluing whose mixed modes vanish on the binding offsets.
(c) At b = nu every global maximiser lies in class C.
(d) At (p,n) in {(5,2),(7,2)}, b in [2,3], (IF_C) holds (Note Cor. 'Parseval closure' (c)). So there, (C) for all networks follows from 'every global maximiser is IF'.
PROOF.
(a) gamma(theta*_split) >= gamma* >= gamma*_C >= gamma(theta*_split), and the split keeps the features (R1c).
(b) Forward direction:
  - theta*_split is a class-C maximiser, hence IF;
  - R7a gives h(theta*) <= h(split); optimality forces P_split = P and equality;
  - equality forces g_R = 0 on binding offsets;
  - R1a excludes DC (and shared neurons at b>2); dead pieces would be deletable.
  Backward direction: R7a and P_split <= P.
(c) gamma* = B_p Gamma_1 is the universal bound. Equality forces single-neuron-budget equality: balanced, single-frequency and no DC, since b = nu > 2.

*Where:* Proof in this statement. Consistency checked in out/e1*.out, orbit_*.out

*Check:* - Every optimum found at p = 5, 7 is IF.
- max|g_R|/mean|g| ~ 1e-6, and the Sigma-orbit is constant at 1.
- At b > 2 the secondary energy is <= 7e-6, as R9(b) predicts.

### R10 [REFUTED]

Decoupling for ALL networks, with the canonical split and price nu^{1-s} (equal to the true price at n = 2, Note Thm 'price of a feature at n=2'), is false. The two inequalities are:
- (D_Q): h <= nu^{1-s} (C_b P / sum Q_l)^s min_delta sum Q_l^s kappa_l(delta);
- (D_split): h <= nu^{1-s} B_p (P/P_feat)^s min_delta sum_{features} (P_j^live)^s kappa_j(delta).
This is the hypothesis of Note Prop. 'decoupling suffices'.
COUNTEREXAMPLE (p,n,b) = (5,2,2), width 6, mixed:
- h/RHS(D_Q) = 1.033200 and h/RHS(D_split) = 1.029689, at gamma = 0.790214 gamma_IF.
- Its split: 0.494862 and 0.493181, gamma_split = 0.378482.
- Every class-C network at (5,2) satisfies (D_Q) (Note Thm 'decoupling when few features are bought cheaply').
So decoupling fails for non-maximisers ONLY through mixing.
Violation size by b:
- p = 5: 1.0332, 1.0147, 1.0047, 1.0002, none at b = 2, 2.25, 2.5, 2.75, 3;
- p = 7: 1.0710, 1.0084, 1.0015, none at b = 2, 2.5, 2.75, 3;
- p = 11, b = 2: 1.110 at gamma = 0.67.
The same search restricted to class C finds max ratio 1.00000 at p = 5 and 7. Every violator's split satisfies (D_Q), with ratios 0.01-0.75.

*Where:* certify_viol.py -> certify_viol.out (net viol_p5_n2_b2.npz); adv_dq.py, adv_dq_classC.py, analyse_viol.py, run_adv*.sh -> out/adv_dq_*.out, adv_dq_classC_p{5,7}_n2_b2.out

*Check:* - Margins computed in exact rational arithmetic, other quantities in 50-digit interval arithmetic.
- Weights are rounded to 12 digits and the ratio intervals are tight.
- The violation was found independently by numpy and torch code.

### R11 [NUMERICAL]

Mechanism: hiding at a single binding pair. All figures for the certified violator, in units of gamma_IF P^s.
- The allocation's Q^s shares are 0.84/0.16, so the binding set is Delta(theta) = {+-1}.
- The small feature is cheap: |beta_2|/a_2 = 1.17.
- The split has Det = 0.392; theta has Det = 0.0011. Mixing hides 99.7% of the detected slack, at payment Delta U = 0.0034 (shared mixing is free at b = 2; secondary energy 2%).
- theta stays below IF only through the coverage gap G = 0.232.
- Hence 'hidden <= (1-lambda) Det_split + C Delta U' fails for C = 1 with lambda > 0.012, and for C = 10 with lambda > 0.09.
- With mu uniform over all offsets, the hidden fraction drops to 25%. This is delta-Fourier separation (the slack identity): (1/p) sum_delta |g_hat_delta(chi_{l,T})| >= (1/2)|beta_{l,T}| for every network.
- The price of the uniform mu is the allocation's excess, 0.32.

*Where:* mi_terms.py -> mi_terms_viol.out; mi_spread.py -> mi_spread.out

*Check:* - The MI identity (with point-mass tests) holds to 9e-8.
- A second violator gives the same picture: hidden fraction 0.996 at the binding pair, 0.31 under uniform mu.

### R12 [NUMERICAL]

p = 3 (b = 2), exact rational margins: mixing is essential and the counterexamples are Sigma-isolated.
- gamma/gamma_IF:
  - exact certificate n = 11: 1.0544;
  - exact certificate n = 13: 1.1092;
  - 1-D certified networks n = 11..18: 1.063..2.212.
- DC energy is 14.0-14.7% of P.
- gamma_split/gamma_IF = -273.4 and -594.0 (exact certificates), and -235 down to -25230 (1-D networks).
- Every non-identity global flip gives a hugely negative margin (n = 11: [-6.3e3, -211]).
- 60 per-neuron flips at n = 11 all give < -2.6e3.
- E_sigma h <= h(split) holds in every case.
- Class-C searches at p = 3 reach exactly the IF value (n = 11, slot-symmetric) and 0.92 (n = 5, general).
So (MM) fails at p = 3 by factors of 1.054-2.21. The class-C side of this is numerical only.

*Where:* p3_orbit.py -> p3_orbit.out (out/p3_orbit.json); p3_classC.py -> p3_classC_n11_sym.out, p3_classC_n5_gen.out

*Check:* - Margins computed in exact Fraction arithmetic over all count vectors.
- They reproduce the certified ratios: 1.054399 (n = 11) and 1.063082 (1-D n = 11).

### R13 [NUMERICAL]

p >= 5: optima are split-exact, and mixing pays only below IF.
Data: ~270 optimised mixed networks (250 grid runs at (5,2), (7,2), (5,3), b in {2, 2.5, 3, 4}, plus 18 orbit runs) and ~80 adversarial runs.
- None exceeds IF.
- Every run with gamma > 0.9 gamma_IF has gamma_split >= gamma - 1e-6.
- Mixing-reliant optima top out at 0.837 (5,2,2) and 0.869 (7,2,2.5).
- IF optima are Sigma-invariant: max|g_R|/mean|g| ~ 1e-6.
- At b > 2 the IF optima are class C: secondary energy <= 7e-6.
- Mixing-reliant optima have spread orbits, e.g. [-0.10, 0.84] at p = 5, far milder than the p = 3 isolation.
Note Section 6 rebuilt at p = 5, n = 6:
- gamma = 0.177, split -1.58;
- with outputs released to all frequencies: 0.182.
Decoupling-violation frontier (best R_Q subject to gamma >= c): violations exist up to gamma ~ 0.80 (p = 5), 0.933 (p = 7), 0.976 (p = 11), and none were found above ~0.85, 0.96, 0.99 respectively.
Gamma-ascent from the near-IF p = 11 violators reaches IF (0.99998) with the mixing removed.

*Where:* exp_random.py, run_e1.sh -> out/e1*.out; orbit_p5.py -> orbit_*.out; exp_sec6.py -> exp_sec6_p5_n6.out; adv_frontier.py -> out/adv_frontier_*; push_gamma.py -> push_gamma_p11.out; test_dsplit.py

*Check:* - IF network checks give 1.000000.
- The programme Gamma_s(5) = 0.88388 (y* equal) was verified analytically.
- These are local searches: evidence, not proof.

### R14 [CONJECTURE]

Residual (MM_inv) for p >= 5: every global maximiser is split-exact. That is: no DC, no dead content, P_split = P, and g_R = 0 on binding offsets; at b > 2 it lies in class C.
- Together with (IF_C) it is equivalent to 'every global maximiser is IF' (R9b), and it gives (C).
- It is the sharpest form found:
  - false at p = 3 (numerical);
  - not relaxable to all networks (R10, certified);
  - no fixed-gap relaxation 'gamma >= lambda gamma_IF => (D_Q)' is p-uniform, since the frontier lambda_0(p) ~ 0.82, 0.95, 0.98 at p = 5, 7, 11.
- The p-uniform alternative is (D_opt): global maximisers satisfy (D_split) with the true price pi. It is trivially true at p = 3 and implies (C) by Note Prop. 'decoupling suffices'.

*Where:* Supported by R9-R13

*Check:* No counterexample in any numerical search at p = 5, 7, 11.

## Obstruction
(MM_opt) concerns the value, not the count.
- Under (IF_C) it is exactly the rigidity half of the value conjecture: every global maximiser is interference-free.
- So it must use p >= 5 (it is false at p = 3) and global optimality (as an inequality over all networks it is false, certified at (5,2,2)).

The obstruction is directional and quantitative.
- Mixed modes hide a cheap feature's interference one offset at a time.
- Exact hiding from a binding set B needs |B|+1 channel coefficients. The channels supply 2|F|-1.
- So any B that is a proper subset of Z_p^x can be hidden from:
  - for free at b = 2, via F2 shared live neurons;
  - otherwise at quadratic cost (DC, dead content, or superadditivity at b > 2).
- On the full set Z_p^x the own mode is isolated in the delta-Fourier domain. There at most half can be hidden, and the payment c_1 = 0.4226 < 1/2 loses.

At sparse primes the programme's binding set B is a proper subset.
- Feature outputs give at most 2K*-1 free channels against |B| >= 2K*: one channel short.
- So whether hiding is ever profitable at an efficient allocation is a comparison of constants (protection slope h_T against payment c_1), uniformly in p.
- This is the (O2) wall. Every mixed mode is now classified and priced, but the constants comparison remains.

The numerical violation frontier rises with p (about 0.82, 0.95, 0.98 gamma_IF at p = 5, 7, 11). Hence no fixed-gap robustness argument can be p-uniform.

## Next steps
- FIRST LEMMA. Claim: for p in {5,7}, n = 2, b in [2,3], every network with gamma >= gamma_IF has no cheap feature. By R9(d) this closes (C) for all networks at (5,2) and (7,2). Plan:
(i) The master inequality on the argmin set gives G <= c_1 sum_{L'} R_l K_l, so the allocation is near uniform.
(ii) Certify the 1-2 variable inequality excess_unif - G = (p/(p-1)) sum x_k^s - Gamma_s <= C*G near uniform shares.
(iii) With mu uniform, delta-Fourier separation charges (1/2)|beta_{l,T}| per cheap feature for any mixing. This closes |L'| = 1, since c_1 < 1/2.
(iv) |L'| = 2: charge both features jointly with a two-character Riesz test on the slack sigma(.,delta) >= 0. Its relation term at p = 5 (chi_{1,T} + chi_{2,T} = -chi_{2,T}) must be bounded; this is the open step.
- Sparse primes: combine 'one channel short' (2K*-1 free channels vs |B| >= 2K*) with the exact prices of R5 into a protection inequality uniform in p. This is (O2) with the free modes restricted to F2 shared neurons.
- Map the decoupling-violation frontier lambda_0(p) at p = 13 and 17, seeding from frontier networks (adv_frontier.py, push_gamma.py). Two questions: does the gap close to 1, and does anything cross IF?
- For the class-C agents: state every class-C result together with its test's Fourier support. By R7(c) each such result then transfers to all networks whose mixed modes avoid that support, which turns the general case into a channel count (R4b).
- Use R6(d) together with bounds on k*(n,s) to settle 2l-pair-free single-output networks at p >= 5. The only single-output mixing left after that is 2l-pair content: partner, DC+2l, and {j, 2l+-j}.
