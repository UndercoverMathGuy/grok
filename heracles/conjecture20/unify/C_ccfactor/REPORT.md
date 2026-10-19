# C_ccfactor: report (saved by the orchestrator from the agent's structured output)

## Headline
CC-FACTOR. I did not prove RD. I did turn it into an exact factorisation and find the weakest per-feature statement that implies it. Every per-feature charging of the interference deficit is either circular or false, and the per-feature detection tools provably cannot see the configurations that block the aggregate statement.

1. Exact identity, for every class-C network and every p, n, b: gamma/B_p = (1 - delta_int) * M(e o zhat) * K_{1/s}(zhat)^{1-s}.
- zhat_l is proportional to P_l^s (the cost profile).
- e_l = [1 - E_{m^(l)} sigma] exp(-(s-1) H_{1/s}(m^(l))) is the within-feature convex-plus-correction efficiency.
- delta_int = 1 - h/M(Re beta) is the only joint term.
- The chain-rule term is exactly Arimoto's conditional Renyi entropy, and it cancels against the efficiency normalisation.

2. RD says this product never exceeds pi * Gamma_s. The exact factorisation CD, h <= pi B_p M(P^s), is the weakest per-feature-charged inequality that implies RD.
- CD holds numerically in class C at every allocation searched: maximum 1.00000, attained by decoupled networks even at allocations where the RD ratio is 0.82.
- For mixed networks CD is false, including at two numerical local maximisers of gamma.

3. Charge rules. The KKT rule is circular: each neuron earns exactly h per unit b-cost, so each per-feature inequality is RD itself. The argmin and Shapley rules are refuted by a certified (5,2,2) network that satisfies CD (ratio 0.942); its cheap feature is charged 6.27x and 2.39x its price.

4. Detection. The per-feature tools give:
- linear background-immune detection that adds up over at most n features on disjoint slot sets;
- a joint pair cone moment that generalises W8.
Second-moment detection is provably blind to antipodal features at an offset pair, and to interference spread evenly over F on every offset set (sum_l Psi^(l) = (p/2)(I+J) is completely positive). At programme binding sets it fails in 135 of 144 cases.

The blocking mechanism is compensation by the background. Atom cancellation blocks only the background-free part (direction-uniform prices, which CD forces). Non-circular content blocks only the extension to general networks.

## Principle
FACTORISATION OF THE VALUE AND WHERE IT CAN BREAK.

Notation, for a class-C network:
- a_i = ||omega_i||^nu; A_l = sum over neurons of feature l of a_i; M_l = A_l / sum a; m^(l) = within-feature masses.
- P_l = sum_{i in l} a_i^{1/s} is the b-cost of feature l.
- zhat_l = P_l^s / sum_k P_k^s is the cost profile.
- e_l = Re beta_l / (B_p P_l^s) is the efficiency of feature l at temperature s.
- delta_int = 1 - h/M(Re beta), with M(v) = min_delta sum_l v_l kappa_l(delta).

Then exactly:

  gamma(theta; s) = B_p * (1 - delta_int) * M(e o zhat) * K_{1/s}(zhat)^{1-s},
  with e_l = [1 - E_{m^(l)} sigma] * exp(-(s-1) H_{1/s}(m^(l))).

How to read it:
- Feature level: M(zhat) K(zhat)^{1-s} is the programme objective, evaluated at the network's own cost profile, so it is at most Gamma_s.
- Within-feature level: e_l is the per-feature convex-plus-correction efficiency. Its supremum over isolated features, after interference, is the price pi.
- Joint level: delta_int is the only place where features interact.
- The Renyi chain rule H(m) = H(zhat) + H^A(neuron | feature) holds with Arimoto's conditional entropy. Its within-feature part cancels against the efficiency normalisation (1 - E sigma = E_zhat[e] exp((s-1) H^A)), so all within-feature Renyi content sits in the e_l.

Two statements follow:
- RD: (1 - delta_int) M(e o zhat) K(zhat)^{1-s} <= pi Gamma_s.
- Exact factorisation CD: (1 - delta_int) M(e o zhat) <= pi M(zhat). In words, the joint interference loss pays for the joint efficiency excess at the network's own allocation.

In variational form, W_S = B_p sup over zhat of Pi_feat(zhat) * omega_S(zhat).
- omega_S(zhat) is the within-feature price at a pinned allocation; omega_S >= pi everywhere and omega_S = pi at the vertices.
- RD only requires omega_S <= pi Gamma_s / Pi_feat, i.e. omega_S = pi at programme maximisers.
- CD requires omega_S = pi at every allocation.

Every attempt to localise the joint term feature by feature either:
- equalises the charges (KKT measures: each neuron earns h per unit b-cost);
- or fails (argmin and Shapley rules).

What survives is an aggregate depth-versus-excess inequality at one binding offset pair. The tools that see a single feature through a background, namely linear Riesz/polydisc tests and the copositive second-moment (cone-moment) inequalities, cannot certify that inequality jointly. Linear tests collide exactly at additive relations. Second moments are blind to antipodal and to evenly spread interference, a completely-positive certificate.

## Results

### CC1 [PROVED]

Renyi chain rule, Arimoto form. Setting: every odd p, n >= 2, s > 1, every class-C network (frequency-pure neurons, no DC). With r = 1/s, K_l = exp H_r(m^(l)) and zhat_l = P_l^s / sum_k P_k^s:
  H_r(m) = H_r(zhat) + H^A_r(N|L),  H^A_r(N|L) = (1/(s-1)) log sum_l M_l K_l^{s-1}.
H^A_r is exactly Arimoto's conditional Renyi entropy of order r of the neuron given the feature.
- zhat is the tilted (escort) feature marginal: zhat_l is proportional to M_l K_l^{s-1}, which is proportional to P_l^s.
- min_l H_r(m^(l)) <= H^A_r <= max_l H_r(m^(l)).
- Decoupled networks (K_l constant) have zhat = M, so the entropy is additive.
- Cost forms: P^s = (sum_l P_l^s) K_r(zhat)^{s-1}, and sum_l P_l^s = (sum a) exp((s-1) H^A_r).
- General networks: apply this to the frequency split (SPLIT R1), whose pieces are the class-C 'neurons'. W5 transfers the Renyi bookkeeping: at b = 2 with no DC, P_b(theta) = P_b(split), so shared neurons change nothing on the cost side.

**Proof / evidence.**

1. P_l^s = A_l K_l^{s-1}. This is s log sum q^r = (s-1) H_r(q) applied to m^(l).
2. Hence sum_i m_i^r = sum_l M_l^r K_l^{1-r} = sum_l (M_l K_l^{s-1})^r, because (s-1) r = 1 - r. This equals Z^r sum_l zhat_l^r with Z = sum_l M_l K_l^{s-1}.
3. Take logs and divide by 1 - r; note r/(1-r) = 1/(s-1).
4. Arimoto's entropy of order alpha is (alpha/(1-alpha)) log sum_l M_l (sum_i (m^(l)_i)^alpha)^{1/alpha}. At alpha = r, (sum_i (m^(l)_i)^r)^{1/r} = K_l^{(1-r)/r} = K_l^{s-1}.
5. The bounds are those of a power mean; the cost forms follow from the same algebra applied to (P_l).

*Check:* c1_accounting.out:
- chain rule error 4.9e-15 on 108 random class-C networks at (5,2), (7,2), (5,3), (11,2), (7,3), (13,2), three values of b each;
- zhat proportional to P^s, error 1e-16;
- bounds hold.
c8_checks.out (a): direct Arimoto formula, error 1.1e-14, including p = 3.
W5 identity on general networks: error 1.8e-15; the b = 2 no-DC cost gap is 1.8e-15.

### CC2 [PROVED]

Factorised convex-plus-correction identity. Setting: every class-C network with h >= 0 (it holds for any h whenever M(Re beta) != 0), every p, n, b. Define:
- delta_int = 1 - h/M(Re beta), in [0,1];
- e_l = Re beta_l / (B_p P_l^s).
Then
  gamma(theta; s) = B_p (1 - delta_int) M(e o zhat) K_r(zhat)^{1-s},  with e_l = [1 - E_{m^(l)} sigma] exp(-(s-1) H_r(m^(l))).
Equivalently gamma = B_p Pi_feat(zhat) Pi_within, where:
- Pi_feat = M(zhat) K_r(zhat)^{1-s} <= Gamma^{(S)}_s;
- Pi_within = h/(B_p M(P^s)) = (1 - delta_int) M(e o zhat)/M(zhat).
The terms of IDENTITY.md are:
- 1 - E_m sigma = E_zhat[e] exp((s-1) H^A_r);
- d_prog = (1 - E_m sigma)(1 - M(yhat)/Gamma_1);
- d_int = (1 - E_m sigma)(M(yhat)/Gamma_1) delta_int;
- the clean profile is yhat, proportional to Re beta, proportional to e o zhat.
So sigma enters only through the per-feature e_l (per neuron, not compensable). d_prog is the programme deficit of the efficiency-weighted cost profile. delta_int is the only joint term. The within-feature Arimoto entropy cancels: all within-feature Renyi content sits in the e_l.

**Proof / evidence.**

1. B_p M(e o zhat) = M(Re beta)/sum_k P_k^s, by homogeneity, since e_l zhat_l = Re beta_l/(B_p sum P^s).
2. K_r(zhat)^{1-s} = sum_k P_k^s / P^s, by CC1.
3. Multiplying, the right side is (h/M(Re beta)) * M(Re beta)/P^s = h/P^s.
4. Formula for e_l: Re beta_l = sum_{i in l} Re beta_il = B_p A_l (1 - E_{m^(l)} sigma) by the definition of sigma_i, and P_l^s = A_l K_l^{s-1}.
5. Pi_feat <= Gamma^{(S)}_s is E_dual D1(iv).
6. Translation to IDENTITY.md: 1 - E sigma = sum Re beta/(B_p sum a) = sum_l e_l P_l^s / sum a, then use CC1. Also M(Re beta) = (sum Re beta) M(yhat), and d_prog, d_int follow from their definitions.

*Check:* c8_checks.out (a): relative error 2.1e-15 for the factorised identity, and 6.7e-14 for the per-feature efficiency identity.
- Networks: 168 random class-C networks, at (5,2), (7,2), (11,2), (5,3), (7,3), (13,2) and (3,2), with b in {2, 2.4, nu-0.2}, including negative margins.
c1_accounting.out:
- gamma = B_p Pi_feat Pi_within, error 1.0e-15;
- Pi_feat <= Gamma^{(S)} (exact vertex enumeration), excess 1.1e-16;
- deficit split e = M(M o eps) - Gamma_1 d_int, error 4e-16;
- IF optima give Pi_feat = Gamma_s and Pi_within = nu^{1-s} to 1e-12, with d_int ~ 1e-15.

### CC3 [PROVED]

Consequences of CC2, for every p, n, b and every class-C theta in N_S.
(i) RD at theta <=> (1 - delta_int) M(e o zhat) K_r(zhat)^{1-s} <= pi Gamma^{(S)}_s.
(ii) Canonical decoupling CD: h <= pi B_p M(P^s) <=> (1 - delta_int) M(e o zhat) <= pi M(zhat). CD => RD at theta.
(iii) No feature cheap at price pi (e_l <= pi for all l) => CD.
(iv) Decoupled networks (relabelled scaled copies of an optimal single feature, ANY allocation) have Pi_within >= pi. If that feature is interference-free, then e_l = pi, delta_int = 0 and Pi_within = pi. This is the KM2 block: always at n = 2, and wherever pi = nu^{1-s}. In that case CD is an equality on the entire decoupled manifold, not only at programme optima. For a non-IF optimal feature, copies may compensate each other in x; then Pi_within > pi, and the failure of CD, are not excluded.

**Proof / evidence.**

(i), (ii): these rewrite CC2. CD => RD because M(P^s) <= Gamma^{(S)}_s P^s: take y = P^s, for which sum y^{1/s} = P.
(iii): M is monotone and delta_int >= 0 (paper Lemma 2).
(iv): gaps of a union add, and the optimal (cone) feature has profile min_x g(x,l) >= pi kappa_1(l) B_p P^s in every direction l. Reason: g = kappa (Re Phi - Im Phi cot(phi/2)) and |cot(pi l/p)| <= cot(pi/p) (GLOBAL R8(a)). Hence h >= pi B_p M(P^s). For IF copies the gaps are constant in x and equal pi B_p P_l^s kappa_l, so equality holds.

*Check:* c4_facts.out (3): E_width X1/X2 IF networks give CD = 1.000000000000 exactly:
- uniform IF at p = 11 (s = 1.35, 1.45, 1.49) and p = 13 (s = 1.45), with RD ratios 0.9976 down to 0.920;
- single IF block at (5, s = 1.45), with RD ratio 0.755.
c1_accounting.out: IF optima give a CD ratio of 1.000000000000.

### CC4 [PROVED]

Two-level variational formula. For zhat in Delta_S let
  omega_S(zhat) = sup { Pi_within(theta) : theta in N^C_S, h > 0, zhat(theta) = zhat }.
Then W^C_S = B_p sup over zhat in Delta_S of Pi_feat(zhat) omega_S(zhat). Moreover:
- omega_S >= pi^C everywhere, and omega_S(e_l) = pi^C at vertices;
- RD^C_S <=> omega_S(zhat) <= pi^C Gamma^{(S)}_s / Pi_feat(zhat) for all zhat;
- CD on N^C_S <=> omega_S = pi^C identically;
- RD forces omega_S = pi^C only at programme maximisers; CD forces it everywhere. The allocation slack Gamma/Pi_feat - 1 is exactly the room RD leaves beyond CD.
Any allocation, any network: for an arbitrary (also mixed) network and ANY allocation Ptilde >= 0 with sum Ptilde_l = P_b(theta), the same algebra gives gamma = B_p Pi_feat(ztilde) h/(B_p M(Ptilde^s)). RD holds at theta iff CD holds for SOME allocation (take Ptilde_l = P_b (y*_l)^{1/s}). So a factorisation theorem has content only for a canonical allocation.

**Proof / evidence.**

1. The formula is CC2 for each theta; take the sup.
2. omega >= pi^C: by CC3(iv), using an optimal class-C single feature (attained, GLOBAL R2).
3. At a vertex e_l the network is a single class-C feature and M(P^s) = kappa_min P_l^s, so omega = pi^C by definition.
4. Since omega >= pi, RD forces equality at the maximisers zhat* of Pi_feat.
5. The any-allocation identity is the same algebra with P replaced by Ptilde. For the equivalence, with Ptilde_l = P_b (y*_l)^r we get M(Ptilde^s) = P_b^s Gamma_s.

*Check:* Same algebra as CC2 (c1_accounting.out, c8_checks.out).
CC14 gives numerical evidence that omega_S(zhat) = pi at non-optimal allocations, e.g. zhat = (0.447, 0.358, 0.195) at p = 11, S = {1,3,4}: CD = 0.99999 while the RD ratio is 0.824.

### CC5 [PROVED]

KKT charges are cost-proportional (allocation through the KKT measure is circular). Setting: every p, n, b; every KKT point theta of gamma(.;s) with h > 0, normalised to unit b-cost, with Fritz-John measure q* (E_dual D3c).
- For every neuron, E_{q*}[g(T_i)] = h a_i^{1/s}, where g(T_i) is the gap function of neuron i's table alone.
- Every set J of neurons is therefore charged h sum_{i in J} a_i^{1/s}; in class C, feature l is charged exactly h P_l.
- Consequently every per-feature inequality of the form E_{q*} g_l <= Lambda P_l is, at that point, equivalent to gamma <= Lambda.
- The delta-marginal/KKT charge rule therefore yields no reduction: each per-feature statement is RD itself.

**Proof / evidence.**

1. D3c: at a KKT point the neurons are balanced, and t_i = E_{q*} g(uhat_i1, ..., what_i) = h nu^{nu/2} a_i^{r-1}.
2. By multilinearity, E_{q*} g(T_i) = (||omega_i||/sqrt(nu))^nu t_i = nu^{-nu/2} a_i * h nu^{nu/2} a_i^{r-1} = h a_i^r.
3. Sum over J.

*Check:* c2_charge.out (B):
- decoupled IF optima at (7,2,2.6), (11,2,2), (13,2,2.3), (7,3,2), with q* computed by NNLS from the active set: max |E_q g(T_i)/(h a_i^r / P) - 1| between 7e-15 and 3.3e-14;
- 10 E_kkt local maxima at (5,2,2) and (7,2,2), including non-IF ones with KKT residual up to 3.6e-6: ratios in [1.0000, 1.0000].

### CC6 [PROVED]

Reduction theorem: the weakest charged within-feature inequality. Setting: class C, every p, n, b, theta in N_S. Pair quantities:
- Rbar(delta) = sum_l Re beta_l kappa_l(delta);
- G(delta) = min_x g(x, delta);
- pair margin h_delta = min(G(delta), G(-delta)) >= h;
- pair depth X = Rbar - h_delta;
- symmetric depth D = Rbar - (G(delta) + G(-delta))/2 <= X;
- excess E_l = Re beta_l - pi B_p P_l^s;
- isolated depth D_l^iso: the same as D for feature l alone.
A charge rule assigns D_l(delta) with sum_l D_l = X (or = D). The within-feature charged inequality is
  WF_l(delta): D_l(delta) >= E_l kappa_l(delta), i.e. [1 - E_{m^(l)} sigma - d_l] exp(-(s-1) H_r(m^(l))) <= pi, with d_l = D_l/(B_p A_l kappa_l).
This is the per-feature IDENTITY with charged interference. With no background and delta = +-l^{-1} it is the definition of pi.
Theorem.
(i) For any rule: WF_l(delta_0) for all l at one delta_0 in Delta(P^s) = argmin_delta sum_l P_l^s kappa_l(delta) implies
  CD_pair(delta_0): X(delta_0) >= sum_l E_l kappa_l(delta_0),
  which implies CD, which implies RD_S at theta.
(ii) Conversely, CD_pair(delta_0) implies WF for the free rule D_l = E_l kappa_l + (X - sum_k E_k kappa_k)/|S|.
So the weakest single-feature-with-background inequality implying RD is CD_pair at a P^s-binding pair, i.e. an aggregate.
Exact factorisation of its symmetric form:
  D(delta) >= sum_l E_l kappa_l(delta) <=> rho(delta) >= taubar(delta),
where:
- rho = D / sum_l D_l^iso is in [0,1] (compensation survival: no prices);
- taubar = sum_l E_l kappa_l / sum_l D_l^iso is the iso-depth-weighted direction-uniform tightness (background-free).

**Proof / evidence.**

1. Summing WF at delta_0 gives h <= h_{delta_0} = Rbar - X <= sum_l (Re beta_l - E_l) kappa_l(delta_0) = pi B_p sum_l P_l^s kappa_l(delta_0) = pi B_p M(P^s).
2. Then CC3(ii).
3. The converse is immediate.
4. rho <= 1 because the minimum of a sum is at least the sum of the minima.
5. Divide by sum_l D_l^iso > 0.

*Check:* c2_charge.out (C): 112 positive-margin perturbed-IF networks at (7,2,2), (11,2,2), (11,2,2.5), (13,2,2.2), (7,3,2):
- CD holds in 112/112;
- CD_avg at a binding offset holds in 112/112.
c5_struct.out (d): on the CD-maximising networks, rho at the binding offsets is 0.57-0.85 and taubar < 0 (no cheap features).

### CC7 [PROVED]

CD forces direction-uniform prices (necessity; this is where atom cancellation enters). Setting: every odd prime p (vacuous at p = 3), every n, b. If CD holds on N_F, then for every single feature theta_1 at frequency 1 and every direction l != +-1:
  min(m(l), m(-l)) <= pi kappa_1(l) B_p P_b(theta_1)^s,  where m(l) = min_x g(x, l).
This is DU_min, a background-free single-feature statement. RD alone does not give it: the construction below has allocation slack Gamma_s - kappa_min > 0.

**Proof / evidence.**

Setup.
- Choose j in F and delta_0 = j^{-1} l.
- Take k = +-j l^{-1}; then k != +-j (since l != +-1) and k delta_0 = +-1.
- theta = theta_1^{(j)} (relabelled, cost eps) together with an optimal single feature at k (cost 1; pi is attained, GLOBAL R2).
Lower bound on h: gaps add, so h >= B_p min_delta [eps^s mhat(j delta) + mhat_opt(k delta)], and mhat_opt >= pi kappa_1 in every direction.
Binding pair of P^s: for eps^s < (kappa_1(2) - kappa_1(1))/2, the unique binding pair is +-delta_0, with M(P^s) = kappa_min + eps^s kappa_1(l).
Offsets delta != +-delta_0 give a value >= pi B_p (kappa_min + gap) - O(eps^s), which exceeds pi B_p M(P^s) for small eps.
Conclusion: if min mhat(+-l) > pi kappa_1(l), the pair +-delta_0 also exceeds pi B_p M(P^s), contradicting CD.

*Check:* c8_checks.out (c), with perturbed non-IF features at (7,2,2), (11,2,2.5), (7,3,2) and eps = 0.3, 0.1, 0.03:
- the gaps-add bound holds, h - bound >= 0 (to 1e-17);
- the CD ratio equals the predicted pair-term ratio to 1e-6;
- the ratio tends to 1 from below as eps tends to 0, exactly when min(m(+-l))/(pi kappa) < 1.
No direction tested violates DU_min. Also c4_facts.out (4): the (5,6) width-6 network has min and average directional ratio 0.17723 in both directions.

### CC8 [PROVED]

Copositive second-moment lemma and joint pair cone moment (multi-feature W8). Setting: class C, every p, n, b; any offset set Delta; any copositive Q on R^Delta; h_Delta = min over x and a in Delta of g(x, a). Then
  sum_{a,b} Q_ab (gbar_a - h_Delta)(gbar_b - h_Delta) >= -(1/2) sum_l V_l <Q, Psi^(l)>,
where:
- Psi^(l)_ab = Re(c_l(a) conj c_l(b)) = kappa_l(a) + kappa_l(b) - kappa_l(a-b);
- V_l = E_x |Phi_l - E Phi_l|^2 = sum_{T nonempty} |beta_{l,T}|^2.
For Delta = {+-delta} with Q off-diagonal (pair cone moment):
  (Rbar(delta) - h_delta)^2 >= Ibar(delta)^2 + sum_l kappa_l(delta) cos(2 pi l delta/p) V_l,  with Ibar = -sum_l Im beta_l sin(2 pi l delta/p).
For one feature at delta = l^{-1} this is exactly E_width W8(i): V <= k_p x^2 |A|^2.

**Proof / evidence.**

1. On Delta, g(x,.) - h_Delta >= 0. Copositivity gives E_x sum_ab Q_ab (g_a - h)(g_b - h) >= 0. Expand into means plus covariances.
2. In class C the fluctuations f_l live on the characters 2l 1_T. Hence E f_l^2 = 0 (circular), and E f_l f_k = E f_l conj(f_k) = 0 for l != k: 2l 1_T + 2k 1_{T'} is never 0, and 2l 1_T = 2k 1_{T'} forces l = k, T = T'.
3. So Cov(g_a, g_b) = sum_l V_l Re(c_l(a) conj c_l(b))/2.
4. For the pair: g(x, +-delta) = R -+ I with R = sum Re Phi_l kappa_l and I = sum Im Phi_l sin phi_l. Also Re(c^2) = kappa^2 - sin^2 phi = -2 kappa cos phi.
5. W8(i): with kappa = 1 - cos(2 pi/p) and x = 1 - h/(kappa |A|), V <= kappa x^2 |A|^2 / cos(2 pi/p) = k_p x^2 |A|^2.

*Check:* c2_charge.out (A): V_l normalisation error 4e-16, circularity 7.5e-16, cross terms 1.1e-15.
c2_charge.out (D): 640 (network, offset) pairs with h_delta > 0 at (5,2), (7,2), (11,2), (5,3), (7,3), (13,2): max (rhs - lhs)/Rbar^2 = -6.1e-3 <= 0, and max rhs/lhs = 0.135 (the bound is loose on random networks). Single features are consistent with W8(i).
c7_quad.out (i): 225 random (network, Delta, nonnegative Q) cases, max (rhs - lhs)/|lhs| = -1.02 <= 0.

### CC9 [PROVED]

Single-pair blindness. Setting: every p, n, b. On Delta = {+-delta}, the weight of feature l in CC8 is
  w_l(Q) = -kappa_l [(Q_++ + Q_--) + 2 Q_+- (kappa_l - 1)].
- No copositive Q gives a positive weight to an ANTIPODAL feature, i.e. one with kappa_l(delta) >= 1, equivalently cos(2 pi l delta/p) <= 0.
- Acute features are detected only with Q_+- > 0, and then antipodal features enter with NEGATIVE weight: they compensate.
- At p = 3 every feature is antipodal at every offset, so the lemma is vacuous there, as it must be.

**Proof / evidence.**

1. w_l > 0 with kappa_l >= 1 requires Q_+- < 0 and 2 |Q_+-| (kappa_l - 1) > Q_++ + Q_--.
2. Copositivity gives Q_++ + Q_-- >= 2 sqrt(Q_++ Q_--) >= 2 |Q_+-|.
3. Together these force kappa_l > 2, which is impossible.

*Check:* c5_struct.out (b): over 199,610 random copositive 2x2 forms and kappa in [0,2]:
- 99,805 antipodal cases, 0 with positive weight;
- 50,475 acute cases with positive weight.

### CC10 [PROVED]

Dense blindness (completely-positive obstruction). Setting: every p and every offset set Delta.
- sum_{l in F} Psi^(l) restricted to Delta equals (p/2)(I + J), which is completely positive.
- Hence, for every class-C network whose fluctuation energies V_l are equal on all of F, every copositive second-moment inequality on every offset set is vacuous: its right side is <= 0.
- In particular the pair cone moment has sum_l kappa_l cos(phi_l) V_l < 0 for equal energies.
- More generally: whenever sum_l V_l Psi^(l)|_Delta is completely positive, no second-moment argument on Delta detects that network's interference.

**Proof / evidence.**

1. sum_{l=1}^{F} kappa_l(a) = p/2 for a != 0, and kappa_l(0) = 0. So the off-diagonal entries are p/2 and the diagonal entries are p.
2. I + J = sum_a e_a e_a^T + 1 1^T is completely positive.
3. COP and CP are dual cones, so <Q, I + J> >= 0 for every copositive Q.

*Check:* c8_checks.out (b): max |sum_l Psi^(l) - (p/2)(I+J)| = 2.3e-13 for p = 5..101 and random Delta. With equal V, max_delta sum_l kappa_l cos(phi_l) = -1.25 < 0.
c4_facts.out (5), arith2 Legendre flat field: the fraction of variance kept by the pair cone moment is -0.0167, -0.0100, -0.0025, -0.0010 at p = 61, 101, 401, 1009 (vacuous).

### CC11 [PROVED]

Background-immune additive linear detection. Setting: class C, every p, n, b. For distinct features l_1, ..., l_k with pairwise DISJOINT nonempty slot sets T_1, ..., T_k (so k <= n), and every delta:
  D(delta) >= sum_i |beta_{l_i,T_i}| |sin(pi l_i delta/p)|,
whatever the other features are.
- Collision characterisation for two-factor tests: product characters 2l 1_T +- 2l' 1_{T'} are feature characters only when T = T' and l +- l' = +-k with k in S (a 3-term relation). With disjoint slot sets there is never a collision.
- This is the multi-feature, background-immune form of S4/R12. Combined with paper Theorem A at n = 2 (per feature: excess <= M/2 and kappa/2 = sin^2 <= |sin|), it reproduces the Note's 'few cheap features' decoupling.

**Proof / evidence.**

1. The test w(x) = prod_i (1 - cos(2 pi <2 l_i 1_{T_i}, x>/p + psi_i)) is >= 0.
2. For |I| >= 2, the character sum_{i in I} sigma_i 2 l_i 1_{T_i} takes the value 2 sigma_i l_i on T_i. This is not constant over its support (the features are distinct), so it is neither 0 nor a feature character +-2k 1_{T''}. Hence E w = 1, and only the single-factor terms pair with g(., +-delta).
3. In class C, g(., delta) has spectrum {0} together with +-(feature characters). Its coefficient at 2l 1_T has modulus |beta_{l,T}| |c_l(delta)|/2 = |beta_{l,T}| |sin(pi l delta/p)|.
4. Choose the phases separately for +-delta and use G(+-delta) <= E[w g(., +-delta)].
5. Collisions: list the entries of 2l 1_T + sigma 2l' 1_{T'} on T\T', T'\T and the intersection of T and T'.

*Check:* c11_linear.out:
- field coefficient modulus equals |beta_{l,T}| to 4.3e-15;
- 504 (network, disjoint family, delta) cases at (7,3), (5,3), (11,2), (7,2), (13,2), (5,4): max (detection - D) = -0.44 <= 0, and max detection/D = 0.30.

### CC12 [REFUTED]

Canonical localisation of d_int by the argmin (test-measure) rule or by the Shapley split of min minus mean, as a single-feature-with-background inequality WF_l(delta) at P^s-binding offsets. Both are false even when CD and RD hold. The KKT rule (CC5) is circular. The isolated rule is background-free (DU) and misses compensation.

**Proof / evidence.**

Certified counterexample at (p,n,b) = (5,2,2), s = 3/2, where pi = 3^{-1/2} (Note Thm q1n2). Script c9_cex.py; weights rounded to 10 digits; exact rational margins and minima; 50-digit costs.
Network: class C, width 7 (cex_charge_rules_p5_n2_b2.npz).
- Feature 1 is effectively one neuron, cheapness e_1/pi = 1.732049 = sqrt 3.
- Feature 2 is a KM2 block plus 2 extra neurons, e_2/pi = 0.9533.
- h = 206.6069 > 0.
- P^s binds at offsets {2, 3}.
- CD ratio 0.941595511664 < 1, so CD and RD hold.
At both binding offsets, feature 1 is charged:
- Shapley charge/price = 2.39361071323;
- argmin charge/price = 6.26635126529.
Mechanism: adding feature 1 LOWERS the background depth, v(12) = 1.768 < v(2) = 1.929 (v(1) = 0.114). The cheap feature is the background's compensator.
More failures:
- c2_charge.out: argmin-rule violations in 9/112 random perturbed-IF networks, all satisfying CD (CD ratios 0.07-0.86, charge ratios up to 2.29).
- c6_shapley.out: Shapley ratios 2.1-3.4 at (7,2), (5,2), (11,2), including a network with CD = 0.9154 at (7,2), S = {1,2,3}.

*Check:* c9_cex.out, c6_shapley.out, c2_charge.out

### CC13 [REFUTED]

Canonical CD for general (mixed) networks, with either natural canonical allocation:
- frequency pieces rescaled to the full neuron cost, i.e. SPLIT's D_split, called CD_prop here;
- raw split costs.
It is false even at numerical LOCAL MAXIMISERS of gamma. Optimal reallocation of shared-neuron costs restores it, but that is equivalent to RD itself (CC4).

**Proof / evidence.**

c4_facts.out (2), SPLIT R10 violators (pi = 3^{1-s} is proved for all networks at n = 2):
- CD_prop = 1.0300, 1.0296, 1.0270 at (5,2,2);
- 1.0662, 1.0436, 1.0427 at (7,2,2);
- 1.0985-1.1061 at (11,2,2);
- their splits give 0.01-0.75.
With optimal per-neuron reallocation, the (5,2,2) ratio equals gamma/(pi B Gamma) = 0.79: every neuron is shared, so the share range is [0,1].
E_kkt local maxima: lm_p5_n2_b2.0_k4 has CD_prop = 1.01464 (gamma = 0.835 of the optimum), and lm_p7_n2_b2.0_k6 has CD_prop = 1.00978.
c8_checks.out (d) and c10_lm7.out: re-optimising from perturbations of size 1e-3 to 1e-1 returns gamma/gamma(k) = 0.999998 and 0.999995, so these are numerical local maximisers.
The mechanism is non-circular content (mixed modes), cost-neutral at b = 2 by W5.

*Check:* c4_facts.out, c8_checks.out, c10_lm7.out

### CC14 [NUMERICAL]

CD holds in class C at every allocation searched. Equivalently omega_S(zhat) = pi (exact factorisation), not only at programme optima. The CD maximum is attained by decoupled IF networks with arbitrary allocations. Networks with width < nu per feature (every feature cheap with interference) stay far below 1.

**Proof / evidence.**

Adversarial torch searches (c3_search.py: L-BFGS with smooth-min annealing; seeds are perturbed decoupled networks at random allocations and Gaussian networks; 16-40 restarts each). pi = 3^{1-s} at n = 2; at n = 3, nu^{1-s} is used, which makes the test stronger.
Maximum CD ratio (RD ratio of the best run in parentheses):
- (7,2,2) S={1,2,3}: 1.00000 (0.999)
- (7,2,2.5) S={1,3}: 1.00000 (0.998)
- (11,2,2) S=F: 1.00000 (0.911)
- (11,2,2) S={1,3,4}: 0.99999 (0.824)
- (11,2,2.5) S={1,2,4}: 1.00000 (0.916)
- (13,2,2) S={1,2,5}: 1.00000 (0.968)
- (13,2,2.2) S=F: 1.00000 (0.945)
- (5,3,2) S=F: 1.00000 (0.900)
- (7,3,2) S=F: 1.00000 (0.826)
The best networks are IF: max |beta_{l,T}|/|beta_l| <= 2.5e-5.
Width < nu: 0.359 (7,2,2), 0.518 (11,2,2), 0.458 (11,2,2.5), 0.545 (13,2,2), 0.235 (5,3,2), 0.318 (7,3,2).

*Check:* out/cd_*.out, best_cd_*.npz, c5_struct.out (d)

### CC15 [NUMERICAL]

At programme maximisers the quadratic route is blocked at the binding set. For p in [5,61] and s in {1.05, 1.2, 1.35, 1.5, 1.75, 2, 2.5, 3, 4}:
(a) every binding offset carries an antipodal support feature in 143/144 cases. The exception is (61, s=4): one all-acute binding offset among 5 (LP tolerance 1e-6).
(b) every support feature is individually quadratically detectable on the binding set in 144/144 cases.
(c) ALL support features are jointly detectable by one entrywise-nonnegative copositive Q in only 9/144 cases.

**Proof / evidence.**

Programme solved by reweighted LP (mechanism_scripts/prog.py); binding tolerance 1e-6. Joint detectability is an LP: find Q_ab >= 0 (a < b in Delta) with <Q, Psi^(l)> <= -1 for all l in S. By Gordan, infeasibility means some lambda >= 0 makes sum_l lambda_l Psi^(l) entrywise nonnegative off the diagonal, i.e. doubly nonnegative.
Undetectable cases include every (5, s), (7, s), (11, s <= 2.5). For p = 5 and 7 they follow from CC10, since there the support is F.

*Check:* c5_struct.out (c), c7_quad.out (ii)

### CC16 [NUMERICAL]

The candidates against BRIEF_CORE section 3:
- p = 3: with one frequency, CD with pi is the definition of pi.
- (5,2,2) mixed violator and E_dual D8/D9 theta_v: canonical CD is refuted for mixed networks. The cost-profile analogue of D9 therefore also fails for general networks, while in class C it holds numerically (CC14).
- E_width X1/X2: IF networks satisfy CD with equality at every allocation. So CD is not a pointwise-in-s entropy inequality.
- Legendre flat design: CC10 makes the quadratic route vacuous on it. The candidates are not support-only.
- (5,6) width-6 network: directionally uniform at 0.177 of nu^{1-s}.
- (FE) at non-global local maxima: canonical CD also fails at non-IF mixed local maxima (CC13). The value-type candidates here need global information too.

**Proof / evidence.**

c4_facts.out:
(1) Certified p = 3 winners: the CD ratio with nu^{1-s} equals gamma/gamma_IF = 1.054399 (n=11), 1.109163 (n=13), and 1.063, 1.193, 1.184, 1.457 for the 1-D certified n = 11..14. So the nu^{1-s} version FAILS, as it must. Every acute or copositive statement is vacuous at p = 3, since cos(2 pi/3) < 0.
(2) Violators: see CC13.
(3) IF networks: CD = 1.000000000000.
(4) (5,6): the single feature at frequency 1 has min and average ratio 0.17723 in both directions.
(5) Legendre flat design: fractions -0.0167 down to -0.0010.

*Check:* c4_facts.out

### CC17 [CONJECTURE]

(CD_C) Exact factorisation in class C. For every odd prime p, every n >= 2, every b in [2,nu], and every class-C network theta: h(theta) <= pi^C B_p M(P(theta)^s). Equivalently omega_S = pi^C on every simplex Delta_S.
Consequences:
- RD for class C, for all S, and with feature weights lambda (replace kappa_l by lambda_l kappa_l in M; the proof of CC3 is unchanged);
- CD_C implies DU_min (CC7).
It is false for mixed networks (CC13). So for all networks it must be combined with SPLIT's (MM_opt) or with a non-canonical reallocation of shared neurons. It is trivially true at p = 3.

**Proof / evidence.**

Supported by CC14 (no violation in any adversarial class-C search, with exact attainment at decoupled IF networks for arbitrary allocations), by CC3(iv), and by the Note's class-C theorem at (5,2) (D_Q).
Open: any general proof needs the joint depth-versus-excess inequality CD_pair at one binding pair. By CC9, CC10 and CC12 it can be proved neither feature by feature nor from first plus second moments alone.

*Check:* out/cd_*.out

## Obstruction
THE PRECISE GENERAL OBSTRUCTION OF THE FACTORISATION ROUTE (every p, n, b).

1. Accounting is exact and closes (CC1, CC2). The value factorises as gamma/B_p = (1 - delta_int) M(e o zhat) K(zhat)^{1-s}.
- The feature level (programme in the cost profile) is bounded by Gamma_s.
- The within-feature level (efficiencies e_l) is controlled by single-feature tools.
- Everything that can beat the product sits in the single joint factor (1 - delta_int), compared against the efficiency excess.
So RD is the statement that the joint interference loss pays for the joint efficiency excess, up to the allocation slack (CC4). Its slack-free version CD holds numerically in class C at every allocation (CC14).

2. LOCALISATION FAILS. The weakest single-feature-with-background inequality that implies RD is CD_pair at one P^s-binding offset pair (CC6). This is an aggregate: the depth there must cover sum_l E_l kappa_l. Any canonical way of charging the depth to features either:
- equalises the charges (KKT: every neuron earns exactly h per unit b-cost, CC5, so each per-feature inequality IS RD); or
- is false (argmin and Shapley: a certified (5,2,2) network with CD = 0.942 whose cheap one-neuron feature is charged 6.27x and 2.39x its price, CC12). The cheap feature is the background's compensator: adding it lowers the background's depth.
The background-free charge (isolated depth) gives only DU. CD needs DU_min in every direction (CC7) plus compensation survival rho >= taubar, an exact and price-free joint condition (CC6).

3. DETECTION FAILS JOINTLY. The per-feature tools see a feature through any class-C background:
- linear Riesz/polydisc tests (CC11);
- the joint copositive second-moment inequalities (CC8), which contain W8.
But:
(a) linear tests add only over features on pairwise disjoint slot sets (at most n features), and product tests collide exactly at 3-term relations l +- l' = +-k, k in S, on a shared slot set;
(b) second moments on an offset pair cannot see antipodal features (CC9). At the binding offsets of programme maximisers some support feature is antipodal in 143/144 cases (CC15);
(c) second moments on ANY offset set cannot see interference spread evenly over F, because sum_l Psi^(l) = (p/2)(I+J) is completely positive (CC10). Joint second-moment detection of the support fails at the binding set in 135/144 cases (CC15), and the Legendre flat field makes it vacuous.
Payment (W6(c)) adds over all cheap features.
So the blocking mechanism for CD and RD in class C is COMPENSATION BY THE BACKGROUND. Precisely, it requires at least n+1 cheap features that are:
- arithmetically related on shared slot sets, AND
- antipodal at the binding pairs or evenly energised.
This blockage needs neither atom cancellation nor non-circular content.

4. ATOM CANCELLATION blocks only the background-free single-feature part: DU_min in every direction, which CD forces (CC7). W8 and the pair cone moment settle aligned atoms at the feature's own binding direction only. Single-pair blindness (CC9) shows that at antipodal directions only linear detection is available. Against it, cancellation makes the payment linear with n-dependent constants (O3).

5. NON-CIRCULAR CONTENT (mixed modes) blocks only the extension from class C to all networks. Canonical CD is false for mixed networks even at numerical local maximisers of gamma (CC13). Only a non-canonical reallocation of shared neurons survives, and that is RD itself (CC4). For all networks the factorisation can therefore only be used at global maximisers through SPLIT (MM_opt), or with an allocation rule that is not local.

6. p = 3 consistency:
- every candidate reduces there to the definition of pi;
- the nu^{1-s} versions fail at all certified winners (1.054-1.457);
- every acute or copositive statement is vacuous (cos(2 pi/3) < 0);
- no statement here uses p >= p_0, b near nu or large n.

## Next steps
- Go beyond second moments, which CC10 shows is mandatory. Use weighted copositive/SOS certificates E_x[w(x) Q(g(x,.) - h)] >= 0, with Riesz-product weights w >= 0 that mix first-order (background-immune, CC11) and second-order (CC8) information. First test numerically whether such certificates detect the completely positive (evenly spread) energy profile and the antipodal features at programme binding sets. LP and SDP over (w, Q) at p <= 23 is a sanity search, not a proof.
- Counterexample search aimed at the obstruction. Build class-C networks whose interference energies lie in the completely positive direction (equal V_l on all of F, or the DNN certificates lambda found by c7_quad.py at programme binding sets), carry relations l +- l' = k on a shared slot set, and have features antipodal at the binding pairs. Maximise the CD and RD ratios. If CD fails while RD holds, the exact factorisation is dead and only the allocation slack of CC4 can save RD. Coordinate with HUNT, whose Fejer flat-bottom designs are this class.
- Prove DU_min, i.e. direction-uniform single-feature prices in every direction, for every n. It is necessary for CD (CC7). For aligned atoms at acute directions it follows from the pair cone moment (CC8 with one feature). The missing part is antipodal directions with atom cancellation: polydisc linear detection against cancellation-linear payment.
- Use the variational formula (CC4) in the opposite direction. RD needs omega_S(zhat) <= pi only up to the allocation slack Gamma_s/Pi_feat(zhat) - 1, which vanishes only at programme maximisers. Look for a 'local CD' near optimal allocations (stability of omega_S at zhat*), combined with a crude global bound, as a route to RD that does not need the exact factorisation. This would be stated for every p, n, b, with slack measured by programme quantities.
- General networks: find an allocation of shared-neuron cost that is canonical and survives CC13. For example, charge each piece by its share of the KKT earnings, which CC5 shows is cost-proportional at critical points. Test it on the SPLIT violators and on the E_kkt local maxima k4 at (5,2,2) and k6 at (7,2,2); canonical CD fails at both.
