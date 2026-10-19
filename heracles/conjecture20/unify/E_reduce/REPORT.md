# E_reduce: report (saved by the orchestrator from the agent's structured output)

## Headline
REDUCE. (1) Proved for every odd p, every n >= 2, every S and every feature weight lambda: "(E_S) for almost every s" is equivalent to "W_S = D_S on all of [1, nu/2]", which is equivalent to "(E_S) at every s < nu/2" (Theorem A). So (E) gives gamma* = pi B_p Gamma_s and the lower half of (C) at every b, b = 2 included by continuity. (2) Upper half. I add a feature-weight (tax) deformation, and Danskin then gives a first-order rigidity result (Theorem B): the vector nu E(theta*) of output-frequency energy shares of any global maximiser is a convex combination of the budget-share vectors z(y) of programme maximisers. Corollaries: every maximiser's feature set is a union of optimal supports, and the count is exact for almost every feature weight (Theorem G). The obstruction is proved: dilation symmetry makes this first-order information empty exactly when the upper half has content, because the orbit average of the optimal z has full support. The residual is "non-straddling", which a frequency-resolved entropy inequality (FE) closes through a strict-concavity (Jensen-Shannon) gap. (3) At b = 2, continuity gives the value, the lower half and the union-of-supports statement. The full count at b = 2 also needs (FE), applied to the split network when shared neurons are free. (4) Target theorem T: (E^lambda) near lambda = 1, plus (FE), plus "optimal single features share one cost-share entropy" (true except on the countable kink set of pi) together imply (C) and the value for all p, n, b. The hypotheses hold with equality on every optimum found numerically and are vacuous at p = 3.

## Principle
COST DEFORMATIONS ARE GIBBS ENVELOPES; THEIR FIRST-ORDER DATA ARE LINEAR OBSERVABLES; COUNTING NEEDS STRICT CONCAVITY ON TOP.

Any deformation that changes only the cost leaves the margin h fixed. Then log gamma = log h - (smooth function of the deformation parameters), and every value function is an upper envelope. Its one-sided derivatives (Danskin) are extremes of linear observables over the maximisers:
- temperature s: minus the cost-share entropy H;
- output-frequency tax mu_l: -(nu/2) E_l, where E_l is the cost-weighted share of output energy at frequency l.
The decoupled value D = pi B_p Gamma is an envelope of the same kind.

So a value identity W = D on a neighbourhood gives an identity of convex hulls: conv{(H, nu E)(maximisers)} = conv{(H, nu E)(decoupled optima)}. From this:
- the value and the lower half of (C) follow from the s-direction alone, with an "almost every s" hypothesis (Theorem A);
- the feature sets follow from the mu-directions, but only up to convex hulls (Theorem B, Corollary U).

Dilation symmetry (Z_p^x/{+-1} acts regularly on F) makes the set of programme maximisers a non-trivial orbit whenever an optimal support is not all of F. The orbit average then has full support, so no envelope argument can see the upper half (Proposition O). Breaking the symmetry with generic feature weights restores uniqueness, and with it the exact count (Theorem G). At the symmetric point lambda = 1 the missing ingredient is pointwise and strictly concave in nu E:
- (FE): H(theta*) >= H(nu E(theta*)) + H^(1)_min(s), a frequency-resolved (E);
- its Jensen-Shannon gap excludes straddling maximisers.

One-line reading. The temperature fixes the value; feature taxes fix the frequency usage up to symmetry; one entropy inequality, resolved by frequency, fixes the count.

## Results

### R0-setting [PROVED]

Definitions used throughout. Proofs are in /home/user/grok/heracles/conjecture20/unify/E_reduce/reduce_proofs.tex.

Networks and cost.
- a_i = ||omega_i||^nu, P_sigma = sum_i a_i^{1/sigma} = sum_i ||omega_i||^{nu/sigma}.
- Cost shares w_i = a_i^{1/sigma}/P_sigma; entropy H(theta,sigma) = -sum w_i log w_i.

Feature weights (new object; it does real work in R7-R11).
- For lambda in (0,inf)^F let G_l(x,delta) = Re(Phi_l(x)(1-rho^{-l delta})).
- g_lambda = sum_l lambda_l G_l, h_lambda = min g_lambda, gamma_lambda = h_lambda/P_sigma^sigma.
- Equivalently, h_lambda is the margin of the network whose outputs are multiplied by Lambda = P_0 + sum lambda_l P_l.
- lambda = 1 is the original problem.

Tax picture.
- tau_mu multiplies the frequency-l output by sqrt(mu_l).
- ||omega||_mu^2 = sum_t ||u_t||^2 + ||P_0 w||^2 + sum_l mu_l ||P_l w||^2.
- Then h(theta) = h_lambda(tau_mu theta) with lambda = mu^{-1/2}, and cost(tau_mu theta) = cost_mu(theta).

Classes and programme.
- N_S = {beta_j = 0 for all j not in S}; W^lambda_S(sigma) = sup over N_S of gamma_lambda.
- m_lambda(y) = min_{delta != 0} sum_l lambda_l y_l kappa_l(delta).
- Gamma^{lambda,(S)}_sigma = max{m_lambda(y) : spt y in S, sum y^{1/sigma} <= 1}, equivalently max over A != 0 of m_lambda(A)/||A||_{1/sigma}.
- M^{lambda,(S)} is its set of maximisers; z(y)_l = y_l^{1/sigma} are the budget shares.
- D^lambda_S = pi(sigma) B_p Gamma^{lambda,(S)}_sigma.

Observables.
- Output-frequency usage E_l(theta,sigma) = sum_i w_i ||P_l w_i||^2/||omega_i||^2.
- H^(1)_min/max(sigma): least/largest entropy of optimal single features; Delta^(1) = H^(1)_max - H^(1)_min.
- H^dec,lambda_S = H^(1)_min + min over M^{lambda,(S)} of H(z(y)).

Hypotheses.
- (E^lambda_S) at sigma: every maximiser of gamma_lambda on N_S has H >= H^dec,lambda_S.
- (FE) at sigma for a maximiser theta: H(theta) >= H(nu E(theta)) + H^(1)_min(sigma).

**Proof / evidence.**

Definitions. The tax identities hold because tau_mu only rescales orthogonal output planes:
- every gap G_l of tau_mu theta is sqrt(mu_l) times that of theta;
- the norms are as stated;
- beta_l(tau_mu theta) = sqrt(mu_l) beta_l(theta), so every N_S is preserved.

*Check:* check_bounds.py: tax derivative and weighted-margin code. elib.py implements every object.

### R1-basic [PROVED]

(a) Width reduction. For every theta and sigma there is theta' with:
- the same table and width <= D = p^nu;
- P_sigma(theta') <= P_sigma(theta);
- neurons that are factorwise rescalings of neurons of theta.
So N_S and the single-feature class are preserved.
(b) Balancing the factors of a neuron and deleting its output DC change no weighted gap and no beta_j, and do not increase the cost (strictly, unless the neuron is already balanced and output-DC-free). Hence every maximiser, of any gamma_lambda on any N_S or of the single-feature efficiency, is balanced and output-DC-free, and so sum_l nu E_l = 1.
(c) W^lambda_S(sigma) = max of gamma_lambda over the compact set K_D intersected with N_S, where K_D = {width <= D, sum ||omega_i||^2 = 1}. W^lambda_S, pi and Gamma^{lambda,(S)} are continuous and positive on [1, nu/2] x (0,inf)^F.

**Proof / evidence.**

(a) This is GLOBAL R2 (refereed). Write mu_i = prod_t ||u_it|| ||w_i||. AM-GM gives ||omega_i||^b >= nu^{b/2} mu_i^{1/sigma}, with equality iff the neuron is balanced. The function sum mu_i^{1/sigma} is concave and non-decreasing along the recession cone of the pointed polyhedron {mu >= 0 : sum mu_i T^_i = T}, so it attains its minimum at a vertex, which has <= p^nu non-zero entries. Balance those neurons.
- N_S is a condition on the table, so it is preserved.
- The single-feature class is preserved because the new neurons are rescaled old ones.
(b) Rescaling with factor product 1 keeps the rank-one term. P_0 w adds a constant in c to f(x)[.], which cancels in every gap and in every beta_j (j != 0). The cost claims follow from AM-GM and orthogonality. For a balanced, output-DC-free neuron, sum_l ||P_l w||^2 = ||w||^2 = ||omega||^2/nu, so sum_l nu E_l = sum_i w_i = 1.
(c) Scale invariance holds because b sigma = nu. By (a) the sup over h_lambda > 0 is attained on K_D intersected with N_S, which is compact. On K_D, P_sigma >= D^{1-b/2} > 0 (power mean, b >= 2), so gamma_lambda is jointly continuous and its maximum is continuous.
- pi: the same argument on the single-feature class.
- Gamma^{lambda,(S)}_sigma = max over the simplex Delta_S of m_lambda(z^sigma).
- Positivity: pi >= nu^{1-sigma}, Gamma^{lambda,(S)} >= lambda_j kappa_min for j in S, and W >= D (R2).

*Check:* check_chain.out: |sum nu E - 1| <= 5.6e-16 on balanced DC-free random networks at (5,2), (7,3), (11,2), random b. check_weighted.out: sum nu E = 1.000000 at every optimum found.

### R2-bounds [PROVED]

(a) Weighted restricted universal bound: for theta in N_S and every sigma, h_lambda(theta) <= B_p Gamma^{lambda,(S)}_sigma P_sigma(theta)^sigma.
(b) Decoupled networks. Let theta_1 be a single feature with c(theta_1) = min_x cm(Phi_1) > 0, and let A >= 0, A != 0, spt A in S. The union Theta(theta_1, A) of the relabelled copies (a -> l a), each scaled by A_l^{1/nu} in every factor:
- lies in N_S;
- satisfies gamma_lambda(Theta; sigma) >= psi(sigma) for every sigma and lambda, where psi(sigma) = c(theta_1) m_lambda(A)/(sum_{l,i} (A_l a_i)^{1/sigma})^sigma = e(theta_1;sigma) B_p m_lambda(A)/||A||_{1/sigma} <= D^lambda_S(sigma);
- attains equality in the last step at sigma = s if theta_1 is optimal at s and A is in M^{lambda,(S)}(s);
- has entropy H(Theta,sigma) = H(z) + H(theta_1,sigma), with z_l = A_l^{1/sigma}/sum A^{1/sigma}.
In particular W^lambda_S >= D^lambda_S (the restricted, weighted price-programme theorem).
(c) W^lambda_S(1) = D^lambda_S(1) = B_p Gamma^{lambda,(S)}_1.

**Proof / evidence.**

(a) E_x Phi_l = conj(beta_l), so the average of E_x g_lambda(., delta) and E_x g_lambda(., -delta) is sum_l lambda_l Re(beta_l) kappa_l(delta) (paper Lemma 2 with weights). For theta in N_S only l in S occur. With Q_l = sum_i |beta_il|^{1/sigma}, Re beta_l <= |beta_l| <= Q_l^sigma =: A_l (l1 <= lr). So h_lambda <= m_lambda(A) <= Gamma ||A||_{1/sigma} = Gamma (sum Q_l)^sigma <= Gamma (C_b P)^sigma = B_p Gamma P^sigma, by the single-neuron budget with C_b^sigma = B_p.
(b) Copy l has outputs in span{1, cos_l, sin_l}, so it contributes only to beta_l and to G_l. By the cone lemma its gap is A_l g_1(lx, l delta) >= A_l c kappa_1(l delta) = A_l c kappa_l(delta). Weighted gaps add over the union, so h_lambda(Theta) >= c m_lambda(A).
- Relabelling permutes coordinates, so neuron i of copy l has ||omega||^nu = A_l a_i. This gives the cost and the factorisation (sum_{l,i} (A_l a_i)^{1/sigma})^sigma = ||A||_{1/sigma} P_sigma(theta_1)^sigma.
- The shares are z_l w_i, so the entropy adds.
- e <= pi and m_lambda(A)/||A|| <= Gamma give the last inequality.
(c) pi(1) = 1, since nu^{1-s} <= pi <= 1. Combine (a) and (b).

*Check:* check_bounds.out: max h_lambda/(B_p Gamma^lambda_S P^s) = 0.8307 <= 1 over 600 random networks in random classes N_S (p in {5,7,11}, n in {2,3}, random lambda in [0.6,1.6]^F, b in {2, random, nu}). check_decoupled.out: min gamma_lambda(Theta)/psi - 1 = -3.3e-16 (equality up to rounding); |H(Theta) - H(z) - H(theta_1)| <= 4.4e-16; weighted IF networks at b = nu attain B_p Gamma^lambda_1 to 3.7e-15.

### R3-danskin [PROVED]

Danskin, elementary form. Let:
- X be compact metric and U open (or convex, with directions pointing inward);
- f: X x U -> R be continuous, with grad_xi f continuous;
- v = max over X of f.
Then v'(xi; eta) = max over argmax of <grad_xi f, eta>.

Applications.
(i) d/ds log gamma(theta;s) = -H (temperature identity). Hence log W^lambda_S has right derivative -H*_min and left derivative -H*_max, the extremes over ALL maximisers of any width.
(ii) In the tax picture, d/dmu_l log gamma_mu(theta) at mu = 1 equals -(nu/2) E_l(theta).
(iii) log pi has one-sided derivatives -H^(1)_min and -H^(1)_max. So Delta^(1)(s) > 0 only at kinks of log pi, which form a countable set.

**Proof / evidence.**

>= : for x* in the argmax at xi, v(xi + t eta) - v(xi) >= f(x*, xi + t eta) - f(x*, xi).
<= : take t_k -> 0 realising the limsup and maximisers x_k at xi + t_k eta, with x_k -> x_bar along a subsequence. Then x_bar is in the argmax at xi (continuity of f and v). By the mean value theorem, v(xi + t_k eta) - v(xi) <= f(x_k, xi + t_k eta) - f(x_k, xi) = t_k <grad f(x_k, xi + tau_k eta), eta>, and uniform continuity gives the limit.

(i) Let Z = sum a_i^{1/s}. Then d/ds[s log Z] = log Z - (1/s) sum w_i log a_i = -sum w_i log w_i. H is jointly continuous for theta != 0, including zero neurons (a^{1/s} is continuous at 0). Apply on X = K_D intersected with N_S and {h_lambda >= eta}, which contains all maximisers for s in a compact interval. Enlarging D does not change v, so the extremes are over all widths.
(ii) d/dmu_l of ||omega_i||_mu^b equals (b/2) ||omega_i||^{b-2} ||P_l w_i||^2. Multiply by -s/P and use s b = nu.
(iii) Same argument on the single-feature class. A function with one-sided derivatives everywhere has distinct one-sided derivatives at only countably many points.

*Check:* check_bounds.out: |d/ds log gamma + H| <= 9.7e-10 and |d/dmu_l log gamma_mu + (nu/2) E_l| <= 1.2e-9 (finite differences on random networks at (5,2), (7,3), (11,2)). check_programme.out (1), at the p=5 transition s_c = 1.85520596: right slope of log Gamma = -0.000007 (Danskin: 0) and left slope = -0.693147 (Danskin: -log 2).

### R4-dini-ae [PROVED]

Monotonicity with an almost-everywhere hypothesis. Suppose:
- L is continuous on [a,c];
- D^+L <= phi on [a,c), with phi <= C < infinity everywhere;
- phi <= 0 outside a Lebesgue-null set N.
Then L is non-increasing.

**Proof / evidence.**

Step 1 (Dini). If M is continuous with D^+M <= 0 on [a,c), then M is non-increasing. Fix a <= a' < c' <= c and eta > 0, and set M_eta(t) = M(t) - M(a') - eta(t - a'). Let t* = sup{t : M_eta <= 0 on [a',t]}. Then M_eta(t*) <= 0. If t* < c', then D^+ M_eta(t*) <= -eta < 0 gives M_eta < 0 just to the right of t*, a contradiction. Let eta -> 0.
Step 2. Choose an open U containing N with |U| < eps/C (outer regularity of Lebesgue measure). Put G = C 1_U and I(t) = int_a^t G.
- On U, I has right derivative C; elsewhere D_+ I >= 0. So D_+ I >= G >= phi everywhere.
- Hence M = L - I has D^+M <= phi - G <= 0.
- By Step 1, L(c') - L(a') <= I(c') - I(a') <= eps. Let eps -> 0.
The bound C is essential: the Cantor function has derivative 0 a.e. and increases. No Vitali-Caratheodory and no measurability of phi are needed.

*Check:* check_dini.out: (1) Cantor function: c(1) - c(0) = 1 while difference quotients at 0 grow like (3/2)^k (2.25, 11.4, 57.7, 292), illustrating why the bound C is needed. (2) Random L with slope C on an open set of measure 3.3e-3 and negative slope elsewhere: max_{a<c} L(c) - L(a) = 9.8e-4 <= eps = 1e-2.

### R5-thmA-equivalence [PROVED]

THEOREM A (reduction and equivalence). Fix lambda in (0,inf)^F and S. The following are equivalent:
(i) (E^lambda_S) at almost every s in [1, nu/2];
(ii) W^lambda_S = D^lambda_S on all of [1, nu/2];
(iii) H*_{S,min}(s) = H^dec,lambda_S(s) for every s in [1, nu/2); the maximal entropies also agree on (1, nu/2];
(iv) (E^lambda_S) at every s in [1, nu/2).
In particular the almost-everywhere hypothesis improves itself to every s < nu/2. At s = nu/2 (b = 2) (E) is not asserted (see R13).

**Proof / evidence.**

(i) => (ii). L = log W - log D is continuous (R1c), L >= 0 and L(1) = 0 (R2).
- Upper Dini of log W: by R3, D^+ log W(s) <= -H(theta_bar, s) for a maximiser theta_bar of width <= D. This is the elementary half: take maximisers at s_k decreasing to s, use the subsequence limit, and the temperature identity integrated over [s, s_k].
- Lower Dini of log D: for any optimal single feature theta_1 at s and any y in M^{lambda,(S)}(s), the curve psi of R2(b) with A = y lies below D on [1, nu/2] and touches it at s. Its logarithm is a Gibbs function with derivative -(H(theta_1,s) + H(z(y))). So D_+ log D(s) >= -H^dec(s).
- Hence D^+ L(s) <= phi(s) := H^dec(s) - H(theta_bar(s), s).
- phi <= C := log(D|S|): a width-reduced optimal single feature has H <= log D, H(z) <= log |S|, and H >= 0.
- phi <= 0 wherever (E) holds.
- R4 gives L <= L(1) = 0, so L = 0.
(ii) => (iii). By R3, log W has right derivative -H*_min. Also log D = log pi + log B_p + log Gamma^{lambda,(S)}, where log pi has right derivative -H^(1)_min. log Gamma^{lambda,(S)}_s = max over A in Delta_S with m_lambda(A) >= eta of [log m_lambda(A) - s log sum A_l^{1/s}] is a Gibbs envelope with right derivative -min over M of H(z(y)). Equal functions have equal one-sided derivatives.
(iii) => (iv) => (i): immediate.

*Check:* check_programme.out (2): the right slope of log D_S (n = 2, where pi = 3^{1-s} is proved and every optimal single feature has H = log 3) equals -H^dec_S to within 1.3e-7 at 15 points: p in {7, 11, 13}, S in {F, {1,2}, {1,2,3}}, s in {1.1, 1.3, 1.45}. check_weighted.out / check_weighted2.out: at every optimum found for s > 1, H = H^dec (to 1e-5) and W/D = 0.999998-1.000000, as (iii) requires.

### R6-value-lower-half [CONDITIONAL]

Hypothesis: (E_S) at almost every s, for all S (lambda = 1).
Conclusions, for every odd p, every n >= 2 and every b in [2, nu], b = 2 included:
(a) gamma*_b = pi_{n,b}(p) B_p Gamma_s(p) (Conjecture 20'). In fact (E_F) a.e. is EQUIVALENT to this, since N_F is all networks.
(b) Lower half of (C): the feature set S* of every global maximiser contains the support of a programme maximiser, i.e. Gamma^{(S*)}_s = Gamma_s.

**Proof / evidence.**

(a) Theorem A with S = F, since W_F = gamma*.
(b) Theorem A gives W_S = D_S on the CLOSED interval [1, nu/2]: continuity transfers it to s = nu/2. The global maximiser theta* lies in N_{S*}, so gamma* = gamma(theta*) <= W_{S*} = pi B_p Gamma^{(S*)}_s <= pi B_p Gamma_s <= gamma* (price-programme theorem). Hence equality, and Gamma^{(S*)}_s = Gamma_s. Only (E_{S*}) is used for the inequality; the price-programme theorem supplies gamma* >= pi B_p Gamma_s.

*Check:* p=3, check_p3.out: F = {1}. Every network is single-feature, so maximisers are optimal single features, H^dec = H^(1)_min, and (E) is vacuous. W = kappa_min B_3 pi = D identically, consistent with the certified networks (gamma/gamma_IF = 1.0631-2.2120, H = log m < log nu): the theorem predicts the PRICE value, not the IF value.

### R7-tax-decoupled-family [PROVED]

Tax picture and decoupled envelope.
(a) In the tax picture, log W^{mu^{-1/2}}_S(sigma) = max over K_D intersected with N_S of f(theta; sigma, mu), where f = log h(theta) - sigma log sum_i ||omega_i||_mu^{nu/sigma}. At mu = 1 the gradient of f is (-H, -(nu/2) E).
(b) Let the decoupled family consist of tuples ((theta^(l))_{l in S}, A): single features of width <= D relabelled to l, and A in the simplex. Define Psi = log N - sigma log sum_{l,i} (A_l ||omega^(l)_i||_mu^nu)^{1/sigma}, with N = min_delta sum_l A_l c(theta^(l)) kappa_l(delta). Then max Psi = log D^{mu^{-1/2}}_S(sigma) for all (sigma, mu).
(c) At mu = 1 the maximisers of Psi are: optimal single features on spt A, and A_l B_l^sigma proportional to some y in M^{(S)}. Their gradients form the set G_D = {( -H(z(y)) - sum_l z_l h_l , -z(y)/2 )}, with h_l in the set of entropies of optimal single features, which lies in [H^(1)_min, H^(1)_max].

**Proof / evidence.**

(a) By R0 and R1. The derivative of ||omega_i||_mu^b is computed in R3(ii).
(b) <= : put theta~ = tau_{mu_l} theta^(l). Then c(theta^(l)) = mu_l^{-1/2} c(theta~) <= mu_l^{-1/2} pi B_p B_l^sigma, with B_l = sum_i ||omega^(l)_i||_mu^{nu/sigma}. Insert into N and use the homogeneous programme with amplitudes A_l B_l^sigma and weights lambda = mu^{-1/2}.
>= : take optimal theta~ and programme-optimal amplitudes.
(c) If a copy l with A_l > 0 is below price, every delta-term of N strictly decreases, because kappa_l(delta) > 0 for all delta != 0. Optimal single features are balanced and output-DC-free (R1b), so ||P_l w||^2/||omega||^2 = 1/nu and E_l(Theta) = z_l/nu. The entropy is H(z) + sum z_l H(theta^(l)) by the chain rule. The family is compact after normalisation (restrict to N >= eta), and Psi is C^1 in (sigma, mu), so R3 applies.

*Check:* check_programme.out (4), the D side in the tax directions at p = 11, s = 1.5, where M has 5 maximisers. In all 10 directions the finite-difference derivative of log Gamma^{mu^{-1/2}} matches the Danskin value max_y <-z(y)/2, eta> to 1.5e-5: 0 for a tax, +0.284727 for a subsidy.

### R8-thmB-first-order-rigidity [CONDITIONAL]

THEOREM B. Hypothesis: W^lambda_S(sigma) = D^lambda_S(sigma) for sigma in (s0 - eps, s0] and |lambda - 1| < eps, with s0 in (1, nu/2].
Conclusion: every maximiser theta* of gamma(.; s0) on N_S admits a finitely supported probability (c_k) on M^{(S)}(s0) and a number h_bar in [H^(1)_min(s0), H^(1)_max(s0)] with
  nu E(theta*, s0) = sum_k c_k z(y_k)   and   H(theta*, s0) <= sum_k c_k H(z(y_k)) + h_bar.
The second relation is an equality when s0 < nu/2 and the hypothesis also holds on [s0, s0 + eps).
In words: a maximiser's output-frequency usage is a mixture of the programme maximisers' budget shares.

**Proof / evidence.**

Let v_W = log W and v_D = log D as functions of xi = (sigma, mu) near xi0 = (s0, 1). They agree on {sigma <= s0}.
- For every direction eta in Pi = {eta_sigma <= 0}, the point xi0 + t eta stays in that set, so v_W'(xi0; eta) = v_D'(xi0; eta).
- By R3 and R7, the support functions of the compact gradient sets G_W (over maximisers of any width: enlarging D leaves v_W unchanged) and G_D agree on Pi.
- Let C = polar(Pi) = {(t,0) : t >= 0}. For compact convex K, the support function of K + C equals that of K on Pi and is +infinity off Pi. Hence conv G_W + C = conv G_D + C.
- Write g(theta*) = (-H, -(nu/2) E) as sum_k c_k g_k + (t, 0) with t >= 0 and g_k in G_D. The E-coordinates give nu E = sum c_k z_k. The H-coordinate gives H = sum c_k [H(z_k) + h_bar_k] - t, with h_bar_k = sum_l z_{k,l} h_{k,l}; set h_bar = sum_k c_k h_bar_k.
- Two-sided case: C = {0}.

*Check:* Every optimum found satisfies it with a single k. check_weighted.out: |nu E - z(y^lambda)| <= 2.3e-6 at (5,2) and (7,2), b in {2, 2.2, 2.5, 3}, 7 weightings. check_weighted2.out: at the degenerate point p = 11, b = 2 (5 programme maximisers), each optimum matches exactly one z(y) to 9.4e-7, e.g. nu E = (0.4305, 0, 0.5695, 0, 0), features (1,3).

### R9-union-of-supports [CONDITIONAL]

COROLLARY U. Hypothesis: W^lambda_S(s) = D^lambda_S(s) for all S and all lambda near 1, at this s only. Valid at every s in [1, nu/2], b = 2 included.
Conclusion: every global maximiser theta* with feature set S* satisfies
  nu E(theta*) in conv{ z(y) : y in M(s), spt y in S* }   and   S* = union{ spt y : y in M(s), spt y in S* }.
So (C) holds at theta* whenever S* contains only one optimal support. (C) can fail only through a maximiser whose feature set is a union of two or more distinct optimal supports.

**Proof / evidence.**

Run the mu-part of Theorem B at fixed s inside N_{S*}. Only mu-directions are used, and both signs are available. theta* is a maximiser in N_{S*}. By R6(b) (lambda = 1, S*), Gamma^{(S*)} = Gamma, so the maximisers of Gamma^{(S*)} are exactly the programme maximisers supported in S*. Then
  S* in spt(nu E(theta*)) in union_k spt y_k in S*.
The first inclusion holds because a feature j needs some w-hat_i(j) != 0, hence E_j > 0.

*Check:* Every optimum found has S* equal to a single optimal support. Examples: p = 11, b = 2: features (1,3) or (2,5) out of five optimal 2-supports. (5,3), b = 2.1: (1) or (2). check_weighted2.out, check_generic.out.

### R10-dilation-obstruction [PROVED]

PROPOSITION O (why value identities cannot give the upper half).
- Z_p^x/{+-1} acts regularly on F and fixes Gamma_s.
- If a programme maximiser y has spt y != F, its orbit contains >= 2 maximisers with distinct supports and equal H(z).
- The orbit average z_bar has full support F.
- The point (H(z) + H^(1)_min, z_bar) lies in conv G_D. It therefore satisfies every conclusion of Theorems A and B and every first-order (envelope) identity, for all S, lambda and s.
So no argument that uses only value functions and their one-sided derivatives can exclude a maximiser using all |F| frequencies, at any (p, n, b) where the upper half of (C) has content. The same applies to the subsidy/removal route: subsidising j shows only that j lies in the union of the optimal supports, and that union is all of F.

**Proof / evidence.**

Dilation j -> +-kj permutes F transitively, and Gamma is invariant under it. The union of the supports along an orbit is invariant and non-empty, hence equals F. Dilation permutes the coordinates of z, so H(z) is constant on the orbit. Take h_l = H^(1)_min for all l and equal weights on the orbit: the point is in conv G_D.
Its entropy H(z) + H^(1)_min equals H^dec, so it is consistent with (E) and with Theorem A(iii). Corollary U then allows S* = F.

*Check:* check_chain.out / check_programme.out (3). The union of the optimal supports is F in each case:
- p=11, s=1.5: 5 maximisers, supports {1,3},{2,5},{1,4},{4,5},{2,3}; orbit average has support 5/5; JS gap H(z_bar) - H(z) = 0.926.
- p=13, s=1.5: 6 maximisers; JS 0.700.
- p=7, s=1.8: 3 maximisers; JS 0.419.
- p=5, s=1.9: 2 maximisers; JS 0.693.
H(z) is constant on each orbit (to 1e-6).

### R11-thmG-generic-weights [CONDITIONAL]

THEOREM G. Hypothesis: W^lambda_S(s) = D^lambda_S(s) for all S and all lambda in an open set Lambda_0, at a fixed s in [1, nu/2] (supplied by (E^lambda) a.e. via Theorem A).
(a) Every maximiser of the lambda-weighted problem has nu E in conv{ z(y) : y in M^lambda(s) }.
(b) [PROVED unconditionally] M^lambda(s) is a single point for Lebesgue-almost every lambda.
(c) Hence for almost every lambda in Lambda_0, every global maximiser of the lambda-weighted margin has nu E = z(y^lambda) and feature set exactly spt y^lambda. (C) holds for generic feature weights with no further hypothesis; the symmetric point lambda = 1 is the only obstruction.
COROLLARY S (lambda-stability). A global maximiser at lambda = 1 that is a limit of maximisers at weights lambda_k -> 1 with unique programme maximisers satisfies (C). A straddler is never such a limit.

**Proof / evidence.**

(a) Theorem B's mu-part at fixed s, around mu0 = lambda^{-2}; Lambda_0 is open, so all directions are available. Under tau_{mu0}, E^{mu0}_l(theta) = E_l(tau theta)/mu0_l on both sides, a linear bijection. The decoupled maximisers have E_l = z_l(y)/nu.
(b) Put t = log lambda and B_l = lambda_l A_l. Then log Gamma^lambda_s = max over B of [log m(B) - s log sum_l B_l^{1/s} e^{-t_l/s}].
- Each bracket has t-gradient q = z(y) and t-Hessian -(1/s)(diag q - q q^T), which is >= -(1/s) I.
- So v(t) + |t|^2/(2s) is a maximum of convex functions, hence convex, hence differentiable almost everywhere.
- Where v is differentiable, R3 makes eta -> max_y <z(y), eta> linear, so all maximisers share the same z; and z determines y (y_l = z_l^s).
(c) By (a) and (b), nu E = z(y^lambda), so S* is inside spt y^lambda. R6(b) for the lambda-problem gives a maximiser supported in S*, which must be y^lambda.
Corollary S: take limits. E is continuous on K_D and the programme argmax has a closed graph, so nu E(theta_bar) = z(y_bar); Corollary U in N_{S_bar} then gives S_bar = spt y_bar.

*Check:* check_generic.out:
- 200 random lambda within 2% of 1: always exactly 1 programme maximiser, at p=11 s=1.5, p=13 s=1.5, p=7 s=1.8 and p=5 s=1.9 (versus 5, 6, 3 and 2 maximisers at lambda = 1).
- Weighted searches at random lambda, (11,2,2) and (7,3,2): gamma/D = 0.999998 and 0.999999; |nu E - z(y^lambda)| <= 9.2e-7; features = spt y^lambda.
- check_weighted2.out at (5,3,2.1) with lambda = (1, 1.05): the unique support is {2}, and the best optimum has features (2); the {1}-networks sit at 0.952 = 1/1.05.

### R12-removal [PROVED]

Removal inequality.
- For any network with b >= 2, deleting the frequency-j output of every neuron gives theta^{-j} with P(theta^{-j}) <= (1 - E_j) P(theta).
- At a global maximiser, h(theta^{-j}) <= h(theta*)(1 - E_j(theta*))^s: every unit of frequency-j output energy must earn margin.
- As E_j -> 0 this is the subsidy derivative of Theorem B. It is consistent with every straddler, which uses each j in its union with E_j > 0. So finite removal adds nothing beyond Corollary U.

**Proof / evidence.**

P(theta^{-j})/P = sum_i w_i (1 - eps_ij)^{b/2} <= sum_i w_i (1 - eps_ij) = 1 - E_j, with eps_ij = ||P_j w_i||^2/||omega_i||^2 and b/2 >= 1. Then h(theta^{-j}) <= gamma* P(theta^{-j})^s.
delta-Fourier isolation (p52 R12) gives the exact mean effect of the removal on each offset. But a straddler uses j through its fluctuating field at the binding points (compensation), which no fixed test controls. This is the (O2) wall, now located exactly at unions of optimal supports.

*Check:* check_removal.out:
- Cost and margin inequalities hold on IF optima and on found optima at (5,2,2.5), (7,2,2.2), (7,2,3) and (11,2,2).
- Maximal violation: +4.4e-16 for unused j at p = 11, i.e. equality: removing an unused feature changes nothing.
- For used features the margin slack is -0.39 to -0.53.

### R13-thmC-upper-half [CONDITIONAL]

THEOREM C. Hypotheses: Theorem B's hypothesis for S = S* (and lambda = 1, S*), and (FE) for the global maximiser theta* at s0 in (1, nu/2].
Then JS := H(sum c_k z_k) - sum c_k H(z_k) <= h_bar - H^(1)_min <= Delta^(1)(s0). Consequently:
(a) If Delta^(1)(s0) = 0, i.e. all optimal single features share one entropy (true except on the countable kink set of log pi, and everywhere where pi is differentiable, e.g. n = 2 where pi = 3^{1-s}): nu E(theta*) = z(y) for ONE programme maximiser y, and S* = spt y. This is (C), with output-energy shares exactly equal to the programme's budget shares.
(b) If s0 < nu/2 and H(theta*) = H*_min(s0), the same holds without Delta^(1) = 0.
REMARKS.
- (FE) is used only to exclude feature sets that are unions of >= 2 optimal supports.
- If M(s0) is a single point and s0 < nu/2, (FE) follows from Theorem B. By Proposition O this happens only when the upper half is vacuous.
- For class-C maximisers the chain rule gives H = H(nu E) + sum_l nu E_l H(theta^[l]). So (FE) reads: the feature sub-networks of a maximiser are, on average, at least as spread as the most concentrated optimal single feature (a coupled entropic-width bound). Shared neurons subtract H(l|i).
- At p = 3, nu E = 1 and (FE) is vacuous. Under the hull, (FE) implies (E), by concavity of H.

**Proof / evidence.**

theta* is a maximiser in N_{S*}. Theorem B in N_{S*} gives nu E = sum c_k z_k with spt z_k inside S*, and H(theta*) <= sum c_k H(z_k) + h_bar. The y_k are programme maximisers by R6(b).
(FE) gives H(theta*) >= H(sum c_k z_k) + H^(1)_min. Subtract.
JS = 0 forces all z_k with c_k > 0 to be equal (strict concavity of Shannon entropy), and distinct maximisers have distinct z. So nu E = z(y) and S* in spt(nu E) = spt y in S*.
(b) H(theta*) is minimal over conv G_W = conv G_D, which forces the representation onto the face {h_l = H^(1)_min}. So h_bar = H^(1)_min and JS <= 0.

*Check:* (FE) slack H - (log nu + H(nu E)) at every optimum found with s > 1 is 0 to 2.1e-14: (5,2) and (7,2), b in {2, 2.2, 2.5}, several weightings. At s = 1 (b = nu) the slack is +0.14: maximisers are not unique there, and the theorem needs s0 > 1. At p = 3 it is vacuous (check_p3.out: nu E_1 = 1.000000, output-DC fraction <= 1.5e-32). check_chain.out: class-C chain rule exact to 8.9e-16. The straddling configurations excluded have JS = 0.42-0.93 (R10).

### R14-endpoint-b2 [CONDITIONAL]

THEOREM D (b = 2, s = nu/2). Hypothesis: (E^lambda_S) a.e. for all S and lambda near 1.
(a) Continuity gives W^lambda_S(nu/2) = D^lambda_S(nu/2). Hence gamma* = pi B_p Gamma_{nu/2}, the lower half of (C), and Corollary U (feature sets are unions of optimal supports) all hold at b = 2.
(b) Theorem B holds at nu/2 in its one-sided form, and Theorem C(a) gives (C) at b = 2 for every maximiser that satisfies (FE), when Delta^(1)(nu/2) = 0.
(c) [PROVED, no hypothesis] For a DC-free network theta at b = 2 and its frequency split:
- the features are equal;
- P(theta) = P(theta_split) and E(theta) = E(theta_split);
- H(theta) <= H(theta_split), strictly if some neuron has two live frequency pieces.
With DC, E(theta) = (P_split/P) E(theta_split).
Consequently, if a maximiser theta_g is a free gluing of a maximiser theta_split (p52 R17), then H(theta_g) < H(theta_split), and (E) at s = nu/2 and (FE) for theta_g can fail. This is why (E) is needed only a.e. (C) for theta_g still follows from (FE) for the class-C maximiser theta_split, because features are preserved.
(d) Every maximiser at b = 2 that is a limit of maximisers at s_k increasing to nu/2 with nu E(theta_k) = z(y_k) (for example under (FE) for b slightly > 2) satisfies (C). Such limits have maximal entropy H*_max(nu/2). Minimal-entropy maximisers at b = 2, such as free gluings, are invisible to continuity.

**Proof / evidence.**

(a) Theorem A(ii) holds on the closed interval; then R6(b) and R9.
(c) SPLIT R1(a),(c): atoms are preserved, and the cost is preserved at b = 2 without DC. With no DC, ||omega_i||^2 = sum_k ||omega_i^(k)||^2, so the piece shares refine the neuron shares; the chain rule gives H(split) = H(theta) + H(k|i) >= H(theta). E_l = sum_i ||P_l w_i||^2/P for both networks.
(d) theta_k -> theta_bar and y_k -> y_bar in M(nu/2) (closed graph), so nu E(theta_bar) = z(y_bar) and S_bar is inside spt y_bar. Theorem B in N_{S_bar} gives spt(nu E(theta_bar)) inside S_bar, so S_bar = spt y_bar. Maximisers along s_k increasing to nu/2 have entropies converging to H*_max (left Danskin).
WHAT THE COUNT AT b = 2 NEEDS EXTRA: (FE) at b = 2 for every maximiser whose split is not a maximiser, and, for those whose split is a maximiser, (FE) for the split. Equivalently, (FE) for class-C maximisers plus (MM_opt) at b = 2 (SPLIT R9).

*Check:* check_b2.out:
- (c) Identity holds to 1.4e-16; DC-free random networks have H(theta) - H(split) <= -0.571.
- Narrow-width searches at b = 2 found no maximiser below the decoupled width: (5,2) width 3-5 reach at most 0.837 of D; (7,2) width <= 9 reach at most 0.980; (5,3) width <= 3 reach about 0.
So free gluings are a theoretical possibility only; none was observed.

### R15-target-theorem [CONDITIONAL]

TARGET THEOREM T, for every odd prime p and every n >= 2. Assume:
(H1) for every S in F and every feature weight lambda in a neighbourhood of 1: (E^lambda_S) at almost every s in [1, nu/2];
(H2)_b (FE) at s = nu/b for every global maximiser;
(H3)_b all optimal single features at s = nu/b have one cost-share entropy (automatic outside the countable kink set of log pi).
Then for every b in [2, nu]:
- from (H1) alone:
  - gamma*_b = pi_{n,b}(p) B_p Gamma_s(p);
  - every global maximiser's feature set contains an optimal support and is a union of optimal supports;
  - for almost every feature weight, the weighted count is exact;
- adding (H2)_b and (H3)_b: every global maximiser theta* uses exactly the features spt y of a programme maximiser y, with nu E(theta*) = z(y).
At p = 3 all hypotheses are vacuous and the conclusions trivial: the theorem gives the PRICE value, not the refuted IF value.

**Proof / evidence.**

Assembly:
- Theorem A (R5) for each lambda gives W^lambda_S = D^lambda_S everywhere.
- R6 gives the value and the lower half; R9 gives the union-of-supports statement; R11 gives the generic count.
- R13 gives the full count.
- R14 covers b = 2: the value and the lower half by continuity; the count from (H2) at b = 2, through the split if shared neurons are free.
Minimality, which each hypothesis is needed for:
- (H1) at lambda = 1 is needed for the value: it is equivalent to Conjecture 20' (R6a).
- (H1) for lambda != 1 is needed for any frequency information: the s-derivative sees only H.
- (H2) is needed for the upper half by Proposition O.
- (H3) is only a single-feature regularity condition, and fails on at most a countable set.

*Check:* See R16.

### R16-hypotheses-on-known-optima [NUMERICAL]

Sanity check of (H1), (H2), (H3) on every known optimum. None is violated.
(i) (H1) in value form: W^lambda/D^lambda = 0.999998-1.000000 from local searches (6-10 starts each) at:
- (5,2), b in {2, 2.2, 2.5, 3};
- (7,2), b in {2, 2.2, 2.5};
- (11,2,2) and (7,3,2) with random lambda;
- (5,3,2.1);
with weights lambda in {1, (1,1.2), (1.2,1), (1,0.85), (1,1.1), (1,1.3), (1,1.15,0.9), (0.9,1,1.1), random within 3%}. The entropy of each optimum equals H^dec,lambda: (E^lambda) holds with equality.
(ii) (H2) holds with equality (|slack| <= 2.1e-14) at every optimum with s > 1, and with slack +0.14 at s = 1.
(iii) (H3): at n = 2, pi = 3^{1-s} is differentiable, and every optimal single feature has H = log 3.
(iv) p = 3, the certified networks n = 11..18 (oned_certified):
- gamma/gamma_IF = 1.063085, 1.193354, 1.183652, 1.456927, 1.617043, 1.595918, 2.011274, 2.211978 (they reproduce the certificates);
- H = log m with m = 8..13, below log nu;
- nu E_1 = 1.000000, with no output DC.
So (E), (FE) and (H1) are all vacuous there, as required: these networks violate the IF entropic width, not the hypotheses.

**Proof / evidence.**

Scripts in /home/user/grok/heracles/conjecture20/unify/E_reduce/:
- check_weighted.py, check_weighted2.py, check_generic.py: weighted smooth-min L-BFGS from the refereed GLOBAL library;
- check_p3.py: margin by exact count-vector enumeration in float.
All runs together take under 5 minutes on 1 core.

*Check:* Outputs: check_weighted.out, check_weighted2.out, check_generic.out, check_p3.out.

### R17-FE-conjecture [CONJECTURE]

(FE) for every odd p, every n >= 2 and every b in (2, nu]: every global maximiser theta* satisfies H(theta*, s) >= H(nu E(theta*, s)) + H^(1)_min(s).
For class-C maximisers, equivalently: sum_l nu E_l H(theta*^[l]) >= H^(1)_min. That is, the per-feature sub-networks of an optimum are, on average, never more concentrated than the most concentrated optimal single feature.

**Proof / evidence.**

Evidence:
- equality at every optimum found (R16);
- vacuous at p = 3;
- equality for every decoupled optimum built from minimal-entropy single features.
It is false in general at b = 2 if a free gluing (shared neurons with harmless mixed modes) is a maximiser; none was found (R14).

*Check:* R16.

## Obstruction
PRECISE GENERAL OBSTRUCTION TO THE UPPER HALF: SYMMETRY PLUS LINEARITY OF FIRST-ORDER DATA.

1. Every value identity available, for all classes N_S, all feature weights lambda and all temperatures s, determines the maximisers only through the convex hull of linear observables: the entropy H for s, and the output-frequency usage nu E for the feature taxes. The hull is conv{(H(theta*), nu E(theta*))} = conv{(H^(1) + H(z(y)), z(y))} over decoupled optima (Theorem B). This is the complete first-order content (Danskin); second-order envelope information gives only one-sided inequalities.

2. The dilation group Z_p^x/{+-1} acts regularly on F. So whenever an optimal support is not all of F (precisely when the upper half has content), the programme maximisers form an orbit of size >= 2. The orbit average z_bar has full support and the same H(z) (Proposition O). A hypothetical "straddling" maximiser with nu E = z_bar and H = H^dec is consistent with every value identity and every derivative identity, and it would use all |F| frequencies. Value identities alone therefore give only:
- the lower half;
- "the feature set is a union of optimal supports" (Corollary U);
- the exact count for generic feature weights (Theorem G).
The perturbation and removal route (subsidy derivative, removal inequality R12) and the s-derivative route ("all maximisers share H* = H^dec at a.e. s") are special cases of this first-order data. A "strict (E)" (equality only for decoupled maximisers) is just the conclusion restated, because the straddler has H = H^dec exactly.

3. The exact residual is NON-STRADDLING: no global maximiser has a feature set equal to a union of >= 2 distinct optimal supports. Equivalently (Corollary S), every maximiser survives some generic small reweighting of the features. The cleanest general condition delivering it is (FE): H(theta*) >= H(nu E(theta*)) + H^(1)_min. Its Jensen-Shannon gap, 0.42-0.93 on the tested orbits, separates straddlers.
- For class-C maximisers (FE) is a coupled per-feature entropic-width bound.
- It is NOT implied by the price problem: the frequency sub-networks of a maximiser are not optimal single features, because they carry interference compensated by other features. So (FE) meets the same wall as (E) itself, now in conditional form (O2).
- Shared neurons lower H by H(l|i). At b > 2 this costs budget strictly (SPLIT R1b), but no known inequality converts budget slack into entropy.

4. Endpoint b = 2. Shared neurons are free (p52 R17), so a gluing changes H while leaving features and E unchanged; (E) and (FE) can fail at s = nu/2 itself. The value, the lower half and Corollary U transfer by continuity. The count needs (FE) for the class-C split together with (MM_opt) at b = 2. Continuity reaches only the maximal-entropy maximisers at b = 2.

5. Kinks of pi (a countable set of b). Straddling is bounded only by JS <= Delta^(1)(s), the jump in the entropy of optimal single features. There, only minimal-entropy maximisers are proved to satisfy (C).

6. (H1), i.e. (E^lambda) near lambda = 1, is itself open. It is the brief's (E) with feature-weighted margins, and every ingredient of the reduction is lambda-uniform. Proving (E) remains the core problem; this task reduces the count to (E^lambda) + (FE) and shows that something of type (FE) cannot be avoided.

## Next steps
- Make every existing proof of (E)-type statements lambda-uniform. The dense-regime theorem, the slack principle and the proven ranges at (5,2) and (7,2) have strictly positive tolerances that depend continuously on the weights. Check that they give W^lambda_S = D^lambda_S for lambda near 1 on the same b-ranges; this upgrades every proven value range to Corollary U and to Theorem G. Because K* = F at (5,2) and (7,2), that is a real test only at larger p.
- Attack (FE) in class C at b > 2 as a conditional entropic-width statement. Show that a feature sub-network theta^[l] of a maximiser with H(theta^[l]) < H^(1)_min cannot be compensated by the other features. Natural tool: delta-Fourier isolation (p52 R12) applied to feature l alone, combined with the identity nu E_l = z_l(y) from Theorem B, which fixes the budget of each feature exactly. This turns (FE) into a single-feature statement with an externally imposed interference budget.
- Alternative to (FE): lambda-stability (Corollary S). Prove that the argmax is lower semicontinuous in the feature weights at lambda = 1, i.e. that every maximiser survives a generic small reweighting. A quantitative second-order (Hessian in mu) condition at maximisers is the natural candidate. The tax Hessian is explicit: -(nu/2) times the derivative of E along mu.
- b = 2: decide whether a free gluing with harmless mixed modes can be a maximiser for p >= 5. By SPLIT R7a, at an IF core the mixed table must vanish on every binding offset. A channel count (SPLIT R4b) at the IF optimum's binding set may exclude it in general. If excluded, (FE) at b = 2 reduces to class C.
- Kinks of pi: prove Delta^(1)(s) = 0, i.e. log pi differentiable, by showing optimal single features have a unique cost-share entropy. At n = 2 this already holds (pi = 3^{1-s}). In general it follows from (Q1), pi = nu^{1-s}, together with KM2 rigidity (the IF block is the unique cost-optimal IF feature, entropy log nu).
- Referee targets in this report:
- the support-function step of Theorem B with the one-sided cone C at s = nu/2;
- the claim that the maximum of the decoupled family Psi equals log D^{mu^{-1/2}} (R7b), with balanced and DC-free maximisers;
- the a.e.-uniqueness argument of Theorem G(b) (semiconvexity in t = log lambda).
The numerical checks are in /home/user/grok/heracles/conjecture20/unify/E_reduce/check_*.out, and the complete proofs in reduce_proofs.tex in the same directory.
