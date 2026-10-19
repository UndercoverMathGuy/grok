# E_dual: report (saved by the orchestrator from the agent's structured output)

## Headline
DUAL-ROUTE (E_dual). No general proof of (E_S) or (EW_S). The duality route gives the following.

(1) An exact Renyi-Legendre form of the temperature problem. Then:
- the free energy is F = log gamma_1 + KL(w_s||w_1), and sup F = +infinity (cancelling dust);
- the fixed-entropy problem is the s=1 problem under an entropy cap;
- decoupled networks are product states with additive Renyi entropy, so W_S = D_S is a multiplicativity statement (Amosov-Holevo-Werner type).

(2) Linear (test-measure) duality, i.e. the Linial-Shraibman / trace-norm / E_q g type, is provably blind to s and to entropy caps. It is exact only at s=1. There (E_F) holds, with D^+ L_F(1) = 0.

(3) (EW_S) for all s <=> (E_S) a.e. <=> W_S == D_S. No entropic weakening of the restricted price-programme identity exists.

(4) Forster / Srebro-Shraibman transfer as an entropic Lemma 7: atom shares are majorised by the spectrum of the chain Gram matrix, and the entropy deficit is bounded by interval interference.

(5) Refuted with a certified network at (5,2,2):
- the all-network Forster-type inequality whose extremals are the decoupled networks;
- the restricted identity W_C = D_C on pinned amplitude-profile classes.

(6) The upper half of (C) reduces to a slack inequality on amplitude-floor classes (two-sided slack principle). Exactness is needed only as the floor eps -> 0, where the tolerance scales like eps^{1/s} (numerical).

Files are in /home/user/grok/heracles/conjecture20/unify/E_dual/. Proofs are in e_dual_proofs.tex; checks are the *.py scripts with their *.out files.

## Principle
THE RENYI-LEGENDRE DUALITY

Notation: a_i = ||omega_i||^nu; p_i = a_i / sum a (nu-cost shares, independent of s); w(s) = escort of p of order 1/s (the cost shares); gamma_1 = gamma(.;1); K_alpha = exp(H_alpha) (Hill number).

Exact form:
  log gamma(theta;s) = log gamma_1(theta) - (s-1) H_{1/s}(p_theta)
                     = min_pi [log gamma_1 + KL(pi||p) - (s-1) H(pi)].
So (s-1) and the Shannon entropy H of the cost shares are Legendre-conjugate. F is the value at sigma = 1 of the tangent line at s: F = log gamma_1 + KL(w_s||w_1).

The programme has exactly the same form over frequencies: Gamma_s = max_yhat M(yhat) K_{1/s}(yhat)^{1-s}. The single-feature price has it over the neurons of one feature: pi(s) = sup e(theta_1;1) K_{1/s}^{1-s}.

Decoupled networks are tensor products, p_Theta = yhat (x) p_{theta_1}, and Renyi entropy is additive on products at every order. Hence:
- log D_S = log B_p + [single-feature Renyi problem] + [allocation Renyi problem];
- (E_S) / W_S = D_S is the statement that coupling never beats product states. Structurally this is the multiplicativity of a maximal Renyi functional under tensor products (Amosov-Holevo-Werner). That conjecture is true for special classes of channels (King) and false in general (Werner-Holevo, Hayden-Winter, Hastings). Here too it fails off the optimum (D8, D9).

WHAT THE DUALITY DOES
- Sorts every network by (s=1 margin, Renyi effective width). A network can beat the decoupled value only by being effectively narrower than its own IF projection: K_{1/s}(p_theta) < nu_pi K_{1/s}(yhat_theta) (D7).
- Makes the s=1 endpoint exact. There all maximisers are IF and H >= log nu + log F (D4').
- Splits (C) into two slack principles: feature classes for the lower half (GLOBAL R6) and amplitude-floor classes for the upper half (D10).

KNOWN-MATHEMATICS LEDGER (searched 2026-10-19)
- Trace-norm / nuclear-spectral duality (Srebro-Shraibman COLT 2005; Friedland-Lim, Math. Comp. 2018: the tensor spectral norm is dual to the nuclear norm).
  - Transfers EXACTLY, but only at s=1: W(1) = nu^{-nu/2} min_q ||E_q g||_sigma = B_p Gamma_1, attained by q uniform.
  - For s>1 it gives only the s=1 value. The convex hull of every l_r ball, or entropy-capped ball, is the nuclear ball.
  - The l_r quasi-norm has dual l_infinity and Banach envelope l_1 (Peetre / Shapiro), so no linear functional sees r < 1 (D3b).
- Linial-Shraibman gamma_2 / max-norm and margin complexity ~ 1/discrepancy.
  - The max-norm charges per-row (per-input) factor norms. Our cost is a sum over hidden units, which is trace-norm type, so gamma_2 is the wrong norm.
  - The mc ~ 1/disc theorem is a linear duality, so D3b applies to it.
  - What transfers is the template "test distribution P gives a lower bound on the factorisation norm". That template is already our Lemma 2 / delta-Fourier tests.
- Forster (sign-rank >= N/||M||, radial isotropic position). Does not transfer.
  - Flattening the label tensor gives matrix rank 2 (cos, sin), while the tensor rank is nu (KM3). Forster bounds matrix dimension, not the number of rank-one terms in a product decomposition.
  - Isotropic rescaling has no analogue for a sum-type cost.
  - The transferable core is "number of terms >= effective rank of a PSD matrix built from them" (Schur-Horn / Gram majorisation). That is D5.
- Gibbs / Donsker-Varadhan and the Renyi derivative of the escort distribution: this is D1. It is standard, so it cannot by itself carry the value content (consistent with known/ L4).

Sources: Linial-Shraibman (www.cs.huji.ac.il/~nati/PAPERS/lcc.pdf); Srebro-Shraibman (www2.mta.ac.il/~adish/Pubs/Papers/SSCOLT05.pdf); Friedland-Lim (ams.org/mcom/2018-87-311); Nie arXiv:1605.08823; Albiac-Ansorena-type Banach envelopes arXiv:1811.01265 and arXiv:2508.00490; Forster / radial isotropic position (ias.edu/node/66406, arXiv:2107.05582); Amosov-Holevo-Werner multiplicativity and counterexamples (arXiv:0807.4753, quant-ph/0203003, quant-ph/0410063).

## Results

### D1 [PROVED]

Renyi-Legendre form. For every network theta with h>0 and every s>0, with nu-cost shares p (s-free) and cost shares w(s):
(i) log gamma(theta;s) = log gamma_1(theta) - (s-1) H_{1/s}(p). Equivalently (sum a_i^{1/s})^s = (sum a_i) K_{1/s}(p)^{s-1}.
(ii) log gamma(theta;s) = min over pi in the simplex of [log gamma_1 + KL(pi||p) - (s-1) H(pi)], attained uniquely at pi = w(s).
(iii) F(theta;s) := log gamma + (s-1) H(w(s)) = log gamma_1 + KL(w(s)||p) = log gamma_1 + psi*(H(w(s))). Here psi(beta) = beta H_{1/(1+beta)}(p) is convex, with psi' = H(w(1+beta)), and psi* is its convex conjugate.
(iv) Gamma_s = max over yhat in Delta_F of M(yhat) K_{1/s}(yhat)^{1-s}, where M(yhat) = min_delta sum_j yhat_j kappa_j(delta).
Reading: the s-dependence of a network's normalised margin is entirely a Renyi entropy of order 1/s of an s-free distribution. The free energy is the s=1 margin plus the KL gap between the shares at s and at 1.

**Proof / evidence.**

(i) sum a^{1/s} = (sum a)^{1/s} sum p^{1/s}, and s log sum p^{1/s} = (s-1) H_{1/s}(p).
(ii) Gibbs principle: log sum e^{x_i} = max_pi [<pi,x> + H(pi)], with unique maximiser pi proportional to e^x. Take x_i = (1/s) log p_i and use sum pi log p = -H(pi) - KL(pi||p). This gives s log sum p^{1/s} = max_pi [(s-1) H(pi) - KL(pi||p)], with maximiser w(s). Insert into (i).
(iii) Evaluate (ii) at pi = w(s). psi(beta) = (1+beta) log sum p^{1/(1+beta)} has derivative H(w(1+beta)) (GLOBAL R1), which is non-decreasing, so psi is convex. At eta = psi'(s-1): psi*(eta) = (s-1) eta - psi(s-1) = (s-1)H(w) - (s-1)H_{1/s}(p) = KL(w||p).
(iv) M is 1-homogeneous, so Gamma_s = max_yhat M(yhat) / (sum yhat^r)^s, and (sum yhat^r)^s = K_r(yhat)^{s-1} by (i).
This contains LARGEN R12 (the participation-number identity for one feature) as the special case of one feature's atoms.

*Check:* check_identities.py -> check_identities.out:
- (A) 200 random cost vectors at 40 digits: max error 3.7e-40.
- (B,C) perturbed IF networks at (5,2),(7,2),(7,3),(11,2), s in {1.2, 1.5, nu/2}: |F - (log gamma_1 + KL)| <= 1.8e-15. The Gibbs value at 30 random pi per case is always >= log gamma (min excess 0.011), with equality at w_s to 1.8e-15.
- (E) the programme Renyi form holds to 4e-16 at (7,1.3),(11,1.45),(11,1.8),(13,1.5).

### D1v [PROVED]

Decoupled networks are product states. Let theta_1 be a single feature, y >= 0 nonzero, and Theta the union of relabelled copies of theta_1 with every weight of copy j multiplied by y_j^{1/nu}. Then:
- p_Theta = yhat (x) p_{theta_1}, so H_alpha(p_Theta) = H_alpha(yhat) + H_alpha(p_{theta_1}) for every order alpha;
- the clean profile of Theta is yhat;
- log D_S(s) = log B_p + sup_{theta_1} [log e(theta_1;1) - (s-1) H_{1/s}(p_{theta_1})] + max_{yhat in Delta_S} [log M(yhat) - (s-1) H_{1/s}(yhat)], where e(theta_1;1) = min_x cm(Phi_1) / (B_p sum a).
In particular pi(s) = sup e(theta_1;1) K_{1/s}(p_{theta_1})^{1-s}.
Reading: W_S >= D_S says product states are feasible. W_S = D_S says no coupled (entangled) state beats the best product state, i.e. multiplicativity of a maximal Renyi functional.

**Proof / evidence.**

Relabelling permutes coordinates, so neuron (j,i) has a = y_j a_i. Hence p_(j,i) = yhat_j p_i, and sum (yhat_j p_i)^alpha = sum yhat^alpha * sum p^alpha. Also |beta_j(Theta)| = y_j |beta_1(theta_1)|. The value formula combines D_S = pi B_p Gamma^(S)_s with D1(i) applied to theta_1 (the cone margin is s-free) and D1(iv) applied to yhat.

*Check:* check_identities.out (E): for the decoupled IF optimum at (p,n,s) = (7,2,1.3), (11,2,1.45), (11,3,1.8), (13,2,1.5):
- H_r(net) - [log nu + H_r(yhat)] <= 9e-16;
- log gamma - [log(B_p M(yhat)) - (s-1) H_r] <= 5.3e-15.

### D2 [PROVED]

Question (a), first part. The free energy has no finite supremum: for every s>1 and every theta with h>0, sup F(theta';s) = +infinity over networks theta' with the same table. Moreover, for every eta and eps>0 there is theta' with the same table, H(theta',s) >= eta and gamma(theta';s) >= (1-eps) gamma(theta;s).
So entropy floors cost nothing, only entropy caps constrain, and the only meaningful restricted problem is U(s,eta) = sup{gamma(theta;s) : H(theta,s) <= eta}.

**Proof / evidence.**

Append M pairs of neurons (u_1,...,u_n,w) and (-u_1,u_2,...,u_n,w). Their tables are T and -T, so the table is unchanged. Give each a^{1/s} = c/(2M).
- Then log gamma = log h - s log(Z_0 + c), independent of M, where Z_0 = sum a_i^{1/s}.
- H >= (c/(Z_0+c)) log(2M(Z_0+c)/c), which tends to infinity.
- The total nu-mass 2M (c/2M)^s tends to 0.
- Choosing (1 + c/Z_0)^{-s} >= 1 - eps gives the second claim.

*Check:* check_identities.out (D): IF optimum at (7,2,s=1.5) with dust of s-mass 5% and M = 10..10^12. log gamma is constant at -5.931; H rises 2.43 -> 3.63 and F rises -4.72 -> -4.11, linearly in log M.

### D3a [PROVED]

Question (a), exact dual of the fixed-entropy problem. For S in F, r in (0,1] and eta >= 0, let Phi_{S,r}(eta) = sup{log gamma_1(theta) : theta in N_S, H_r(p_theta) <= eta}. Then
  log W_S(s) = sup_{eta>=0} [Phi_{S,1/s}(eta) - (s-1) eta],
and the problem at temperature s with fixed Renyi entropy is the s=1 problem (a convex objective with nu-cost) under an entropy cap.
For decoupled networks, Phi^dec is the sup-convolution of the single-feature frontier and the allocation frontier, and its transform is the sum (D1v).
Direction of the bound: 'margin h with l_r cost P needs entropy >= f(h,P)' is equivalent to an upper bound on Phi_r. Note h/P^s = gamma_1 K_r^{1-s}, so the l_r cost is the l_1 cost times K_r^{s-1}.

**Proof / evidence.**

If H_{1/s}(p_theta) = eta, then by D1(i) log gamma(theta;s) = log gamma_1 - (s-1)eta <= Phi(eta) - (s-1)eta. Conversely, if H_{1/s} <= eta then log gamma(theta;s) >= log gamma_1 - (s-1)eta; take the sup over such theta. The decoupled statement is D1v.

*Check:* This is the same identity as D1(i), checked there. The (5,2,2) frontiers computed in D11 use exactly this decomposition.

### D3b [PROVED]

Linear (test-measure) duality is blind to s and to entropy caps. Let q be any probability on Z_p^n x Z_p^*, r in (0,1] and eta >= 0. Then
  sup{<E_q g, f> : f = sum_i mu_i T_i, mu in the simplex, H_r(mu) <= eta, T_i unit rank-one} = ||E_q g||_sigma (the tensor spectral norm).
Hence every bound of the form h <= <E_q g, f> <= (dual norm) x (cost), whether Linial-Shraibman, Srebro-Shraibman or KKT-certificate style, gives at best
  nu^{-nu/2} inf_q ||E_q g||_sigma = B_p Gamma_1 = W_F(1),
whatever s and eta. Yet for p >= 5 and s > 1, W(s) <= B_p Gamma_s < B_p Gamma_1.
At s=1 the duality is exact: W(1) = nu^{-nu/2} min_q ||E_q g||_sigma, with q uniform.
Consequence: for s>1 there is no exact dual in terms of a single dual tensor. Any proof must be two-level: tests reduce the table to finitely many Fourier statistics, and the l_r structure enters only through non-convex 'projected cost' inequalities (single-neuron budget, Lemma 7 and its robust forms).

**Proof / evidence.**

The constraint set contains every unit rank-one (take mu a point mass, so H_r = 0) and lies in the nuclear unit ball. A linear functional has the same supremum on a set and on its closed convex hull, and the hull of the unit rank-ones is the nuclear ball.

Balancing (AM-GM) gives ||theta||^nu >= nu^{nu/2} sum ||T_i||_F. So gamma(theta;s) <= gamma_1(theta) <= h(f) / (nu^{nu/2} N_1(f)) <= nu^{-nu/2} ||E_q g||_sigma for every q.

With q uniform, E_q g = (p/(p-1)) sum_j Re beta_j. The single-neuron budget at r=1 gives ||E_q g||_sigma <= (p/(p-1)) B_p nu^{nu/2}, and IF networks attain B_p Gamma_1.

Strict inequality Gamma_s < Gamma_1 for p >= 5, s > 1: by D1(iv), Gamma_s = M(yhat*) K_r(yhat*)^{1-s}. Either K_r > 1, or yhat* is a point mass with M = kappa_min < p/(p-1) = Gamma_1. At p = 3 there is no gap (Gamma_s = 3/2 for all s); this is consistent with the counterexamples, and no value claim is made.

*Check:* Logical. The s=1 equality is checked in check_entropic.out (3): s=1 optimisations at (5,2),(7,2),(5,3) reach gamma/(B_p Gamma_1) = 1.000000. The strict gap Gamma_1/Gamma_s is visible in every programme value computed (e.g. Gamma_1.5(5) = 0.8839 < 1.25).

### D3c [PROVED]

KKT dual tensor. At every local maximiser of gamma(.;s) with h>0, normalised to unit b-cost, there is a probability q* on the active constraints such that
  t_i := E_{q*} g(u_i1-hat, ..., u_in-hat, w_i-hat) = h nu^{nu/2} a_i^{1/s - 1}  (all factors unit vectors).
Consequently:
- F(theta;s) = E_w log t_i - (nu/2) log nu <= log ||E_{q*} g||_sigma - (nu/2) log nu;
- KL(w_s||w_1) <= log(nu^{-nu/2} ||E_{q*} g||_sigma / gamma_1(theta));
- for s>1 the neurons attain the spectral norm of their own dual tensor only if all neurons have equal size.
At the decoupled IF optimum with unequal feature amplitudes, the larger features' neurons sit strictly below the spectral norm. So the s=1 certificate structure (all rank-ones attain the dual norm) cannot exist at s>1: this answers (b) in the negative for spectral certificates.

**Proof / evidence.**

Write the problem as max t subject to g(x,delta) >= t and sum ||omega_i||^b <= 1. MFCQ holds at every feasible point with h>0, via the direction (-theta, Delta t = -nu h - 1): active g - t increase at rate 1, and the cost decreases at rate b. So KKT multipliers exist: q >= 0 with sum q = 1 on active constraints, and grad_{omega_i} E_q g = lambda b ||omega_i||^{b-2} omega_i.

Pairing with each factor of neuron i gives equal factor norms ||omega_i||/sqrt(nu) and E_q g(omega_i) = lambda b ||omega_i||^b / nu. Summing gives h = lambda b / nu. Multilinearity gives E_q g(omega_i) = (||omega_i||/sqrt(nu))^nu t_i, so t_i = h nu^{nu/2} ||omega_i||^{b-nu} = h nu^{nu/2} a_i^{r-1}.

At unit cost w_i = a_i^r, so E_w log t = log h + (nu/2) log nu + (s-1) H(w) (as in BRIEF_E sec.2). Finally t_i <= ||E_{q*} g||_sigma; combine with D1(iii). a -> a^{r-1} is injective for r < 1.

*Check:* check_kkt.py -> check_kkt.out: decoupled IF optima at (7,2,1.3), (11,2,1.5), (7,3,1.8), (13,2,1.4), with q = uniform(x) x programme-dual mu* restricted to the binding offsets:
- stationarity residual <= 1.5e-15;
- max |t_i / (h nu^{nu/2} a_i^{r-1}) - 1| <= 6.4e-15;
- F - [E_w log t - (nu/2) log nu] <= 5.3e-15;
- t_i/max t ranges 0.77-1.00 when amplitudes are unequal: the spectral norm is not attained by all neurons.
A first run without complementary slackness (mu on non-binding offsets) violated the identity, as it must.

### D4 [PROVED]

Logical status of the target. For every S in F:
  (EW_S) for all s in [1,nu/2]  <=>  (E_S) for a.e. s  <=>  W_S == D_S on [1,nu/2].
When these hold, every maximiser in N_S has H = H^dec_S(s) at a.e. s (equality, not merely >=).
So the 'entropic width theorem' (EW_S) has no content beyond the restricted price-programme value identity. There is no weaker entropic statement to prove: any proof of (E_S) proves W_S = D_S. The target entropy H^dec involves the optimal single features, which are unknown for n >= 3.

**Proof / evidence.**

W_S == D_S => (EW_S): gamma <= W_S = D_S for every theta.

(EW_S) for all s => W_S == D_S: on {L_S > 0} every maximiser has H >= H^dec_S, since otherwise gamma = W_S <= D_S. So D^+L_S <= 0 there; with L_S(1) = 0 and continuity, L_S == 0 (BRIEF_E sec.1, the Dini-derivative argument of GLOBAL R3).

(E_S) a.e. => W_S == D_S: BRIEF_E with the Vitali-Caratheodory fix (GLOBAL VERIFY R4).

W_S == D_S => (E_S) a.e., with equality: log W_S is locally Lipschitz (its slopes are entropies of width-bounded networks, GLOBAL R2), hence differentiable a.e. (Rademacher). At such s all maximisers have H = -(log W_S)'. Every decoupled network built from an optimal single feature and a maximiser of Gamma^(S)_s attains D_S = W_S, so it is a maximiser and has the same entropy.

*Check:* Logical. Consistent with p=3, where W = D trivially (F = {1}, decoupled = single-feature optimum), so (E) is vacuous. The p=3 networks have H = log m < log nu (check_p3_gain.out: m = 8..13 equal neurons).

### D4' [PROVED]

(E_F) holds at s=1 (b = nu). Every maximiser theta of gamma(.;1), of any width and with arbitrary inputs:
- is interference-free, frequency-pure, DC-free and balanced;
- has beta_j = 2h/p for all j in F, and real positive atoms;
- has H(p_theta) >= log nu + log F = H^dec_F(1), with equality iff each frequency is carried by exactly nu neurons of equal cost.
Consequently D^+(log W_F - log D_F)(1) = 0, i.e. L_F(s) = o(s-1). This is the s->1 'minimum-output-entropy additivity' analogue of the multiplicativity conjecture, and it is a theorem.

**Proof / evidence.**

Chain of inequalities: h <= E_{x,delta uniform} g = (p/(p-1)) sum_j Re beta_j <= (p/(p-1)) sum_{i,j} |beta_ij| <= (p/(p-1)) B_p sum a_i, and W(1) = B_p Gamma_1 = (p/(p-1)) B_p. So every step is an equality.

(1) g == h. For fixed x, phi_x(c') = f(x)[y - c'] - fbar(x) has mean 0 and phi_x(0) - phi_x(delta) = h. Hence phi_x = h(1_0 - 1/p) for every x, so Phi_l == 2h/p: the network is IF with beta_l = 2h/p.
(2) sum_j |beta_j| = sum_ij |beta_ij| forces all atoms of frequency j to have the phase of beta_j (real positive).
(3) Equality in the r=1 single-neuron budget forces single-frequency, DC-free, balanced neurons with |beta_ij| = B_p a_i. The budget chain is: Hoelder with nu factors; ||.||_nu <= ||.||_2, which is strict unless each factor has one frequency, since nu >= 3; Parseval without DC; AM-GM.
(4) D5 (aligned, IF) gives collision <= 1/nu for the atom shares within each frequency, so H(p|j) >= H_2 >= log nu. The frequency shares are 1/F. The chain rule gives H(p) >= log F + log nu, with equality iff each conditional is uniform on nu atoms.

H^dec_F(1) = log nu + log F:
- Gamma_1 has the unique maximiser 1/F (GLOBAL VERIFY R7: C^2 = (p/4)I - J/2).
- An optimal single feature at s=1 has min cm(Phi_1) = Re beta_1 = B_p sum a. Since cm is concave and cm <= Re, this forces Phi_1 == beta_1 (IF, aligned), so H >= log nu, with equality for the paper's block.

Finally:
- Danskin at the endpoint gives D^+ log W_F(1) = -(log nu + log F).
- log gamma(Theta_IF;s) <= log gamma_IF(s) <= log D_F(s), with equality at s=1, gives D_+ log D_F(1) >= -(log nu + log F).

*Check:* check_entropic.out (3): 12 gamma(.;1) optimisations from Gaussian starts at (5,2), (7,2), (5,3), widths 8-10. Every run reaching gamma/(B_p Gamma_1) = 1.000000 has H(p) >= log(nu F):
- 1.80-2.02 vs 1.7918 at (5,2);
- 2.205-2.224 vs 2.1972 at (7,2);
- 2.0794-2.162 vs 2.0794 at (5,3).

### D5 [PROVED]

Entropic Lemma 7: the transferable Forster / Srebro-Shraibman mechanism. Setting: atoms beta_l != 0 with |xi_lt| = 1; L = sum |beta_l|; A = sum beta_l; beta_T = sum_l beta_l prod_{t in T} conj(xi_lt); beta_I for the binomial(nu,2) intervals I = {k+1..k'}, 0 <= k < k' <= n. Then
  sum_l (|beta_l|/L)^2 <= 1/nu + (2/nu^2) sum_I (|beta_I|/L + sqrt(2(1 - |A|/L)))^2.
For aligned atoms (all with the phase of A) the last term is (2/nu^2) sum_I |beta_I|^2/L^2.
Moreover (|beta_l|/L)_l is majorised by spec(M)/(nu L), where M = sum |beta_l| m_l m_l^* and m_l = (1, conj xi_l1, conj xi_l1 conj xi_l2, ...). So every Schur-concave entropy of the atom shares (Shannon, every Renyi order) dominates that of the spectrum.
If the atoms are aligned and IF:
- H_alpha >= log nu for all alpha <= 2, with equality iff exactly nu equal atoms with orthogonal chain vectors (the KM2 configuration);
- sum |beta_l|^r >= nu^{1-r} |A|^r, which is Paper Lemma 7 for aligned atoms, by a Frobenius instead of a determinant argument.
Reading: the effective number of atoms of a feature is at least nu divided by (1 + normalised interval interference + phase misalignment).

**Proof / evidence.**

M is PSD, and ||M||_F^2 = sum_{l,l'} |beta_l||beta_l'| |<m_l,m_l'>|^2 >= nu^2 sum |beta_l|^2 (drop the terms with l != l'; ||m_l||^2 = nu).

The entries are M_kk = L, and for k' > k, M_{k'k} = sum_l |beta_l| prod_{t in I} conj xi_lt =: beta-tilde_I. With phi = arg A and psi_l = arg beta_l - phi, Cauchy-Schwarz gives
  |beta-tilde_I - e^{-i phi} beta_I| <= sum |beta_l| |1 - e^{i psi_l}| <= sqrt(L) (sum |beta_l| 2(1 - cos psi_l))^{1/2} = sqrt(2L(L - |A|)).
Hence ||M||_F^2 <= nu L^2 + 2 sum_I (|beta_I| + sqrt(2L(L-|A|)))^2; divide by nu^2 L^2.

Majorisation: G = V*V with V = [sqrt(|beta_l|) m_l] has diagonal nu|beta_l| and the same nonzero spectrum as M = VV*; apply Schur's theorem.

Equality: collision 1/nu forces <m_l, m_l'> = 0 for all l != l', so N <= nu, and Cauchy-Schwarz then forces nu equal atoms.

Lemma 7: sum |beta|^r = L^r K_r^{1-r} >= L^r K_2^{1-r} >= L^r nu^{1-r} (Hill numbers decrease with the order, r <= 2), and L = |A| for aligned atoms.

*Check:* check_entropic.out (1): 20000 random atom systems, n = 1..6, nu to 3nu atoms: aligned, random phases, near-IF (KM2 blocks plus noise), and exact KM2.
- max(lhs - rhs) = 4.4e-16 (general bound);
- 2.2e-16 (aligned bound);
- |collision - 1/nu| at KM2 = 5.6e-17 (equality).

### D6 [PROVED]

SPECIAL CASE, reported and not pursued further. Aligned class-C single features. Let p >= 5 and theta be a class-C single-frequency single feature (frequency 1, no DC) whose atoms share one phase. Let c = min_x cm(Phi_1(x)) and tau_p = (1 - cos(2pi/p)) / cos(2pi/p). Then:
(a) sum_{T != empty} |beta_T|^2 <= tau_p (Re A - c)^2;
(b) |A| - nu^{1-s} Q^s <= (2(s-1)/(nu |A|)) sum_I |beta_I|^2, so the cheapness gain of aligned atoms is second order in interference;
(c) if 2(s-1) tau_p <= nu, then c <= nu^{1-s} Q^s: no such feature beats the IF price. This holds for all n and b when p >= 7 (tau_p <= 1), and for p = 5 when s - 1 <= 0.2236 nu.
This is an alignment-restricted, parameter-conditioned statement. Per the rules it is reported only to locate the obstruction (see D6n).

**Proof / evidence.**

Phi_1(x) = sum_T conj(beta_T) z^T with z_t = rho^{2x_t} ranging over mu_p^n. The monomials z^T are orthonormal, and E(Phi - E Phi)^2 = 0 because the exponents lie in {0,1,2}.

(a) Phi - c lies in K_p, so Re (Phi - c)^2 >= cos(2pi/p) |Phi - c|^2 (2pi/p < pi/2 for p >= 5). Hence E|Phi - c|^2 <= (X^2 - Y^2)/cos(2pi/p) with X + iY = E Phi - c, while E|Phi - c|^2 = X^2 + Y^2 + sum_{T != empty} |beta_T|^2.

(b) D5 (aligned) gives K_2 >= nu / (1 + 2R_I^2/(nu L^2)), with R_I^2 = sum_I |beta_I|^2. By D1(i) for the atoms (LARGEN R12), nu^{1-s} Q^s = (K_r/nu)^{s-1} L >= L (1 + 2R_I^2/(nu L^2))^{1-s} >= L - 2(s-1) R_I^2/(nu L), by Bernoulli. Also L = |A|.

(c) If c <= 0 there is nothing to prove. Otherwise, by (a) and (b), nu^{1-s} Q^s >= |A| - (2(s-1)tau_p/nu)(|A| - c)^2/|A| >= c, since 0 <= |A| - c <= |A|.

At p = 3, cos(2pi/3) < 0 and no statement is made, consistent with the counterexamples.

*Check:* check_aligned.py -> check_aligned.out: 4000 random aligned near-IF single features, p in {5,7,11,13}, n = 2..5 with p^n <= 20000, 419 with positive margin.
- (a): max excess -1.4e-5;
- (b) collision bound: -1.2e-11;
- (c): max c/(nu^{1-s} Q^s) - 1 = -0.073 over b in {2, 2.5, nu - 0.5} where 2(s-1)tau_p <= nu.

### D6n [NUMERICAL]

The alignment hypothesis in D5/D6 is an artefact of the Frobenius / Schur-Horn method. Near the KM2 block the cheapness gain g = |A| - nu^{1-s} Q^s is SECOND order in the interference R for ALL phases. Far from IF it is linear: a single atom has g/R = (1 - nu^{1-s})/sqrt(2^n - 1).

**Proof / evidence.**

gain_rays.py -> gain_rays.out: 4000 random rays from the KM2 block at each (n,s) in {(2,1.25), (2,1.5), (3,1.25), (3,2)}, aligned and misaligned perturbations. max g/R scales like eps in every case:
- n=2, s=1.5, misaligned: 2.4e-2, 2.7e-3, 2.6e-4, 3.0e-5 at eps = 1e-1..1e-4;
- aligned: 2.9e-2 ... 3.3e-5.
gain_order.out (constrained optimisation at R/|A| = 1e-1, 1e-2, 1e-3; partial run): aligned max g/R = 6.8e-3, 6.7e-4, 6.7e-5; misaligned 7.0e-3 at 1e-1.

*Check:* Evidence only. The Frobenius bound's phase term sqrt(2(1 - |A|/L)) is first order in the phase spread, while the true effect is second order. That is why D6 needs alignment.

### D7 [PROVED]

Static entropic gain bound, for all networks and all p, n, b. For every theta with h>0 and clean profile yhat_theta = |beta|/sum|beta|:
  gamma(theta;s) <= B_p M(yhat_theta) K_{1/s}(p_theta)^{1-s}.
Hence, with nu_pi(s) := pi(s)^{-1/(s-1)} in [1, nu],
  gamma/(pi B_p Gamma_s) <= (nu_pi K_{1/s}(yhat)/K_{1/s}(p))^{s-1} * M(yhat) K_{1/s}(yhat)^{1-s}/Gamma_s,
with the last factor <= 1. Replacing nu_pi by nu gives the same bound for gamma/gamma_IF.
So a network can beat D_F (resp. gamma_IF) only if K_{1/s}(p_theta) < nu_pi K_{1/s}(yhat_theta) (resp. < nu K_{1/s}(yhat)): it must be effectively narrower than its own decoupled IF projection.
This is a static, all-network, pointwise-in-s form of GLOBAL R3/R4 ('coupling can win only by concentrating') and the multi-feature form of LARGEN R12. It is not a value claim.

**Proof / evidence.**

Paper Lemma 2 gives h <= sum_j |beta_j| kappa_j(delta) for all delta, so h <= M(yhat) sum_j |beta_j| <= M(yhat) B_p sum a_i (single-neuron budget at r = 1, valid for every neuron and every b). Hence gamma_1 <= B_p M(yhat). Apply D1(i) and D1(iv).

*Check:* - check_entropic.out (2): 66 positive-margin networks (perturbed IF, random supports, unequal neuron sizes) at (5,2), (7,2), (5,3), (7,3), (11,2), b in {2, 2.5, nu - 0.3}: max lhs/rhs = 0.678 <= 1.
- check_p3_gain.out: the certified p=3 networks (n = 11..18) have gamma/gamma_IF = 1.06..2.21 against bounds (nu/K_r)^{s-1} = 7.6..25.7. K_r = 8..13 < nu, as the bound requires of any winner.

### D8 [REFUTED]

Question (c). The natural all-network Forster-type inequality with decoupled extremals is false:
  H(theta,s) >= eta^dec(gamma(theta;s)) for all theta in N_S,
where eta^dec(g) is the least entropy of a decoupled network (relabelled scaled copies of an optimal single feature, with any allocation y on S) having gamma >= g.

**Proof / evidence.**

Counterexample at (p,n,b) = (5,2,2), s = 3/2, S = F = {1,2}.

The network is SPLIT R10's certified violator theta_v (width 6), with weights rounded to 12 digits. Its margin is exact rational (split/certify_viol.out): gamma = 0.790214100 gamma_IF. Its cost-share entropy, computed from the exact rational neuron norms at 50 digits, is H = 1.559104.

Decoupled side at n=2, p >= 5:
- Note Thm q1n2's chain min cm <= Re beta_1 - cos(pi/p) max_T|beta_T| <= 3^{1-s} Q^s, with cos(pi/p) > 1/2, forces every optimal single feature to be IF. Lemma 7 equality (r < 1) then forces the KM2 block.
- So decoupled networks are two-block IF networks with y2/y1 = t, value R(t) gamma_IF with R(t) = (kappa_a + t kappa_b)(1 + t^{2/3})^{-3/2}/Gamma on [0,1] (t > 1 is the mirror case), and entropy log 3 + H(z(t)).
- With u = t^{1/3}, d log R/dt = kappa_b/(kappa_a + u^3 kappa_b) - 1/(u(1+u^2)), which vanishes only at u = kappa_a/kappa_b. So R decreases on [0, 0.0557] and then increases.
- R(0) = 0.781758 < 0.790214, so the level is first reached at t0 = 0.2693156. H(z(t)) is increasing on [0,1].
- Therefore eta^dec(0.790214) = log 3 + H(z(t0)) = 1.704571 > 1.559104 = H(theta_v).

What survives: any f that is monotone and <= log 3 on (0.7818, 0.7902] is consistent. Only the exact decoupled frontier (which jumps from log 3 to 1.70) fails. So an entropy lower bound of Forster type, with decoupled extremals, can hold only at the optimal level gamma = W, which is (E) itself (D4).

*Check:* certify_E.py -> certify_E.out (50-digit; exact-rational inputs).

Numerical corroboration over decoupled networks built from ARBITRARY (non-optimal) single features of widths 2-6: frontier.py (entropy-capped penalty optimisation, 6 starts per width and cap) -> dec_p5n2b2.out.
- At caps 1.30, 1.45, 1.559 the best is 0.781758 (one IF block, H = log 3).
- At cap 1.704 the best is 0.789771.

### D9 [REFUTED]

The restricted price-programme identity W_C = D_C := pi B_p max_{yhat in C} M(yhat) K_{1/s}(yhat)^{1-s} is false for amplitude-profile classes C = {theta : |beta|/sum|beta| in C} pinned away from the optimal profiles. Decoupled networks are extremal only near the optimum: coupling (mixing) wins on pinned profiles.

**Proof / evidence.**

At (5,2,2), s = 3/2: theta_v has clean profile yhat = (0.81142854, 0.18857146), computed by 50-digit exact DFT of the rational weights. Its gamma/gamma_IF = 0.7902141 exceeds D_{yhat}/gamma_IF = M(yhat) K_r(yhat)^{1-s}/Gamma_s = 0.7773209, by 1.66%. Here pi_{2,b}(5) = 3^{1-s} by Note Thm q1n2.

The same holds for the closed neighbourhood {|yhat_1 - 0.8114| <= eta} with eta = 0.01 (D_C = 0.78268) and eta = 0.02 (D_C = 0.78819).

The other two SPLIT violators behave the same way (pinned_profile.out): 0.785339 > 0.77199 and 0.776161 > 0.76344.

Feature classes N_S (C = Delta_S) are NOT refuted. At (5,2), W_F = D_F is the p52 theorem, and W_{1} = D_{1} numerically (GLOBAL R12).

*Check:* certify_E.out, pinned_profile.out. The margin is exact (split/certify_viol.out). The profile and D_C are computed in 50-digit arithmetic from exact rationals; the slack of 1.66% is far above rounding.

### D10 [PROVED]

Upper half of (C) through amplitude-floor classes: a two-sided slack principle.

For closed C in Delta_F let A_C = {theta : h>0, yhat_theta in C}. This class is table-defined: invariant under width reduction and scaling, and independent of s. Let W_C = sup over A_C of gamma, Gamma^C_s = max_{yhat in C} M K_{1/s}^{1-s}, and D_C = pi B_p Gamma^C_s.

(a) D_C <= W_C for every s, and W_C(1) = B_p Gamma^C_1 = D_C(1). So the continuation framework of BRIEF_E applies verbatim to profile classes.

(b) Fix s at which all programme maximisers have K* features. Let K be the set of optimal supports, and assume the lower half. Then every global maximiser has its feature set in K iff
  W_{C(S,eps)}(s) < W(s) for all eps in (0,1] and all S strictly containing some member of K,
where C(S,eps) = {yhat_k >= eps for all k in S}.
For such S, tau(S,eps) := Gamma_s / Gamma^{C(S,eps)}_s - 1 > 0, and it suffices that W_{C(S,eps)} <= (1+tau) D_{C(S,eps)} with tau < tau(S,eps).

By D9, exact identities on such classes can fail far from the optimum. So the upper half must be proved through this slack form, with exactness needed only as eps -> 0.

**Proof / evidence.**

(a) Decoupled networks with allocation y proportional to yhat in C lie in A_C (D1v), and gamma >= pi B_p M(yhat) K_r(yhat)^{1-s} (Companion Thm pp; pi is attained, GLOBAL R2). At s = 1, gamma_1 <= B_p M(yhat_theta) <= B_p Gamma^C_1 (proof of D7), attained by IF networks; pi(1) = 1.

(b), direction 'iff' from right to left: a maximiser with feature set S' not in K contains some K in K, hence S' strictly contains K. Then theta lies in A_{C(S', eps)} with eps = min_{S'} yhat_k > 0, so W_C >= W.

(b), the other direction: if W_{C(S,eps)}(s) = W(s), the sup is attained (closed table-defined class; GLOBAL R2 compactness; h bounded below on near-maximisers keeps sum|beta| > 0). The attaining network is a maximiser whose feature set contains S, which has more than K* elements, so its feature set is not in K.

tau > 0: C(S,eps) is closed and contains no programme maximiser, whose support would have to contain S.

Sufficiency: W_C <= (1+tau) pi B_p Gamma^C < pi B_p Gamma_s <= W (price-programme theorem).

*Check:* Logical, plus the tolerance table (floor_tolerance.py -> floor_tolerance.out): tau(S,eps) for S = (optimal support) + {j}.
- p=11, s=1.5: tau ~ 1.2e-4, 2.4e-3, 1.0e-2 at eps = 1e-6, 1e-4, 1e-3;
- p=13, s=2: 4.2e-3, 4.2e-2, 0.13;
- p=11, s=2.5: 6.6e-3, 4.2e-2, 0.106;
- p=17, s=1.5: 1.0e-4, 2.2e-3, 9.8e-3.
So tau ~ c eps^{1/s}: small extra features are expensive at rate eps^{1/s}. Some reweighted-LP entries at larger eps (e.g. p=11, eps=0.01) are not monotone; they are upper bounds, and true tau is non-decreasing in eps. Near the p=11 transition (s_c = 1.4895) tau stays <= 0.4% even at eps = 0.05, matching GLOBAL R6's vanishing tolerance at transitions.

### D11 [NUMERICAL]

Entropy-margin frontiers at (5,2,2), s = 3/2.
- Decoupled networks from arbitrary single features (widths 2-6) never beat the two-block IF frontier.
- Lagrangian scans (maximise log gamma - lam H) only see the concave hull of the frontier. Points like theta_v (H = 1.559, 0.790) lie far below it: the hull at H = 1.559 is about exp(-0.083).
- The local all-network searches under an entropy cap did not even recover the one-block IF network: widths 6, caps 1.40-1.56 gave at most 0.683.
Conclusion: all-network entropy frontiers are non-concave and badly found by local search, so counterexample search for (EW) at the optimum requires seeded searches.

**Proof / evidence.**

frontier.py (modes dec/all, Lagrangian lam >= 0, or entropy cap lam < 0 via quadratic penalty) -> dec_p5n2b2.out and console runs recorded above. A few cap runs failed to enforce the cap (reported H = log 6 at caps 1.65/1.70); they are discarded.

*Check:* Evidence only.

## Obstruction
WHERE THE DUAL ROUTE STOPS, FOR ALL p, n, b

1. There is nothing weaker to prove (D4). (EW_S) for all s, (E_S) a.e., and W_S == D_S are equivalent, and at a.e. s the maximiser entropy equals H^dec exactly. The target entropy H^dec involves the optimal single features, which are unknown for n >= 3, p >= 5 (O3/O4). Any entropic proof is therefore a proof of the restricted price-programme value identity, including the single-feature price.

2. Linear duality is blind to the temperature (D3b).
- Every test-measure certificate h <= <E_q g, f> sees the l_r ball, and every entropy-capped ball, only through its convex hull, which is the nuclear ball.
- This is the quasi-Banach fact that l_r (r < 1) has dual l_infinity and Banach envelope l_1. Margin-complexity / gamma_2 / trace-norm duality (Linial-Shraibman, Srebro-Shraibman) therefore yields only the s=1 value B_p Gamma_1. The gap is Gamma_1/Gamma_s > 1 for p >= 5.
- At s>1 even the KKT dual tensor of the decoupled optimum is not attained by all its neurons (t_i proportional to a_i^{r-1}, D3c), so no spectral certificate exists.
- Any proof must be two-level: tests reduce to finitely many Fourier statistics, and non-convex projected-cost inequalities carry the l_r structure. That is exactly the master-inequality / ledger architecture, so the dual route reduces to it rather than bypassing it.

3. Entropy-from-margin needs global optimality (D8, D9).
- The natural all-network Forster-type inequality with decoupled extremals is false: certified at (5,2,2), H = 1.559 < 1.705 at the same gamma.
- The restricted identity also fails on pinned amplitude profiles (1.66% excess).
- So decoupled networks are extremal only at or near optimal profiles. Any valid inequality must use gamma = W, or approximate decoupling with a tolerance, as in the slack principles.

4. The Forster / Schur-Horn mechanism controls only aligned atoms (D5, D6, D6n).
- The effective number of atoms is bounded below by an effective rank (Gram / Schur-Horn), with deficit bounded by interval interference. This is quadratic for aligned atoms, and combined with circularity it gives (Q1) for aligned class-C single features when 2(s-1) tau_p <= nu.
- But phase misalignment enters the Frobenius bound at FIRST order in the phase spread, while the true gain is SECOND order (numerical). Only the determinantal Lemma-7 mechanism sees phases exactly.
- Its robust version (E_k-perturbation) controls the entropy deficit only LINEARLY in each of the binomial(nu,2) interval coefficients, times junk entries up to L/|A|.
- The available margin detections are either quadratic in aggregate (circularity, constant tau_p; fine) or linear per coefficient with p- or n-dependent constants (delta-Fourier isolation: about 1/2 per coefficient, phase-blind; Rogosinski: (4/p) cos(pi/p)^{-k}).
- Matching them requires p >= p_0(n) or s <= s_0, which is forbidden. This is O3/O4 rediscovered from the entropy side.

5. Cross-feature hiding is untouched.
- D7 shows that a network can beat D only by being effectively narrower than its IF projection, i.e. by carrying a cheap feature.
- Whether the cheap feature's interference can be hidden by other features' modes (F2 shared neurons at b=2, relations 2l = +-j +- q) is O1/O2.
- The dual route supplies no detection that is blind to those channels beyond R12. Its tests force mass on every offset and so detect only max_T |beta_T|.

6. Upper half (D10). It reduces to approximate decoupling on amplitude-floor classes. The tolerance is positive and scales like eps^{1/s}, so exactness is needed only for vanishing extra features, which is the first-order 'no small extra feature' problem (NS). At b=2 this involves the by-products of multi-frequency neurons, which no known inequality charges.

## Next steps
- Determinantal robust Lemma 7 with quadratic error. Prove sum |beta_l|^r >= nu^{1-r}|A|^r - C(n,s) R^2/|A|^{2-r} near the IF manifold for ALL phases (D6n says the truth is quadratic). Route: perturb det(I + tH/A) = sum_k t^k E_k with lower-triangle entries beta_I/A; show the first-order terms cancel after integrating against t^{-1-r}dt, using the KM2 Fourier structure (Z_nu-diagonalisation). Combined with circularity (R^2 <= tau_p (Re A - c)^2), this gives local IF-price rigidity for class-C single features at every p >= 5, n, b. Test numerically first whether C(n,s) is bounded uniformly in n. This is the only part of the dual route that could become general.
- Two-sided slack programme. Compute tau(S,eps) (D10) exactly by vertex enumeration of the floor-constrained coverage polytope. Combine it with existing approximate-decoupling bounds (Note's dense theorem, GLOBAL R6 / R12, p52 ledgers) to close the upper half wherever the lower half is closed. At (5,2) and (7,2) the upper half is vacuous (K* = F). The first test case is a K* < F regime such as p = 11, n = 2 near s = 1.5. That is a computer-assisted check at one parameter, so it is a sanity check only, not a proof route.
- Use D7 as a filter in all counterexample searches for (C) and the value conjecture. Any winner must have K_{1/s}(p_theta) < nu_pi K_{1/s}(yhat_theta). Seed searches with entropy-capped penalties (frontier.py mode 'allv' seeds from known violators) rather than Lagrangian scans, which miss the non-concave frontier (D11).
- Explore the multiplicativity analogy (Amosov-Holevo-Werner / King) seriously. King's proof for entanglement-breaking channels uses a factorisation of the 'classical' side. The analogue here is class C, where the cost is additive across frequencies (KM1) and coupling is only through min_x of a sum of fields with disjoint interference character sets. Try to show that the margin functional, restricted to class-C tables whose interference characters are dissociated, is multiplicative in the AHW sense. Relations (O1) are the analogue of entanglement and must be handled separately.
- Do not pursue (i) sup-of-F duals (D2), (ii) single-dual-tensor certificates at s>1 (D3b, D3c), (iii) all-network or pinned-profile entropy/decoupling inequalities (D8, D9), or (iv) Frobenius / Schur-Horn entropy bounds for misaligned atoms (D6n). Each is provably blind or refuted.
