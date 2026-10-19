# E_kkt: report (saved by the orchestrator from the agent's structured output)

## Headline
KKT route to (E_S): not proved. What is proved holds for all p, n and b. (1) A KKT calculus: every maximiser is balanced, each neuron's unit tuple is a singular tuple of the dual multilinear form E_q g, and the critical values obey t_i = nu^{nu/2} gamma w_i^{1-s}; so entropy is a function of the critical values, H = a E_w log(t/nu^{nu/2}gamma) with a = 1/(s-1). (2) Co-maximisers: decoupled IF unions on any S have an x-uniform KKT measure p^{-n}mu*, with t = nu^{nu/2} B_p K_j(mu*). Both cross-duality inequalities between co-maximisers hold for every network, so they carry no information. (3) Departures need cheap features (b>2): a maximiser on N_S that is IF, or has no cheap feature, is a union of KM2 blocks with a programme allocation and satisfies (E_S). So a jump departure of L_S needs a co-maximiser with a cheap feature. (4) A second-order departure lemma in s. (5) When the optimal allocation is uniform on a proper support, the decoupled optimum's own x-uniform dual is maximised by non-feature frequencies (numerically also in all 9 non-uniform cases tested), so dual optimality cannot give the upper half of (C) for s>1. Obstruction: the KKT relations bound entropy only from above, and co-maximality adds nothing beyond Lemma 2. Entropy-dropping level crossings really occur between decoupled branches (p=7, n=3, s_c=1.5475). The residual is exactly a value bound for concentrated networks with a cheap feature, which is equivalent to (E_S); at n=2 it contains the relations core O1.

## Principle
At a KKT point the critical values are the cost shares in disguise: t_i = nu^{nu/2} gamma w_i^{1-s}. So H = a E_w log(t_i/(nu^{nu/2}gamma)), and at co-maximisers the entropy order equals the order of the w-weighted mean log critical values. The only inequality KKT adds, t_i <= tau(q) (injective norm of the dual tensor), bounds entropy from ABOVE. Every LOWER bound on entropy must come from the margin, and the one general mechanism is Lemma 7/cheapness. For aligned, budget-tight features, cheapness = exp((s-1)(log nu - H_s)) exactly, with H_s the Renyi entropy of order s of the within-feature cost shares. With global optimality this becomes: a maximiser with no cheap feature is a decoupled IF union (entropy >= H^dec). Hence a departure from the decoupled value must be a concentrated, cheap, interfering co-maximiser. Its interference lowers the critical values (sum_i rho_i^nu iota_i = -Det_mu, linear detection in class C), but comparing that loss with the cheapness gain is the payment-versus-detection value inequality, which KKT cannot supply.

## Results

### R1 [PROVED]

KKT structure theorem. Valid for any odd p, n>=2, b in [2,nu], s=nu/b. Let theta=(omega_i)_{i<=m} be a local maximiser of gamma(.;s) among networks of its width (in particular any global maximiser), with h>0 and nonzero neurons; factors v_it (t=1..n inputs, t=0 output). For a probability q on A=Z_p^n x Z_p^*, put T_q(omega)=E_q[g_omega], which is multilinear in the nu factors, and tau(q)=max of T_q over tuples of unit factors (the injective norm of the dual tensor M_q[x,c]=sum_delta q(x,delta)([c=y(x)]-[c=y(x)-delta])).
(a) There is a probability q on argmin g with sum_a q_a grad g_a = (s h/P) grad P.
(b) Every neuron is balanced, ||v_it||^2=||omega_i||^2/nu, and T_q(omega_i)=h w_i>0. (Cost shares equal margin shares.)
(c) The unit tuple u^_i is a singular tuple of T_q: grad_t T_q(u^_i)=t_i u^_it for every factor t, with singular value t_i=T_q(u^_i)=nu^{nu/2}(h/P)a_i^{-(s-1)/s}. Equivalently, the CRITICAL-VALUE LAW t_i = nu^{nu/2} gamma w_i^{1-s}.
(d) For s>1, with a=1/(s-1): w_i=t_i^{-a}/sum_j t_j^{-a}; gamma=nu^{-nu/2}(sum_i t_i^{-a})^{-(s-1)}; log(nu^{nu/2}gamma)=E_w log t-(s-1)H; F=log gamma+(s-1)H=E_w log t-(nu/2)log nu; H=a E_w log(t_i/(nu^{nu/2}gamma)). For s=1 all t_i=nu^{nu/2}gamma.
(e) t_i does not depend on the choice of KKT measure q, and t_i<=tau(q) for every KKT measure. Hence w_i>=(nu^{nu/2}gamma/tau(q))^a, width m<=(tau(q)/(nu^{nu/2}gamma))^a, and H<=a log(tau(q)/(nu^{nu/2}gamma)). Also tau(q)<=sqrt2 min(1,||q_X||_2), where q_X is the x-marginal.
(f) For every probability q: tau(q)>=tau(p^{-n}mu_q)=nu^{nu/2}B_p max_l |K~_l(mu_q)|, where mu_q is the delta-marginal and K~_l(mu)=sum_delta mu_delta(1-rho^{-l delta}).

**Proof / evidence.**

(a) h is a minimum of finitely many polynomials, so h'(theta;v)=min_{a in A(theta)}<grad g_a,v>, and P is C^1 for b>=2. At a local maximum, min_{a in A}<grad g_a,v>/h - s<grad P,v>/P <= 0 for every v. If 0 were not in C=conv{grad g_a/h - s grad P/P : a in A}, strict separation would give v with <c,v> > 0 on C, a contradiction. So 0 = sum q_a(grad g_a/h - s grad P/P).
(b) g_a depends on factor v_it only through g_{omega_i}(a), which is linear in v_it, so <v_it, grad_it g_a> = g_{omega_i}(a). Also <v_it, grad_it P> = b||omega_i||^{b-2}||v_it||^2. Pair the (i,t) block of (a) with v_it and use sb=nu: T_q(omega_i) = (nu h/P)||omega_i||^{b-2}||v_it||^2 for every t. The left side does not depend on t, so all factor norms are equal (and nonzero), and T_q(omega_i)=(h/P)||omega_i||^b = h w_i.
(c) Put rho_i=||omega_i||/sqrt nu. Multilinearity gives grad_it T_q(omega_i)=rho_i^{nu-1} grad_t T_q(u^_i), and (a) gives grad_it T_q(omega_i)=(nu h/P)||omega_i||^{b-2} rho_i u^_it. So grad_t T_q(u^_i)=t_i u^_it with t_i=nu^{b/2}(h/P)rho_i^{b-nu}=nu^{nu/2}(h/P)a_i^{(b-nu)/nu}. Pairing with u^_it gives T_q(u^_i)=t_i. Then use a_i^{-(s-1)/s}=(P w_i)^{1-s} and h/P^s=gamma.
(d) From the law, w_i=(nu^{nu/2}gamma/t_i)^a. Summing gives (nu^{nu/2}gamma)^{-a}=sum t_i^{-a}. Taking E_w log of the law gives E_w log t = log(nu^{nu/2}gamma)+(s-1)H.
(e) The law expresses t_i through gamma and w_i only, so t_i is q-independent; t_i<=tau(q) by definition. For unit factors, |w^(y)-w^(y-delta)|<=sqrt2 and |prod_t u^_t(x_t)|<=1. Hence |T_q(u^)| <= sqrt2 sum_x q_X(x)|U^(x)| <= sqrt2 min(1,||q_X||_2), because ||U^||_2=1.
(f) For alpha in Z_p^n let R_alpha shift input t by alpha_t and the output by sum alpha_t; this is orthogonal on every factor. Since y(x)-sum alpha=y(x-alpha), g_{R_alpha omega}(x,delta)=g_omega(x-alpha,delta), so T_{alpha.q}(R_alpha u^)=T_q(u^) and tau is translation invariant. tau is convex in q, and the translation average of q is p^{-n}mu_q. For x-uniform q, T_q(omega)=sum_l Re(conj(beta_l(omega))K~_l(mu)), because E_x Phi_l=conj(beta_l). The budget at b=nu gives sum_l|beta_l(u^)|<=B_p nu^{nu/2}, attained by a frequency-pure balanced unit neuron with suitable phases.

*Check:* check_kkt_if.out: IF optima at (p,n,b)=(5,2,2.5),(5,2,2),(7,2,2.2),(5,3,2.5),(7,3,3). KKT residual <=1e-13; max|t-t_pred|/t <=9e-14; critical-point residual <=4e-13; identity log(nu^{nu/2}gamma)=E_w log t-(s-1)H to 4e-15. lm_*.out: polished local maxima at (5,2,2),(7,2,2),(5,3,2),(5,3,2.6); identity to 1e-11 on well-polished ones. check_p3_sym.out: certified p=3 winners at n=11,12; KKT residual 1e-7 and 3e-6, critical-value law to 1.5e-7 and 1.5e-6, identity to 7e-12. check_tau.out: 16 polished maximisers; nu^{nu/2}B_p max|K~|<=tau<=sqrt2 min(1,||q_X||), t<=tau, law error <=6.5e-8, share floor respected.

### R2 [CONDITIONAL]

Restricted classes. Let theta be a local maximiser on N_S (S strictly inside F). Hypothesis (LICQ): the gradients of {Re beta_j, Im beta_j : j not in S} are linearly independent at theta. Then R1 holds with T_q replaced by the augmented form T_{q,eta}=E_q g + Re sum_{j not in S} conj(eta_j) beta_j, for some multipliers eta_j. Without LICQ, R1 holds verbatim for every N_S maximiser that is also a KKT point of the unconstrained problem. This covers all decoupled IF unions (R3a).

**Proof / evidence.**

Under LICQ, N_S is a smooth submanifold near theta. The argument of R1(a) on its tangent space gives sum q_a grad g_a - (sh/P) grad P in span{grad Re beta_j, grad Im beta_j}. beta_j(omega) is multilinear in the factors, so the Euler pairings of R1(b),(c) go through with T_{q,eta}. Summing T_{q,eta}(omega_i) over i gives h, because beta_j(theta)=0 on N_S. Caveat: LICQ fails at class-C points, where grad beta_j=0 for every j with no frequency-j content. That is why R3a is stated with an unconstrained KKT measure.

*Check:* Not separately checked; it is used only through R3a, which is checked.

### R3 [PROVED]

Co-maximisers. Let S be a subset of F, s in (1,nu/2], y a maximiser of Gamma^{(S)}_s, and mu a programme KKT dual of y: a symmetric probability on the binding offsets B(y) with K_j(mu)=Gamma^{(S)}_s y_j^{r-1} for j in spt y. Let Theta be the IF union of KM2 blocks with allocation y.
(a) q*=p^{-n}mu is an unconstrained KKT measure of Theta, and on block j the critical value is t=nu^{nu/2}B_p K_j(mu).
(b) Critical-value law of co-maximisers. All maximisers at s satisfy t_i=nu^{nu/2}W(s)w_i^{1-s}: their neurons lie on ONE curve in the (w,t) plane. Two co-maximisers have H_A-H_B=a(E_{w_A}log t^A - E_{w_B}log t^B). A coupled co-maximiser with lower entropy sits further along the same curve, at larger shares and smaller critical values.
(c) Cross-duality is vacuous. For every network theta' and every probability mu' on Z_p^*: E_{p^{-n}mu'} g_{theta'}=sum_l Re(conj(beta_l)K~_l(mu'))>=h(theta'). For every IF network Theta' and every probability q: E_q g_{Theta'}=sum_l Re(conj(beta_l)K~_l(mu_q))>=h(Theta'). So for a decoupled IF co-maximiser Theta and any co-maximiser theta_c, both weak-duality inequalities h(theta_c)<=E_{q*}g_{theta_c} and h(Theta)<=E_{q_c}g_Theta hold for ALL networks. They carry no information beyond Paper Lemma 2.

**Proof / evidence.**

Existence of mu: apply the Gordan argument of R1(a) to the programme on the open orthant of spt y, where y^r is smooth. This gives K_j(mu)=Gamma y_j^{r-1}. Symmetrise, using kappa_j(delta)=kappa_j(-delta).
(a) E_{q*}g_{theta'}=sum_l K_l(mu) Re beta_l(theta') for every theta'. So sum q*_a grad g_a = grad[sum_l K_l Re beta_l](Theta). At a frequency-j neuron, beta_l (l != j) has zero gradient: it is multilinear in frequency-l components, which vanish in all factors. Also grad_{v_t} Re beta_j = (Re beta_ij/||v_t||^2) v_t: both lie in plane j, and the in-plane phase derivative vanishes at a positive real atom. (Indeed beta = C (p/2) lambda e^{i alpha} for v_t = lambda cos(2pi j x/p + alpha), so arg C = -alpha.) So stationarity reads K_j B_p a_i = (h/P)a_i^{1/s}. In Theta (cost P), a_i=P^s nu^{-s} y_j and h=nu^{1-s}B_p Gamma^{(S)} P^s. This reduces to K_j(mu)=Gamma^{(S)}y_j^{r-1}, true by assumption. g_Theta(x,delta)=nu^{1-s}B_p P^s sum_j y_j kappa_j(delta) is x-independent and minimal exactly on B(y), so spt q* lies in argmin g. Critical values: T_{q*}(u^_i)=K_j Re beta_j(u^_i)=K_j B_p nu^{nu/2}, since the unit neuron is budget-tight.
(b) R1(c) with gamma=W(s); subtract the entropy identities.
(c) E_x Phi_l=conj(beta_l) and g=sum_l Re(Phi_l c_l(delta)) give the formula, and an average of g is >= min g. For IF Theta', g_{Theta'}(x,delta)=sum_l Re(conj(beta_l)c_l(delta)) is x-independent.

*Check:* check_lemmas.out (K2(a)), at (5,2,2.3,{1,2}), (7,2,2,{1,2,3}), (7,2,2,{1,2}), (7,3,2.5,{1,3}), (11,2,2.2,{1,2,4}), (5,3,2,{1}): programme-dual residual <=1e-13, KKT stationarity residual of q* <=9e-14, q* supported on the active set. check_k7.out: t=nu^{nu/2}B_p K_j(mu*) to 7e-16 at (11,2,2.2,{1,2,4}), (13,2,2.4,{1,2,3,5}), (7,3,2.5,{1,3}), with unequal K_j. check_lemmas.out (K2(b)): Lemma-2 averaging on 300 random networks (p in {3,5,7}, DC, multi-frequency); formula error 7e-16, max(h-E g) = -0.29 <= 0. Note: KKT measures are not unique. NNLS finds non-uniform ones (TV distance 0.6-0.9 from x-uniform) with larger tau, while the critical values are identical, as R1(e) predicts.

### R4 [PROVED]

IF maximisers satisfy (E_S). Let s in (1,nu/2) (i.e. b>2), S a subset of F, and theta in N_S IF on S (beta_{l,T}=0 for l in S, T nonempty) with gamma(theta;s)=W_S(s). Then:
- W_S(s)=D_S(s) and pi(s)=nu^{1-s};
- every neuron is balanced, DC-free and frequency-pure;
- each feature j is carried by exactly nu equal neurons forming a Lemma-7 (KM2) block;
- the budget shares are z(y) for a maximiser y of Gamma^{(S)}_s, and the features are spt y;
- H(theta;s)=log nu + H(z(y)) >= H^dec_S(s).

**Proof / evidence.**

Chain (Lemma 2; Lemma 7 for each IF feature; budget):
h <= min_delta sum_{l in S} Re beta_l kappa_l <= min_delta sum |beta_l| kappa_l <= nu^{1-s} min_delta sum Q_l^s kappa_l <= nu^{1-s} Gamma^{(S)}(sum_S Q_l)^s <= nu^{1-s} B_p Gamma^{(S)} P^s <= D_S P^s <= W_S P^s = h.
So every step is an equality.
(i) sum_{l in S}Q_l=C_b P. Since sum_{l in F}Q_l <= C_b P neuron by neuron, every neuron is budget-tight and carries no atom off S. By the budget-equality lemma (Hoelder with nu factors, then ||x||_b < ||x||_2 for b>2 unless x has one nonzero entry, then AM-GM), each neuron is frequency-pure, balanced and DC-free.
(ii) pi=nu^{1-s} and W_S=D_S.
(iii) y_l=Q_l^s/(sum Q)^s attains Gamma^{(S)}.
(iv) |beta_l|=nu^{1-s}Q_l^s whenever Q_l>0. Otherwise y'_l=|beta_l|/(nu^{1-s}(sum Q)^s) <= y_l, strictly somewhere, has the same min_delta sum y' kappa = Gamma^{(S)} with sum y'^r < 1, and rescaling y' to the budget beats Gamma^{(S)}.
(v) By the equality clause of Lemma 7 (KM2), each feature has exactly nu nonzero atoms of equal modulus. Budget-tight frequency-pure neurons have |beta_il|=B_p a_i, so these are nu equal neurons of total cost Q_l/C_b. So z_l=y_l^r, each neuron has share z_l/nu, and H=log nu + H(z).
(vi) The IF block is an optimal single feature (efficiency nu^{1-s}=pi) of entropy log nu, so H^(1)_min <= log nu and H >= H^dec_S.

*Check:* check_lemmas.out (K3): 200 random IF unions (p in {5,7,11}, n in {2,3}, random b, supports, block sizes, phases): max gamma/(nu^{1-s}B_p Gamma^{(S)}) - 1 = -1.6e-15 <= 0. At the programme allocations (7,2,2.4,{1,2}), (7,3,2.5,{1,2,3}), (11,2,2.6,F): ratio 1.0000000000, and H = log nu + H(z) to 1e-8.

### R5 [PROVED]

Departures need cheap features.
(i) Every class-C network with features in S and no cheap feature (|beta_l| <= nu^{1-s}Q_l^s) satisfies, at every b in [2,nu]:
gamma(theta;s) <= nu^{1-s}B_p Gamma^{(S)}_s - P^{-s} max_{l in F, T nonempty} |beta_{l,T}| sqrt(kappa_l(delta°)/2),
with delta° in argmin_delta sum_{l in S} Q_l^s kappa_l(delta). So no-cheap networks lose LINEARLY in interference.
(ii) For s in (1,nu/2), every maximiser on N_S with no cheap feature is IF; so R4 applies and it satisfies (E_S).
(iii) If L_S departs at s_0 in (1,nu/2) with positive right slope (a jump: L_S'_+(s_0)=H^dec_S-H*_S>0), the co-maximiser theta_0 at s_0 with H(theta_0;s_0) < H^dec_S(s_0) has a cheap feature. At n=2 the cheap feature lives in a network with >= 2 output frequencies, since single-feature networks are never cheap (Note Thm q1n2).

**Proof / evidence.**

(i) Choose the sign of ±delta° whose mean gap is <= sum_l Re beta_l kappa_l(delta°) (their average, by Lemma 2). The Riesz test 1-cos(2pi<chi,x>/p+psi) >= 0 gives min_x g <= E_x g - |g^_delta(chi)| for every chi != 0. By R6, in class C |g^_{±delta}(2l 1_T)| = |beta_{l,T}| |sin(pi l delta/p)| = |beta_{l,T}| sqrt(kappa_l(delta)/2), for every l in F. Then
h <= sum_S Re beta_l kappa_l(delta°) - det <= nu^{1-s} sum_S Q_l^s kappa_l(delta°) - det = nu^{1-s} min_delta sum_S Q_l^s kappa_l - det <= nu^{1-s} Gamma^{(S)}(sum_S Q_l)^s - det <= nu^{1-s} B_p Gamma^{(S)} P^s - det.
(ii) The chain of R4, with the no-cheap hypothesis in place of Lemma 7, ends in D_S P^s <= W_S P^s = h. So all steps are equalities, every neuron is budget-tight, and (b>2, budget-equality lemma) theta is in class C. Then (i) gives det <= 0, and since kappa_l(delta°)>0 for every l in F, all beta_{l,T}=0.
(iii) L_S'_+=H^dec_S-H*_S (Danskin both sides, as in E_split/E_reduce). A positive value gives a maximiser with H < H^dec_S, which cannot be no-cheap by (ii)+R4.

*Check:* check_k3plus.out: 190 no-cheap class-C perturbations of IF optima at p in {5,7}, n in {2,3}, b in [2,nu], supports S=F or |S|=F-1, perturbation sizes 1e-4..1e-1. max gamma/(nu^{1-s}B_p Gamma^{(S)}) = 0.99957 <= 1; max (h - [bound - det])/bound = -3.3e-4 <= 0. Detection/bound ranges over [7e-5, 0.1], so the linear term is genuinely active.

### R6 [PROVED]

One constant fibre forces IF in class C, and the KKT characterisation of decoupling.
For class-C networks, g^_delta(2l 1_T) = (1/2) conj(beta_{l,T}) c_l(delta) for all l in F, T nonempty, delta != 0. Hence if g(.,delta_0) is constant in x for a single offset delta_0, the network is IF; in particular this holds if argmin g carries an x-uniform probability p^{-n}mu.
Corollary (s in (1,nu/2)): a class-C global maximiser admits an x-uniform KKT measure iff it is a decoupled IF union with a programme allocation. In class C, decoupling is exactly x-uniformity of the dual.

**Proof / evidence.**

SPLIT R4 gives g^_delta(chi) = (1/2) sum_j [phi_j(chi) c_j(delta) + conj(phi_j(-chi) c_j(delta))], with phi_j(chi) = 2p^{-nu} sum_i w^_i(j) prod_t u^_it(chi_t - j). A frequency-j class-C neuron has u^_it supported on {±j}. For chi = 2l 1_T, phi_j(chi) != 0 needs 2l = 2j, i.e. j=l, and then phi_l(chi) = conj(beta_{l,T}). phi_j(-chi) != 0 needs -2l in {0,2j}, which is impossible for l,j in F. Also c_l(delta) != 0 for delta != 0. So a constant fibre kills every beta_{l,T}. Corollary: (=>) by the lemma and R4; (<=) by R3(a).

*Check:* check_lemmas.out (K6): the identity holds to 2.2e-16 relative on 60 random class-C networks at p in {5,7,11}, n in {2,3}, for all l, T and delta.

### R7 [PROVED]

Critical values versus entropy (the provable form of 'concentrated networks cannot have all critical values high').
(a) If feature l is carried by budget-tight neurons with aligned atoms (|beta_l| = sum_i |beta_il|), its cheapness is exactly |beta_l|/(nu^{1-s}Q_l^s) = exp((s-1)(log nu - H_s(w^{(l)}))), with H_s the Renyi entropy of order s of the within-feature cost shares. So cheap <=> H_s < log nu, and Lemma 7 <=> aligned IF features have H_s >= log nu.
(b) Every network satisfies gamma <= B_p min_delta sum_l (sum_i v_il^s) kappa_l(delta) <= B_p Gamma_s exp(-(s-1) min_l H_s(v^{(l)})), where v_il = |beta_il|^r/(C_b P) are the atom cost shares (sum v <= 1).
(c) Let q be any probability on argmin g with delta-marginal mu. Split each critical value t_i = c_i + iota_i, where c_i = T_{p^{-n}mu}(u^_i) is the clean part and iota_i the interference part. Then sum_i rho_i^nu iota_i = -Det_mu := -sum_delta mu_delta (E_x g(.,delta) - h) <= 0, with rho_i^nu = prod_t ||v_it||. In class C, Det_mu >= sum_delta mu_delta max_{l,T} |beta_{l,T}| sqrt(kappa_l(delta)/2), which is > 0 unless the network is IF.
Reading: concentration (Renyi deficit) is exactly cheapness. Cheapness forces interference (Lemma 7). Interference pushes the interference parts of the critical values negative on aggregate, by at least the linearly detected interference. The quantitative comparison of this loss with the cheapness gain is the payment problem and is NOT supplied by KKT.

**Proof / evidence.**

(a) For budget-tight neurons |beta_il| = B_p a_i. So Q_l = B_p^r P_l, where P_l is the feature's cost, and |beta_l| = B_p sum_i a_i. The ratio is nu^{s-1} sum_i a_i/P_l^s = nu^{s-1} sum_i (w^{(l)}_i)^s = exp((s-1)(log nu - H_s)).
(b) Lemma 2 and |beta_l| <= sum_i |beta_il| give h <= min_delta sum_l sum_i |beta_il| kappa_l. Also |beta_il| = B_p P^s v_il^s, because C_b^s = B_p. Then min_delta sum y_l kappa_l <= Gamma_s (sum y_l^r)^s, with y_l = sum_i v_il^s, y_l^r = V_l exp(-(s-1) r H_s(v^{(l)})), and sum V_l <= 1 by the budget.
(c) sum_i rho_i^nu T_q(u^_i) = E_q g = h, and sum_i rho_i^nu c_i = E_{p^{-n}mu} g. Then E_x g(.,delta) - h >= E_x g - min_x g >= max_chi |g^_delta(chi)| (Riesz test), and R6 evaluates this in class C.

*Check:* check_lemmas.out (K5): identity (a) on 200 random aligned features (p in {5,7,11}, n in {2,3}, random b, 1-8 atoms), relative error 6.7e-16. (K5b): bound (b) on 300 random positive-margin networks with DC and mixing, max gamma/bound - 1 = -1.1e-3 <= 0. check_k7.out: identity (c) to 7e-16 on 150 random class-C networks; detection bound minus Det_mu <= -0.42 (always <= 0).

### R8 [PROVED]

Departure lemma (second order in s). Let L_S(s_0)=0 with s_0 in [1,nu/2). Let Theta_0 be a decoupled maximiser at s_0 and psi its Gibbs curve (E_reduce Lemma 2(b)): psi <= D_S, psi(s_0)=D_S(s_0), (log psi)' = -H(Theta_0;.), (log psi)'' = -Var(Theta_0;.)/sigma. For every theta in N_S with h>0 and every s>s_0:
log(gamma(theta;s)/D_S(s)) <= -Delta(theta) + [H(Theta_0;s_0) - H(theta;s_0)](s-s_0) + int_{s_0}^{s} (s-tau)[Var(Theta_0;tau) - Var(theta;tau)]/tau dtau,
where Delta(theta) = log D_S(s_0) - log gamma(theta;s_0) >= 0.
Hence a network that overtakes D_S just after s_0 must, at s_0, be entropy-lower than every decoupled maximiser to first order, or have flatter cost shares (lower Var(log w)) to second order, by more than its value deficit. Together with L'_{S+} = H^dec - H*, a departure is either a JUMP (a co-maximiser at s_0 with H < H^dec, which carries a cheap feature by R5(iii)) or TANGENTIAL (overtaking networks with H(.;s_0) -> H^dec(s_0) and Delta = o(s-s_0)).

**Proof / evidence.**

log D_S >= log psi. phi = log gamma(theta;.) - log psi is smooth, with phi(s_0) = -Delta, phi' = H(Theta_0) - H(theta) and phi'' = (Var_{Theta_0} - Var_theta)/sigma. This uses the temperature identities (GLOBAL R1) for both Gibbs curves; psi has the Gibbs form const - sigma log sum_{l,i}(A_l a_i)^{1/sigma}, whose shares are Theta_0's cost shares. Taylor with integral remainder gives the bound. Delta >= 0 because gamma(theta;s_0) <= W_S(s_0) = D_S(s_0).

*Check:* check_departure.out (1): the Taylor identity with integral remainder, for 20 random pairs of fixed networks at p=5, n=2, s from 1.1 to 1.45, holds to 1.8e-15.

### R9 [PROVED]

The decoupled optimum's own dual prefers non-features (proved for uniform optimal allocations). Let s>1 and y a maximiser of Gamma_s that is uniform on its support S, with S != F and |S|=k (this includes every K*=1 regime). Let mu be any programme KKT dual. Then (1/(|F|-k)) sum_{l not in S} K_l(mu) > K_j(mu) for every j in S. Consequences for the decoupled IF optimum:
- its x-uniform KKT measure q* has tau(q*) = nu^{nu/2}B_p max_l K_l(mu), attained ONLY by neurons at non-feature frequencies;
- this value is strictly above every neuron's critical value.
At s=1, complementary slackness puts the support inside argmax_l K_l(mu), so convex duality certifies 'no extra features'. For s>1 the dual favours non-features, which are excluded only by the concavity of the cost (y^r has infinite slope at 0). An upper-half proof must therefore use the SIZE of a would-be extra feature, not dual optimality.

**Proof / evidence.**

(1) sum_{l in F} kappa_l(delta) = p/2 for every delta != 0, so sum_{l in F} K_l(mu) = p/2 for every probability mu.
(2) On S, K_j = Gamma_s y_j^{r-1} = Gamma_s k^{s-1} = (1/k) min_delta sum_{j in S} kappa_j(delta), since y_j = k^{-s}.
(3) This value is strictly below the average (1/k)(1/(p-1)) sum_{delta != 0} sum_S kappa_j(delta) = p/(p-1). Strictness: if f(delta) = sum_{j in ±S} rho^{j delta} were constant c on Z_p^*, then f = c + (2k-c)1_0. Comparing Fourier transforms, p 1_{±S} = cp 1_0 + (2k-c). At any m outside ±S and 0 this forces c = 2k, and then m = 0 gives 2kp = 0; this is impossible when S != F.
(4) Hence (p/2 - kK_S)/(|F|-k) > K_S <=> K_S < p/(2|F|) = p/(p-1).
(5) The tau statement follows from R1(f) (equality for x-uniform q) and R3(a). Equality in T <= K_max sum_l |beta_l(u^)| <= K_max B_p nu^{nu/2} needs all atom mass at frequencies with K_l = K_max, all of which lie off S.

*Check:* check_offsupport.out: p in {7,11,13,17}, s in {1.1,1.3,1.5,1.8}. The proof covers only the uniform case p=13, s=1.8 (spt 2, max/min y = 1.000000); the other 9 proper-support cases (p=7 s=1.8; p=11 s=1.5,1.8; p=13 s=1.3,1.5; p=17 s=1.3,1.5,1.8) have non-uniform optimal allocations (max/min of y on the support 1.51-2.30). In all 10 cases max off-support K_l/Gamma exceeds max on-support (e.g. p=13, s=1.5: 3.48 vs 1.86; p=17, s=1.8: 5.19 vs 2.73). The non-uniform statement is labelled NUMERICAL (R9b).

### R9b [NUMERICAL]

The conclusion of R9 also holds for the non-uniform optimal allocations tested: in all 10 proper-support cases (p in {7,11,13,17}, s in {1.1,...,1.8}) max_{l not in spt} K_l(mu*) > max_{j in spt} K_j(mu*).

**Proof / evidence.**

No proof for non-uniform y. Programme optimality gives no constraint on off-support K_l (adding mass eps costs eps^r >> eps), so a proof would need the coverage geometry of kappa.

*Check:* check_offsupport.out.

### R10 [NUMERICAL]

KKT data of local maximisers. Low-entropy KKT points (H < H^dec) are common at p>=5, and all of them are strictly below the IF value. The certified p=3 winners satisfy every KKT identity, and their critical-value deficit is smaller than (s-1) x entropy deficit. The IF analogue of (E) therefore fails at p=3, consistently with (E) itself being vacuous there.

**Proof / evidence.**

search_localmax.py, gamma/gamma_IF with H-H^dec(IF):
- (5,2,2): IF optima 1.000000 (H = H^dec); non-IF 0.8370 and 0.8348 (H-H^dec = -0.192, -0.194; cheapness 1.10-1.11; interference 0.18-0.21).
- (7,2,2): 2-feature IF on {1,3} or {1,2}: 0.980055 (-0.424); non-IF 0.874 (-0.266), 0.842 (-0.254), 0.752 (-0.592), 0.560 (-0.671).
- (5,3,2.6): 1-feature IF 0.8029 (-0.693); non-IF 0.7857 (-0.168); IF optimum 1.000000.
- (5,3,2), where K*=1: 2-feature IF 0.9045 (+0.693); non-IF 0.765 (+0.536).
p=3 winners (slot-symmetric SLSQP polish, check_p3_sym.py):
- n=11: gamma/gamma_IF 1.0631, H = 2.0794 < log 12 = 2.4849, critical-value deficit 1.966 < (s-1)(log nu - H) = 2.027;
- n=12: 1.1934; deficit 1.846 < 2.023;
- n=13: 1.1839; deficit 1.850 < 2.019 (KKT residual 3e-3, less polished).

*Check:* lm_p5_n2_b2.0.out, lm_p7_n2_b2.0.out, lm_p5_n3_b2.0.out, lm_p5_n3_b2.6.out, check_p3_sym.out. These are local searches: evidence, not proof.

### R11 [NUMERICAL]

Level crossings with an entropy drop really occur, between DECOUPLED branches. At p=7, n=3 the IF unions with exact supports {1,2,3} and {1,2} are co-maximal at the programme transition s_c = 1.5475377 (ratio 1.000000). Their entropies are 2.4849 vs 2.0617, and their mean log critical values -4.4308 vs -4.6626, differing by exactly (s_c-1) dH. The lower-entropy branch overtakes (ratio 0.97807 at s=1.6). Both are KKT points with residual <= 1e-13. So 'co-maximal + lower entropy + overtaking' is not contradictory in itself. (E) forbids it only for coupled networks, through cheapness (R5).

**Proof / evidence.**

Exact-support restricted programmes by vertex enumeration; IF unions; NNLS KKT measures; identities checked to 6e-15.

*Check:* check_departure.out (2).

### R12 [PROVED]

Exact content of the KKT route (obstruction proposition). Fix p, n, S.
(a) (E_S) a.e. <=> W_S = D_S <=> every theta in N_S with a CHEAP feature and H(theta;s) < H^dec_S(s) has gamma(theta;s) <= D_S(s).
(b) At every maximiser, the KKT form of (E_S), E_w log t >= log(nu^{nu/2}W_S) + (s-1)H^dec_S, is (E_S) itself, by the critical-value law. The remaining KKT relations (t_i <= tau(q), q >= 0 on argmin) bound H only from above (R1(e)).
(c) Co-maximality adds no inequality beyond Lemma 2 (R3(c)).
(d) At n=2, (E_F) implies gamma* = gamma_IF for every p and b in [2,3], hence the relations-case and class-C value bounds at every prime. These are open problems (O1; class C beyond p=31 at b=2).

**Proof / evidence.**

(a) The first equivalence and (EW_S) <=> (E_S) are E_reduce Thm A and E_split E4. If the restricted statement holds, every theta with H < H^dec has gamma <= D_S: cheap ones by hypothesis; non-cheap ones by the chain Lemma 2 + no-cheap + budget, gamma <= nu^{1-s}B_p Gamma^{(S)} <= D_S. That is (EW_S). The converse direction is immediate.
(b) R1(c),(e).
(c) R3(c).
(d) pi_{2,b} = 3^{1-s} (Note Thm q1n2) and (a).

*Check:* Logical consequence of R1, R3 and the imported (refereed) equivalences; the numerical pieces are checked in R1-R11.

### R13 [REFUTED]

Claim: at a KKT point (in particular at a decoupled optimum), every neuron attains the injective norm of the dual tensor (t_i = tau(q)), i.e. neurons are dual best responses.

**Proof / evidence.**

Counterexample, rigorous by R3(a) + R9: the 1-feature IF block at p=5, n=3, b=2 (s=2, K*=1, y uniform). It is a KKT point with x-uniform q* = p^{-n}mu*, mu* on delta = ±1. Its neurons have t = nu^{nu/2}B_p K_1(mu*) with K_1 = kappa_1(1) = 0.691. But tau(q*) = nu^{nu/2}B_p K_2(mu*), with K_2 = kappa_2(1) = 1.809. Numerically, with NNLS measures, t/tau is in [0.12, 0.77] at all 16 polished maximisers (check_tau.out).

*Check:* check_offsupport.out, check_tau.out, check_kkt_if.out.

## Obstruction
Why KKT cannot close (E_S), stated precisely and generally:
1. (E_S) a.e. is EQUIVALENT to the value identity W_S = D_S, and to a value bound for concentrated networks with a cheap feature (R12a). At n=2 this contains the IF value at every prime, hence the relations core O1 and the class-C closure at b=2 beyond p=31. No route can be easier than these.
2. The KKT system is tautological for entropy. At every maximiser t_i = nu^{nu/2} W w_i^{1-s} (R1c), so 'critical values of the coupled winner are at least the decoupled ones' is literally (E_S). The only non-tautological relations are t_i <= tau(q) and q >= 0 on argmin g. They bound H from above (width/entropy caps, R1e) and never from below.
3. Co-maximality adds nothing. The decoupled optimum has an x-uniform dual q*, and both cross-dual inequalities between co-maximisers hold for every network, by Lemma 2 (R3c). The critical-value law is the same curve for all co-maximisers (R3b), so sharing the value only says that a lower-entropy co-maximiser sits further along it.
4. Level crossings with an entropy drop are generic and occur in this very problem among decoupled branches at programme transitions (R11). Every KKT/Danskin/second-order identity holds for both branches. What distinguishes a forbidden (coupled) overtaking is not visible to KKT data. It is visible only through cheapness (R5): a jump co-maximiser must carry a cheap feature, and in class C it has no x-uniform KKT measure (R6).
5. The second-order analysis in s (R8) only sorts overtaking into jumps and tangential (bifurcation) departures. It cannot exclude either: level crossings are allowed by the calculus, and excluding bifurcations needs local rigidity of the decoupled optimum in theta (second order in theta, with mixing), which is not known in general.
6. Upper half: for s>1 the decoupled optimum's own dual is maximised by non-features (R9; proved for uniform allocations, numerically in all 9 non-uniform cases tested). Dual optimality, the convex-end mechanism for 'no extra features', is therefore unavailable; extra features are excluded only by cost concavity, i.e. by their size.
The precise residual is: for b>2 and every p, n, S, show that a network in N_S with a cheap feature and H < H^dec_S has gamma <= D_S. In KKT terms: the negative aggregate interference part of the critical values, sum_i rho_i^nu iota_i = -Det_mu (R7c, linear detection in class C), must outweigh the cheapness gain, which equals the Renyi entropy deficit (R7a). This is the payment-versus-detection comparison of the master inequality. The KKT marginal mu is supported on binding offsets, possibly a proper subset of Z_p^*, which is exactly where known payments (c_1 = 1-3^{1-s} vs 1/2) stop applying uniformly. The p=3 winners (R10) satisfy every relation used here and violate the IF analogue of (E). So any argument must use the true price pi and, at n=2 where pi = nu^{1-s}, the arithmetic of p>=5.

## Next steps
- Attack the residual of R12(a) in its sharpest form: at b>2, bound the value of networks in N_S that have a cheap feature and H < H^dec_S. Use R7: write the margin loss as sum_i rho_i^nu iota_i = -Det_mu, with mu the KKT delta-marginal, and the cheapness gain as exp((s-1)(log nu - H_s)) - 1. A general p>=5 inequality 'Renyi deficit of a feature <= C x detected interference at the binding offsets, with C below the detection constant' would prove (E_S).
- Exclude tangential departures (bifurcations from a decoupled maximiser) by second-order rigidity in theta. At a decoupled IF union the x-uniform measure q* (R3a) makes the Lagrangian on the critical cone equal to the clean form sum_l K_l(mu*) Re beta_l - lambda P, which is separable over neurons. Prove strict negativity modulo the KM2 symmetry orbit, including mixed and compensating directions; the Note's near-IF theorem covers only class C and the relations case at n=2. Then (E_S) <=> 'no jump co-maximiser' (E_split E5), and by R5(iii) every jump co-maximiser has a cheap feature.
- Upper half: since the dual prefers non-features (R9), prove a 'no small extra atoms' lemma from cost concavity at b>2. An extra atom of size eps^nu costs >= (b/2-1) eps^2 in superadditivity slack, but the first-order mixed-mode gain is ~eps. So the lemma must control first-order compensation content at maximisers, i.e. (MM_opt) restricted to small content.
- Prove R9 for non-uniform optimal allocations (or find a counterexample) using the coverage geometry: sum_{l in F} K_l(mu) = p/2 together with K_j(mu*) = Gamma y_j^{r-1} on the support.
- Combine R5(iii) with E_reduce's tax-direction first-order rigidity (Thm B) to localise the cheap feature of a jump co-maximiser inside an optimal support, and then test the residual inequality numerically at (7,2) and (5,3) near the known transitions, seeding from the stored local maxima lm_*.npz.
