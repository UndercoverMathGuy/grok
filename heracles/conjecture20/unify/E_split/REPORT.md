# E_split: report (saved by the orchestrator from the agent's structured output)

## Headline
SPLIT-ROUTE verdict: the four ingredients cannot prove (E_S) or (EW_S) for all p, n, b. Two facts below are proved for every p, n, b and S. First, (E_S) is exactly the restricted value identity W_S ≡ D_S. Equivalently, M_S = L_S/(s-1) is non-increasing, or every maximiser has free energy F at least the decoupled one (E4). Second, on class C the split, Σ and superadditivity are all trivial (E8). So the route reduces (E_S) to two value statements: a price-relative mixing bound, and class-C decoupling. At n=2, class-C decoupling contains the class-C IF value at every p, which is open. Proved in general along the way: (E1) exact free-energy bookkeeping for split, block split, uneven replication, merging, Σ-orbits and dilation unions; (E2) entropic Lemma 7 (escort entropy >= log nu at every temperature, equivalent to Lemma 7; alignment is necessary); (E3) convex-end rigidity: at b=nu every maximiser is class C, IF and aligned, with H >= H^dec and D^+L_S(1)=0; (E5) local-to-global: any failure of (E_S) bifurcates from a maximiser at a touching point with H <= H^dec; (E6) at every maximiser, cost shares equal binding-margin shares; (E7) mode budgets: a Pythagorean ledger for one own feature at every b, and a multi-feature ledger with constant (b-2)/(2nu) for b>2, which is provably impossible at b=2. Upper half of (C): no general rigidity follows (E10 and O6).

## Principle
TEMPERATURE DUALITY. The temperature identity d/ds log gamma = -H turns entropy statements into value statements and back. Integrating an entropy lower bound over s gives a value bound; differentiating a value identity gives an entropy bound.
- Single-feature level (E2): Lemma 7 at every r is equivalent to 'the escort (cost-share) entropy of an aligned IF feature is >= log nu at every temperature'.
- Network level (E4): (E_S) is equivalent to W_S ≡ D_S. The natural monotone quantity is M_S = (log W_S - log D_S)/(s-1). Its right derivative is -J_S/(s-1)^2, where J_S = min F(maximisers) - F^dec_S. M_S starts at 0 at the convex end, because maximisers there are rigid (E3).
So the entropy formulation adds no strength. Its genuine gains are two:
- localisation: a failure must bifurcate from a low-entropy co-maximiser at a touching point (E5);
- a concrete reading: H(theta*) is the entropy of the binding-margin shares (E6).

THE SPLIT IS A COST/ENTROPY DEVICE THAT IS BLIND TO DECOUPLING. The four ingredients (frequency or block split, Σ-averaging, replication, superadditivity) act only on:
- the cost and entropy bookkeeping (E1);
- the non-own modes, which E7 prices quadratically off the endpoint.
They keep all own data, and they are identities on class C (E8c). The content of (E_S) lies elsewhere:
- cross-feature compensation inside the own data (class-C decoupling);
- foreign output channels (already for |S|=1: zero-mean outputs at other frequencies);
- the unknown shape of optimal single features (direction-uniform price, E4 corollary).
None of these is touched by the ingredients.

## Results

### E1 [PROVED]

Exact free-energy calculus, valid for all p, n, b in [2,nu], every s = nu/b and every network with h > 0.
Notation: c_i = ||omega_i||^b, Z = sum c_i, w_i = c_i/Z, H = Shannon entropy, H_q = Renyi entropy of order q, esc_q(w)_i = w_i^q/sum_k w_k^q, gamma = h/Z^s, F = log gamma + (s-1)H.
(a) F = log h - log Z - (s-1) E_w log c_i = log h - log Z - E_w log ||omega_i||^{nu-b}.
(b) F(theta;s) is the value at sigma = 1 of the tangent to sigma -> log gamma(theta;sigma) at s. Moreover dF/ds = (s-1)Var_w(log w)/s >= 0, and F(theta;s) = log gamma(theta;1) + int_1^s (sigma-1)Var/sigma dsigma >= log gamma(theta;1).
(c) Replication with cost fractions t = (t_1..t_k), applied to every neuron: Z is unchanged, h -> e^{-(s-1)H_s(t)} h, H -> H + H(t), F -> F + (s-1)(H(t) - H_s(t)) >= F. Equality holds iff t is uniform: then the rate is exactly (s-1) nats of log gamma per nat of entropy. The rate (s-1)H_s(t)/H(t) tends to 0 as t -> (1,0), so entropy can be raised at arbitrarily small relative loss of gamma.
(d) Merging omega and lambda*omega into (1+lambda^nu)^{1/nu}*omega keeps the table and strictly lowers the cost for s > 1. So no maximiser (global, or on N_S) contains two neurons with proportional tables. For an isolated pair, F(pair) - F(merged) = (s-1)(H(t) - H_s(t)) >= 0. Inside a network, the sign of Delta F is not fixed.
(e) Every element of Σ (global or per neuron) preserves every c_i. Hence w, H and Z are constant on Σ-orbits, and F(sigma theta) - F(theta) = log(h(sigma theta)/h(theta)).
(f) Frequency split. Let e_ik be the energy of neuron i in plane k in {0} u F, summed over all nu factors, and rho_ik = e_ik/E_i. Put lambda_i = sum_{k in F} rho_ik^{b/2} <= (1 - rho_i0)^{b/2}, pi_ik = rho_ik^{b/2}/lambda_i and what_i = w_i lambda_i/sum_j w_j lambda_j. Then
- Z_split = Z sum_i w_i lambda_i;
- H(split) = H(what) + sum_i what_i H(pi_i) (a chain rule with escort reweighting).
At b = 2 without DC, H(split) = H(theta) + sum_i w_i H(rho_i) >= H(theta). The same formulas hold for every block split (any partition of {0} u F into blocks). A block split is the average over the block sign group, keeps every beta_j and beta_{j,T}, and never raises the cost.
(g) Dilation union. Let Theta be the union of the copies theta∘j, j in J, each scaled to cost z_j P. Then
- H(Theta) = H(theta) + H(z);
- h(Theta) >= min_delta sum_j z_j^s m(j delta), where m(delta) = min_x g(x,delta);
- F(Theta) - F(theta) >= (s-1)(H(z) - H_s(z)) + log min_delta sum_j esc_s(z)_j m(j delta)/h >= 0.
(h) F is unbounded above on every N_S, at every s > 1.

**Proof / evidence.**

(a) H = log Z - sum w log c, and (s-1) log c_i = log ||omega_i||^{b(s-1)} = log ||omega_i||^{nu-b}.
(b) With phi(sigma) = log gamma(theta;sigma), GLOBAL R1 gives phi' = -H and phi'' = -Var/s. The tangent at s, evaluated at sigma = 1, is phi(s) + (1-s)phi'(s) = F. Its s-derivative is (1-s)phi'' >= 0. Concavity of phi gives F >= phi(1).
(c) A copy with cost t_m c_i is t_m^{1/b} omega_i, whose table is t_m^{nu/b} = t_m^s times the original. Since sum_m t_m^s = e^{(1-s)H_s(t)}, and the shares become w_i t_m, the formulas follow. Also H_s <= H, with equality iff t is uniform. Near t = (1-tau, tau): H_s ~ s tau/(s-1) and H ~ tau log(1/tau).
(d) For x > 0 and s > 1, (1+x)^{1/s} < 1 + x^{1/s}; take x = lambda^nu. An isolated pair is the uneven replication of the merged neuron, so (c) gives its F.
(e) Each element of Σ is orthogonal on every factor of every neuron.
(f) The pieces satisfy ||omega_i^(k)||^2 = e_ik, so c_ik = c_i rho_ik^{b/2}. The split shares are c_ik/Z_split = what_i pi_ik; apply the chain rule. Block split: take plane projections P_B = sum_{k in B} P_k. A mode whose factors carry block labels (B_1..B_n; B_0) receives the sign prod_t s_{B_t,t} s_{B_0,t}. This is trivial iff all labels agree, and beta_{j,T} has all its labels in the block of j. Superadditivity of t^{b/2} gives the cost bound (SPLIT R2(5) generalised).
(g) Neuron i of copy j has share z_j w_i. Gaps of a union add, and copy j contributes g(jx, j delta) scaled by z_j^s. Then use log sum_j z_j^s = -(s-1)H_s(z) and m >= h.
(h) Add N pairs (v, ±w) whose tables cancel, with total cost eps Z. Then gamma is multiplied by (1+eps)^{-s} while H >= (eps/(1+eps)) log(2N) - O(1), which tends to infinity.
Answer to (b). The operations act as follows:
- replication and dilation unions never lower F;
- merging and balancing raise gamma, but no maximiser has proportional pairs or unbalanced neurons, so they are exhausted there;
- Σ-flips and the split leave the entropy fixed (Σ) or raise it at b=2 without DC (split). Their effect on gamma is the uncontrolled ratio h(split)/h or h(sigma theta)/h.
So no listed operation turns a low-entropy maximiser into a better network unless h is controlled under the split or Σ (an (MM)-type value statement), or the profile slack is controlled (direction uniformity).

*Check:* check_calculus.py -> check_calculus.out, with networks at (p,n) in {(5,2),(7,2),(5,3),(7,3),(11,2)}:
- the three formulas for F, the finite-difference tangent intercept and the integral identity agree to 3e-9;
- replication gives Delta F = 0 to 1e-15, and uneven replication matches (s-1)(H - H_s) exactly;
- the isolated-pair merge formula holds to 1.6e-15; inside networks, 66 of 192 merges raised F;
- the split cost formula and the entropy chain rule hold to 9e-16;
- block split = block-sign-group average (to 3e-15), own data preserved (4e-16), cost not raised;
- dilation-union entropy is additive (9e-16) and the F lower bound holds;
- 10^4 cancelling pairs raise F by 0.154 at a fixed 6.6% loss of gamma.

### E2 [PROVED]

Entropic Lemma 7. Let w_1..w_m > 0 with sum w = 1, and |xi_lt| = 1, satisfy sum_l w_l prod_{t in T} xi_lt = 0 for every non-empty T in [n] (an aligned interference-free feature). Put psi(r) = log sum_l w_l^r. Then
  H(esc_r(w)) = psi(r) - r psi'(r) >= log nu   for every r in (0,1].
Conversely, H(esc_r) >= log nu on (0,1] implies psi(r) >= (1-r) log nu, which is Paper Lemma 7 for that configuration.
Reading: at temperature s = 1/r, a budget-tight neuron has cost proportional to |beta_i|^r, so its cost shares are esc_r of the atom moduli. Lemma 7 at every r is therefore the same statement as 'an aligned IF feature always has at least nu effective neurons in cost share'.

**Proof / evidence.**

Paper Lemma 7 gives psi(r) >= (1-r) log nu on (0,1). Since psi(1) = 0, it follows that psi'(1) <= -log nu. psi is convex (a log-sum-exp in r), so psi'(r) <= psi'(1) for r <= 1. Hence psi(r) - r psi'(r) >= (1-r) log nu + r log nu.
Conversely, d/dr(psi(r)/r) = -H(esc_r)/r^2 <= -log nu/r^2. Integrating over [r,1] gives psi(r) >= (1-r) log nu.
Equality at some r < 1 forces equality in Lemma 7 at that r, hence the KM2 block.

*Check:* check_entropic_l7.py -> .out. Exact IF configurations (residual < 1e-12) were found by least squares, for n = 1..4 and m = nu..nu+3 atoms, about 60 per case, with non-uniform weights (max(w)*m - 1 up to 2.4).
- min over r of [H(esc_r) - log nu] >= -3e-16, i.e. >= 0;
- min over r of [psi(r) - (1-r) log nu] >= -3e-16.

### E2b [REFUTED]

Refuted: the entropic Lemma 7 without alignment, i.e. 'every interference-free feature (complex atoms beta_l) has atom-modulus entropy >= log nu'.

**Proof / evidence.**

Explicit counterexample at n = 2 with three atoms (nonaligned_example.out):
- |beta_l| = (0.49144832, 0.10252906, 0.40602261);
- arg beta_l = (4.59829238, 3.03313222, 1.07566332);
- xi angles: rows (6.24048247, 2.48936183), (5.18952748, 1.43840683), (0.21132144, 2.7433861).
All three IF sums vanish to 7e-16, |beta| = 0.1249, and Shannon H = 0.9486 < log 3 = 1.0986. Lemma 7 itself still holds, with large room, because alignment is poor (|beta|/sum|beta_l| = 0.125).
Generic non-aligned configurations reach H - log nu = -0.38 at n = 4 (check_entropic_l7.out).

*Check:* nonaligned_example.py: residuals printed; Lemma 7 holds at r = 0.5, 2/3, 1.

### E3 [PROVED]

Convex-end rigidity, for every p, n and S in F. At s = 1 (b = nu), theta in N_S maximises gamma iff all four of the following hold:
(i) every neuron has all nu factors in one frequency plane j(i) in S, no DC, and equal factor norms;
(ii) all atoms beta_{i,j(i)} are positive reals;
(iii) beta_{j,T} = 0 for every j in S and non-empty T;
(iv) y_j = sum_i |beta_ij|/(B_p sum_i a_i) maximises Gamma_1^{(S)}.
Every maximiser has H(theta;1) = H(y) + sum_j y_j H(w^{(j)}) >= log nu + min H(z*(1)) = H^dec_S(1), and the decoupled block networks attain equality. Hence H*_S(1) = H^dec_S(1), D^+L_S(1) = 0 and L_S(s)/(s-1) -> 0 as s decreases to 1. In words, (E_S) holds at the convex end with no slack.

**Proof / evidence.**

For every delta, h <= A(delta) <= B(delta), where A(delta) = sum_{j in S} Re beta_j kappa_j(delta) (Paper Lemma 2, symmetrised over ±delta) and B(delta) = sum_{j in S}(sum_i |beta_ij|) kappa_j(delta). Next, min_delta B <= (sum_{j in F} sum_i |beta_ij|) Gamma^{(S)}_1 <= B_p sum_i a_i Gamma^{(S)}_1, using the r = 1 budget sum_j |beta_ij| <= B_p a_i.
Budget equality case. Put A_t(j) = |vhat_t(j)|/sqrt((p/2)) ||v_t||. Then sum_j prod_t A_t(j) <= prod_t ||A_t||_nu <= prod_t ||A_t||_2 <= 1, followed by AM-GM. Equality forces:
- no DC;
- ||A_t||_nu = ||A_t||_2 with nu >= 3, i.e. one non-zero entry per factor;
- the same frequency in all factors (otherwise the left side is 0);
- equal factor norms.
This is (i).
Now suppose h = B_p sum a Gamma_1^{(S)}. At delta* in argmin B we get A(delta*) = B(delta*), and every kappa_j(delta*) > 0. So Re beta_j = sum_i |beta_ij| for j in S, atoms off S vanish and every neuron is tight: (ii). The allocation y is a programme maximiser: (iv). Moreover h = E_x g(.,±delta*) <= g, so g(.,delta*) is constant in x.
In a network of type (i), the Fourier coefficient of g(.,delta) at 2l 1_T is (1/2) conj(beta_{l,T}) c_l(delta), because 2l 1_T = ±2j 1_T' forces j = l and T' = T for odd p. Since c_l ≠ 0, this gives (iii). The converse is the IF computation.
Entropy. With tight neurons, w_i = |beta_i|/sum|beta|, so H = H(y) + sum_j y_j H(w^{(j)}), and each w^{(j)} is an aligned IF configuration. E2 at r = 1 gives H(w^{(j)}) >= log nu.
Optimal single features at s = 1 are the case S = {1}: a cone margin equal to B_p sum a forces cm(Phi) ≡ Re Phi ≡ Re beta. Their least entropy is log nu (the block), and z*(1) = y*(1).
Danskin at the endpoint then gives L'_+(1) = 0.

*Check:* check_convex_end.py -> .out: torch L-BFGS maximisation at b = nu, from random and IF-perturbed starts.
- (5,2): near-maximisers (gamma/(B Gamma_1) >= 0.99998) have H - H^dec(1) in [-1e-5, +0.25];
- (7,2): [0, +0.0005];
- (5,3): 0.0000.
In all cases max|beta_T|/max|beta| <= 2.6e-6 and the off-dominant-frequency energy share is <= 1.6e-5. So the maximisers are IF and class C, with H >= H^dec(1) = log nu + H(z*).

### E4 [PROVED]

Equivalence theorem and the monotone quantity, for every p, n, S. The following are equivalent:
(i) (E_S) for a.e. s;
(ii) (E_S) for every s in [1, nu/2);
(iii) (EW_S);
(iv) (J_S): for a.e. s, every maximiser theta of N_S has F(theta;s) >= F^dec_S(s) := log D_S(s) + (s-1)H^dec_S(s);
(v) M_S(s) := L_S(s)/(s-1) is non-increasing on (1, nu/2];
(vi) W_S ≡ D_S on [1, nu/2].
Also M_S'_+ = -J_S/(s-1)^2, with J_S = min over maximisers of F minus F^dec_S.
Corollaries. (E_S) implies:
(a) Gamma^{P,(S)}_s = pi Gamma^{(S)}_s, i.e. direction-uniform single-feature prices on S (GLOBAL R8 restricted to S);
(b) for S = {l}: W_{l} ≡ pi B_p kappa_min, i.e. zero-mean output fields at frequencies other than l never raise a single feature's margin;
(c) at n = 2, gamma* = gamma_IF for all p and b in [2,3], and hence the class-C IF value at every p.
Answer to (c) of the task: the only monotone quantity that works is M_S, equivalently the free-energy comparison (J_S). It is equivalent to (E_S) globally and weaker pointwise: a concentrated maximiser is allowed if it beats D_S by at least (s-1) times its entropy deficit.

**Proof / evidence.**

Two Danskin facts are used: (log W_S)'_+ = -H*_S (GLOBAL R2; width reduction preserves N_S) and (log D_S)'_+ = -H^dec_S. For the latter, apply Danskin to pi (GLOBAL R2, single-feature version) and to Gamma^{(S)}_s = max over y in the simplex of m(y)/(sum y^{1/s})^s, whose s-derivative is -H(z). Both functions are locally Lipschitz, since entropies are bounded by log(width) <= log D.
Then J_S = L_S - (s-1)L'_+ gives the formula for M'_+.
- (vi)=>(ii): L ≡ 0 forces H* = H^dec everywhere.
- (ii)=>(i) is trivial.
- (i)=>(iv): F(theta) = log W_S + (s-1)H(theta) and W_S >= D_S.
- (iv)=>(v): M_S is absolutely continuous with M' <= 0 a.e.
- (v)=>(vi): M_S >= 0 and M_S(1+) = 0 by E3, so M_S ≡ 0.
- (vi)=>(iii): gamma <= W_S = D_S.
- (iii)=>(vi): where L > 0, every maximiser beats D_S, hence has H >= H^dec, so L'_+ <= 0 on {L > 0}. The Dini monotonicity argument of GLOBAL R3 then gives L ≡ 0.
Corollary (a): W_S >= B_p Gamma^{P,(S)} >= D_S, since a union of relabelled single features lies in N_S. (b): Gamma^{({l})}_s = kappa_min. (c): pi_{2,b} = 3^{1-s} (Note), and class C contains the IF block.

*Check:* check_monotone.py -> .out: 200 synthetic envelopes of temperature curves, 4200 points. The relation M'_+ = -J/(s-1)^2 holds to relative error 1e-4 (finite differences, kinks included). The convex-end input is E3's check.

### E5 [PROVED]

Bifurcation and local-to-global reduction. Suppose W_S is not identically D_S. Take s_2 with L_S(s_2) > 0 and put s_0 = max{s <= s_2 : L_S(s) = 0}. For any s_k decreasing to s_0 and any maximisers theta_k at s_k (width <= D, normalised):
- H(theta_k; s_0) < H^dec_S(s_0) + o(1);
- gamma(theta_k; s_0) -> W_S(s_0);
- every limit point theta_0 is a maximiser at s_0 with H(theta_0; s_0) <= H^dec_S(s_0).
If s_0 = 1, theta_0 is class C, IF and aligned, with H = H^dec_S(1): each feature has atom entropy exactly log nu and the allocation is a least-entropy maximiser of Gamma_1^{(S)}.
Consequently (E_S) <=> (Loc_S). (Loc_S) says: at every s_0 with L_S(s_0) = 0, and every maximiser theta_0 there with H(theta_0) <= H^dec_S(s_0), there are a neighbourhood U of theta_0 and eps > 0 such that gamma(theta;s) <= D_S(s) for all theta in U ∩ N_S and s in (s_0, s_0 + eps).

**Proof / evidence.**

phi_k = log gamma(theta_k;.) is concave. Hence phi_k(s_k) <= phi_k(s_0) - H(theta_k;s_0)(s_k - s_0), with phi_k(s_0) <= log W_S(s_0) = log D_S(s_0). On the other side, phi_k(s_k) = log W_S(s_k) > log D_S(s_k) = log D_S(s_0) - H^dec(s_0)(s_k - s_0) + o(s_k - s_0). This gives the entropy bound.
Concavity also gives phi_k(s_0) >= phi_k(s_k) + H(theta_k;s_0)(s_k - s_0) >= log W_S(s_k) -> log W_S(s_0).
The limit statement follows from:
- compactness of the normalised width-<=D class;
- closedness of N_S;
- joint continuity of gamma on {h > 0}, where h is bounded below along the sequence;
- continuity of H(.; s_0).
The case s_0 = 1 is E3. Along theta_k, (Loc) fails at theta_0. Conversely, (E_S) gives gamma <= W_S = D_S.

*Check:* check_monotone.out: at the first departure point of synthetic envelopes, the winners satisfy H(theta_k, s_0) - H^D(s_0) <= -0.0077 over 561 cases, as predicted. The logical structure is pure analysis.

### E6 [PROVED]

Cost shares are margin shares. Let theta* maximise gamma(.;s) under all neuron rescalings (any global maximiser qualifies), and let A be the argmin set of g. There is a probability q on A such that E_q[g_i] = h w_i for every neuron i, where g_i is the gap of neuron i alone. So H(theta*;s) is the entropy of how the binding margin is shared among neurons, and (E) asks that this margin be spread over at least e^{H^dec} effective neurons.
Neuron-level bounds give only share floors, i.e. upper bounds on H: w_i >= (nu^{nu/2} gamma*/2)^{1/(s-1)}.

**Proof / evidence.**

log gamma(lambda) = log min_a sum_i lambda_i^nu g_i(a) - s log sum_i lambda_i^b c_i. Its directional derivative at lambda = 1 along v is min_{a in A} sum_i v_i nu (g_i(a)/h - w_i), using sb = nu. Optimality makes this <= 0 for every v. By the separation theorem, 0 lies in conv{(g_i(a)/h - w_i)_i : a in A}.
Share floor: h w_i = E_q g_i <= max|g_i| <= 2 nu^{-nu/2} a_i = 2 nu^{-nu/2}(Z w_i)^s.

*Check:* check_convex_end.out and check_kkt_unequal.out: LP feasibility of q on the near-argmin set.
- (5,2,2.5), (7,2,2.7) and (5,3,3.2) IF maximisers: residual <= 3.1e-8;
- converged non-IF local maxima at (7,2,2.4) with unequal shares (0.167, 0.12, 0.07): residual <= 5.3e-7.
Non-converged runs (argmin set of size 1) are not stationary and were excluded.

### E7 [PROVED]

Mode budgets: the quantitative form of 'shared neurons pay strictly off the endpoint'. For one neuron, let e = B_p ||omega||^nu, let beta_j (j in F) be its atoms, and let N = sum_{j in F} sum_{k not in {±j}^n} |2p^{-nu} what(j) prod_t uhat_t(k_t)|^2 be its non-own mode mass.
(i) sum_j |beta_j|^2 + 2^{-n} N <= e^2, for all p, n.
(ii) For b in [2,nu] and r = b/nu: sum_j |beta_j|^r + ((b-2)/(2nu)) e^{r-2} 2^{-n} N <= e^r. No positive constant is possible at b = 2: a neuron whose factors share one frequency distribution a on F, with no DC, has sum_j |beta_j|^{2/nu} = e^{2/nu} and 2^{-n} N = e^2 (1 - sum_j a_j^nu) > 0. This generalises p52 R17.
(iii) Network forms.
- One own feature l, every b (generalising p52 R2-R3 to all p, n): B_p P^s >= sqrt(Q_l^{2s} + 2^{-n} R_l^2). Here R_l = sum_i sqrt(N_i^{(l)}), where N_i^{(l)} counts every mode of neuron i except the own modes of feature l.
- Several features, b > 2: sum_j Q_j <= B_p^r P [1 - ((b-2)/(2nu)) 2^{-n} (R/(B_p P^s))^2], with R = sum_i sqrt(N_i). R bounds the l2 norm of all non-own field coefficients.

**Proof / evidence.**

Let a_t(k) be the plane-k energy fraction of factor t (t = 0 is the output), and let Ehat = 2(2p)^{-nu/2} prod_t ||v_t|| <= e (AM-GM). Then:
- |beta_j| = Ehat prod_t a_t(j)^{1/2};
- total mode mass = 2^n Ehat^2 (1 - a_0(0)), by Parseval;
- own mass = 2^n Ehat^2 x, with x = sum_j prod_t a_t(j).
Hence 2^{-n} N = Ehat^2 (1 - a_0(0) - x), which proves (i).
For (ii), put g_j = prod_t a_t(j)^{1/nu}, q = b/2 and M = max g_j; AM-GM gives sum g_j <= 1. Then sum_j |beta_j|^r = Ehat^r sum g_j^q <= Ehat^r M^{q-1}, and x >= M^nu. The function eta(u) = 1 - u^{q-1} - ((q-1)/nu)(1 - u^nu) has eta(1) = 0 and eta'(u) = (q-1)u^{q-2}(u^{nu-q+1} - 1) <= 0. So 1 - M^{q-1} >= ((q-1)/nu)(1 - x) >= ((q-1)/nu) 2^{-n} N/Ehat^2. Finally use Ehat <= e and r < 2.
(iii) One own feature: as in (i), |beta_il|^2 + 2^{-n} N_i^{(l)} <= e_i^2. Apply Minkowski in l^{2/r} to the pairs (|beta_il|^r, rho_i^r), then (sum rho^r)^s >= sum rho. Several features: Cauchy-Schwarz gives (sum sqrt N_i)^2 <= sum e_i^{r-2} N_i * sum e_i^{2-r}, and sum e^{2-r} <= (sum e^r)^{(2-r)/r}.

*Check:* check_mode_budget.py -> .out.
- 20000 random neurons (Gaussian, sparse, near-pure with leakage 1e-4 to 0.3, common-spread, unbalanced with DC; p in {3,...,13}, n in {2,3,4}, five values of b): (i) max excess 2.4e-15; (ii) max excess 1.1e-15; the observed slack is at least 1.686 times the guaranteed slack.
- Equal splits at b = 2: budget = 1.000000000000 with 2^{-n}N/e^2 = 1 - K^{1-nu} exactly.
- Network forms on 3000 random networks: -1.1e-2 and -0.17 (both <= 0).

### E8 [PROVED]

The split route as a decomposition. Class C means every neuron has all its factors in one plane k in F. Write pi^C for the class-C price, W^C_S for the class-C restricted optimum and D^C_S = pi^C B_p Gamma^{(S)}. The restricted price-programme theorem inside C gives W^C_S >= D^C_S.
(a) (E_S) and (pi = pi^C) together hold iff (W_S = W^C_S) and (W^C_S ≡ D^C_S) both hold. The latter pair is the absolute route: value-(MM) on N_S plus class-C decoupling.
(b) (E^C_S) together with (MM^rel_S): pi^C W_S <= pi W^C_S implies (E_S); and (E_S) implies (MM^rel_S). This is the price-relative route, and it is consistent at p = 3.
(c) On class C, the split is the identity, Σ fixes every table, and the superadditivity slack is zero. So all four ingredients are vacuous there.

**Proof / evidence.**

(a) <=: W_S = D^C_S <= D_S <= W_S. =>: W_S = D_S = D^C_S <= W^C_S <= W_S.
(b) W_S <= (pi/pi^C) W^C_S = (pi/pi^C) D^C_S = D_S. Conversely, W_S = D_S <= (pi/pi^C) W^C_S, because W^C_S >= pi^C B_p Gamma^{(S)}.
(c) A plane-k neuron's table is multiplied by prod_t s_{k,t} * prod_t s_{k,t} = 1. The split of a single-plane neuron is the neuron itself.

*Check:* Algebraic. Class-C invariance is checked in check_calculus.out (6): the group average equals the split, and is the identity on single-plane neurons.

### E9 [REFUTED]

Refuted: 'the frequency split never lowers the cost-share entropy', beyond the case b = 2 without DC.

**Proof / evidence.**

Explicit networks at p = 11, n = 2 (check_calculus.out):
- b = 3: a small neuron (norm^2 ratio 0.1) spread equally over 5 frequencies, plus a pure neuron. H(theta) = 0.1370 and H(split) = 0.0959.
- b = 2 with DC: two neurons of equal norm, one of them 90% DC. H drops from log 2 = 0.6931 to 0.3046.
The exact mechanism is the reweighting term H(what) - H(w) of E1(f): mixed neurons lose cost weight.

*Check:* check_calculus.out section (4).

### E10 [CONDITIONAL]

Upper half of (C), under the hypothesis W_S ≡ D_S for every S (equivalently (E_S) for all S). Then:
- the lower half of (C) holds (GLOBAL);
- at every s where log D_S is differentiable (a.e.), every maximiser on N_S has H(theta;s) = H^dec_S(s) exactly;
- at a.e. s, every maximiser also satisfies Var_w(log w) >= s (d/ds) H^dec_S(s).
These identities do NOT exclude extra features (see the obstruction).

**Proof / evidence.**

Apply Danskin from both sides: D^- = -max H = D^+ = -min H at differentiability points. For the variance, note that log W_S is semiconvex, as a sup of curves with phi'' >= -C on the width-<=D class, so it is twice differentiable a.e. (Alexandrov). The concave curve log gamma(theta*;.) touches it from below with equal first derivative, which forces -Var/s <= (log W_S)'' = -(H^dec_S)'.

*Check:* Consistent with GLOBAL R2's kink check at the (5,3) transition.

### E11 [NUMERICAL]

Evidence for the obstructions, cited and re-read:
(a) At p = 3 the certified counterexamples have split margins -284.5 (n = 11) and -746.3 (n = 13), against positive margins before the split. Any network-level comparison gamma(theta) <= gamma(theta_split) therefore fails at p = 3, while (E) holds there trivially.
(b) At (5,2), b = 2, the relaxation with non-own modes free, which is the most that tests blind to non-own modes can certify, reaches 1.034 gamma_IF = 1.32 D_{1} at a single atom. So delta-isolation plus own-data budgets cannot prove even (E_{1}).
(c) (E_{1}) at (5,2) is open. The best rigorous bound is W_{1} <= 0.995 gamma_IF = 1.273 D_{1} at s = 3/2 (p52 R9); numerically W_{1}/D_{1} = 1.000000 (GLOBAL R12; VERIFY v7).

**Proof / evidence.**

(a) SPLIT R12 with VERIFY v_p3.py (exact arithmetic). (b) p52 R15 free relaxation. (c) p52 R9 together with Gamma_{3/2}(5)/kappa_a = 1.279.

*Check:* Cited outputs: split/verify/v_p3.out; p52 free_sup.out; global/verify/v7_w1_search.out.

## Obstruction
O1. No free lunch. (E_S), (EW_S) and (J_S) are each equivalent to the restricted value identity W_S ≡ pi B_p Gamma^{(S)} on [1, nu/2] (E4). The same holds one level down: entropic Lemma 7 is equivalent to Lemma 7 (E2). Any proof of (E_S) is therefore a value proof. (E_S) carries three single-feature or decoupling contents that no entropy device supplies:
- direction-uniform prices on S (E4a);
- for |S| = 1, foreign zero-mean outputs never help a single feature (E4b). This is open already at (5,2), where the best bound is 1.273 D_{1} (E11c);
- at n = 2, the class-C IF value at every p (E4c). This is open beyond p = 31 at b = 2, where the relations core and the sharp energy constants sit.

O2. The ingredients are vacuous exactly where the content lies (E8c). On class C:
- the split is the identity;
- Σ fixes every table;
- the superadditivity slack is zero;
- replication and dilation unions are exact but value-neutral (F is unchanged or rises with no control of gamma; E1).
Only delta-isolation remains, and it sees nothing but own data. The own-data relaxation overshoots D_{1} by 32% at (5,2,2) (E11b). So the route cannot touch class-C decoupling.

O3. Wrong projection.
- The absolute route needs pi = pi^C (E8a). Network-level split comparisons fail at p = 3, where the certified splits have margins -284 and -746 (E11a), even though (E) holds there trivially.
- The price-relative route needs (MM^rel_S): pi^C W_S <= pi W^C_S. This compares the mixing gains of coupled and single-feature networks. The split erases both alike, and no ingredient attributes mixed modes to features.
- The obstruction is arithmetic. Of the 2|F|-1 channels reaching feature l's sign-flipped characters (SPLIT R4b), 2|F|-2 are foreign: their output j differs from l. Foreign channels do not exist in single-output networks, so the decoupled side cannot use them, while the coupled side can. This is already the whole of (E_{l}).

O4. Prices are quadratic, gains are linear.
- E7 shows that strict superadditivity buys a mode budget with constant (b-2)/(2nu) for b > 2, and no positive constant exists at b = 2. Either way the price of a mixed mode is quadratic in its amplitude.
- Delta-isolated tests never see mixed modes. Tests that do see them pay linearly (n·R).
- Closing the comparison therefore needs case-specific certificate constants, as in the computer-assisted (5,2) proof. No general inequality follows.

O5. Local-to-global needs the unknown single-feature optimum.
- (E_S) is equivalent to (Loc_S) at touching points (E5).
- At s_0 = 1 the low-entropy maximisers are explicit (E3): aligned IF class-C networks with within-feature entropy log nu. But (Loc) there is a statement about b near nu, which the hard rules disallow as a result.
- At touching points s_0 > 1, the low-entropy maximisers are co-maximisers of unions of relabelled OPTIMAL single features. Their structure is the unknown price problem: IF blocks if pi = nu^{1-s}, but p = 3 shows they can be non-IF.
- So the continuation scheme needs a local decoupling theorem around arbitrary optimal single features, uniformly in s_0. Nothing in the route provides it.

O6. Upper half.
- Under W_S ≡ D_S for all S, maximisers have entropy exactly H^dec at a.e. s (E10). But a feature with cost share eps changes the entropy by only O(eps log(1/eps)), which concentration elsewhere can offset.
- Extra features are compensation fields whose clean part is non-zero. Letting that clean part tend to 0 connects N_{S*} continuously to N_{S* minus {l}}, so no value or entropy identity separates them.
- Excluding extra features needs strict rigidity: at a maximiser, every output field at a frequency outside spt y* vanishes identically.
- The only strictness among the ingredients is superadditivity at b > 2. It acts on the cost side and gives only quadratic prices (E7), and it is silent on class-C extra features, which carry no mixed modes at all.

## Next steps
- Attack (E_{l}) for |S| = 1 at all p, n: 'foreign zero-mean outputs never help a single feature'. It is the smallest instance of (E) that is not definitional, and E7(iii) already supplies a general Pythagorean ledger for every b. The missing piece is a general detection certificate for foreign channels. Without one, the computer-assisted (5,2) argument does not generalise.
- Entropic width in margin-share form (E6). Show that within each feature, the binding-margin shares E_q[g_i]/h have entropy >= log nu minus a function of the interference that the Fritz-John measure q detects. This is a robust version of E2 that charges entropy deficits to beta_{j,T}. It is the only way an entropy lower bound can enter, since neuron-level bounds give share floors only.
- Price-relative mixing (MM^rel_S). Look for an attribution of mixed modes to features that is compatible with p = 3. A block split whose blocks are the frequency orbits l·Supp(theta_1) is cost-non-increasing and keeps own data (E1f), but it exists only when these orbits are disjoint. Find the precise arithmetic condition, and the replacement when DC is shared.
- Counterexample search for (E_S) with |S| = 2 at p = 7, 11 and b in (2,3): maximise gamma on N_S with outputs on all frequencies, and compare against D_S. Check numerically the pointwise form (J_S) and the monotonicity of M_S = L_S/(s-1) along s. Any violation would refute Conjecture 20' restricted to S.
- Do not pursue the split route as the main mechanism for (E). Use E8's decomposition to route class-C decoupling to the class-C agents (O1, relations core) and foreign channels to the compensation agents (O2). Keep E3, E4 and E5 as the continuation framework, and E7 as the general mode ledger for b > 2.
