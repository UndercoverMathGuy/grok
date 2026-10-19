# gibbs: report (round 2; saved by the orchestrator from the agent's structured output)

## Headline
GIBBS (resumed). Joint detection for class C at n=2 now has the sharp constant (p-1)/(p+1). A new two-plane Bhatia-Davis inequality (PBD#) is proved via Cauchy-Davenport, and a sawtooth attains the constant. PB2 (constant 1) is REFUTED at every p, and the earlier numerical "min 1.0003" was an optimiser failure. With a new computer-assisted SHARP L2 payment, |A| - 3^{1-s}Q^s <= ((1-3^{1-s})/sqrt3) R (equality: one neuron), class C at n=2 is closed (every class-C maximiser is IF with K* features) for EVERY prime p <= 31 and every b in [2,3]. The range grows with b: p <= 53 for b >= 2.2 and p <= 61 for b >= 2.25. With P12 and an S_3 symmetry, the closure reaches p <= 53 at b=2 when each cheap feature's interference sits on one slot-set. "p <= 59 at b=2 for all of class C" cannot come from the L2 route: both of its constants are now sharp.

## Principle
RANK-PLANE TEST. Take a sum of three ridge functions F = f1(x1)+f2(x2)+f3(x3) and two parallel planes Pi_c, Pi_c' (sum x_i = c). Their worst depth and worst height are controlled by the three value DISTRIBUTIONS alone, through Cauchy-Davenport: top/bottom level sets whose sizes sum to p+2 meet on every plane. The CD-admissible rank triples y in [0,p-1]^3 with sum y in {p-1, 2p-2} carry the uniform measure nu. Under nu:
- each rank is uniform;
- each pair of ranks has law (1 + [a+b = p-1])/(p(p+1)), i.e. it is independent up to one anti-diagonal.
So Bhatia-Davis holds jointly for all three ridge functions, losing only an anti-monotone pairing of size at most 2/(p+1).

This theta-adapted test (built from the sorted profiles of the network) detects all slot-sets T at once, with no case split.

What it removes:
(a) The 2^n - 1 = 3 loss of Note Thm 'decoupling when few features are bought cheaply' (now (p+1)/(p-1), sharp).
(b) The split between the relations case (T={1,2}) and mixed T: one argument covers all of class C at n=2.
(c) For general n, slicing by n-2 coordinates reduces to the n=2 core in every slice. This improves 1/(2^n-1) by a factor 2 to 3.6 (n=3: kappa/(2+kappa), about 1/3 instead of 1/7).

Entropy (Gibbs) reading of the payment side. The cost of a feature's atoms relative to their mass is at least 3^{-(s-1)(1-H_3(pi))}, where pi = spectrum/3 of the twisted Gram matrix and H_3 is the base-3 Shannon entropy. This entropy bound is what makes the L2 payment for all s reduce to the single case s = 3/2: x^Lambda is concave in x = 3^{1-s}.

## Results

### R1 [REFUTED]

PB2 (gibbs/pb2_pure.py): for mean-zero f1,f2,f3 on Z_p and planes c != c', (-min_{Pi_c} F)(max_{Pi_c'} F) >= V = Var f1 + Var f2 + Var f3. Equivalently, joint L2 detection with constant 1 for class C at n=2: A_delta A'_delta >= E_delta. Also refuted: the claim in gibbs/jl2_p11_n2_prod.out that the minimum of AA'/E is 1.0003 at p=11 (the true minimum is 5/6).

**Proof / evidence.**

Counterexample, valid at every odd prime p.
- Take f1 = f2 = f3 = r with r(x) = (-x mod p) - (p-1)/2, and c = 1, c' = 2.
- On Pi_c the residues r_i in [0, p-1] satisfy sum r_i == -c (mod p).
- Depth on Pi_1 = 3(p-1)/2 - (p-1) = (p-1)/2.
- Height on Pi_2 = (2p-2) - 3(p-1)/2 = (p-1)/2.
- V = 3(p^2-1)/12 = (p^2-1)/4.
- Ratio = (p-1)/(p+1) < 1.

Checks.
- Exact rational arithmetic for p = 3, 5, 7, 11, 13, 17: m_1 M_2 / V = 1/2, 2/3, 3/4, 5/6, 6/7, 8/9 (gibbs2/pbd_sharp_check.py, output pbd_sharp_check.out, part S5).
- Network level: an explicit class-C network at n=2 realises D_1 = D_2 = r, D_12(w) = r(1-w), delta = 1. It has AA'/E = (p-1)/(p+1) exactly for p = 5..17 (gibbs2/classC_net.py, part N3; classC_net.out).
- The previous agent's own check_planebd.py already printed 'min ... (PB2 conjecture: >= 1) = 0.6667'. joint_l2.py's smoothed-min L-BFGS never reached the sawtooth.

Why it matters: profiles D_T are arbitrary mean-zero functions (D_T = phi_T - tau_delta phi_T is a bijection, and any (A,B,C,D) is realisable with 4 atoms per frequency). So the constant 1 is false for class C itself, not just for the pure inequality.

*Check:* 

### R2 [PROVED]

Reconstruction of the previous agent's PBD, which was claimed but whose proof was lost.

Setting: mean-zero f_i on Z_p, m = -min_{Pi_c} F, M = max_{Pi_c'} F, eps = min_{i<j} sqrt(osc_i osc_j)/p. Then
- m M + m eps >= V - sqrt(V) eps;
- (m+M)/2 >= c_p sqrt(V), with c_p = sqrt(1-eta) - eta/2 and eta = sqrt(2/(3p)).

Superseded by R3.

**Proof / evidence.**

Setup.
- Sort each f_i decreasingly: A_i(0) >= ... >= A_i(p-1). Let H(y) = sum_i A_i(y_i).
- Let Y be uniform on {y in [0,p-1]^3 : sum y == p-2 (mod p)}. Any two coordinates are then independent and uniform, and sum Y lies in {p-2, 2p-2}.

Bottom constraint (holds on all of spt Y). The bottom sets L_i = {f_i <= A_i(y_i)} have sizes >= p - y_i, so sum |L_i| >= p+2. By Cauchy-Davenport, every plane Pi_c then has a point with F <= H(y). Hence H >= -m.

Top constraint. Where sum y = 2p-2, the top sets have total size 2p+1 >= p+2, so H <= M. Where sum y = p-2, raising one rank by 1 restores it, so H(y) <= M + R(y) with R(y) = min_i g_i(y_i+1), where g_i(k) = A_i(k-1) - A_i(k) >= 0 are the gaps.

Expansion.
- E[(H+m)(M+R-H)] >= 0.
- With E H = 0 and E H^2 = V this gives m M >= V - m E R - E[H R].

Bounding R.
- R <= sqrt(g_i g_j) for any pair i != j.
- E sqrt(g_i) <= sqrt(sum_k g_i(k)/p) = sqrt(osc_i/p).
- So E R <= eps and E R^2 <= eps^2, and Cauchy-Schwarz gives E[H R] <= sqrt(V) eps.

Corollary.
- osc_i^2 <= 2p V_i, so eps^2 <= 2V/(3p).
- AM-GM on m(M + eta sqrt V) >= (1-eta)V gives the c_p form.

Check: re-run of gibbs/check_planebd.py: minimum slack 0.0855 >= 0 and corollary slack 0.146 >= 0, over 5400 instances.

*Check:* 

### R3 [PROVED]

PBD# (sharp plane Bhatia-Davis).

Setting: p prime; f1, f2, f3 : Z_p -> R with sum_x f_i(x) = 0; F(x) = sum_i f_i(x_i); Pi_c = {x in Z_p^3 : x1+x2+x3 = c}; V_i = (1/p) sum_x f_i(x)^2; V = sum_i V_i.

Claim: for ALL c, c' in Z_p,
(-min_{Pi_c} F)(max_{Pi_c'} F) >= V + (2/(p(p+1))) sum_{i<j} sum_a A_i(a) A_j(p-1-a) >= V - (2/(p+1)) sum_{i<j} sqrt(V_i V_j) >= ((p-1)/(p+1)) V,
where A_i is the decreasing rearrangement of f_i.

Sharpness: all three bounds are equalities for the sawtooth of R1 with (c, c') = (1, 2).

**Proof / evidence.**

Write m = -min_{Pi_c} F and M = max_{Pi_c'} F.

Step 1 (ranks). Put U_i(y) = {x : f_i(x) >= A_i(y)} and L_i(y) = {x : f_i(x) <= A_i(y)}. Then |U_i(y)| >= y+1 and |L_i(y)| >= p-y. Let H(y) = sum_i A_i(y_i).

Step 2 (Cauchy-Davenport). For nonempty sets in Z_p, |S1+S2+S3| >= min(p, |S1|+|S2|+|S3|-2).
- If sum y_i >= p-1, then sum |U_i| >= p+2, so U1+U2+U3 = Z_p. Every plane Pi_c' then contains x with x_i in U_i(y_i), and M >= F(x) >= H(y).
- If sum y_i <= 2p-2, then sum |L_i| >= 3p - sum y >= p+2. Every Pi_c contains x with F(x) <= H(y), so H(y) >= -m.
Hence -m <= H(y) <= M whenever p-1 <= sum y <= 2p-2. This is plane-independent.

Step 3 (the test nu). Let nu be uniform on S = {y in [0,p-1]^3 : sum y in {p-1, 2p-2}}.
- |S| = 2 C(p+1, 2) = p(p+1). Use y -> (p-1)-y for the second level.
- For a pair (y_i, y_j): on the level sum = p-1 the pair ranges over {a+b <= p-1}, each point once; on the level 2p-2 it ranges over {a+b >= p-1}, each point once.
- So nu_ij(a,b) = (1 + [a+b = p-1])/(p(p+1)). Its single marginals are (p+1)/(p(p+1)) = 1/p, i.e. uniform.

Step 4 (Bhatia-Davis under nu). On S, (H+m)(M-H) >= 0, so mM >= E_nu H^2 - (M-m) E_nu H.
- E_nu H = sum_i (1/p) sum_a A_i(a) = 0.
- E_nu H^2 = sum_i V_i + 2 sum_{i<j} E_nu[A_i A_j] = V + (2/(p(p+1))) sum_{i<j} sum_a A_i(a) A_j(p-1-a), since the product-measure part vanishes (sum A_i = 0).

Step 5. By Cauchy-Schwarz, sum_a A_i(a) A_j(p-1-a) >= -p sqrt(V_i V_j). Also sum_{i<j} sqrt(V_i V_j) <= V. QED.

Checks (gibbs2/pbd_sharp_check.py, output pbd_sharp_check.out).
- p in {3,5,7,11,13,17,19,23,29}, 8 families x 60 trials, all pairs of planes.
- Step 1 slack min 0 (>= 0). Step 3 marginals and pair law exact (integer counts). Step 4 identity error <= 5e-15.
- Theorem: (mM - bound)/V >= -1.4e-15. The minimum of mM/V over the families equals (p-1)/(p+1), attained by the sawtooth, at every p.
- Adversarial hill-climbing (pbd_adversarial.py): 0.666671 at p=5 and 0.750023 at p=7 (both converge onto the sawtooth from above); 0.855 at p=11. Never below the bound.

*Check:* 

### R4 [PROVED]

Joint detection for class C at n=2, sharp.

Setting: theta in class C, n = 2, p an odd prime, delta != 0. Let A_delta = E_x g(.,delta) - min_x g(x,delta) (<= m_delta) and A'_delta = A_{-delta} (<= m_{-delta}). Let V_T = sum_l |beta_{l,T}|^2 kappa_l(delta) and E_delta = sum_T V_T.

Claim:
m_delta m_{-delta} >= A_delta A'_delta >= E_delta - (2/(p+1)) sum_{T<T'} sqrt(V_T V_T') >= ((p-1)/(p+1)) E_delta.
In particular E_delta <= ((p+1)/(p-1)) mbar_delta^2. The constant is attained by a class-C network (R1).

**Proof / evidence.**

Profile structure.
- In class C, Phi_l = sum_T a_{l,T} rho^{2l s_T(x)} with a_{l,T} = conj(beta_{l,T}) (companion Lemma 11; derived again here from the neuron formula).
- Put phi_T(w) = sum_l Re(a_{l,T} rho^{lw}) and D_T(w) = phi_T(w) - phi_T(w - delta).
- Then Z_delta = g(.,delta) - E g = D_1(2x1) + D_2(2x2) + D_12(2x1+2x2), and Z_{-delta}(x) = -sum_T D_T(2 s_T(x) + delta).

Reduction to R3.
- Set u = 2x1, v = 2x2 (bijections, p odd), f1 = D_1, f2 = D_2, f3(z) = D_12(-z).
- Then A_delta = -min_{Pi_0} F and A'_delta = max_{u,v}[D_1(u+delta) + D_2(v+delta) + D_12(u+v+delta)] = max_{Pi_delta} F.
- V = sum_T Var D_T = sum_{l,T} |a_{l,T}|^2 |c_l|^2/2 = E_delta, since |c_l|^2 = 2 kappa_l and the characters 2l 1_T are distinct and not negatives of each other for l in F.
- R3 applies. Finally mbar >= sqrt(m_delta m_{-delta}) by AM-GM.

Checks (gibbs2/classC_net.py, output classC_net.out). Random class-C networks at p = 5, 7, 11, 13, all delta:
- ridge-sum identity error <= 1.2e-13 and reflection identity error <= 2e-13;
- E formula relative error <= 5e-15;
- min (AA' - bound)/E >= 0.898 and min (m m' - AA')/E >= 0.029;
- sawtooth network: AA'/E = (p-1)/(p+1) to 8 digits.

*Check:* 

### R5 [PROVED]

Decoupling without the 2^n - 1 loss. This extends Note Thm 'decoupling when few features are bought cheaply' and Cor 'Parseval closure' from the relations case to ALL of class C at n=2.

Setting: theta in class C, n = 2, b in [2,3], s = 3/b. Let a_l = 3^{1-s} Q_l^s, L' = {l : |beta_l| > a_l}, R_l^2 = sum_T |beta_{l,T}|^2, and delta* a minimiser of sum_l a_l kappa_l(delta). Suppose |beta_l| - a_l <= c_l R_l for l in L' (any constants c_l >= 0).

Claim: if sum_{l in L'} c_l^2 kappa_l(delta*) < (p-1)/(p+1), then
h(theta) <= W(theta) := min_delta sum_l a_l kappa_l(delta) <= 3^{1-s} B_p Gamma_s(p) ||theta||^3,
and equality forces theta to be interference-free.
With Theorem A (c_l = 1/(2 sqrt3)) the condition reads sum_{L'} kappa < 12 (p-1)/(p+1). The Note's threshold was 4.

**Proof / evidence.**

Step 1. Averaging g(.,+-delta*) over x gives h = sum_l Re(beta_l) kappa_l(delta*) - mbar.

Step 2. Bound the clean amplitudes.
- For l not in L': Re beta_l <= |beta_l| <= a_l.
- For l in L': Re beta_l <= a_l + c_l R_l.

Step 3. Cauchy-Schwarz and R4:
h <= sum_l a_l kappa_l + sqrt(sum_{L'} c_l^2 kappa_l) sqrt(E_{delta*}) - mbar <= sum_l a_l kappa_l + mbar ( sqrt(((p+1)/(p-1)) sum_{L'} c_l^2 kappa_l) - 1 ).
The bracket is < 0, so h <= W(theta).

Step 4. W(theta) <= 3^{1-s} (sum_l Q_l)^s Gamma_s <= 3^{1-s} (C_b P)^s Gamma_s, by the single-neuron budget and C_b^s = B_p.

Step 5 (equality). Equality forces mbar = 0, so E_{delta*} = 0 by R4. Since kappa_l(delta*) > 0 for all l, every beta_{l,T} = 0, so theta is IF.

Checks (gibbs2/classC_net.py, part N4, p = 7..23).
- min[chain RHS - h] >= 1.7e-3.
- min[W - h] >= 2.3e-2 over nets meeting the condition.

*Check:* 

### R6 [PROVED]

SHARP L2 PAYMENT at n=2 (computer-assisted).

Setting: atoms beta_i in C and xi_i, eta_i in T. A = sum beta_i, B = sum beta_i xi_i, C = sum beta_i eta_i, D = sum beta_i xi_i eta_i, Q = sum |beta_i|^{1/s}, R = sqrt(|B|^2 + |C|^2 + |D|^2), theta = 3^{1-s}.

Claim: for all atoms and every s in [1, 3/2],
|A| - theta Q^s <= ((1 - theta)/sqrt3) R.
Equality holds for a single atom. At b=2 the constant is (sqrt3 - 1)/3 = 0.24402, versus Theorem A's 1/(2 sqrt3) = 0.2887. The constant is optimal for every s.

**Proof / evidence.**

Notation: rho_i = |beta_i|, p_i = beta_i/rho_i, L = sum rho_i.

Step 1 (twisted Gram matrix of Note Thm P12).
- u_i = (1, p_i xi_i, p_i^2 xi_i eta_i), H = sum rho_i u_i u_i^*.
- H has diagonal L, H21 = B, H32 = C, H31 = D' = sum rho_i p_i^2 xi_i eta_i.
- Hhat = H/L is a correlation matrix with spectrum nu (sum nu = 3).
- Put b = |B|/L, c' = |C|/L, delta = |D'|/L, beta = sqrt(b^2 + c'^2), P = beta^2 + delta^2, q = 2 Re(Hhat21 Hhat32 conj(Hhat31)).
- nu_k = 1 + t_k, where t_k are the roots of t^3 - P t - q. Also q >= P-1 (det >= 0) and |q| <= 2bc'delta.

Step 2 (cost, Rotfel'd). sum_k lambda_k(H)^r <= sum_i (3 rho_i)^r = 3^r Q, hence theta Q^s >= L M_{1/s}(nu), with M_r(nu) = (mean nu^r)^{1/r}.

Step 3 (entropy/Jensen). Weighted AM-GM with weights nu_k/3 gives M_{1/s}(nu) >= x^Lambda, where x = 3^{1-s} and Lambda = sum_k (nu_k/3) log_3 nu_k = 1 - H_3(nu/3), so Lambda is in [0,1].

Step 4 (gain). For a unit lambda >= 0, R >= lambda.(|B|,|C|,|D|). The Note's Cauchy-Schwarz gain step (|A| - C'|D| <= L sqrt(1 + C'^2 - 2C'delta)) with C' = c lambda_3, and lambda_1 : lambda_2 = b : c', give
|A| - cR <= L G_x, where G_x = sqrt(1 + c^2 lambda_3^2 - 2c lambda_3 delta) - c beta sqrt(1 - lambda_3^2) and c = c(x) = (1-x)/sqrt3.

Step 5 (monotonicity, by divided differences).
- d/dq sum h(1+t_k) = g[t1,t2,t3] with g(t) = 1 + ln(1+t), g'' < 0. So Lambda decreases in q.
- d/dP sum h(1+t_k) = G[t1,t2,t3] with G(t) = t(1 + ln(1+t)), G'' = (2+t)/(1+t)^2 > 0. So Lambda increases in P.
- On q = P-1 the spectrum is ((3 +- sqrt(4P-3))/2, 0), which spreads as P grows; Lambda is Schur-convex.
- Hence, with q* = max(P-1, -beta^2 delta) <= q: Lambda(nu) <= Lambda(P, q*).
- (beta, delta) always lies in Rgn = {0 <= delta <= 1, beta^2 <= 1+delta}, because P - 1 <= q <= beta^2 delta.

Step 6 (certificate at s = 3/2). With c0 = (sqrt3-1)/3 and x0 = 3^{-1/2}: for every (beta, delta) in Rgn there is lambda_3 in [0,1] with G0 <= x0^{Lambda(P, q*)}.
- Region A, P <= 1/25 (analytic). Take lambda_3 = delta/(c0 + 2 beta). Then
  G0 <= 1 - c0 beta - c0 delta^2/(2(c0 + 2beta)),
  x0^Lambda >= 1 - (P + beta^2 delta/2 + P^2/6 + E5)/6, with E5 <= P tau^3/(10(1-tau)) and tau = 2 sqrt(P/3).
  So margin >= beta [c0 - beta/6 - beta delta/12 - kA beta] + delta^2 [c0/(2(c0 + 2beta)) - 1/6 - kA], with kA <= 0.001378.
  On beta, delta <= 1/5 both brackets are >= 0.2071 and >= 0.0214 (interval-checked).
- Region B, P >= 2.71 (analytic). Here q* = P-1 and nu = (3 - sigma, sigma, 0) with sigma <= 1/10.
  Take lambda_3 = 1/sqrt3. Feasibility gives delta >= 1 - sigma and beta >= sqrt(2 - 3sigma + sigma^2), so G0 <= Gbar(sigma), where
  Gbar(sigma) = sqrt((1 - c0/sqrt3)^2 + 2c0 sigma/sqrt3) - c0 sqrt(2/3) sqrt(2 - 3sigma + sigma^2), and Gbar(0) = x0.
  The bound (3-sigma) ln(3-sigma) <= 3ln3 - sigma(1 + ln3) + sigma^2/3 gives x0^Lambda - x0 >= (x0/6)[sigma(1 + ln(3/sigma)) - sigma^2/3].
  So margin >= sigma (0.4203 - max Gbar') >= sigma (0.4203 - 0.3932) > 0.
- Region C (the rest): adaptive boxes.
  G is decreasing in beta and delta, so G <= G(beta_lo, delta_lo, lambda_3).
  Lambda <= Lambda(P_hi, max(q_lo, P_hi - 1)), with q_lo = max(P_lo - 1, -beta_hi^2 delta_hi). This is a path argument using Step 5, valid in both cases q* >= P_hi - 1 and q* < P_hi - 1.
  Roots are isolated by exact rational sign changes (the farthest root; the other two from the quadratic factor). Logs, exps and square roots use mpmath interval arithmetic.
  Result: 210 boxes, 0 failures, minimum margin 3.13e-4. The same holds from starting grids 16 and 32 (255 and 832 boxes).

Step 7 (all s by concavity in x). G_x is convex in x (Euclidean norm of an affine map of x, minus a linear term). x^Lambda is concave. G_1 = 1 = 1^Lambda. So x^Lambda - G_x >= 0 on [x0, 1]. Chaining Steps 4, 3, 2:
|A| - c(x) R <= L G_x <= L x^Lambda <= L M_{1/s}(nu) <= theta Q^s.

Scripts and outputs.
- Certificate: gibbs2/l2pay_cert.py -> l2pay_cert_N8.out, l2pay_cert_fine.out.
- Chain check on 20000 random atom systems, s in [1, 1.5] (l2pay_chain_check.out): every step slack >= -1.4e-10. Step (d) re-checked at 50 digits: >= -1e-15, which is float-rounded inputs at the equality q = q*.
- Monotonicity violations: 0.
- Numerical sup over atoms (payment_l2.out, P1) equals the single-atom value at s = 1.1, 1.2, 1.3, 1.4, 1.5.
- Network check: per feature, cR - (|beta| - a) >= -2.8e-15, with equality at single neurons (classC_closure_check.out, K1).

*Check:* 

### R7 [PROVED]

Class C at n=2 is closed: (IF_C) and hence (C_C).

Main claim: for EVERY prime p <= 31 and EVERY b in [2,3], every class-C network satisfies gamma <= 3^{1-s} B_p Gamma_s(p), with equality only for interference-free networks. Every maximiser within class C is therefore IF and uses exactly |spt y*| features (Paper Thm 8).

More generally the same holds for every prime p with p(p+1)/(2(p-1)) < 3/(1 - 3^{1-s})^2:
- p <= 31 at b = 2;
- p <= 37 for b >= 2.1;
- p <= 53 for b >= 2.2;
- p <= 61 for b >= 2.25;
- p <= 101 for b >= 2.4;
- p <= 151 for b >= 2.5;
- p <= 661 for b >= 2.75.

At every prime, the conclusion holds for networks whose cheap set satisfies sum_{L'} kappa_l(delta*) < 3(p-1)/((p+1)(1 - 3^{1-s})^2). At b = 2 this is 16.79 (p-1)/(p+1); in particular it holds whenever |L'| <= 5, and |L'| <= 7 once p >= 23.

Via SPLIT R9(a,b), for these (p, n=2, b): (C) for ALL networks holds iff every global maximiser is IF. This extends SPLIT R9(d) from p in {5,7} to every p <= 31.

**Proof / evidence.**

Proof.
- Apply R5 with c_l = c(s) = (1 - 3^{1-s})/sqrt3 (from R6) and sum_{l in F} kappa_l(delta) = p/2 for every delta != 0.
- c(s) increases with s, so b = 2 is the worst case.
- Table: gibbs2/closure_table.py -> closure_table.out.
- With Theorem A alone, the range is p <= 19.
- IF networks attain the bound (Paper Thm 8), so class-C maximisers are IF.

Checks (gibbs2/classC_closure_check.py -> classC_closure_check.out).
- p in {11,13,17,19,23,29,31,37}, b in {2, 2.5}:
  - (K1) payment holds per feature;
  - (K2) min[chain - h] >= 1.4e-3;
  - (K3) min[W - h] >= 3.7e-2 on every net meeting the condition.
- (K4) Local adversarial search at p = 29, 31, b = 2: best (h - W)/W = -0.99 and -1.20 (the theorem requires <= 0).
- Consistent with earlier numerics: all optima found at p = 5, 7, 11 are IF.

*Check:* 

### R8 [PROVED]

S_3 symmetry, single-T features, and an improved near-IF buffer.

(a) The payment problem sup{|A| - theta Q^s : |B|, |C|, |D| given} is symmetric in (|B|, |C|, |D|).

(b) Hence P12 (c12 = 1 - (2/3)^{s-1}) also holds when C = D = 0 or B = D = 0. Two consequences:
- The Note's relations-case closure (p <= 59, or sum_{L'} kappa < 29.7, at b = 2) holds verbatim when all interference sits on T = {1} or on T = {2}.
- A class-C network in which EACH cheap feature's interference lies on a single slot-set (different features may use different T) is closed when sum_{L'} kappa < (p-1)/((p+1) c12^2). This covers every p <= 53 at b = 2, and every b in [2,3].

(c) Symmetrised quadratic payment: |A| - theta Q^s <= Ksym R^2/L, with Ksym = (2(s-1)/3)(1 + eta/3) and eta = 4(s-1)/(7-4s). Ksym = 5/9 at b = 2, versus the Note's K2 = 1.

(d) Near-IF rigidity in class C (Note Thm nearif(b)) holds under (sum_l E_l kappa_l)^{1/2} < sqrt((p-1)/(p+1)) min_{L'} L_l / Ksym. This buffer is about 3x larger than the Note's.

**Proof / evidence.**

(a) Write beta_i = r_i x_i Y_i Z_i with unit x, Y, Z, where xi = conj(Y)^2 and eta = conj(Z)^2. Then (A, B, C, conj D) = sum r_i (xYZ, x conj(Y) Z, x Y conj(Z), conj(x) Y Z). Swapping x and Y keeps |A|, Q, L, |C| and swaps |B| with |D|; swapping x and Z swaps |C| with |D|.

(b) P12 + (a) gives c12 per single-T feature. Then use R5 with c_l = c12; for the uniform single-T case, use per-T Bhatia-Davis exactly as in the Note.

(c) The Note's proof of Prop quadpay(i) gives |A| - theta Q^s <= (2(s-1)/(3L))(|B|^2 + |C|^2 + (1+eta)|D|^2). I re-derived each step: Rotfel'd, Jensen, e^{-y} >= 1-y, nu log nu <= (nu-1) + (nu-1)^2, the |D'| <= |D| + sqrt(2L(L - |A|)) step, and the eta choice. Apply it to the three S_3 images and keep the minimum: R^2 + eta min(|B|^2, |C|^2, |D|^2) <= (1 + eta/3) R^2.

(d) As in the Note, with R4 in place of the factor 3 and (c) in place of K2.

Checks.
- S_3 symmetry: deviation 3.6e-15 over 2000 random systems (payment_sym.out).
- Symmetrised quadratic bound: adversarial ratio <= 0.796 for s in [1.05, 1.5] (payment_quad.out).

*Check:* 

### R9 [PROVED]

Slicing lemma, general n.

Setting: D_T : Z_p -> R mean zero for each nonempty T in [n]; Z(x) = sum_T D_T(s_T(x)); Z^d(x) = sum_T D_T(s_T(x) + d); kappa = (p-1)/(p+1).

Claim: for every pair i != j,
(-min Z)(max Z^d) >= kappa sum_{T meets {i,j}} V_T + sum_{T avoids {i,j}} C_T(d),
where C_T(d) = E_w[D_T(w) D_T(w+d)] = V_T - sum_l |beta_{l,T}|^2 kappa_l^2 for class-C profiles. Also (-min Z)(max Z^d) >= max_T V_T.

Guaranteed joint constants (worst case over energy splits, C_T = -V_T) for kappa -> 1: 1/3, 0.182, 0.100, 0.048, 0.029 for n = 3..7, versus the Note's 1/(2^n - 1) = 1/7, 1/15, 1/31, 1/63, 1/127. For n = 3 the constant is kappa/(2+kappa).

**Proof / evidence.**

Fix x'' = (x_k)_{k not in {i,j}}.
- The slice is an n=2 ridge sum Psi(x'') + G_1(x_i) + G_2(x_j) + G_12(x_i + x_j), where G_S(w) = sum_U D_{S u U}(w + s_U(x'')) and Psi(x'') = sum_{U nonempty} D_U(s_U(x'')).
- Z^d on the same slice is Psi_d + G(. + d) in all three core arguments.
- R3's Steps 1-2 give, for all slices x'' and all y in spt nu: Psi(x'') + H^{x''}(y) >= -A and Psi_d(x'') + H^{x''}(y) <= A'.
- Expand E_{x'', nu}[(Psi + H + A)(A' - Psi_d - H)] >= 0. The cross terms vanish: E_y H = 0 for each slice, E Psi = E Psi_d = 0, and distinct s_U are independent.
- This gives AA' >= E[H^2] + sum_U C_U(d) >= kappa E_{x''} V^{x''} + sum_U C_U(d), and E_{x''} V^{x''} = sum_{T meets {i,j}} V_T.
- Per-T: condition on s_T.
- Constants: the linear programme in gibbs2/slice_lp.py -> slice_lp.out.

Checks.
- gibbs2/slice_n3.py -> slice_n3.out: n = 3 (p = 5, 7, 11) and n = 4 (p = 5, 7). min (AA' - best bound)/E >= 0.31.
- The previous agent's fibre certificates (its star splits are the 1-dimensional-core case) also re-check fine: check_fibre_rerun.out, slack >= 0.

*Check:* 

### R10 [PROVED]

Quantitative transfer to general networks at n=2 (this is what SPLIT R7(c) can give here).

Setting: any network theta at n = 2, with theta_split its frequency split and g_R = g_theta - g_split the gap of the mixed modes. Let rho = (1/2)[max_x g_R(x, delta*) + max_x g_R(x, -delta*)].

Claim: if c(s) sqrt(sum_{L'} kappa_l) sqrt(E_{delta*}) + rho <= sqrt(((p-1)/(p+1)) E_{delta*}), then h(theta) <= W(theta) <= IF value * ||theta||^3.

**Proof / evidence.**

- The split keeps the atoms, beta_{l,T}, a_l, E_x g and the cheap set (SPLIT R1), and does not raise the cost.
- The all-DC table is constant in c, so g_theta = g_split + g_R.
- min_x(g_split + g_R) <= min g_split + max g_R, hence m_{+-delta}(theta) >= A_{+-delta}(split) - max g_R(.,+-delta).
- So mbar(theta) >= sqrt(A A'(split)) - rho >= sqrt(kappa_p E) - rho. Then run R5.

Obstruction to anything stronger: R4 uses point extrema on all of Z_p^2. For mixed T, the span V_delta of the class-C characters is Z_p^2, so SPLIT R7(c) transfers it only to networks without mixed modes. SPLIT R10 shows that all-network decoupling is false, so only this perturbative form can hold without global optimality.

*Check:* 

## Obstruction
1. The L2 route at b=2 stops at p=31, and this is sharp on both sides.
- Detection: (p-1)/(p+1) is attained by the sawtooth profile, realised by a class-C network.
- Payment: (1-3^{1-s})/sqrt3 is attained by a single neuron, and is certified as the supremum.
- So any argument that bounds the cheap gain by per-feature L2 payments and the slack by an energy-only detection closes exactly the primes with p(p+1)/(2(p-1)) < 3/(1-3^{1-s})^2. That is p <= 31 at b=2, and p <= 61 only for b >= 2.25.
- The orchestrator's expected "p <= 59 for all of class C at b=2" assumed P12's constant c12 = 0.1835. That constant is valid only for features whose interference sits on one slot-set (R8: p <= 53). A single neuron spreads its interference equally over all three T and has payment ratio 0.244 > c12.
- Beyond p=31 at b=2, a proof must see PHASES: single-neuron features have product fields, (1+zeta1 rho^u)(1+zeta2 rho^v), while the detection-extremal profile is a sawtooth. Whether these two extremals are compatible is the open point.

2. Transfer to general networks.
- The joint detection uses point extrema on all of Z_p^2. SPLIT R7(c) is therefore vacuous for mixed T: the span of the class-C characters is everything.
- Only the perturbative R10 holds. (C) at (p,2), p <= 31, is now exactly (MM_opt): every global maximiser is IF.

3. n >= 3.
- There is no Cauchy-Davenport analogue for the 2^n-1 ridge forms on Z_p^n (codimension 2^n-1-n). The union-type bounds are too weak, because the two sides need sum of ranks >= ~(2^n-1-n)(p-1) and <= ~n(p-1) at the same time.
- Slicing to an n=2 core gives only constant-factor gains, and payment constants for n >= 3 are not sharp.

## Next steps
- Class C, n=2, b=2, 37 <= p <= 59. Two complementary steps:
- Numerically maximise sum_{L'} kappa_l (|beta_l| - a_l) - mbar_{delta*} with the EXACT mbar over class-C networks (e.g. one or two neurons per frequency), to measure how much the phase structure leaves.
- If single-neuron (product) features cannot form sawtooth profiles, prove a 'product-field detection' lemma. For instance, apply the rank-plane test to profiles whose T={1,2} coefficient is the product of the {1} and {2} coefficients over the clean one.
- (MM_opt) for n=2, p <= 31, all b: with IF_C now proved there, SPLIT R9 makes (C) equivalent to 'every global maximiser is IF'. Attack it with the channel arithmetic (partner 3l, foreign 2l+-j) together with the quantitative transfer R10.
- n = 3, two steps:
- Build a twisted 4x4 Gram matrix with the entropy bound (as in R6) to get a sharp L2 payment at n=3.
- Combine it with the slicing lemma R9 (joint constant kappa/(2+kappa)) to restate Note Thm loc at n=3 with explicit constants.
- Look for a Cauchy-Davenport/Pollard-type theorem for the 7 ridge forms of class C at n=3, which would push the joint detection constant toward 1.
- Referee items for the orchestrator:
- Every inherited claim in gibbs/ is now settled. PBD is proved; its proof was reconstructed in R2. PB2 is refuted (R1). The jl2 'min 1.0003' is wrong (true minimum (p-1)/(p+1)). closure_n2b's 'p=11 CLOSED / p=13 not closed' is superseded by R7.
- All scripts and outputs are in /home/user/grok/heracles/conjecture20/unify/gibbs2/. Key files: pbd_sharp_check.py, classC_net.py, l2pay_cert.py, classC_closure_check.py, closure_table.py, slice_n3.py, slice_lp.py, payment_l2.py, l2pay_chain_check.py, l2pay_stepd_hp.py, plus PROGRESS_NOTES.txt.
