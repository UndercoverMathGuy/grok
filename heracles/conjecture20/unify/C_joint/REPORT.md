# C_joint: report (saved by the orchestrator from the agent's structured output; referee did not run: session limit)

## Headline
JOINT (class C first, then all networks). I did not prove RD. What I found:

(1) Detecting many features together is not the obstacle. One inequality (FR) catches all features at once with no loss and has the same shape as the joint payment: the gain is at most the cost-weighted mean of one quantity per feature. FR holds for every network, every odd p, every n >= 2 and every b in [2, nu], and is an equality at interference-free (IF) optima.

(2) What FR leaves is one quantity per feature plus a pointwise covering question. Every cheap feature in every stored non-IF optimum has field-row efficiency Xi_l between 1.016 and 1.33, so the minimum over x is still needed. The exact sufficient condition is the covering statement PW: some point x has nonpositive excess.

(3) The relations core is present at every n. A sharp single-set payment for aligned atoms is now proved for every n, with constant c_n(s) = 1 - (n/nu)^{s-1}; at n = 2 it is P12 restricted to aligned atoms. With it, the aligned single-set class at (p, n, b) is exactly the one-variable relations problem with amplification (1+1/n)^{s-1}. That amplification grows to e^{1/2} on the b = 2 diagonal.

(4) The brief's three new directions all fail as general mechanisms:
- (1) and (2): an explicit Legendre-symbol network is flat at every offset at once. Its depth is (2/sqrt p) times the interference the payment charges. So no detection inequality that ignores how efficient the allocation is can hold for all p.
- (3): the offset phases act as one common translation exactly when the interference family is balanced. Every single-set network is balanced, so the offset arithmetic gives no rigidity where the hard core lives. For unbalanced families the translation is lost, and so is the oscillation bound.

Numerically no failure of RD was found.

All files are in /home/user/grok/heracles/conjecture20/unify/C_joint/. Proofs are in joint_proofs.tex.

## Principle
THE l_r CURRENCY MAKES DETECTION ADDITIVE; WHAT REMAINS IS POINTWISE AND ARITHMETIC.

(i) The field at a point is itself an allocation.
- Pair delta with -delta at any fixed x: h <= sum_l Re Phi_l(x) kappa_l(delta).
- So the field row (Re Phi_l(x))_l is a clean allocation whose coverage bounds the margin.
- The programme then gives h <= Gamma_s (sum_l (Re Phi_l(x))_+^r)^s.

(ii) Averaging is free in this currency.
- The r-th powers are summed over features before anything else. So the x-average is exactly the sum of single-feature averages: no cross terms and no loss of sqrt(K).
- The averaged bound costs only the step min over x <= mean over x, but that step is lossy for features with low-field points (part iii).
- Concavity of t^r turns each feature's fluctuation into a quadratic loss r(1-r)E_l/(4|A_l|^2), matching the quadratic payment.
- The result (FR) says: gain <= cost-weighted mean of Xi_l = G_l^r D_l, where G_l is cheapness and D_l is a per-feature detection factor.

(iii) What averaging cannot see is the minimum over x.
- Two kinds of feature have Xi_l > 1: single-set ('relations-type') features, and wild features whose field leaves the right half plane.
- For these the margin is decided at the points where their field is small. A network beats IF only if the other features' excess covers every such point (PW).

(iv) The core is one-dimensional at every n, and the same Gauss sums control flatness and efficiency.
- For single-set networks the interference is an exact coboundary in one variable w = 2x_T.
- The delta-phases are a common translation there (balanced family), so slot-wise tools (Riesz products, Bonami) and delta-arithmetic see nothing.
- Legendre interference is flat at every offset.
- The cyclotomic supports that carry flat interference are themselves efficient covers with a defect of the same order p^{-1/2}. The missing inverse theorem must therefore tie flatness (or covering) to coverage efficiency.

## Results

### J1 [PROVED]

FIELD-ROW THEOREM (FR): a joint detection inequality with the shape of the joint payment. It holds for EVERY product network (any inputs, outputs, DC, width), every odd prime p, every n >= 2 and every b in [2, nu].

Definitions:
- psi_l(x) = (Re Phi_l(x))_+^r, with r = 1/s;
- M_l = E_x psi_l(x);
- Q_l = sum_i |beta_il|^r (clean atoms);
- Xi_l = nu^{1-r} M_l / Q_l.

Then:
- h(theta) <= Gamma_s (min_x sum_l psi_l(x))^s <= Gamma_s (sum_l M_l)^s;
- gamma(theta; s) <= B_p Gamma_s (sum_l M_l / sum_l Q_l)^s = gamma_IF (sum_l Q_l Xi_l / sum_l Q_l)^s.

The sums run over all of F. Non-features have Q_l = 0 and enter only the numerator.

Variants:
- class C with features in S: Gamma_s may be replaced by Gamma^{(S)}_s;
- feature weights lambda: replace Gamma_s by Gamma^lambda_s.

Equality holds for IF networks at programme allocations.

Reading:
- Write G_l = |A_l|/(nu^{1-s} Q_l^s) for the cheapness. Then Xi_l = G_l^r D_l with D_l = E_x (Re Phi_l)_+^r / |A_l|^r.
- D_l = 1 is Lemma 2 plus the budget.
- For a small circular fluctuation, D_l = 1 - r(1-r) E_l / (4|A_l|^2) + O(|f_l|^3). This is a quadratic detection that adds over features with no loss of sqrt(K).
- So RD-IF reduces, for networks whose features satisfy Xi_l <= 1, to single-feature inequalities.

**Proof / evidence.**

1. Fix x and delta != 0. The field decomposition (valid for every network) gives c_l(delta) + c_l(-delta) = 2 kappa_l(delta). Hence h <= min(g(x,delta), g(x,-delta)) <= (g(x,delta) + g(x,-delta))/2 = sum_l Re Phi_l(x) kappa_l(delta).
2. Put y_l = Re Phi_l(x). Since kappa >= 0, min_delta sum_l y_l kappa_l(delta) <= min_delta sum_l (y_l)_+ kappa_l(delta).
3. If y_+ != 0, the vector y_+/(sum (y_l)_+^r)^s is feasible for Gamma_s, so the right side is <= Gamma_s (sum_l psi_l(x))^s. This also holds trivially if y_+ = 0.
4. Take the minimum over x. Since min <= mean, linearity of E_x gives the averaged bound.
5. Cost: the single-neuron budget sum_l |beta_il|^r <= B_p^r ||omega_i||^b holds for every neuron, so sum_l Q_l <= B_p^r P_b and P_b^s >= B_p^{-1}(sum_l Q_l)^s. Also (nu^{r-1})^s = nu^{1-s}.
6. Equality: an IF network at a programme allocation has Phi_l = A_l > 0 constant and Q_l = nu^{1-r} A_l^r (KM2), so every step is an equality.
7. Restricted case: in class C, Phi_l = 0 off S.
8. The expansion of D_l is a Taylor expansion together with circularity, E(Re f)^2 = E|f|^2/2.

*Check:* Scripts in /home/user/grok/heracles/conjecture20/unify/C_joint/:
- check_FR.out:
  - first inequality on 120 random networks (class C and general): max relative violation -6.3e-3;
  - FR on 36 positive-margin class-C perturbations of IF optima: max gamma/(gamma_IF avgXi^s) - 1 = -2.7e-2;
  - avgXi^s = 1.00000000 on 5 IF optima.
- check_FR_mixed.out: 73 mixed/DC perturbations of IF networks, max -3.3e-2. Also 28 stored E_kkt local maxima: gamma/FR <= 1.0000, with equality on the IF ones.
- check_FR_adv.out:
  - the 12 stored SPLIT/E_dual violators at p = 5, 7, 11 (including the (5,2,2) violator and the D8 networks): gamma/FR in [0.47, 0.9997];
  - the (5,6) width-6 network: 0.31;
  - the p=3 winners at n = 11, 13: 0.238 and 0.217. FR does not imply IF at p = 3, because there Xi^s = 4.4 and 5.1, consistent with BRIEF_CORE section 3.

### J2 [PROVED]

ALIGNED SINGLE-SET PAYMENT, EVERY n. This generalises P12 (restricted to aligned atoms) from n = 2 to all n.

Hypotheses:
- atoms beta_i = rho_i alpha with rho_i > 0 and a common phase |alpha| = 1;
- xi_it unimodular;
- for a fixed nonempty slot set T, B_{T'} = sum_i beta_i prod_{t in T'} xi_it = 0 for every nonempty T' != T.

Put L = sum rho_i = |A|, D = B_T, theta = |D|/L, Q = sum rho_i^r. Then for every r in (0,1]:
  nu^{1-s} Q^s >= L F_n(theta) >= L - c_n(s)|D|,
where
- F_n = nu^{1-s} Phi_n^s;
- Phi_n(theta) = (1/2)[(1+theta)^r (nu+(n-1)theta)^{1-r} + (1-theta)^r (nu-(n-1)theta)^{1-r}];
- c_n(s) = 1 - (n/nu)^{s-1}.

Sharpness:
- both inequalities are equalities at theta = 0 (the KM2 block);
- both are equalities at theta = 1 (n equal atoms at n-th roots of unity);
- the bound does not depend on |T|.

Also c_n(s) < 1 - e^{-1/2} < 1/2 for every n and every s <= nu/2.

**Proof / evidence.**

Chain matrix.
- Order the slots so that T = {1, ..., k}.
- Let m_i = (1, xi_i1, xi_i1 xi_i2, ...) and H = sum rho_i m_i m_i^*, which is PSD.
- For t > t', H_{tt'} = conj(alpha) B_{[t'+1,t]}. This vanishes unless (t, t') = (k, 0).
- So H = L I + conj(alpha) D e_k e_0^* + h.c.

Leverage (theta < 1).
- rho_i m_i m_i^* <= H gives rho_i w_i <= 1, with w_i = m_i^* H^{-1} m_i.
- The 2x2 block inverse gives L w_i = (nu - 2) + 2(1 - theta X_i)/(1 - theta^2), where X_i = Re(alpha conj(D) y_i)/|D| is in [-1, 1].
- Moreover sum_i rho_i X_i = L theta.

Cost.
- rho_i^r >= rho_i w_i^{1-r}, so Q >= L^{r-1} sum_i rho_i psi(X_i), with psi concave.
- Among laws on [-1,1] with mean theta, the two-point law on {-1, 1} minimises E psi. This gives Q >= L^r Phi_n(theta).
- At theta = 1 use the pseudo-inverse H^+: all X_i = 1, w_i = n/L, so Q >= n^{1-r} L^r.

Concavity of F_n.
- Write Phi = (1/2)(G_+ + G_-) with G_pm = u^r v^{1-r}, u = 1 pm theta, v = nu pm N theta, N = n - 1.
- Then G'' = -r(1-r) G (u'/u - v'/v)^2, and u'/u - v'/v = pm 2/(uv).
- F_n'' <= 0 is equivalent to Phi'^2 <= 4r^2 Phi sum_pm G_pm/(u_pm v_pm)^2.
- By Cauchy-Schwarz it suffices that N theta - 2r <= (N + 2r) T <= N theta + 2r, where T = tanh((1-r) artanh(theta) + r artanh(eps theta)) and eps = N/(N+2).
- Lower bound: tanh is concave on [0, inf), so T >= theta(1 - 2r/(N+2)), and (N + 2r)(1 - 2r/(N+2)) = N + 4r(1-r)/(N+2) >= N.
- Upper bound: artanh((N theta + 2r)/(N + 2r)) = (1/2) log((N(1+theta) + 4r)/(N(1-theta))) is concave in r. It equals the affine function (1-r) artanh(theta) + r artanh(eps theta) at r = 0 and exceeds it at r = 1, hence on all of [0,1].
- So F_n is concave, and F_n >= its chord 1 - c_n theta.

The bound on c_n: c_n(s) <= 1 - (n/(n+1))^{(n-1)/2} < 1 - e^{-1/2}.

Full proof: joint_proofs.tex, Theorem 2.

*Check:* - phi_check.out: F_n - (1 - c_n theta) on a 20001-point grid, n <= 60, 60 values of s. Minimum interior slack 8e-11 >= 0; maximum second difference 8e-16 (concave).
- phi_suff.out: the tanh sufficient condition for n <= 400 and r in (0,1]. Minimum slacks 2e-13 and 2e-4 (both >= 0).
- pay_n2.out and pay_n3.out: constrained optimisation over aligned atom systems (m <= 2n; T = [n], {1}, {1,2}). The maximum of (|A| - c_n|B_T|)/(nu^{1-s} Q^s) is 1.000000, attained at the IF block and at the n-atom block.

### J3 [NUMERICAL]

The single-set payment of J2 also holds for NON-aligned atoms at n = 3. At n = 2 this is the proved P12. Conjectured for all n.

**Proof / evidence.**

- payment_T2.py uses penalty continuation with BFGS and random non-aligned starts, m in {3, 4, 5, 6}, T = {1,2,3} and T = {1}, s in {1.5, 1.75, 2}.
- The maximum ratio is 1.000000 in every case, with residual constraint violation <= 1e-8.
- Proof obstruction: the P12 twist u_{i,t} = p_i^t prod xi controls only adjacent chain entries, so for n >= 3 the entries of length >= 2 are not sign-flip coefficients.

*Check:* /home/user/grok/heracles/conjecture20/unify/C_joint/pay_n3.out; pay_n2.out (n = 2 reproduces P12).

### J4 [PROVED]

THE RELATIONS CORE IS PRESENT AT EVERY n (single-set embedding).

Construction. For l in F take alpha_l, beta_l >= 0 and phi_l real. Feature l is carried by:
- (i) the KM2 block: nu balanced class-C neurons with atoms alpha_l/nu and xi_it = e^{2 pi i k/nu};
- (ii) n balanced neurons with atoms beta_l/n, xi_it = e^{2 pi i k/n}, and an extra factor e^{i phi_l} in slot 1.

Then, with w = 2y(x):
- Phi_l(x) = alpha_l + beta_l + beta_l e^{i phi_l} rho^{lw};
- gamma = B_p min_{w,delta} sum_l [(alpha_l + beta_l) kappa_l(delta) + beta_l Re(e^{i phi_l} rho^{lw} c_l(delta))] / (nu^{1-r} sum alpha_l^r + n^{1-r} sum beta_l^r)^s.

Consequences:
- RD restricted to class C at (p, n, b) implies RT_n(p, s): this ratio is <= pi_{n,b}(p) B_p Gamma^{(S)}_s for all (alpha, beta, phi).
- In IF-equivalent units the single-set part has amplification tau_n(s) = (1+1/n)^{s-1} = 1/(1 - c_n(s)). By J2 no aligned single-set atom system does better.
- RT_2 is exactly the relations case O1. For s <= 1.5, n = 2 is the hardest. For s > 1.5 the instances are new; on the b = 2 diagonal the amplification increases to e^{1/2} = 1.6487 (n=3: 4/3; n=5: 1.44; n=7: 1.493).

**Proof / evidence.**

- A balanced class-C neuron at frequency l has field c_i prod_t (1 + xi_it rho^{2l x_t}), with |c_i| = B_p ||omega_i||^nu and ||omega_i||^b = (|c_i|/B_p)^r.
- Block (i): B_{T'} = (alpha/nu) sum_k e^{2 pi i k|T'|/nu} = 0 for 1 <= |T'| <= n.
- Block (ii): B_{T'} = (beta/n) e^{i phi [1 in T']} sum_k e^{2 pi i k|T'|/n}. This is 0 unless |T'| = n, where it equals beta e^{i phi}, and rho^{2l sum_t x_t} = rho^{lw}.
- Add the fields and the costs nu(alpha/nu)^r + n(beta/n)^r.
- The amplification follows from n^{1-r} = nu^{1-r}(n/nu)^{1-r} and (n/nu)^{(1-r)s} = (n/nu)^{s-1}.

*Check:* check_embed.out: 28 random networks with p <= 13 and n <= 4. The explicit tables, run through the independent E_width Net code, reproduce the margin and gamma to 3.9e-14 relative.

### J5 [PROVED]

DIRECTION (3) SETTLED: the offset phases act as a translation exactly for balanced families.

Setup: f_l(x) = sum_{T in calT} B_{l,T} rho^{2l x_T} and d_delta = sum_l Re(f_l c_l(delta)).

Translation lemma. Suppose some u in Z_p^n has <1_T, u> = 1 for every T in calT ('calT balanced'). Then d_{-delta}(x) = -d_delta(x + (delta/2)u).

Consequence: every class-C network with a balanced interference family satisfies the oscillation bound at every n:
  h <= sum_l Re A_l kappa_l(delta) - (1/2) osc_x d_delta <= sum_l Re A_l kappa_l(delta) - (sum_l kappa_l(delta) E_l)^{1/2}.

Which families are balanced:
- calT is balanced iff no relation sum lambda_T 1_T = 0 has sum lambda_T != 0 (mod p).
- Every single-set family is balanced.
- Every single-level family with p not dividing |T| is balanced, even with |T| >= 2. So the brief's premise that no common translation exists for |T| >= 2 is too strong.
- Mixed families such as {1}, {1,2} can also be balanced.

Answer to direction (3): this rigidity does NOT prevent simultaneous flat-bottoming.
- The hard core (J4) lives in single-set networks, which are balanced: for them the offsets differ only by a translation and real weights.
- J6 gives a single-set network flat at every offset.
- For unbalanced families the translation, and with it the oscillation bound, is lost. That helps the network, not detection.

**Proof / evidence.**

- Translation: f_l(x + (delta/2)u) = rho^{l delta} f_l(x) and c_l(-delta) = -rho^{l delta} c_l(delta).
- Margin bound: take separate minimisers for plus and minus delta. Then min d_{-delta} = -max d_delta. Average the two mean gaps, which gives sum Re A_l kappa_l.
- Then Bhatia-Davis, and Parseval E d_delta^2 = sum_l kappa_l E_l, because the characters are distinct.
- Balance is the solvability of a linear system over Z_p (Fredholm alternative).

*Check:* check_translation.out.
- Balanced families ({1},{2}; {1,2}; all 2-sets at n = 3; singletons at n = 3; {1},{1,2}): identity error <= 1.3e-13, and the oscillation bound holds exactly.
- NUMERICAL counterpart: for the unbalanced family {1},{2},{1,2} at n = 2 the oscillation bound fails by up to 17.6% (200 random networks; worst case at p = 11, delta = 3).

### J6 [PROVED]

DIRECTIONS (1) AND (2) SETTLED NEGATIVELY: a network flat at every offset.

Construction: in J4 take alpha = 0 and beta_l e^{i phi_l} = 2 chihat(l), where chi is the Legendre symbol mod p (so |beta_l| = 2/sqrt p). The interference potential is then chi itself.

For EVERY delta != 0:
  depth_delta := max_w [chi(w - delta) - chi(w)] <= 2 = (2/sqrt p) sum_{l in F} |beta_l| kappa_l(delta).
This holds for every odd p and every n.

Consequently this inequality is REFUTED for every constant c and every nonempty offset set Delta:
  'depth_delta >= c sum_l v_l kappa_l(delta) for some delta in Delta'.
It fails once p > 4/c^2. The same applies to the analogous inequalities obtained from:
- Riesz products, Lambda(p) or Khintchine bounds;
- Bonami hypercontractivity over the slots;
- Paley-Zygmund or second-moment arguments;
- tilted or extreme-value test measures.
So any joint detection that proves RD must use the efficiency of the allocation.

Slot-wise tools are blind here in addition: the fluctuation depends on x only through the single linear form x_T.

**Proof / evidence.**

- chi is real with chihat(0) = 0 and |chihat(l)| = p^{-1/2} (Gauss sum).
- So chi(w) = sum_{l in F} 2 Re(chihat(l) rho^{lw}).
- Also |chi| <= 1 and sum_{l in F} kappa_l(delta) = p/2.
- This strengthens arith2 R10 from a single offset to all offsets, and from n = 2 to every n.

*Check:* cyclotomic.out part (a): for p = 11, 13, 101, 103, 401, 1009, 1013, max |phi - chi| <= 5e-13, and depth/(sum v kappa) = 2/sqrt p exactly at every delta (max = min).

### J7 [PROVED]

PARSEVAL CLOSURE AT EVERY n (generalises the Note's Cor. parseval).

Hypotheses:
- every feature of a class-C network is an aligned single-set atom system;
- the family {T_l} is balanced;
- L' is the cheap set (features with B_{l,T_l} != 0).

Conclusion: if c_n(s)^2 max_delta sum_{l in L'} kappa_l(delta) <= 1, then gamma <= gamma_IF^{(S)}. In particular this holds:
- whenever |L'| <= 3 (because c_n < 0.3935 and kappa <= 2);
- whenever p < 2/c_n(s)^2.

**Proof / evidence.**

- By J2, Re A_l <= tilde a_l + c_n v_l, where tilde a_l = nu^{1-s} Q_l^s and v_l = |B_{l,T_l}|.
- By J5 with E_l = v_l^2: h <= sum tilde a_l kappa_l + c_n sum_{L'} v_l kappa_l - (sum v_l^2 kappa_l)^{1/2}.
- By Cauchy-Schwarz, h <= sum tilde a_l kappa_l(delta) at delta = argmin sum tilde a kappa.
- Finish with Gamma^{(S)} and the budget.

*Check:* check_parseval.out: 150 explicit networks with p in {7, ..., 19} and n in {2, 3, 4, 5, 7}, half of them phase/amplitude-optimised, with |L'| <= 3 or p < 2/c_n^2. Max gamma/gamma_IF = 0.999987 <= 1.

### J8 [PROVED]

p-UNIFORM GAIN CAP FOR ALIGNED SINGLE-SET NETWORKS.

For any number of features and any sets T_l:
  gamma <= gamma_IF Xi*_{n,p}(s)^s,
where Xi*_{n,p}(s) = sup over theta in [0,1] and phase of [(1/p) sum_{k in Z_p} (1 + theta cos(phase + 2 pi k/p))_+^r] / F_n(theta)^r.

Numerical values of the constant:
- n = 2, b = 2: 1.0536 at p = 101, giving a gain cap of 1.081 (arith2 R6 had 1.108 at n = 2);
- n = 3, b = 2: 1.0396;
- Xi* = 1, which closes the class, at e.g. (n=4, s=1.5) and (n=5, s=2) for p >= 7, at (n=8, s=2) for every p >= 3, and at (n=20, s=3);
- Xi* > 1 near b = 2 for small n or small p.

**Proof / evidence.**

- Re Phi_l depends on x only through lw, with w = 2x_T uniform on Z_p.
- Re A_l <= L_l, and J2 gives Q_l >= nu^{1-r} L_l^r F_n(theta_l)^r.
- Insert these into J1.

*Check:* xi_pp.out (p-point constants for p = 3 ... 101) and xi_single_T.out (torus constants for n <= 30). These are NUMERICAL evaluations of the constant; the theorem is the inequality.

### J9 [PROVED]

ONE CHEAP SINGLE-SET FEATURE IS ALWAYS CAUGHT. This is the single-feature case of the covering form PW.

Define the excess function Psi(x) = sum_l [(Re Phi_l(x))_+^r - nu^{r-1} Q_l]. By J1, beating gamma_IF requires Psi > 0 at every x.

If all features but one are interference-free and the remaining one is an aligned single-set system, then min_x Psi <= 0, so gamma <= gamma_IF. This holds at every odd p and every n.

**Proof / evidence.**

- IF features: (Re A_l)^r <= nu^{r-1} Q_l by Lemma 7.
- The single-set feature has Re Phi = Re A + v cos(phase + 2 pi lw/p). As lw runs over Z_p, some x gives Re Phi <= L - v cos(pi/p).
- J2 gives nu^{1-s} Q^s >= L - c_n v >= L - v cos(pi/p), because c_n < 1/2 <= cos(pi/p).
- Raise to the power r.

*Check:* check_PW1.out: 3000 explicit networks, p in {3, 5, 7, 11, 13, 31, 101}, n in [2, 15], random s. Max of min_x Psi / (nu^{r-1} Q_cheap) = -1.05e-3 <= 0.

### J10 [NUMERICAL]

WHAT FR LEAVES ON REAL NETWORKS: per-feature field-row efficiencies.

Stored non-IF optima and violators:
- every cheap feature (G_l > 1) has Xi_l in [1.016, 1.331];
- every non-cheap feature has Xi_l in [0.960, 0.986];
- example, the (5,2,2) violator: cost shares (0.748, 0.252), Xi = (0.985, 1.078).
- So FR alone leaves 1-22% of slack. The real loss sits in the minimum over x (compensation fails at the low-field points).

Single atoms are the wildest features found. Xi of one atom (torus model):
- 1.24 at (n, b) = (2, 2);
- 1.50 at (3, 2);
- 1.76 at (4, 2);
- 2.03 at (5, 2);
- 2.63 at (7, 2);
- 3.68 at (10, 2).

At n = 2 the supremum over 1-3-atom features is the single atom: 1.2374 at b = 2.

The pointwise form of the FR bound is <= gamma_IF on all 40 stored networks (max Xi_min^s = 1.0000, attained at IF networks). It is tight (h/Gamma(min_x sum psi)^s = 1.0000) for several non-IF local maxima. At p = 3 it is not <= gamma_IF for the DC winners: 1.058.

**Proof / evidence.**

Exact field rows on Z_p^n for the tables in E_kkt/lm_*.npz and split/viol_*.npz, and Monte Carlo on the torus for single atoms. Nelder-Mead over m-atom systems at n = 2.

*Check:* xi_of_networks.out, check_PW.out, xi_atom.out, xi_general.py (log of the n = 2 part in the task output).

### J11 [NUMERICAL]

SETTLING THE FLAT-BOTTOM HEURISTIC OF BRIEF_CORE, as far as numerics go: no failure of RD was found.

(a) Single-set family searches:
- p <= 61, with exact Gamma for p <= 23;
- n in {2, 3, 5, 7, 15} at b = 2;
- the optimiser always removes the interference, and best R <= 1.

(b) Legendre network on all of F: R = tau_n(s)(1 - 2/sqrt p) e(s,p) <= 0.97 up to p = 4007.

(c) Cyclotomic constructions (quartic or octic character on index-2 or index-4 classes): R <= 0.91 for p <= 1009.

(d) The dense regime has width O(1/p): cosine-modulated allocations beat the uniform one from s - 1 of about 5/p.

(e) The constructions that achieve all-offset flatness (Gauss sums on cyclotomic classes) also make their supports efficient covers, with the same O(p^{-1/2}) defect. Heuristic constants at n = 2: flatness defect 2d/sqrt p against amplification <= 0.58d/sqrt p over the window in which H_d stays optimal, a safety factor of about 3.4.

**Proof / evidence.**

- tl_search2.py and run_tl_batch.sh: soft-min L-BFGS with tau-annealing; the IF reference is computed self-consistently.
- legendre_dense.py and cyclotomic.py: R is an upper bound, since Gamma is estimated from below by LP and uniform/cyclotomic allocations.
- dense_probe.py.

*Check:* tl_batch.out, legendre_dense.out, cyclotomic.out, dense_probe.out.

### J12 [CONJECTURE]

(PW_C) POINTWISE COVERING: the exact residual of joint detection.

For every odd prime p and every class-C network there is an x with
  sum_l (Re Phi_l(x))_+^r <= nu^{r-1} sum_l Q_l.

It implies the class-C IF value through J1, hence RD in class C wherever pi = nu^{1-s} (for example at n = 2).

What is known:
- the single-feature case and the one-cheap-single-set-feature case are proved (J9);
- with several cheap features it says that the low-field points of the features with Xi_l > 1 cannot all be covered by the excess of the others.

**Proof / evidence.**

Evidence:
- holds on all 40 stored networks (J10);
- holds on 3000 one-cheap-feature networks (J9);
- fails only for the non-class-C (DC) p = 3 winners, consistent with BRIEF_CORE section 3.

The argument needed: the low-field points of a cheap feature at frequency l can be covered by a feature at l' only through correlation of their field functions. Over x in Z_p^n such correlation arises only from harmonic or additive relations among the frequencies. Kronecker-type simultaneous hitting holds for dissociated S but fails generically once |S| > log_3 p < K* (pigeonhole), which is the known capacity obstruction.

*Check:* check_PW.out, check_PW1.out.

## Obstruction
The precise general obstruction, for every p, n and b.

1. CONTAINMENT (J4). Any proof of RD in class C must prove RT_n(p, s): the one-variable relations inequality, for every n, every s in [1, nu/2] and every p.
- Its amplification is tau_n(s) = (1+1/n)^{s-1}, sharp for aligned atoms by J2.
- tau_n(s) reaches e^{1/2} on the b = 2 diagonal.
- RT_2 is O1.
- So the relations core is not an n = 2 artifact, and at b = 2 with n >= 3 its constant c_n = 1 - (n/nu)^{s-1} (up to 0.393) is larger than c_12 = 0.184.

2. NO EFFICIENCY-BLIND DETECTION (J6). The Legendre single-set network has depth (2/sqrt p) times the payment-charged interference at EVERY offset at once. So none of the following can prove RT_n for all p unless it uses the efficiency of the allocation:
- linear or Riesz-product tests;
- Lambda(p), Khintchine or Bonami bounds;
- Paley-Zygmund;
- tilted or extreme-value measures;
- any choice of offset set.
Slot-wise tools are blind in addition: single-set fluctuations depend on x only through x_T.

3. NO OFFSET RIGIDITY WHERE THE CORE LIVES (J5). The offset phases are a common translation exactly for balanced families, which include all single-set networks. There the offsets differ only by real weights and a shift. Unbalanced families lose the translation, and with it the oscillation bound; that favours the network.

4. JOINTNESS IS NOT THE OBSTRUCTION (J1). The l_r field-row bound detects all features at once with no loss of sqrt(K). It has exactly the shape of the joint payment and is tight at IF optima. What it cannot do is replace the minimum over x by the mean for features with Xi_l > 1:
- relations-type single-set features at small n or b near 2 (Xi <= 1.054);
- wild features with holes (Xi up to 3.7 at (10, 2)).
Every cheap feature of every stored non-IF optimum has Xi_l > 1 (J10).

5. THE EXACT RESIDUAL is the pointwise covering statement PW_C (J12). With several cheap features it asks for an inverse theorem: 'if the excess functions of an efficient allocation's features cover Z_p^n (equivalently, the single-set interference is flat at all binding offsets), the allocation is compressible.'
- Such a theorem must couple flatness or covering to coverage efficiency, because the same Gauss sums produce both (J11e).
- Kronecker or dissociation arguments stop at |S| = log_3 p < K*.
- In the near-dense window s - 1 = Theta(p^{-1/2}) any proof must be sharp to within a constant factor of about 3.4. This is a diagnostic, not a target statement.

Mixed networks. J1 and J9 hold for every network: J1 was checked on DC/mixed networks, and its clean single-neuron budget needs no channel classification. Since RD for all networks implies RD in class C, the class-C obstruction above is inherited. The channel and mode-budget step was therefore not pursued.

## Next steps
- Two-feature PW at every p and n. Take one cheap single-set feature at frequency l and one at frequency l' (or one wild feature plus one tame one). Prove that the low-field set of the first cannot be covered by the excess of the second unless l'/l = k'/k mod p with small k, k'. The tool would be a 'harmonic height' decorrelation lemma: Cov(psi_{l'}, 1_{hole(l)}) is bounded by the tail of the slot harmonics beyond the height H(l'/l). This is the first genuinely joint case of PW and needs no additive combinatorics beyond heights.
- Turn FR-min into an exchange (compression) argument at every n. If Psi > 0 everywhere, show that the field row (Re Phi(x*))_+ at a near-minimiser, suitably modulated as in arith2 R8/R9, is a cheaper clean cover than the network. That would be the covering analogue of 'flat implies compressible', with the payment of J2 supplying the exchange rate c_n(s).
- Strengthen the per-feature detection factor D_l inside FR without losing additivity. Replace (Re Phi_l)_+ by a cone-adjusted real part that uses the imaginary channel on half of the offsets (FR drops Im completely, while W8 shows that Im is the main single-feature detector). Target: Xi_l <= 1 for every circular aligned feature at every n, which would leave only compensation (PW) open.
- Extend J2 to non-aligned atoms for n >= 3 (J3 is numerical). Combine the leverage-score argument with the W6(b) twisted Hermitian matrix and charge the twist error sqrt(2L(L-|A|)) against the factor |A|/L; equality cases are KM2 and the n-atom block.
- Use the explicit single-set family (J4, cjlib.TL) at n in {3, 5}, b = 2 and p in [61, 400] as a counterexample test bed. There Parseval no longer closes (binding kappa-mass about 6-9 against 1/c_n^2 = 16 and 10.7). Look for supports with many additive or harmonic relations but near-optimal coverage; the bounds R <= 1 recorded so far come from local search only.
- Map the near-dense window s - 1 = Theta(p^{-1/2}) with exact programme values. Use cyclotomic and Rudin-Shapiro half-sets as competitors and Legendre and quartic-character interference as attackers, to pin down the safety factor (heuristically about 3.4 at n = 2). Any correct inverse theorem has to be sharp to within it there.
