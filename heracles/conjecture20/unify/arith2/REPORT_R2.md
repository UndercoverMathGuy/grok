# arith2: report (round 2; saved by the orchestrator from the agent's structured output)

## Headline
ARITH (round 2, resumed). The relations core (O1) at n=2 is NOT closed at b=2 for any p beyond 59. What is new: (i) A two-atom cap tau'=(3/2)^{s-1} gives a p-dependent efficiency closure e_thr(p,s): 0.997 at p=61, tending to 0.8165 (the Note had 0.759). It also gives the b-dependent all-network range p<2/c(s)^2: p<=59, 113, 317, 10369 at b=2, 2.25, 2.5, 2.9, and every prime once b>3-O(p^{-1/2}). (ii) A phase-free reduction: O1 at p follows from TDP*(p), a water-filled Parseval condition, which also has a delta-free form. Numerically TDP* holds on everything tested up to p=30011, but it must fail once K* is about 55 or more. (iii) An interval-certified gain bound: at every prime and every b in [2,3], a relations-case network has gamma <= 1.108 x IF. (iv) A compression lemma (an exchange argument) plus a certified doubling lemma. At b=2 and at EVERY prime, no network reaches IF if its cheap set is a disjoint union of constant-price dilated progressions (any lengths, any number of blocks) plus a Parseval-small rest. A single progression with constant prices never reaches IF at any length. What remains open is cheap sets that are at once large, tau'-incompressible and flat. A Legendre-symbol construction refutes every flatness bound that depends only on the support. All files are in /home/user/grok/heracles/conjecture20/unify/arith2/.

## Principle
COVERAGE AMPLIFICATION vs DETECTION vs COMPRESSION.
(1) Amplification is bounded. In the relations case a cheap feature's clean amplitude is at most tau'=(3/2)^{s-1} times its IF price (R1). Its interference is also at most tau' times the price. The identity 1+c12*tau'=tau' ties this to P12.
(2) Detection by Parseval. Lemma osc plus exact Parseval on Z_p cancels the interference bonus of any part of the cheap set whose kappa-mass is <=1/c12^2 (=29.70 at b=2).
(3) Optimality enters as incompressibility. A sub-block of a programme maximiser can never be replaced by a cheaper clean allocation that dominates its coverage profile at every offset. Conversely, a cheap block whose coverage can be bought for <=(2/3)^{s-1} of its cost is closed, because interference amplifies it by at most tau'=(2/3)^{-(s-1)} (R8).
So interference can only pay on a part of the cheap set that is at once:
 (a) LARGE: kappa-mass >29.7 at the relevant offsets;
 (b) tau'-INCOMPRESSIBLE;
 (c) FLAT: osc D_delta close to 2(E D_delta^2)^{1/2}.
This replaces the Note's trilemma 'flat + efficient + small'.
Cases removed:
- every p<2/c(s)^2 (R2b);
- every allocation with e(a)<e_thr(p,s) (R2a);
- at every prime and b=2, every cheap set that is a union of constant-price dilated progressions plus a Parseval-small rest (R9).
Every net margin gain over IF is <=10.8% (R6).
Fourier supports (SPLIT R7(c)): every test used here is a function of w=2(x1+x2), i.e. has Fourier support Lambda=Z_p*(1,1). Averaging over x1-x2 gives h(theta) <= min_delta min_w E[g_theta(.,delta)|w], so every result transfers verbatim to networks whose split is in the relations case and whose mixed modes vanish on Lambda\{0}.

## Results

### R0 [NUMERICAL]

Reconstruction of the previous arith agent's work (unify/arith/), with re-runs and verdicts.

**Proof / evidence.**

twoatom_check.py: the claim is correct and is now PROVED as R1. The re-run gives max violations 2.8e-14, 3.6e-10 and -4.2e-9, all at rounding level (arith2/twoatom_rerun.out).

xi_const.py / xi_const.out: numerical Xi_rel. At s=1.5 the torus value is 1.06537 (gain 1.0996); at p=5 it is 1.06691. xi_certify.py certifies at fixed s. Both are superseded by R6, which covers all s in [1,3/2] and finite p, and which does not need the quadratic payment.

pw_adv*.py, pw_fixed_t.py (label-row closure, NUMERICAL):
- On programme supports, adversarial phases give max min_w C(w)/sum a^r = 0.9984-0.9989 for p<=1009 and 0.9977-0.9979 at p=10007.
- On uniform progressions with t=0.825 at p=1009, label rows alone FAIL for N>=40 (R=1.007 at N=40, 1.024 at N=80). This is exactly the family closed by compression in R9.

pw_moments.py: the statement 'degree-2 test closes for K<64.1' is a TORUS-model computation with independent phases. It ignores additive relations, so it is evidence and not a Z_p proof.

compress.py: finite-p reweighted LP. Its output is an uncertified upper bound on rho. Re-run in arith2/compress_rerun.out:
- p=211: rho = 0.926, 0.701, 0.553 for N = 10, 30, 60;
- p=1009: rho = 1.000, 0.744, 0.753.
Superseded by the p-uniform certificates of R9.

pw_identity_check.py: the identity min(g(x,d),g(x,-d)) = <Re Phi(x),kappa(d)> - |<Im Phi(x), sin(2 pi l d/p)>| is correct. It follows in one line from g = sum_l Re(Phi_l (1-rho^{-l d})).

*Check:* arith2/twoatom_rerun.out, arith2/compress_rerun.out; outputs of the previous agent in unify/arith/*.out

### R1 [PROVED]

Two-atom cap. Let beta_i in C and xi_i in T, with L=sum|beta_i|, Q=sum|beta_i|^r (0<r<=1) and B=sum beta_i xi_i. Then
  max_i |beta_i| <= (L+|B|)/2  and  Q >= ((L+|B|)/2)^r + ((L-|B|)/2)^r.
The same holds with |B| replaced by min(|B|,|C|,|D|).

Relations case (B=C=0, n=2, s in [1,3/2], a_l = 3^{1-s} Q_l^s):
  L_l <= 2^{1-s} Q_l^s = tau' a_l,  tau' = (3/2)^{s-1}.
Hence Re beta_l <= |A_l| <= L_l <= tau' a_l and v_l = |beta_{l,12}| <= tau' a_l. Together with P12, Re beta_l <= min(a_l + c v_l, tau' a_l) with c = c12 = 1-(2/3)^{s-1}, and 1 + c tau' = tau'.

Equality case: two aligned atoms, beta=(1/2,1/2) and xi=eta=(i,-i). They give L = |A| = |D| = 2^{1-s}Q^s, and equality in P12 at the same time.

**Proof / evidence.**

1. Let |beta_1| be the largest atom. Then |B| >= |beta_1| - sum_{i>=2}|beta_i| = 2|beta_1| - L.
2. x -> sum x_i^r is Schur-concave. On {x>=0, sum x = L, max x <= M} with M = (L+|B|)/2 >= L/2, its minimum is at (M, L-M, 0, ...).
3. M^r + (L-M)^r is decreasing in M on [L/2, L].
4. At B=0 this gives Q >= 2^{1-r} L^r, i.e. L <= 2^{-(1-r)s} Q^s = 2^{1-s} Q^s = (3/2)^{s-1} 3^{1-s} Q^s.
5. |A| = |sum beta_i| <= L and |D| = |sum beta_i xi_i eta_i| <= L.
6. 1 + c tau' = 1 + (tau' - 1) = tau'.
The min(|B|,|C|,|D|) version is the same argument applied to C or D. QED

*Check:* arith2/twoatom_rerun.out: 2e5 random atom systems, n<=8, s in [1,1.5]. All three inequalities hold to rounding, and the equality case is reproduced (bonus/(c12|D|) = 1.0).

### R2 [PROVED]

p-dependent Parseval-efficiency closure. Setting: relations case, n=2, s in [1,3/2], any width, any odd prime p. Notation: a = IF prices, v = interference moduli, m(a) = min_delta (Ka)_delta, e(a) = m(a)/(Gamma_s ||a||), ||a|| = (sum a^r)^s.

For every delta != 0:
  h <= (Ka)_delta + c(Kv)_delta - (sum_l v_l^2 kappa_l(delta))^{1/2} <= (Ka)_delta [1 + tau'(c - (2/p)^{1/2})_+].
Hence h <= e(a) [1 + tau'(c - sqrt(2/p))_+] Gamma_s ||a||.

(a) Every network with e(a) < e_thr(p,s) := 1/(1 + tau'(c - sqrt(2/p))_+) has gamma < IF. At b=2: e_thr = 0.9970 (p=61), 0.9870 (67), 0.9502 (101), 0.8545 (1009), 0.8282 (10007), tending to (2/3)^{1/2} = 0.8165. The Note had 0.759.

(b) If p < 2/c(s)^2, every relations-case network with v != 0 has gamma < IF, so maximisers in the relations case are IF and use K* features (Note, Prop count). Largest prime covered:
- b=2: p<=59
- b=2.1: 73
- b=2.25: 113
- b=2.4: 211
- b=2.5: 317
- b=2.75: 1523
- b=2.9: 10369

(c) Equivalently, at every prime the relations case is closed for b in (b_p, 3], where b_p = 3/(1 + ln(1 - sqrt(2/p))/ln(2/3)) = 3 - O(p^{-1/2}). Examples: b_61 = 2.0098, b_1009 = 2.6971.

**Proof / evidence.**

1. Note Lemma osc (separate minimisers for +-delta, and D_{-delta}(w) = -D_delta(w+delta)) gives h <= sum_l kappa_l(delta) Re beta_l - (1/2) osc_w D_delta. P12 gives Re beta_l <= a_l + c v_l.
2. Bhatia-Davis gives (1/2) osc D >= (E_w D^2)^{1/2}. Exact Parseval on Z_p gives E_w D_delta^2 = sum v_l^2 kappa_l(delta), because w = 2(x1+x2) is uniform on Z_p and l != +-l' for distinct l, l' in F.
3. Cauchy-Schwarz: (Kv)_delta <= (sum v^2 kappa)^{1/2} (sum_{spt v} kappa_l(delta))^{1/2} <= (sum v^2 kappa)^{1/2} (p/2)^{1/2}, using sum_{l in F} kappa_l(delta) = p/2.
4. Hence c(Kv) - (sum v^2 kappa)^{1/2} <= (c - sqrt(2/p))(Kv) <= (c - sqrt(2/p))_+ tau'(Ka), using R1.
5. Take delta = argmin (Ka). This gives (a).
6. For (b): if c < sqrt(2/p), then h <= m(a) - (sqrt(2/p) - c)(Kv)_{delta0}. Since kappa_l(delta0) > 0 for all l in F, h < m(a) <= Gamma_s ||a|| unless v = 0.
7. For (c): solve c(3/b) = sqrt(2/p) for b. QED

Honesty note: (b) is the Note's Cor. parseval with P12, evaluated at general s. What is new is the tau' cap and the interpolation (a).

*Check:* arith2/check_ethr.out: 60 phase-optimised (Adam) configurations, p in {31,...,151}, s in [1,1.5]. Max h_model/[m(a)(1 + tau'(c - sqrt(2/p))_+)] = 0.8404; at p<=59, max h/(Gamma ||a||) = 0.5468. Ranges and e_thr values: arith2/ranges.out.

### R3 [PROVED]

Phase-free reduction TDP*, and its delta-free form.

(i) For every delta,
  h <= (Ka)_delta + tau' E0(a, kappa(delta)),
where E0(a, kappa) := max_{0<=u<=a} [c sum u_l kappa_l - (sum u_l^2 kappa_l)^{1/2}].
Water-filling formula: if c^2 sum_{spt a} kappa_l > 1, then E0 = sum_l a_l kappa_l (c - lambda a_l)_+, with lambda the root of sum_l min(c, lambda a_l)^2 kappa_l = 1; otherwise E0 = 0. E0 is concave and 1-homogeneous in a.
Hence TDP*(p) := [Phi*(a) := min_delta ((Ka)_delta + tau' E0(a, kappa(delta)))/||a|| <= Gamma_s(p) for all a >= 0] implies that no relations-case network at p beats IF. The strict form needed for the count follows if, in addition, maxima of Phi* occur only at minimising offsets with E0 = 0.

(ii) Delta-free form. Let N_c = floor(1/(2c^2)); N_c = 14 at b=2. For every a and every T in F with |T| <= N_c,
  h <= m(a_T + tau' a_{F\T}).

**Proof / evidence.**

(i) The inequality.
1. From R2, step 1: h <= (Ka)_delta + c(Kv)_delta - (sum v^2 kappa)^{1/2}.
2. Write v = tau' u with 0 <= u <= a (R1). Then the last two terms equal tau'[c sum u kappa - (sum u^2 kappa)^{1/2}].

(i) Water-filling.
1. f(u) = c sum u kappa - N(u), with N = (sum u^2 kappa)^{1/2}, is concave. For u != 0, d f/d u_l = kappa_l (c - u_l/N).
2. KKT on the box [0,a] gives u_l = min(a_l, cN).
3. Self-consistency N^2 = sum min(a_l, cN)^2 kappa_l is equivalent to sum min(lambda a_l, c)^2 kappa_l = 1 with lambda = 1/N. The left side is continuous and nondecreasing in lambda, from 0 to c^2 kappa(spt a), so a root exists iff c^2 kappa(spt a) > 1.
4. Substituting gives f = sum kappa_l min(a_l, cN)(c - min(lambda a_l, c)) = sum_{lambda a_l < c} a_l kappa_l (c - lambda a_l).
5. If c^2 kappa(spt a) <= 1, then f(u) <= (c kappa(spt a)^{1/2} - 1) N <= 0.
6. KKT is sufficient for concave f.
7. Concavity of E0: a partial maximum of a jointly concave function over the convex cone {0 <= u <= a}.

(ii)
1. kappa <= 2, so sum_T kappa_l(delta) <= 2 N_c <= 1/c^2 for every delta.
2. For u <= a: c sum_T u kappa <= c (sum_T kappa)^{1/2} (sum_T u^2 kappa)^{1/2} <= (sum u^2 kappa)^{1/2}. Hence E0 <= c sum_{l not in T} a_l kappa_l(delta).
3. Then (Ka) + c tau' (K a_{T^c}) = (K(a_T + (1 + c tau') a_{T^c})) = (K(a_T + tau' a_{T^c})). QED

*Check:* arith2/check_chain.out: 300 random configurations, p in {31,...,131}, s in [1,1.5]. The chain h_model <= O1 <= Parseval <= tail bound holds (max violations -2.2e-4, -2.7e-2, -5.2e-2). E0 <= tail bound holds to 1e-17, and E0 matches a brute-force maximiser to 1e-6.

### R4 [NUMERICAL]

Status of TDP* and TDP14 at b=2. Both hold on everything tested up to p=30011. Any argument that uses only Parseval detection must fail at the programme optimum once K* is about 55 or more, i.e. p roughly 1e12-1e13.

**Proof / evidence.**

(i) Heuristic global maximisation (reweighted LP) of m(a_T + tau' a_rest)/(Gamma_ref ||a||) at p = 61, 101, 211, 401, 1009 gives exactly 1.00000, attained at the programme optimum (tdp14.out). Perturbative searches give local maxima 0.91-0.95 (tdp_search_small.out).

(ii) On all 34 near-optimal allocations on file, p from 61 to 30011, the kappa-mass of the support at the binding offsets is 0.48-0.72 K (e.g. [8.66, 11.64] for K=18 at p=30011). This is far below 29.7, so E0 = 0 there and Phi* = e Gamma <= Gamma (tail_on_runs.out).

(iii) Forced-spread allocations (up to 100 support points): Phi*/Gamma <= 0.914 (p=1009) and 0.897 (p=10007). All with e < e_thr are already closed by R2 (tdp_forced_*.out). Spread restricted optima at p=1009 give Phi*/Gamma <= 0.977 (tdp_map_B_1009.out).

(iv) Prediction. The binding kappa-mass is about (0.5-0.65) K*. Once K* >~ 29.7/0.55 = 55, every binding offset of y* has E0 > 0, so Phi*(y*) > Gamma_s: TDP* fails at the optimum itself. K* is about 1.6-2 ln p.

*Check:* Scripts: tdp14.py, tdp_search.py, tail_on_runs.py, tdp_forced.py, tdp_map.py (arith2/). The p=100003 run ran out of memory (dense 50001^2 kernel), so the data stop at 30011.

### R5 [PROVED]

Label-row principle and the relaxation chain.

For every network (any n, any class) and every x:
  h <= min_delta sum_l Re Phi_l(x) kappa_l(delta) <= Gamma_s ||(Re Phi(x))_+||   (label-row bound).

Relations case. Put psi_delta(w) = sum_l v_l kappa_l(delta) cos(2 pi l w/p + theta_l) = (D_delta(w) + D_{-delta}(w))/2. Then for every delta:
  O1(delta) := (KA)_delta - (1/2) osc D_delta <= CP(delta) := (KA)_delta + min_w psi_delta(w),
and
  min_delta CP <= LR := Gamma_s min_w ||(A + v cos(2 pi l w/p + theta))_+||.
So O1 <= CP <= LR: Lemma osc is the strongest of the three relaxations.

Test support: functions of w, i.e. Lambda = Z_p(1,1).

**Proof / evidence.**

1. g(x,d) + g(x,-d) = sum_l Re(Phi_l(x)(2 - rho^{ld} - rho^{-ld})) = 2 sum_l Re Phi_l(x) kappa_l(d), and h <= min(g(x,d), g(x,-d)).
2. sum_l Re Phi_l kappa_l <= sum_l (Re Phi_l)_+ kappa_l, and min_d (Ky)_d <= Gamma_s ||y|| for y >= 0 by definition of Gamma_s.
3. For CP: D_d + D_{-d} = 2 phi(w) - phi(w-d) - phi(w+d) = 2 psi_d, and min D_{-d} = -max D_d. Hence min psi_d >= (min D_d - max D_d)/2.
4. In the relations case Re Phi_l(x) = Re beta_l + v_l cos(2 pi l w/p + arg W_l). QED

*Check:* arith2/check_lr.out: 240 random configurations, p in {61, 101, 211}. Max violations: h <= O1: -2.7e-4; O1 <= CP: +7e-15 (rounding); CP <= LR: -0.20.

### R6 [PROVED]

p-uniform gain bound (computer-assisted, interval arithmetic). Relations case, n=2, any width.

For every odd prime p and every s in [1,3/2]: gamma(theta) <= Xi_p(s)^s * nu^{1-s} B_p Gamma_s(p), where
  Xi_p(s) := sup_{t>=0, vartheta} (1/p) sum_{k in Z_p} (1 + t cos(vartheta + 2 pi k/p))_+^r / pi(t)^r,
  pi(t) = 1 - c t for t <= 1, and pi(t) = t(1-c) for t >= 1.

Certified values:
- Xi_torus(s) <= 1.06800 on [1,3/2]; <= 1.0164 on [1,1.02]; <= 1.0468 on [1.30,1.32].
- Xi_p <= Xi_torus + alpha_p (3/2)^{(s-1)/s}, with alpha_p <= (8/3) Gamma(1/3)^{-1} sum_{j>=1} 1/(jp (jp-1)^{2/3}).
- alpha_61 <= 2.3e-3, which gives Xi_p^s <= 1.1077 for all p >= 61.

With R2(b) (closure for p <= 59 at every b in [2,3]): at every prime and every b in [2,3], interference raises the relations-case margin by at most 10.8% over IF. This is a value statement only for n=2, p>=5, so it does not touch the p=3 counterexamples.

**Proof / evidence.**

Reduction to one feature.
1. R5 with min_w <= E_w gives h <= Gamma_s (E_w sum_l (A_l + v_l cos(2 pi l w/p + theta_l))_+^r)^s.
2. w -> l w is a bijection of Z_p, so each feature's average is the p-point average f_p of f(U) = (A + v cos U)_+^r.
3. If A > 0, put t = v/A. P12 gives a >= A(1 - c t); R1 gives a >= v/tau' = t A (1-c). So a >= pi(t) A and f_p <= Xi_p a^r.
4. If A <= 0: (cos U)_+ <= (1 + cos U)/2 gives f_p <= (tau' a)^r 2^{-r}(c0 + alpha_p) < a^r.
5. Summing: h <= Xi_p^s Gamma_s ||a||.

Torus certificate (mpmath.iv, 25 x 800 boxes in (s,t)).
1. For t in [0,1]: E(1 + t cos)^r = 1 - sum_{k>=1} |binom(r,2k)| binom(2k,k) 4^{-k} t^{2k}. Every term is <= 0, so truncation gives an upper bound.
2. This is increasing in r on [2/3,1], because |binom(r,m)| decreases in r there for m >= 2 (d log/dr = 1/r - sum 1/(j-r) < 0). It is decreasing in t.
3. pi(t,s) >= 1 - c(s2) t2 on each box. Bound each box by E(t1, 1/s1)/pi_low(t2)^{1/s1}.
4. For t >= 1, f is decreasing in t. The supremum is c0(r)(3/2)^{(s-1)r} with c0(r) = 2^r Gamma(r+1/2)/(sqrt(pi) Gamma(r+1)), enclosed over r-intervals.
5. The quadratic payment is NOT needed: running the identical certificate without it gives the same 1.06800 (xi_certify_noquad.out).

Finite p.
1. With t = 2q/(1+q^2): f = (1+q^2)^{-r} (1 + q e^{iU})^r (1 + q e^{-iU})^r, so fhat(m) = (1+q^2)^{-r} sum_b binom(r, b+m) binom(r, b) q^{2b+m}.
2. |binom(r,k)| decreases in k, and sum_{b>=1} |binom(r,b)| = 1, so |fhat(m)| <= 2 |binom(r,m)|.
3. f_p - Ef = sum_{j!=0} fhat(jp) e^{ijp vartheta}.
4. Gautschi gives |binom(r,m)| < r/(Gamma(1-r) m (m-1)^r). Also r/Gamma(1-r) is decreasing on [2/3,1) (interval-checked).
5. For t >= 1, the p-average of (1/t + cos)_+^r decreases in t, so the case reduces to t = 1. QED

*Check:* arith2/xi_certify_all_s.out, xi_certify_noquad.out, xi_finite_p.out.
Numerical values below the certified bounds: torus 1.06537; p=5: 1.06691 (arith/xi_const.out).
arith2/check_lr.out: on 240 random P12-extremal configurations (A = a + cv, v <= tau' a, so t <= 1), max E_w sum row^r / sum a^r = 1.0633, below Xi_p (1.0683-1.0706).

### R7 [PROVED]

KKT identity at programme maximisers, and first-order label rows.

If y* maximises Gamma_s(p) with sum y^r = 1, there is a probability mu on the binding set B = argmin_delta (Ky*)_delta with
  K_l(mu) := sum_delta mu_delta kappa_l(delta) = Gamma_s y*_l^{r-1}  for every l in spt y*.

Consequence. For interference v on spt y* with phases theta, the label-row cost C(w) = sum_l (y*_l + v_l cos(2 pi l w/p + theta_l))^r satisfies
  C(w) = 1 + (r/Gamma_s) psi_mu(w) + O(|v|^2),  psi_mu = sum_delta mu_delta psi_delta (the CP detection of R5).
So at the optimum the label-row test and the mu-averaged common-point detection agree to first order. Moreover min_w psi_mu < 0 unless v = 0, since E_w psi_mu = 0 and K_l(mu) > 0.

**Proof / evidence.**

1. On S = spt y*, (y*, t* = Gamma) is a local maximum of t subject to (K_S y)_delta >= t and sum y^r <= 1.
2. MFCQ holds with direction (-eps y*, -1). So there are multipliers mu_delta >= 0 on the active delta and lambda >= 0, with sum mu = 1 and K_l(mu) = lambda r y_l^{r-1}.
3. Multiply by y_l and sum: lambda r = sum_delta mu_delta (Ky*)_delta = Gamma_s.
4. Expand (y + v cos)^r = y^r + r y^{r-1} v cos + O(v^2). QED

Route assessment for unicorn KKT. For a putative unicorn (a maximiser of Q/||a|| with v != 0) the KKT system only gives K_l(mu) <= R* a_l^{r-1}, with equality off the cap v_l = tau' a_l. This is underdetermined and local, consistent with the BRIEF dead end 'KKT alone'.

*Check:* arith2/check_misc.out:
- the KKT LP is feasible at p = 31, ..., 1009 (|B| = K*) with residual <= 2.7e-14;
- the first-order identity error scales as eps^2: 4.7e-6 at eps = 1e-2, 5.4e-8 at eps = 1e-3.

### R8 [PROVED]

Compression lemma: an exchange at the reduced level, using optimality as incompressibility.

Setting: relations case, n=2, any p, any width, s in (1,3/2]. Let L' be the cheap set, with L' contained in C' + B (disjoint subsets of F), where sum_{l in C'} kappa_l(delta) <= 1/c^2 for every delta != 0. Suppose b'' >= 0 satisfies (Kb'')_delta >= tau' (K a_B)_delta for every delta. Then
  h <= m(a_{F\B} + b'') <= Gamma_s (sum_{l not in B} a_l^r + ||b''||^r)^s.

Define the compression ratio rho(a_B) := min{ ||b'|| : Kb' >= K a_B pointwise } / ||a_B||. If rho(a_B) < (2/3)^{s-1} = 1/tau' and B is non-empty, then gamma(theta) < IF.

Why this is the optimality notion: sub-blocks of programme maximisers have rho >= 1, since replacing a compressible block lowers the cost without lowering any coverage.

Test support: Lambda = Z_p(1,1).

**Proof / evidence.**

Fix delta.
1. Lemma osc, with P12 on C', R1 on B (A_l <= tau' a_l) and A_l <= a_l off L', gives
  h <= (K a_{F\B})_delta + tau'(K a_B)_delta + c(K v_{C'})_delta - (1/2) osc D_delta.
2. Parseval over all of spt v, then dropping the nonnegative terms of B:
  (1/2) osc D_delta >= (sum_{C'} v_l^2 kappa_l(delta))^{1/2} >= c(K v_{C'})_delta,
  by Cauchy-Schwarz and the kappa-mass hypothesis.
3. Hence h <= (K(a_{F\B} + tau' a_B))_delta <= (K(a_{F\B} + b''))_delta for every delta.
4. So h <= m(a_{F\B} + b'') <= Gamma_s ||a_{F\B} + b''||. Subadditivity (x+y)^r <= x^r + y^r on overlapping frequencies gives the cost bound.
5. With b'' = tau' b': ||b''||^r = tau'^r rho^r ||a_B||^r < sum_B a^r. QED

Why the exchange must be done at the reduced level (network-level obstruction).
- The ablation identity g_{theta-N} = g_theta - g_N gives h(theta - N) >= h(theta) - max g_N (check_misc.out (3)), so a network-level exchange would need (Kb'')_delta >= max_x g_{N_B}(x,delta).
- That maximum contains the term max_w D^B_delta >= (sum_B v^2 kappa)^{1/2}, which is O(1/p) at delta = d^{-1}.
- Any fixed clean dominator covers only O(p^{-2}) there.
- So the exchange must be done after Lemma osc, at the price of the factor tau'.

*Check:* arith2/check_compress.out: 40 exact Z_p models, p in {101, 211, 307}, blocks d{1..q} with q in {17, ..., 40}, coordinate-adversarial phases. max[h_model - m(a_{B^c} + b'')] = -0.71 and max[cost difference] = -0.14.

### R9 [PROVED]

Uniform progressions are compressible at every scale; consequently progression cheap sets are closed at every prime at b=2 (computer-assisted: interval arithmetic plus an analytic doubling lemma).

(a) For s = 3/2 and every q >= 17 there is b'_q >= 0 on the positive integers with
  B_q(x) = sum_l b'_{q,l}(1 - cos l x) >= P_q(x) = sum_{j<=q}(1 - cos j x)  for all real x,
and ||b'_q|| < (2/3)^{1/2} q^{3/2}.
Certified ratios rho_q:
- q = 17: 0.8009; q = 18: 0.7867; q = 19-23: <= 0.7592; q = 24: 0.7290;
- q = 30: 0.6819; q = 48: 0.6454; q = 52: 0.5889; q = 64: 0.5358;
- q = 84: 0.5112; q = 110: 0.4858; q = 144: 0.4189;
- along all doubling chains: <= 0.5125.
q = 16 is NOT below the threshold (0.8206).

(b) Let b = 2, p any odd prime, any width. Suppose the cheap set L' is contained in C' + B_1 + ... + B_m (pairwise disjoint), where:
- each B_i = {+-d_i j mod p : 1 <= j <= q_i} with q_i >= 17;
- each B_i carries a constant IF price;
- max_delta sum_{C'} kappa_l(delta) <= 29.70.
If m >= 1 then gamma < IF.
In particular a cheap set that is a single dilated progression with constant prices never reaches IF, at any length q. For q <= 23 no price condition is needed: max_x P_23 = 28.61 < 29.70, so Parseval applies.

**Proof / evidence.**

(b) from (a) and R8.
1. For l = +-d j in B_i: kappa_l(delta) = 1 - cos(j x) with x = 2 pi d_i delta/p.
2. Hence (K a_{B_i})_delta = abar_i P_{q_i}(x) <= abar_i B_{q_i}(x) = abar_i (K D_{d_i} b'_{q_i})_delta. Here D_d dilates by d and folds mod p with +- identified. Terms with p | d l are 0 and are dropped; merging frequencies only lowers sum(.)^r.
3. Take b'' = tau' sum_i abar_i D_{d_i} b'_{q_i}. Then ||b''||^r <= sum_i (tau' rho_{q_i})^r abar_i^r q_i < sum_{B} a^r.

(a), step 1 (direct certificates). For q in {17, ..., 26, 28, 30, 32, 36, 40, 44, 48, 52, 56, 60, 64, 84, 110, 144}:
- b' is found by reweighted LP, scaled by 1+eta and rounded UP to rationals with denominator 1e9;
- on [0, x_s]: 1 - cos t lies in [t^2/2 - t^4/24, t^2/2], with x_s^2 = 12(S2' - S2)/S4';
- on [x_s, pi]: cells with g >= min(g(x_i), g(x_{i+1})) - (S2' + S2) h^2/8;
- every certificate is re-verified in mpmath.iv (min cell margin >= 8.8e-5);
- evenness and 2 pi-periodicity give all real x.

(a), step 2 (monotonicity). P_{q'} <= P_q for q' <= q, so b'_q also serves q' with ratio rho_q (q/q')^{3/2}. This fills [17, 144].

(a), step 3 (doubling lemma, analytic, q >= 84). Identities:
  P_{2q}(x) = 2 P_q(2x) + E_q(x),  E_q(x) = -sin(qx) sin((q+1/2)x)/cos(x/2),
  P_q(z) = q + 1/2 - sin((q+1/2)z)/(2 sin(z/2)).
If B >= P_q, set B_2(x) = 2(1+eps) B(2x) + lam(1 - cos x) with eps = 1.01/q and lam = 1.2q. Then B_2 - P_{2q} >= R := 2 eps P_q(2x) + lam(1 - cos x) - E_q(x), and R >= 0 on [0, pi]:
- On [0, pi/(2q)]: E_q <= 0.
- On [pi/(2q), 2/sqrt q]: E_q <= 1/cos(1/sqrt q) <= 1.006, and P_q(2x) >= q + 1/2 - pi/(4x) >= (q+1)/2, so 2 eps P_q(2x) >= 1.01(q+1)/q.
- On [2/sqrt q, pi - sqrt 2]: E_q <= 1/sin(sqrt2/2) = 1.540, and lam(1 - cos x) >= 2.4(1 - 1/(3q)) >= 2.39.
- On [pi - sqrt 2, pi], with x = pi - y: |E_q| = |sin(qy) cos((q+1/2)y)|/sin(y/2) <= min(2q/(1 - y^2/24), 2.182/y), and lam(1 + cos y) >= 1.2q(2 - y^2/2) dominates. For y >= 1/q use that y(2 - y^2/2) is concave, so its minimum is at an endpoint.
Cost: frequency 1 is odd and all other frequencies are even, so rho_{2q}^r <= 2^{r-1}(1+eps)^r rho_q^r + lam^r/(2q).

(a), step 4 (induction with outward rounding). Chains q0 2^k from q0 in {84, 110, 144}:
- sup rho = 0.51247, 0.48579, 0.41890 respectively;
- the bound is preserved for all k;
- each q is covered by monotonicity on [0.733q, q];
- consecutive ladder ratios are <= 1.31 < 1/0.733;
- so every q >= 62 is covered. QED

*Check:* arith2/ certificate files:
- compress_certify_uniform.out, compress_certify_warm.out, compress_certify_big.out;
- JSON certificates compress_cert_*_s1.5.json;
- cert_iv_verify.out: every certificate re-verified in interval arithmetic.

arith2/doubling_check.out:
- closed forms hold to 1e-11;
- R >= 0 on 4e5-point grids for q from 84 to 20000;
- the explicit doubled dominator for q = 168 was certified DIRECTLY, with rho = 0.5125 as the lemma predicts;
- the interval-arithmetic chain bounds are printed there.

arith2/coverage_check.out: no uncovered q in [17, 2000]; worst ratio 0.8009.
arith2/check_compress.out (lemma level).

b = 2.25 is only partly certified (threshold 0.8736): rho = 0.782 (q=51), 0.745 (64), 0.723 (84), 0.694 (110) (compress_certify_b225.out). The b = 2.25 version of (b) stays open for q in 52-56, 65-72, 85-92 and for q > 110.

### R10 [REFUTED]

Refuted candidate lemma: 'support-only detection'. It claims there is sigma0 > 0 such that for every prime p, every delta != 0 and every interference (v >= 0 on F, any phases): (1/2) osc_w D_delta >= sigma0 (Kv)_delta. Equivalently, the Note's sigma_delta(F) would be bounded below uniformly in p.

False: for p = 1 mod 4, sigma_1(F) <= (pi/(2 sqrt p))(1 + o(1)).

**Proof / evidence.**

Counterexample.
1. Let chi be the Legendre symbol mod p, p = 1 mod 4. It is real, even and has mean 0, and |chihat(l)| = p^{-1/2} (Gauss sum).
2. Put phihat(l) = chihat(l)/(1 - e^{-2 pi i l/p}) for l != 0. Then phi is real and D_1 = phi - phi(. - 1) = chi, so (1/2) osc D_1 = 1.
3. v_l = 1/(sqrt p |sin(pi l/p)|), so (Kv)_1 = (2/sqrt p) sum_{l in F} |sin(pi l/p)| ~ 2 sqrt p/pi.
4. The ratio c12 (Kv)_1/((1/2) osc D_1) is 0.91 (p=61), 1.17 (p=101), 2.34 (p=401), 3.71 (p=1009).
5. The cheapest allocation that can carry this interference is a = v/tau'. Its efficiency is 0.32, 0.26, 0.15, 0.10, so it is closed by R2. Interference can thus beat detection at a single offset by a factor about sqrt p, and is excluded only by inefficiency.
Conclusion: any proof of O1 must use a property of the allocation (efficiency or incompressibility), not of the support alone.

*Check:* arith2/legendre_flat.out: max|D_1 - chi| <= 6e-13.

### R11 [CONJECTURE]

Flat implies compressible (b=2, every prime). Suppose an allocation a, interference v <= tau' a and phases have Q(a,v,theta) >= Gamma_s ||a|| (necessarily e(a) >= e_thr(p) >= 0.8165). Then the cheap set splits as C' + B with max_delta kappa_{C'} <= 29.7 and rho(a_B) < (2/3)^{1/2}.
By R8 this is contradictory, so the conjecture implies O1 and hence the relations case of (C) at every prime.

**Proof / evidence.**

Evidence:
(i) The classical carriers of flat polynomials, progressions with equal moduli, are compressible at every length >= 17 (R9). Their ratio decreases roughly like ln q/sqrt q (0.80 at q=17, 0.42 at q=144).
(ii) Label rows alone fail on long progressions (R0), but compression closes them.
(iii) Up to p = 30011, near-optimal supports have binding kappa-mass <= 12.2, which is the Parseval part (R4).
(iv) The Legendre configuration (R10) is flat but has e <= 0.32.
Not proved: an inverse theorem that turns flatness of D_delta at the binding offsets into compressibility.

*Check:* Evidence files: compress_certify_*.out, pw_fixed_progression.out (unify/arith), tail_on_runs.out, legendre_flat.out.

## Obstruction
1. The rigorous range at b=2 cannot yet be extended past p=59.
- For p >= 61, R2 leaves only allocations with e(a) > e_thr(p): 0.997 at p=61, 0.987 at 67, 0.95 at 101.
- Closing them means proving TDP*(p), or O1 directly, over the non-convex r-ball (r < 1).
- Allocations with E0 = 0 at a minimising offset are free, since m(a) <= Gamma_s ||a|| by definition. But nearly-optimal allocations with many small extra weights exist (by continuity, a = y* + eps*1 has e -> 1 with support all of F).
- So one needs either a certified branch-and-bound over supports (plausible at p = 61, 67 with 30-33 frequencies) or a structure theorem for near-optimal allocations. Neither exists. Gamma_s(p) itself has no certified upper bound beyond small p; only explicit lower bounds exist.

2. Parseval detection is asymptotically insufficient.
- At the programme optimum the binding kappa-mass is about (0.5-0.65) K*.
- Once K* >~ 55, every binding offset of y* has E0 > 0, so TDP*, and every argument that uses only E D_delta^2, fails at the optimum itself.
- Beyond that point detection must use flatness deficits, i.e. osc versus L^2.

3. Flatness cannot be controlled through the support alone.
- R10 (Legendre symbol) gives perfectly flat first differences with flat spectrum on F. It beats detection by a factor of about sqrt p, and is excluded only by inefficiency.
- Joint use of all binding offsets does not help within L^2: h <= min_delta f(delta) is already stronger than any mu-average. Label rows (a common x for all delta) fail on long progressions.
- The missing lemma must therefore couple flatness with efficiency or incompressibility: 'large and tau'-incompressible implies not flat' (R11). This is an inverse theorem for real trigonometric polynomials restricted to efficient covers.

4. Exchange at the network level is blocked by a delta -> 0 barrier.
- Replacing a cheap sub-network by a cheaper IF block requires dominating max_x g_N.
- max_x g_N contains the non-kappa-weighted term max_w D^B_delta, which is O(1/p) at delta = d^{-1}, while any clean coverage there is O(p^{-2}).
- So exchanges must be made after Lemma osc. That costs the amplification factor tau', which is why the compression threshold is (2/3)^{s-1} and not 1.

5. b > 2.
- The compression threshold (2/3)^{s-1} tends to 1 as b -> 3, but compression ratios also tend to 1.
- The doubling lemma's constants (lam = 1.2q) are too lossy at s = 4/3, so R9(b) is proved only at b = 2. For b close to 3, R2(c) covers everything anyway.

## Next steps
- Inverse-theorem route to R11. Prove Bhatia-Davis stability: if (1/2) osc D_delta <= (1+eta)(E D_delta^2)^{1/2}, then D_delta is L^2-close to a two-valued function. Combine this with 'spectrum inside the cheap set' and with efficiency e(a) >= 0.8165. Square-wave-like flat D have harmonic-weighted progression spectra with an effective count of only about 2.5 frequencies, so they are thin in the E0 sense. Test numerically first: maximise c(Kv)_delta/((1/2) osc D_delta) over allocations with e >= 0.8165 and incompressible cheap parts.
- Extend R9 to the other natural supports of flat polynomials: certify compression for non-constant price profiles (Fejer, harmonic 1/j, bounded ratio), for 2-dimensional GAPs d1{1..q1} + d2{1..q2}, and for Bohr sets. Reuse compress_certify3.py, cert_iv_verify.py and a doubling lemma in each direction.
- Certified TDP*(p) at p = 61 and 67, b = 2: only allocations with e > 0.997 or 0.987 and binding kappa-mass > 29.7 matter. Run a branch-and-bound over supports with LP relaxations of the water-filled E0, which is concave in a (R3). This would push the rigorous b=2 range a few primes and test the method.
- Sharpen the doubling lemma (lam = (1+o(1))q via sharper E_q bounds near x = pi) and add base certificates at s = 4/3. This would give R9(b) for b in (2, 2.5], where the Parseval single-block range is already q <= 50 at b = 2.25.
- For K* >~ 55, where Parseval fails at the optimum: replace E D^2 by the exponential-tilt or 4th-moment detection restricted to the binding set. Start with an analysis of the large-deviation model in the Note ('P12 + tilt + (RM)' closes every kappa with margin 1.10 there), now with tau' instead of 3^{s-1}.
- Combine the p-uniform gain bound (R6: at most 10.8% over IF) with the GLOBAL R7 transition identities. At a K*-transition the count is a value statement, and R6 bounds how far any relations-case competitor can sit above IF.
