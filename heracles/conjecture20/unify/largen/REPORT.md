# largen: report (from the agent's structured output; the harness blocks subagents from writing report files)
## Headline
At large n and b=2, the existing tools for the single-feature price cannot work: the circularity constant k*(n,nu/2) of Companion Thm 14 grows with n (certified k*(32,16.5) >= 1.9057 > k_7 = 1.8199, k*(64,32.5) >= 2.7635), and the pointwise and polydisc (seed S4) payments also have constants that grow with n. The extremal atom systems behind this are spiky fields far outside the cone (-22x to -940x the IF price), so the routes fail, not (Q1). The searches agree: the best network found at p=5 (n<=12) and p=7 (n<=7) is always the IF block, with spurious clean amplitudes <= 1e-11 (numerically zero). One principle organises the angle and gives new n-uniform tools: the feature field is analytic in the label phases exactly when the inputs are frequency-pure. This yields circularity to order p-1 and a Rogosinski subordination bound, and it places (O4) and spurious features in the same, non-analytic, part. No proof of (C) at large n was obtained.

## Principle
ANALYTICITY OF THE FEATURE FIELD IN THE LABEL PHASES.

1. The object. A slot factor a -> u(a) rho^a is a polynomial in the phase v = rho^a:
   - class C: affine in z = v^2;
   - frequency 1 with constant (DC) components: degree 2 in v;
   - general inputs: realness of u brings in conjugate phases (at p=5, frequency-2 content creates the exponents 3 and -1), so the field is not analytic.

2. Why analyticity helps. Averaging the inputs slot by slot over product measures turns the field into a polynomial on a convex set of phase moments. That set contains a polydisc: radius cos(pi/p) for class C; radius r_p for DC networks, with r_5 = 0.612, r_7 = 0.801, r_11 = 0.919 and r_3 = 0. So the cone condition at the p^n inputs becomes: an analytic map sends a disc into a sector of half-angle pi/p. Two n-uniform families of detection inequalities follow:
   - (i) moment identities E(Phi-A)^k = 0 up to order p-1 (class C) or (p-1)/2 (DC), hence cone-moment bounds. The Companion's circularity is the case k=2.
   - (ii) slice subordination: every analytic slice has |k-th coefficient| * cos(pi/p)^k <= (4/p)|A-c|. This grows linearly in p, and on middle-order interference it is sqrt(C(n,k)) times stronger than L^2.

3. What it unifies.
   - The circularity theorem is its second-order shadow.
   - The p=3 counterexamples are exactly where analyticity fails: with DC inputs, 2 = -1 mod 3 gives r_3 = 0.
   - The two remaining large-n dangers have the same root. Non-circular cheats (O4) and spurious features both need input content at a second frequency, which is the non-analytic part.

4. Which cases it removes.
   - For frequency-pure fields, p=3 and p>=5 need no separate treatment, and every statement is uniform in n.
   - Width bounds for arbitrary sign patterns come from the same line and slice geometry (R1, R4).
   - What remains for (Q1) at large n is (a) one payment inequality in the subordination currency, and (b) the non-analytic part, which is the same object as the spurious-feature question.

5. Reformulation that exposes the large-n structure (R12). (Q1) is equivalent to c/L <= (K_r/nu)^{s-1}, where c/L is the cancellation efficiency of the atoms and K_r their r-participation number. At b=2 this is a contest between exponential rates. At fixed s it is not, and there the L^2 constant k*(n,s) decreases in n (R7).

## Results

### R1 [PROVED]

Width theorem for arbitrary sign patterns. Let p be an odd prime and n>=1. Take a slot-symmetric single-feature network: neuron i uses the same real input u_i in every slot, with any sign pattern, and its output is at frequency 1. If its margin is positive, its width satisfies m >= K+1-floor((K+1)/p) with K = floor(n/2), so roughly m >= (n/2)(1-1/p). Previously only shared-sign networks were covered (Note: nu - floor(nu/p)).

*Where:* Proof (given here).
1. By Companion Lemma 3, positive margin means Phi(N) lies in the open sector |arg| < theta = pi/p for every count vector N. This condition is open, so we may perturb zero entries of the u_i.
2. Let d = (p-1)/2, choose d' with 2d' = d mod p, and set a = b+d' and N(k) = (n-2k)e_b + 2k e_a for k = 0..K. Then Phi(N(k)) = e^{i(pi-theta)k} S(k), with S(k) = sum_i C_i r_i^k, C_i = W_i v_i(b)^n and r_i = (u_i(a)/u_i(b))^2 > 0. This uses rho^{2d'} = rho^d = e^{i(pi-theta)}: moving two slots at a time squares the sign ratio, so S is a real-base exponential sum whatever the signs.
3. arg S(k) lies in an arc of half-width theta centred at -(pi-theta)k. The Note's width-theorem argument now applies verbatim with n replaced by K: Laguerre's rule bounds the sign changes of h(k) = Im(e^{-i psi}S(k)) by m-1, while the signs at determined k are forced to alternate. Hence m-1 >= |D|-1 with |D| = K+1-floor((K+1)/p).
Why only n/2: a one-slot step gives Phi = e^{i(pi-theta)j} S_+ + e^{-i theta j} S_-, and the S_- part rotates slowly.
Check script: /home/user/grok/heracles/conjecture20/unify/largen/theory_checks.py, part (W).

*Check:* theory_checks.py (W): positive-margin slot-symmetric networks with perturbed, mixed-sign inputs (p=5,7; n=6,7,9). The forced alternation was tested 412 times and violated 0 times.

### R2 [PROVED]

Higher-order circularity and cone-moment bounds.
(a) Class C single feature: E_x (Phi-A)^k = 0 for 1 <= k <= p-1.
(b) Single frequency with DC: E_x (Phi-A)^k = 0 for 1 <= k <= (p-1)/2.
(c) If Phi(x) lies in c+K for every x (c real, K the closed sector of half-angle pi/p), then in both classes E|Phi-c|^k <= Re((A-c)^k)/cos(k pi/p) for 1 <= k <= (p-1)/2.
Both orders are sharp: the next moment is generically non-zero. Companion Lemma circ and Lemma cc are the case k=2. At p=3 with DC, part (b) is empty.

*Where:* Proof (given here).
(a) Write Phi - A = sum_{T nonempty} beta_T rho^{2 s_T(x)}. A product of k such terms is rho^{2 sum_t m_t x_t}, where 0 <= m_t <= k. Its mean is non-zero only if every m_t = 0 mod p. For k < p that forces all m_t = 0, which is impossible because every T is non-empty.
(b) The same argument with per-slot exponents in {0,1,2}, whose sums stay in [0,2k], inside [0,p-1] when 2k < p.
(c) E(Phi-c)^k = (A-c)^k by binomial expansion and (a)/(b). Since (Phi-c)^k lies in the sector of half-angle k pi/p < pi/2, Re (Phi-c)^k >= cos(k pi/p)|Phi-c|^k; take expectations.
Check script: /home/user/grok/heracles/conjecture20/unify/largen/theory_checks.py, parts (A) and (A').

*Check:* (A): the moment ratios are about 1e-17 up to the predicted order and about 1e-2 at the next order, for p=5,7,11, with and without DC.
(A'): on 198 positive-margin class-C networks, max LHS/RHS = 0.973 <= 1.

### R3 [PROVED]

Slice subordination: an n-uniform coefficient detection. Take a class C single feature with field polynomial P(z) = sum_i beta_i prod_t (1 + xi_it z_t), and let c be real with Phi(x) in c+K for every x (for example c = min cm). Then for every non-empty S in [n], every zeta in T^S and every k >= 1:
  cos(pi/p)^k |sum_{T in S, |T|=k} beta_T zeta^T| <= (4/p)|A-c|.
This holds for every odd p, every n and every width.
Consequences:
- Pointwise: Re A - c >= (p/4) cos(pi/p)^{|T|+1} |beta_T|, which is linear in p.
- Slot-symmetric atoms: C(n,k) cos^k(pi/p) |m_k| <= (4/p)|A-c|, where m_k = sum_i beta_i xi_i^k. This is sqrt(C(n,k)) stronger than L^2.
- Single frequency with DC at p >= 5: the same holds with radius r_p (r_5 = 0.612, r_7 = 0.801). There is no analytic slice at p=3 (r_3 = 0).

*Where:* Proof (given here).
1. P is multi-affine, so averaging the slots over product measures gives P on the polygon product conv(mu_p)^n, which therefore maps into the convex cone. That polygon contains the disc of radius R = cos(pi/p).
2. The slice f(w) = P(R zeta w on S, 0 off S) is a polynomial with f(0) = A and coefficients f_k = R^k sum_{|T|=k} beta_T zeta^T.
3. For c' < c, f is subordinate to the conformal map G of the disc onto the convex sector c' + int K, normalised by G(0) = A. Explicitly G(w) = c' + M(w)^{2/p}, with M(w) = (beta + conj(beta) w)/(1-w) and beta = (A-c')^{p/2}.
4. Rogosinski's theorem (for a convex univalent G, |f_k| <= |G'(0)|; Duren, Univalent Functions, Thm 6.4) gives |f_k| <= (4/p)|A-c'|. Let c' increase to c.
The radius r_p comes from /home/user/grok/heracles/conjecture20/unify/largen/rp_radius.py.

*Check:* theory_checks.py (B): 198 positive-margin class-C networks (p=5,7,11; n=2,3), with random S, zeta and every k: max LHS/RHS = 0.70 <= 1.
rp_radius.py (LP): r_3 = 0, r_5 = 0.612, r_7 = 0.801, r_11 = 0.919, r_13 = 0.942.

### R4 [PROVED]

Robust width from subordination (slot-symmetric class C, distinct atom phases xi_i, R = cos(pi/p)). For every polynomial q of degree <= n with q(0) = 1:
  |A| <= sum_i |beta_i| |q(xi_i)| + (4/p)|A-c| N_R(q),   N_R(q) = sum_{k>=1} |q_k| / (C(n,k) R^k).
(a) A positive margin forces m > n pR^2/(4+pR); the factor is 0.407, 0.551, 0.696, 0.737 at p = 5, 7, 11, 13, and tends to 1-4/p.
(b) Sharp width: if p cos(pi/p)^{n+1} > 4n (true whenever p >= 4n+5), every positive-margin slot-symmetric class-C single feature has at least nu neurons, the IF width.
(c) If the nu atom phases are equally spaced (masses and phases of beta arbitrary) and p cos(pi/p) >= 4 sum_k 1/(C(n,k)R^k), then c <= nu min|beta_i| <= nu^{1-s} Q^s. So (Q1) holds for these systems.

*Where:* Proof (given here).
1. The inequality: sum_k q_k m_k = sum_i beta_i q(xi_i). With q_0 = 1 this gives A = sum_i beta_i q(xi_i) - sum_{k>=1} q_k m_k; bound each m_k by R3.
2. (a) Take q = prod_i (z - xi_i) / prod_i (-xi_i). Then |q_k| <= C(m,k) <= C(n,k)(m/n)^k, and positive margin forces R|A-c| <= Re(A-c) < Re A <= (4/p)|A-c| N_R.
3. (b) For m <= n, N_R <= n R^{-n}. For p >= 4n+5, cos^{n+1}(pi/p) >= 1 - pi^2/(32(n+1)).
4. (c) Take q_j to be the normalised geometric series vanishing at every atom except j. Then |q_k| = 1 and |q_j(xi_j)| = nu; finish with power mean >= minimum.
Check scripts: /home/user/grok/heracles/conjecture20/unify/largen/check_R4.py and check_R4b.py.

*Check:* check_R4.py: 24 positive-margin networks, 20 polynomials q each: max |A|/RHS = 0.9975 <= 1.
check_R4b.py: (c) holds wherever it applies (p = 11, 13, 17): max c/(nu min|beta|) = 0.91. (b): 4000 random n-atom systems at each of (p,n) = (13,2), (17,3), (23,4), (29,5) all have c < 0.
The inequality p cos^{n+1}(pi/p) > 4n at p = 4n+5 was verified for all n <= 20000 (min slack 3.77).

### R5 [PROVED]

Computer-assisted, certified growth of the circularity constant at b=2. With k*(n,s) = sup (Re A - nu^{1-s}Q^s)/sqrt(E/2):
  k*(n,nu/2) >= 1.0551, 1.2146, 1.4267, 1.5417, 1.6543, 1.9057, 2.1233, 2.3634, 2.6946, 2.7635
  for n = 9, 12, 17, 20, 24, 32, 40, 48, 56, 64.
Since k_7 = sqrt(cot^2(pi/7) - 1) = 1.8199 < 1.9057, the hypothesis of Companion Thm 14 fails at (p,n,b) = (7,32,2). Its bound Lambda(k_7) is then strictly above the IF price, so Thm 14 cannot prove (Q1) there. At n = 9 the Note's 1.042 improves to 1.0551.

*Where:* The atom systems are explicit and slot-symmetric (xi_it = xi_i, so E = sum_k C(n,k) |m_k|^2). They are stored as exact dyadic float64 values in /home/user/grok/heracles/conjecture20/unify/largen/ (kstar_opt_n*.npy, comb_opt_n20_b2.0.npy, arcpolish_n24.npy, arcpolish_n32.npy, arcbig_n{40,48,56,64}.npy).
The ratio is enclosed with 200-bit interval arithmetic by kstar_certify.py; output in kstar_certified.out.
The searches are kstar_large.py (general atoms), kstar_sym.py / kstar_cont.py / kstar_comb.py (slot-symmetric), and kstar_exact.py / kstar_bign.py / kstar_arcpolish.py (exact arc family, beta = G^{-1}1).

*Check:* The general (non-symmetric) search reproduces the Note's values for n <= 8: 0.345, 0.543, 0.673, 0.764, 0.840, 0.928, 0.998. Its optima are slot-symmetric (inspect_kstar.py). The float64, mpmath and interval evaluations agree.

### R6 [CONJECTURE]

k*(n, nu/2) tends to infinity, roughly linearly: about 0.03 per unit n between n=24 and n=64. If so, Thm 14 cannot be n-uniform at any fixed p. At the current slope k_11 = 3.256 would be passed near n of about 75.
Proved ingredient: for fixed slot-symmetric phases, E >= |A|^2/([q]_n^2 - 1) for every q with q(xi_i) = 0 and q(0) = 1, where [q]_n^2 = sum_k |q_k|^2/C(n,k) is the Bombieri norm (Cauchy-Schwarz on A = -sum_{k>=1} q_k m_k). Equality holds at the least-squares atoms beta = G^{-1}1, with G_ij = (1 + conj(xi_i) xi_j)^n - 1.

*Where:* Data: /home/user/grok/heracles/conjecture20/unify/largen/kstar_bign.out and kstar_scan32.out. Structure of the extremisers: show_opt.py.
The extremisers have about 0.75n atoms, equally spaced on roughly a half circle, with Gaussian masses and atom phase linear in position. Along them IF/|A| tends to 0 while E/|A|^2 stays at about 0.2-0.5.

*Check:* The certified values in R5 rise monotonically. The least-squares formula sqrt(2 * 1^T G^{-1} 1) matches the direct evaluations.

### R7 [NUMERICAL]

At fixed s (so b = nu/s grows with n), k*(n,s) decreases in n and is largest at b=2:
- s=2: 0.543, 0.436, 0.339, 0.277, 0.237, 0.208 for n = 3, 5, 8, 11, 14, 17;
- s=4: 0.928, 0.876, 0.781, 0.673, 0.593 for n = 7, 8, 10, 13, 16;
- s=4.5: 0.998, 0.938, 0.851, 0.742 for n = 8, 9, 11, 14.
So b -> 2 is the only regime in which the L^2 route breaks down at large n.

*Where:* /home/user/grok/heracles/conjecture20/unify/largen/kstar_fixeds.py; outputs kstar_fixeds2.out, kstar_fixeds4.out, kstar_fixeds45.out.

*Check:* At n = 2s-1 the values agree with the b=2 values from the independent searches (0.543, 0.928, 0.998).

### R7c [CONDITIONAL]

Assume k*(n,s) <= k*(2s-1,s) for all n >= 2s-1, as observed in R7. Then Companion Thm 14 proves (Q1) for single-frequency networks at p=5 for every n whenever s <= 4, because k*(7,4) = 0.928 < k_5 = 0.9457. At p=7 the same holds for s up to about 15.

*Where:* Combine Companion Thm 14 with R7 and the hypothesis.

*Check:* The hypothesis holds numerically for s = 2, 4, 4.5 up to n = 17, 16, 14.

### R8a [REFUTED]

Seed S4's n-uniform polydisc payment is false. Take the explicit family of n atoms beta_i = A/n with xi_it = e^{2 pi i i/n}. Then P(z) = A(1 + z_1...z_n) exactly, and its gain is A(1-(n/nu)^{s-1}) > 0, while the sup of |P-A| over the polydisc D_R^n is A R^n. So any constant C in |A| - nu^{1-s}Q^s <= C sup_{D_R^n} |P-A| must satisfy C >= (1-(n/nu)^{s-1}) R^{-n}. At R = 1/2 that is >= 0.28 * 2^n, for every s > 1.

*Where:* Explicit family, analysed here; numbers from /home/user/grok/heracles/conjecture20/unify/largen/payment_consts.py.

*Check:* payment_consts.py: at b=2, R=1/2 the required constant is 4.6, 86.5, 2.4e4, 1.6e9 and 7.1e18 for n = 4, 8, 16, 32, 64.

### R8b [NUMERICAL]

The pointwise payment constant c_1(n,nu/2) = sup gain / max_T |beta_T| grows with n: >= 2.68, 3.58, 5.16, 6.17, 10.42, 12.75 at n = 9, 12, 17, 20, 48, 64. Riesz-type tests detect only about 1, so every pointwise route fails at large n when b=2.

*Where:* /home/user/grok/heracles/conjecture20/unify/largen/payment_consts.py, evaluated at 80 digits on the explicit R5 atom systems.

*Check:* The same systems give the certified k* values of R5, so the numbers are internally consistent.

### R9 [NUMERICAL]

The atom systems that maximise k* are far from the cone. Realised as class-C slot-symmetric networks, their cone margin divided by the IF price is:
- n=9: -22, -42, -68 at p = 5, 7, 11;
- n=12: -69, -98, -189;
- n=17: -406, -496 at p = 5, 7;
- n=20: -729, -940.
max|P|/|A| grows with n, from about 6 to about 147: the fields are spiky on concentrated inputs. L^2 does not see these spikes; the subordination bound R3 does.

*Where:* /home/user/grok/heracles/conjecture20/unify/largen/cone_of_atoms.py.

*Check:* The margins are computed over all count vectors exactly in double precision. The sign of the result is robust.

### R10 [NUMERICAL]

Multi-frequency single-feature searches and the spurious-feature check. Setup: slot-symmetric networks at b=2, arbitrary inputs, outputs on ALL frequencies.
- Best found: always the IF block (ratio about 1), at p=5 for n = 3..12 and at p=7 for n = 4..7.
- Spurious clean amplitudes at the best network: relative size <= 1e-11, and 1e-53 to 1e-69 for n >= 9, i.e. zero.
- Every non-IF local optimum loses by a factor >= 3.8 and carries spurious features of relative size 1e-6 to 1e-2.
- The Note's 1-D network (p=5, n=6, ratio 0.17723, reproduced) becomes the Note's KKT local maximiser when all frequencies are released (ratio 0.2643, spurious 1.3e-4), reproduced independently.
- Continued in n, that network decays: 0.264, 0.220, 0.181 at n = 6, 7, 8.

*Where:* /home/user/grok/heracles/conjecture20/unify/largen/mfnet.py, oned_seed.py, oned_cont.py; outputs mf_p5_small.out, mf_p5_6to10.out, mf_p5_11to12.out, mf_p7_3to7.out, oned_cont_p5.out.

*Check:* test_mf.py: the IF block gives ratio 1.0000 at p=5 and the Note's one-feature value 0.80085 at (7,3). The 1-D seed ratio matches the Note's saved value to 1e-12.

### R10b [REFUTED]

The Note suggests that at p>=5 every positive-margin single-feature network needs nu neurons ('No width below nu is feasible ...'). Its own saved network mechanism_scripts/kkt/oned_p5_n6.json is a counterexample: p=5, n=6, width 6 < nu = 7, positive margin. The shared-sign bound nu - floor(nu/p) = 6 is attained, consistent with the Note's Table 2.

*Where:* Recomputed independently in /home/user/grok/heracles/conjecture20/unify/largen/oned_seed.py: ratio 0.17723 > 0.

*Check:* Independent table and margin code (mfnet.py) agrees with the stored numerical_ratio to 1e-12.

### R11 [NUMERICAL]

The payment constant in the subordination currency, c_sub(n,s,R) = sup gain / max_k C(n,k) R^k |m_k| over slot-symmetric atoms at b=2. By R3, (4/p) c_sub(n,s,cos(pi/p)) < cos(pi/p) would give (Q1) for slot-symmetric class C.
- At R=1: 0.26, 0.38, 0.44, 0.40, 0.41, 0.37 for n = 2, 3, 4, 6, 8, 10. This looks bounded; the explicit family tends to 1-e^{-1/2} = 0.393.
- At R = cos(pi/7): 0.49, 0.84, 0.94 for n = 6, 8, 10, tracking the explicit family's 0.39 R^{-n}.
So disc slices reach (Q1) only for n up to order p^2 log p.

*Where:* /home/user/grok/heracles/conjecture20/unify/largen/csub.py; outputs csub_a.out, csub_b.out.

*Check:* These are lower bounds from local search (unreliable for n >= 12). They are consistent with the explicit-family lower bound.

### R12 [PROVED]

Rate reformulation. nu^{1-s} Q^s = (K_r/nu)^{s-1} L, where L = sum |beta_i| and K_r = Q^{1/(1-r)}/L^{r/(1-r)} is the r-participation number. Hence (Q1) holds iff c/L <= (K_r/nu)^{s-1}: cancellation efficiency against effective width. For general inputs use the masses M_i = B_p ||omega_i||^nu.
At b=2 the exponent is (n-1)/2, so this is a contest between exponential rates. With the Note's 1-D measurements:
- p=3: c/L about e^{-0.13n}, against (2/3)^{n/2} = e^{-0.20n}: violated, which is the counterexample mechanism;
- p=5: c/L about e^{-0.23n}, against (4/5)^{n/2} = e^{-0.11n}: satisfied.

*Where:* Identity: Q/L^r = K_r^{1-r} and (1-r)s = s-1.

*Check:* Exact identity. The rates use the Note's Table 2 and its gain accounting (0.5 log(2p/(p-1)) per unit n, measured log(zeta)/n).

## Obstruction
The large-n case stops in four places.

1. b=2 needs exponentially sharp payments. Any atom system with K_r <= (1-eps)nu has an IF price of at most e^{-eps(n-1)/2} L. So a proof of (Q1) must detect cancellation efficiencies that small. Payments that charge the gain to the sign-flip coefficients through l^2 (k*), l^infinity (c_1) or a polydisc of radius below 1 (seed S4) all have constants that grow with n (R5 certified, R8). The witnesses are slot-symmetric 'Gaussian combs': measures on an arc whose binomially weighted moments are small while their total mass is 10^2-10^4 times |A|. Their fields are spiky and leave the cone (R9).

2. Analytic slices see high-order interference only weakly. They control all interference of order |T| below about (p^2/pi^2) log p, but lose cos(pi/p)^k on degree-k coefficients. At fixed p and large n, interference of order |T| >> p^2 is visible only through aliasing at the roots of unity: on mu_p a polynomial of degree >= p is seen only through its coefficients summed modulo p. The family A(1+z_1...z_n) sits exactly there. Anchored slices (slice one slot, fix the others at roots of unity) detect it, but chaining anchored centres costs (1+4/(pR))^{|U|}.

3. Non-analytic inputs have no tools. Frequency-2 content at p=5 has neither moment identities nor subordination; only line arguments of Laguerre type remain (R1), and they bound width, not cancellation rate. This non-analytic part is exactly where (O4) cheats and spurious features would have to live. None was found numerically (R10).

4. The missing lemma is the payment in the subordination currency: gain <= C max_k C(n,k)|m_k| with C independent of n. Numerically C is about 0.4 at R=1.

Dead ends recorded here:
- the n-uniform polydisc payment of seed S4 (R8a);
- an n-uniform L^2/circularity route at b=2 (R5);
- induction on n by merging two slots: the merged network gains DC inputs, and its IF price at nu-1 is larger than at nu for the same atoms, so (Q1) at nu-1 is weaker than (Q1) at nu;
- the Laplace/tropical route, which reduced to the rate contest of R12 but did not yield a rate functional.

## Next steps
- FIRST LEMMA (subordination payment, slot-symmetric class C, R=1 model). Prove Re A - nu^{1-s}Q^s <= C max_{1<=k<=n} C(n,k)|m_k|, with m_k = sum_i beta_i xi_i^k and C independent of n. Numerically C is about 0.4; the explicit family needs C >= 1-e^{-1/2}. In words: a measure on the circle whose one-sided moments 1..n are binomially small must be spread over nu points in the l_r sense. Natural tools: the R4 duality (q vanishing on the heavy atoms) and the Bombieri norm of R6.
- Replace disc slices by anchored slices (slots fixed at roots of unity) for interference of order |T| >> p^2. Bound the growth of the anchored centres with the R2 moment identities rather than the multiplicative walk (1+4/(pR))^{|U|}. Target: (Q1) for class C uniformly in n at p >= p_0.
- Port R3 and R4 to single-frequency networks with DC at p >= 5, using the slice radius r_p (r_5=0.612, r_7=0.801). This would cover the DC part of the Companion's single-frequency class, uniformly in n, including the case b=2.
- Non-analytic part, which is (O4) together with spurious features. Combine the S2/S3 split (frequency mixing is free at b=2 without DC) with the numerical fact (R10) that non-analytic local optima lose by >= 3.8x. Target statement: the non-analytic component of a field costs unused budget at least proportional to the cone violation it repairs.
- Derive the cancellation rate of the 1-D family by Laplace-transform inversion of a rotating field. This would confirm -0.12 per unit n at p=5 and +0.07 at p=3, and turn R12 into a theorem for that family.
- Optional numerics: certify k* beyond k_11 = 3.256 (expected near n of about 75), and run R10 at p=7 for n = 8-9 with better 1-D seeds.
