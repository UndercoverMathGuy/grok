# C_ccstab: report (saved by the orchestrator from the agent's structured output)

## Headline
CC-STAB: I did not prove RD. Treating RD as a quantitative stability statement for the solved convex problem settles four questions, for every odd p, every n >= 2 and every b in [2, nu].

(1) THE TRUE ORDERS ARE THE REVERSE OF THE TASK'S GUESS.
- The interference deficit d_int is FIRST order in the interference amplitude. There is an explicit lower bound (Prop O).
- The per-neuron misalignment E_m sigma is second order. It splits exactly into five nonnegative gaps.
- The concentration a feature can buy with interference is SECOND order (via W6(c)).
- So at every interference-free (IF) equality case, RD is conically (first-order) stable. The task's model of quadratic deficit against first-order Renyi concentration is wrong.

(2) THE LOCAL REGIME IS NOW A THEOREM (Theorem L, class C, any phases, quantitative).
- gamma <= (1 - X) gamma_IF^{(S)} <= (1 - X) D_S, with X explicit.
- For aligned atoms this holds whenever the slack has no spike taller than nu/(2(s-1)) times an interference-weighted mean clean amplitude, at the binding offset. That factor is at least (n+1)/(n-1).
- The radius is parameter-free.
- The large-deficit regime is trivial.
- Everything hard sits in between:
  - Legendre networks (C_joint J6) violate Theorem L's hypothesis at every tested (p, n);
  - every stored competitive coupled network is NOT class C and relies on mixed modes.

(3) NO CERTIFICATE ON A FIXED-CONCENTRATION SLICE CAN WORK.
- The deficit is exactly 'neuron misalignment + mean of the slack field'.
- But its minimum at fixed Renyi concentration is NOT attained on decoupled networks: refuted at (5,2,2), with 0.8699 against 0.8185.
- Only the s-tilted functional has decoupled extremals, and that is RD itself.

(4) RD IS NOT HARDEST AT b = 2.
- For each network the hardest exponent is where its entropy H(theta, s) first rises above the decoupled entropy H^dec_S(s) (an entropy crossing).
- log Gamma_s is an upper envelope of finitely many concave vertex curves, so H^dec rises along each branch and drops at programme transitions.
- An exact slack-transport inequality reduces RD to b = 2 only for networks whose integrated entropy grows no faster than that of the b = 2 decoupled pair.
- Decoupled maximisers of earlier branches are tight there and slack at b = 2 (0.085, 0.118, 0.177 at p = 11, 13, 17).

Files are in /home/user/grok/heracles/conjecture20/unify/C_ccstab/. Proofs are in ccstab_proofs.tex; the scripts c1-c9 have matching .out files; sclib.py is the library.

## Principle
RD IS A SHARP (FIRST-ORDER, LP / HOFFMAN-TYPE) STABILITY STATEMENT LOCALLY, AND A TILTED (LEGENDRE) STATEMENT GLOBALLY.

(i) The deficits are moments of one nonnegative object, the slack field sigma(x, delta) = g(x, delta) - h >= 0.
- d_prog + d_int is its mean.
- d_int is the minimum over offsets of its x-mean.
- The interference energy is its x-centred second moment (Parseval in delta).
- 'Payment adds, detection does not' becomes concrete: a nonnegative function has second moment <= max * mean. Detection is therefore first order in the amplitude, and it degrades only through SPIKES (tall slack excursions).

(ii) Concentration bought by interference is second order (W6(c)), while detection is first order. So at every equality case the trade is first order against second order, and the local regime is easy: Theorem L, with an explicit spike threshold nu/(2(s-1)) times the harmonic-mean clean amplitude.

(iii) Decoupled networks are extremal only for the s-TILTED functional log gamma(theta;1) - (s-1) H_{1/s}(m_theta), never on slices of fixed concentration.
- Fixed Renyi concentration: refuted here.
- Fixed Shannon entropy: refuted by D8.
- Fixed profile: refuted by D9.
- The tilt makes the hardest exponent depend on the network: it is the entropy crossing, located at the convex kinks of log D_S (programme transitions) or at b = 2.

What this unifies:
- the IDENTITY (C_cc), the Note's slack identity and E_width W4/W6;
- C_joint J5/J7 (oscillation and Parseval closure) as single-offset cases of Theorem L;
- E3 and D4' (convex-end rigidity) as the X = 0 case.

What it removes:
- slice certificates;
- any reduction to the single exponent b = 2;
- Bianchi-Egnell quadratic stability as the model. The convex problem is an LP over the nuclear ball, so its stability is linear, not quadratic.

What it leaves:
- spiky or flat slack fields (the Legendre / relations core);
- compensation at a single offset through mixed modes, which is exactly where the stored violators live.

## Results

### S1 [PROVED]

SLACK FORM OF THE IDENTITY, valid for every product network with any margin h, every odd p and every n.

Notation:
- N := B_p Gamma_1 sum_i a_i;
- sigma(x,delta) = g(x,delta) - h >= 0 is the slack field;
- sbar(delta) = E_x sigma(x,delta) and sbar_sym(delta) = (sbar(delta) + sbar(-delta))/2.

Then:
- d_int N = min_delta sbar_sym(delta);
- d_prog N = E_delta sbar - min_delta sbar_sym;
- hence 1 - gamma(theta;1)/(B_p Gamma_1) = E_m sigma_neur + E_{x,delta} sigma / N.

Moreover the total interference energy is the x-centred second moment of the slack:
  sum_{l in F} E_x |Phi_l - E Phi_l|^2 = (2/p^2) E_x [ p sum_delta tau^2 - (sum_delta tau)^2 ],  with tau = sigma - sbar.

Reading: the programme and interference deficits are the mean of one nonnegative field, and interference is its fluctuation.

**Proof / evidence.**

1. The uniform dual gives E_{x,delta} g = Gamma_1 sum_l Re beta_l, and g = h + sigma. Hence Gamma_1 sum Re beta_l = h + E sigma.
2. Pairing delta with -delta: sum_l Re beta_l kappa_l(delta) = (1/2) E_x (g(.,delta) + g(.,-delta)) = h + sbar_sym(delta). This uses E_x Phi_l = conj(beta_l) and c_l(delta) + c_l(-delta) = 2 kappa_l(delta).
3. Substitute both into the definitions of d_prog and d_int in C_cc/IDENTITY.md.
4. Energy: the Note's slack identity Phi_j(x) = (2/p)(h - sum_delta sigma(x,delta) rho^{j delta}) gives f_j(x) = -(2/p) sum_delta tau(x,delta) rho^{j delta}.
5. For real tau on Z_p^*, sum_{j=1}^{p-1} |tauhat(j)|^2 = p sum tau^2 - (sum tau)^2, and j, -j give conjugate values.

*Check:* c1_slackform.py -> c1_slackform.out. Random networks at (5,2), (7,2), (5,3), (3,2), (11,2), (7,3), general and frequency-pure:
- identity error 8.5e-16;
- |d_int - min sbar_sym/N| <= 1.8e-15;
- |d_prog - (mean - min)/N| <= 1.3e-15;
- slack Parseval relative error 1.3e-15.
IF optima have x-independent slack (x-variance <= 1e-29), d_int = 0, and d_prog = 0.316 at (11,2,2), 0.211 at (7,3,2).

### S2 [PROVED]

JOINT DETECTION, valid for every network, plus a spike bound in class C.
(a) sum_l E_l <= 2(1 - 1/p) sigma_max N (d_prog + d_int), where sigma_max = max_{x,delta} sigma.
(b) Uniform-offset variance identity for every network, with F_delta = g(.,delta) - E_x g(.,delta) and f_l = Phi_l - E Phi_l:
  E_delta E_x F_delta^2 = (Gamma_1/2) [ sum_l E_l + 2 E_x (Re sum_l f_l)^2 ] >= (Gamma_1/2) sum_l E_l.
(c) In class C, sigma_max <= 2^{n+2} N / Gamma_1. Hence d_prog + d_int >= Gamma_1 sum_l E_l / (2^{n+3} N^2): a GLOBAL quadratic (Hermitian-square) certificate.

Reading:
- Detection is additive over features once all offsets are averaged.
- The certificate's constant 2^{-(n+3)} comes from Riesz-product spikes.
- At a single offset, the off-diagonal (cross-feature) terms do not vanish off class C. That is compensation.

**Proof / evidence.**

(a) E_x tau^2 <= E_x sigma^2 and sigma^2 <= sigma_max sigma. Drop -(sum tau)^2 in S1.
(b) F_delta = Re sum_l f_l c_l(delta). For l, l' in F:
- E_delta c_l conj(c_l') = Gamma_1 (1 + [l = l']);
- E_delta c_l c_l' = Gamma_1, because l + l' is not 0 mod p.
Write F^2 = (1/2)|sum f c|^2 + (1/2) Re (sum f c)^2.
(c) In class C the field of a neuron at frequency l is conj(beta_il) prod_t (1 + e_it z_t) with |e_it| = 1. So:
- |Phi_l| <= 2^n sum_i |beta_il|;
- |g| <= 2^{n+1} B_p sum a, because |c_l| <= 2 and the r = 1 budget holds;
- sigma = g - h <= 2 max |g|.

*Check:* c1_slackform.out:
- min of [2(1-1/p) smax N D' - sum E]/N^2 = 0.90 (>= 0);
- min of [2^{n+2} N/Gamma_1 - smax]/N = 7.68 (>= 0);
- uniform-offset identity relative error 5.8e-16;
- the single-offset diagonal formula deviates by up to 70% on general networks (cross terms are real).
c5_adversarial.out: on 24 stored non-IF networks the ratio 2(1-1/p) smax N D'/sum E lies in [2.4, 1.1e5] (>= 1), with smax/N in [0.6, 4.6].

### L [PROVED]

THEOREM L (local regime, quantitative). Valid for every odd p (p = 3 included), every n >= 2, every b in [2, nu] and every S.

Let theta be in N_S and of class C: all nu factors of each neuron lie in one frequency plane j(i).

Notation:
- A_l = beta_l; L_l = sum_i |beta_il|; Q_l = sum_i |beta_il|^r; a_l = nu^{1-s} Q_l^s;
- E_l = sum_{T nonempty} |beta_{l,T}|^2;
- V'_l = (2/(nu L_l^2)) sum_I (|beta_{l,I}| + sqrt(2 L_l (L_l - |A_l|)))^2, summed over the n(n+1)/2 intervals I;
- a_l^- = L_l max{ (1 + V'_l)^{-(s-1)}, nu^{1-s} };
- delta_0 in argmin_delta sum_{l in S} a_l kappa_l(delta), and O = max( osc_x g(.,delta_0), osc_x g(.,-delta_0) );
- X = [ sum_l kappa_l(delta_0) E_l / O - sum_{l in S} kappa_l(delta_0)(|A_l| - a_l^-) ] / sum_{l in S} kappa_l(delta_0) a_l.

If X >= 0, then gamma(theta;s) <= (1 - X) gamma_IF^{(S)}(s) <= (1 - X) D_S(s).

In particular RD holds with explicit first-order slack: log(D_S/gamma) >= X. The hypothesis X >= 0 says that the detected interference at the binding offset beats the payment bound.

**Proof / evidence.**

Margin step.
- h <= (1/2)(min_x g(.,delta_0) + min_x g(.,-delta_0)) = sum_l Re beta_l kappa_l(delta_0) - (m_{delta_0} + m_{-delta_0})/2, where m_delta = -min_x F_delta.
- Mean-zero lemma: if E F = 0 then m (max F - min F) >= E F^2. Proof: E F_+^2 <= max F E F_+ = max F E F_- <= max F m, and E F_-^2 <= m^2.
- Class-C variance lemma: u_t(x) rho^{l x} = (conj(U_t) + U_t rho^{2lx})/2 gives f_l = sum_{T nonempty} conj(beta_{l,T}) rho^{2l <1_T, x>}, using uhat_t(+-l) = (p/2) U_t^{(+-)}.
  - The characters 2l 1_T (l in F, T nonempty) are nonzero and distinct, and 2l 1_T != -2l' 1_{T'}: that would need l + l' = 0 on T intersect T' and 2l = 0 elsewhere.
  - So E f_l conj(f_l') = [l = l'] E_l and E f_l f_l' = 0, hence E_x F_delta^2 = sum_l kappa_l(delta) E_l.
- Therefore m_{+-delta_0} >= sum_l kappa_l E_l / O.

Payment step.
- E_width W6(c) gives nu^{1-s} Q^s >= L (1 + V)^{-(s-1)} with V = (2/(nu L^2)) sum_{t > t'} |K_{tt'}|^2.
- Since K_{t0} = beta_{[1,t]} and |K_{tt'} - conj(alpha) beta_{[t'+1,t]}| <= sqrt(2L(L - |A|)), we have V <= V'.
- Also Q >= L^r, so a >= nu^{1-s} L. Hence a_l >= a_l^-.

Conclusion.
- Combining, h <= sum_S kappa_l(delta_0) a_l (1 - X).
- sum_S kappa_l(delta_0) a_l = min_delta sum_S a_l kappa_l <= Gamma^{(S)}_s (sum a_l^r)^s = nu^{1-s} Gamma^{(S)}_s (sum Q_l)^s <= nu^{1-s} Gamma^{(S)}_s B_p P_b^s, by the class-C budget.
- Finally pi >= nu^{1-s}.

No step uses p >= 5. This is consistent with BRIEF_CORE section 3 because the p = 3 winners are not class C (they use DC).

*Check:* c4_local.py -> c4_local.out. 621 positive-margin class-C networks:
- p in {3, 5, 7, 11}, n in {2, 3, 4}, b in {2, 2.4, 2.5, 3};
- aligned, misaligned, and J4 single-set (cheap) families at perturbation sizes 0 to 1.

Every step checked (all >= 0 up to rounding):
- min[a_l - a_l^-] = -7.8e-16;
- variance identity error 7.5e-16;
- mean-zero lemma >= 3e-31;
- margin step -1.3e-15;
- min[(1 - X) - gamma/gamma_IF] = -6.7e-16.

(L) holds on 587 networks. Where it holds, max gamma/gamma_IF = 1.000000 (IF only); where it fails, the max is 0.688.

### L1 [PROVED]

COROLLARY L1 (aligned atoms; parameter-free spike radius). If within each feature all atoms share one phase, then X >= 0 whenever
  O <= (nu/(2(s-1))) Abar,  with Abar = sum_l kappa_l(delta_0) E_l / sum_l kappa_l(delta_0) E_l/|A_l|,
the interference-weighted harmonic mean of the clean amplitudes.

Since O <= max_x sigma(x, +-delta_0): an aligned class-C network whose slack at the binding offset has no spike taller than nu/(2(s-1)) Abar never beats gamma_IF^{(S)}. That threshold factor is at least (n+1)/(n-1) at b = 2.

**Proof / evidence.**

Alignment gives L = |A|, zero twist, and V' = 2E^I/(nu|A|^2) with E^I <= E. Bernoulli gives (1 + V)^{-(s-1)} >= 1 - (s-1)V, so |A_l| - a_l^- <= (2(s-1)/nu) E_l/|A_l|. Substitute into X.

*Check:* c4_local.out: 'L1 => L' holds on all 621 networks.

The theorem is silent on the stored and Legendre networks:
- the stored violators and non-IF local maxima are not class C (2-8% of their energy lies outside the dominant plane). Their frequency splits, which are class C, keep only 1-62% of the original gamma/gamma_IF and sit at L1 ratio 1.28-4.70 > 1 (c5_adversarial.out);
- Legendre networks: L1 ratio 2(n-1) sqrt(p)/(n+1) >= 1.49 at b = 2 (c9_legendre.out).

### L2 [PROVED]

COROLLARY L2 (any phases). X >= 0 whenever
  sum_{l in S} kappa_l(delta_0) [ (4(s-1)/(nu L_l)) E^I_l + (4n(s-1) - 1)_+ (L_l - |A_l|) ] <= sum_l kappa_l(delta_0) E_l / O.

Near an IF configuration the left side is second order (E^I and the cancellation L - |A| are both quadratic), while the right side is first order. So class-C networks near IF never beat gamma_IF, whatever the atom phases.

**Proof / evidence.**

(s-1) L V' <= (2(s-1)/(nu L)) [ 2E^I + 2 #I tau^2 ], with #I = n nu/2 and tau^2 = 2L(L - |A|), which equals (4(s-1)/(nu L)) E^I + 4n(s-1)(L - |A|). Then |A| - a^- <= |A| - L(1 - (s-1)V') by Bernoulli.

*Check:* c4_local.out: 'L2 => L' holds on all 621 networks. L2 holds on 12-16 of every 16-19 misaligned networks per case.

### O [PROVED]

TRUE ORDERS (this settles the task's question; the task's guess was the reverse).

(i) Five-gap decomposition. sigma_i B_p a_i = G_amgm + G_dc + G_mix + G_hol + G_phase, each >= 0. These are the gaps of the budget chain (AM-GM, DC mass, l_nu versus l_2 mixing, Hoelder frequency mismatch, atom phase). They vanish exactly on aligned, balanced, DC-free single-frequency neurons, so E_m sigma is second order there.

(ii) Class C, first-order interference deficit:
  d_int N >= min_delta sum_l kappa_l(delta) E_l / O_delta >= kappa_min sqrt(sum E_l) / (4 sqrt((2^n - 1) K')),
with kappa_min = 1 - cos(2pi/p) and K' the number of features with E_l > 0.

(iii) Aligned atoms: the concentration a feature can buy is second order, log(|A_l|/a_l) <= (s-1) log(1 + 2E^I_l/(nu |A_l|^2)). Under any phases near IF it is second order as well (L2).

Consequence: at every IF network the RD-IF slack is at least X = (first order) - (second order). RD is conically stable at its interference-free equality cases.

**Proof / evidence.**

(i) The budget chain is sum_j |beta_ij| = c prod ||v_t|| sum_j prod_t A_t(j) <= c prod ||v_t|| prod ||A_t||_nu <= c prod ||v_t|| prod ||A_t||_2 <= c prod ||v_t|| <= B_p a_i, with c = 2(2p)^{-nu/2} and A_t(j) = |vhat_t(j)|/(sqrt(p/2)||v_t||). Written as a telescoping sum plus the phase gap sum |beta| - sum Re beta. The equality cases are those of E3 / E7.
(ii) sbar_sym(delta) >= (m_delta + m_{-delta})/2. Then apply the mean-zero and variance lemmas. Bound O_delta <= 2 sum_l |c_l| max|f_l| <= 4 sum_l sum_T |beta_{l,T}| and use Cauchy-Schwarz.
(iii) Restates W6(c).

*Check:* c8_sigma_gaps.out: on 1000 neurons the gap identity holds to 9.9e-16, and the minimum gap is 0.
c4_local.out: the d_int lower bound holds with min slack 2.8e-17 (>= 0).

### O-num [NUMERICAL]

Orders along rays theta_0 + eps V from the decoupled IF block-union maximiser at b = 2, at (5,2), (7,2), (11,2), (5,3). Fitted log-log slopes for eps in [1e-4, 1e-2]:
- E_m sigma: 2.00.
- d_int: 1.00, in every direction (general, aligned class C, phase-only, mass-only).
- Renyi drop H_{1/s}(m_0) - H_{1/s}(m): 2.00 where masses are uniform within the support. It is about 1 at (11,2), but only through allocation (decoupled) directions, where d_prog moves at first order too.
- RD-IF slack log(gamma_IF/gamma): 1.00-1.02 in every direction.

**Proof / evidence.**

c6_orders.py, 6 random directions per kind, 7 eps values.

*Check:* c6_orders.out

### H1 [PROVED]

PROGRAMME ENVELOPE (valid for every p and S).
- log Gamma^{(S)}_s = max over v in V_S of phi_v(s), where V_S is the finite, s-free vertex set of {v >= 0, spt v within S, K v >= 1}.
- phi_v(s) = -log sum v - (s-1) H_{1/s}(vhat) is concave, with phi_v'(s) = -H(esc_{1/s} vhat).
- Hence the programme part of H^dec_S is non-decreasing along each branch and drops at every transition. H^dec_S is not monotone.
- So the per-network RD slack S_theta(s) = log D_S(s) - log gamma(theta;1) + (s-1) H_{1/s}(m_theta) has derivative H(theta,s) - H^dec_S(s) at differentiability points of log D_S. Its minimum (the hardest exponent for theta) is an entropy crossing: on a kink, or at b = 2.

**Proof / evidence.**

- sum v^{1/s} is concave in v and tends to infinity along recession directions, so its minimum over the polyhedron is at a vertex; hence Gamma = max_v (sum v^{1/s})^{-s}.
- The derivative is W1 applied to vhat, and escort entropy is non-decreasing in s.
- An upper envelope of crossing concave curves has upward derivative jumps, i.e. drops of H.

*Check:* c2_hardest.out, H^dec(s) at n = 2 (pi = 3^{1-s}):
- p = 11: rises at 14 grid steps (at most +2.2e-4 per 0.01), drops twice (largest drop 0.526);
- p = 13: 20 rises, largest drop 0.437;
- p = 17: 25 rises, largest drop 0.354;
- p = 5, 7: constant, with no transition in [1, 1.5].

### H2 [PROVED]

SLACK TRANSPORT (a partial reduction to b = 2). Let 1 <= s <= s_1 <= nu/2. Let (theta_1, y*) be any s_1-optimal pair: an optimal single feature and a maximiser of Gamma^{(S)}_{s_1}. Write H^pair(sigma) = H(theta_1, sigma) + H(esc_{1/sigma} yhat*). Then for every theta in N_S:
  S_theta(s) >= S_theta(s_1) + int_s^{s_1} [ H^pair(sigma) - H(theta, sigma) ] d sigma.
Hence RD at s_1 implies RD at s for every theta with int_s^{s_1} H(theta,.) <= int_s^{s_1} H^pair.

In words: RD at b = 2 controls exactly the networks whose integrated cost-share entropy grows no faster than that of the b = 2 decoupled pair.

**Proof / evidence.**

1. gamma^dec(sigma) := e(theta_1;sigma) B_p G_sigma(y*) <= pi(sigma) B_p Gamma_sigma = D_S(sigma), with equality at s_1.
2. (log gamma^dec)' = -H^pair, by W1 for theta_1 and D1(iv) for y*.
3. log gamma(theta;s) - log gamma(theta;s_1) = int_s^{s_1} H(theta,.).
4. Subtract. No RD hypothesis is used except at s_1 in the corollary.

*Check:* c5_adversarial.out: at n = 2, s_1 = 1.5, on all stored n = 2 networks and s in {1.05, 1.2, 1.35}, the minimum of [S(s) - S(1.5) - int(H^pair - H)] is -1.2e-8 (quadrature error).

### H3 [PROVED]

NO SINGLE-EXPONENT REDUCTION, network by network.
- For every s_0, each s_0-optimal decoupled network has S(s_0) = 0. Whenever it is not optimal at nu/2, it has S(nu/2) > 0.
- So no argument that uses only the s-free data (gamma(theta;1), m_theta) can deduce RD at s_0 from RD at nu/2. Replacing gamma(theta;1) by e^{eps} gamma(theta;1), with eps < S(nu/2), gives data that satisfy RD at nu/2 and violate it at s_0.
- So the RD family is genuinely one-parameter. Only H2's partial transport survives.

**Proof / evidence.**

Immediate from the definitions. The strict case occurs (see H3-num).

*Check:* c2_hardest.out: first-branch decoupled maximisers have S identically 0 on [1, 1.35) and S(1.5) = 0.0853 (p = 11), 0.1182 (p = 13), 0.1769 (p = 17).

### H3-num [NUMERICAL]

Where RD is hardest, at n = 2.

p = 5, 7 (no transition):
- every perturbed IF network and every stored violator is hardest at b = 2.

p = 11:
- perturbations of first-branch decoupled maximisers are hardest at the transition s = 1.34-1.35, with slack at b = 2 larger by 0.085;
- the stored p = 11 violators are hardest at s = 1.35, where H(theta,s*) - H^dec(s*) = +0.18 (they cross at the kink).

IF networks on every vertex of the coverage polyhedron (counts are b = 2 / programme transitions / interior / s = 1):
- p = 11: 14/15/2/0 of 31;
- p = 13: 53/31/3/0 of 87;
- p = 17: 102/283/6/0 of 391.

So RD is hardest at b = 2 or at programme transitions, with rare interior exceptions; it is never hardest at s = 1.

**Proof / evidence.**

Exact vertex enumeration of Gamma_s (E_reduce) on 51-101-point s grids; exact margins of the networks.

*Check:* c2_hardest.out, c2b_vertices.out

### P [PROVED]

PROFILE FORM. RD_S holds iff, for all theta in N_S,
  log gamma(theta;1) <= U_S(m_theta) := inf over s in [1, nu/2] of [ log D_S(s) + (s-1) H_{1/s}(m_theta) ].

U_S is Schur-concave in the mass vector. The minimiser in U_S(m_theta) is theta's hardest exponent.

Far regime (trivial): every theta with gamma(theta;1) <= D_S(s) satisfies RD at s, because H_{1/s} >= 0.

**Proof / evidence.**

- W1: log gamma(theta;s) = log gamma(theta;1) - (s-1) H_{1/s}(m).
- Renyi entropies are Schur-concave.

*Check:* A logical restatement; the far regime is consistent with every .out file.

### R [REFUTED]

Decoupled networks minimise the convex-end deficit at given Renyi concentration'. Equivalently: for all theta, 1 - D(theta) <= sup{ 1 - D(Theta) : Theta decoupled, H_{1/s}(m_Theta) <= H_{1/s}(m_theta) }.

So no identity 'deficit = explicit nonnegative form whose minimum at given Renyi concentration is attained on decoupled networks' can exist. Only the s-tilted problem (P) can have decoupled extremals, and that is RD itself. Together with D8 (Shannon slices) and D9 (profile slices), every slice certificate is now refuted.

**Proof / evidence.**

Counterexample at (5,2,2), s = 3/2, order 2/3: the certified SPLIT violator theta_v (exact rational margin in split/certify_viol.out).
- It has H_{2/3}(m) = 1.53770 and gamma(theta;1)/B_p = 0.86993.
- Decoupled networks of D_F at n = 2 are KM2 two-block unions (Note Thm q1n2 together with KM2 rigidity, as in D8). For them H_{2/3} = log 3 + H_{2/3}(yhat) and gamma_1/B_p = M(yhat) = 0.691 + 1.118 min(yhat).
- Under the cap H_{2/3}(yhat) <= 0.43909 the best is M = 0.81848, at yhat = (0.886, 0.114).
- theta_v exceeds this by 6.3%. The two sister violators give 0.8606 against 0.8127 and 0.8435 against 0.8029.

Caveat: decoupled networks built from non-optimal single features of Renyi entropy below log 3 were not searched.

*Check:* c3_frontier.out

### G [NUMERICAL]

WHERE THE HARD CASES LIVE: the census of BRIEF_CORE section 3 and the stored networks.

(a) Stored networks (SPLIT violators and E_kkt non-IF local maxima, 24 networks):
- None is class C: 1.6-7.9% of their energy lies outside the dominant plane.
- Their frequency splits keep only 1-62% of the original gamma/gamma_IF (e.g. viol_p5:0 0.790 -> 0.379). So they rely on mixed modes, and Theorem L is silent on them.
- The violators pay mainly in d_prog (0.17-0.64, even at p = 5, 7 where the decoupled optimum has d_prog = 0) and in E_m sigma (0.03-0.06). Their d_int is only 0.0008-0.003: they are flat at the binding offset. Two local maxima pay mainly in d_int (0.17, 0.30).

(b) Legendre networks (C_joint J6):
- (L) fails at every tested (p,n) in {5,...,19} x {2} and {5,7,11} x {3};
- L1 ratio = 2(n-1) sqrt(p)/(n+1) at b = 2 (>= 1.49);
- gamma/gamma_IF = 0.13-0.54.

(c) Single class-C neuron (maximal concentration). Second-moment detection at the binding offset is 0.53-1.23 |beta| kappa_min, against a needed payment 1 - nu^{1-s} = 0.42-0.99. It is enough at n = 2 and too weak at p = 5, n >= 3. So averaged detection is not a uniform route in the global regime: the minimum over x (the cone) is needed.

**Proof / evidence.**

Exact margins and fields on stored tables; Theorem-L ledgers.

*Check:* c5_adversarial.out, c9_legendre.out, c7_singleneuron.out

## Obstruction
The stability route stops here, for every p, n and b.

1. THE EQUALITY CASES ARE FAR FROM THE CONVEX OPTIMISERS.
- At s > 1, RD is tight on decoupled maximisers with d_prog = O(1): 0.316 at (11,2,2), 0.211 at (7,3,2), 0.447 at (5,3,2).
- Stability around the solved convex end (E3) therefore controls only a neighbourhood of s = 1. Any statement extracted from it is of 'b near nu' type, which is forbidden.
- Example: S2's global quadratic certificate turns Theorem L into a statement only when s - 1 <= nu 2^{-(n+3)} Gamma_1 min|A_l|/N, i.e. b near nu. That special case was not pursued.

2. NOTHING HARD IS LOCAL.
- At every interference-free equality case the trade is first-order deficit against second-order concentration (Prop O), and RD is conically stable.
- Theorem L settles the local regime in class C with an explicit, parameter-free spike threshold.
- Bianchi-Egnell quadratic stability is the wrong model. The convex problem is an LP over the nuclear ball (D3b), so its stability is linear (Hoffman-type).

3. NO SLICE CERTIFICATE EXISTS.
- The deficit is an explicit nonnegative functional (S1).
- Its minimiser at fixed Renyi concentration is not decoupled (R). The same fails at fixed Shannon entropy (D8) and at fixed profile (D9).
- Decoupled networks are extremal only for the s-tilted functional, which is RD (P).
- So any certificate must carry the tilt s - 1 and the decoupled entropy H^dec(s). Because H^dec is non-monotone (H1), the hardest exponent depends on the network (H2, H3); b = 2 is not a sufficient single test.

4. THE MIDDLE (GLOBAL) REGIME IS THE OLD CORE.
- For class C it consists of spiky or flat slack fields. Legendre networks (J6) violate Theorem L's hypothesis at every tested (p, n) and satisfy RD; J6 shows any allocation-blind detection fails there.
- Off class C, single-offset detection fails: cross-feature terms change E_x F_delta^2 by up to 70% (S2(b)), which is exactly compensation through mixed modes.
- Only the uniform-offset average detects additively (S2(b)), and it costs the convex-end value Gamma_1 > Gamma_s.
- All stored competitive coupled networks are in this regime: non-class-C, mixed-mode reliant, and paying in d_prog rather than d_int.
- A general proof needs a 'compensation costs coverage' inequality: flattening a binding offset through mixed modes must move the delta-profile of the mean slack (d_prog) by at least the concentration gained, measured against the s-programme rather than the convex end. No general version is in hand.

## Next steps
- Multi-offset Theorem L. Replace the single binding offset delta_0 by the s-programme dual measure mu_s on the binding set.
- Detection then becomes the Gram form G_{ll'}(mu) = 1 - muhat(l) - muhat(l') + muhat(l - l') plus a pseudo-variance term (S2(b) is the case mu uniform, with G = Gamma_1 (I + J)).
- If lambda_min of the Gram form over programme duals is bounded below by a programme-only quantity, Theorem L extends beyond class C. Compensation would then be charged at the s-price, not the convex-end price.
- First test: compute lambda_min(G(mu_s)) exactly by vertex enumeration for p <= 23.
- 'Compensation costs coverage'.
- Data: the stored violators have d_int <= 0.003 but d_prog = 0.17-0.34 at p = 5, 7, where the decoupled optimum has d_prog = 0. Their splits lose 38-99% of the margin, so mixed modes do the flattening.
- Use SPLIT's explicit 2|F|-1 mixed-mode channels and S1 (the delta-profile of the mean slack fixes the clean allocation through beta_j = (2/p)(h - sum_delta sbar(delta) rho^{j delta})).
- Goal: an inequality saying that a mixed-mode field cancelling the x-variance at the binding offsets forces the slack's delta-profile away from the programme profile, at the rate needed by the exchange inequality.
- Spike control at maximisers. Theorem L needs the slack at the binding offset to have no spike taller than (n+1)/(n-1) Abar at b = 2.
- Use the KKT measure of E6 (binding-margin shares equal cost shares) to bound the height of the slack field on the support of the dual at a maximiser.
- Any such bound proves RD in class C for maximisers in the aligned case. The non-aligned case still needs the cancellation term of L2.
- Profile functional. RD holds iff gamma(theta;1) <= exp U_S(m_theta), with U_S Schur-concave.
- Look for a network-level Lorenz (majorisation) form of Lemma 7: show that m_theta is majorised by a decoupled mass profile up to a loss of convex-end efficiency.
- The entropic Lemma 7 (E2 / D5 Schur-Horn) does this within one feature.
- Do not pursue:
- slice certificates at fixed concentration, entropy or profile (R, D8, D9);
- reduction to b = 2 (H3);
- global quadratic (Hermitian-square) certificates, whose constant is 2^{-(n+3)} (S2);
- local analysis at the convex end (item 1 of the obstruction).
