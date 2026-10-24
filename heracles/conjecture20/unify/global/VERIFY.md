# Referee verdicts on global/REPORT.md (round 2; saved by the orchestrator from the referee's structured output)

## Overall
I re-derived all eleven claims line by line, and none is false. Seven are CONFIRMED: R1, R2, R3, R5, R6, R8, R9 and R13. Three are MINOR_GAP: R4, R7 and R12. I found no SERIOUS_GAP and no FALSE claim.

None of the PROVED claims implies the IF value at p=3:
- R3 and R4 are conditional, and their hypotheses fail on the exact p=3 certificates. At n=11, H=log 8 < log 12; at n=13, H=log 10 < log 14. I re-checked gamma/gamma_IF exactly: 1.054399 and 1.109163.
- R9 uses p>=5 exactly where it is needed.
- R6 and R13 use pi>=nu^{1-s} and gamma*>=gamma_IF only as lower bounds.

The three gaps:
- **R4(b):** the integrand is not upper semicontinuous, so the proof's "as in R3" step is invalid as written. The conclusion still holds via Vitali-Caratheodory.
- **R7:** the p=7 transition at s=2.3247 is listed as in range for n=3; it needs n>=4. The "all p>=11" statement was only checked in floating point up to 199, and the certificate's wording is false as written.
- **R12:** the hypothesis range (1.331,1.5] and Thm dense (s<1.330839) leave the sliver [1.330839,1.331] uncovered.

The headline interpretation of R7 is overstated. W_{K<}(s_c)=gamma*(s_c) is equivalent to the count statement "some maximiser at s_c has <=K_< features". It involves neither pi nor Gamma. So it does not show that the count cannot be separated from the price-programme value; only the slack-principle method needs exact decoupling at s_c.

All my independent scripts and outputs are in /home/user/grok/heracles/conjecture20/unify/global/verify/:
- vnet.py: own library.
- v1_temperature: 50-digit check of R1.
- v2_programme: 40-digit vertex enumeration for the tolerances, transitions, dense threshold and slack numbers.
- v3_dense_end and v3b_small_primes: interval certificates for primes 11..1000 plus a lacunary proof for p>=892.
- v4_directional and v4b: random and adversarial tests of R9 (i), (ii), (iii) and Theorem A.
- v5_sharefloor: R13.
- v6_p3: exact p=3 checks.
- v7_w1_search: independent search for the R12 hypothesis.
- v8_profile_union: R8.
- v9_budget_AD: R6 corollary (a).

## Salvage
- **R4(b), corrected:** the integrated bound log gamma* - log gamma_IF <= int_1^s phi still holds for the bounded Borel integrand phi. Take an lsc majorant G>=phi with int G <= int phi + eps; then D_+ int G >= G, so L - int G is non-increasing.
- **R7 strengthened (computer-assisted with interval arithmetic, now complete):** EVERY prime p>=11 has a K*-transition in (1,1.5).
  - Programme maximisers are vertices.
  - The only possible full-support vertex is (2/p)1, of value (p/2)F^{-s}.
  - K*(1,p)=F, uniquely, because C^2=(p/4)I-(1/2)J for the cosine matrix.
  - Interval-certified allocations beat (p/2)F^{-1.5} for all primes 11<=p<=1000.
  - The lacunary support {2^k : 2^{K-1}>=p/3} gives Gamma_1.5 >= 1.5 K^{-1.5} > (p/2)F^{-1.5} for all p>=892.
- **R7 corrected range:** p=7's second transition (2.3247) is in range only for n>=4.
- **R7 corrected reading:** W_{K<}(s_c)=gamma*(s_c) is equivalent to "a global maximiser with <=K_< features exists at s_c" (a count statement). It is not a price-programme value identity.
- **R6/R12 corrected threshold:** the p=5, n=2 dense threshold is s<1.330839 (b>2.25424), not 1.331.
- **R12 corrected:** (C) at (5,2) for all b in [2,3] follows from W_{1}(s) < 3^{1-s} B_5 Gamma_s on s in [1.33,1.5]. The slack there is at least 27.9168% (1.279168 at s=1.5).
- **R12 at (7,2):** the hypothesis is needed on s in (1.1539,1.5], i.e. b in [2,2.600), with slack 2.035% (b=2) to about 18.4% (b=2.6).
- **R12 evidence:** an independent 144-start search finds no compensation gain (max ratio 1.000000).

## Verdicts

### R1-temperature: CONFIRMED

Re-derived line by line. Write a_i=||omega_i||^nu, so ||omega_i||^b=a_i^{1/s}, and Z=sum a_i^{1/s}. Then log gamma = log h - s log Z, and h does not depend on s for a fixed network. Phi'=log Z - xbar/s = -sum w_i log w_i = H. Next, dw_i/ds=-w_i(x_i-xbar)/s^2 and dxbar/ds=-Var_w(x)/s^2, so Phi''=Var_w(x)/s^3=Var_w(log w)/s. Hence d/ds log gamma=-H and d2/ds2 log gamma=-V/s<=0. The only hypotheses are h>0 and zero neurons dropped; zero neurons contribute nothing anyway. The identity holds for every s>0, not only for [1,nu/2]. It holds for b>2 and b=2 alike, for arbitrary inputs with DC, and for any width. The cone-efficiency version is identical, because its numerator min_x cm(Phi_1) is also s-free. Independent check (verify/v1_temperature.py -> v1_temperature.out): 40 random networks, p in {3,5,7}, n in {2,3}, s in {1, 1.37, nu/2}, 50-digit central differences. Results: max|dlog/ds+H|=8.6e-26 and max|d2+V/s|=1.7e-25. The same script checks vnet's conventions: E Phi_1=conj(beta_1), IF-block gaps = kappa Re beta, and the single-neuron budget is an equality on IF neurons.

### R2-danskin: CONFIRMED

Reconstructed the proof completely.
(1) Width reduction. The table is sum mu_i T^_i with unit rank-one T^_i. On the pointed polyhedron {mu>=0 : sum mu_i T^_i = table}, which lies in the orthant, any point is a convex combination of vertices plus a recession direction d>=0. sum mu^r (r=b/nu<=1) is monotone along d and concave, so its minimum is attained at a vertex, i.e. a basic feasible solution with at most rank<=p^{n+1} non-zeros. Balancing gives ||omega_i||^b=nu^{b/2} mu_i^r (AM-GM equality), so the cost does not rise. The table, and hence h, is unchanged.
(2) Compactness. K_D={width<=D, sum||omega_i||^2=1} is compact. gamma is scale-invariant. P_b>=D^{1-b/2} by the power mean.
(3) Every maximiser has h=gamma*(s) P^s >= gamma_IF(s) D^{s-nu/2}, which is bounded below on a compact J. So the argmax over K_D equals the argmax over the s-independent compact set {h>=eta_J} cap K_D.
(4) f=log gamma and d_s f=-H(w(theta,s)) are jointly continuous there; H is continuous on the simplex including zero shares. Danskin's theorem (Bonnans-Shapiro Thm 4.13 / Danskin 1967: compact parameter set, f and d_s f continuous) gives the one-sided derivatives: v'(s;+1)=max over the argmax of (-H), and v'(s;-1)=max of H. So D+ = -min H and D- = -max H, both attained because the argmax is compact.
(5) Enlarging D: for D'>=D, gamma* is the same function, so min/max H over argmax(K_{D'}) equals that over argmax(K_D). Every finite-width maximiser, padded with zero neurons, lies in some K_{D'}.
The single-feature version holds with D_1=3p^n (the class is preserved by width reduction, which keeps a subset of the original neurons) and pi>=nu^{1-s}>0.
Consequences check out: kinks are convex because D+>=D-. A cross-check against a proved fact: at n=2, pi=3^{1-s} is differentiable, so R2 forces every optimal single feature to have H=log 3 exactly. This is consistent: m equal IF neurons give efficiency m^{1-s}, maximised only at m=nu.
Minor: the Bonnans-Shapiro numbering (Prop. 4.12) may be off; immaterial.

### R3-entropy-criterion-price: CONFIRMED

L(s)=log pi(s)+(s-1)log nu.
- Continuity: Berge, using R2's compact set and P_b bounded away from 0.
- L>=0: the IF block has e=nu^{1-s}. I checked via the budget equality |beta_i|=B_p||omega_i||^nu for IF neurons and Q=nu^{1-r}|beta|^r.
- L(1)=0: min cm <= E Re Phi_1 = Re beta_1 <= sum_i|beta_i1| <= B_p sum||omega_i||^nu.
- D+L = log nu - H_min on (1,nu/2) by R2.
(a) If L(s2)>0, take s1=max{s<=s2 : L=0}. On (s1,s2) every optimal single feature has e=pi>nu^{1-s}, so the hypothesis gives D+L<=0. The monotonicity argument written out (M=L-L(a')-eps(s-a'), t=sup{M<=0}, M(t)=0, D+M(t)>=0>-eps) is correct, so L(s2)<=L(s1)=0, a contradiction. ('Dini's lemma' is a misnomer: this is the Dini-derivative monotonicity theorem.)
(b) H_min is lsc: by Berge the argmax has a closed graph and H is continuous. So g=(log nu-H_min)^+ is usc and bounded, and continuous g_k decreasing to g with L-int g_k non-increasing gives the bound by monotone convergence. Correct.
No value claim: the hypothesis must fail at p=3. Verified on the exact rational certificates (verify/v6_p3.py -> v6_p3.out):
- n=11: gamma/gamma_IF=1.054399 exactly (>1); cost shares 0.1239-0.1266; H=2.07942 vs log m=2.07944 < log nu=2.48491.
- n=13: 1.109163; H=2.30257 < log 14=2.63906.
The fixed-network integrated identity also holds (0.0529708 vs 0.0529707).

### R4-entropy-criterion-value: MINOR_GAP

(a) Correct.
- L=log gamma* - log gamma_IF is continuous. Gamma_s is continuous: it is the max over finitely many vertices of (sum v^{1/s})^{-s}.
- L>=0 by Prop. 9, and L(1)=0 by the dense corollary.
- Key step: the IF network built from a programme maximiser y*(s) is IF at every s'. By Thm 8, gamma(theta_IF;s')<=gamma_IF(s'), with equality at s'=s. So the lower right Dini derivative D_+ log gamma_IF(s) >= d/ds log gamma(theta_IF;s) = -(log nu + H(z*)): nu equal neurons per block, with block shares z_j=y_j^{1/s}.
- D+(f-g) <= D+f - D_+g gives D+L <= log nu + min H(z*) - H*_min. The monotonicity argument then closes (a). Checked: the entropies of IF networks equal log nu + H(z*), consistent with entropy_check.out.
(b) Gap. The proof says 'conclude as in R3', but R3's device (approximating a usc integrand from above by continuous g_k) does not apply here. The integrand phi=(log nu + min H(z*(sigma)) - H*_min(sigma))^+ contains +min H(z*), which is only LOWER semicontinuous. It jumps down at programme transitions: at p=5, s_c=1.8552 it is log 2 to the left and 0 at and to the right of s_c (verify/v2_programme.out). So phi need not be usc. The conclusion is still true: phi is bounded and Borel. By Vitali-Caratheodory, take an lsc G>=phi with int G <= int phi + eps. Then D_+ int G >= G, L - int G is non-increasing, and let eps->0. Easy fix, but the step as written is unjustified.
No value claim: at p=3 the hypothesis fails (v6_p3.out), as the report says.

### R5-decoupling-is-value: CONFIRMED

(=>) From (D) with spt P = features, put z_j=(P_j/P)^s. Then sum z_j^{1/s}=sum P_j/P<=1 and h(theta*)<=pi B_p P^s min_delta sum z_j kappa_j <= pi B_p P^s Gamma_s, so gamma*<=pi B_p Gamma_s. The price-programme theorem gives >=. So equality holds, z attains Gamma_s, and spt z = feature set.
(<=) Take P_j=(y*_j)^{1/s} P. Then sum P_j = P sum (y*)^r <= P, and P_j^s = y*_j P^s. So pi B_p min_delta sum P_j^s kappa_j = pi B_p Gamma_s P^s = gamma* P^s = h(theta*), and P_j>0 iff j is in spt y* = features.
Hypotheses are explicit: a global maximiser, arbitrary inputs, all b. The only import is the price-programme theorem.
No conflict at p=3: there every network is a single feature (output frequencies 0 and +-1 only), so (D) holds trivially with P_1=P and gamma*=pi B_3 kappa_min with pi>nu^{1-s}. This is not the IF value. The remark that (D) at one maximiser implies Conjecture 20' is correct and is a fair correction of the brief's lever.

### R6-slack-principle: CONFIRMED

Proof reconstructed.
- z_j=(P_j/P)^s is feasible with spt z in S(theta*).
- The price-programme theorem and (AD_eps) give min_delta sum z_j kappa_j >= Gamma_s/(1+eps) > (1-delta_min) Gamma_s, using 1+eps_tol=1/(1-delta_min).
- If S contained no optimal support, then Gamma^(S) <= (1-delta_min) Gamma_s by definition, a contradiction.
- delta_min>0 because S is finite and the restricted maximum is attained: if Gamma^(S)=Gamma_s, its maximiser would be an optimal support inside S.
- Small point: S=empty always 'contains no optimal support' and gives 1-0=1, so delta_min<=1 and eps_tol=infinity, consistent with the stated convention.
Corollary (a): Lemma 2 gives h <= sum_{j in S} |beta_j| kappa_j. Also |beta_j| <= Q_j^s (l^r dominates l^1) = B_p P_j^s, and sum P_j = sum Q_j/C_b <= P by the budget. So 1+eps=1/pi <= nu^{s-1}. Tested on 300 random arbitrary networks (verify/v9_budget_AD.py -> v9_budget_AD.out): budget ratio <=1, Lemma 2 and the AD bound hold, sum P_j/P<=1.
Corollaries (b) and (c) are correct: dilations act transitively on F; at (5,2) and (7,2), K*=F on [1,1.5].
I rigorously verified K*=F on [1,1.5] at p=5 and p=7 (verify/v2_programme.py): vertex values are non-increasing in s, and on 3000 monotone cells the uniform vertex is strictly best, with min ratios 1.27917 and 1.02024.
The tolerance table was reproduced at 40 digits for p=5,7,11,13 at all ten s values (e.g. p=5, s=1.5: eps_tol=0.279168; p=7, s=1.5: 0.0203505).
One numerical nit: 6^s<10.854 is s<1.330839, not '1.3309'. The Note's 's<=1.331' is also slightly overstated: at s=1.331, 3^{s-1} kappa_min/Gamma_s=1.00029>1. This matters only for R12's range.

### R7-transition-identity: MINOR_GAP

The stated implication is correct and essentially immediate.
- W_K is continuous: '<=K features' is closed (beta is continuous, and a union of coordinate subspaces is closed), and width reduction keeps the table.
- On the K_< side, (C) makes every maximiser have K_< features, so W_{K<}=gamma*.
- On the other side, the maximum defining W_{K<} is attained by a non-maximiser, so W_{K<}<gamma*.
- Continuity gives equality at s_c.
I recomputed the transitions at 40 digits by vertex enumeration (verify/v2_programme.out): p=5: 1.85520596; p=7: 1.54753766 and 2.32468130; p=11: 1.34539230 and 1.48952159; also p=13: 1.2850, 1.4067, 1.6776.
Problems:
(1) Range error. p=7's s_c=2.3247 is listed as 'in range for n>=3', but nu/2=2 at n=3, so it needs n>=4.
(2) 'Transitions for all p>=11' was only floating-point and only for 11<=p<=199. The certificate wording 'exceeds (p/(p-1))F^{1-s}, which bounds every full-support allocation' is false as written: near-singleton full-support allocations exist. The correct reason: programme maximisers are vertices (strict concavity along segments), and the only possible full-support vertex is v=(2/p)1, of value (p/2)F^{-s}. K*(1,p)=F is unique because the cosine matrix C satisfies C^2=(p/4)I-(1/2)J, which is invertible.
I CLOSED the gap for all p:
- interval-certified (mpmath.iv) allocations for every prime 11..1000 (verify/v3_dense_end.py with v3b_small_primes.py for p=11; ratios >=1.05, p=11: 1.089);
- for large p, an analytic lacunary bound: support {2^k}, 2^{K-1}>=p/3, so every delta has some ||2^k delta/p||>=1/3, giving Gamma_1.5>=1.5K^{-1.5} > (p/2)F^{-1.5} for all p>=892.
So every prime p>=11 does have a transition in (1,1.5).
(3) Misleading reading. By attainment, W_{K<}(s_c)=gamma*(s_c) is EQUIVALENT to 'some global maximiser at s_c has <=K_< features'. That is a count statement, and it follows from (C) near s_c by compactness. It involves neither pi nor Gamma. The headline's inference that 'the count cannot be separated from the price-programme value' at transitions does not follow from R7. What does follow, from R6, is only that the slack-principle route needs exact decoupling as s->s_c.

### R8-profile-programme: CONFIRMED

(a) The relabelled copy u(j.), w(j.) has f'(x)[c]=f(jx)[jc], so g'(x,delta)=g(jx,jdelta). Its minimum over x is m(jdelta) B_p P^s, and its cost is unchanged. Scaling each copy to cost P_j P and taking the union (gaps add; min of a sum >= sum of mins) gives gamma >= B_p min_delta sum P_j^s m_j(jdelta).
Second inequality: for single features (output at frequency 1, DC cancels), g(x,l)=Re(Phi_1 c_l). Since |cot(pi l/p)| <= cot(pi/p), Re(z c_l) >= kappa_1(l) cm(z), so m(l) >= e kappa_1(l), and kappa_1(j delta)=kappa_j(delta).
(b) Follows from R5 and (a).
(c) Exactly one j in F has j delta = +-1 (mod p). With two or more features in spt y*, some y_j>0 term is strict at every one of the finitely many delta.
(d) min_delta <= the average over +-delta, and kappa_1(l)=kappa_1(-l).
Checks:
- verify/v8_profile_union.py -> v8_profile_union.out: relabelling identity exact (error 0); union margin minus the predicted bound >= 5.5e-3 > 0 over 60 random non-IF unions at p in {5,7,11}, n in {2,3}, random b.
- verify/v4b_directional_random.out: the cone lemma m(l) >= e kappa_1(l) holds to 2.7e-13 (roundoff at equality cases) on 2000 random single features.
No value claim.

### R9-directional-n2: CONFIRMED

Re-derived. This is the Note's Thm q1n2 argument extended from delta=+-1 to a general offset l.
Modes: Phi_1-hat(chi) has the single input index k=chi-1. For chi=2*1_T the modes at +-chi are the own sign-flipped mode a (|a|=|beta_{1,T}|) and the own partner (input -3 on T) only. This holds for arbitrary inputs and any odd p, since chi != -chi.
Gap coefficient: G_l-hat(chi) = (a c_l + conj(b) conj(c_l))/2, with c_{-l}=conj(c_l). The Riesz test 1-cos(<chi,x>+psi)>=0 has mean 1 and gives m(l) <= E G_l - |G_l-hat(chi)|.
Algebra: c_l=2i sin(pi l/p) e^{-i pi l/p}. X e^{i alpha} - Y e^{-i alpha} = 2ia sin 2alpha, so |X|+|Y| >= 2|a||sin 2alpha|. Hence the summed detection is >= 2 kappa_1(l)|cos(pi l/p)||a|, and the averaged mean is kappa_1(l) Re beta_1. This gives (i).
(ii): single-frequency inputs give modes in {0,1,2}^2, and -2 = p-2 is not in {0,1,2} for p>=5, so b=0 and the detection is |a||sin(pi l/p)|.
(iii): Theorem A (Part I: |A| - R/(2 sqrt3) <= 3^{1-s}Q^s for 1<=s<=3/2, i.e. b in [2,3]) gives Re beta_1 <= 3^{1-s}Q^s + M/2, with R<=sqrt3 M. Then M(sin^2-|sin|)<=0 or kappa M(1/2-|cos|)<=0, and Q<=C_b P_b.
The profile-programme consequence follows from R8(d).
Checks:
- verify/v4_directional.py and v4b_directional_random.py: 3200 random single features. p in {5,7,11,13}, b in [2,3]; inputs Gaussian, partner-heavy, DC, sign-vector or IF-perturbed. No violation of (i), (ii), (iii), Theorem A or the cone lemma.
- Adversarial Powell maximisation of the violations at 5 (p,m,b) settings: (i) reaches equality (<=1e-15), never above; (ii) <=-0.098; Theorem A <=-1.5e-4.
The one imported ingredient, Part I's Theorem A, is used inside its hypotheses. The p>=5 assumption is used exactly where needed, and the report correctly notes that (ii) fails at p=3.

### R13-share-floor: CONFIRMED

Step 1. Optimality gives h(theta*\A) <= gamma*(P-P_A)^s; this is valid even if h(theta*\A)<=0. At a minimiser (x_R,d_R) of theta*\A, h(theta*) <= g_{theta*}(x_R,d_R) = h(theta*\A) + g_A(x_R,d_R). Convexity of t^s (s>=1) gives P^s-(P-P_A)^s >= s(P-P_A)^{s-1} P_A.
Step 2. For a class-C neuron at frequency j (no DC): |u(a)| <= sqrt(2/p)||u||, and |w(y)-w(y-d)| <= 2 sqrt(2/p)||w||. AM-GM then gives |g_i| <= 2(2/p)^{nu/2} nu^{-nu/2} ||omega||^nu = 2^{n+1} B_p ||omega||^nu, and sum ||omega_i||^nu <= P_A^s.
Step 3. With gamma* >= nu^{1-s} B_p Gamma_s, rearranging gives P_A/(P-P_A) >= psi.
Quantifiers: the frequency-piece corollary needs theta* to lie in class C (or to be a maximiser within class C), and b<nu.
Checks (verify/v5_sharefloor.py -> v5_sharefloor.out):
- piece bound on 400 random class-C pieces (p in {5,7,11}, n in {2,3}, random b): max ratio 0.883<=1;
- the decomposition inequality: 0 violations;
- psi(11,2,2)=3.363e-3, matching the report;
- on the (5,2) IF optimum, every one of the 62 neuron subsets satisfies g_A >= gamma_IF(P^s-(P-P_A)^s), min ratio 1.119;
- on the p=3 exact certificates (DC, so outside class C; only the general inequality applies, with gamma_cex standing in for gamma*): ratios 188 and 550.
No value claim.

### R12-reduction-52-72: MINOR_GAP

Reduction logic is correct.
(5,2): K*=F={1,2} on all of s in [1,1.5], rigorously; the transition is at 1.8552. So (C) is equivalent to 'no maximiser has a single feature'. Dilation by 2 maps feature set {1} to {2} and preserves gamma, so W_{2}=W_{1}. If W_{1}(s) < 3^{1-s} B_5 Gamma_s <= gamma* (price-programme theorem with pi_{2,b}=3^{1-s}), no maximiser has one feature. That is exactly W_{1} < (1+eps_tol) 3^{1-s} B_5 kappa_min with 1+eps_tol = Gamma_s/kappa_min.
(7,2): any proper feature set lies in a 2-set; the three 2-sets are dilation-equivalent (verified: equal ratios); and (1+eps_tol) Gamma^(S) <= Gamma_s for every 2-set, so the stated sufficient condition works.
Slack numbers confirmed at 40 digits (verify/v2_programme.out):
- (5,2): Gamma_s/kappa_min = 1.279168 at s=1.5. This is the minimum on the range, since Gamma_s decreases; it is 1.4383 at the dense threshold. So 27.9% is right.
- (7,2): 1.020350 at b=2, 1.081410 at b=2.2, 1.160484 at b=2.5, so '2-16%' is right on those b. Dense covers b>~2.600 (s<1.1539); on (2.5,2.6] the slack is 16-18.4%.
Gap: the hypothesis is stated on (1.331,1.5], but Thm dense holds only for s<1.330839. On [1.330839,1.331] the dense condition fails (ratio 1.00029 at s=1.331), so a sliver (b in [2.25395,2.25424]) is covered by neither part. Trivial fix: state the hypothesis on [1.33,1.5]; the slack there is about 43.8%.
Hypothesis evidence: my own adversarial penalised search (verify/v7_w1_search.py -> v7_w1_search.out, own gradient checked to 6e-10): 144 starts at b=2, 2.1, 2.2, widths 3-9, from Gaussian, IF+noise, sign-vector and DC+partner starts. Max W_{1}/(3^{1-s} B_5 kappa_min) = 1.000000 with |beta_2| ~ 1e-17, against the 1.279-1.406 needed. This is numerical evidence only; the hypothesis is not proved.
