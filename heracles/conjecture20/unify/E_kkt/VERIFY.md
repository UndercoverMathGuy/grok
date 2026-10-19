# Referee verdicts on E_kkt/REPORT.md (saved by the orchestrator)

## Overall
I re-derived all eleven claims and checked them with independent code in /home/user/grok/heracles/conjecture20/unify/E_kkt/verify/: vlib.py, v_r1.py, v_r1b.py, v_r3.py, v_r6.py, v_p3.py, with outputs in v_*.out. The code uses its own optimiser, LP duals, tensor contractions, FFTs and programme solver.

**Confirmed:** R1 (KKT structure, critical-value law, tau bounds), R4 (IF maximisers satisfy (E_S) for b>2), R6 (one constant fibre forces IF in class C), R9 (uniform case) and the R13 refutation. These are correct and general within their stated ranges.

**Minor gaps, mostly overstated side-statements:**
- R3(c): the equality is false for non-class-C 'IF' networks; there is an explicit mixed-neuron counterexample at p=7.
- R7(a): false at b=2 for shared budget-tight neurons; there is an explicit counterexample. The fix is to require frequency-pure neurons.
- R5(iii): the n=2 remark misquotes Note Thm q1n2, which bounds the price and does not exclude cheap features. The conclusion is true at every n by a short replacement argument.
- R8: the dichotomy reading needs a compactness/width-reduction argument.
- R2: R1(e),(f) do not transfer to the augmented form.
- R12(b),(c): these are informal meta-claims.

**What is settled:**
- KKT plus Lemma 2/7 plus the budget prove (E_S) for every maximiser with no cheap feature, at every b in (2,nu).
- The KKT form of (E_S) is (E_S) itself.

**What remains open:** maximisers that have a cheap feature (for b in (2,nu)), and the endpoint b=2. Neither is touched; this is the same residual the brief already identifies.

**p=3:** no value claim conflicts with the p=3 counterexamples. They are cheap (ratio about 6), not in class C (input DC), and satisfy R7(b).

## Generality audit
- **R1:** no case split. It holds for every odd p (including p=3, checked numerically), n>=2 and b in [2,nu]. The width/entropy bounds in (e) need s>1 (b<nu).
- **R3:** needs s in (1,nu/2], i.e. it excludes b=nu. Its proof uses nu>=3.
- **R4, R5(ii)/(iii), and the corollary of R6:** need b in (2,nu) strictly. The budget-equality lemma fails at b=2, where budget-tight mixed neurons exist (ratio 1.000 at b=2 versus 0.935 at b=2.2). So b=2 can only enter by continuity, and only for the value. For (E_S) a.e. this is acceptable, but it is a real endpoint exclusion.
- **R7(a):** as stated, it is false at b=2 unless neurons are frequency-pure.
- **R9:** limited by a hypothesis (the optimal allocation is uniform on a proper support). That holds only in special regimes, such as K*=1 or symmetric pairs (p=13 {2,3}; p=17 {3,5}). The proof's averaging step genuinely needs uniformity: for the non-uniform allocation at p=23, s=1.2, the stronger inequality fails. So R9 is NOT a general theorem; the general version is only numerical.
- **R12(d):** n=2 only, by design.
- **R5(iii):** its n=2-specific remark should become an every-n statement with the corrected argument.
- **Imported results:** none of the claims uses anything special to a given (p,n). The imported results are Lemma 7 / KM2, E_reduce Lemma 2(b) and Thm A, and Note Thm q1n2. I checked their statements, and each is used within its hypotheses.

## Salvage
1. **R3(c):** state the equality only for class-C IF networks (or any network whose gap is x-independent). The inequality '>= h' holds for every q and every network.
2. **R7(a):** add the hypothesis 'frequency-pure' (automatic for b>2). At b=2, rewrite it with atom cost shares v_il, which is what (b) already uses.
3. **R5(iii):** replace the q1n2 citation with this argument. A co-maximiser with a single output frequency has gamma<=pi B_p kappa_min<=D_S=gamma. So it is an optimal single feature with Gamma^{(S)}=kappa_min, hence H>=H^(1)_min=H^dec_S. This gives '>=2 output frequencies' at every n.
4. **R8:** state the tangential case for width-reduced overtaking networks (Lemma 1(a) keeps gamma(.;s)), with the compactness and continuity-of-H argument. Restrict the 'cheap' clause to s_0>1.
5. **R2:** claim R1(a)-(d) only. Define tau for the augmented form without the sqrt2 bound or the averaging bound.
6. **R12(b),(c):** relabel as remarks: 'R1(e) gives only upper bounds', and 'the cross weak-duality inequalities are implied by Lemma 2'.
7. **R9:** keep the label PROVED only with the uniformity hypothesis prominent, and say that the averaging form fails for some non-uniform allocations (p=23, s=1.2). R9b should claim only 'max off-support K > max on-support K'.

## Verdicts

### R1: CONFIRMED

I re-derived every part. (a) h is a finite min of polynomials, so h'(theta;v)=min_{a in A}<grad g_a,v>; P is C^1 for b>=2; Gordan/strict separation gives q. (b) Euler pairing <v_it,grad_it g_a>=g_{omega_i}(a) and <v_it,grad_it P>=b||omega_i||^{b-2}||v_it||^2 with sb=nu give equal factor norms and T_q(omega_i)=h w_i. (c) I checked the exponent arithmetic: t_i=nu^{b/2}(h/P)rho^{b-nu}=nu^{nu/2}(h/P)||omega||^{b-nu}=nu^{nu/2}gamma w_i^{1-s}. (d) is algebra. (e) holds; the width and entropy bounds need s>1, which is implicit in a=1/(s-1). (f) tau is a sup of linear functionals in q, so it is convex; it is translation invariant (R_alpha is orthogonal and y(x)-sum alpha=y(x-alpha)); E_x g=sum_l Re(conj(beta_l)c_l) for arbitrary networks; the b=nu budget holds for every neuron, with equality at a pure balanced neuron. No hidden case split: it holds for every odd p (including p=3), n>=2 and b in [2,nu]. Independent check (verify/v_r1.py, my own SLSQP local maxima, LP KKT measure, own contractions), at (3,2,2),(3,2,2.6),(3,3,2),(5,2,2.3),(5,2,2),(7,2,2.5): KKT residual <=2e-6; balance, T_q=h w, singular-tuple and law errors <=5e-6; identity <=7e-9; tau(x-uniform) matches the formula to 1e-16; low<=tau<=sqrt2 min(1,||q_X||) and max t/tau in [0.45,0.74]. One caveat: a numerically dying neuron (relative norm 1e-9) violates the law until it is pruned. This matches the nonzero-neuron hypothesis and is not a defect of the theorem.

### R2: MINOR_GAP

Under LICQ, N_S is locally a smooth manifold, and the Gordan argument projected onto the tangent space correctly gives sum q grad g - (sh/P)grad P in span{grad Re beta_j, grad Im beta_j}. The Euler pairings and the critical-value law then hold for T_{q,eta}, because beta_j is multilinear per neuron and sums to 0 on N_S. So R1(a)-(d) transfer. But 'R1 holds with T_q replaced by T_{q,eta}' overclaims for R1(e),(f): the augmented form is not E_q of a gap, so tau<=sqrt2 min(1,||q_X||) and the translation-averaging lower bound do not transfer. The 'without LICQ' clause is a tautology, since R1(b)-(e) use only equation (a). The label CONDITIONAL is appropriate.

### R3: MINOR_GAP

(a) is correct. Existence of mu follows from Gordan on the open face of spt y, and symmetrisation is valid because kappa is even. At a pure frequency-j neuron the gradients of beta_l (l!=j) vanish, since nu-1>=2 other factors have no l-content. grad_{v_t}Re beta_j is parallel to v_t at a positive real atom. The stationarity reduces to K_j=Gamma y_j^{r-1}. Independent check (verify/v_r3.py, own programme solver, LP dual, own gradient): KKT residual <=3.4e-15, q* on argmin, and critical values nu^{nu/2}B_p K_j to 7e-16, at (5,2,2.3),(7,2,2),(7,3,2.5),(11,2,2.2),(5,3,2),(3,3,2.5). (b) is R1(c) with gamma=W. The reading 'at larger shares and smaller critical values' holds only on average (E_w log), not neuron by neuron. (c) Its second equality, E_q g_{Theta'}=sum Re(conj beta_l K~_l(mu_q)) 'for every IF network', is FALSE with the agent's definition of IF (beta_{l,T}=0). Counterexample (v_r6.out): one mixed neuron at p=7 with inputs at frequencies 1,2 and output at 3 has every beta_l and beta_{l,T}=0, yet g ranges over [-1.54,1.90], so a point-mass q gives 1.90, not 0. The equality needs class C (x-independent g). The inequality '>= h' is trivially true for every q and every network, and the intended application (a class-C decoupled Theta) is fine.

### R4: CONFIRMED

Each link of the chain is valid for arbitrary networks in N_S:
- Lemma 2 restricted to S.
- Re<=|.|.
- Lemma 7 for IF features, which holds for arbitrary inputs via Paper Lemma 6 (|beta_l|<=nu^{1-s}Q_l^s is equivalent to Q_l>=nu^{1-r}|beta_l|^r).
- Programme homogeneity.
- Single-neuron budget.
- pi>=nu^{1-s} (KM2 block).
- W_S>=D_S (E_reduce Lemma 2(b), which I checked).

Consequences of the equalities:
- (i) Budget-equality lemma: I rebuilt the Hoelder/Parseval/AM-GM proof. For b>2, ||A||_b=||A||_2 forces a single entry, and differing entries give zero.
- (iv) The rescaling argument is correct; it needs Gamma^{(S)}>0, which holds for nonempty S because a single frequency covers every offset at prime p.
- (v) The KM2 equality clause requires r<1, i.e. s>1, which is satisfied.

Generality: R4 needs b in (2,nu). It does not hold at b=2 (budget-tight mixed neurons exist there; v_r6.out: ratio 1.000000 at b=2 versus 0.935 at b=2.2). That is acceptable only because (E_S) is needed a.e.

Consistency with p=3: R4 is an implication from the existence of an IF maximiser. At p=3, n>=11, b near 2 the counterexamples beat the IF value, so no IF maximiser exists and R4 is vacuous. There is no contradiction.

### R5: MINOR_GAP

(i) is correct at every b. The choice of sign between +-delta works by Lemma 2 averaging. The Riesz test min g<=E g-|ghat(chi)| is valid. The R6 identity gives |ghat|=|beta_{l,T}|sqrt(kappa_l/2) for every l in F. No-cheapness is used only on S. Independent check (v_r6.py, own construction, including b=2, b=nu=4, and non-feature frequencies carrying cancelled atoms but nonzero beta_{l,T}): over 274 no-cheap class-C samples, max(gamma-bound)/gIF <=-0.27, with det/gIF up to 0.37. (ii) is correct for b in (2,nu): the equality chain forces budget-tightness, hence class C, hence det=0. (iii) The main sentence is correct and needs only D^+L<=H^dec-H*. However the n=2 justification misquotes Note Thm q1n2. That theorem bounds the PRICE (cone margin <=3^{1-s}Q^s); it does not say single features are never cheap. A single pure neuron is cheap: |beta|=Q^s>3^{1-s}Q^s. The conclusion is still true, at EVERY n, by a different argument. A single-output-frequency co-maximiser has gamma<=pi B_p kappa_min<=D_S=gamma. So it is an optimal single feature with Gamma^{(S)}=kappa_min, hence H>=H^(1)_min=H^dec_S. That replacement argument should be written in.

### R6: CONFIRMED

I re-derived the Fourier computation: ghat_delta(chi)=p^{-nu}sum_k what(k)c_k(delta)prod_t uhat_t(chi_t-k). For chi=2l1_T and a pure frequency-j neuron, only k=j=l survives, because l+j!=0 mod p for l,j in F. This gives (1/2)conj(beta_{l,T})c_l(delta), with c_l(delta)!=0 for delta!=0. Independent check (v_r6.py, own FFT, 40 random class-C networks at p in {5,7,11}, n in {2,3}): error 2.1e-16. The corollary's (=>) direction uses the lemma and R4 (b in (2,nu)); its (<=) direction uses R3(a). KM2 forces equal atoms beta/nu, so the atoms are positive real and R3(a) applies. Generality: the lemma holds for every p, n and b. The corollary is restricted to b in (2,nu), as stated.

### R7: MINOR_GAP

(b) is correct for every network and every b: Lemma 2, |beta_l|<=sum|beta_il|, C_b^s=B_p, programme homogeneity and the Renyi rewriting. It holds on the p=3 counterexamples (v_p3.out: gamma/bound=0.34 at n=11 and 0.42 at n=13). (c) is correct: sum rho^nu T_q(u^)=E_q g=h, the clean part is E_{p^{-n}mu}g, and the detection bound follows from Riesz plus R6. (a) is FALSE as stated at b=2. The proof step 'budget-tight => |beta_il|=B_p a_i' requires frequency-pure neurons, which is automatic only for b>2. Counterexample (v_r6.out): at p=7, n=2, b=2, three budget-tight (ratio 1.0) aligned neurons each spread over frequencies 1 and 2 with different profiles. Feature 1 has cheapness ratio 1.242, while exp((s-1)(log nu-H_s(w))) gives 1.057. Fix: add 'frequency-pure neurons' (or b>2). The 'Reading' paragraph is interpretation, not proof.

### R8: MINOR_GAP

The displayed inequality is correct. I checked E_reduce Lemma 2(b): psi(sigma)=e(theta_1;sigma)B_p m(A)/||A||_{1/sigma}<=D_S, it touches D_S at s_0, and it has the Gibbs form const-sigma log sum(A_l a_i)^{1/sigma}. So (log psi)'=-H(Theta_0) and (log psi)''=-Var/sigma, and Taylor with integral remainder is exact. Delta>=0 because L(s_0)=0. The dichotomy sentence is not proved as stated. In the tangential case, 'overtaking networks have H(.;s_0)->H^dec and Delta=o(s-s_0)' needs a compactness argument: width-reduce the overtaking networks (Lemma 1(a) keeps gamma(.;s)), take limits, which are maximisers at s_0 with H>=H*=H^dec, and use continuity of H. That works for width-bounded representatives but not for arbitrary networks. Also, the 'jump => cheap feature by R5(iii)' clause is outside R5's range at s_0=1.

### R9: CONFIRMED

I re-derived each step:
- sum_{l in F}kappa_l(delta)=p/2.
- Under uniformity, K_j=k^{-1}min_delta sum_S kappa_j.
- The mean over delta is p/(p-1).
- Strictness: if sum_{+-S}rho^{j delta} is constant c off 0, comparing Fourier transforms at m not in +-S and not 0 forces c=2k, and m=0 then gives 2kp=0. This requires S!=F.
- The averaging inequality is equivalent to K_S<p/(p-1).
- 'Attained only off S' follows from the equality analysis at b=nu (pure neuron, all mass on argmax K).

Independent check (v_r3.py) for p in {7,...,23}: sum_F K=p/2 exactly, and the uniform cases (p=13 s>=1.8; p=17 s in {2.2,3}; p=7 s=3, K*=1) satisfy the claim.

Generality: the theorem only covers uniform optimal allocations, which occur only in special (p,s) regimes. The uniform hypothesis is genuinely needed. For a non-uniform allocation (p=23, s=1.2, k=8), the stronger 'mean off-support K > every on-support K' FAILS (on-support 1.138 > off-support mean 1.133, and K_j exceeds p/(p-1)). The weaker NUMERICAL R9b (max off > max on) held in all my cases.

### R12: MINOR_GAP

(a) is correct. Networks with no cheap feature satisfy gamma<=nu^{1-s}B_p Gamma^{(S)}<=D_S for ALL networks in N_S (no class-C assumption needed). I also checked that the restricted (EW_S) implies W_S=D_S by a modified Dini argument: on {L>0} every maximiser has H>=H^dec, so D^+L<=0 there, and L vanishes at the left endpoints. (d) is correct given the imports: Note Thm q1n2 really gives pi_{2,b}=3^{1-s}, at every p including p=3. The p=3 counterexamples are cheap (cheapness ratio 6.0 at n=11, 6.2 at n=13; v_p3.out), so nothing contradicts the no-cheap value bound. (b) Its first sentence is a correct tautology. Its second, 'the remaining KKT relations bound H only from above', is an unproved meta-claim: it is shown only that R1(e) gives upper bounds. (c) 'Co-maximality adds no inequality beyond Lemma 2' likewise overgeneralises R3(c), which covers only the two cross weak-duality inequalities, and those are trivial (average >= min). Both should be reworded as observations.

### R13: CONFIRMED

The refutation is valid and in fact stronger than stated. At (p,n,b)=(5,3,2) the single-feature KM2 block is a KKT point by R3(a); verify/v_r3.py checked stationarity to 3e-15. Every KKT measure q has mu_q on the binding offsets {+-1}, so tau(q)>=tau(p^{-n}mu_q)>=nu^{nu/2}B_p|K~_2(mu_q)|>=nu^{nu/2}B_p kappa_2(1)=nu^{nu/2}B_p*1.809. The critical value t=nu^{nu/2}B_p*0.691 does not depend on q. So t<tau for EVERY KKT measure, not just q*. Numerically K=[0.691,1.809] and tau(q*)=3.618e-2, matching the formula to 4e-16. My independent local maxima also give max t/tau in [0.45,0.74].
