# Referee verdicts on gibbs/REPORT_R2.md (saved by the orchestrator)

## Overall
I could not break any of the ten claims; two have minor gaps.

- **Confirmed (8):**
  - R1, the sawtooth refutation of PB2: re-verified exactly and rebuilt from actual class-C neurons.
  - R2 and R3: full re-derivations; R3 also passes an exhaustive exact check at p=3.
  - R4 and R5: re-derived, and checked on class-C networks built from neurons with my own Fourier conventions.
  - R6, the computer-assisted sharp L2 payment: every reduction re-derived, including the box path argument and the infeasible P > 3 boxes. The agent's certificate re-runs CERTIFIED, and my independent interval certificate (different grid, roots and lambda) passes with 0 failures. Adversarial atom searches find nothing above 1e-14.
  - R9: re-derived, with LP constants reproduced and brute-force checks at n=3 and n=4.
  - R10: re-derived and checked on general networks with DC and mixed frequencies.
- **Minor gaps (2):**
  - R7: the closure range is right (p <= 31 at b=2, and the whole table), but '(C) for all networks iff every global maximiser is IF' should be 'if'. SPLIT R9 does not give the converse.
  - R8: correct mathematics, but 'about 3x larger buffer' holds only near b=2 and large p.
- **p=3 consistency:** the value claims R5 and R7 cover class C at n=2, including p=3. The known p=3 counterexamples use input DC and n >= 11, so there is no conflict. Adversarial class-C maximisation at p=3, 5, 7, 11 never exceeds the IF value: the best is 0.99999989 at p=3.

My verification scripts are in /home/user/grok/heracles/conjecture20/unify/gibbs/verify/: v_pbd.py, v_r2.py, netlib.py, v_r4.py, v_r6_direct.py, v_r6_cert.py, v_r6_chain.py, v_r6_chain_hp.py, v_r7.py, v_r7_p3.py, v_r8.py, v_r9.py, v_r10.py.

## Salvage
- **R7:** replace 'iff' with: 'for p <= 31 (b=2), and the larger ranges in the table for larger b, (C) for all networks follows once every global maximiser is IF (equivalently (MM_opt), by SPLIT R9(a,b) with (IF_C)).' Everything else in R7 stands.
- **R8(d):** the buffer improves by the factor 9 sqrt3/(17-8s) * sqrt((p-1)/(p+1)) over the Note's. At b=2 that is 3.118 sqrt((p-1)/(p+1)): 2.2x at p=3, 2.55x at p=5, about 3.1x for large p. It falls to about sqrt3 * sqrt((p-1)/(p+1)) as b -> 3.
- **R4:** 'the constant is attained' should read 'the constant (p-1)/(p+1) in A_delta A'_delta >= kappa_p E_delta is attained by a class-C network'. Sharpness of E <= ((p+1)/(p-1)) mbar^2 is not shown.
- **R6:** the agent's script converts the root-enclosure endpoints with round-to-nearest. A fully directed-rounding version (e.g. verify/v_r6_cert.py, which caps P at 3 and uses trigonometric interval roots) should be the archived certificate.

## Verdicts

### R1: CONFIRMED

I re-derived the counterexample. For the sawtooth on Pi_1 the residue sums are at least p-1, so the depth is (p-1)/2. On Pi_2 they are at most 2p-2, so the height is (p-1)/2. With V=(p^2-1)/4 the ratio is (p-1)/(p+1).

Exact checks (Fractions, verify/v_pbd.py): the ratio is 1/2, 2/3, 3/4, 5/6, 6/7, 8/9 for p=3..17, and (1,2) is the worst plane pair over all (c,c').

Network level (verify/v_r4.py, own neuron code): least squares over 8 class-C neurons per frequency realises D1=D2=r, D12(w)=r(1-w) exactly (residual 1e-14). Computing from the actual logit table gives AA'/E=(p-1)/(p+1) for p=3..13.

The PB2 <=> class-C equivalence holds: the profiles are arbitrary mean-zero functions, and translations and dilations map (c,c') to (0,delta).

joint_l2.py minimises over all free a_{l,T} with no restriction, so its 1.0003 at p=11 is an optimiser failure. The true minimum is 5/6: the R4 lower bound plus the sawtooth at d=1.

### R2: CONFIRMED

I rebuilt the whole proof.
- Y is uniform on {sum y = p-2 mod p}. Pairs are independent and uniform, and sum Y is in {p-2, 2p-2}.
- Bottom constraint: Cauchy-Davenport with sum|L_i| >= p+2, so H >= -m.
- Top constraint: H <= M on level 2p-2. On level p-2, H(y) <= M + g_i(y_i+1) for every i.
- Expansion: E[(H+m)(M+R-H)] >= 0. This needs E H = 0 and E H^2 = V, both true by pairwise independence.
- Bounds on R (set g(p):=0): E sqrt(g_i) <= sqrt(osc_i/p), E R^2 <= E g_i E g_j <= eps^2, then Cauchy-Schwarz.
- Corollary: osc^2 <= 2pV_i and min-pair <= average-pair give eps^2 <= 2V/(3p); then AM-GM.

Checks: the agent's check_planebd.py re-run gives slacks 0.0855 and 0.146. My own check (verify/v_r2.py, p=3..11, all plane pairs, 5 families) gives slack >= 0, with exact 0 attained at p=3.

### R3: CONFIRMED

Every step re-derived.
- Step 1 (ranks): |U_i(y)| >= y+1 and |L_i(y)| >= p-y.
- Step 2: three-set Cauchy-Davenport. sum(y_i+1) >= p+2 when sum y >= p-1, and 3p - sum y >= p+2 when sum y <= 2p-2. Both are plane-independent, so they hold for ALL c, c' (c = c' included).
- Step 3: |S| = 2C(p+1,2) via y -> (p-1)-y. The pair counts are 1 off the antidiagonal and 2 on it, so the marginals are exactly 1/p.
- Step 4: (H+m)(M-H) >= 0 on S with E_nu H = 0. The product-measure part of E A_i A_j vanishes because sum A_i = 0.
- Step 5: Cauchy-Schwarz, plus sum_{i<j} sqrt(V_i V_j) <= V.

No hypothesis beyond p prime and mean zero is used.

Sharpness: the sorted sawtooth gives A(a)A(p-1-a) = -A(a)^2, so all three bounds are equalities.

Checks (verify/v_pbd.py):
- Exhaustive at p=3 over all 46656 integer triples with values in [-3,3]: 0 violations of mM >= first bound (exact Fractions); min mM/V = 1/2.
- 30000 integer samples at p=5 and 20000 at p=7: 0 violations.
- Nelder-Mead on the exact ratio: minima 0.5000 (p=3), 0.751 and 1.437 (p=5,7). All are at or above the bound (p-1)/(p+1).

### R4: CONFIRMED

Re-derived.
- Class C: Phi_l involves only frequency-l neurons, so g - E g = sum_T D_T(2 s_T) with D_T = phi_T - phi_T(.-delta).
- Z_{-delta} = -sum_T D_T(2 s_T + delta), so A = -min_{Pi_0} F and A' = max_{Pi_delta} F with f3(z) = D12(-z).
- Var D_T = sum_l |a_{l,T}|^2 |c_l|^2/2 = sum_l |beta_{l,T}|^2 kappa_l, using that l in F are distinct and l != -l'.
- m_delta >= A_delta >= 0, and AM-GM. R3 then applies verbatim, for every odd prime including p=3.

Independent check (verify/v_r4.py): neuron-level class-C networks with my own Fourier convention, p=3,5,7,11,13, all delta.
- E from sum |beta_{l,T}|^2 kappa matches the profile variances to 7e-15.
- min (mm' - AA')/E >= 0.03, min (AA' - b2)/E >= 0.60, b2 >= b3 to rounding.

Wording caveat: 'the constant is attained' is shown only for the A A' form (R1 sawtooth network). Equality in E <= ((p+1)/(p-1)) mbar^2 would also need h = min_x g(x,+-delta), which was not shown.

### R5: CONFIRMED

Each step checked.
- Step 1: averaging over +-delta gives h = sum Re beta_l kappa_l - mbar. This holds for any network, since E_x Phi_l = conj beta_l.
- Steps 2-3: Cauchy-Schwarz over L', then R4 gives sqrt(E) <= sqrt((p+1)/(p-1)) mbar.
- Step 4: Gamma_s with y_l = Q_l^s/(sum Q)^s, then the single-neuron budget. A class-C neuron has one atom, and (C_b P)^s = B_p ||theta||^3.
- Equality: either equality (h = W, or gamma = IF) forces mbar*bracket = 0, so mbar = 0, so E = 0. kappa_l(delta*) > 0 since p is prime, so every beta_{l,T} = 0.
- Theorem A's sqrt(E_l/2)/sqrt6 equals R_l/(2 sqrt3), so the threshold 12(p-1)/(p+1) vs the Note's 4 is right.

Own check (verify/v_r7.py, p=11,13,17, b=2 and 2.5): the chain RHS - h >= 4e-3, W - h >= 0.039 on nets meeting the condition, and the payment holds per feature to 3e-15.

### R6: CONFIRMED

Every reduction step re-derived.
- Twisted Gram matrix and characteristic polynomial: t^3 - Pt - q with q = 2Re(xy conj z), det >= 0 gives q >= P-1, and |q| <= 2bc'delta.
- Rotfel'd for t^r (r in [2/3,1]) with ||u_i||^2 = 3.
- Weighted AM-GM gives M_{1/s}(nu) >= x^Lambda.
- Gain step with C' = c lambda_3 and lambda_1:lambda_2 = b:c'.
- Divided-difference monotonicity: g'' < 0 and G'' = (2+t)/(1+t)^2 > 0. Schur-convexity along q = P-1.
- Valid spectra force P <= 3.
- Concavity in x: G_x is convex (norm of an affine map minus a linear term), x^Lambda is concave, and G_1 = 1.

Path argument behind the box bound Lambda(P_hi, max(q_lo, P_hi-1)):
- Case q* >= P_hi - 1: raise P at fixed q*, then lower q.
- Case q* < P_hi - 1: raise P to q*+1 (q* >= -1/4 always), then follow the line q = P-1.
- For boxes with P_hi > 3 the script evaluates at an infeasible point. There the largest nu exceeds 3, so the computed Lambda exceeds 1, which is >= the true Lambda. This is conservative.

Regions:
- Region A: Taylor series via Newton sums (sum t^3 = 3q, sum t^4 = 2P^2), tail bound, and lambda = delta/(c+2beta); checked by hand.
- Region B: q* = P-1, delta >= 1-sigma, beta^2 >= 2-3sigma+sigma^2, the (3-sigma)ln(3-sigma) bound, and 0.4203 > 0.3932; checked by hand.

Re-ran l2pay_cert.py: CERTIFIED at N=8 and 16.

Independent re-certification (verify/v_r6_cert.py), with a decimal grid, P capped at 3, trigonometric interval root enclosures and a different lambda choice: N=10 gives 226 boxes ok, 0 fail, margin 1.1e-4; N=23 gives 441 ok, 0 fail. Float grids of regions A and B are both >= 0.

Adversarial search over atoms (verify/v_r6_direct.py; N<=6, 400 restarts, s=1.5,1.35,1.2,1.08): max (|A| - th Q^s - cR)/L <= 1.7e-14, attained only at single atoms.

Random chain check (verify/v_r6_chain.py, 20000 systems): every link >= 0. The only negative values, -4e-8, are single atoms evaluated on a lambda grid that misses 1/sqrt3; re-checked at 60 digits.

Tiny caveat: the agent's script converts the root-enclosure endpoints and lambda to mpf with round-to-nearest, about 1e-40 error. This is immaterial at margin 3e-4, and my certificate does not depend on it.

### R7: MINOR_GAP

Main claim (IF_C)+(C_C) for class C at n=2: confirmed.
- It follows from R5 with c(s) from R6 and sum_{l in F} kappa_l = p/2.
- My recomputed table (verify/v_r7.py) matches every row: p <= 31 (b=2), 37 (2.1), 53 (2.2), 61 (2.25), 101 (2.4), 151 (2.5), 661 (2.75); Theorem A alone gives p <= 19. c(s) is increasing in s, so b=2 is the worst case.
- The |L'| <= 5 and |L'| <= 7 (p >= 23) corollaries hold.
- Value check at p=3: R7 covers p=3 class C, n=2. The known counterexamples have input DC and n >= 11, so they are outside class C; no conflict.
- Adversarial gamma maximisation in class C (verify/v_r7_p3.py, v_r7.py): best gamma/gamma_IF is 0.99999989 (p=3), 0.9999927 (p=5), 0.98 (p=7) and 0.72-0.82 (p=11), never above 1.

Gap: the sentence '(C) for ALL networks holds iff every global maximiser is IF' overstates SPLIT R9.
- R9(a,b) with (IF_C) gives: every global maximiser IF <=> (MM_opt) => (C).
- The converse, (C) => every maximiser is IF, is not established anywhere.
- So the correct extension of SPLIT R9(d) is 'if', not 'iff'. SPLIT R9 is itself an unrefereed round-1 citation.

### R8: MINOR_GAP

Mathematics confirmed.
- (a) Verified the explicit swap maps x<->Y (|B|<->|D|) and x<->Z (|C|<->|D|). They keep beta_i, Q and L. Deviation 3.6e-15 (verify/v_r8.py).
- (b) Follows from P12 plus (a), then R5 with c_l = c12. 1/c12^2 = 29.70 and p(p+1)/(2(p-1)) gives p <= 53 (59 gives 30.5); c12 decreases with b. Adversarial check on C=D=0 systems: max (|A| - c12|B| - th Q^s)/L = -4e-9 (two-atom equality).
- (c) Re-derived the Note's quadpay(i) intermediate: Rotfel'd, Jensen, e^{-y} >= 1-y, nu ln nu <= (nu-1)+(nu-1)^2 (true on [0,inf)), |D'| <= |D| + sqrt(2L(L-|A|)), and eta chosen so the L-|A| terms cancel. The S3 minimum gives (1+eta/3)R^2, and Ksym = 5/9 at s=3/2. Adversarial ratio <= 0.78 (s=1.1..1.5).
- (d) Correct, with R4 in place of the factor 3.

Gap: 'about 3x larger than the Note's buffer' holds only near b=2 and large p. The exact ratio is 9 sqrt3/(17-8s) * sqrt((p-1)/(p+1)): 2.2x at p=3 and 2.55x at p=5 (b=2), falling to about sqrt3 * sqrt((p-1)/(p+1)) as b -> 3.

### R9: CONFIRMED

Re-derived.
- The slice decomposition G_S(w) = sum_U D_{S u U}(w + s_U(x'')) uses all U including the empty set, and is mean zero.
- R3 Steps 1-2 hold for every plane pair, so Psi + H >= -A and Psi_d + H <= A' on every slice for y in spt nu.
- Cross terms vanish: E_y H^{x''} = 0 for each slice, E Psi = E Psi_d = 0, and distinct non-empty s_U are jointly uniform.
- Hence AA' >= E H^2 + sum_{U avoids} C_U(d), with E H^2 >= kappa E_{x''} V^{x''} = kappa sum_{T meets} V_T.
- Per-T bound: conditional expectation plus Bhatia-Davis.
- The class-C formula C_T = V_T - sum |beta|^2 kappa^2 is correct.

My own LP (verify/v_r9.py) reproduces the constants:
- kappa = 1: 1, 1/3, 0.1818, 0.1000, 0.0476, 0.0286 for n = 2..7;
- kappa = 5/6: 0.2941; kappa = 2/3: 0.25 (n=3).
This equals kappa/(2+kappa) at n=3.

Brute force at n=3 (p=3,5,7) and n=4 (p=5), all d, random, sawtooth and spike profiles: min (AA' - best slicing bound)/E >= 0.34.

### R10: CONFIRMED

Re-derived.
- The split keeps beta_l, beta_{l,T}, every atom (so Q_l, a_l and the cheap set) and E_x g, and does not raise the cost.
- The all-DC part is constant in c, so g = g_split + g_R.
- min(g_split + g_R) <= min g_split + max g_R gives m_delta(theta) >= A_delta(split) - max g_R.
- AM-GM and R4 on the class-C split give mbar >= sqrt(kappa_p E) - rho.
- R5 Steps 1-4 hold for any network, and R6 applies to arbitrary-input atoms (Paper Lemma 6).

Own split code (verify/v_r10.py) on general networks with DC and mixed frequencies, plus class-C cores with small mixed/DC perturbations (p=5,7, b=2 and 2.5):
- coefficient and Q differences <= 7.5e-15; cost(split) <= cost;
- min[m_d - (A_d(split) - max g_R)] >= 1e-9; min[mbar - (sqrt(kE) - rho)] >= 0.025;
- chain >= 0.03; W - h >= 0.038 on the 50/150 nets meeting the condition.

The result is weak (perturbative), as the claim itself says. The obstruction paragraph is commentary.
