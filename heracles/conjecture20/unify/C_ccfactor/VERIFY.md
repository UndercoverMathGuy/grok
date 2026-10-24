# Referee verdicts on C_ccfactor/REPORT.md (saved by the orchestrator)

## Overall
I checked 16 claims. 12 hold as stated (including both refutations), 4 have minor gaps, and CC15 is false.

The accounting and detection results are correct and general: CC1, CC2, CC5 and CC8-CC11. I re-derived each line and confirmed it numerically with independent code in /home/user/grok/heracles/conjecture20/unify/C_ccfactor/verify/ (vlib.py, v1-v9).

The reduction claims CC3, CC4 and CC6 have correct algebra but one shared scoping gap. The step CD => RD_S (via M(P^s) <= Gamma^{(S)} P^s) needs every frequency that carries neurons to lie in S.
- BRIEF_CORE defines N_S by clean support, and class C allows 'ghost' frequencies: neurons whose atoms cancel, so beta_k = 0 but beta_{k,T} != 0 and P_k > 0.
- At (5,2,2), S = {1}, I exhibit M(P^s)/(Gamma^{(S)} P^s) = 1.127, so CD does not imply RD_S there.
- These ghosts are exactly the open 'foreign zero-mean output field' case in E_SYNTHESIS, so the CD route silently omits it.
- A quick search at (5,2,2) and (7,2,2) found that ghosts do not help, but that is evidence only.

CC7 is correct; one parenthetical reason is misstated ('Gamma_s - kappa_min > 0' fails when K* = 1, e.g. p = 5 at s >= 1.855).

Both refutations hold:
- CC12: the certified counterexample reproduces, with the argmin rule robust to rounding.
- CC13: the SPLIT violators and the E_kkt local maximisers reproduce.

CC14 reproduces and also holds at (5,2,2) in my own search.

CC15 is FALSE. It was computed at points that are not programme maximisers in 65 of 144 cases (prog.solve with 4 restarts is a local heuristic). At exact maximisers (p <= 19) there are all-acute binding offsets, and one nonnegative Q jointly detects every support feature in many p >= 11 cases.

CC16 is consistent, apart from a misattribution to CC10 for the Legendre design; I verified the stronger statement directly.

## Generality audit
Fully uniform, with no case split or parameter limit:
- CC1, CC2 (boxed identity), CC5 (in fact for every network, not only class C), CC8, CC9, CC10 and CC11 (main inequality) hold for every odd p, every n >= 2 and every b in [2, nu]. p = 3 is included; there CC9 and CC10 are vacuous, as they must be.
- CC7 holds for every p >= 5 and every n and b, and is vacuous at p = 3. It uses only kappa_1(2) > kappa_1(1) and the cone lemma.

Hidden special-casing found:
1. Ghost frequencies. CD => RD_S (CC3(ii), CC4's equality and equivalences, CC6's last arrow) and Pi_feat <= Gamma^{(S)} (CC2) silently assume that the cost support of the class-C network lies in S. This is a domain restriction, not a parameter limit, but it excludes exactly the foreign-zero-mean-field case.
2. CC2's IDENTITY.md translation of d_prog and d_int needs sum Re beta > 0. This is automatic for h > 0 but fails for some h < 0 networks, which IDENTITY.md covers.
3. CC1's W5 bullet holds with equality only at b = 2 with no DC; for b > 2 it is an inequality.
4. CC3(iv)'s equality case needs an IF optimal feature, which is known only at n = 2 or where pi = nu^{1-s}. The inequality Pi_within >= pi is general. This is stated in the claim.
5. CC6's rho needs at least one non-IF feature.

The numerical claims are limited in range: CC14 to p <= 13 and n <= 3; CC15 to p <= 61 and was computed with a non-global solver. The refutations (CC12 at (5,2,2); CC13 at (5/7/11, 2, 2)) are single-parameter counterexamples, which is legitimate for a REFUTED label. No PROVED claim rides a parameter to a limit.

## Salvage
1. CC3, CC4, CC6: either define N^C_S by cost support (every frequency carrying neurons lies in S), or state CD with allocations restricted to S. Then add a separate statement 'ghost frequencies never help in class C': if theta in N_S has neurons at k not in S with beta_k = 0, then gamma(theta) <= sup over ghost-free networks in N_S. Label it CONJECTURE, with my numerics at (5,2,2) and (7,2,2) for S = {1} (best 0.99999; ghosts shrink to zero). This is the class-C form of the open foreign-zero-mean-field problem and must be proved for the CD route to give RD_S.
2. CC15: recompute with exact programme maximisers (vertex enumeration for p <= 19; many more restarts or exhaustive small-support search beyond). At true maximisers with p >= 11, a single nonnegative Q detects every support feature on the binding set in many cases, e.g. the pair (1, -3) at (11, s = 2). The copositive route is therefore NOT blocked there, which is worth pursuing. The blocking statement survives where the support is all of F (s near 1, by CC10) and at p = 5 and 7 where K* >= 2.
3. CC7: replace the slack reason by '(1+eps)^s - 1 ~ s eps is much larger than eps^s', i.e. the Pi_feat slack.
4. CC2: add the condition sum Re beta > 0 to the IDENTITY.md translation.
5. CC11: change 'k in S' to 'any frequency carrying sign-flipped content', and either prove or drop the 'reproduces the few-cheap-features decoupling' aside.
6. CC16(5): cite the direct computation (all off-diagonals of sum V_l Psi^(l) >= 0 for the Legendre design) instead of CC10.

## Verdicts

### CC1: CONFIRMED

I re-derived every line. P_l^s = A_l K_l^{s-1}, and sum_i m_i^r = sum_l (M_l K_l^{s-1})^r because (s-1)r = 1-r. With r/(1-r) = 1/(s-1) this gives H_r(m) = H_r(zhat) + (1/(s-1)) log Z. At alpha = r, Arimoto's formula has inner factor (sum (m^(l))^r)^{1/r} = K_l^{s-1}. The bounds are those of a power mean of order s-1 > 0, and both cost forms are the same algebra. The identity is pure combinatorics of a positive measure split into blocks, so class C is not even needed.

Independent check (verify/v1_identities.py, own library, 144 random class-C networks):
- p in {3,5,7,11}, n in {2,3,4}, b = 2, random, and nu - 1e-3.
- Chain rule error 1.5e-12 (amplified only at s - 1 ~ 3e-4); Arimoto 8e-13; cost forms 9e-16; bounds never violated.

The W5 bullet is only a b = 2, no-DC statement; for b > 2 SPLIT gives only an inequality. It is labelled as such.

### CC2: MINOR_GAP

The main identity is correct. B_p M(e o zhat) = M(Re beta)/sum P^s by 1-homogeneity of M, and K_r(zhat)^{1-s} = sum P^s / P^s by CC1. e_l follows from Re beta_l = B_p A_l (1 - E sigma), valid in class C because atoms vanish off the neuron's frequency. Checked (v1): factorisation 1.3e-14; e_l 2.3e-12; Pi_feat Pi_within split 1.3e-14; 1 - E sigma translation 7e-16.

Caveats:
(a) Pi_feat <= Gamma^{(S)} needs zhat supported on S. For networks in N_S (clean support) carrying 'ghost' frequencies (neurons whose atoms cancel, beta_k = 0, P_k > 0), only Pi_feat <= Gamma^{(supp P)} holds; see CC3.
(b) The IDENTITY.md translations use M(Re beta) = (sum Re beta) M(yhat), which needs sum Re beta > 0. This is automatic for h > 0, since M(Re beta) <= Gamma_1 sum Re beta. It FAILS for h < 0 networks with sum Re beta < 0 (57 such random cases in v1). IDENTITY.md advertises 'any h', and the parenthetical 'it holds for any h' covers only the boxed identity.
(c) h = 0 = M(Re beta) leaves delta_int undefined.
All three are scoping issues and the boxed identity stands.

### CC3: MINOR_GAP

(i), (iii) and (iv) are correct.
- (iii): h <= M(Re beta) <= M(pi B P^s) by Lemma 2 and monotonicity of M. v1 found no violation.
- (iv): the cone lemma gives min_x g(x,delta) >= kappa_1(delta) min cm, because |cot(pi l delta/p)| <= cot(pi/p); relabelling turns kappa_1(l delta) into kappa_l(delta); gaps of a union add. I checked it for NON-IF features too: copies at random allocations have Pi_within >= the feature's own cone price, min excess 7.6e-14 (v6 (b)).

Gap in (ii), 'CD => RD at theta' for theta in N_S. The step M(P^s) <= Gamma^{(S)} P^s (take y = P^s) needs supp(P) contained in S. BRIEF_CORE defines N_S by CLEAN support, and class C admits ghost frequencies k not in S: neurons at k whose atoms cancel exactly, so beta_k = 0 while beta_{k,T} != 0 and P_k > 0. Then M(P^s)/(Gamma^{(S)} P^s) can exceed 1. v2_ghost.py gives 1.127 at (5,2,2), S = {1}. So CD with the canonical allocation does NOT imply RD_S on all of N^C_S.

The statement is true once N^C_S is defined by cost support. The excluded case is not harmless for the programme: class-C ghosts are exactly the 'foreign zero-mean output fields' that E_SYNTHESIS lists as open (item 2). My ghost search at (5,2,2) and (7,2,2), S = {1}, found no gain (best 0.99999; ghosts shrink to 0; v2b), but that is evidence only.

Side note: 'always at n = 2' and the IF equality case hold only where an IF optimal feature exists (n = 2 by Note Thm q1n2, or pi = nu^{1-s}). The claim says so.

### CC4: MINOR_GAP

The algebra is correct. omega_S >= pi^C holds by taking decoupled copies of an optimal class-C feature with P_l proportional to zhat_l^{1/s}; M(P^s) > 0 always. Attainment of pi^C holds (GLOBAL R2 width reduction keeps class C and the set is compact) but is not even needed. At a vertex, M(P^s) = kappa_min P_l^s, so omega = pi^C. The equivalences RD <=> omega <= pi Gamma/Pi_feat and CD <=> omega = pi^C follow. The any-allocation identity holds for arbitrary mixed networks: v1 error 6e-16. Taking Ptilde_l = P_b (y*_l)^{1/s} gives M(Ptilde^s) = P_b^s Gamma, so RD <=> CD for some allocation.

Same scoping gap as CC3. With N^C_S defined by clean support, ghost networks have zhat(theta) outside Delta_S, so the formula W^C_S = B_p sup_{Delta_S} Pi_feat omega_S is only '>='. Equality, and hence RD^C_S <=> the omega condition, holds on networks whose cost support lies in S. The 'some allocation' direction needs Ptilde supported on S.

### CC5: CONFIRMED

Correct, and even simpler than stated. At any Clarke/KKT point, sum q grad g = s(h/P) grad P. Pairing with omega_i and using Euler's identity for the multilinear table gives <omega_i, grad_{omega_i} E_q g> = nu E_q g(T_i). Since <omega_i, grad P> = b ||omega_i||^b and s b = nu, E_q g(T_i) = h ||omega_i||^b / P = h a_i^{1/s}/P.

No balancing or class-C assumption is needed; it holds for every network, p, n and b. E_dual D3c supplies existence of multipliers at local maximisers with h > 0 (MFCQ). The per-feature equivalence E_q g_l <= Lambda P_l <=> gamma <= Lambda requires P_l > 0 and a feature-independent Lambda.

Independent NNLS check at E_kkt lm_p5_n2_b2.0_k0 (a mixed network): KKT residual 1.4e-9; E_q g(T_i)/(h a_i^r/P) in [1.000000, 1.000000] (v6 (c)).

### CC6: MINOR_GAP

Every algebraic step is correct and was checked (v1, random pi):
- WF_l <=> bracket form: 0 mismatches.
- CD_pair(delta_0) => CD: 0 violations.
- rho in [0,1] (min of a sum >= sum of minima): 0 violations.
Summing WF at delta_0 gives h <= h_{delta_0} <= pi B M(P^s), and the converse with the free rule is immediate.

Caveats:
(a) The last arrow, CD => RD_S, inherits the ghost-frequency scoping gap of CC3; the chain ends at RD_S only when supp P is inside S.
(b) The rho/taubar factorisation needs sum_l D_l^iso > 0, i.e. at least one non-IF feature. For in-class-C circular fields, D_l^iso = 0 iff f_l = 0.
(c) 'Weakest inequality implying RD' is meant within single-offset charged inequalities; CD_pair(delta_0) is strictly stronger than CD because h_{delta_0} >= h.
The content is largely definitional but sound.

### CC7: CONFIRMED

The proof is correct for every p >= 5, n and b (vacuous at p = 3), with theta_1 in class C so the union is in N^C_F:
- relabelling gives min_x g'(x,delta) = m(j delta);
- k = +-j l^{-1} != +-j because l != +-1, and k delta_0 = +-1;
- if eps^s < (kappa_1(2) - kappa_1(1))/2 then +-delta_0 is the unique binding pair, since kappa_k(delta) >= kappa_1(2) elsewhere;
- off delta_0 the lower bound is >= pi(kappa_min + gap) - O(eps^s).
Attainment of pi is not needed: a feature with price pi - eta, eta << eps^s, suffices.

Independent check (v6 (a)) at (5,2,2), (7,3,2), (11,2,2.4) with my own relabelling: the binding pair is as predicted, and the CD ratio equals the predicted pair term to 1e-6 and tends to 1 from below.

Minor wording error: the parenthetical reason 'allocation slack Gamma_s - kappa_min > 0' is wrong in general. For example Gamma_s = kappa_min at p = 5, s >= 1.855, where K* = 1. RD fails to give DU_min because RD charges (1+eps)^s - 1 ~ s eps, which is much larger than eps^s; this is the Pi_feat slack. The conclusion is unaffected.

Note: at n = 2, DU_min for single-frequency features is already GLOBAL R9(iii), so this route yields no class-C CD counterexample at n = 2. The statement is close to GLOBAL R8(c).

### CC8: CONFIRMED

Correct. Pointwise, sum Q (g_a - h_Delta)(g_b - h_Delta) >= 0 by copositivity; taking expectations gives the means term plus Cov.

In class C, E f_l f_k = 0: 2l 1_T + 2k 1_{T'} = 0 would need T = T' and l = -k, impossible in F. Also E f_l conj(f_k) = 0 for l != k. Hence Cov(g_a, g_b) = sum_l V_l Psi^(l)_ab / 2.

For the pair, Psi_{+-} = 2 kappa (kappa - 1) = -2 kappa cos(phi), which gives the stated pair cone moment. W8(i) follows with k_p = sec(2 pi/p) - 1, after a phase rotation (|A| >= Re beta).

Independent checks (v3, 350 cases, n = 2, 3, 4):
- covariance identity 2e-15; V_l = sum_T |beta_{l,T}|^2 to 1.7e-15;
- the inequality held for random PSD+N and non-negative Q AND for the Horn matrix (copositive but not PSD+N);
- the pair cone moment held at every offset.

### CC9: CONFIRMED

Exact algebra. On {+-delta}, -(1/2)<Q, Psi> = -kappa[(Q_++ + Q_--) + 2 Q_+- (kappa - 1)] for symmetric Q (checked to 7e-14 in v3). With kappa >= 1, copositivity gives Q_++ + Q_-- >= 2|Q_+-|, so w <= -2 kappa |Q_+-| (2 - kappa) <= 0. kappa = 1 is impossible for odd p.

Acute features need Q_+- > 0, and then antipodal weights are < 0. At p = 3, cos(2 pi/3) < 0 at every offset, so the lemma is vacuous there. General in p, n and b.

### CC10: CONFIRMED

sum_{l=1}^F kappa_l(a) = F + 1/2 = p/2 for a != 0, and kappa(0) = 0. So the sum is (p/2)(I + J) on any Delta, including Delta containing +-a (2a != 0). I + J is completely positive, and CP is the dual of COP, so the right side is <= 0 when the V_l are equal; the left side is >= 0 automatically, so the inequality is vacuous. The pair-moment sum equals -p V/4 exactly.

Checked to 1.2e-14 (v3), including p = 3. Fully general.

### CC11: CONFIRMED

The main inequality is correct. With disjoint T_i and distinct l_i, every multi-factor character takes the value 2 sigma_i l_i on T_i. It is non-constant, hence neither 0 nor +-2k 1_{T''}, so E w = 1. A single-factor character -2l 1_T never equals 2k 1_{T'}, because k = -l is not in F. The coefficient modulus is |beta_{l,T}| |c_l(delta)|/2 = |beta_{l,T}| |sin(pi l delta/p)|; optimise phases separately for +-delta.

Independent check (v3), random class-C networks at (5,2), (7,2), (11,2), (13,2), (5,3), (7,3), (5,4) with random disjoint families: min(D - detection) = +0.012 >= 0.

Minor:
(a) In the collision list, 'k in S' should read 'any frequency carrying sign-flipped content' (ghost frequencies included).
(b) The l = l' cases, e.g. 2l 1_{T u T'} or the 2l 1_T - 2l 1_{T'} difference, are excluded only by the distinctness hypothesis.
(c) The aside that combining with paper Theorem A 'reproduces the Note's few-cheap-features decoupling' has no proof here and is unverified.

### CC12: CONFIRMED

The refutation is valid. I recomputed it with my own code (v4_cex.py):
- h = 206.6068869 > 0, exact Fraction margin;
- CD ratio 0.9415955 < 1; binding offsets {2,3}, a single +- pair;
- cheapness sqrt3 and 0.9533;
- Shapley charge/price 2.39361071 and argmin charge/price 6.26635127 at both offsets;
- depth game v(1) = 0.114, v(2) = 1.929, v(12) = 1.768.

Robustness of the discontinuous argmin rule:
- the stored rounded weights are class C only up to 1.7e-10 relative off-plane amplitude;
- the exactly projected (unrounded) class-C network gives identical values;
- the argmin is unique, with a relative gap of 5e-4 to the second-smallest gap, far above the rounding.
So WF fails for the cheap feature under both rules while CD (hence RD; this network has no ghosts) holds.

### CC13: CONFIRMED

Reproduced with independent code (v5_mixed.py):
- SPLIT violators: CD_prop = 1.0300/1.0296/1.0270 (p = 5), 1.0662/1.0436/1.0427 (p = 7), 1.0985-1.1061 (p = 11);
- CD_split is also > 1;
- E_kkt lm_p5_n2_b2.0_k4: CD_prop 1.01464; lm_p7_n2_b2.0_k6: CD_prop 1.00978.
So canonical CD for mixed networks is refuted by the certified SPLIT violators.

Caveats:
- The 'local maximiser' status is numerical: re-optimisation returns 0.999998 gamma and never exceeds it.
- The violators carry small DC, 0.1-0.9% of the squared norm (v5), so 'cost-neutral at b = 2 by W5' is only approximate; W5 equality needs no DC.

### CC14: CONFIRMED

This is a numerical claim and it reproduces:
- re-evaluating all stored best_cd_*.npz with my code gives max CD ratio 1.000000;
- the width < nu runs give 0.235-0.545 as stated;
- my own torch search at (5,2,2), S = {1,2} (not in the agent's grid), seeded from the CC12 counterexample and random class-C starts with unequal widths, peaks at 0.9999996 (v8, v8b).

Limitations:
- p <= 13 and n <= 3;
- equal fixed widths per feature in the agent's runs;
- smooth-min L-BFGS;
- no ghost frequencies in any run. My separate ghost search at S = {1}, p = 5 and 7 (v2b) found no gain.
At n = 3, nu^{1-s} replaces pi, which is the stronger test, as stated. This is evidence, not proof.

### CC15: FALSE

The 'programme maximisers' used by c5_struct (c) and c7_quad (ii) come from mechanism_scripts/prog.solve(restarts=4), a reweighted-LP local heuristic. Against exact vertex enumeration (p <= 19) and an exhaustive K <= 2 lower bound (p >= 23), it is SUBOPTIMAL in 65 of the 144 grid cases (v7b_prog_audit.out), often by a factor of 2-5 in value. Examples:
- p = 5, s >= 2: true K* = 1, value kappa_min = 0.691, while prog returns a uniform 2-support with value 0.625;
- p = 7, s >= 2.5: true K* = 1;
- p = 13, s = 2: 0.288 vs the true 0.328.

At EXACT maximisers (v7_binding.py, p <= 19, s in {1.05, 1.5, 2, 3, 4}):
(a) all-acute binding offsets exist at (13, s = 2), (13, 3), (17, 3), (19, 3) and (19, 4). For example p = 13, s = 2 has support {4,6} and binding {4,6} with every cos > 0, contradicting '143/144'.
(c) joint detection by one entrywise-nonnegative Q is FEASIBLE at (11, s = 1.5, 2, 3), (13, 2, 3), (17, 1.5, 2, 3) and (19, 1.5, 2, 3, 4). At (11, s = 2), the single pair (1, -3) has negative Psi entries for both support features 1 and 4. This contradicts 'only 9/144' and the listed undetectable '(11, s <= 2.5)'.

Also, 'for p = 5 and 7 they follow from CC10 since the support is F' is wrong for p = 7, s >= 1.75 (K* <= 2 < |F| = 3) and for p = 5, s >= 2 (K* = 1). The counts and the 'quadratic route is blocked at programme maximisers' conclusion do not hold as stated.

### CC16: CONFIRMED

The numerical consistency statements check out:
- p = 3: CD with pi is the definition of pi; the CD_IF ratio equals gamma/gamma_IF (1.054, 1.109, ...) as it must.
- The mixed violator is the same network as E_dual's theta_v.
- IF networks satisfy CD = 1 at every allocation (CC3(iv)).
- (5,6) width-6: 0.177.

Minor attribution error in (5): CC10 needs EQUAL energies, but the Legendre design has V_l proportional to 1/kappa_l(1). I checked the stronger fact directly (v9_legendre.py):
- the pair cone moment is vacuous at EVERY delta, not just delta = 1 (max fraction -0.0167, -0.0100, -0.0025 at p = 61, 101, 401);
- all off-diagonal entries of sum_l V_l Psi^(l) are >= 0, so no entrywise-nonnegative Q on any Delta detects it.
The conclusion therefore holds, but by direct computation rather than by CC10.
