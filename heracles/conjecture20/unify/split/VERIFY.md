# Referee verdicts on split/REPORT.md (round 2; saved by the orchestrator from the referee's structured output)

## Overall
The SPLIT backbone holds up under hostile refereeing. R1, R2, R4, R5, R6 and R8 are confirmed: I reconstructed complete proofs and wrote independent scripts in /home/user/grok/heracles/conjecture20/unify/split/verify/. Those scripts use FFT-mask projections and read own data off the table's FFT rather than from atom formulas, and test random, sparse/dead, heavy-DC and adversarially optimised networks at p = 3 to 13 and n = 2 to 4. Every check passes.

Four claims carry precise defects:
- R3: the stabiliser misses the output-DC sign, and the 'cannot attain' remark is false for IF networks.
- R7: the 'equality iff g_R = 0 on binding offsets' clause is FALSE in one direction under either reading. Certified constructions at p=5 are in v_r7_r9.py and v_r7_r9_extra.py; only 'equality implies g_R = 0 on split-binding offsets' survives.
- R9: correct only with split-binding offsets. Its backward proof cites R7a outside its hypothesis; R9(d)'s content is already Note Prop count(b); ROUND1's 'iff' is unsupported.
- R10: the (5,2,2) violator is re-certified exactly, in Q(sqrt5) plus intervals, matching the reported 1.033200, 1.029689 and 0.790214. But it refutes only the canonical-split forms of decoupling. The Note's existential (D) holds for it (ratio 0.790 with P_j = P/2), so 'no all-network inequality can replace (MM)' and ROUND1's 'must use global optimality' are overclaims.

Tested against the exact p=3 certificates (v_p3.py), no SPLIT claim implies the IF value at p=3. The certificates are not IF (|beta_{1,T}| up to 1.86), their splits have margin -284 and -746, R8 holds with huge room, and R2's Jensen inequality holds. Mathematically, the reduction to (MM_opt) is correct but, as the report admits, it is a value-type statement.

All scripts and outputs are in that verify/ directory: vlib.py, v_r1_r2, v_r3_r4, v_r5_r6, v_r7_r9, v_r7_r9_extra, v_r10_certify, v_p3 (.py and .out).

## Salvage
R3 (corrected): plane-preserving uniform maps that fix all beta_j and beta_{j,T} are exactly Sigma' x {output-DC sign}. The torus average keeps the clean modes plus the all-DC mode. For networks with interference, the IF projection generally costs more than the split; for IF networks it equals the split.

R7(a) (corrected): if theta is IF, then h(theta) <= h(theta_split). If equality holds, then g_R(.,delta) = 0 for every delta that is binding for theta_split. The converse is false: certified at p=5 by an IF block plus a mixed table A cos(2 pi (x1-x2)/5) e(c-y).

R9(b) (corrected): define the binding offsets as argmin_delta min_x g_split(.,delta). Under (IF_C), the following are equivalent:
- (MM_opt);
- every global maximiser is IF;
- every global maximiser has no DC (input or output), no dead content, P_split = P, and g = g_split on the split's binding offsets.
The proof of (iii) => (i) is direct: h(split) = min g_theta*(.,delta_s) >= h(theta*). At b > 2, these maximisers lie in class C.

R9(d) needs no (IF_C): 'every global maximiser is IF => (C)' holds at every (p,n,b) by Note Prop count(b).

R10 (corrected scope): CERTIFIED that the canonical-split decoupling inequalities (D_Q) and (D_split), with the true price 3^{1-s}, fail for a non-maximiser mixed network at (5,2,2). Its split satisfies both, and (MM) fails for this network (gamma_split = 0.378 gamma_IF < gamma = 0.790 gamma_IF). NOT refuted: the Note's existential hypothesis (D) 'for some split' as an all-network inequality, which this network satisfies with ratio 0.790.

## Verdicts

### R1: CONFIRMED

I rebuilt the proof line by line.
(a) The planes P_0 and P_k (k in F) are orthogonal in every factor, so ||omega||^2 = e_0 + sum_k e_k. Since t -> t^{b/2} is superadditive and increasing for b >= 2, sum_k e_k^{b/2} <= (e_0+E)^{b/2}. Equality holds iff e_0 = 0 and (b = 2 or at most one e_k > 0); a pure-DC neuron is strict.
(b) With q = b/2 > 1, three tangent-line steps give: (E+e_0)^q - E^q >= q E^{q-1} e_0, and E^q - sum e_k^q >= (q-1) M^{q-1} (E-M), because sum e_k^q <= M^{q-1} E. At b = 2 the slack is exactly e_0, so P - P_live = DC energy + dead energy, where a dead piece is one with a zero factor (equivalently, zero atom).
(c) Piece (i,k) has uhat(+-j) = 0 for j != k and keeps uhat(+-j), what(j) for k = j. So atoms, xi, beta_j, beta_{j,T}, Q_j, a_j, the cheap set, Delta, G and E_x g are all preserved; G depends only on the Q_l (Note, sec:arch). theta_split has no DC, so it lies inside the Note's class C, which allows output DC.
Independent checks (verify/v_r1_r2.py -> v_r1_r2.out), using my own FFT-mask projections and reading beta_T off the table's FFT rather than from atom formulas:
- p,n in {(3,2),(3,3),(5,2),(5,3),(7,2),(7,3),(11,2),(5,4)}; b in {2, 2.3, 2.5, 3, nu}; random, sparse (exact dead pieces) and heavy-DC networks.
- cost(split) - cost <= 0 in every case. The R1(b) bound minus the slack is <= 7e-15. |slack_b2 - e_0| <= 7e-15.
- beta, beta_T, atoms and E_x g are preserved to 1e-15.
- The split table equals the diagonal-mode projection of the table, to 1e-15. This also shows the split table is linear in the table.
- Equality cases behave as stated.
- On the p=3 certificates (v_p3.py, exact): cost(split) <= cost.

### R2: CONFIRMED

(1) Each map is +-1 on orthogonal planes, so it is orthogonal per factor and preserves cost. In beta_{j,T}, each uhat_t(+-j) appears once and conj(what(j)) once, so the signs give (prod_t s_{j,t})^2 = 1. The DC signs d_t never touch beta_j.
(2) The sign of a block with labels (kappa_1..kappa_n; k) is as stated. It is a trivial character iff kappa = (k,...,k): any kappa_t = 0 brings in d_t (or s_{k,t} without DC flips), and any kappa_t not in {0,k} brings in s_{k,t} s_{kappa_t,t}.
(3) Averaging kills every block with a non-trivial character. Without DC flips, the all-DC block also survives. With DC flips, only output-DC blocks remain, and they are constant in c, so they are gap-free.
(4) h is a minimum of linear functionals of the table, so it is concave; Jensen applies.
(5) The partial subgroup keeps the (k..k;k) blocks for k in K and the blocks with all labels outside K. That is exactly the table of {omega^(k): k in K} u {(I - sum_K P_k) omega}, and superadditivity gives the cost inequality.
Independent checks (v_r1_r2.py), with exact group averages at (3,2), (5,2), (5,3), (7,2), (3,4) on random, heavy-DC and sparse networks:
- Sigma-average equals split + all-DC block (3.6e-15). With DC flips, the gaps of avg - split are 2.4e-15.
- Per-neuron averages agree (3.6e-15). The partial-split identity holds (8.9e-15) and its cost is non-increasing.
- Cost and own data are invariant (4.5e-13, 8.7e-16). E_sigma h <= h(split) everywhere; equality occurs only for networks already in class C.
- On the exact p=3 certificates (v_p3.py), by slot symmetry h(sigma theta) depends only on |S|: E_sigma h = -4218 <= h(split) = -284.5 at n=11, and -17412 <= -746.3 at n=13.

### R3: MINOR_GAP

(a) The argument is correct. Identify each plane with C via vhat(k); an orthogonal plane map is z -> e^{i phi} z (rotation) or e^{i phi} conj z (reflection). For one-neuron networks, beta_k is holomorphic in each uhat_t(k) and antiholomorphic in what(k), so reflections are excluded. Fixing beta_k gives sum_t phi_{t,k} = psi_k; fixing beta_{k,{t}} gives 2 phi_{t,k} = 0, so phi_{t,k} is in {0, pi}.
The statement 'are exactly Sigma'' is slightly false. Sigma' as defined in R2 never flips the output DC, yet the sign flip of the output DC component also fixes every beta_j and beta_{j,T} (checked: change 8e-16). The stabiliser is Sigma' x Z_2. This is harmless, since output-DC blocks never enter gaps, but the statement should be corrected.
(b) This is correct. The phase of a mode is linear in the angles with integer coefficients, and it vanishes identically only for clean modes (j..j;-j) and the all-DC mode. The all-DC mode is omitted from the statement but appears in the check.
The remark that the larger average 'is the IF projection, which the split cost cannot attain' is false as a universal statement: for an IF block, the clean projection IS the split table at the same cost (checked: 7.8e-16). It holds only for networks that carry interference.
Independent checks (v_r3_r4.py), at (7,3):
- 200 random rotation sets match the predicted factor e^{-2i sum_T phi} on beta_{k,T} (2e-16).
- 0/pi rotations fix all own data.
- Generic rotations change some beta_T (min change 0.08).
- Slot reflections change beta_k under every output rotation.
- An exact torus average (N=5 grid) at (5,2) keeps the clean and all-DC modes (7e-15) and kills all others (7e-15).

### R4: CONFIRMED

(a) From u(a) rho^{la} = p^{-1} sum_c uhat(c-l) rho^{ca}, I get phi_j(chi) = 2 p^{-nu} sum_i what_i(j) prod_t uhat_it(chi_t - j). So for each output j, character chi corresponds to exactly one input mode q = chi - j1.
- q_t in {+-j} iff chi_t in {0, 2j}; this gives phi_j(0) = conj beta_j and phi_j(2j1_T) = conj beta_{j,T}.
- Hence mixed modes vanish on {0,2j}^n and E_x g_R = 0.
(b) The formula for ghat_delta follows from g = sum_j Re(Phi_j c_j) and Re z = (z + conj z)/2. At chi_{l,T}, a direct mode needs input 2l - j on T. This is clean only for j = l, since 2l = +-2j forces j = l when p is odd. The conjugate mode needs -2l - j, i.e. |3l| for j = l, and |2l + j| = j is impossible. That gives exactly 1 + 2(|F|-1) = 2|F|-1 channels. DC appears iff j = |2l mod p| (or 3l = 0, i.e. p = 3).
(c) Exhaustive check for n <= 6: every chi in Z_3^n equals 2*1_T - 2*1_S.
Independent checks (v_r3_r4.py), at (5,2), (7,2), (7,3), (11,2), (13,2), (5,3), Gaussian and heavy-DC:
- The phi formula matches the FFT of the field computed from the table (<= 1.9e-15). The clean identifications hold (<= 4e-16).
- The mixed table R = f - f_split - allDC has zero content on {0,2j}^n (<= 4.5e-16), and E_x g_R = 0.
- The mixed part of ghat_delta at +-chi_{l,T} equals the sum over the listed channels (<= 2.5e-15). An assertion confirmed in every case that exactly 2|F|-1 channels occur, with input |3l| or |2l -+ j| on T and on-frequency input off T.

### R5: CONFIRMED

Per neuron, ||omega||^b - sum_{k live} ||omega^(k)||^b = (R1 slack) + sum_{dead k} ||omega^(k)||^b >= 0.
- At b = 2 this is DC + dead energy, so equality holds iff neither is present.
- At b > 2, any neuron with two non-zero pieces (DC counted) has strictly positive slack by R1(b).
- Each live piece has a single non-zero atom, at its own frequency, with |beta_ik|^r <= C_b ||omega_i^(k)||^b. Summing gives sum_l Q_l <= C_b P_live.
- Using U = nu^{1-s}((C_bP)^s - (sum Q)^s) Gamma_s (the Note's definition, sec:arch) and C_b^s = B_p, I get U >= nu^{1-s} B_p Gamma_s (P^s - P_live^s).
- The F2 condition is R4(b) restricted to neurons whose j- and q-pieces are both live (q != 0, because DC is never live).
- The 'quadratic' parenthesis is cited from Note Lemma 'paid compensation is quadratic'; I did not re-referee that lemma.
Independent checks (v_r5_r6.py), at (5,2), (7,2), (5,3), (3,2), (11,2), b in {2, 2.5, 3, nu}, Gaussian/sparse/heavy-DC:
- (sum Q - C_b P_live)/(C_b P) <= 0. The maximum is 0, reached only by a trivial all-dead case. My first run used a 1e-13 liveness tolerance and produced an 'inf' artefact (Q of order 1e-32 against P_live = 0); I fixed the tolerance.
- At b = 2, P - P_live = DC + dead energy to 6e-14.
- Every multi-piece neuron at b > 2 has positive slack (min 0.028).

### R6: CONFIRMED

(a) Parseval with E rho^{(chi+chi')x} = [chi + chi' = 0]. A non-zero term needs q_t + q'_t = -(l+k) in every slot, with chi != 0.
(b) In class C, q is in {+-l} and q' in {+-k}. Of the four sums, l + k = -(l+k) would need l + k = 0, and the others would need 2l = 0 or 2k = 0; none is possible for l, k in F and odd p, p = 3 included. Output DC is irrelevant.
(c) Forward direction: if a = a' then q_t = q'_t = -l, i.e. chi_t = 0. So chi != 0 needs a slot with a != a' and +-a +- a' = -2l, which is a 2l-pair.
The converse is stated only for generic weights and has no proof in the report. I supply one. Take a single witness neuron: output l, slot t supported on the pair planes {a, a'}, the other slots at frequency l only. Then the pseudo-variance is 2 phi(chi) phi(-chi), a product of non-zero Fourier coefficients. This also covers the doubly related pair {0, |2l|}. So the polynomial is not identically zero, and genericity follows.
(d) The companion proof of Thm 14 uses only the cone lemma, Lemma cc (Var X = Var Y from circularity, then (Re A - c)^2 >= (cot^2 - 1) sigma^2), Parseval (sigma^2 >= (1/2) sum_T |phi_l(2l1_T)|^2 = (1/2) sum_T |beta_{l,T}|^2 by R4(a)), and the cost bound ||theta||^nu >= B_p^{-1} (sum_i |beta_il|^r)^s. Every one of these holds for arbitrary inputs, so the extension is valid for p >= 5. It is vacuous beyond the companion's own case at p = 5, where the only pair-free set containing l = 1 is {0,1}.
Independent checks (v_r5_r6.py):
- Class-C joint pseudo-covariances, including p = 3 and output DC, are <= 3e-16.
- 300 random single-output networks at (5,2), (7,2), (7,3), (11,2), (13,2): no violation of the forward implication and no exception to the converse.
- Adversarial L-BFGS (with IF-seeded restarts) on pair-free classes at p=5 {0,1}; p=7 {1,2} and {0,1}; p=11 {0,1,4} and {1,2,5}; p=7, n=3 {1,2}: max of (min cm - (Re beta - k_p sigma))/|Re beta| is <= -0.0104, and max gamma / single-feature IF value is <= 0.9997.

### R7: MINOR_GAP

The main inequality is proved correctly. A = V_delta^perp is a subspace of F_p^n. Averaging over A-cosets multiplies ghat(chi) by [chi in V_delta], so the average equals g_split + g_R^V, and the minimum is at most the coset average. Corollaries (b) (modes reaching c*1_T share the signed input c - j on T) and (c) (tests supported on Lambda see the same E[w g]; U can only grow, other terms preserved) are also correct. For (a): in the IF case V = {0}, and E g_R = 0 gives h(theta) <= h(split).
The clause 'with equality iff g_R = 0 on every binding offset' is FALSE under either reading of 'binding offset'. I constructed explicit counterexamples (v_r7_r9.py, v_r7_r9_extra.py), at p=5, n=2:
- Construction: theta = the IF frequency-1 block plus neurons realising R(x)[c] = A cos(2 pi (x1 - x2)/5) e(c - y). Here e(0) = e(+-1) = 0 and e(+-2) = -1, and R is realised exactly via SVD as a sum of rank-one product neurons.
- R has no diagonal modes, so theta stays IF (max |beta_T| = 3e-16), table(split) = core table, g_R(.,+-1) = 0 and g_R(.,+-2) = A r(x).
- Split-binding reading: the split binds at {1,4}, where g_R = 0. Yet h(theta) = 0.4749 < h(split) = 0.5182, so the 'if' direction fails.
- theta-binding reading: tuning A to a tie gives h(theta) = h(split) to 2e-16. theta now binds at {1,2,3,4}, and g_R != 0 on 2 and 3 (max 1.04), so the 'only if' direction fails.
The true statement is: h(theta) = h(split) implies g_R(.,delta) = 0 on every offset that is binding for the SPLIT. That is all R9(b) needs.
Other checks:
- The main inequality on random networks at (5,2), (7,2), (5,3), (7,3): max of min g_theta - min(g_split + g_R^V) is 1.4e-14.
- Glued IF blocks plus input/output DC: h(theta) - h(split) <= -2.37 is the largest value, so always negative.
- The p=3 certificates are not IF (|beta_{1,T}| up to 1.86 and 0.98), so R7(a) does not contradict them.

### R8: CONFIRMED

The chain holds for every network. E_x g(.,delta) = sum Re beta_l kappa_l(delta) + sum Im beta_l sin(2 pi l delta/p); averaging over +-delta gives paper Lemma 2. Then Re beta_j <= |beta_j| <= sum_i |beta_ij| <= Q_j^s (r <= 1), and Q_j <= C_b P_j^live (live-piece budget; dead pieces have zero atom), with kappa >= 0. The second inequality uses y_j = (P_j^live/P_live)^s, which has sum y^r = 1. The consequence uses gamma* >= pi B_p Gamma_s (price-programme theorem) and pi >= nu^{1-s}. Global maximisers exist: a concave-cost vertex argument bounds the width by p^nu, and the problem is compact at fixed width.
Independent checks:
- Adversarial L-BFGS maximisation of h/(B_p min_delta sum (P_j^live)^s kappa_j) at (3,2), (5,2), (7,2), (5,3), (3,3), b in {2, 2.5, nu} (v_r7_r9.py): sup = 1.0000, reached by IF-type networks, never exceeded.
- Exact p=3 certificates: ratio 1.05e-5 (n=11) and 4.4e-7 (n=13). So R8 is consistent with the p=3 counterexamples and does not imply the IF value (its constant is B_p, not nu^{1-s} B_p).
- At b = nu, gamma/(B_p Gamma_1) <= P_live/P on all networks tested.

### R9: MINOR_GAP

(a) Correct, and nearly tautological: gamma(theta*_split) >= gamma* >= gamma*_C >= gamma(theta*_split), and R1(c) carries the features.
(b) Correct only with 'binding offsets' read as the SPLIT's binding offsets, which the report never defines.
- (i) => (ii): under (IF_C), theta*_split is a class-C maximiser, hence IF, and it shares beta_T with theta*.
- (ii) => (iii): R7a and P_split <= P give gamma(split) >= gamma*. Hence h and P are equal, so there is no DC (input or output) by R1a, no dead pieces (they could be deleted to beat gamma*_C), and, by the TRUE half of R7a, g_R = 0 on split-binding offsets.
- The written backward direction cites R7a, which needs theta* to be IF; (iii) does not assert that. The correct argument is direct: h(split) = min g_split(.,delta_s) = min g_theta*(.,delta_s) >= h(theta*), together with P_split = P.
- Under the theta-binding reading, (ii) => (iii) is not established: my R7 tie example shows IF networks with h(theta) = h(split) and g_R != 0 on a theta-binding offset.
(c) Correct. At s = 1, R8 gives gamma <= B_p Gamma_1 P_live/P, and the IF value equals the universal bound, so P_live = P. Since b = nu > 2, there are no DC, dead or shared pieces, i.e. class C.
(d) Correct, given the cited Note Cor. 'Parseval closure'(c): Thm loc with sum_F kappa = p/2 < 4 at p = 5, 7. But the content '(C) follows from every global maximiser being IF' holds at EVERY (p,n,b) by Note Prop 'what is unconditional'(b), without (IF_C).
The ROUND1 phrasing '(C) for all networks IFF every global maximiser is IF' at (5,2) and (7,2) is unsupported: only the 'if' direction is proved.
Checks (v_r7_r9.py, v_r7_r9_extra.py): at (5,2), b = 3, ascent from IF + mixing noise reaches gamma/gamma* = 0.99996 with P_live/P -> 1.000000 (start 0.87-0.90). No part of R9 implies the IF value at p = 3: (MM_opt) and 'every maximiser is IF' both fail there, consistently with (b).

### R10: MINOR_GAP

I re-certified the counterexample independently (v_r10_certify.py -> v_r10_certify.out), by a different route from certify_viol.py.
- Every algebraic quantity is computed EXACTLY in Q(sqrt5), with exact signs: |what|^2, |uhat|^2, |beta_il|^2, piece energies, kappa = (5 -+ sqrt5)/4, the exact rational margin h, and the exact margin of the split in Q(sqrt5).
- Only the cube roots and 3/2 powers use 80-digit mpmath intervals.
- I ran two weight sets: the exact binary rationals of the stored float64 weights, and the agent's 12-digit rounding.
Both give, for theta: h/RHS(D_Q) = 1.0331995438..., h/RHS(D_split) = 1.0296885360..., gamma/gamma_IF = 0.7902141005..., all pieces live, both features present.
For the exactly computed split: 0.4948622306, 0.4931805950 and gamma_split/gamma_IF = 0.3784816929.
Gamma_{3/2}(5) = 5/(4 sqrt2) is confirmed, with the maximiser at u = 1/2, despite the non-convex constraint. The price nu^{1-s} is the true price at n=2 for b in [2,3] (Note Thm q1n2), and 'every class-C network at (5,2) satisfies (D_Q)' follows from Note Thm loc. So canonical-split decoupling (P_j proportional to Q_j, or to P_j^live) fails for this non-maximiser, and (MM) fails for it too (0.378 < 0.790).
The gap is the scope. The sentence 'This is the hypothesis of Note Prop. decoupling suffices' is inaccurate: the Note's (D) is existential ('for some P_j'). The same violator satisfies it with P_j = P/2, the programme maximiser, at ratio 0.790. So R10 does not refute the existential all-network inequality, which at (5,2) is essentially the value statement gamma(theta) <= gamma_IF(supp). The headline 'no all-network inequality can replace (MM)' and ROUND1's 'any proof of (C) must use global optimality, not an all-network inequality' are therefore not established.
