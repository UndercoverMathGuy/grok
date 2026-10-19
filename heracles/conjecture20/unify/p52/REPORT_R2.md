# p52: report (round 2; saved by the orchestrator from the agent's structured output)

## Headline
(C) is PROVED at (p,n)=(5,2) for every b in [2,3]. The proof is computer-assisted with interval arithmetic: every network with only one feature has gamma <= 0.995 gamma_IF, and b=2 dominates all b. At (7,2), (C) is PROVED for b in [2.5381,3], extending the Note's dense range b >= 2.600. Single features at (7,2) are excluded for every b in [2,3] (gamma <= 0.9682 gamma_IF). A new detection lemma, delta-Fourier isolation, holds for every p, n and network. The remaining (7,2) gap b in [2,2.538) has three precise obstructions: a phase-blind cap near b=2.50 (numerical), failure of the free relaxation for b <= 2.14 (numerical), and free non-own modes from shared neurons at b=2 (proved). All work, scripts and logs are in /home/user/grok/heracles/conjecture20/unify/p52/. The LaTeX write-up is p52_count_theorem_proof.tex.

## Principle
Price every compensating mode in the same currency as the cost, using three ledgers, then remove the unknown compensation mass R.

Ledger 1, the margin certificate (R4). For any probability q on (x, delta), h <= sum q g. This sum splits into:
- an exact pairing with the own coefficients A, B, C, D;
- a pairing with every other mode, bounded by R*n through one l2 number n of the certificate.

Ledger 2, the mode budget (R2-R3). This turns the non-own l2 mass R into cost through a Pythagorean inequality: 3^{1-s} B P^s >= sqrt(P_1^2 + 3^{2-2s} R^2/4).

Ledger 3, the twisted Rotfel'd payment (R7). This prices the own data, with three chain matrices and a circle reduction.

R is then removed by Cauchy-Schwarz: sup_R (a + nR)/sqrt(F^2 + cR^2) <= sqrt(a_+^2/F^2 + n^2/c). By s-monotonicity (R8), b=2 dominates every b in [2,3].

Detection uses delta-Fourier isolation (R12). A test whose delta-profile is exactly rho^{j delta} pairs with no Fourier coefficient of any field except:
- the clean coefficients;
- feature j's own sign-flip coefficient at chi_{j,T} = 2j 1_T.
This holds for every odd p, every n and every network (any width, DC and multi-frequency inputs, outputs on all frequencies).

What the principle removes:
- At (5,2): every maximiser with one feature, for every b. Hence (C) at (5,2).
- At (7,2): one-feature maximisers for every b, and two-feature maximisers for b >= 2.538.

What it does not remove:
- Two-feature networks at (7,2) for b < 2.538.
- At b=2, ledger 2 has no multi-feature analogue. Shared neurons produce non-own mass 3e^2 at zero marginal cost (R17).

## Results

### R1 [PROVED]

Reduction at (5,2). Let p=5, n=2, b in [2,3], s=3/b in [1,3/2], r=1/s.
(i) Gamma_s(5) = (5/2) 2^{-s}. Its unique maximiser is y* = (2^{-s}, 2^{-s}), so K* = 2.
(ii) Suppose every network theta with beta_2(theta) = 0 has gamma(theta) < gamma_IF = 3^{1-s} B_5 Gamma_s. Then every global maximiser of the L_{2,b} margin uses exactly 2 features.

**Proof / evidence.**

Notation: kappa_a = 1 - cos72 and kappa_b = 1 - cos144, so kappa_a + kappa_b = 5/2. The coverages are kappa_1(+-1) = kappa_2(+-2) = kappa_a and kappa_1(+-2) = kappa_2(+-1) = kappa_b.

(i) Take y >= 0 with y_1^r + y_2^r <= 1. The minimum over delta of sum_j kappa_j(delta) y_j is at most the mean of the delta=1 and delta=2 values, which is (5/4)(y_1 + y_2). By the power mean, y_1 + y_2 <= 2^{1-s}. So the minimum is <= (5/2) 2^{-s}.
- Equality forces y_1 = y_2. For r < 1 this is the power-mean equality case. For r = 1 the two delta-values must agree, i.e. (kappa_b - kappa_a)(y_1 - y_2) = 0.
- One-point supports give at most kappa_a = 0.691 < (5/2) 2^{-s}, because s < 1.855.

(ii) Let theta* be a global maximiser. Then gamma* >= gamma_IF (paper Thm 8).
- With no feature, h <= 0 by paper Lemma 2.
- If its only feature is 2, apply the relabelling a -> 3a in both input slots and in the output: u'_t(a) = u_t(3a), w'(c) = w(3c).
- This gives g'(x, delta) = g(3x, 3 delta) and the same cost.
- Its transforms are uhat'(k) = uhat(2k), so beta'_1 = beta_2(theta*) != 0 and beta'_2 = conj(beta_1(theta*)) = 0.
- The relabelled network has beta_2 = 0 and gamma >= gamma_IF, which contradicts the hypothesis.

*Check:* sanity52.out L_A: |beta'_1 - beta_2| = 5.6e-17; margin difference 0.

### R2 [PROVED]

Mode budget at p=5. Take a neuron omega = (u_1, u_2, w) in (R^5)^3 and define:
- m(j;k) = (2/125) what(j) uhat_1(k_1) uhat_2(k_2), for j in {1,2} and k in Z_5^2;
- e = B_5 ||omega||^3, with B_5 = 2*30^{-3/2};
- beta(omega) = (2/125) conj(what(1)) uhat_1(1) uhat_2(1).
The own modes of feature 1 are (1; +-1, +-1). Then sum over non-own (j,k) of |m(j;k)|^2 <= 4(e^2 - |beta(omega)|^2).

**Proof / evidence.**

1. Parseval gives sum_k |uhat_t(k)|^2 = 5||u_t||^2. Since w is real, |what(1)|^2 + |what(2)|^2 <= (5/2)||w||^2.
2. So sum over (j,k) of |m|^2 <= (2/125)^2 (5/2)(25) ||w||^2 ||u_1||^2 ||u_2||^2.
3. By AM-GM this is <= (2/125)^2 (125/2) (||omega||^2/3)^3 = 4 B_5^2 ||omega||^6 = 4e^2.
4. Each of the 4 own modes has modulus |beta(omega)|, because |uhat(-1)| = |uhat(1)|.

*Check:* sanity52.out L_B: max over random neurons of [non-own mass - 4(e^2 - |beta|^2)] = -7.1e-6, i.e. <= 0.

### R3 [PROVED]

Pythagorean cost. For a network define:
- atoms beta_i = beta(omega_i) and e_i = B_5 ||omega_i||^3;
- rho_i = (e_i^2 - |beta_i|^2)^{1/2} and R = 2 sum_i rho_i;
- Q = sum_i |beta_i|^r, P_1 = 3^{1-s} Q^s and P = sum_i ||omega_i||^b.
Then 3^{1-s} B_5 P^s >= sqrt(P_1^2 + 3^{2-2s} R^2/4).
Also, the vector of non-own field coefficients phi_j(chi) = E_x Phi_j(x) rho^{-<chi,x>} = sum_i m_i(j; chi - j1) has l2 norm <= R.

**Proof / evidence.**

1. e_i^r = B_5^r ||omega_i||^b, so B_5 P^s = (sum_i e_i^r)^s.
2. Put t = 2/r >= 2. By Minkowski in l^t on R^2: sum_i (|beta_i|^2 + rho_i^2)^{r/2} = sum_i ||(|beta_i|^r, rho_i^r)||_t >= ||(Q, sum rho_i^r)||_t.
3. Raise to the power s (rs = 1): B_5 P^s >= (Q^{2s} + (sum rho_i^r)^{2s})^{1/2}.
4. Since r <= 1, (sum rho_i^r)^s >= sum rho_i = R/2. Multiply by 3^{1-s}.
5. The second claim follows from the triangle inequality over neurons and R2. The map chi -> k = chi - j1 is a bijection.

*Check:* sanity52.out L_C: max over random networks (s = 1.5, 1.25, 1.1) of [sqrt(P_1^2 + cR^2) - 3^{1-s} B_5 P^s] = -0.0062.

### R4 [PROVED]

Margin ledger (certificate lemma) at (5,2). Let theta have beta_2 = 0, with own data A = beta_1 and B, C, D = beta_{1,{1}}, beta_{1,{2}}, beta_{1,{12}}. For any probability q on Z_5^2 x Z_5^* define:
- C_j(x) = sum_delta q(x,delta) c_j(delta), with c_j(delta) = 1 - rho^{-j delta};
- Chat_j(psi) = (1/25) sum_x C_j(x) rho^{-<psi,x>};
- a_0 = 25 Chat_1(0) and a_T = 25 Chat_1(-chi_T), with chi_T = 2*1_T;
- n = 25 ||(Chat_j(-chi)) over non-own (j,chi) != (2,0)||_2.
Then h <= Re(conj(A) a_0) + sum_T Re(conj(beta_{1,T}) a_T) + R n.

**Proof / evidence.**

1. h <= sum q g, because q is a probability.
2. Write g = sum_j Re(Phi_j c_j) and Phi_j = sum_chi phi_j(chi) rho^{<chi,x>}. Then sum q g = 25 Re sum_{j,chi} phi_j(chi) Chat_j(-chi).
3. Own coefficients: phi_1(0) = conj(A), phi_1(chi_T) = conj(beta_{1,T}) (input frequencies chi_t - 1 in {+-1}; paper Lemma 6), and phi_2(0) = conj(beta_2) = 0.
4. The remaining terms are bounded by Cauchy-Schwarz, using the l2 bound R from R3.

*Check:* sanity52.out L_D: max[h - sum q g] = -0.72 and max[sum q g - bound] = -0.087, over random HS certificates and random networks with any phases.

### R5 [PROVED]

Elimination of R. For a real, n >= 0, F > 0 and c > 0: sup over R >= 0 of (a + nR)/sqrt(F^2 + cR^2) <= sqrt(max(a,0)^2/F^2 + n^2/c).
Together with R3 and R4, for beta_2 = 0 and any lower bound F <= P_1: gamma/gamma_IF <= sqrt(max(a,0)^2/F^2 + n^2/c_s)/Gamma_s, where c_s = 3^{2-2s}/4 (equal to 1/12 at s = 3/2).

**Proof / evidence.**

1. a + nR <= (max(a,0)/F) F + (n/sqrt c)(sqrt c R).
2. By Cauchy-Schwarz this is <= sqrt(max(a,0)^2/F^2 + n^2/c) * sqrt(F^2 + cR^2).
3. For the corollary, gamma/gamma_IF = h/(3^{1-s} B_5 P^s Gamma_s), with h <= a + nR (R4) and 3^{1-s} B_5 P^s >= sqrt(F^2 + c_s R^2) (R3).

*Check:* sanity52b.out R4: max[grid sup - closed form] = 0, i.e. <= 0, over 2000 random cases.

### R6 [PROVED]

Hermitian-square certificates and normal form at (5,2).

Setup:
- E = {0,1,2}^2 and v(theta) = (e^{i<e,theta>}) for e in E.
- PSD 9x9 matrices G_delta with sum_delta tr G_delta = 1.
- q(x,delta) = (1/25) v(theta(x))^* G_delta v(theta(x)), where theta(x) = 2pi(2x)/5.

Claims:
- q is a probability.
- The family is equivariant under the torus action (B,C,D) -> (B e^{i t1}, C e^{i t2}, D e^{i(t1+t2)}), acting by G -> D_t^* G D_t with D_t = diag(e^{-i<e,t>}). This action preserves a_0, n, each pairing Re(conj(beta_T) a_T), the moduli |B|, |C|, |D|, and phi_pay = arg(A D conj(B C)).
- Slot swap and conjugation are network symmetries that preserve margin, cost, P_1 and R.

Consequence: it suffices to bound normalised data. These are |A| = 1, A = e^{i alpha} with alpha in [0, pi/2], B = m1 >= C = m2 >= 0 real, and D = m12 e^{i(phi_pay - alpha)}. If Re A <= 0, then h <= 0 by paper Lemma 2.

**Proof / evidence.**

1. q >= 0 because it is a Hermitian form.
2. sum_x v^* G v = sum_{e,e'} G[e,e'] sum_x e^{i<e'-e, theta(x)>} = 25 tr G. This holds because, for f in {-2..2}^2, 2f = 0 mod 5 forces f = 0 (no aliasing).
3. v(theta)^* D_t^* G D_t v(theta) = v(theta - t)^* G v(theta - t). This is still >= 0 on the whole torus, and its Fourier sums pick up the phases e^{-i<f,t>}. So a_T rotates opposite to beta_T and every modulus is unchanged.
4. phi_pay is invariant because D rotates by the product of the rotations of B and C.
5. A certificate found for the normal form, rotated back, certifies the original data with the same value. The payment depends only on (|B|, |C|, |D|, phi_pay).
6. Network symmetries: slot swap maps (A,B,C,D) to (A,C,B,D). The map x -> -x, c -> -c conjugates all four.
7. The bound of R5 is invariant under (A, B, C, D, R) -> lambda(...), which allows |A| = 1.

*Check:* sanity52b.out R7: the rotated certificates give exactly the same bound (0.4047540787) at three angle pairs, and the rotated density stays >= 0 on the grid.

### R7 [PROVED]

Twisted Rotfel'd payment with circle reduction, for s in [1, 3/2]. Notation: beta_i = varrho_i p_i with |p_i| = 1, L = sum varrho_i, alpha = A/|A|, and rho_L = sqrt(2L(L - |A|)). Each chain
  u_i in {(1, p xi, p^2 xi eta), (1, p xi, p xi eta), (1, p eta, p xi eta)}
gives H = sum varrho_i u_i u_i^*. This matrix has diagonal L, two exact entries Y, Z among B, C, D, and one entry X' with |X' - alpha^{+-1} X| <= rho_L. Then:
(a) P_1 >= L M_r(nu(H/L)), the power mean with exponent r of the eigenvalues.
(b) The eigenvalues are 1 + mu_k, where mu^3 - P mu - q = 0, P = (|X'|^2 + |Y|^2 + |Z|^2)/L^2 and q = 2Re(.)/L^3. S_r = sum_k (1 + mu_k)^r is decreasing in P and increasing in q on the feasible set K = {0 <= P <= 3, max(P-1, -2(P/3)^{3/2}) <= q <= 2(P/3)^{3/2}}.
(c) S_min(Pbar, qlow) := min of S_r over K with P <= Pbar and q >= qlow equals S_r(Pbar, max(qlow, Pbar - 1, -2(Pbar/3)^{3/2})). It is +infinity if qlow > 2(Pbar/3)^{3/2}.
(d) The minimum over the disk of X' is at least the minimum over its boundary circle. On the circle, with angle psi:
  Pbar = (mY^2 + mZ^2 + mX^2 + rho_L^2 + 2 mX rho_L cos(psi - phi_pay))/L^2,
  q = 2 mY mZ (mX cos phi_pay + rho_L cos psi)/L^3.
Hence P_1 >= min over L >= Lmin and psi of L (S_min/3)^s, where Lmin = max(|A|, |B|, |C|, |D|).
Also P_1 >= max(|A| - c_1(s) M [Note Thm c1sharp], |A| - sqrt(|B|^2 + |C|^2 + |D|^2)/(2 sqrt 3) [Part I Thm A], 3^{1-s} max(|A|, M)).

**Proof / evidence.**

Entries of H. For the first chain, H_21 = sum varrho p xi = B, H_32 = sum varrho p eta = C and H_31 = sum beta p xi eta = X'. Then |X' - alpha D| <= sum varrho |p - alpha| <= sqrt(L * sum varrho |p - alpha|^2) = sqrt(2L(L - |A|)). The other two chains are the same computation with the uncertain entry next to conj(alpha) C or conj(alpha) B.

(a) Rotfel'd: tr f(sum X_i) <= sum tr f(X_i) for f(t) = t^r concave with f(0) = 0. Here tr f(varrho u u^*) = (3 varrho)^r. So 3^r Q >= tr H^r = L^r sum nu_k^r, i.e. P_1 = 3^{1-s} Q^s >= L((1/3) sum nu_k^r)^{1/r}.

(b) The characteristic polynomial of a zero-diagonal Hermitian 3x3 matrix is mu^3 - P mu - q.
- d mu_k/dq = 1/p'(mu_k) and d mu_k/dP = mu_k/p'(mu_k).
- So dS/dq = r g[mu_1, mu_2, mu_3] (second divided difference) = r g''(xi)/2 > 0, with g = (1+mu)^{r-1} and g'' = (r-1)(r-2)(1+mu)^{r-3} > 0.
- Likewise dS/dP = r h[mu_1, mu_2, mu_3] < 0, with h = mu(1+mu)^{r-1} and h'' = (r-1)(1+mu)^{r-3}(r mu + 2) < 0 for mu >= -1.
- Degenerate (repeated-root) points follow by continuity.

(c) For fixed P, the minimum over q is at q = max(qlow, lowK(P)). The minimum over P is at Pbar, for these reasons:
- In the interior S(P, qlow) decreases in P.
- On the PSD boundary q = P - 1 the eigenvalues are 0 and (3 +- sqrt(4P-3))/2, and S decreases in P.
- On the discriminant boundary q = -2(P/3)^{3/2}, S = 2(1+t)^r + (1-2t)^r with t = sqrt(P/3), which decreases because (1-2t)^{r-1} > (1+t)^{r-1}.
- The two boundary curves cross at P = 3/4. The piecewise argument is continuous at the switch points.

(d) Let X' be interior and let q = 2Re(conj(X') c) + const. Pick d != 0 with Re(conj(X') d) >= 0 and Re(conj(d) c) <= 0. Two closed half-planes through 0 always share a nonzero direction. Move along d to the circle: |X'| does not decrease and q does not increase. So the true (P, q) is admissible for S_min at the circle point.

The circle parametrisation is exact. With phi_pay = arg(Y Z conj(alpha^{+-1} X)) and X' = alpha^{+-1} X + rho e^{i theta}: Re(Y Z conj X') = mX mY mZ cos phi_pay + rho mY mZ cos psi, where psi = arg(YZ) - theta. Then |X'|^2 = mX^2 + rho^2 + 2 mX rho cos(psi - phi_pay).

For L >= 2.5 Lmin we use P_1 >= 3^{1-s} L, since Q^s >= L.

The last three bounds are the Note's theorems (round 1). They were re-checked here numerically through I4 and through endtoend52.

*Check:* sanity52.out:
- L_F: max[twisted bound - P_1] = 1.8e-15 and max[|X' - alpha^{+-1} X| - rho_L] = 2.0e-15, i.e. <= 0 up to rounding, over random atom systems.
- L_S: 0 monotonicity violations out of 15964.
endtoend52.out: P_1 >= F for every network tested.

### R8 [PROVED]

s-monotonicity: b=2 dominates. Fix the own data and R. The ratio bound of R5, computed at s = 3/2 with payment F_{3/2}, dominates the true ratio at every s in [1, 3/2].

**Proof / evidence.**

1. Gamma_s = (5/2) 2^{-s} and c_s = 3^{2-2s}/4 decrease in s.
2. The certificate data a and n do not depend on s.
3. Every payment bound used is non-increasing in s:
- |A| - c_1(s) M, with c_1 = 1 - 3^{1-s} increasing;
- Part I Thm A's bound, which does not depend on s;
- 3^{1-s} max(|A|, M), which decreases;
- L M_r(nu), with M_r increasing in r = 1/s. The minimum over (L, psi) of non-increasing functions is non-increasing.
4. Hence F_{3/2} <= F_s <= P_1(s). The denominator Gamma_s sqrt(F^2 + c_s R^2) is smallest at s = 3/2. The trivial bound kappa_a cos(alpha)/(Gamma F) is handled the same way.

*Check:* The B&B runs only at s = 3/2. endtoend52.out also checks the full chain at s = 1.25 (all ok).

### R9 [PROVED]

Single-feature bound at (5,2), computer-assisted. For every b in [2,3] and every product network theta at (5,2) with beta_2(theta) = 0: gamma(theta) <= 0.995 gamma_IF(b). This covers any width, arbitrary inputs (including DC and multi-frequency content) and outputs on all frequencies.

**Proof / evidence.**

By R6 and R8 it suffices to work at s = 3/2. There we must show, for all normalised data and all R >= 0, that min(trivial, certificate) <= 0.995. Here Gamma = Gamma_{3/2}(5) = 2.5 * 2^{-3/2}, and:
- trivial = kappa_a cos(alpha)/(Gamma F), from h <= kappa_a Re A and cost >= P_1 >= F;
- certificate = sqrt(max(a,0)^2/F^2 + 12 n^2)/Gamma, from R5.

Large interference. If max(m1, m2, m12) >= 1.4, then F >= 1.4/sqrt3 = 0.808 >= THR_I = kappa_a/(0.995 Gamma) = 0.7857, so the trivial bound suffices.

The rest is covered by bb52.py over (m1, m2, m12) in [0,1.4]^3 with m1 >= m2 and phi_pay in [-pi, pi], in 4D boxes.

Payment on each box (pay_adapt.py): F is a rigorous lower bound, the maximum of:
- the three bounds of R7;
- the best of the three chains. Each chain uses (L, psi)-cells in outward-rounded interval arithmetic (iv.py), verified cubic-root enclosures (pay_rig.LB_Smin) and the tail L >= 2.5 Lmin.

Region I. If F >= THR_I, the box is done.

Region II. Otherwise alpha runs over [0, alpha_max(F)]. Here alpha_max is verified so that kappa_a cos(alpha)/(Gamma F) <= 0.995 beyond it. This range is split into cells. On each cell, either the trivial bound passes or a degree-2 HS certificate gives a rigorous bound <= 0.995 on the whole cell (cert_hs.eval_cells, interval cos/sin of alpha and phi). The certificates are built as follows:
- hs2.elim_ratio solves an SDP in floating point at the cell centre.
- cert_hs.rigorise rewrites it as G_delta = sum lam_k u_k u_k^* with float lam_k >= 0, which is exactly PSD.
- The mass, a_0, a_T and n are enclosed with explicit summation error bounds. Dividing by the enclosed mass makes q an exact probability.
- Failing boxes are bisected (widest of the m-widths and the phi-width/3) down to width 2e-3; reaching that width counts as FAIL.

Official run: ./run_certificate.sh, four phi_pay chunks.
- 575 Region I boxes and 1540 Region II boxes.
- 1842 SDP certificates.
- Worst certified Region II bound <= 0.995.
- fail = 0 in every chunk.
Logs cert_l0_7.log, cert_l1_6.log, cert_l2_5.log and cert_l3_4.log end with "DONE ... fail=0". Each chunk took 732-1170 s on one core.

*Check:* endtoend52.out: actual networks with beta_2 = 0 (random, IF-perturbed, single atom plus compensators, locally optimised), at s = 1.5 and 1.25. In every case gamma/gamma_IF <= bound and P_1 >= F. The largest gamma/gamma_IF observed is 0.7818 = kappa_a/Gamma_{3/2}, the single-feature IF block. Numerically no compensation gain appears.

### R10 [PROVED]

(C) at (5,2) for every b in [2,3]. Every global maximiser of the L_{2,b} margin at (p,n) = (5,2) uses exactly 2 = |spt y*| features. This holds for networks of any width, with any inputs and outputs.

**Proof / evidence.**

Apply R1 with the hypothesis supplied by R9 (0.995 < 1). Maximisers satisfy gamma* >= gamma_IF (paper Thm 8).

*Check:* As for R1 and R9. The proof is also written in p52_count_theorem_proof.tex.

### R11 [PROVED]

(7,2): the programme and single features.
(i) For s >= 1, Gamma_s(7) = (7/2) 3^{-s}, with the unique maximiser y* = (3^{-s}, 3^{-s}, 3^{-s}). So K* = 3.
(ii) For every b in [2,3], every network at (7,2) with exactly one feature has gamma <= (3^{2s-1} kappa_min/(7/2)) gamma_IF <= 0.9682 gamma_IF, where kappa_min = 1 - cos(2pi/7).

**Proof / evidence.**

(i) sum_{j=1}^3 kappa_j(delta) = 3 - sum_j cos(2pi j delta/7) = 7/2 for every delta != 0. The minimum over delta is at most the average, (7/6) sum_j y_j, which is <= (7/6) 3^{1-s} by the power mean.
- For s > 1, equality forces y to be uniform.
- For s = 1, equality forces K y = (7/6)1, where K = [[a,b,c],[b,c,a],[c,a,b]] is the matrix of kappa_j(delta). Since det K = -(a+b+c)((a-b)^2 + (b-c)^2 + (c-a)^2)/2 != 0, this gives y = 1/3.

(ii) Take delta with j delta = +-1. Then paper Lemma 2 gives h <= kappa_min Re A <= kappa_min |A|. The single-neuron budget gives B_7 P^s >= Q^s >= L >= |A|. So gamma <= kappa_min B_7, while gamma_IF = 3^{1-s} B_7 (7/2) 3^{-s}. Their ratio 3^{2s-1} kappa_min/3.5 increases in s and equals 0.96817 at s = 3/2.

*Check:* sanity72.out I5: random search reproduces (7/2) 3^{-s} at s = 1, 1.175, 1.5. sanity52b.out R10: maximum ratio 0.96817.

### R12 [PROVED]

delta-Fourier isolation, for every odd p, every n >= 2 and every product network (any width, arbitrary inputs, outputs on every frequency). Let:
- mu be a symmetric probability on Z_p^*, and K_l(mu) = sum_delta mu_delta kappa_l(delta);
- j in F = {1, ..., (p-1)/2} and T a non-empty subset of [n], with chi_{j,T} = 2j 1_T.
Then h <= sum_{l in F} K_l(mu) Re beta_l - (p/2) min_delta mu_delta |beta_{j,T}|.
Exactly: for w in C, q_w(x,delta) = p^{-n}[mu_delta + 2Re(w rho^{j delta} rho^{-<chi_{j,T},x>})] satisfies sum q_w g = sum_l K_l(mu) Re beta_l - p Re(w conj beta_{j,T}). So the test pairs with no Fourier coefficient of any field except the clean ones and feature j's own coefficient at chi_{j,T}.

**Proof / evidence.**

1. g = sum_{l in F} Re(Phi_l c_l); the DC output cancels.
2. So E_x[g rho^{-<chi,x>}] = (1/2) sum_l [phi_l(chi) c_l(delta) + conj(phi_l(-chi) c_l(delta))].
3. sum_delta rho^{j delta} c_l(delta) = sum_{delta != 0} rho^{j delta} - sum_{delta != 0} rho^{(j-l) delta} = -p [l = j]. Also sum_delta rho^{j delta} conj(c_l(delta)) = -p [l = -j] = 0 for l, j in F.
4. Hence sum_delta rho^{j delta} E_x[g rho^{-<chi,x>}] = -(p/2) phi_j(chi).
5. phi_j(chi_{j,T}) = conj(beta_{j,T}), because the input frequencies chi_t - j lie in {+-j}.
6. The mean term is sum_delta mu_delta sum_l Re(conj(beta_l) c_l(delta)) = sum_l K_l Re beta_l, since mu is symmetric.
7. Take w = (min mu/2) beta_{j,T}/|beta_{j,T}|. Then q_w >= 0, sum q_w = 1 (chi != 0) and h <= sum q_w g.

Remark: rigid tests force mass on every delta. So against adversarial phases, several such tests together detect only max_{j,T} |beta_{j,T}| (see R16).

*Check:* sanity72.out (p = 7):
- I1: the identity holds to 2.0e-14 relative, on 300 random networks with DC and multi-frequency inputs, all j and T.
- I2: the aligned test has min q = 6.5e-11 >= 0, and max[h - bound] = -2.06.

### R13 [PROVED]

Cost ledger at (p,2) for several features. For s in [1, 3/2] and every network:
  3^{1-s} B_p P^s >= (sum_{j in F} (F_j A_j)^r)^s,
where:
- A_j = |beta_j| and x_j = max_T |beta_{j,T}|/A_j (features with A_j = 0 are omitted);
- F_j = max(1 - c_1 x_j, 3^{1-s} max(1, x_j)), with c_1 = 1 - 3^{1-s}.

**Proof / evidence.**

1. Single-neuron budget: sum_j |beta_ij|^{2/3} <= e_i^{2/3}, by AM-GM on (|what(j)|^2, |uhat_1(j)|^2, |uhat_2(j)|^2) and Parseval sum_{j in F} |uhat(j)|^2 <= (p/2)||u||^2.
2. Since ||.||_r <= ||.||_{2/3} for r >= 2/3, sum_j Q_j <= sum_i e_i^r = B_p^r P, with Q_j = sum_i |beta_ij|^r.
3. Hence 3^{1-s} B_p P^s >= 3^{1-s} (sum_j Q_j)^s = (sum_j (3^{1-s} Q_j^s)^r)^s.
4. Finally 3^{1-s} Q_j^s >= F_j A_j. This uses Note Thm c1sharp (|A| <= 3^{1-s} Q^s + c_1 max(|B|, |C|, |D|) for all atoms and 1 <= s <= 3/2) and Q_j^s >= L_j >= max(A_j, M_j).

*Check:* sanity72.out:
- I3: budget slack <= 1.7e-16 (equality for IF and single-atom neurons).
- I4: the ledger never exceeds the cost (<= 8.7e-16 relative) at s = 1, 1.1, 1.175, 1.5, on random networks, single atoms, IF blocks and mixtures.

### R14 [PROVED]

(C) at (7,2) for every b in [2.5381, 3], computer-assisted. For s in [1, 1.182] every network at (7,2) with exactly two features has gamma <= 0.9995 gamma_IF (<= 0.999 gamma_IF for s <= 1.175). With R11, every global maximiser uses all 3 = |spt y*| features for b in [3/1.182, 3] = [2.5381, 3]. The Note's dense theorem gave b >= 2.600 (s <= 1.154).

**Proof / evidence.**

Reduction. The relabelling a -> 2a permutes the frequency classes 1 -> 2 -> 3 -> 1 (since 4 = -3 and 6 = -1), so WLOG the features are {1,2}.

Bound. R12, with one test on the largest |beta_{j,T}| and mu chosen per box, together with R13 gives
  gamma/gamma_IF <= [sum_{j=1,2} K_j(mu) A_j - (7/2) min mu * max_j M_j] / [(7/2) 3^{-s} (sum_j (F_j A_j)^r)^s].
This is scale-free, so we may take A_1 + A_2 = 1 and tau = A_2.

Verification by simple72_rig.py:
- Boxes cover (tau, x_1, x_2) in [0,1] x [0,1.75]^2, plus the slabs x_j = infinity. The slabs are valid because F_j >= 3^{1-s} x_j >= 3^{1-s} * 1.75 >= 1 (asserted). Dropping M_j from the max only weakens the bound.
- mu comes from an LP at the box centre and is renormalised exactly by the enclosed sum 2 sum mu_d.
- Every s-dependent quantity, including Gamma_s, kappa and c_1, is enclosed in mpmath interval arithmetic with s ranging over the s-cell. Results are rounded upward.
- Boxes are bisected until the bound is <= TARGET.

Runs:
- s in [1, 1.175], TARGET 0.999: 39,676 boxes, 0 failures (simple72_rig.out).
- s in [1.175, 1.182], s-step 5e-4, TARGET 0.9995: 78,408 boxes, 0 failures (simple72_rig_ext.out).
Each run takes about 2 minutes.

Conclusion. gamma* >= gamma_IF. With zero features h <= 0, with one feature gamma <= 0.9682 gamma_IF, and with two features gamma <= 0.9995 gamma_IF.

*Check:* sanity72.out I6: locally optimised two-feature networks (6 neurons, outputs on frequencies {0,1,2}) at s = 1, 1.1, 1.175 all satisfy gamma/gamma_IF <= the simple bound (max difference -0.015). The best found is 0.813.

### R15 [NUMERICAL]

Which ingredients the (5,2) proof needs, at s = 3/2. Each value below is the supremum over own data of the ratio bound obtained with that set of ingredients.

**Proof / evidence.**

- Free relaxation, with non-own modes uncosted: 1.034 at a single atom. So ledger 2 (the Pythagorean cost) is necessary.
- L2 route with the Note's payments: >= 0.99686 (l2plb_s1.5.out).
- Max-T tests: 1.066 (maxT_s1.5.out).
- Degree-1 HS certificates with the decoupled twisted payment: 0.956.
- Degree 2 with the decoupled twisted payment: 0.919.
- Degree 2 with the circle 3-chain payment: 0.869 (hs_sup4_*.out). This is the combination certified, at 0.995.
- The true single-feature optimum found is kappa_a/Gamma = 0.7818, the IF block.

*Check:* Scripts and outputs are in /home/user/grok/heracles/conjecture20/unify/p52/: free_sup.py, l2plb.py, maxT.py, hs_sup*.py and their .out files.

### R16 [NUMERICAL]

(7,2) landscape and the limit of the phase-blind route.
(a) The two-feature IF value relative to the three-feature IF value, Gamma^{({1,2})}_s/Gamma_s:

| b | 3 | 2.6 | 2.538 | 2.5 | 2.31 | 2.14 | 2 |
|---|---|---|---|---|---|---|---|
| ratio | 0.789 | 0.845 | 0.855 | 0.862 | 0.900 | 0.939 | 0.980 |

At b = 2 the slack is only 2%.
(b) Phase-blind cap. Any bound that uses only (A_j, max_T |beta_{j,T}|), rigid tests and the best phase-blind payment curve stops at s about 1.20, i.e. b about 2.50.
(c) The free relaxation (phase-aware LP, exact atom costs) exceeds 1 for s >= 1.4.

**Proof / evidence.**

(b) The decoupled-twisted curve F_tw(x) (ftw_curve_s1.2.out, ftw_curve_s1.3.out) agrees within 0.006 with the numerically minimised true curve F*(x) = inf{P_1 : |A| = 1, max_T |beta_T| <= x} (fstar_s1.2_m3.out, fstar_s1.2_m5.out). Using F_tw, the bound's supremum is:

| s | 1.18 | 1.2 | 1.22 | 1.25 |
|---|---|---|---|---|
| sup | 0.977 | 1.0006 | 1.024 | 1.061 |

(simple72_tw_s*.out). Adding the Note's quadratic payment changes nothing (simple72q.py).

The worst configuration is:
- A_2/A_1 = 2cos(pi/7). The clean margins then tie at delta = +-1, +-3 (2.579), while delta = +-2 gives 4.648.
- Feature 1 is a single atom (x_1 = 1, payment 3^{1-s}).
- max_j M_j sits at the break-even 1.18 A_1. There a rigid test's gain (7/2) mu_2 M equals its mean-margin cost 2 mu_2 (4.648 - 2.579) = 4.14 mu_2.

(c) free72_*.out: sup >= 0.795 (s = 1.2, 1 atom per feature), 0.876 (s = 1.2, 2 atoms), 0.932 (s = 1.3), 1.086 (s = 1.4), 1.255 (s = 1.5).

*Check:* Values (a) are recomputed by direct maximisation of the restricted programme. The search in (b) is a grid plus Nelder-Mead. The searches in (c) use few restarts, so they are lower bounds on the supremum.

### R17 [PROVED]

Shared-neuron obstruction at b = 2 (p = 7, two features). Take a neuron whose input factors and output are split equally between frequencies 1 and 2, with balanced norms. Then |beta_1|^{2/3} + |beta_2|^{2/3} = e^{2/3}, so the cost budget is tight. Yet its non-own mode mass is exactly 3e^2.
Consequences:
- At b = 2, multi-feature neurons produce non-own modes at zero marginal cost. The Pythagorean ledger R3, which drives the (5,2) proof, has no two-feature analogue.
- At b > 2 the same neuron pays: sum_j |beta_j|^r = 2^{1 - 3r/2} e^r < e^r.

**Proof / evidence.**

Notation: E_j = |what(j)| |uhat_1(j)| |uhat_2(j)|.
1. Equal split gives |what(j)|^2 = (p/4)||w||^2 and similarly for the inputs. With ||w|| = ||u_t|| = ||omega||/sqrt3: |beta_j| = (2/p^3) E_j = 2(12p)^{-3/2} ||omega||^3 = 2^{-3/2} e. So sum_j |beta_j|^{2/3} = e^{2/3}.
2. Total mode mass: sum_{j in F} sum_k |m|^2 = (2/p^3)^2 (p/2)||w||^2 p^2 ||u_1||^2 ||u_2||^2 = (2/(27 p^3)) ||omega||^6 = 4e^2.
3. Own mass: 4|beta_1|^2 + 4|beta_2|^2 = e^2. So the non-own mass is 3e^2.
4. At p = 7 these cross modes reach feature 2's own characters through the F2 channel 4 = -1 - 2 (mod 7).

*Check:* sanity52b.out R13: |b1|^{2/3} + |b2|^{2/3} = 0.793701 = e^{2/3}; non-own mass / e^2 = 3.0000.

## Obstruction
(5,2) is closed for every b in [2,3]. The open case is (7,2) with b in [2, 2.538). Three obstructions are pinned down.

(1) Phase-blind cap at b of about 2.50 (R16b, numerical). Rigid isolated tests need mass on every offset delta. At the tie configuration A_2/A_1 = 2cos(pi/7), the clean margins at delta = +-1, +-3 are equal and much smaller than at +-2. So buying detection costs 4.14 per unit of test mass, against a gain of 3.5 M. A single-atom feature with |B| = |C| = |D| = |A| sits below the break-even M = 1.18|A_1| and is never detected. Any argument that uses only (A_j, max_T |beta_{j,T}|) stops at s of about 1.20. This holds even with the best phase-blind payment curve (decoupled twisted, which matches the true curve within 0.006).

(2) For b <= 2.14 the free relaxation fails (R16c, numerical). With phases and exact atom costs, but with non-own modes free, the bound already exceeds 1 (1.255 at b = 2). So, as at (5,2), the cost of non-own modes must enter.

(3) At b = 2 that cost can be zero (R17, proved). Shared neurons split between frequencies 1 and 2 are budget-tight, yet carry non-own mass 3e^2. Through the F2 relation 4 = -1 - 2 (mod 7) these modes reach feature 2's own characters. The Pythagorean ledger of the (5,2) proof therefore has no two-feature version at b = 2. Only a channel-resolved ledger could work: one that leaves unpriced exactly the cross modes shared neurons can make, and shows the tests can be chosen blind to them.

Combined with GLOBAL's 2% slack at b = 2 (the two-feature IF value is 0.980 gamma_IF), no slack-based argument reaches (7,2) at b = 2.

## Next steps
- (7,2), b in [2.14, 2.54): build a two-feature version of the (5,2) certificate search, restricted to the free relaxation. Use the exact LP dual: q >= 0 is checked exactly on the 49 x 6 grid, with isolated tests at both features' own characters and twisted payments per feature. Torus-normalise one feature. Use the max-structure of rigid tests (R12) to decouple the two features' detections. The search space is about 9-11 dimensional.
- (7,2), b in [2, 2.14): prove a channel-resolved ledger. First classify which non-own modes of multi-feature neurons reach own characters; at p=7 this is only the F2 channel 4 = -1-2 and the own partners. Then choose delta-isolated tests that annihilate those characters, and price every remaining non-own mode Pythagorean-style, as in R3.
- Use R12 (delta-Fourier isolation, valid for every p and n) as the generic detection primitive at larger p. Compute the phase-blind cap of the simple bound at p = 11 and 13 to see how the provable range above the dense threshold scales with p.
- Optional robustness checks: re-run run_certificate.sh with TARGET 0.98 to measure the true (5,2) slack; it is numerically about 0.78 vs 0.995 certified. Separately, port the s=3/2 machinery (pay_adapt, cert_hs) to general r, so the (7,2) payments can use the twisted curve rigorously; this would move the (7,2) threshold from b = 2.538 to about 2.50.
