# known: report (round 2; saved by the orchestrator from the agent's structured output)

## Headline
KNOWN-MATH: I found no existing theorem that implies (C), (MM), decoupling (D), the transition identity, or any of O1-O4. I also found no known false additivity conjecture of which (D) is an instance, so there is no hidden landmine. Three tensor facts are proved, each checked numerically. KM1: the frequency split is the pinching inequality of the l_r quasi-nuclear norm, and b=2 is exactly its threshold. So Strassen/Shitov-type non-additivity cannot occur in the cost; any failure of (D) must come from the margin. KM2: equality in Paper Lemma 7 holds only for the diagonal Fourier configuration xi_lt = c_t omega^l with all beta_l = beta/nu. Orthogonal chain vectors (complex Hadamard matrices) alone are not enough. KM3: the cosine tensor Re(z_1...z_nu) has l_r quasi-nuclear norm nu^{1/r-1} 2^{nu-1} and real rank nu, with a unique l_r-minimal decomposition for r<1. This is the multilinear, cost-weighted form of the Comon-Ottaviani / Causa-Re / Blekherman / Reznick theorem that hyperbolic binary forms have real rank equal to their degree. One import is conditional: Bohnenblust-Hille (BH) combined with LARGEN R3 gives n-free l_{2k/(k+1)} control of the degree-k sign-flipped coefficients. The ledger below says, entry by entry, which known theorems provably cannot supply each missing piece.

## Principle
THE IDENTIFICATION. The problem is "max-margin under an l_r atomic quasi-norm over rank-one tensors (r = b/nu, a Schatten-type quasi-norm for tensors), with a translation-invariant concave objective". Each piece of the machinery then has a known-math counterpart.

(i) Cost side = tensor-norm theory (KM1).
- Seed S2 / SPLIT R1 is the pinching inequality sum_k N_r(Pi_k f)^r <= N_r(f)^r. It holds iff b >= 2.
- At r=1 this is the known additivity of the nuclear norm on direct sums (Kong-Lee-Wang, via Cochrane arXiv:2001.07202).
- As r -> 0 it becomes Strassen's direct-sum conjecture, which Shitov (Acta Math. 2019) showed false.
- In matrices (nu=2) it is the classical pinching inequality, which fails for Schatten-r with r<1, exactly at b<2.
- Consequence: the problem's range b in [2,nu] is precisely where mixed (cross-block) neurons can never represent a block-diagonal table more cheaply. Strassen-type non-additivity is excluded, and (MM)/(D) are purely questions about the objective (the margin).

(ii) Single-feature IF cost = real-rank theory of hyperbolic forms (KM2, KM3).
- Lemma 7 is an l_r strengthening of "real rank of Re(z_1...z_nu) is nu".
- The symmetric shadow is the theorem that a real binary form of degree d has real Waring rank d iff all its roots are real and distinct (Comon-Ottaviani; Causa-Re 2011; Blekherman 2015; Reznick 2013). At nu=3 it is the 2x2x2 negative-hyperdeterminant case (de Silva-Lim 2008; Seigal-Sturmfels 2017).
- The r -> 0 count bound is a Fisher/Delsarte-type design bound: the nu chain functions are orthonormal on the support.
- Rank-nu decompositions form a (nu-1)-parameter family (Sylvester apolarity). The l_r cost with r<1 selects a unique one, the regular polygon, a "tight design" (KM2). Complex-Hadamard freedom (Craigen's F_4(a), Haagerup's classification) is killed by the non-interval (JF) conditions.

(iii) Programme = l_{2/nu} bridge max-margin in the Fourier domain. At b=2, Gamma_s is exactly the Gunasekar-Lee-Soudry-Srebro (NeurIPS 2018) depth-L linear-convolutional implicit bias with L = nu, applied to the coverage vectors kappa(delta). Equivalently it is the best spectral gap of a weighted Cayley graph on Z_p under an l_r budget (Alon-Roichman). That paper gives only stationary points, and nothing about support identification beyond the paper's own Theta(log p).

(iv) Relations core O1 = a coverage-weighted Sidon (anti-flatness) bound on near-optimal Cayley expander generating sets of size about 1.4 ln p. Coverage-free versions are false (Rudin-Shapiro). Known flatness and Sidon theorems are either the wrong object (Chowla), too weak (McGehee-Pigno-Smith / Konyagin L^1 bounds), non-explicit (Pisier), or consistent but undecisive (Rudin's size bound).

(v) Large-n payments = Bohr / Bohnenblust-Hille theory on the polydisc.
- LARGEN R3 is Rogosinski subordination.
- Dimension-free control is available degree by degree in l_{2k/(k+1)} (BH, hypercontractive constants), but not in l_1 over all T (Bohr radius ~ sqrt(log n/n)).

(vi) The temperature identity is the escort-distribution / Renyi-entropy derivative of log ||a||_{1/s} together with Danskin's theorem. It is standard, so it cannot carry the value content of the transition identity.

WHICH CASES THIS REMOVES.
- From the decoupling question: every cost-side mechanism, including Shitov-type straddling decompositions, for all b in [2,nu].
- From the IF analysis: all non-Fourier equality configurations at every nu (in particular nu=4 and nu=6, where non-Fourier complex Hadamard families exist).
- From the search for imports, these whole classes, with reasons: p-free norm or additivity theorems (refuted by p=3); coverage-free flatness theorems (refuted by APs); unweighted-l_1 polydisc payments (Bohr).

## Results

### KM1 [PROVED]

Pinching / additivity of the l_r quasi-nuclear norm, with sharp threshold b=2.

Setting.
- N_r(f)^r = inf { sum_i ||T_i||_F^r : f = sum_i T_i, T_i real rank one }, for r in (0,1] and f in V_1 (x) ... (x) V_nu.
- Each V_t = (+)_{k in K} V_t^(k) is an orthogonal decomposition, and Pi_k = (x)_t P_t^(k) projects onto the k-th diagonal block.

Claims.
(a) If b := r nu >= 2, then sum_k N_r(Pi_k f)^r <= N_r(f)^r for every f. Hence N_r((+)_k A_k)^r = sum_k N_r(A_k)^r: the quasi-norm is additive on direct sums.
(b) If b < 2, and some two blocks are non-zero in every slot, there is a rank-one f with sum_k N_r(Pi_k f)^r = 2^{1-b/2} > 1 >= N_r(f)^r.

Network reading (via seed S1 balancing, and S2):
- For b in [2,nu], no network can realise a given frequency-block-diagonal table more cheaply by using neurons that straddle blocks (mixed modes).
- So the Strassen/Shitov non-additivity mechanism cannot occur in the cost. Any failure of (MM) or (D) must come from the margin objective.

**Proof / evidence.**

Proof of (a).
1. Take any decomposition f = sum_i (x)_t v_it.
2. Then Pi_k f = sum_i (x)_t P_t^(k) v_it, so sum_k N_r(Pi_k f)^r <= sum_i sum_k prod_t ||P_t^(k) v_it||^r.
3. Put a_kt = ||P_t^(k) v_it||. Hoelder with nu factors gives
   sum_k prod_t a_kt^r <= prod_t (sum_k a_kt^{r nu})^{1/nu} <= prod_t (sum_k a_kt^2)^{r/2} = prod_t ||v_it||^r.
   The middle step uses ||x||_q <= ||x||_2 for q = r nu >= 2.
4. Take the infimum over decompositions.
5. For a direct sum, concatenating decompositions gives the reverse inequality.

Proof of (b).
1. Let f = (x)_t (e_t + e'_t)/sqrt2, with e_t, e'_t unit vectors in two different blocks. Then N_r(f) <= 1.
2. The pinched tensor is 2^{-nu/2}((x)e_t + (x)e'_t).
3. Flatten slot 1 against the rest. Rank-one tensors map to rank-one matrices with the same Frobenius norm, so N_r(g) >= N_r^mat(flat g).
4. For matrices, N_r^mat is the Schatten-r quasi-norm, by the r-subadditivity ||sum A_i||_{S_r}^r <= sum ||A_i||_{S_r}^r for 0 < r <= 1 (Rotfel'd 1969; McCarthy 1967; this is the inequality the Paper itself cites via Thompson 1976).
5. Each block piece is rank one with norm 2^{-nu/2}, so the pinched sum of r-th powers equals 2 * 2^{-r nu/2} = 2^{1-b/2} > 1 when b < 2.

Numerical check (/home/user/grok/heracles/conjecture20/unify/knownmath/cosine_tensor_check.py, output cosine_tensor_check.out).
- 20000 random block splittings per case, nu in {2,3,4}.
- The termwise ratio (pinched/original) is at most 1.0000 for b in {2, 2.5, nu}.
- The ratio exceeds 1 for b = 1.5 (1.38, 1.34, 1.22).
- The equal-split example gives exactly 2^{1-b/2}.

Known-math placement.
- r=1: nuclear-norm additivity on direct sums (Kong-Lee-Wang, quoted in R. Cochrane, 'Nuclear norm under tensor Kronecker products', arXiv:2001.07202).
- r -> 0: Strassen's direct-sum conjecture, false (Y. Shitov, Acta Math. 222 (2019) 363-379; revisited in arXiv:2507.17890).
- nu=2: the matrix pinching inequality (Bhatia, Matrix Analysis, Ch. IV), which fails for Schatten r < 1.

What KM1 is not. It is the S2/SPLIT R1 cost claim restated with its sharp threshold and its literature placement. It adds the counterexample for b < 2 and the conclusion that Strassen-type obstructions are absent. It does not prove (MM).

*Check:* cosine_tensor_check.out, section 'KM1 pinching'

### KM2 [PROVED]

Equality case of Paper Lemma 7 (the tight case).

Setting: n >= 1, nu = n+1, 0 < r < 1, beta != 0, and (beta_l, xi_lt) satisfy (JF):
  sum_l beta_l prod_{t in T} xi_lt = beta [T = empty]  for all T in [n],
with equality sum_l |beta_l|^r = nu^{1-r} |beta|^r.

Claim: after relabelling l = 0..n,
- beta_l = beta/nu for every l (equal phases, not only equal moduli), and
- xi_lt = c_t omega^l, with omega = e^{2 pi i/nu} and |c_t| = 1. The same root of unity appears in every slot.
Conversely, these configurations attain equality.

Network reading: up to independent rotations of the input slots, with the compensating output rotation, every cost-optimal interference-free feature is the Paper's Prop. 9 block.

The Paper's characterisation by 'pairwise orthogonal chain vectors' is correct only together with (JF). Orthogonality alone allows every dephased complex Hadamard matrix, for example Craigen's family F_4(a) at nu=4, and these violate (JF) at the non-interval set T={1,3}.

**Proof / evidence.**

Starting point. By Lemma 7's equality clause, exactly nu of the beta_l are non-zero, each has |beta_l| = |beta|/nu, and the vectors u_l = m_l/sqrt(nu) are orthonormal. So U = [u_0 ... u_n] is unitary. Normalise beta = 1 and put d_l = nu beta_l, so |d_l| = 1.

(i) Equal phases.
- H = sum_l d_l u_l u_l^* = U D U^* is a normal matrix.
- By the chain lemma, H is upper triangular with unit diagonal.
- A normal triangular matrix is diagonal, because ||H||_F^2 = sum |lambda_i|^2 = sum |H_ii|^2.
- So H = I and D = I, i.e. every beta_l = beta/nu.

(ii) Generating identity. (JF) now reads sum_l prod_{t in T} xi_lt = nu [T = empty]. Equivalently,
  sum_l prod_{t=1}^n (1 + z_t xi_lt) = nu   for all z in C^n.

(iii) Killing all terms but one.
- Fix l_0 and a bijection sigma: [n] -> [nu] minus {l_0}. Set z_t = -conj(xi_{sigma(t),t}).
- Every term with l != l_0 contains the factor 1 - 1 = 0. Hence prod_t w_{sigma(t),t} = nu, where w_lt := 1 - xi_{l_0 t}/xi_lt.
- Every w_lt is therefore non-zero.
- For n=1 the identity sum_l xi_l = 0 with two points already gives xi_1 = -xi_0.

(iv) Rank one (n >= 2). Swapping sigma on two slots gives w_At w_Bu = w_Bt w_Au. So the n x n matrix (w_lt) has rank one: w_lt = alpha_l gamma_t, with all entries non-zero.

(v) All gamma_t are equal.
- Since |xi| = 1, we have |1 - alpha_l gamma_t| = 1. So each gamma_t != 0 lies on every circle |z - 1/alpha_l| = 1/|alpha_l|, and all these circles pass through 0.
- If all alpha_l were equal, two points xi_A and xi_B would coincide. Then m_A = m_B, which contradicts orthogonality.
- Two distinct circles through 0 meet in at most one other point. So gamma_t = gamma for every t.
- Hence xi_lt = xi_{l_0 t} zeta_l, with zeta_l = (1 - alpha_l gamma)^{-1} independent of t and zeta_{l_0} = 1.

(vi) Roots of unity.
- (JF) gives prod_{t in T} xi_{l_0 t} * sum_l zeta_l^{|T|} = 0. So the power sums p_k(zeta) vanish for k = 1..nu-1.
- By Newton's identities, e_1 = ... = e_{nu-1} = 0. So the zeta_l are the nu roots of z^nu - c.
- zeta_{l_0} = 1 forces c = 1, so {zeta_l} = mu_nu.

Converse. For 1 <= |T| <= n < nu, sum_l omega^{l|T|} = 0, and the chain vectors m_l are rows of a column-twisted Fourier matrix.

Numerical checks.
- /home/user/grok/heracles/conjecture20/unify/knownmath/tight_designs.py (output tight_designs.out). Random-start solves of the 2^n - 1 conditions with nu points, for n = 2..5: 300, 300, 300 and 142 solutions. None is off the roots of unity; the maximum deviation is 1.8e-15.
- /home/user/grok/heracles/conjecture20/unify/knownmath/tight_diag_check.py (output tight_diag_check.out), for n = 2..5:
  - the diagonal structure holds to 2.3e-15 and |zeta^nu - 1| <= 6.8e-15;
  - the key identity prod_t (1 - xi_{l_0 t}/xi_{sigma(t) t}) = nu holds for every l_0 and sigma to 1.8e-14;
  - Craigen F_4(a) has orthogonal chain vectors but violates (JF) by 1.18, 3.37 and 4.0 at a = 0.3, 1.0 and pi/2. The violation is 0 only at a = 0, the Fourier case.

Known-math placement.
- Tight designs: nu points with nu orthonormal chain functions, a Fisher/Delsarte bound; the 'tight design is a regular polygon' pattern.
- Complex Hadamard matrices: Haagerup 1996 classifies them up to order 5; Craigen's continuous family exists at order 4; Tadej-Zyczkowski catalogue.
- Uniqueness of the cost-minimal (l_r, r<1) decomposition among the (nu-1)-parameter family of rank-nu real decompositions given by Sylvester apolarity.

*Check:* tight_designs.out, tight_diag_check.out

### KM3 [PROVED]

The l_r quasi-nuclear norm of the cosine tensor.

Let C_nu in (R^2)^{(x) nu} be the coefficient tensor of cos(theta_1 + ... + theta_nu) = Re prod_t (cos theta_t + i sin theta_t). Here nu = n+1 >= 2, and the output slot is just another slot. Then:
(a) N_r(C_nu)^r = nu^{1-r} 2^{(nu-1) r} for every r in (0,1], in any ambient spaces that contain the planes isometrically. In particular:
  - ||C_nu||_* = 2^{nu-1} = ||C_nu||_F^2 / ||C_nu||_sigma;
  - the real tensor rank of C_nu is exactly nu.
(b) For r < 1, the minimisers are exactly the nu-term equally spaced decompositions C_nu = (2^{nu-1}/nu) sum_{l=0}^n (x)_t (cos phi_lt, -sin phi_lt), with phi_lt = phi_t + pi l/nu for t <= n and phi_{l,nu} = phi_nu - n pi l/nu, where sum_t phi_t = 0, up to sign flips of pairs of factors.

Network reading: this is Paper Lemma 7 together with KM2, in tensor language. A clean feature with no interference has cost at least nu^{1-r} times its nuclear cost, needs nu neurons, and is unique for r < 1.

**Proof / evidence.**

Proof.
1. Reduction to the planes. Projecting each factor onto its plane is a product of contractions. It fixes C_nu and does not increase ||T_i|| (it strictly decreases a term with an out-of-plane part). So it suffices to treat in-plane decompositions, and minimisers lie in-plane.
2. Translation to (JF). For T_i = (x)_t v_it, write z_it = v_it1 + i v_it2, beta_i = prod_t conj(z_it) (so |beta_i| = ||T_i||_F), and xi_it = z_it/conj(z_it).
3. Expand prod_t Re(conj(z_it) e^{i theta_t}) in the characters e^{i sum_t eps_t theta_t}, and match against Re e^{i sum_t theta_t}. Conjugation pairs eps with -eps, so one may fix eps_nu = +1. With T = {t <= n : eps_t = -1}, the matching conditions are exactly
   sum_i beta_i prod_{t in T} xi_it = 2^{nu-1} [T = empty].
4. Lemma 7 then gives the lower bound in (a). Its coefficient bound e_nu(x) >= 1 forces at least nu non-zero terms.
5. Attainment by the block: each eps other than the all-ones vector (up to sign) carries a phase omega^{+-l m} with 1 <= m <= n, and these sum to 0 over l.
6. An independent nuclear-norm check: ||C_nu||_sigma = max Re prod z_t = 1 and ||C_nu||_F^2 = 2^{nu-1} give ||C_nu||_* >= 2^{nu-1}.
7. Uniqueness for r < 1 is KM2.

Numerical check (/home/user/grok/heracles/conjecture20/unify/knownmath/cosine_tensor_check.py, output cosine_tensor_check.out).
- The block reproduces C_nu to 6e-15 for nu = 2..6.
- ||C||_F^2 = 2^{nu-1}, and the sampled spectral norm is 1.000000.
- 40 random exact decompositions with nu to nu+3 terms, for nu = 3, 4 and r in {2/nu, 0.75, 1}: every cost/bound ratio is >= 1. The minima are 1.0616, 1.0075, 1.0013 (nu=3) and 1.0415, 1.1621, 1.0309 (nu=4).

Known-math placement.
- Symmetric shadow: Re((x+iy)^nu) is a hyperbolic binary form (all roots real and distinct), so its real Waring rank is nu:
  - Comon-Ottaviani, Linear Multilinear Algebra 60 (2012), arXiv:0909.4865;
  - Causa-Re, Annali di Matematica 190 (2011), arXiv:1006.5127;
  - Blekherman, Found. Comput. Math. 15 (2015), arXiv:1205.3257;
  - Reznick, 'On the length of binary forms' (2013).
- nu=3: a 2x2x2 tensor with negative hyperdeterminant has real rank 3 and complex rank 2 (de Silva-Lim, SIMAX 30 (2008); Seigal-Sturmfels, arXiv:1609.09245).
- Friedland-Lim (Math. Comp. 87 (2018)): symmetric nuclear norm = nuclear norm (a Banach-type theorem). At r = 1 this makes slot-symmetric decompositions optimal; for r < 1 KM2 gives the analogous uniqueness.
- Mohlenkamp-Monzon's sin(sum) identity is the attaining decomposition.
- I found no published statement of the real rank of Re(z_1...z_d) for d >= 4, or of any l_r (r < 1) minimal-cost statement. Lemma 7 together with KM2 appears to be new in that form.

*Check:* cosine_tensor_check.out, sections 'KM3 identity' and 'KM3 lower bound'

### KM4 [CONDITIONAL]

Dimension-free degree-k detection from the cone margin (import of the Bohnenblust-Hille inequality).

Hypotheses:
(H1) LARGEN R3 (slice subordination). I re-derived it; see the evidence field.
(H2) The polynomial Bohnenblust-Hille inequality: for every k-homogeneous polynomial on C^n,
  (sum_alpha |c_alpha|^{2k/(k+1)})^{(k+1)/(2k)} <= BH_k sup_{D^n} |P|,
with BH_k independent of n and BH_k <= C^{sqrt(k log k)}. Equivalently, BH_k <= C_eps (1+eps)^k for every eps > 0.

Setting: a class C single-frequency single feature with field polynomial P(z) = sum_i beta_i prod_t (1 + xi_it z_t), Phi(x) in c + K for every x, and A = sum_i beta_i.

Claim: for every k <= n,
  ( sum_{|T|=k} |beta_T|^{2k/(k+1)} )^{(k+1)/(2k)} <= BH_k (4/p) |A - c| / cos(pi/p)^k,
uniformly in n and in the width.

Reading: the cone margin controls the whole degree-k layer of sign-flipped coefficients in l_{2k/(k+1)}, with no loss in n. This is strictly stronger than the pointwise (max_T) detection used in LARGEN R8b.

What Bohr theory forbids: an unweighted l_1 (sum over all T) detection at a fixed polydisc radius. The Bohr radius of D^n is ~ sqrt(log n/n), and LARGEN R8a is a degree-n instance of that loss.

**Proof / evidence.**

Proof.
1. With S = [n], LARGEN R3 bounds the k-homogeneous part H_k(zeta) = sum_{|T|=k} beta_T zeta^T at every zeta in T^n:
   cos(pi/p)^k |H_k(zeta)| <= (4/p) |A - c|.
2. By the maximum principle, sup over D^n of |H_k| equals sup over T^n.
3. Apply (H2) to H_k.

Re-derivation of R3 (checked here).
- P is multi-affine. Averaging over product measures gives P(conv(mu_p)^n) in c + K, and conv(mu_p) contains the disc of radius cos(pi/p).
- The slice f(w) = P(R zeta w) has f(0) = A, and f maps the unit disc into the convex sector {|arg(z - c)| <= pi/p}. (Note: cm(z) >= 0 means |arg z| <= pi/p.)
- So f is subordinate to G(w) = c' + M(w)^{2/p}, where M(w) = (beta + conj(beta) w)/(1 - w) maps the disc onto a half-plane and beta = (A - c')^{p/2}.
- |G'(0)| = (2/p)|beta|^{2/p - 1} 2 Re(beta) <= (4/p)|A - c'|.
- Rogosinski's theorem for convex univalent G (Duren, Univalent Functions, Thm 6.4) bounds each coefficient: |f_k| <= |G'(0)|. Let c' increase to c.

References.
- Defant, Frerick, Ortega-Cerda, Ounaies, Seip, Annals of Math. 174 (2011) 485-497 (hypercontractive BH).
- Bayart, Pellegrino, Seoane-Sepulveda, Adv. Math. 264 (2014), arXiv:1310.2834 (subexponential BH; Bohr radius ~ sqrt(log n/n)).
- Defant-Mastylo-Perez for Boolean-cube analogues.
- Multilinear and tetrahedral BH constants grow only polynomially in k, which may give a better constant for multi-affine P.

No separate numerical check was run: the step from R3 is the cited theorem, and LARGEN's own check of R3 (max LHS/RHS = 0.70 over 198 networks) covers the input.

Assessment.
- This gives an n-free currency for LARGEN's missing payment.
- It does not by itself bound the gain |A| - nu^{1-s}Q^s. That would need a robust Lemma 7 in which the gain is charged to the interval coefficients beta_T (the only ones entering the chain-lemma matrix H), measured in l_{2k/(k+1)} per degree.

*Check:* inherits LARGEN R3 check (theory_checks.py (B)); R3 proof re-derived here

### L1-additivity [REFUTED]

Refuted proposition: '(MM), or decoupling (D), is implied by a known additivity or symmetrisation theorem for tensor norms or ranks (Strassen-type, Friedland-Lim/Banach-type, nuclear-norm additivity), or is an instance of a known false additivity conjecture.'

Ledger:
(1) Strassen's direct-sum conjecture for rank: false (Shitov, Acta Math. 2019; https://projecteuclid.org/journals/acta-mathematica/volume-222/issue-2/Counterexamples-to-Strassens-direct-sum-conjecture/10.4310/ACTA.2019.v222.n2.a3.full). This is the r -> 0 limit of N_r.
(2) Nuclear-norm additivity on direct sums: true (Kong-Lee-Wang; Cochrane, arXiv:2001.07202). This is r = 1.
(3) Friedland-Lim, 'Nuclear norm of higher-order tensors', Math. Comp. 87 (2018), https://www.ams.org/mcom/2018-87-311/S0025-5718-2017-03239-7/: symmetric nuclear norm = nuclear norm, a Banach-type theorem. Nie, 'Symmetric tensor nuclear norms', arXiv:1605.08823. Derksen, FoCM 2016.
(4) Comon's conjecture (symmetric rank = rank): false (Shitov, SIAM J. Appl. Algebra Geom. 2018).
(5) Additivity of minimum output entropy: false (Hastings, Nature Physics 2009). This is the closest structural analogue: additivity of an optimum value in a max-min problem, where cross-block (entangled / mixed) strategies may help.

**Proof / evidence.**

Two independent reasons.

(a) Cost-side theorems do not apply to what is open. KM1 proves that the cost is additive and pinching-monotone for all b in [2,nu]. So the Shitov mechanism (cheaper decompositions that straddle blocks) is impossible here, and (1)-(4) say nothing more about (MM) or (D), which concern the margin.

(b) No p-free theorem can imply (MM).
- At p=3 the certified counterexamples are networks in which mixed (DC) modes raise the margin above that of the split network (BRIEF S3; Disproof; Note Table 2).
- So (MM) is false at p=3. Any theorem that uses only the tensor and norm structure, and not Z_p arithmetic with p >= 5, would also apply at p=3, and is therefore refuted.
- Likewise, SPLIT R10 (certified) refutes decoupling for all networks at (5,2,2). Any import must use global optimality.

Assessment of (5). Decoupling has Hastings shape: additivity that fails for some non-optimal networks (SPLIT R10) and is conjectured only at the optimum. Hastings' violations need high-dimensional random constructions. The analogous risk here is a large-p mechanism using the 2|F|-1 mixed channels. Numerics up to p=11 (GLOBAL R11, R12) show exactly zero compensation gain. No theorem either way was found. Decoupling is not an instance of any known false conjecture.

*Check:* KM1 checks; p=3 counterexample certificates (heracles/conjecture20/exact_certificate.py); SPLIT R10 certificate

### L2-flatness [REFUTED]

Refuted proposition: 'a known flat-polynomial, Sidon-constant or Chowla-type theorem that does not use coverage (spectral gap) implies the relations core O1.'

Ledger (closest results):
(1) Chowla cosine problem: min_x sum_{a in A} cos(a x) <= -|A|^{c}.
  - Bedert, arXiv:2509.05260: exponent 1/7 - o(1) in v2; the current abstract states 1/5 - o(1).
  - Jin-Milojevic-Tomon-Zhang, arXiv:2509.03490: c about 1/10, via small eigenvalues of graphs.
(2) Flat Littlewood polynomials exist (Balister-Bollobas-Morris-Sahasrabudhe-Tiba, Annals 2020). Rudin-Shapiro polynomials have sup <= sqrt2 ||b||_2 on arithmetic progressions.
(3) L^1 bounds:
  - McGehee-Pigno-Smith / Konyagin (Annals 1981): ||sum c_k e(a_k x)||_1 >= c sum |c_k|/k.
  - Green-Konyagin (Canad. J. Math. 2009): Littlewood's problem modulo a prime, for dense sets.
(4) Pisier (1983): S is Sidon iff every finite subset has a proportional quasi-independent subset, with non-explicit constants. Rudin (1960): a Sidon set with constant S in a finite group G has |S| = O(S^2 log |G|). I stated this from memory, without tracking the constant.
(5) Salem-Zygmund / Sudakov minoration: for random phases, sup ~ ||b||_2 sqrt(log p).

**Proof / evidence.**

Counterexample to every coverage-free version of O1.
- Take S = {1, ..., K} in Z_p, with K a power of 2 and K << p, and Rudin-Shapiro signs b_l = +-1 (allowed in O1, where phases are free).
- Then osc <= 2 sup <= 2 sqrt(2K).
- Choose delta with K delta/p about 1/2. Then sum_l b_l s_l(delta) = sum_{l <= K} |sin(pi l delta/p)| >= c K.
- So sigma_delta(S) <= sqrt(2K)/(cK) -> 0. Any theorem giving sigma(S) >= c_0 > 0 from |S| or arithmetic alone is false.

Why the listed results cannot supply O1.
- O1 is a lower bound that must use the efficiency/coverage of the support (Note Prop. effsidon; 'thin/costly/blind' trichotomy). Arithmetic progressions are poor covers, which is exactly why they escape.
- (1) concerns unit coefficients with zero phases, and its interesting side is the minimum, whereas O1 needs the oscillation for arbitrary phases. With zero phases the oscillation is trivially >= K.
- (3) gives at most order log K * (average b), short of the required order K * (average b) by a factor K/log K.
- (4) gives non-explicit constants, far above the needed 1/0.367.
- Rudin's size bound is consistent with O1: K* ~ 1.4 ln p against C S^2 ln p, with S about 4. It does not decide it.
- (5): at K ~ log p, random phases already give sup of order sum b. So O1 is precisely a deterministic Sudakov minoration for every phase choice, under the hypothesis that the points of Z_p are separated in the coverage metric d(delta)^2 = sum_l b_l kappa_l(delta). I found no such theorem.

The Note's Laplace identity is the heat-kernel return probability of the weighted Cayley graph (continuous-time random walk). Spectral gap -> return probability is classical, but it is moduli-only, and the Note already shows that moduli-only arguments provably fall short under Theorem A's constant.

*Check:* Rudin-Shapiro bound sup <= sqrt(2K) is classical; the delta choice is elementary

### L3-largen-bohr [REFUTED]

Refuted proposition: 'a polydisc test at a fixed radius R < 1 (seed S4), or any l_1-over-T payment, gives an n-uniform single-feature payment.'

Ledger:
- Bohr radius of D^n ~ sqrt(log n / n) (Bayart-Pellegrino-Seoane-Sepulveda, Adv. Math. 2014, arXiv:1310.2834; earlier bounds of Boas-Khavinson 1997).
- Hypercontractive / subexponential Bohnenblust-Hille (Defant-Frerick-Ortega-Cerda-Ounaies-Seip, Annals 2011).
- Rogosinski subordination (Duren, Thm 6.4).
- Bohr phenomenon for subordination into convex domains (Aizenberg; Bohr-Rogosinski radii, arXiv:2004.08895).

**Proof / evidence.**

- LARGEN R8a, which I checked as stated: the explicit family P = A(1 + z_1 ... z_n) has gain A(1 - (n/nu)^{s-1}) and sup_{D_R^n} |P - A| = A R^n. So the payment constant is at least 0.28 * 2^n at R = 1/2. A top-degree monomial is invisible at radius R < 1.
- More generally, the Bohr radius theorem says sup-norm control of sum |c_alpha| R^{|alpha|} holds uniformly only for R of order sqrt(log n/n). So unweighted l_1 currencies are excluded.
- The admissible n-free currencies are the slice-wise / homogeneous ones: Rogosinski on slices (LARGEN R3) and Bohnenblust-Hille per degree (KM4).

*Check:* LARGEN payment_consts.py numbers (4.6, 86.5, 2.4e4 at n=4, 8, 16) match 0.28*2^n growth

### L4-temperature-transition [REFUTED]

Refuted proposition: 'the transition identity (GLOBAL R7) follows from envelope / thermodynamic formalism (Danskin, escort distributions, log-convexity of l_q norms).'

Ledger:
- d/ds [s log sum_i a_i^{1/s}] = Shannon entropy of the escort distribution. This is the Renyi / free-energy derivative (Beck-Schloegl, Thermodynamics of Chaotic Systems, 1993).
- Lyapunov / Riesz-Thorin log-convexity of norms.
- Danskin's theorem (Bonnans-Shapiro, Prop. 4.12).
- GLOBAL R1 and R2 are exactly these.

**Proof / evidence.**

- The transition identity W_{K<}(s_c) = gamma*(s_c) equates the optimal values of two non-convex network classes at one parameter value.
- Envelope theory constrains only one-sided derivatives of an upper envelope of concave curves. Two different families of curves can cross at s_c without any identity between their sup-values: generic upper envelopes switch branches at crossings that the programme does not predict.
- So the identity carries value content beyond R1/R2. GLOBAL R7 itself shows it is equivalent to (C) near s_c, which is open.
- No parametric-optimisation theorem relating the switching points of two different optimisation problems was found.

*Check:* logical; GLOBAL R11 numerics are consistent with the identity at six transitions

### L5-grokking-literature [NUMERICAL]

Status of (C) in the machine-learning literature: open. Every published max-margin result found concerns the convex end b = nu, or training dynamics, or sufficiency rather than optimality.

**Proof / evidence.**

Literature search results (sources in the ledger):
- Morwani-Edelman-Oncescu-Zhao-Kakade, ICLR 2024 (arXiv:2311.07568): L_{2,nu} margin (b = nu); the max-margin solution uses all frequencies.
- Li-Liang-Shi-Song-Zhou, AISTATS 2025 (arXiv:2402.09469): L_{2,k+1} margin for k inputs, again b = nu; width >= 2^{2k-2}(p-1).
- McCracken et al. 2025 (arXiv:2505.18266): O(log p) frequencies suffice for a large margin. This is sufficiency, not optimality, and they also restate that the max-margin solution uses all frequencies at b = nu.
- He et al. 2026 (arXiv:2602.16849): gradient-flow dynamics, a 'diversification condition', and weight-decay sparsification. These are empirical or dynamical statements, not characterisations of optima.
- arXiv:2608.04451 (2026): a single ReLU neuron freezes with Fourier energy spread over all frequencies. Dynamics only.
- Gunasekar-Lee-Soudry-Srebro, NeurIPS 2018 (arXiv:1806.00468): depth-L linear convolutional networks give an l_{2/L} bridge penalty in the frequency domain. Taking the abstract's description, together with my recollection that the theorem identifies first-order stationary points only (I could not confirm this from the page), at b = 2 this is exactly the programme Gamma_s with L = nu. It gives no global or support statement.
- Dai-Karzand-Srebro, NeurIPS 2021: representation costs; Schatten / l_p quasi-norms.
- arXiv:2607.05546 (2026): depth-L ReLU representation cost is a 2/L-quasi-norm; the native space is quasi-Banach.

No paper proves a feature count at b < nu.

*Check:* web searches 2026-10-18 (arXiv abstract pages fetched for 2001.07202, 2509.05260, 1806.00468, 2602.16849, 2402.09469, 2608.04451)

## Obstruction
Why no import closes (C).

1. Cost side.
- Every known tensor theorem that bears on decoupling concerns the cost: rank or norm additivity, symmetric = general, pinching.
- KM1 shows the cost side is already settled throughout b in [2,nu]: pinching holds and Strassen-type straddling gains are impossible.
- What remains, (MM) and (D), concerns the margin, a min over (x,delta) of linear functionals, under a non-convex budget.
- Such statements are false at p=3 (mixed DC modes help) and false for non-optimal networks (SPLIT R10). So any valid theorem must use p >= 5 arithmetic and global optimality.
- No theorem in tensor-norm theory, real algebraic geometry or quantum additivity has that form. The nearest analogue, minimum-output-entropy additivity, is of exactly this optimum-only type, and it is false in high dimension. It is a structural warning, not an instance.

2. Relations core O1.
- O1 is a coverage-conditional anti-flatness (deterministic Sudakov) statement for supports of size about 1.4 ln p.
- At that size, random phases already achieve the Sidon scale, so the question is purely whether special phases can flatten an efficient cover.
- Known flatness results go the wrong way (flat polynomials exist on progressions). Known anti-flatness results are either the wrong object (Chowla: unit coefficients, zero phases, the minimum rather than the oscillation), too weak (MPS/Konyagin: a log K L^1 floor), or non-explicit (Pisier).

3. Large n.
- Bohr theory proves that unweighted l_1 payments cannot be n-uniform.
- The n-free currencies are slice/degree-wise: Rogosinski, Bohnenblust-Hille (KM4).
- No known inequality converts degree-wise l_{2k/(k+1)} control of sign-flipped coefficients into a bound on the gain |A| - nu^{1-s}Q^s. That is a robust (stability) version of Lemma 7, which is not in the literature.

4. Transition identity. It is a value equality between two non-convex classes. Envelope and thermodynamic tools (GLOBAL R1, R2) are standard and provably carry no such content.

## Next steps
- Robust Lemma 7 (most promising, and uses KM2). The chain-lemma matrix H = sum_l beta_l m_l m_l^* sees only INTERVAL sign-flipped coefficients. Its below-diagonal entries are interval sums beta_{[t'+1,t]}. So det(I+tH) = (1+t)^nu + (terms in interval coefficients), and the integration step gives sum_l |beta_l|^r >= nu^{1-r}|beta|^r - (an explicit function of the interval beta_T). Combined with KM4, which controls each degree layer in l_{2k/(k+1)} uniformly in n, and with freedom to choose the slot order (nu! chains), this is a concrete route to the n-uniform payment LARGEN needs. KM2's rigidity (a unique equality point) should give a quadratic stability constant near IF.
- For O1: formulate and test numerically a 'coverage-metric Sudakov' lemma. If every pair of points of Z_p is at distance >= eps in d_b(delta)^2 = sum_l b_l kappa_l(delta), is inf over phases of osc_w sum_l b_l cos(2 pi l w/p + psi_l) >= c eps sqrt(log p)? Test it on programme supports (relations3/supports.json) and on AP-plus-random hybrids. A counterexample would show that O1 needs more than coverage, namely efficiency of the allocation itself.
- Use KM1 to restate (MM) cleanly as 'pinching onto frequency blocks never lowers the optimal margin'. Search for an (x,delta)-test-measure proof that uses only the 2|F|-1 channels at p >= 5. Any such proof must fail at p=3, which is a useful filter for candidate arguments.
- Record KM2 in the Paper. The equality clause of Lemma 7 should read: beta_l = beta/nu and xi_lt = c_t omega^l. 'Orthogonal chain vectors' alone admits Craigen's F_4(a), which violates (JF). The commentary after Prop. 9 ('the doubled phases are the nu-th roots of unity, the equality case') is then a theorem, not an observation.
- If a reviewer asks about the real rank of Re(z_1...z_d) for d >= 4, cite KM3 as following from Lemma 7. The d=3 case is classical (de Silva-Lim; Seigal-Sturmfels), and the symmetric case is the hyperbolic-binary-form theorem (Comon-Ottaviani, Causa-Re, Blekherman, Reznick).
