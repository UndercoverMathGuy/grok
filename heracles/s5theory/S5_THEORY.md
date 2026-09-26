# Feature counts on finite groups: the gated-chain theory ported from Z_p to S5, S4, A5, D_n and composite abelian groups

heracles/s5theory, 2026-09-23. Local CPU, fp64, no pods. New files only (this directory). No background jobs left running.
Companion scripts: `gt_core.py`, `verify_fourier.py`, `programme.py`, `qscan.py`, `robust_cert.py`, `construct_verify.py`,
`kappa_mm.py`, `prog_large.py`, `complex_closure.py`, `junk_ascent.py`, `junk_nu4.py`, `junk_a5_nu4.py`, `junk_s5.py`,
`junk_abelian.py`, `counterexample_d3.py`, `coset_construct.py`, `coset_check.py`, `coset_trained.py`, `rev_check.py`,
`analyse_junk.py`, `analyse_sectors.py`, `anat_trained.py`, `trained_check.py`; outputs in `out/`.

## 0. Status at a glance

| # | statement | status | where |
|---|---|---|---|
| 1 | **Master bound (Theorem G1)**: every gated chain on the full n-operand table of any finite group G, every width, every q in [2, nu]: gamma_q <= (N nu)^{-nu/2} Gamma(c^unres). Own-share form (G1'). | **PROVEN** (hand proof below; every identity fp64-checked on S4, S5, A5, D_7, n = 2, 3, rel. err <= 6e-14) | §2, `verify_fourier.py` |
| 1b | Real vs complex type: all irreps of S5, S4, A5, D_n are real type (Frobenius–Schur indicator +1). | **COMPUTED** (exact characters) | §2.6 |
| 2 | **Vertex lemma**: the programme is solved exactly by enumerating the vertices of a polyhedron that does not depend on the prices; every maximiser is a vertex, so "the feature count" = support size of the optimal vertex. | **PROVEN** | §3 |
| 3 | **Symmetric-class theorem (G2)** with irrep price kappa(d); Schur: the only invariant in the (rho, ..., rho) block is the chain-trace (iterated matrix-multiplication) tensor. | **PROVEN given kappa**; attained (fp64, 8 constructions, ratio 1 to 3e-15) | §4, `construct_verify.py` |
| 4 | **kappa at EVEN nu: kappa(d) = d^{nu(1-r)} exactly** (schoolbook is optimal; each d-dim feature uses exactly d^nu neurons). | **PROVEN** (flattening lemma) | §5 (Theorem K2) |
| 4b | kappa at ODD nu: d^{(nu-1)(1-r)} <= kappa(d) <= d^{nu(1-r)}; at nu = 3, q = 2: kappa(d) in [d^{2/3}, d]. | **PROVEN** (bounds) | §5 (K1, K3) |
| 4c | kappa at nu = 3 equals the schoolbook value d^{3(1-r)} (q = 2: d = 2, 3, [4]; q = 2.5: d = 2, 3). Strassen's rank-7 orbit costs exactly 10 > 8 at q = 2. | **COMPUTED** (no counter-example in every converged run; upper bounds only) | §5.4, `kappa_mm.py` |
| 5 | **S5, even nu (exact)**: the symmetric-class max-margin chain uses exactly ONE irrep, std_sign (4-dim), no sign patch, for q in [2, ~0.9 nu); runner-up std + sign (sign amplitude exactly 1/8) at 0.978 (nu 4) / 0.998 (nu 6). | **PROVEN** (exact rational vertex enumeration) | §6.1 |
| 5b | **S5, nu = 3, q = 2 (the GX bil family): the COUNT is kappa-free** — for every kappa allowed by K1–K3 the optimum is a single irrep with no sign patch. The identity is std_sign iff kappa_5 >= 0.9691 kappa_4 and kappa_6 >= 0.9371 kappa_4 (true under 4c). | count **PROVEN** (rigorous box certificate); identity **CONDITIONAL** (numerically supported) | §6.1, `robust_cert.py` |
| 5c | S4 / A5 at nu = 3, q = 2: count is NOT kappa-free on the proven box (S4: 2 vs 3 irreps; A5: {3a,3b} vs {4} vs {5}). With schoolbook kappa: S4 sign + std_sign (sign 1/6), A5 3a + 3b — the observed supports. | **REFUTED** (robustness) / **CONDITIONAL** (identity) | §6.2–6.3 |
| 5d | D_n, symmetric class: the count equals the cyclic count K*(n, nu/q) and the sign irrep is never used, for every kappa(2) (globally false: §9, D_7). | **PROVEN** given the cyclic programme (same computer-assisted status as Z_p) | §6.4 |
| 6 | Composite abelian groups: complex pairs cost nu^{1-r} (ELEGANT), real (order-2) characters cost 1: a discount of nu^{s-1} 2^{nu/2-1} per share. No 2-torsion defect in the (H)-theorem. Z_120, Z_11^2 predictions. But (T) itself fails on Z_12 (§9). | **PROVEN** (theorem) / **COMPUTED** (predictions) | §7 |
| 7 | **Complex-weight gated chain (real-part readout), any finite abelian group: unconditional exact global theorem**, no junk hypothesis: gamma* = (N nu)^{-nu/2} Gamma^C, attained by ONE neuron per character; for Z_p the count is the same K*(p, nu/q). | **PROVEN** (+ fp64 attainment and ascent checks) | §8 |
| 8 | q = nu endpoint (Morwani's L_{2,nu}) on non-abelian groups: two-sided exact (kappa = 1), and the optimum is NOT always dense: dense on S5 (nu 3), S4; sparse on A5 (nu 3: {3a,3b,4}) and on S5 at nu >= 4 ({std, std_sign, 5a}). | **PROVEN** (exact LP) | §6.5 |
| 9 | Group analogue of (T): no network beats the symmetric-class ceiling. | **REFUTED** on Z_12 (abelian, nu 4: 1.045-1.064 x, (H)-ceiling exact and attained; saved net re-verified), on D_3 (nu 4: explicit 9-neuron coset construction 1.601 x, saved ascent net 1.621 x, independently re-verified; ceiling exact by K2) and S4 (nu 4: 1.27-1.46 x); nu = 3: D_3, D_5, D_7, S4, A5 exceed the schoolbook ceiling by 4-24%. Abelian controls without a sign-fixed pair (Z_7, Z_9, Z_3^2, Z_6, Z_10; Z_8 and Z_12 at nu 3) never exceed it. | §9 |
| 9b | Mechanisms: (a) coset relaxation on groups with a sign character (junk confined to sign-inconsistent triples, 4 instead of 8 neurons per 2-dim feature, count can change: D_7 4 vs 2); (b) aligned junk with |J|/|X| = 1/sqrt d on A5 / S4 (hypothesised structure). | **COMPUTED** (anatomy) / hypothesis | §9.3 |
| 9c | S5: the symmetric optimum std_sign does not move under junk ascent (0.86, 0.99); random-init ascents reach 0.465. Whether a sign + coset optimum beats it globally is unknown. | **OPEN** | §9.4 |

---

## 1. Setting and symbols

| symbol | meaning | plain reading |
|---|---|---|
| G, N = \|G\| | a finite group and its order | the task's group |
| n, nu = n + 1 | number of operands, homogeneity degree | nu = number of weight factors in each product |
| a = (a_1, ..., a_n), P(a) = a_1 a_2 ... a_n | a row of the table and its correct label | the full table has all N^n rows |
| z(a, c) = sum_i w_i(c) prod_j e_ij(a_j) | the gated chain; neuron i has operand embeddings e_ij : G -> R and readout w_i : G -> R | one hidden layer of multiplicative units |
| theta_i, \|\|theta_i\|\|^2 = sum_j \|\|e_ij\|\|^2 + \|\|w_i\|\|^2 | neuron i's weights | |
| P_q = sum_i \|\|theta_i\|\|^q, 2 <= q <= nu, r = q/nu, s = nu/q | penalty; q = 2 is weight decay, q = nu is Morwani's L_{2,nu} | |
| margin = min_a min_{c != P(a)} [z(a, P(a)) - z(a, c)]; gamma_q = margin / P_q^{nu/q} | normalized margin | scale-free |
| C, cl(x) | a conjugacy class; the class of x | |
| rho, d_rho, chi_rho | complex irreducible unitary representation, dimension, character | |
| f^(rho) = sum_x f(x) rho(x)^* | Fourier transform (a d x d matrix) | f(x) = N^{-1} sum_rho d_rho tr(f^(rho) rho(x)) |
| E_ij(rho) = e_ij^(rho), W_i(rho) = w_i^(rho) | neuron i's Fourier blocks | |
| G_rho = sum_i tr( W_i(rho) E_i1(rho)^* ... E_in(rho)^* ) | clean amplitude at rho | how strongly the network writes the "correct-answer" signal through rho |
| m_i(rho) = sum_j \|\|E_ij(rho)\|\|_F^2 + \|\|W_i(rho)\|\|_F^2 | neuron i's Fourier mass at rho | |
| pi = {rho, rho-bar} | a real irrep: either a self-dual rho (eps_pi = 1) or a conjugate pair (eps_pi = 2) | real weights force the two members of a pair to be conjugate |
| cov_pi(C) = d_pi - Re chi_rho(C) | coverage of the wrong offset class C by irrep pi | the cyclic 1 - cos; >= 0, and > 0 for every C != e if pi is faithful |
| kappa_pi >= 1 | the price of a junk-free feature at pi (§4, §5) | "how many neurons' worth of cost a clean feature needs" |
| c_pi = kappa_pi eps_pi^{q/2 - r} d_pi^{q/2} | the programme price | |
| Gamma(c) = max { min_{C != e} sum_pi y_pi cov_pi(C) : y >= 0, sum_pi c_pi y_pi^r <= 1 } | **the programme** | best coverage of the worst wrong class for a unit cost budget |

Unrestricted prices: kappa = 1 ("c^unres"). Symmetric-class prices: kappa_pi = kappa(d_pi) for real-type irreps with d >= 2,
kappa = 1 for real characters, kappa = nu^{1-r} for complex pairs of 1-dim characters (ELEGANT Theorem 1).

Cyclic check: for G = Z_p every nontrivial pi is a pair (eps 2, d 1, cov 1 - cos 2 pi k D/p); with kappa = 1 the prices are all
2^{q/2 - r}, and (N nu)^{-nu/2} Gamma = 2 (2 p nu)^{-nu/2} Gamma_s(p) = B Gamma_s(p), THEORY_nsum's bound; with kappa = nu^{1-r} it is
ELEGANT Theorem 2's nu^{-(nu-q)/q} B Gamma_s(p). So everything below reduces to the proven cyclic theory on Z_p.

## 2. Theorem G1: the master bound for every finite group

> **THEOREM G1.** For every finite group G, every n >= 1, every gated chain of any width on the full table of
> a_1 ... a_n = c, and every q in [2, nu],
>
>     gamma_q  <=  (N nu)^{-nu/2} Gamma(c^unres),     c^unres_pi = eps_pi^{q/2 - r} d_pi^{q/2}.
>
> **Own-share form (G1').** With t_i(pi) := eps_pi d_pi m_i(rho_pi) / N (neuron i's weight mass on the isotypic component pi),
> U_pi := sum_i t_i(pi)^{q/2} and beta_pi := U_pi / sum U,
>
>     gamma_q  <=  (N nu)^{-nu/2} V(beta),     V(beta) = min_{C != e} sum_pi beta_pi^s (c^unres_pi)^{-s} cov_pi(C).
>
> *Plain reading: the worst wrong answer is no better than the average wrong answer of its class; only the "clean"
> class-function part of the logits survives that average; one neuron of a given norm can put only so much clean amplitude
> into irrep pi, and an irrep of dimension d dilutes it by d (Plancherel). G1' is the same bound evaluated at the network's
> own budget shares instead of the best shares.*

### 2.1 L1 (level-set bound). No symmetry argument needed.

For a class C put L_C := { (a, c) : P(a) c^{-1} in C } and zbar_C := mean of z over L_C. For C != {e}, every (a, c) in L_C has
c != P(a), so margin <= z(a, P(a)) - z(a, c). Averaging over L_C: for fixed a the map D -> c = D^{-1} P(a) is injective, so
each row appears exactly |C| times and the first term averages to zbar_e := mean_a z(a, P(a)). Hence

    margin <= zbar_e - zbar_C     for every class C != {e}.                                         (L1)

*Remark (the coordinator's orbit statement, verified).* The group H = G^{n+1} acting by
(a_1, ..., a_n, c) -> (g_0 a_1 g_1^{-1}, g_1 a_2 g_2^{-1}, ..., g_{n-1} a_n g_n^{-1}, g_0 c g_n^{-1}) maps rows to rows and
correct labels to correct labels, and P(a) c^{-1} -> g_0 (P(a) c^{-1}) g_0^{-1}. It is transitive on each L_C: given
(a, c), (a', c') with D' = x D x^{-1} take g_0 = x and g_j = a_j'^{-1} g_{j-1} a_j; the last condition g_0 c g_n^{-1} = c'
reduces to g_0 D g_0^{-1} = D'. The stabiliser is {g_0 in C_G(D)} (the other g_j are then forced), so orbits have size
N^{n+1} / |C_G(D)| = N^n |C| = |L_C|. Averaging z over H therefore gives the class function C -> zbar_C; L1 is the
statement margin(z) <= margin(average), exactly as THEORY_nsum L1. The proof above does not need it.

### 2.2 L2 (Fourier formula for the level-set means)

    zbar_C  =  N^{-nu} sum_rho chi_rho(C^{-1}) G_rho,     G_rho = sum_i tr( W_i(rho) E_i1(rho)^* ... E_in(rho)^* ).   (L2)

*Proof.* Write D := P(a) c^{-1}, c = D^{-1} P(a). Fourier inversion of w_i at D^{-1} a_1 ... a_n gives
N^{-1} sum_rho d_rho tr( W_i rho(D^{-1}) rho(a_1) ... rho(a_n) ); summing e_ij(a_j) rho(a_j) over a_j gives E_ij^* (e real).
Summing over a: sum_a z(a, D^{-1} P(a)) = N^{-1} sum_rho d_rho sum_i tr( W_i rho(D^{-1}) E_i1^* ... E_in^* ). Summing over
D in C, Schur's lemma gives sum_{D in C} rho(D^{-1}) = (|C| chi_rho(C^{-1}) / d_rho) I. Divide by |L_C| = N^n |C|. ∎
Checked by brute force (F1): relative error <= 5.4e-15 on S4 (n = 2, 3), S5, A5, D_7 (n = 2, 3).

### 2.3 L3 (phase step)

Average (L1) at C and at C^{-1} (also a class), use chi(C^{-1}) = conj chi(C):

    margin <= N^{-nu} sum_rho Re(G_rho) (d_rho - Re chi_rho(C)) <= N^{-nu} sum_rho |G_rho| cov_rho(C)   for every C != e.  (L3)

The first expression is real because the left side is; each factor d_rho - Re chi_rho(C) >= 0 since |chi| <= d. Equality in
the second step iff every G_rho with cov_rho(C) > 0 is real and non-negative (all irreps "phase-aligned at the identity").

### 2.4 L4 (per-neuron AM-GM) and L5 (Plancherel)

|tr(X_0 X_1 ... X_n)| <= ||X_0||_F ||X_1 ... X_n||_F <= prod_j ||X_j||_F (Cauchy–Schwarz, then submultiplicativity), and
AM-GM on the nu norms: with G_{i,rho} the neuron's term in G_rho,

    |G_{i,rho}| <= prod_j ||E_ij(rho)||_F ||W_i(rho)||_F <= ( m_i(rho) / nu )^{nu/2}.                       (L4)

Plancherel: sum_x f(x)^2 = N^{-1} sum_rho d_rho ||f^(rho)||_F^2, hence ||theta_i||^2 = N^{-1} sum_rho d_rho m_i(rho).     (L5)
Real weights: f^(rho-bar) = conj f^(rho), so m_i(rho-bar) = m_i(rho) and G_{rho-bar} = conj G_rho: the two members of a pair
carry equal mass and equal |G|. Checked (F3) to 1.5e-16.

### 2.5 Assembly

Group the nontrivial complex irreps into real irreps pi (§1). Then t_i(pi) = eps_pi d_pi m_i(rho_pi)/N, sum_pi t_i(pi) <= ||theta_i||^2
(the trivial irrep's mass is dropped), and by L4, subadditivity of x -> x^r (r <= 1) and superadditivity of x -> x^{q/2} (q >= 2):

    (eps_pi d_pi nu / N)^{q/2} |G_pi|^r  <=  sum_i t_i(pi)^{q/2}  =: U_pi,     sum_pi U_pi <= sum_i ||theta_i||^q = P_q.     (A)

With y_pi := eps_pi |G_pi| (so that L3's right side is N^{-nu} sum_pi y_pi cov_pi(C)), (A) reads
sum_pi c^unres_pi y_pi^r <= (N/nu)^{q/2} P_q. The programme is positively homogeneous: scaling the budget by B scales Gamma by
B^{1/r}. So margin <= N^{-nu} ((N/nu)^{q/2} P_q)^{nu/q} Gamma = N^{-nu} (N/nu)^{nu/2} P_q^{nu/q} Gamma, which is G1. For G1', bound
each y_pi separately by ((N/nu)^{q/2} U_pi / c_pi)^{1/r} and pull out (sum U)^s <= P_q^s. ∎

Where the loss sits (for later): L1 (junk), L3 (phases), L4 (one neuron with rank-one aligned blocks reaches it), the r-subadditivity
(one neuron per irrep reaches it), superadditivity (single-irrep neurons reach it). The unrestricted bound is attained only by
networks whose neurons are rank-one aligned **and** junk-free, which for d >= 2 is impossible (§4): hence the price kappa.

### 2.6 Real, complex and quaternionic type

The proof above is type-blind: it works with complex irreps and only uses that real weights make rho and rho-bar carry
conjugate blocks. Type enters only through eps (pairs) and, in §4, through the invariant theory of the block. Frobenius–Schur
indicators (`verify_fourier.py`, F0), exact: every irrep of S5, S4, A5 and D_7 has indicator +1 (real type); D_n is real type for
all odd n (the 2-dim irreps are rotation–reflection matrices). Abelian groups: order-2 characters are real type, all others are
complex pairs (eps 2, d 1). Quaternionic irreps do not occur in any group of the programme (they would need a separate
invariant count in §4; not attempted).

## 3. The programme is solved exactly by vertex enumeration

> **LEMMA V.** Let P := { y >= 0 : sum_pi y_pi cov_pi(C) >= 1 for every C != e } and f_c(y) := sum_pi c_pi y_pi^r (c > 0,
> 0 < r <= 1). Then Gamma(c) = ( min over the vertices v of P of f_c(v) )^{-1/r}. For r < 1 **every** minimiser of f_c on P is a
> vertex; hence every maximiser of the programme is (a multiple of) an optimal vertex.

*Proof.* Every nontrivial coverage vector is >= 0 and every C != e is covered by a faithful irrep, so P is non-empty, pointed
(y >= 0) and its recession cone is the orthant. Any y in P is y = sum_v lambda_v v + d with lambda a probability vector on
vertices and d >= 0. f_c is coordinatewise non-decreasing and concave, so f_c(y) >= f_c(sum lambda v) >= sum lambda f_c(v) >= min_v
f_c(v). Scaling (§2.5) turns min f_c over P into Gamma. For r < 1, t -> ((1-t)a + tb)^r is strictly concave on [0, 1] whenever
a != b (including a = 0), and y^r is strictly increasing, so equality forces d = 0 and a single vertex. ∎

*Plain reading: the best allocation of a concave cost budget sits at a corner of the "cover every wrong class" polyhedron,
and the corners do not depend on the prices. So one finite list of candidate committees serves every kappa, every q and
every nu; the prices only decide which corner wins, and each corner's cost is LINEAR in the unknown kappa.*

Consequences used below: (i) exact solutions (rational arithmetic for S5 and S4: 43 and 11 vertices; A5: 15, float);
(ii) the "count" of a max-margin network in the symmetric class is the support size of the optimal vertex (Corollary G2.2);
(iii) robustness in kappa is a finite family of linear inequalities (§6). The sign-patch amplitude is a vertex coordinate, so it is
price-free: the S5 vertex {std, sign} has y_std = 1/3, y_sign = 1/6, i.e. A_sign/A_std = 1/(4·2) = **1/8 exactly**
(A_pi = d_pi y_pi / eps_pi is the coefficient of chi_pi/d_pi); S4 {std(_sign), sign}: **1/6**.

## 4. The symmetric class: junk-free features are matrix-multiplication decompositions

### 4.1 The invariant (Schur)

Take a real-type irrep with a real orthogonal model rho (built explicitly in `gt_core.real_irrep`). The Fourier block of the
logit table at (rho, ..., rho; rho),

    X_rho := sum_{a,c} z(a, c) rho(a_1)^T (x) ... (x) rho(a_n)^T (x) rho(c)^T  =  sum_i E_i1 (x) ... (x) E_in (x) W_i

(F2, checked to 6.4e-14), transforms under the task symmetry H = G^{nu} by E_j -> rho(g_j)^T E_j rho(g_{j-1}),
W -> rho(g_n)^T W rho(g_0). Each g_k acts on exactly two tensor indices (one of each of two consecutive factors in the cycle
W - E_1 - ... - E_n - W), so the H-invariants of the block are the tensor product over k of Inv_G(V (x) V), and for real
type Inv_G(V (x) V) = Hom_G(V, V) = R · I (Schur; `verify_fourier.py` F4: dimension 1 for every irrep of S4, S5, A5). Hence

> **PROPOSITION S.** The H-invariant subspace of the (rho, ..., rho) block is the line spanned by the chain-trace tensor
> T_rho, <T_rho, E_1 (x) ... (x) E_n (x) W> = tr(W E_1^T ... E_n^T), ||T_rho||^2 = d^nu, and <T_rho, X_rho> = G_rho.
> T_rho is the iterated matrix-multiplication tensor <d, ..., d> (for n = 2: Strassen's <d, d, d>).

### 4.2 The class and the price

**Definition (class H_G).** A network is *symmetric* if, for every real-type irrep rho of dimension >= 2, its block X_rho is a
multiple of T_rho, and for every complex pair of 1-dim characters its same-character sign-pattern junk vanishes (ELEGANT's
(H)). Real characters impose nothing. Any network whose logit table is invariant under H (z(a, c) a function of
cl(P(a) c^{-1}) — the coordinator's "symmetric") is in H_G; H_G is larger (cross-irrep junk, neurons mixing irreps allowed).

**Definition (price).** For the chain-trace tensor T_d of nu slots,

    N_r(T_d) := inf { sum_i |mu_i|^r : T_d = sum_i mu_i U_i1 (x) ... (x) U_inu, ||U_ij||_F = 1 },     kappa_{nu,r}(d) := N_r(T_d) / d^{nu r}.

(The infimum is over real decompositions of any length.)
For real characters kappa = 1 (the block is 1 x ... x 1; sum mu_i = 1 forces sum |mu|^r >= 1). For complex pairs of 1-dim
characters kappa = nu^{1-r}: this is exactly ELEGANT Theorem 1 (the complex d = 1 case, where the tensor is the real form of
complex multiplication of nu numbers and the minimal decomposition is Gauss-type, nu terms).

### 4.3 Theorem G2

> **THEOREM G2 (symmetric-class ceiling).** Let every irrep of G be real type or a 1-dim character (all groups here).
> For every symmetric network (class H_G) and every q in [2, nu],
>
>     gamma_q <= (N nu)^{-nu/2} Gamma(c^kappa),     c^kappa_pi = kappa_pi eps_pi^{q/2 - r} d_pi^{q/2},
>
> and the supremum over H_G equals the right side: placing, for each irrep of an optimal vertex y*, an (almost) optimal
> decomposition of (y*_pi / d^nu) T_rho as one neuron per term (balanced factors, e(x) = (d/N) tr(E rho(x))), attains it
> (exactly, when kappa is attained — e.g. by the schoolbook at even nu, §5).

*Proof.* Under H_G, X_rho = (G_rho / d^nu) T_rho (Prop S), so {neuron i's rank-one term, with modulus
lambda_i = prod_j ||E_ij||_F ||W_i||_F} is a decomposition of a multiple of T_rho; by the definition of kappa,
sum_i lambda_i^r >= kappa(d) |G_rho|^r. Replace L4's |G_rho|^r <= sum_i (m_i/nu)^{q/2} by kappa |G_rho|^r <= sum_i lambda_i^r <=
sum_i (m_i/nu)^{q/2} and repeat §2.5 with c^kappa. For complex pairs use ELEGANT Theorem 1 at that character (its (KL)
system is well-posed exactly when chi != chi-bar, §7.1). Attainment: each neuron is single-irrep and balanced, so L4, L5 and
the super/subadditivity steps are equalities; the logit is exactly the class function
N^{-nu} sum_pi y*_pi chi_pi(P(a) c^{-1}) so L1 and L3 are equalities; the cost is (d nu / N)^{q/2} sum lambda^r = the price. ∎

Checked by brute force on the full table (`construct_verify.py`): S5 (n 2, q 2 and 2.5, supports {std_sign} and {std, sign}),
A5 {3a, 3b}, S4 {sign, std_sign} (n 2 and n 3), S4 {sign, 2, std_sign} (n 3, q 3): logit = class function to 1e-19, and
gamma_q / closed form = 1.000000000000 in all 8 cases (worst deviation 2.9e-15).

> **COROLLARY G2.2 (count and neurons per feature).** For q < nu, every symmetric network attaining the ceiling has clean
> spectrum (y_pi) proportional to an optimal vertex of P (Lemma V). Its feature count is that vertex's support size. At even nu
> (§5) every such feature at a d-dim real-type irrep is carried by **exactly d^nu neurons**, each rank-one aligned
> (|<T, U_i1 (x) ... >| = 1) with equal moduli.

*Plain reading: a clean feature on a d-dimensional irrep is a way of multiplying d x d matrices, one neuron per scalar
product used. The cheapest way under weight decay, at least at even depth, is the schoolbook algorithm: d^nu equal neurons. So a
4-dim irrep of S5 costs 4^nu neurons, and the price grows like d^{nu(1-r)} — which is what pushes the optimum onto the
cheapest irrep that covers every wrong class.*

## 5. The price kappa(d)

### 5.1 The flattening lemma

> **LEMMA F.** Let X = sum_l mu_l a_l b_l^T with unit vectors (or ||a_l||, ||b_l|| <= 1). For every r in (0, 1],
> sum_l |mu_l|^r >= sum_k sigma_k(X)^r, with equality (r < 1) iff the non-zero |mu_l| are exactly the singular values.

*Proof (self-contained, the ELEGANT route).* Let X = U Sigma V^T (thin SVD). Then Sigma = sum_l mu_l (U^T a_l)(V^T b_l)^T with
||U^T a_l|| <= 1, ||V^T b_l|| <= 1. Sylvester + Cauchy–Binet: E_k(Sigma) = sum_{|L|=k} prod_{l in L} mu_l det(B_L^T A_L), and
|det(B_L^T A_L)|^2 <= det(A_L^T A_L) det(B_L^T B_L) <= 1 (Cauchy–Binet/Cauchy–Schwarz on k-th exterior powers, then Hadamard).
So e_k(sigma) <= e_k(|mu|) for all k, hence prod_k (1 + sigma_k t) <= prod_l (1 + |mu_l| t) for t >= 0, and the Bernstein integral
x^r = (r sin(pi r)/pi) int_0^inf log(1 + x t) t^{-1-r} dt gives the claim; equality forces the two polynomials to coincide. ∎
(This is McCarthy's / Rotfel'd's p-subadditivity of the Schatten quasi-norm, 0 < p <= 1, specialised to rank-one terms.)

Apply it to T with any bipartition S of the slots: a unit product tensor flattens to a b^T with unit a, b, so
**N_r(T) >= sum_k sigma_k(flat_S T)^r**. One may also first apply to the S-side any multilinear map that does not increase the
norm of product vectors (e.g. matrix product of adjacent slots, ||B C||_F <= ||B||_F ||C||_F).

### 5.2 The bounds

> **THEOREM K1.** 1 <= kappa_{nu,r}(d) <= d^{nu(1-r)}. (Upper: the schoolbook decomposition, d^nu terms e_{k0k1} (x) e_{k1k2}
> (x) ... with mu = 1. Lower: the spectral norm of T is 1 and ||T||^2 = d^nu, so sum |mu| >= d^nu (nuclear norm = d^nu), and
> sum |mu|^r >= (sum |mu|)^r.) At r = 1 (q = nu) kappa = 1 exactly: symmetric and unrestricted bounds coincide.

> **THEOREM K2 (even nu: the schoolbook is optimal).** If nu is even, kappa_{nu,r}(d) = d^{nu(1-r)} for every r in (0, 1],
> and every optimal decomposition has exactly d^nu terms of modulus 1 (in T-units).

*Proof.* The slots sit on a cycle of even length nu, consecutive slots sharing one index. Put S = the odd slots. No two slots
of S share an index and together they carry all nu indices, and the same holds for the even slots. So
flat_S T = sum over the d^nu index assignments of (unit vector of the S-part)(unit vector of the complementary part)^T, two
orthonormal families: a partial isometry of rank d^nu, all singular values 1. Lemma F gives N_r >= d^nu, the schoolbook gives
<= d^nu; equality case from Lemma F. ∎  (Rank corollary: the 4-cycle chain tensor tr(ABCD) has tensor rank exactly d^4.)

> **THEOREM K3 (odd nu: lower bound).** If nu is odd, kappa_{nu,r}(d) >= d^{(nu-1)(1-r)}. At nu = 3, q = 2: d^{2/3} <= kappa(d) <= d.

*Proof.* Merge two adjacent slots by the matrix product (norm-non-increasing on products), leaving an even cycle of nu - 1
slots with the merged index summed: the resulting tensor is d · T_{d, nu-1}. Its alternating flattening is d times a
partial isometry of rank d^{nu-1}: singular values d. Lemma F: N_r >= d^{nu-1} d^r, i.e. kappa >= d^{nu-1+r-nu r}. ∎
The plain flattening gives only d^{nu-1-(nu-1/2)r}; the merge gains d^{r/2}. At nu = 3 the family "norm-non-increasing map on
the (2,3) side + flattening" cannot do better: the flattened rank is at most d^2 (the slot-1 side), each image vector has norm
<= d (it is a sum of d unit product vectors), so sum sigma^r <= d^2 d^r, which K3 attains. Closing the remaining factor d^{1-r}
needs a genuinely different argument.

> **PROPOSITION K5 (restriction).** For k <= D and q >= 2: N_r(T_D) >= (D/k) N_r(T_k), i.e.
> kappa(D)/kappa(k) >= (k/D)^{q-1}. (Restrict every factor to a k-subset of indices; average over subsets; AM-GM and
> sum over subsets of ||P_S U P_S||_F^2 <= C(D-1, k-1).) At q = 2: kappa_5 >= 0.8 kappa_4, kappa_6 >= 0.667 kappa_4 — too weak for §6.1's
> identity condition (0.969, 0.937).

### 5.3 Where the schoolbook might lose: small r at odd nu

Rank(<2,2,2>) = 7 < 8, so as r -> 0 (not an admissible q at nu = 3: r >= 2/3) Strassen beats the schoolbook. Within the
admissible range at nu = 3 it does not: the best rank-7 decomposition (Strassen's orbit under GL_2^3, polished) costs
**exactly 10.000000** at q = 2 against the schoolbook 8 (`kappa_mm.py`; the coordinator's "balanced Strassen ≈ 11.5" is the
unpolished raw decomposition, 11.524).

### 5.4 Numerical estimates (upper bounds; `kappa_mm.py`, augmented Lagrangian + L-BFGS, fp64)

Every entry is the cost of an explicit exact decomposition (residual < 1e-7 relative); the method returns upper bounds only.

| nu | q | d | proven interval for kappa | best found | schoolbook | runs converged / at schoolbook |
|---|---|---|---|---|---|---|
| 3 | 2 | 2 | [1.587, 2] | **2.000000** | 2 | 13/13 at R >= 8 all = 8.000000; R = 7: 10.000000 (4/4) |
| 3 | 2 | 3 | [2.080, 3] | **3.000000** | 3 | R >= 32: 8/8 at 27.000000; R = 27: 3 local minima at 29 (26 terms); R = 23: 36.48 (23 terms) |
| 3 | 2.5 | 2 | [1.260, 1.414] | **1.414214** | 1.414 | all converged runs = 8 |
| 3 | 2.5 | 3 | [1.442, 1.732] | **1.732051** | 1.732 | R = 32: 3/3 at 27; R = 27: 29.97, 33.0 |
| 3 | 2 | 4 | [2.520, 4] | **4.000000** | 4 | R = 80: 3/3 at 64.000000; R = 64: 66, 66, 68 (62-63 terms); R = 60: 71.2-72; R = 49 (below rank bound): no convergence |
| 4 | 2 | 2 | {4} (Theorem K2) | 4.000000 | 4 | consistency check (theorem); random starts do not converge at nu >= 4 (augmented Lagrangian stalls), only schoolbook+noise does |
| 5 | 2 | 2 | [5.278, 8] | 8.000000 | 8 | only the schoolbook+noise start converged (random starts stall, res ~ 1): no evidence either way |

*Reading.* In every converged run, nothing beats the schoolbook at nu = 3 within the admissible q range; the minimal-rank
decompositions (Strassen 7, Laderman-type 23) are strictly worse. **Conjecture K (odd nu):** kappa_{nu,r}(d) = d^{nu(1-r)} for all
r >= r_nu with r_3 <= 2/3. Evidence only.

## 6. Robustness of the count (Task 3)

Method: Lemma V makes each vertex's cost linear in kappa = (kappa_d)_d. A claim holds on a box of kappas iff it holds on
every sub-box of a finite subdivision, and on a sub-box it suffices that every "bad" vertex is dominated by one "good" vertex
at all corners (linearity). `robust_cert.py` does this in 40-digit arithmetic with exact rational vertices; `programme.py`
and `qscan.py` give exact argmaxes at fixed prices.

### 6.1 S5 (irreps sign 1, std 4, std_sign 4, 5a 5, 5b 5, 6 6; classes (12), (12)(34), (123), (123)(45), (1234), (12345))

Coverage d - chi: sign [2,0,0,2,2,0]; std [2,4,3,5,4,5]; std_sign [6,4,3,3,4,5]; 5a [4,4,6,4,6,5]; 5b [6,4,6,6,4,5];
6 [6,8,6,6,6,5]. std_sign's worst classes are (123) and (123)(45) at 3/4 of its dimension; std's worst is (12) at 1/2, which
the sign repairs at amplitude exactly 1/8 until (123) (3/4) binds — the same 3/4 as std_sign alone, at extra cost. **Hence
{std, sign} is dominated by {std_sign} for every price vector** (same kappa_4, same worst coverage, extra sign cost).

**Even nu (exact, kappa known).** Argmax of the symmetric class (exact rational vertices, `qscan.py`):

| nu | q range | argmax | runner-up (ratio of Gamma) |
|---|---|---|---|
| 4 | [2, 3.625) | **std_sign** (K = 1, 256 neurons) | std + sign, sign at 1/8 (0.978 at q = 2) |
| 4 | [3.625, 4) | std_sign + 5b | |
| 6 | [2, 5.95) | **std_sign** (K = 1, 4096 neurons) | std + sign (0.998 at q = 2) |
| 8 | [2, 8) | **std_sign** | std + sign (0.9998) |

**nu = 3, q = 2 (the GX bil family): THEOREM R.** For every (kappa_4, kappa_5, kappa_6) in the proven box
[4^{2/3}, 4] x [5^{2/3}, 5] x [6^{2/3}, 6], every optimal vertex has exactly ONE irrep and no sign component
(`robust_cert.py`: 12 sub-boxes, minimal slack 7.6e-3 in 40-digit arithmetic). The irrep is std_sign iff
kappa_5/kappa_4 >= 0.969131 and kappa_6/kappa_4 >= 0.937147 (then uniquely), otherwise 5a/5b (tied) or 6. So:

- **PROVEN (symmetric class, nu = 3, q = 2): K* = 1, no sign patch, whatever the true kappa.**
- CONDITIONAL: identity std_sign. Under the numerically supported schoolbook values (kappa_d = d) the ratios are 1.25 and 1.5,
  far inside. Not provable from K5 (gives 0.8, 0.667).
- Also certified: nu = 5, q = 2 (count 1). Not certified: nu = 3, q = 2.5 (count 1 or 2 on the box: std_sign + 5b enters).

Ladder at nu = 3, q = 2, schoolbook kappa (Gamma relative to the argmax): std_sign 1.000; std + sign 0.944; std + std_sign 0.740;
std_sign + 5b 0.716; 5b = 5a 0.683 (lower-bound kappa: std_sign 1.000, std + sign 0.913, 5a/5b 0.763). Compare GX's registered
"naive" ladder (std_sign 1.00; std + sign 0.94; 5a = 5b 0.68; 6 0.49): identical where comparable, because GX's naive kappa for
real-type irreps is exactly the schoolbook d^{4/3} = d^{1/3} kappa_school(d). GX's "bound" ladder (d^{1/3}) is our unrestricted
Theorem G1 (correct as a bound; it is not the symmetric-class functional, which is what GX's A5 data showed).

### 6.2 S4 (sign 1, 2, std 3, std_sign 3)

nu = 3, q = 2: schoolbook argmax {sign, std_sign} tied exactly with {sign, std} (sign amplitude 1/6), then {sign, 2, std(_sign)}
at 0.997. **Count not kappa-free**: on the proven box the optimum is 2 or 3 irreps (the 2-dim irrep enters when kappa_2 is
near its lower bound 2^{2/3}); `robust_cert.py`: K = 2 and K = 3 both fail to certify. At q = 2.5 the count 3 IS certified.
Even nu (exact): nu 4, q in [2, 2.175): {sign, std_sign} ~ {sign, std} (K = 2); q >= 2.175: + the 2-dim irrep (K = 3);
nu 6, q = 2: {sign, 2, std_sign} (K = 3). Data (GX bil, nu 3): free runs use all four irreps (near-tied top: 1.000/1.000/0.997);
fixed-support std + sign on top (ours: tied top). Consistent, not discriminating.

### 6.3 A5 (3a, 3b, 4, 5; 3a/3b Galois conjugates)

nu = 3, q = 2: schoolbook argmax **3a + 3b** (K = 2); {3a, 3b} beats {4} iff kappa_4/kappa_3 >= 1.0671 and beats {5} iff
kappa_5/kappa_3 >= 1.0341 (schoolbook: 1.333, 1.667). On the proven box the argmax can be {3a,3b}, {4} or {5}: **count not
kappa-free** (1 or 2). Data (GX): 32/32 free runs on 3a + 3b, fixed-support order 3a+3b > 3a > 4 > 5 — exactly the schoolbook
ladder (1.000, 0.782, 0.716, ...). Even nu (exact): nu 4, q = 2: single 3a or 3b (K = 1; 3a + 3b at 0.904); nu 4, q in
[2.175, 3.825): 3a + 3b; nu 6, q = 2: single 3a/3b.

### 6.4 D_n (n odd; sign + (n-1)/2 two-dim real-type irreps rho_j)

> **PROPOSITION D.** For every value of kappa(2), the symmetric-class optimum on D_n (n odd) has no sign component, and its
> 2-dim irreps form an optimal support of the CYCLIC programme Gamma_{nu/q}(n) (coverage 1 - cos 2 pi j k/n), whenever that
> cyclic optimum has fewer than n/4 frequencies. In particular K*(D_n) = K*(Z_n) at the same (nu, q), independent of kappa(2).

*Proof.* Rotations r^k: cov_{rho_j} = 2 - 2 cos(2 pi j k/n), cov_sign = 0. Reflections: cov_{rho_j} = 2 (chi = 0), cov_sign = 2. All
rho_j have the same price, so the rotation constraints alone are the cyclic programme (up to a common factor). (a) Any y feasible
for D_n, restricted to the rho_j, is feasible for the rotation-only programme, and f_c is strictly increasing in y_sign; so
min_{D_n} >= min_rot, with equality only if y_sign = 0. (b) The rotation-only optimum y* is feasible for D_n: put
F(k) = sum_j y*_j cos(2 pi j k/n). If F(k) < 0 for every k != 0, then |F(k)| <= sum y* gives sum_{k != 0} F(k)^2 <= (sum y*)^2,
while orthogonality of cosines (n odd, 1 <= j, j' <= (n-1)/2) gives sum_{k=0}^{n-1} F(k)^2 = (n/2) sum y*_j^2, i.e.
sum_{k != 0} F^2 = (n/2) sum y*^2 - (sum y*)^2 >= (n/(2K) - 1)(sum y*)^2 by Cauchy–Schwarz on the K-element support: a
contradiction when K < n/4. So some k != 0 has F(k) >= 0, i.e. the rotation coverage at r^k is <= 2 sum_j y*_j = the reflection
coverage, and the reflection constraint holds. (a) + (b) give the claim. ∎

At the computed optima (`prog_large.py`, Z_113 and D_113 solved independently) the support sizes are 6 (nu 3, q 2) and 4
(nu 4, q 2), both < 113/4, with reflection coverage 1.53x and 2.03x the rotation minimum. K*(D_113) = K*(Z_113) = 6 (nu 3),
4 (nu 4), with the same computer-assisted status as the cyclic Gamma_s(p) itself. Data: only ReLU runs exist on D_113 (count 4
+ parity at nu 4); the ReLU family is outside this theory. The "parity patch" is ABSENT from the symmetric optimum but is exactly
what the junk-carrying coset optimum of §9 uses (sign + 2-dim coset features), and there the count can differ from Prop D's
(D_7 at nu 3: 4 vs 2). So Prop D is a symmetric-class statement only.

### 6.5 The q = nu endpoint is exact and not always dense (Morwani caveat)

At r = 1, kappa = 1 (K1), so G1 = G2 and the programme is an LP: **two-sided exact for every group**. Its optimum is the
dense "delta" solution (y_pi proportional to d_pi, all irreps, every wrong class covered equally) on S5 at nu = 3 and on S4 at
nu = 3, 4, 6, but **not** on A5 at nu = 3 (optimum {3a, 3b, 4}, Gamma 0.5211 vs dense 0.5039) nor on S5 at nu = 4, 6 ({std,
std_sign, 5a}, 0.2105 vs 0.2017 at nu 4). So "L_{2,nu} uses all features for group composition" is not a theorem for the gated
chain on every group; the abstract's attribution to Morwani et al. should be checked against the exact hypotheses of their
group theorem (quadratic-activation network, and — from memory, unverified — a character-sum condition on G).

## 7. Composite abelian groups (Z_120, Z_11 x Z_11): real characters and 2-torsion

### 7.1 Bookkeeping: no 2-torsion defect in the (H)-theorem

For a finite abelian G the nontrivial characters split into complex pairs {chi, chi-bar} (chi^2 != 1) and real characters
(chi^2 = 1, order 2; Z_120 has exactly one, sgn60(x) = (-1)^x; Z_11^2 has none). ELEGANT's Theorem 2 transfers verbatim to
complex pairs once one checks that the sign-pattern junk coefficients of different characters sit at different multi-indices of
the logit table: the coefficient at (chi^{e_1}, ..., chi^{e_n}; chi) (e in {+-1}^n) equals sum_i c_{i,chi} prod_{j in T} v_{ij,chi}
(T = {j : e_j = -1}), and (chi^{e}, chi) = (psi^{e'}, psi) slotwise forces psi = chi and (e_j - e'_j) chi = 0, i.e. e = e' unless
chi^2 = 1. So for complex pairs the map (character, pattern) -> multi-index is injective and hypothesis (H) is a condition on
the logit table, exactly as for Z_p. For a real character all 2^n patterns collapse onto ONE multi-index (chi, ..., chi; chi): the
block is 1 x ... x 1, there is no junk, one neuron is automatically clean, and kappa = 1. **The coordinator's worry (junk of chi and
chi.sgn on the same row mode) does not arise in the (H)-theorem**: chi.sgn's patterns sit at (chi^{e} sgn, ..., chi sgn), a
different multi-index. It can matter only for junk-DAMAGE arguments beyond (H) (P3C-type row-mode analyses), where
multiplying the row function by sgn(a_1) ... sgn(a_n) maps one to the other; those arguments are open anyway. Empirically
(§9.1-9.2) junk pays on Z_12 at nu 4 (1.06 x the (H)-ceiling) through the coset mechanism of §9.3: it needs a real irrep fixed by
the sign twist (pi (x) sgn ~ pi; on Z_N the pair chi_{N/4}, which exists iff 4 | N; on D_n every 2-dim irrep). Z_6, Z_10 (no such
pair) never exceeded the ceiling; Z_8 (pair f2 exists but is not in the (H)-optimum) did not in 4 seeds. Z_120 has one such pair
(f30), so its (H)-prediction is a symmetric-class statement only.

### 7.2 The real-character discount

Prices (§1): complex pair c = nu^{1-r} 2^{q/2-r}; real character c = 1. Per unit of budget share the objective weight is
w = c^{-s}, so a real character is cheaper by the factor

    w_real / w_pair = nu^{s-1} 2^{nu/2-1}       (= 2.45 at nu 3, q 2;  8 at nu 4, q 2;  144 at nu 6, q 2)

while covering only the half of the group where chi = -1 (at coverage 2, the maximum). In the unrestricted bound (kappa = 1)
the discount is 2^{nu/2-1} (the pair's mass is split between chi and chi-bar), in the complex-weight model (§8) there is none.

### 7.3 Predictions vs data (gated chain, `prog_large.py`; support search + SLSQP, computer-assisted like Gamma_s(p))

| group | nu, q | (H)-argmax | best K by rung (Gamma / argmax) | real char in argmax? | GX bil data (nu 3, 16 seeds) |
|---|---|---|---|---|---|
| Z_120 | 3, 2 | K = 7 | 4: 0.872, 5: 0.970, 6: 0.985, 7: 1, 8: 0.979, 9: 0.969, 12 (with sgn60): 0.975 | no (enters only in near-tied rungs) | K_5pct {2:1, 3:2, 4:2, 5:7, 6:3, 7:1}, sgn60 in 15/16 |
| Z_120 | 4, 2 | K = 4 | 3: 0.841, 5: 0.995, 6: 0.923 | no | (ReLU only) |
| Z_11^2 | 3, 2 | K = 8 | 4: 0.903, 5: 0.984, 6: 0.960, 7: 0.965, 9: 0.967 | – | K_5pct {3:2, 4:2, 5:5, 6:1, 7:2, 8:1} |
| Z_11^2 | 4, 2 | K = 4 | 3: 0.756, 5: 0.976 | – | (ReLU only) |
| Z_113 | 3, 2 / 4, 2 | 6 / 4 | (= kstar.json / THEORY_nsum) | – | control |

At nu = 3 the plateau is flat (every K in 5..12 within 5%), exactly as registered in RESULTS_GX; the data sit inside it.
The ubiquitous sgn60 in the data (largest amplitude in 13/16 nets) is NOT predicted by the (H)-class argmax. Z_120 has one
sign-fixed pair (f30, chi_30 . sgn60 = conj chi_30), so the coset mechanism of §9.3(a) is available there (it pays on Z_12, §9.1);
but f30 appears in only 1/16 trained committees, so it does not explain the sign's prevalence. Not tested at Z_120 scale.

## 8. Theorem C: the complex-weight gated chain is exactly solvable on every finite abelian group

**Model.** E_ij, W_i : G -> C, z(a, c) = Re sum_i W_i(c) prod_j E_ij(a_j), ||theta_i||^2 = sum_j ||E_ij||^2 + ||W_i||^2 (complex
moduli), P_q, gamma_q as before. (Each complex neuron is a weight-tied real sub-network: Re of a product of nu complex numbers
is a sum of 2^{n} real products sharing parameters, priced once by the complex modulus.)

> **THEOREM C.** For every finite abelian group G, every n, every q in [2, nu] and every complex gated chain on the full table,
>
>     gamma_q <= (N nu)^{-nu/2} Gamma^C,   Gamma^C = max { min_{D != 0} sum_chi y_chi (1 - Re chi(D)) : y >= 0, sum_chi y_chi^r <= 1 },
>
> where chi runs over the real characters and one representative of each conjugate pair (all priced 1). Equality holds for the
> network with ONE neuron per character chi of an optimal vertex, E_ij(a) = t_chi chi(a) (every j), W_i(c) = t_chi conj chi(c),
> t_chi^nu = y*_chi. For q < nu every maximiser has clean spectrum proportional to an optimal vertex (Lemma V), so the max-margin
> feature count is the vertex's support size. **No junk hypothesis is needed.** For G = Z_p, Gamma^C = Gamma_{nu/q}(p), so the
> count is exactly the Gamma-optimal K*(p, nu/q) of the real theory.

*Proof.* Fourier on G: f^(chi) = sum_x f(x) conj chi(x), f-check(chi) := sum_x f(x) chi(x). (i) L1 as in §2.1 (Z instead of z,
take real parts). (ii) The level-set mean of Z at offset D = P(a) - c is N^{-nu} sum_chi G_chi conj chi(D) with
G_chi = sum_i W_i^(chi) prod_j E_ij-check(chi) (expand W_i(P(a) - D) by inversion and sum each a_j). (iii) Averaging D and -D and
taking real parts: margin <= N^{-nu} sum_chi |G_chi| (1 - Re chi(D)). (iv) Per neuron and character,
|W^(chi) prod_j E-check(chi)| <= (m_{i,chi}/nu)^{nu/2}, m_{i,chi} = |W_i^(chi)|^2 + sum_j |E_ij-check(chi)|^2, and
sum_chi m_{i,chi} = N ||theta_i||^2 (Plancherel; chi -> chi-bar is a bijection). (v) Subadditivity of x^r and superadditivity of
x^{q/2} give sum_chi (nu/N)^{q/2} |G_chi|^r <= P_q. (vi) chi and chi-bar have identical coverage and price, so they are
linearly dependent columns of P and no vertex contains both (Lemma V): the programme may keep one per pair. (vii) Scaling as in
§2.5. Attainment: the constructed neuron outputs t^nu chi(P(a) - c), so the logit is the class function
sum_chi y*_chi Re chi(D) (L1, L3 tight), each neuron is balanced and single-character (iv, v tight), and there is one neuron
per character (subadditivity tight). ∎

**Checks** (`complex_closure.py`, full table, fp64): constructed networks hit the bound to 1e-12 on Z_7, Z_11, Z_13 (n 2),
Z_11 (q 2.5), Z_7 (n 3, q 2 and 3), Z_12 (contains a real character) and Z_3^2 — e.g. Z_11, q 2: Gamma^C = 0.53572593 on {4, 5}
(= VERIFY's Gamma_{3/2}(11) = 0.535725926 on {1, 3}, a multiplicative image; the vertex enumeration is therefore also an exact
certificate for Gamma_s(p) at small p, which the programme previously lacked). 600 random / near-character complex nets:
max gamma/bound 0.385. Direct ascent from random init: Z_5, n 3: 1.000000 (reaches the bound); Z_7, n 2: 0.559 in 6/6 seeds (a
local maximum of the landscape, never above 1).

**Relation to the real model.** Real weights are a special case, so gamma*_real <= gamma*_complex; the real (H)-class optimum
(ELEGANT Theorem 2) is 2^{1 - nu/2} nu^{-(s-1)} of the complex optimum (0.408 at nu 3 q 2; 0.125 at nu 4 q 2). The two losses
are exactly the two things a real neuron cannot do: put all its Fourier mass on chi rather than splitting it with chi-bar
(2^{1-nu/2}), and produce the clean term without sign-pattern junk (nu^{-(s-1)}, the nu-neuron block). **So the complex model is
the setting in which the paper's count is an unconditional global theorem; for real weights the count is exact on the (H) class
and globally conditional on (T).** What Theorem C does NOT do: it says nothing about real-weight networks' junk (it removes the
junk by changing the model), and for non-abelian groups complex weights still leave the matrix-multiplication price and the
same-irrep junk of §9 (only d = 1 characters become single neurons).

## 9. The group analogue of (T) is FALSE: same-irrep junk pays on non-abelian groups and on Z_12

(T_G) would say: no gated chain on G beats the symmetric-class ceiling of Theorem G2. On Z_p this is the open conjecture (T),
with strong evidence for odd p. **On non-abelian groups, and on abelian groups whose order is divisible by 4, it is false.**
Everything below is q = 2, full table, fp64 true margins.

### 9.1 A certified counterexample at even nu (where kappa, hence the ceiling, is exact)

D_3 (= S_3), n = 3 (nu = 4). By G2 + K2 every symmetric network has
gamma_2 <= 1 / (576 (1/sqrt 6 + 8/sqrt 3)^2) = 6.8699e-5 (optimal vertex {sign 1/6, rho 1/3}; the other vertex {rho 1/2} is worse).

* **Explicit construction (`coset_construct.py`)**: 8 neurons for the 2-dim irrep + 1 sign neuron, built from the identity
  theta_{xy} = theta_x + s_x theta_y (theta = 2 pi k / n rotation angle of x = (k, e), s_x = (-1)^e):
  the "coset feature" cos(theta_{a1} + s_{a1} theta_{a2} + s_{a1} s_{a2} theta_{a3} - theta_c) expands (using s^2 = 1) into 8 separable
  products, versus 16 for the schoolbook. Only two amplitudes are tuned. gamma_2 = 1.0998e-4 = **1.601 x the ceiling**.
* **Ascent optimum, saved** (`counterexample_d3_nu4.py`, weights in `out/counterexample_d3_nu4.npz`): 16 neurons,
  gamma_2 = 1.1138e-4 = **1.621 x the ceiling**, margin recomputed by an independent loop over all 216 rows and 6 classes.
  Unrestricted ascent from random init: 1.597-1.705 x in 6/6 runs (`out/junk_nu4.log`). The master bound G1 is 12.6 x the ceiling.
* S4, n = 3 (exact ceiling 4.2403e-7 on {sign 1/6, std_sign 1/3}): ascent started AT the symmetric optimum (82 neurons, plus 16
  small extra neurons) escapes to 1.269 x without noise and 1.460 x with 2% / 10% noise (`out/junk_a5_nu4.log`).
* A5, n = 3 (exact ceiling 3.2911e-8 on the single irrep 3b): the same escape test returns to 0.99991 (no escape, 700 steps,
  2% noise). So neither single-irrep symmetric optimum tested (A5 at nu 4, S5 at nu 3) moved; every escape found started from a
  support with a sign component (S4, D_n, Z_12) or a Galois pair (A5 at nu 3).

* **Z_12 (abelian!), n = 3 (nu = 4)** (`z12_check.py`): the (H)-ceiling 2.16284e-5 (ELEGANT Theorem 2 with the §7 pricing,
  optimal support {f3, f5, sgn6}) is attained to 8 digits by ELEGANT blocks + one sign neuron (so the pricing is right), and an
  unrestricted ascent net (saved: `out/counterexample_z12_nu4.npz`, margin re-verified by an independent loop over all 1728 rows)
  reaches **1.045 x** (junk_abelian.py seed 3: 1.064 x). Its junk sits in f3 — the pair with chi_3 . sgn6 = conj(chi_3), i.e. the
  real irrep fixed by the sign twist — and only on sign-inconsistent triples (0.68 vs 0.0016): the coset mechanism of §9.3(a).
  **So (T) is false on Z_12**: the cyclic conjecture must be restricted to groups without a sign character carrying a
  sign-fixed irrep (for Z_N: 4 does not divide N; in particular every odd prime p, the paper's setting, is unaffected).

### 9.2 nu = 3 (the GX family; ceilings use the schoolbook kappa, numerically optimal but unproven at odd nu)

| group | best unrestricted ascent / ceiling | escape from the symmetric optimum + noise | GX-trained nets / ceiling | explicit coset construction |
|---|---|---|---|---|
| D_3 | 1.165 (10/12 random starts >= 1.10) | 1.100-1.164 | – | 1.049 |
| D_5 | – | – | – | 1.081 |
| D_7 | 1.238 | 0.995-1.015 | – | 1.042 |
| S4 | 1.135 | 1.095-1.165 | fixed-support std+sign 1.139; 2+std+sign 1.045 | – |
| A5 (simple, no sign) | – | 0.996 / **1.122** / 0.969 (noise 0 / 2% / 10%) | **fixed-support 3a+3b 1.089** | – |
| S5 | 0.465 (random init, mixed all-irrep state) | **0.861, 0.989: does NOT escape** | best 0.814 (std+sign), free 0.30 | – |
| Z_7, Z_9, Z_3^2 (odd, controls) | 0.998-1.000 (nu 3 and nu 4) | | | |
| Z_6, Z_8, Z_10 (nu 3, 4), Z_12 (nu 3), Z_2xZ_4 (nu 3) | 0.919-1.000 | | | |
| **Z_12 (nu 4)** | **1.064** | | | |

(The proven box at nu = 3 allows ceilings up to 1.34 x the schoolbook value for D_3, so the nu = 3 excesses alone do not refute
(T_G); the nu = 4 cases of §9.1 do.)

### 9.3 Two mechanisms (anatomy: `analyse_junk.py`, `analyse_sectors.py`, `coset_check.py`, `coset_trained.py`, `rev_check.py`)

**(a) Coset relaxation (a sign character sgn plus a real irrep fixed by the twist, pi (x) sgn ~ pi: every 2-dim irrep of D_n; the
pair chi_{N/4} of Z_N when 4 | N; the 6 of S5; the 2 of S4).** In the D_3/D_7 ascent optima the
junk of every 2-dim irrep lives ONLY on sign-inconsistent triples (sgn(c) != sgn(a) sgn(b)): residual 2-3e-4 of |z_rho| on
consistent triples, 0.7071 on inconsistent ones. The sign character already separates the two cosets of the index-2
subgroup, so a feature need only be a correct class function on consistent triples. In the Fourier block this relaxes
"X_rho in span(T_rho)" to "the Sigma-even part of X_rho is a multiple of T_rho", where Sigma (conjugation by the intertwiner
of rho and rho (x) sgn) fixes T; the Sigma-odd half (parity sectors with an odd number of anticonformal factors) is free. The
optima carry each 2-dim feature on 4 neurons (schoolbook 8), X = T + J with J Sigma-odd and |J| = |T| (junk fraction exactly
1/sqrt 2, uniform over the 8 parity sectors), and pay for it with a larger sign share. **The count changes**: the D_7 ascent
optimum uses sign + all three 2-dim irreps (K = 4) where the symmetric argmax is {rho_2, rho_3} (K = 2).

**(b) Aligned junk without cosets (A5, and S4's std).** In the GX-trained nets that beat the ceiling the junk is spread over all
triples (A5 has no sign; S4 std: 50/50 consistent/inconsistent), the block is X = alpha T + J with |J|/|X| = 1/sqrt 3 exactly
for every 3-dim block (3a, 3b of A5; std of S4), and J is NOT the reversed-task invariant tr(W E_2^T E_1^T) (fitting both leaves the
residual unchanged). The values 1/sqrt 2 (d = 2) and 1/sqrt 3 (d = 3) fit X = sum of d^2(d-1) mutually orthogonal, fully aligned
atoms of modulus d/(d-1) (price ((d-1)/d)^{1-r} of the schoolbook) — a HYPOTHESIS consistent with both measured junk fractions,
not verified atom by atom.

### 9.4 What this means

* Theorems G1 and G2 stand. But on non-abelian groups G2 is the ceiling of the **junk-free** networks only; the max-margin
  network carries structured same-irrep junk and can sit >= 1.6 x above it (even nu), with a different irrep count (D_7).
* The count predicted by G2 is therefore not "the" max-margin count on non-abelian groups. The true optimum lies between G2 and
  G1 (a factor up to kappa^{s}, e.g. 8 x for d = 4 at nu 3) and is not characterised here.
* Abelian groups without a sign-fixed pair (odd order; Z_6, Z_10) behave like Z_p: no ascent run exceeded the (H) ceiling. Z_12
  (4 | 12) fails through mechanism (a). Odd primes p — the paper's cyclic setting — have no sign character, so nothing here
  touches the cyclic theory as stated for prime p.
* S5: the symmetric optimum (std_sign alone) did not move under junk in these tests. It contains no sign component (a small
  sign component is second order, s > 1), the only sign-fixed S5 irrep is the expensive 6 (6 (x) sgn ~ 6), and mechanism (b) was
  only seen on two-irrep supports (3a + 3b, std + sign). So there is no known mechanism that beats std_sign on S5, but no proof
  either: the S5 global optimum is OPEN (random-init ascents reached only 0.465, in the mixed std + std_sign + sign + ... state
  that the GX free runs also occupy).

## 10. What the paper can safely claim (S5 and non-abelian groups)

**Safe (theorems, proofs in §2–§5, every identity fp64-checked):**
1. *Master bound (every finite group, every gated chain, any n, any width, q in [2, nu]):*
   gamma_q <= (N nu)^{-nu/2} Gamma(c), a finite programme over the real irreps with coverage d - Re chi and price
   eps^{q/2-r} d^{q/2}; Z_p is the special case B Gamma_s(p). The programme is solved EXACTLY by vertex enumeration (Lemma V),
   and its optimal vertex's support is the feature count.
2. *Symmetric (junk-free) class:* the maximum normalized margin over networks whose same-irrep Fourier blocks are
   multiples of the invariant (iterated matrix-multiplication) tensor is exactly (N nu)^{-nu/2} Gamma(c^kappa), attained; a clean
   feature on a d-dim irrep is a decomposition of the matrix-multiplication tensor, and **at even nu the schoolbook is optimal:
   kappa(d) = d^{nu(1-r)}, d^nu neurons per feature** (Theorem K2). At odd nu, d^{(nu-1)(1-r)} <= kappa(d) <= d^{nu(1-r)}.
3. *S5 in the symmetric class:* exactly one feature, the 4-dim irrep std_sign, with no sign patch — exact at every even nu and
   every q in [2, ~0.9 nu); at nu = 3, q = 2 the count "one irrep, no patch" holds for every admissible kappa (certificate), the
   identity std_sign under the (numerically supported) schoolbook kappa. If a sign patch is used on a std-led solution its
   amplitude is exactly 1/8 (vertex coordinate, price-free); S4: 1/6.
4. *D_n in the symmetric class:* the count equals the cyclic count on Z_n and the sign irrep is never used, for every kappa.
5. *Abelian groups:* ELEGANT's (H)-theorem holds verbatim on every finite abelian group (no 2-torsion defect); real characters
   are discounted by nu^{s-1} 2^{nu/2-1}. *Complex weights:* Theorem C is an unconditional exact global theorem for every finite
   abelian group, count = K*(p, nu/q) on Z_p.
6. *q = nu (Morwani's L_{2,nu}):* two-sided exact on every group; dense on S5 (nu 3) and S4, but sparse on A5 and on S5 at nu >= 4.

**Not safe — must NOT be claimed:**
* "The max-margin network on a non-abelian group has K* features" without the qualifier "among symmetric (junk-free)
  networks". It is false on D_3 and S4 (explicit networks 1.27-1.62 x above the symmetric optimum at nu = 4, where that optimum is
  exact), and the count changes on D_7. For S5 it is open.
* "(T) holds for cyclic / abelian groups" in general: it fails on Z_12 at nu 4 (1.06 x, coset mechanism). State the cyclic
  conjecture for odd moduli (or 4 not dividing N); the paper's prime-p results are unaffected. Z_120 (4 | 120) predictions are
  symmetric-class statements.
* "The S5 count is proven" in the global sense. What is proven is the symmetric-class count (and at nu = 3 its kappa-robustness).
* "The theory predicts the S5 sign patch": the symmetric theory's S5 argmax has NO patch; the 1/8 patch belongs to the runner-up
  {std, sign} (0.944 at nu 3, 0.978 at nu 4, 0.998 at nu 6).

**Suggested wording for the group paragraph.** "For group composition the same argument gives an exact margin ceiling in which
each irreducible representation pays a price set by how cheaply the matrix-multiplication tensor can be written as a sum of
products (at even degree: the schoolbook algorithm, d^nu neurons). Among networks without same-irrep interference this
ceiling is attained and fixes the feature count (one irrep on S5, the Galois pair on A5, as observed). Unlike on cyclic groups of
prime order, networks with structured interference can exceed it on some groups (dihedral groups, S4, Z_12), so there the count
is exact only for the interference-free class."

**Data consistency (not proof).** A5 (GX bil, nu 3): symmetric argmax 3a + 3b = 32/32 free runs, fixed-support order
3a+3b > 3a > 4 > 5 = the schoolbook ladder. S5 (GX bil, nu 3): symmetric argmax std_sign alone; free runs use std + std_sign + sign
at 0.30 of the ceiling — the support the coset mechanism would use, but the trained junk is not coset-confined (§9.3), so these
nets are at neither optimum. S4: near-tied top three supports, data uses all four irreps. D_113 (ReLU only): the "parity patch"
is absent from the symmetric optimum and present in the coset mechanism. Z_120 / Z_11^2: data inside the flat nu 3 plateau.
All 189 GX bil nets satisfy G1 and G1' (no violation; `trained_check.py`).

## 11. Corrections to inherited material

| where | statement | correction |
|---|---|---|
| RESULTS_GX §0 | true price of irrep rho is "in [d^{1/3}, d^{4/3}]" (GX normalisation) | the junk-free price is in [d^{1/3} d^{(nu-1)(1-r)}, d^{1/3} d^{nu(1-r)}] = [d, d^{4/3}] at nu 3 q 2 (Theorem K3), and EXACTLY d^{4/3}-type (schoolbook) at even nu (K2) |
| RESULTS_GX §3, §4.4 | "the attained price is the junk-free CP cost of <d,d,d>" | trained nets beat the junk-free ceiling: A5 fixed-support 3a+3b at 1.089 x, S4 std+sign at 1.139 x, both with junk fraction 1/sqrt 3. The schoolbook LADDER ORDER is right (A5); the nets are not junk-free |
| RESULTS_GX `kappa()` naive, complex-type | price 2 (4-neuron {0, pi/2} block) | the (H)-price of a complex pair is nu^{1-r} 2^{q/2-r} = 6^{1/3} = 1.817 at nu 3, q 2 (ELEGANT's 3-neuron block; same units as GX's bound value 2^{1/3}); only the real-character discount changes (§7.2) |
| RESULTS_S5 / GX (c) | S5 sign patch predicted at 0.125 | 1/8 is exact but belongs to the runner-up vertex {std, sign}; the gated-chain symmetric argmax is std_sign with no patch |
| coordinator item 3 | "balanced Strassen ~ 11.5 > 8" | raw Strassen is 11.52; balanced over GL_2^3 it is exactly 10.000 (still > 8) |
| VERIFY / GOPT_STATE3 | "Gamma_s(p) has no optimality certificate" | Lemma V is one: exhaustive vertex enumeration (reproduces Gamma_{3/2}(11) = 0.535725926, K = 2); practical up to p ~ 17 without a double-description code |
| ABSTRACT | Morwani: L_{2,nu} "uses all features for ... group composition" | in the gated chain at q = nu the exact optimum is sparse on A5 (nu 3) and S5 (nu >= 4); check Morwani et al.'s hypotheses (their network, and any condition on G) before citing it for groups |
| programme.py (this directory, fixed during the session) | the unrestricted price of complex pairs used ELEGANT's kappa | fixed before any G1 number quoted in this file; the Z_7 "G1" column of `out/junk_nu4.log` predates the fix and is wrong (its ceiling column is right); S5/S4/A5/D_n numbers never touched complex pairs |

## 12. Open problems (ranked for the paper)

1. **S5 global optimum**: does a sign + coset construction (std and std_sign blocks jointly, A5-irrep features) beat std_sign?
   A direct construction in the style of `coset_construct.py` (features of the index-2 subgroup A5 extended to S5, sign neuron
   for the cosets) would settle it in an afternoon. If it does, the S5 count in the global sense is "one A5-type feature + sign".
2. **A theory of the coset relaxation**: price of the Sigma-relaxed tensor (numerically 4 atoms for d = 2, = the K3 lower bound
   d^{(nu-1)(1-r)} at nu 3), plus the sign tax needed to dominate the inconsistent-triple junk; a "coset programme" should replace
   G2 on groups with a sign character.
3. **Mechanism (b)** on simple groups (A5): verify the d^2(d-1)-atom structure. At nu 3 the escape from {3a, 3b} reaches 1.122 x
   (conditional on kappa(3) = 3); at nu 4 the A5 optimum is a single irrep and did not move, so there is no rigorous simple-group
   counterexample yet. The natural rigorous test is A5 at nu 4 with q in [2.175, 3.825), where the exact optimum is the pair 3a + 3b.
4. kappa at odd nu: close the factor d^{1-r} between K3 and the schoolbook (needs a non-flattening argument).
5. The global (T) for Z_p remains open (unchanged), and holds in the complex model (Theorem C).

## 13. Reproduce

    uv run python verify_fourier.py                       # F0-F5: FS indicators, level-set formula, block, Plancherel, Schur
    uv run python programme.py && uv run python qscan.py   # exact programmes (S5, S4, A5), q scans
    uv run --with mpmath python robust_cert.py             # Theorem R certificates
    uv run python construct_verify.py                      # attainment of G2 (8 constructions)
    uv run --with scipy python kappa_mm.py --nu 3 --q 2 --d 2 3 4 --starts 3   # kappa upper bounds (see run_kappa.sh)
    uv run --with scipy python prog_large.py               # Z_120, Z_11^2, Z_113, D_113
    uv run python complex_closure.py                       # Theorem C checks
    uv run --with safetensors --with scipy python trained_check.py   # GX nets vs G1 / G1' / G2
    uv run --with mpmath python counterexample_d3.py       # certified (T_G) counterexample, D_3 nu 4
    uv run --with scipy python coset_construct.py          # explicit coset construction
    uv run python junk_ascent.py; uv run python junk_nu4.py; uv run python junk_abelian.py; uv run python junk_s5.py
