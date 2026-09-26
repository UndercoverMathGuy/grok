# ARCH: a change of architecture for (T) -- operators, Schatten quasi-norms, and the absolute chain matrix

Ninth pass (sections 1-8) and tenth pass (section 10, after the outage; status in section 9). Continues `GOPT_STATE{,2,3}.md`, `ELEGANT.md`, `ELEGANT2.md`, `DEPTHA.md`, `P3B.md`, `P3C.md`,
`VERIFY.md`. Local CPU, fp64, no pods, $0. New files only: `ARCH.md`, `arch_*.py`, `FINAL_RUNS/ARCH/`.
Nothing inherited edited, no commits. Written incrementally; section 9 is always the current status.

Forbidden by brief and not used: junk energy eta, Cl2_r curves, peeling, kill points, cell certificates,
partition test densities.

---

## 0. Symbols (plain reading with each)

| symbol | meaning | plain reading |
|---|---|---|
| `n`, `D = nu = n+1` | operands, homogeneity degree = chain dimension | how many numbers are added; everything lives in `C^D` |
| `q in [2,nu]`, `r = q/nu in (0,1]` | penalty exponent and its ratio | `q=2` weight decay; `r<1` is why splitting a feature over neurons costs |
| `c_l in C`, `v_lj in T` (`l=1..N`, `j=1..n`) | atom amplitude, slot phase | one neuron's Fourier datum at one frequency; `N` arbitrary, no purity needed |
| `C_T = sum_l c_l prod_{j in T} v_lj` | pattern amplitudes | `A := C_empty` clean, the other `2^n-1` are junk |
| `h(u) = sum_l c_l prod_j (1+u_j v_lj) = sum_T C_T u^T` | generating polynomial | what the feature outputs under a probe `u in T^n` |
| `m = min_{T^n} Re h`, `hmin = min_{T^n} |h|` | signed aimed output, unsigned aimed output | `m <= Re A`; `hmin >= m` |
| `m_l = (1, v_l1, v_l1 v_l2, ..)` | chain vector, all entries unimodular, `||m_l||^2 = D` | running phase products |
| `M = sum_l c_l m_l m_l^*` | chain matrix (ELEGANT.md) | diagonal `A`, below-diagonal = interval junk `L`, above = mirror moments `U` (not determined by `h`) |
| **`M_+ = sum_l |c_l| m_l m_l^*`** | **absolute chain matrix** (new) | the chain matrix of the population with every amplitude replaced by its modulus; Hermitian PSD, trace `D*S`, `S = sum|c_l|` |
| `sigma_j(X)`, `lambda_j(X)` | singular values, eigenvalues | |
| `||X||_{S_r}^r = sum_j sigma_j(X)^r` | Schatten-`r` quasi-norm | for `r<1` it rewards low rank, like the penalty rewards few neurons |
| `E_k(X)` | coefficient of `t^k` in `det(I+tX)` | `k`-th elementary symmetric function of the eigenvalues |

---

## 1. The one-line P1 proof (coordinator's seed (a)) -- CHECKED, correct

> **THEOREM P1' (junk-free feature cost, every N, n, r).** If `C_I = 0` for every non-empty *interval*
> pattern `I` then `sum_l |c_l|^r >= D^{1-r} |A|^r` for every `r in (0,1]`.

*Proof.* `M` is upper triangular with diagonal `A` (ELEGANT Lemma 1), so all `D` eigenvalues equal `A`.
Weyl's majorant theorem (`sum_j |lambda_j(X)|^r <= sum_j sigma_j(X)^r`, every `r>0`) and Rotfel'd's
theorem (`X -> sum_j f(sigma_j(X))` is subadditive for every concave non-decreasing `f`, `f(0)=0`, all
matrices, not only PSD) give

    D |A|^r = sum_j |lambda_j(M)|^r <= sum_j sigma_j(M)^r <= sum_l sum_j sigma_j(c_l m_l m_l^*)^r = sum_l (D|c_l|)^r.  QED

*Plain reading: the cost is a Schatten quasi-norm of a sum of rank-one per-neuron operators; the clean
amplitude is its spectrum; "spectrum <= singular values <= sum of pieces".* No Cauchy-Binet, no Hadamard,
no Bernstein integral.

Remarks for the paper.
* Generalisation: the same two lines give `sum_l f(D|c_l|) >= D f(|A|)` for every `f` that is concave,
  non-decreasing, `f(0)=0` (Rotfel'd) **and** with `t -> f(e^t)` convex (Weyl). This covers `x^r`,
  `log(1+x)`. It correctly EXCLUDES `min(x,theta)` (`f(e^t)` not convex) -- which is the capped-mass
  conjecture that GOPT_STATE lists as FALSE. So the hypothesis is the natural one.
* Equality (`r<1`): Weyl equality forces `M` normal, triangular + normal = diagonal = `A I`; Rotfel'd
  equality forces the rank-one pieces to have orthogonal ranges: orthonormal chain frame, equal moduli.
  Same equality class as ELEGANT Cor 1.2.
* Numerical check: `arch_p1.py` (section 8).

The structural reading that the rest of this note develops: **COST = Schatten-r quasi-norm of a sum of
per-neuron rank-one operators; SIGNAL = spectral data of the sum.**

---

## 2. The Schatten cost principle (PROVED; any junk, any N, impure neurons)

> **PROPOSITION S0.** For every population (any `N`, any junk, complex `c_l`), every `r in (0,1]`:
>
>     D^r sum_l |c_l|^r  >=  tr(M_+^r)  >=  ||M||_{S_r}^r  >=  sum_j |lambda_j(M)|^r .

*Proof.* First: Rotfel'd for the PSD rank-one pieces `|c_l| m_l m_l^*`. Second: `M = X^* Phi X`,
`M_+ = X^* X` with `X` the `N x D` matrix of rows `sqrt|c_l| m_l^*` and `Phi = diag(c_l/|c_l|)` unitary;
Horn: `prod_{j<=k} sigma_j(X^* Phi X) <= prod_{j<=k} sigma_j(X)^2 = prod_{j<=k} lambda_j(M_+)`, and
log-majorisation implies `sum sigma_j^r <= sum lambda_j(M_+)^r`. Third: Weyl. QED

> **COROLLARY S1 (network level, impure neurons included).** For any gated-chain network, with atoms
> `c_ik = prod_j ehat_ij(k) conj(what_i(k))` and phases `v_ijk` (Theorem D's dictionary), and
> `M_k^+` the absolute chain matrix of the atoms at frequency `k`:
>
>     (p/(2 nu))^{q/2} P_q  >=  sum_{k in H} sum_i |c_ik|^r  >=  D^{-r} sum_{k in H} tr((M_k^+)^r).

*Proof.* Per neuron, generalised Hoelder with `nu` factors: `sum_k |c_ik|^{2/nu} <= prod_j (sum_k|ehat_ij(k)|^2)^{1/nu} (sum_k |what_i(k)|^2)^{1/nu}`,
AM-GM, Parseval (`sum_{k in H}|ehat(k)|^2 <= p||e||^2/2`); for `q>2` use `sum_k x_k^{q/2} <= (sum_k x_k)^{q/2}`.
This is VERIFY Link 5 step 4; nothing about purity is used. Then S0 per frequency. QED

*Consequence: obstruction (iv) (impure neurons) does not exist on the cost side. The network's cost
dominates a spectral functional of `K` small PSD matrices. All the difficulty is: **what does margin >= mu
force on the spectra of the `M_k^+`?***

---

## 3. CURRENT BEST CANDIDATE: Conjecture (E+)  -- the absolute chain matrix

> **CONJECTURE (E+).** For every population (any `N`, any junk), with `m = min_{T^n} Re h`:
>
>     det(I + t M_+) >= (1 + t m_+ )^D   for all t >= 0,      (m_+ := max(m,0))
>     i.e.  E_k(M_+) >= C(D,k) m^k ,  k = 1..D.

**Why it is the right object.**
* `(E+) => C1` (Re form), every `N`, every `n`, every `r`, in one line and with NO Cauchy-Binet: by S0 and
  the Bernstein representation `x^r = c_r int log(1+xt) t^{-1-r} dt`,
  `D^r cost >= tr(M_+^r) = c_r int log det(I+tM_+) t^{-1-r} dt >= D m^r`. So `cost >= D^{1-r} m^r`.
  (It also implies ELEGANT2's ladder C2 in Re form, `e_k(|c|) >= C(D,k)(m/D)^k`, by Cauchy-Binet+Hadamard with
  all terms positive.)
* **Sharp at the block**: `M_+ = A I`, `m = A`, equality for every `k`.
* It is phrased in neither `|A|` nor eta: obstruction (ii) is respected. And it **breaks the twist symmetry in
  the right way**: the `hmin` version of (E+) is FALSE at `n=2` (measured `det M_+ / hmin^3 -> 0`, because the
  twisted block has chain vectors `(1, w^{-l}, 1)`, a degenerate frame), while the `Re` version survives.
  The network only ever needs the `Re` version (an all-junk feature has negative margin).
* `M_+` is PSD, so the refutations of ELEGANT2 sec 6.1-6.2 (complex `M`: `|det M|/hmin^D -> 0`,
  `Re spec(M) >= m` false by a factor -350) do not apply: they are all driven by LARGE MIRROR moments, and
  in `M_+` the mirror is the adjoint of the junk of the absolute population.
* What is FALSE: `lambda_min(M_+) >= m` (measured ratio down to `1e-5` at `n=1`, `0.002` at `n=2`). So (E+)
  is genuinely a statement about elementary symmetric functions / `log det`, not about the bottom eigenvalue.

**Numerics so far** (`arch_mplus.py`, `arch_eplus.py`; Nelder-Mead adversary minimising each ratio
`E_k(M_+)/(C(D,k)m^k)`; block-perturbed, random and aligned starts; `FINAL_RUNS/ARCH/mplus.json`, `eplus_*.json`):

| n | N | min ratio k=2 | k=3 |
|---|---|---|---|
| 1 | 2 | 1.000000 | - |
| 1 | 3 | 1.000005 | - |
| 1 | 4 | 1.001148 | - |
| 1 | 6 | 1.017045 | - |
| 2 | 3 | 1.000174 | 1.000001 |
| 2 | 4 | 1.001629 | 1.008097 |
| 2 | 5 | 1.025465 | 1.085013 |
| 2 | 6 | 1.020806 | 1.309439 |
| 2 | 8 | 1.025694 | 1.017205 |

Never below 1; minimum attained at the block. `hmin` variant: `n=1` holds (1.000000), `n=2` FAILS
(`k=3`: 0.000, `k=2`: 0.972) -- recorded as a refutation.

**Proof status.** OPEN. Facts:
* `n=1`, `N=2`: exact identity `det M_+ = exp(2H)` (H = mean of `log|h|` over the two kill points), so
  `det M_+ >= hmin^2`. (This is Theorem A rewritten.)
* `n=1`, all `N`: `det M_+ = (S-|P|)(S+|P|)`, `P = sum|c_l|v_l`; Cauchy-Schwarz gives `hmin^2 <= 2S(S-|P|)`,
  which is short of the claim by the factor `2S/(S+|P|)`. Not closed yet.
* `N=D`, `n>=2`: `det M_+ = prod|c_l| |det[m_l]|^2`, so with ELEGANT2's chain identity (E+)_{k=D} becomes
  a comparison between the chain determinant and the geometric mean of slot Vandermondes. Not closed.

---

## 4. Seed (b) evaluated: compressions of multiplication operators

Setting: `f = Re h` on `T^n`, `V` a `d`-dimensional subspace of `L^2(T^n)`, `T = P_V M_f P_V` Hermitian with
`lambda_min(T) >= m`; `T = sum_l T_l`; Rotfel'd gives `d m^r <= sum_l ||T_l||_{S_r}^r`.

**4.1 What the duality barrier becomes.** `lambda_min(T_V) = min over test densities of the form |psi|^2,
psi in V`. So one eigenvalue = one ADAPTIVE test measure (that is P3C's aiming, in cleaner clothes). What is
beyond test measures is the simultaneous use of all `d` eigenvalues + non-commutative subadditivity. The
functional `z -> ||L(z)||_{S_r}` is non-convex, so P3B's barrier (`conv K_q = K_nu`) does not forbid it:
the convex hull of the `S_r` ball is the `S_1` ball, exactly mirroring `conv K_q = K_nu`.

**4.2 PROVED: the chain space with shift 1/2 is the unique sharp translation-invariant choice, and it gives**

> **THEOREM H (aligned populations).** If all `c_l > 0` (after a common rotation) then `M = M_+ = 2H - A I`
> where `H = P_V M_{Re h} P_V` on the chain monomials, hence `lambda_j(M_+) >= 2m - A` and
> `cost >= D^{1-r} (2m - A)_+^r`, every `N`, any junk. Sharp at the block, but in the wrong variable
> (`2m-A <= m`).

Uniqueness: for `V` spanned by characters `S subset Z^n`, the per-atom Hermitian compression is
`c_l D_l (I + Adj_S/2) D_l^*` with `Adj_S` the graph on `S` joining `a,b` iff `a-b in +-({0,1}^n minus 0)`.
Sharpness at the block forces `(1-s)I + Adj/2` to be a multiple of a projection of rank `|S|/D`, i.e.
`Adj_S` = disjoint union of cliques `K_D`; cliques of this graph are exactly chains of nested sets, so
`S` = a chain and the scalar shift is `s = 1/2`. Nothing else is sharp.

**4.3 PROVED NO-GO: no fixed (or Haar-averaged) Hermitian compression is sharp, already at n=1.**
Sharpness at `s=0` requires `P_V M_{Re g_w} P_V = D * (projection)` for a.e. atom position `w`. At `n=1`,
`Re g_w = 1 + cos(theta+alpha) in [0,2]` and `D = 2 = max`, so every range vector must be supported where
`cos = 1`, a single point, for every `alpha` simultaneously: impossible. For non-aligned atoms (phase `phi`)
the chain compression is `|c|(cos(phi) J + sin(phi) K)/2` with `K = i sgn(k-k')` of full rank; at `n=1`,
`phi = pi/2` its `S_r^r` is `2^{1-r}` times the aligned value -- exactly cancelling the gain `D^{1-r}`.
This is the operator analogue of GLOBAL_OPT4's Theorem O. **Seed (b) in fixed-V form is dead for sharpness;
the adaptive-V form at N = n+1 is ELEGANT2's Theorem A (V = point masses at kill points).**

**4.4 The coordinator's computation (c) re-derived.** Analytic (non-Hermitian) compression of `M_{g_l}` on
chain monomials `= c_l D_l L_0 D_l^*`, `L_0` = all-ones lower triangular: correct. Its eigenvalues are all
`A` WHATEVER the junk, so eigenvalues carry no information there; `sum sigma^r >= D|A|^r` by AM-GM on
`prod sigma = |A|^D`. The loss `||L_0||_{S_r}^r > D` is unavoidable (4.3).

**4.5 Why M is not a compression of M_h.** `m_l m_l^*` is the chain compression of the REAL function
`2 Re g_l - 1` times the complex number `c_l`; so `M` is the compression of `F = h + h^# - A`,
`h^#(u) = sum_T U_T conj(u^T)` the mirror polynomial. Numerical range of `M` sits in `conv F(T^n)`, not
`conv h(T^n)`: this is the exact reason ELEGANT2's R2/R3 fail. `M_+` replaces `F` by the real function
`2 Re h_+ - S` of the absolute population.

---

## 5. Matrix reductions with FREE mirror completion (tested; true in the needed range, but not natural)

> **(S)**: for every `D x D` matrix `M = A I + L + U` (`L` strictly lower = junk of `h`, `U` strictly upper
> ARBITRARY): `||M||_{S_r}^r >= D m(h)^r`.

If true at a given `(n,r)` it proves C1 (Re form) for every `N` via S0, with no reference to atoms.
* `n=1`: TRUE for `r >= r_0 = 0.44`, FALSE below (rank-one completion `U = 1/C`: ratio 0.939 at `r=0.4`,
  0.802 at `r=0.3`; grid search `arch_schatten2.py`). So the Schatten norm of the complex `M` is a strictly
  weaker proxy than the true atomic cost (which obeys C1 for every `r`, ELEGANT2 Thm B): **flatness of the
  atoms (unimodular entries) matters at small r**. This is what `M_+` + (E+) repairs: (E+) is conjectured for
  all `r` at once because it is `r`-free.
* `n=2`: no violation for `r in {0.4,0.5,0.6,2/3,0.8,0.9}` (Powell + Nelder-Mead, 150 structured starts
  including rank-one and Hermitian completions); minimum = 1 at `M = I`. Near `I`: `sum sigma^r = D -
  (1-r)||L||_F^2 + ..` at the best `U = ((2-r)/r) L^*`, second order, against the first-order loss
  `m <= 1 - max|C_T|`.
* Verdict: a computer-closable route at `q=2`, `nu=3`; threshold in `r` is not natural; superseded by (E+).

> **(S_K)** (K >= 2, n=2): `min_{a,D} sum_k Re[(1-w^{kD}) h_k(w^{-2ka})] <= Gamma_{1/r}(p) (sum_k ||M_k||_{S_r}^r / D)^{1/r}`
> with every `U_k` free.

`arch_SK.py`, `arch_SK2.py` (torch, softmin ascent): `p=7`: best `0.9999988`; `p=11`: best `0.962` -- but the
optimiser does not hold the (non-symmetric, 2-frequency) Gamma optimum at `p=11`, so this is a weak test.
No violation seen. Needs an epigraph solver to be meaningful.

---

## 6. The margin side for K >= 2 in operator form (what is proved, and what it is equivalent to)

For any network (impure included) and each `D != 0`, `F_D(a) := z_{s(a)}(a) - z_{s(a)-D}(a) >= margin` on
`Z_p^n`, so every Toeplitz matrix `[Fhat_D(a-b)]_{a,b in S}` is `>= margin * I`.
* `S` = chain of frequency `k`: **operator Lemma Q**: `margin <= g(D) + lambda_min(Herm(L_k(D)))`, `g(D)` the
  clean level-set margin, `L_k(D)` the strictly-lower matrix of level-`D` junk coefficients at the modes
  `2k 1_I` (for pure networks `= W_{kD} L_k / 2^n`). One frequency's interval junk acts jointly, not one mode
  at a time.
* `S` = product of chains over the support (`D^K` characters, if distinct = dissociate): the compression is
  a KRONECKER SUM, so `margin <= g(D) + sum_k lambda_min(Herm(W_{kD} L_k))/2^n`: damage ADDS across
  frequencies -- obstruction (iii) handled -- but the eigenvector is a product and `|psi|^2` is a Riesz
  product: this is equivalent to P3C's product densities, with the same dissociativity requirement. Cleaner,
  not new.

---

## 7. Dead ends (with breaking points)

1. Fixed/Haar-averaged Hermitian compression: not sharp at `n=1` (4.3).
2. Character-spanned `V` other than a chain, or shift `s != 1/2`: not sharp at the block (4.2).
3. `h`-determined rank-one kernels (flattenings `Phi^T Z X` of the coefficient tensor): the block maps to a
   rank-one matrix, bound collapses to the trivial `cost >= |A|^r`. The mirror is ESSENTIAL to see `n+1`.
4. `2^n`-dimensional full monomial space (`n_l = (v^T)_T`): block frame not orthogonal
   (`<n_l,n_l'> = (1+w^{l'-l})^n != 0`), Rotfel'd strict, not sharp.
5. `(S)` with free `U` at small `r` (n=1, `r<0.44`).
6. `lambda_min(M_+) >= m`: false (ratio `1e-5`). `(E+)` with `hmin`: false at `n=2` (twist).
7. Diagonal similarity `S M S^{-1}`: per-atom norm becomes `|c| ||s|| ||1/s|| >= D|c|`, equality only for
   constant `s`; cannot help at the block.

---

## 8. Numerical record

| script | what | output |
|---|---|---|
| `arch_schatten.py`, `arch_schatten2.py` | (S) free-U, n=1,2 | `schatten_free_U.json`, `schatten_free_U_v2.json` |
| `arch_SK.py`, `arch_SK2.py p r nstart seed` | (S_K), n=2 | `SK_*.json`, `SK2_*.json` |
| `arch_mplus.py` | first test of M_+ conjectures | `mplus.json` |
| `arch_eplus.py n N m|hmin nstart seed` | (E+) adversary | `eplus_*.json` |

---

## 10. Tenth pass (continuation after the outage): (E+) reduced, n = 1 PROVED, dual form found

New scripts `arch_core2.py`, `arch_det.py`, `arch_pw.py`, `arch_ls.py`, `arch_local.py`, `arch_local2.py`;
outputs `FINAL_RUNS/ARCH/det_*.json|log`, `pw_*.json`, `ls_*.json`, `local*_*.json`.

### 10.1 PROVED: (E+) is its top rung. Maclaurin.

`M_+` is PSD, so with eigenvalues `lambda_j >= 0`, Maclaurin's inequality gives
`E_k(M_+) >= C(D,k) (det M_+)^{k/D}` for every `k`. Hence

> **(E+)  <=>  det M_+ >= m^D**   (whenever `m > 0`).

*Plain reading: it is enough to control the geometric mean of the eigenvalues of the absolute chain matrix.*
(Consistent with the predecessor's tables: the `k = D` ratio is always the smallest at a minimiser.)

### 10.2 PROVED: (E+) at n = 1, every N.

`D = 2`, `M_+ = [[S, conj P],[P, S]]`, `P = sum_l x_l v_l`, `x_l = |c_l|`, `det M_+ = S^2 - |P|^2`.

*Step 1 (probe bound; every n).* For every probability measure `rho` on `T^n`, with
`Lambda_rho(v) := int prod_j (1 + u_j v_j) drho(u) = sum_T rhohat_T v^T`,

    m  <=  int Re h drho  =  Re sum_l c_l Lambda_rho(v_l)  <=  sum_l x_l |Lambda_rho(v_l)| .

*Step 2 (centroid).* At `n = 1`, `Lambda_rho(v) = 1 + z v` with `z = rhohat_1` ranging over the closed unit disc.
Take `z = -conj(P)/S` (`|z| <= 1`). Then `|1 + z v_l| = |v_l - P/S|` (unimodularity) and by Cauchy-Schwarz

    m^2 <= ( sum_l x_l |v_l - P/S| )^2 <= S * sum_l x_l |v_l - P/S|^2 = S (S - |P|^2/S) = S^2 - |P|^2 = det M_+ .  QED

*Plain reading: aim the probe at the reflected centroid of the absolute population; the loss of the probe is
the population's variance about its centroid, which is exactly the determinant.* Equality class at `n=1` is large
(e.g. `x = (1/2,1/2)`, `v = (1, e^{i a})` with the right amplitude phases: `det = m^2 = sin^2(a/2)`), so the
`n = 1` case is NOT rigid at the block; the rigidity of (E+) for `n >= 2` (numerics: minimum only at the block)
is a different regime.

### 10.3 PROVED: three equivalent forms. (E+) <=> (F) <=> (PW).

Let `mu = sum_l x_l delta_{v_l}` (the absolute population), `M_mu := M_+`,
`F(mu) := inf_rho sum_l x_l |Lambda_rho(v_l)|` (inf over probability measures `rho` on `T^n`), and
`q_B(v) := m(v)^* B m(v)` for `B` a `D x D` PD matrix (a real trig polynomial on the torus supported on interval
patterns; `int q_B = tr B`).

> **(F)**   `det M_mu >= F(mu)^D` for every positive `mu`.
> **(PW)**  for every `B > 0` with `det B = 1` there is a probability measure `rho` on `T^n` with
>           `|Lambda_rho(v)| <= q_B(v)/D` for all `v in T^n`.

*Proofs.* (E+)⇒(F): the payoff `Re int int g(v) prod_j(1+u_j v_j) drho(u) dmu(v)` is linear in `rho` and in the
amplitude function `g` (`|g| <= 1`, weak* compact convex); Sion gives `F(mu) = max_{|g|<=1} min_u Re h_g(u)`, i.e.
`F(mu)` is the largest signed aimed output of a population with moduli `<= x`; monotonicity of `det` in the
weights and (E+) for that population give `det M_mu >= F(mu)^D`. (F)⇒(E+): `m(c) <= F(|c|)`.
(F)⇔(PW): `det(M)^{1/D} = (1/D) min{tr(MB): B > 0, det B = 1}` (AM-GM; minimiser `B* = det^{1/D} M^{-1}`) and
`tr(M_mu B) = int q_B dmu`; so (F) says `inf_mu sup_rho int (q_B/D - |Lambda_rho|) dmu >= 0` for every `B`,
which by Sion (linear in `mu`, concave in `rho`) equals `sup_rho min_v [q_B(v)/D - |Lambda_rho(v)|] >= 0`.

*Plain reading: (E+) is equivalent to a population-free statement about `D x D` positive matrices: every
positive-definite quadratic form on chain vectors of unit determinant dominates, pointwise on the torus, the modulus
of the generating polynomial of some probability measure.* At `n = 1` the probe is the Poincaré-disc point of `B`
(`B = cosh(s) I + sinh(s) sigma_theta` ↦ `z = tanh(s) e^{i theta}`) and (PW) is the identity
`(q_B/2)^2 - |1 + z v|^2 = (Re(B_10 conj v) + 2|B_10|^2/tr B)^2`.

### 10.4 Numerics (tenth pass)

* `arch_det.py` (Nelder-Mead + Powell on `det M_+/m^D`, six start families incl. twisted/collapsed/aligned):
  `n=1`: `1.000000000` (N=3, non-block minimiser), `1.0014` (N=6); `n=2`: `1.0000118` (N=3), `1.00043` (4),
  `1.0061` (5), `1.0043` (6), `1.0029` (8), all at the block; `n=3`: `1.11 .. 1.56` (N=4,5,6; weak search, coarse
  inner grid). Never below 1.
* `arch_pw.py` (PW at `n=2`, 16 random `B` of four kinds incl. `B ∝ M_+^{-1}` of populations): the naive probe
  `omega_I = 2 B_I/tr B` (the `n=1` formula) FAILS in 9/16 (slack down to `-0.18`), but an optimised 6-atom
  probe achieves positive slack in 16/16 (min `+4.9e-4`, near-block `B`). At `n=1` the naive probe is exact (12/12).
* `arch_ls.py`: the least-squares chain probe `Lambda_LS = e_0^* M^{-1} m(v)/(M^{-1})_00` (the `n=1` probe in
  another guise) FAILS at `n=2`: pointwise 15/20, even mu-integrated (`int|Lambda_LS|dmu / det^{1/3}` up to
  `1.16`), inadmissible 3/20. So the probe must use NON-CHAIN monomials at `n >= 2`.
* `arch_local.py`, `arch_local2.py` (second order at `B = I + eps H`): the condition is
  `exists zero-mean multi-affine real trig poly g: (2/D^2)(Im R)^2 + g <= ||H_off||^2/D` on the torus,
  `R = sum_{k<k'} H_{kk'} v^{(k,k']}`. With the naive `g` it fails at `n=2` (worst ratio 1.5 vs threshold 1);
  with the LP-optimal `g` the sup over `H` of the violation is `-5.7e-15`: second-order (PW) HOLDS at `n=2` and is
  exactly TIGHT (double tangency on the torus), as at `n=1` where the identity is a perfect square.

### 10.6 PROVED: the upper-triangular contraction theorem (UT). Torus-free.

For `B > 0` (`D x D`) define

    T(B) := max { Re tr Y : Y upper triangular (chain order), [[B, Y],[Y^*, B]] >= 0 } .

> **THEOREM UT.** `T(B) >= D det(B)^{1/D}` for every `B > 0` and every `D`.

*Proof.* The constraint `[[B,Y],[Y^*,B]] >= 0` is equivalent to `Y = B^{1/2} W B^{1/2}` with `||W||_op <= 1`.
The program is an SDP, strictly feasible (`Y = eps I`) and bounded (`Re tr Y <= tr B`), so strong duality holds.
Lagrangian with `Z = [[Z_1, Z_0],[Z_0^*, Z_2]] >= 0`: `Re tr Y + tr(Z [[B,Y],[Y^*,B]]) = tr((Z_1+Z_2)B) + Re tr((I + 2 Z_0^*) Y)`;
boundedness over the free upper-triangular entries forces `I + 2Z_0^*` to be strictly upper triangular, i.e.
`Z_0 = -K/2` with `K` **unit lower triangular** (diagonal 1, strictly lower part free). Dual:
`min tr((Z_1+Z_2)B)` s.t. `[[Z_1, -K/2],[-K^*/2, Z_2]] >= 0`. With `Z_2 = C > 0` the Schur complement gives
`Z_1 >= (1/4) K C^{-1} K^*`, so the dual value is `min_{K,C} [ tr(CB) + (1/4) tr(C^{-1} K^* B K) ]`. For fixed `K` the
convex function of `C` is minimised at the Riccati point `CBC = K^*BK/4` and the minimum is
`2 tr((B^{1/2} (K^*BK/4) B^{1/2})^{1/2}) = tr |B^{1/2} K B^{1/2}| = ||B^{1/2} K B^{1/2}||_{S_1}` (nuclear norm).
Hence `T(B) = min over unit lower triangular K of ||B^{1/2} K B^{1/2}||_{S_1}`. AM-GM on the `D` singular values:
`||X||_{S_1} >= D |det X|^{1/D}`, and `det(B^{1/2} K B^{1/2}) = det B * det K = det B`. QED

*Plain reading: among all `B`-contractions (`Y = B^{1/2} W B^{1/2}`, `W` a contraction) that are upper triangular,
the largest trace is at least `D` times the geometric mean of the eigenvalues of `B`. The dual says the same
thing as: no unit-triangular change of basis can push all singular values of `B` below its geometric mean.*

Complementary slackness (worked out for the record; verified numerically to `1e-8`): the optimal `Y*` is
`Y* = B^{1/2} U B^{1/2}` with `U` the **unitary** polar factor of `B^{1/2} K* B^{1/2}`, `K*` the dual-optimal `K`.
Numerics (`arch_ut.py`, `arch_ut2.py`, `arch_adm.py`): `T(B) >= D` in 65/65 random `B` (det 1) at `n = 1,2,3`,
minimum `3.000011` / `4.0005` near the block (`B` near `I`), `T(B) = D` exactly only at `n = 1` (where (E+) has
its large equality class).

**Consequence (torus-free half of (PW)).** Let `det B = 1` and `Y` be optimal (or any feasible `Y` with
`Re tr Y >= D`; rotate so `tr Y > 0`). For every `v`, the `2x2` compression of `[[B,Y],[Y^*,B]]` by `(m(v), m(v))`
gives `|m(v)^* Y m(v)| <= m(v)^* B m(v) = q_B(v)`, and because `Y` is upper triangular and the chain entries are
unimodular, `m(v)^* Y m(v) = tr Y + sum_{j<k} Y_jk v^{(j,k]}` is an ANALYTIC interval polynomial (the mirror
never enters: `tr(Y U) = 0` for `U` strictly upper). So

    Lambda_Y(v) := m(v)^* Y m(v) / tr Y      satisfies      |Lambda_Y(v)| <= q_B(v)/D    on T^n.

### 10.7 The remaining gap is ADMISSIBILITY, and exactly where it sits (settled by refutations)

(PW) needs `Lambda_Y = Lambda_rho` for a probability measure `rho`; equivalently (M. Riesz extension) the functional
`h -> Re tr(Y M_h^low)/tr Y` (`M_h^low = A I + L`, the lower-triangular chain compression of `h`) is `>= 0` on every
multi-affine `h` with `Re h >= 0` on the torus; equivalently `Lambda_Y` lies in the moment body `Omega_n`.

**10.7.1 REFUTED: "feasible + `Re tr Y >= D` ⇒ admissible."** Explicit extreme points of the feasible set with
`Re tr Y = D` and `|omega_{(0,2]}| = 1.25 > 1` (`n=2`, Wishart `B`, `arch_ellip.py` debug run; LP infeasible and the
elliptope test negative, in agreement). The earlier 156/156 (`arch_adm2.py`) was a sampling artefact.

**10.7.2 REFUTED: "the (UT)-optimal `Y*` is admissible."** For badly conditioned `B` (cond 193 and 527, Wishart)
the optimal probe has `|omega_{(1,2]}| = 1.06 .. 1.09 > 1`; LP infeasible. (The 42 earlier successes were on
`cond <= ~100`.) So the optimal `Y*` cannot be used as such; there is slack (`T(B) = 3.8 .. 4.4` there) but it
must be spent on admissibility.

**10.7.3 PROVED (n = 2): admissibility is an LMI.** Write `w = (u_1, u_1 u_2)`; then the multi-affine moments of a
measure on `T^2` are the entries of the unit-diagonal PSD matrix `R = E[(1,w_1,w_2)(1,w_1,w_2)^*]`
(`R_10 = E u_1 = omega_{(0,1]}`, `R_20 = E u_1u_2 = omega_{(0,2]}`, `R_21 = E u_2 = omega_{(1,2]}`). Conversely,
every `3 x 3` complex PSD unit-diagonal matrix is a convex combination of `z z^*` with unimodular `z`: an extreme
point of the complex elliptope of size `D` has rank `r` with `r^2 <= D` (the face of the PSD cone through a rank-`r`
matrix has real dimension `r^2`, and only `D` diagonal constraints cut it), so for `D = 3` every extreme point is
rank one. Hence

> at `n = 2`: `Lambda_Y` is admissible  <=>  `R_Y := I + (Y_up + Y_up^*)/tr Y >= 0`  (`Y_up` = strictly upper part,
> `tr Y > 0` after a phase rotation). `R_Y >= 0` is NECESSARY at every `n` (it is the chain matrix of the probe measure).

*Plain reading: at two operands a probe is legal iff its own absolute chain matrix is positive semidefinite.* For
`D >= 4` rank-2 extreme points exist and the elliptope is strictly larger than the moment body (and the non-interval
moments must vanish), so the LMI is only necessary there.

**10.7.4 REFUTED as a complete route: the elliptope-constrained matrix program.** `T_ell(B) := max{tr Y : Y UT,
tr Y real, [[B,Y],[Y^*,B]] >= 0, (tr Y) I + Y_up + Y_up^* >= 0}` (an SDP). `arch_ut3.py`: `T_ell(B) >= 3` for
60/60 random `B` (`n=2`, cond up to `1.9e3`; the LMI binds only above cond `~500`), BUT an adversarial descent finds
`B` with one eigenvalue `~4e-6` (cond `5e8`) and `T_ell(B) ~ 0.9 < 3`, and the limit is structural: if `B e ~ 0` for
a generic unit vector `e`, feasibility forces `Y e ~ 0 ~ e^* Y`, which with `Y` UT leaves a one-parameter family
`Y = c Y_0`, and the LMI then forces `c = 0`. The loss is the MATRIX relaxation: `[[B,Y],[Y^*,B]] >= 0` tests all
vectors, whereas (PW) only needs `|m(v)^* Y m(v)| <= q_B(v)` on CHAIN vectors (unimodular entries). Indeed at such a
`B` the (PW) probe is essentially the kill-point mass at `-conj(v_0)` (`e ~ m(v_0)`), which vanishes to second order
at `v_0` and is dominated pointwise, but whose `Y` is far from matrix-feasible.

**10.7.5 The chain-restricted, admissibility-constrained program is (PW) itself (no free lunch).** Replacing the
matrix constraint by the pointwise one and adding the elliptope LMI gives a semi-infinite SDP whose Lagrangian dual
(multipliers: a measure `sigma` on `T^n x T`, a PSD `Z`, a real `lambda`) reads: minimise `int q_B d|tau|` over complex
measures `tau` on `T^n` with `tauhat(empty) = 1 + tr Z - i lambda` and `tauhat((j,k]) = 2 Z_kj`, `Z >= 0`. At `n = 2`
this constraint says exactly `Re h_tau >= 1` on `T^2` (the dual cone of the elliptope is the cone of multi-affine
polynomials with non-negative real part), and `int q_B d|tau| >= D det(B)^{1/D} det(M_+(|tau|))^{1/D}`, so the dual
bound needed is `det M_+(|tau|) >= 1` for every complex population with `m >= 1`: (E+) for the dual population.
The reformulation is an identity, not a lever. (Without the LMI the dual has `tauhat((j,k]) = 0`, `M(tau)` is
unit upper triangular, `det M_+(|tau|) >= |det M(tau)| = 1` by Andréief + Horn -- that is the junk-free theorem P1',
and it is why Theorem UT is provable while the admissible version is not: **Theorem UT is the junk-free (E+) in dual
clothing.**)

**10.7.6 What (PW) at `n = 2` is, concretely.** For every `3 x 3` `B > 0` with `det B = 1` find
`omega = (omega_1, omega_2, omega_12)` with `[[1, conj omega_1, conj omega_12],[omega_1, 1, conj omega_2],[omega_12, omega_2, 1]] >= 0`
and `|1 + omega_1 v_1 + omega_2 v_2 + omega_12 v_1 v_2| <= q_B(v)/3` on `T^2`. Forced structure: first order at
`B = I + eps H` is `omega_I = (2/3) eps H_I`; the second-order correction is not the naive one and the inequality is
exactly tangent (10.4). Explicit admissible candidates tested (`arch_cand.py`): `R_omega = (1-a) I + a corr(B)`
(the `n=1` answer is `a = 1`, `corr(B) = diag(B)^{-1/2} B diag(B)^{-1/2}`) fails at second order for every `a`
(`a = 2/3` is forced and the slack is `-0.02 .. -0.6`); `-corr(B^{-1})` mixtures fail too. Atom-parametrised
optimisation over measures (10 atoms, Powell) finds positive slack in every population-derived `B` tested (cond up to
580), consistent with (E+)'s direct adversary (`det/m^3 >= 1.0000118`, min at the block).

**10.7.7 Consequence for the programme.** The torus-free matrix theorem UT (10.6) proves, for every population,
that some interval polynomial `Lambda` with constant term 1 has `sum_l x_l |Lambda(v_l)| <= det(M_+)^{1/D}` -- the
"phase-blind" half of (F). The whole remaining content of (E+) is that `Lambda` can be taken to be a MOMENT
FUNCTIONAL; at `n = 2` that is one LMI (10.7.3) and the problem is the explicit finite-dimensional statement 10.7.6.

### 10.8 What (E+) buys at network level (for the record; nothing here is new machinery)

* (E+) at `(n, all N)` ⇒ **C1 in Re form** at that `n`: `sum_l |c_l|^r >= (n+1)^{1-r} m_+^r` for every `N`, every
  `r in (0,1]` (S0 + Bernstein, §3). At `n = 2` this is exactly DEPTHA's Corollary DA-1 without the `r >= 0.6`
  restriction and without a certificate.
* `K = 1`, pure single-frequency neurons, **any width**: ELEGANT2 Corollary N1 holds verbatim with the width
  restriction removed (the width was only used to invoke Theorem A at `N = n+1`), i.e. `(T)` up to the depth factor
  `(1 - eps_p)^{-1} = 1 + O(p^{-2})`.
* `K >= 2`: (E+) is a full-torus statement; THEOREM P3C-1 consumes the DEPTH-`A` version (`m_A` = min over the
  polydisc of radius `A* < 1`). (E+) does not give it (a probe on the torus is not a probe of depth `A`), so the
  `K >= 2` glue is unchanged: DEPTHA's certificate at `nu = 3`, depth `1/2`, on dissociate supports.

### 10.9 Target 2 (K >= 2 in operator form) -- not reached

Not attempted beyond the remark above. The natural formulation in this note's language: per frequency
`det(M_k^+)^{1/D} >= (m_A)_k` would feed P3C's step 4 with `y_k = (m_A)_k`, and the share weights `b_k` then emerge
from step 5 exactly as in P3C; the missing object is the depth-`A` (grid-realisable) analogue of (PW), i.e. probes
`rho` supported on product sets with `|rhohat_T| <= A^{|T|}`-type constraints. Open.

### 10.5 What is refuted on the way (probe constructions)
* Any probe dominated by the geometric mean `prod_k |Lambda_k|^{2/D}` of the Cholesky polynomials of `B`
  (the `n=1` mechanism `Lambda_rho = Lambda_0 Lambda_1 / v`): fails at second order for `D >= 3` (products of
  overlapping tail-chain polynomials are not multi-affine; the second-order defect has non-multi-affine modes).
* The chain-supported LS probe and the analytic-part probe: see 10.4.
* The Riesz/"half" density `2 Re Lambda_Y(conj u) - 1` (needs `Re Lambda_Y >= 1/2` on the torus): `min Re Lambda_Y`
  reaches `0.02` (`n=1`, strongly correlated `B`), `-0.16` (`n=2`), `-0.46` (`n=3`) (`arch_half.py`). Dead.
* Interval-supported SOS densities `m^* Gamma m` (`Gamma >= 0`): the constrained SDP value drops below `D` in 20/30
  (`arch_ut2.py`); degree-1 full-monomial SOS densities certify only 9/24 (`n=2`), 3/18 (`n=3`); degree-2 certify
  19/24, 18/18 (`arch_adm3.py`). Not a clean route.
* The Markov measure `u_1 ~ Poisson(omega_1)`, `u_2 | u_1 ~ Poisson(alpha + beta conj u_1)` realises every
  (UT)-optimal probe found at `n=2` with `|alpha| + |beta| <= 0.885` (`arch_markov.py`, 80/80) -- but only on
  well-conditioned `B`; superseded by the exact LMI characterisation 10.7.3.

### 10.10 Numerical record (tenth pass; all fp64, local CPU; SDPs via `uv run --with cvxpy`, Clarabel)

| script | what | key numbers |
|---|---|---|
| `arch_det.py n N nstart seed` | adversary on `det M_+/m^D` | `n=1`: 1.000000000 (N=3), 1.0014 (N=6); `n=2`: 1.0000118/1.00043/1.0061/1.0043/1.0029 (N=3,4,5,6,8); `n=3`: 1.11..1.56 (weak search) |
| `arch_pw.py n ntrials seed K` | (PW) with optimised 6-atom probes vs naive probe | naive fails 9/16 at `n=2`; optimised slack `>= +4.9e-4` 16/16; `n=1` naive exact |
| `arch_ls.py` | LS chain probe vs Christoffel function | `n=2`: pointwise fails 15/20, mu-integrated fails (1.16), inadmissible 3/20 |
| `arch_local.py`, `arch_local2.py` | second order at `B = I + eps H` | naive correction fails (1.5 vs 1); LP-optimal correction: sup violation `-5.7e-15` (tight) |
| `arch_sdp.py kind seed eps` | matrix-SOS certificate of (PW) per `B` (`n=2`) | worst direction `eps=0.3`: `s* = +2.7e-4`, Gram rank 4-5 |
| `arch_ut.py`, `arch_ut2.py` | Theorem UT numerically; SOS-density side-condition | `T(B) >= D` 65/65; `T_adm < D` 20/30 |
| `arch_adm.py`, `arch_adm2.py` | structure of `Y*` (unitary `W`); admissibility of extreme `Y` | `W` unitary 22/22 (err `< 1e-7`); "156/156 admissible" superseded by 10.7.1 |
| `arch_ellip.py` | elliptope test `lambda_min(R_Y)` | optimal `Y*`: `-0.096` (`n=2`, cond 193/527 -> inadmissible, LP agrees); extreme: `-0.47` |
| `arch_ut3.py n ntrials seed [adv]` | elliptope-constrained matrix program `T_ell(B)` | random `B`: `>= 3.00004` (60/60, cond to 1.9e3); adversarial `B` (cond 5e8): `0.90` |
| `arch_half.py`, `arch_adm3.py`, `arch_markov.py`, `arch_cand.py` | routes in 10.5 / 10.7.6 | see 10.5, 10.7.6 |
| (predecessor, unrecorded) `eplus_n3_*.json` | `E_k` ratios at `n=3`, `N=4,5,6` | min `1.017 .. 1.40`, never below 1 |

Caveats: `arch_cand.py`'s penalised Nelder-Mead over the elliptope is unreliable (negative "optimised" slacks there
are optimiser failures; the atom-parametrised optimiser gives positive slack on the same `B`). LP admissibility
tests use an inner polytope of the moment body (grid 72^2 / 20^3), so "LP-feasible" is a proof of admissibility
and "LP-infeasible" is only indicative -- but every LP-infeasible case here is confirmed by `|omega| > 1` or by the
`n=2` LMI.

## 9. Status (current, end of tenth pass)

**PROVED (every step written out, fp64-checked).**
* Ninth pass: P1' one-liner (+ generalisation, + equality); S0; S1 (impure neurons free on the cost side);
  Theorem H (aligned, `2m-A`); uniqueness 4.2; no-go 4.3.
* 10.1 `(E+) <=> det M_+ >= m^D` (Maclaurin).
* 10.2 **(E+) at `n = 1`, every `N`** (probe bound + reflected centroid + Cauchy-Schwarz; two lines).
* 10.3 **`(E+) <=> (F) <=> (PW)`**: (E+) is equivalent to the population-free statement "every `D x D` PD `B` of unit
  determinant dominates, pointwise on the torus, `D` times the modulus of the generating polynomial of some
  probability measure" (Sion, twice).
* 10.6 **Theorem UT**: `max{Re tr Y : Y upper triangular, [[B,Y],[Y^*,B]] >= 0} >= D det(B)^{1/D}` for every
  `B > 0` (SDP duality; dual = `min_{K unit lower triangular} ||B^{1/2} K B^{1/2}||_{S_1}`; AM-GM with `det K = 1`).
  Its dual is the junk-free (E+) (10.7.5). Consequence: for every population some interval polynomial `Lambda`
  with constant term 1 has `sum_l x_l |Lambda(v_l)| <= det(M_+)^{1/D}` -- the phase-blind half of (F).
* 10.7.3 **`n = 2`: admissibility of an interval probe is the LMI `R_Y >= 0`** (complex `3 x 3` elliptope has only
  rank-one extreme points).

**REFUTED (with witnesses).** Ninth pass: fixed-V sharpness; (S) at `n=1`, `r<0.44`; `lambda_min(M_+) >= m`;
hmin-form of (E+) at `n=2`. Tenth pass: the naive `n=1` probe `2B_I/tr B` at `D >= 3` (second order); geometric-mean
probes (second order); LS chain probe; "feasible + `tr Y >= D` ⇒ admissible" (`|omega| = 1.25`); "optimal `Y*` is
admissible" (cond 193/527); the elliptope-constrained matrix program as a route to all `B` (degenerate `B`,
`T_ell -> 0`); Riesz/half density; interval SOS densities; the hmin-variant tests unchanged.

**CONJECTURED, numerically solid, never violated: (E+) for `n >= 2`.** Adversary on `det M_+/m^D` never below 1
(minimum at the block: `1.0000118` at `n=2`); (PW) certified by matrix-SOS at sampled `B`; second-order (PW) exact
and tight at `n=2`. **The single remaining content is admissibility**: at `n = 2` it is the explicit
finite-dimensional statement 10.7.6 (one LMI + one pointwise inequality on `T^2`, three complex unknowns); no
construction of the probe from `B` is known beyond first order. Theorem UT covers exactly the part of (E+) that
does not need it.

**OPEN / not done.** (i) 10.7.6 (⇒ (E+) at `n=2` ⇒ C1 Re-form at `nu=3` for every `N`, every `r`, no certificate);
(ii) `n >= 3` (admissibility needs vanishing non-interval moments; elliptope no longer sufficient); (iii) target 2,
the depth-`A`/`K >= 2` operator form (10.9); (iv) a strong `n=3` adversary for `det/m^4` (the current one is weak,
coarse inner grid).

**Reproduce (tenth pass).** `uv run python arch_det.py 2 4 3 1`; `uv run --with cvxpy python arch_ut.py 2 20 1`;
`uv run --with cvxpy python arch_ellip.py 2 24 8`; `uv run --with cvxpy python arch_ut3.py 2 30 9 6`;
`uv run python arch_local2.py 2 6 1`. No background jobs, monitors or sleep loops were left running.
