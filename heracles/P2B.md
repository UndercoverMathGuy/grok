# P2B: the Quadratic Product Lemma, made quantitative

Sixth pass on the global-optimality programme, and the direct successor of `P2.md`.
Briefing `GOPT_STATE.md` + `GOPT_STATE2.md`; inherited `THEORY_nsum.md`,
`LEMMA_ATTEMPT.md` (agent 1), `GLOBAL_OPT.md` (agent 2), `GLOBAL_OPT3.md` (agent 3),
`GLOBAL_OPT4.md` (agent 4), `ADVERSARY.md`, `P3A.md`, `P2.md`.

My problem, verbatim: **prove the Quadratic Product Lemma (QPL)** — for `N = n+1`
complex atoms `c_l` with unit slot phases `v_lj`, clean amplitude `C_empty = sum_l c_l = 1`
and junk energy `sum_{T != empty}|C_T|^2 <= eta^2`,

    (n+1)^{n+1} prod_l |c_l|  >=  exp(- C eta^2)      for eta <= eta_0,

with an explicit `C` and a useful `eta_0`. By `P2`'s Theorem P2-3 and Corollary P2-5 this
turns the junk-cost curve's only proven minorant from *linear* (which yields a proven junk
threshold of exactly **zero**) into a *quadratic* one (which yields a positive threshold).

**Status of this note: see §1.** The write-up is incremental; each section is written only
after its numbers are in `heracles/FINAL_RUNS/P2B/`.

Everything float64 or mpmath, local CPU, **no pods, $0**. Scripts `heracles/p2b_*.py`,
outputs `heracles/FINAL_RUNS/P2B/`.

---

## 0. Symbols, with a plain reading of each

Inherited symbols keep their meaning (`GOPT_STATE.md` §"Setting"/"Atom picture", `P2.md` §0).
The ones this note adds or leans on:

| symbol | meaning | plain reading |
|---|---|---|
| `n` | number of operands; `nu = n+1` is the homogeneity degree | how many numbers are added |
| `N = n+1` | number of atoms at one frequency | one feature is carried by `n+1` neurons |
| `c_l`, `l = 1..N` | atom `l`'s complex clean amplitude | one neuron's signed contribution at this frequency |
| `v_lj`, `j = 1..n` | unit-modulus slot phase | phase of neuron `l`'s `j`-th operand embedding |
| `C_T = sum_l c_l prod_{j in T} v_lj` | sign-pattern amplitude | `T = empty` is the **clean** part, the other `2^n - 1` are **junk** |
| `eta^2 = sum_{T != empty}|C_T|^2` | junk energy, at `C_empty = 1` | "how much of this feature's output is garbage", in `l_2` |
| `delta = sqrt(2^n - 1) eta` | Cauchy–Schwarz sup-norm bound on the junk | the junk polynomial is nowhere bigger than this |
| `h(u) = sum_l c_l prod_j (1 + u_j v_lj) = sum_T C_T u^T` | the population's generating polynomial | `h = 1 + g`, `g` = junk part; `|g| <= delta` on the torus |
| `u^{l0,sigma}`, `u_j = -conj(v_{sigma(j),j})` | a **kill point** | the torus point annihilating every atom except `l0`; `sigma` is a bijection from the `n` slots onto the `n` atoms `!= l0` |
| `H = mean of log|h|` over the `(n+1)!` kill points | the kill-point average | what the surviving atom "sees" there |
| `V_j = prod_{l<m}|v_lj - v_mj|` | slot `j`'s Vandermonde product | how spread out slot `j`'s phases are; maximal at roots of unity |
| `Delta = (n+1)log(n+1) - (1/n) sum_j log V_j^2` | the **Hadamard defect**, `>= 0` | how far the slot phases are from rotated `(n+1)`-st roots of unity |
| `m_T = mean of u^T` over the kill points | **kill-point moments** | how far the kill-point set is from being Haar-distributed |
| `S_j = sum_l v_lj`, `Q_j = sum_l v_lj^2` | slot `j`'s power sums | vanish exactly at rotated roots of unity |
| `Z = sum_l conj(v_l1 v_l2)` (`n = 2`) | the **cross-slot moment** | the one quantity the Hadamard defect cannot see |
| `eps(v)` | the smallest junk energy the phases `v` admit | `eta >= eps(v)` for every choice of atoms |
| `Theta(eta_0)` | `sup { ||m||^2 / Delta : eps(v) <= eta_0 }` | the **coupling constant**: how big the kill-point moments can be per unit of Hadamard defect, among phase configurations compatible with small junk |
| `Pi(eta)` | `min (n+1)^{n+1} prod_l |c_l|` at junk `<= eta` | the object QPL bounds below |
| `Phi = -log[(n+1)^{n+1} prod_l |c_l|]` | the **product deficit** | QPL says `Phi <= C eta^2` |

Normalisation throughout: `C_empty = 1`, i.e. `sum_l c_l = 1` exactly.

---

## 1. Status and headline

**STATUS: the Quadratic Product Lemma is PROVED at `n = 2` (`nu = 3`, the paper's lead cell),
computer-assisted; OPEN for `n >= 3`.** Consequence: the junk-cost curve now has a *proved*
quadratic minorant, and the programme's first *non-zero* proved junk thresholds.

> **MAIN RESULT (Theorem P2B-1 + Corollary P2B-2).** For every three-atom single-frequency
> population at `n = 2` with `C_empty = 1` and junk energy `<= eta^2`, `eta <= eta_0 = 0.15`,
>
>     27 * |c_1 c_2 c_3|  >=  exp( - 5.224738 * eta^2 ) ,
>
> and consequently, for every `r in (0,1]`,
>
>     Cl2_r|_{N=3}(eta)  >=  3^{1-r} exp( - 1.741580 * r * eta^2 ) .
>
> The inherited proved minorant was *linear* in `eta` and therefore gave a junk threshold of
> exactly `0`; this one gives `eta <= 0.1401` for `P3A`'s `K`-free chord criterion and
> `eta <= sqrt3/(2 K * 5.2247) = 0.1657/K` for agent 4's `(JT)`, at `nu = 3, q = 2`.

**What is new, in order of importance.**

1. **Lemma M4 (§4.3), proved by hand, is the coupling `P2` §4.4 asked for**, at `n = 2`. The
   cross-slot moment `Z = sum_l conj(v_l1 v_l2)` — the one quantity invisible to the Hadamard
   defect, and the reason `P2` stalled — is the **midpoint** of the two per-slot second power
   sums `Q_1, Q_2` along the Möbius flow that carries slot 1 to slot 2. The midpoint of a
   modulus exceeds the average of the endpoints only by the flow's second-order defect, which
   is exactly the Möbius centre `a`, which is exactly the junk: `eps = |a|/sqrt(2+|a|^2)`.
   One display: `|Z| <= (|Q_1|+|Q_2|)/2 + sum_l 2 sin^2(psi_l) <= (|Q_1|+|Q_2|)/2 + 6|a|^2`,
   and `6|a|^2 = O(eta^2)` is invisible at the order that sets the constant.
2. **Proposition M2 (§3.3), proved for every `n`:** `Phi <= (n+1)[eta ||m|| + Cdel ||G|| eta^2] - Delta`.
   This is the whole of QPL reduced to a statement about **phases only** — the kill-point
   moment vector `m` against the Hadamard defect `Delta` — with an explicit, sharp scalar
   lemma (M1) for the non-linearity of `log`, which `P2` did not treat.
3. **Lemma M5 (§4.4, §6): a rigorous computer-assisted certificate** of the per-slot statement
   `8|S|^2 + |Q|^2 <= rho * 36 Delta_1` (`rho = 1.15`; the constant `rho = 1` is REFUTED by
   `1.38%`). Certificate = an analytic second-order argument with an exact Fourier third-
   derivative bound on two small balls, plus a terminating `129 179`-box interval branch and
   bound (mpmath `iv`) over the rest of the two-torus.
4. **`P2` audited (§2).** Every claim I use is re-derived and re-checked independently and
   holds; one *proof* in `P2` is wrong (Theorem P2-2's scalar step, §2.3) and is repaired; the
   collapse point of the product curve is identified exactly as `1/sqrt(2^n - 1)` (§2.5).
5. **`P2`'s F2-2 half-closed (§7.1):** `N > 3` reduces to `N = 3` whenever the tail mass
   `tau <= (eta_0 - eta)/(sqrt3 + eta_0)`, for `r <= 0.75` (so in particular at `q = 2`).
6. **The chain-frame route of `ELEGANT.md` evaluated for junk (§8):** its structural claims are
   confirmed and sharpened (`E_2(M/A) = binom(D,2) - tr(L'U')` exactly), but `||U'||` is an
   `O(1)`, indeed unbounded, function of the phases even at `eta = 0`, so that route is
   *also* linear in `eta` until it is given the same coupling lemma. It is a better packaging
   (uniform in `N`), not a shorter proof.

**Honest accounting of the constant.** The sharp curvature is `c_2 = 9/8 = 1.125`
(Prop P2-4, re-verified in §2.4). P2B proves `5.2247` at `eta_0 = 0.15`, and its floor as
`eta_0 -> 0` is `9 rho/8 + 3 Cdel_0 ||G||_block = 1.294 + 2.25 = 3.54`. So the *structure* is
right and the *constant* is a factor `3.1` too big; §9 F2B-5 says exactly where.

---

## 1b. Reproducing everything

From `heracles/`, with `uv run python`. Total about 35 minutes on one core; no pods, `$0`.

    uv run python p2b_verify.py                                  # SS2: audit of P2 (verify.json)
    uv run python p2b_lemmas.py --ngrid 500                      # SS4.2-4.4: (PS) sup, Moebius dict
    uv run python p2b_scan.py --ngrid 12 --npolish 200           # F2B-1/F2B-2 (the D-vs-Delta trap)
    uv run python p2b_assemble.py --trials 900                   # SS5: all checks + C(eta_0)
    uv run python p2b_cert_ps.py --rho 1.15 --tol 1e-6 --maxboxes 2000000   # SS6: the certificate
    uv run python p2b_ntail.py --eta0 0.15 --C 5.224738          # SS7.1: the N > 3 reduction
    uv run python p2b_elegant.py                                 # SS8: the chain-frame evaluation

Outputs land in `heracles/FINAL_RUNS/P2B/` as `*.json` plus the console `*.log`.

---

## 2. Verification of the inherited material I rely on (`p2b_verify.py`)

`P2.md` had not been hand-checked. I re-derived and re-verified, by an independent code path,
every statement of `P2` that `P2B` uses. Output: `FINAL_RUNS/P2B/verify.json`.

### 2.1 The chain identity — CONFIRMED (and it is an identity, not an inequality)

> **(CHAIN).** For *every* `c in C^{n+1}` and every unit-modulus `v_lj`,
>
>     log[ (n+1)^{n+1} prod_l |c_l| ]  =  (n+1) H  +  Delta ,
>     H     := mean over the (n+1)! kill points of log|h| ,
>     Delta := (n+1) log(n+1) - (1/n) sum_j log V_j^2  >=  0 .

*Proof (re-derived).* At the kill point `u^{l0,sigma}` every atom `l != l0` has the zero
factor `1 + u_j v_lj` with `j = sigma^{-1}(l)`, so
`h(u^{l0,sigma}) = c_{l0} prod_j (1 - conj(v_{sigma(j)j}) v_{l0 j})` and, taking moduli,
`log|h(u^{l0,sigma})| = log|c_{l0}| + sum_j log d^{(j)}_{sigma(j),l0}` with
`d^{(j)}_{lm} = |v_lj - v_mj|` (using `|1 - conj(a) b| = |a - b|` for `|a| = |b| = 1`).
Averaging over the `n!` bijections `sigma`: a pair `(j,l)`, `l != l0`, occurs as
`sigma(j) = l` in exactly `(n-1)!` of them, so the phase term averages to
`(1/n) sum_j sum_{l != l0} log d^{(j)}_{l,l0}`. Summing over `l0` and using that the ordered
double product is the square of the unordered one, `sum_{l0} sum_{l != l0} log d^{(j)}_{l,l0}
= log V_j^2`. Hence `sum_l log|c_l| + (1/n) sum_j log V_j^2 = (n+1) H`. Add
`(n+1)log(n+1)`. `Delta >= 0` is Hadamard's inequality `V_j <= (n+1)^{(n+1)/2}`. ∎

*Plain reading: the product of the atom moduli is exactly the kill-point average of `log|h|`,
corrected by how far the slot phases are from Fekete (roots-of-unity) position. When there is
no junk, `h = 1`, `H = 0`, and the identity IS the Key Lemma.*

**Check.** `max |lhs - rhs|` over random `(c,v)`: `6.7e-16` (`n=1`), `3.6e-15` (`n=2`),
`1.4e-13` (`n=3`), `3.5e-14` (`n=4`). `min Delta` over random phases:
`0.000000 / 0.0340 / 0.4212 / 1.0370` at `n = 1..4` — never negative.

### 2.2 The kill-point moments — CONFIRMED, with a closed form for every `n`

> **(MOM).** For `T` a non-empty set of slots with `|T| = t <= n`,
>
>     m_T  =  (-1)^t * (n-t)! (n+1-t) / ((n+1) n!)  *  sum over INJECTIONS tau: T -> atoms
>                                                       of prod_{j in T} conj(v_{tau(j), j}) .
>
> In particular `m_{ {j} } = -conj(S_j)/(n+1)` for every `n`, and at `n = 2`
> `m_{ {1,2} } = ( conj(S_1) conj(S_2) - Z )/6` with `Z = sum_l conj(v_l1 v_l2)`.

*Proof.* `u^T = (-1)^t prod_{j in T} conj(v_{sigma(j),j})`. For fixed `l0`, the restriction of
`sigma` to `T` is a uniformly random injection into the `n` atoms `!= l0`, and each injection
extends in `(n-t)!` ways; averaging also over `l0` and counting, for a fixed injection `tau`
into the full atom set, the `n+1-t` admissible `l0` (those outside `tau`'s image) gives the
displayed coefficient. `t = 1`: coefficient `-(n-1)! n/((n+1) n!) = -1/(n+1)` and the
injection sum is `S_j`, conjugated. `n = t = 2`: coefficient `0! * 1/(3 * 2) = 1/6` and the
injection sum is `sum_{a != b} conj(v_{a1} v_{b2}) = conj(S_1)conj(S_2) - Z`. ∎

**Check.** brute-force average over the explicitly enumerated `(n+1)!` kill points vs the
closed form: max abs error `0`, `2.2e-16`, `2.2e-16`, `1.6e-16` at `n = 1..4`; `P2`'s two
explicit `n = 2` formulas agree to `2.8e-16`.

### 2.3 Theorem P2-2 — TRUE, but its stated proof of the scalar step is WRONG; repaired

`P2` §4.3 uses: *for `lam in (0, Lam]`, `log lam <= (lam-1) - (lam-1)^2/(2 Lam)`, because
"both sides and their first derivatives agree at `lam = 1` and at `lam = Lam`".*

The parenthetical reason is **false**: at `lam = Lam` the gap is
`(Lam-1) - (Lam-1)^2/(2Lam) - log Lam`, which equals `0.0569, 0.2347, 0.4887, 0.7906, 1.4827`
at `Lam = 2,3,4,5,7` — not `0`. The inequality itself is nevertheless **true**, by:

> `psi(lam) := (lam-1) - (lam-1)^2/(2Lam) - log lam` has
> `psi'(lam) = 1 - (lam-1)/Lam - 1/lam = (lam-1)(Lam-lam) / (lam Lam)`,
> which is `<= 0` on `(0,1]` and `>= 0` on `[1, Lam]`. So `psi` is minimised at `lam = 1`,
> where `psi(1) = 0`. Hence `psi >= 0` on `(0, Lam]`. ∎

(Numerically `max(lhs - rhs)` over a `2e5`-point grid is `<= -8.3e-12` for `Lam = 2,3,4,5,7`.)
With the repair, Theorem P2-2 stands: `A := W W*/(n+1)` is Hermitian PSD with unit diagonal and
trace `n+1`, so all eigenvalues lie in `(0, n+1]`; summing the scalar inequality over them
(`sum (lam_i - 1) = 0`, `sum (lam_i - 1)^2 = ||A - I||_F^2`) gives
`log det A <= -||A-I||_F^2/(2(n+1))`, i.e.

    V^2 <= (n+1)^{n+1} exp( -(n+1)^{-3} sum_{k=1}^{n} (n+1-k)|P_k|^2 ) .

**Check.** `max V^2/bound` over `2e4` random circle configurations: `0.999999996` (`n=1`),
`0.9996` (`n=2`), `0.9997` (`n=3`), `0.9775` (`n=4`), `0.9096` (`n=5`) — never above `1`.

**But I do not use Theorem P2-2 at `n = 2`.** It is loose there by exactly a factor `3`
(the scalar step's `1/(2 Lam) = 1/6` against the sharp local `1/2`). Instead:

> **(VAND2, exact).** For three unit-modulus numbers with `S = sum z_l`, `Q = sum z_l^2`,
>
>     V^2 = prod_{l<m}|z_l - z_m|^2 = 27 - 6|S|^2 - 3|Q|^2 + 2 Re( conj(S)^2 Q ) .

*Proof.* `V^2 = det(W W*)` with `W_{a,l} = z_l^a`, `a = 0,1,2`; `(WW*)_{ab} = P_{a-b}` with
`P_0 = 3, P_1 = S, P_2 = Q`; expanding the `3x3` determinant of
`[[3, conj S, conj Q],[S,3,conj S],[Q,S,3]]` gives the display. ∎
(Checked to `2.8e-14` on `2e4` random configurations.)

Consequently, with `u_j := 1 - V_j^2/27 in [0,1]` (non-negative by Hadamard),

> **(D-BOUND).** `Delta = -(1/2) sum_j log(1 - u_j)  >=  D := (1/2) sum_j u_j`,
> and `D` is a **trigonometric polynomial** in the phases, sharp to second order at the block.

### 2.4 Prop P2-4's constant — CONFIRMED independently

Rebuilt the second-order programme from scratch (independent code path from `p2_secord.py`)
and bisected the PSD condition: `c_n/[(n+1)^2/(4n)] = 1.0000000000` at `n = 1..6`
(`c_n = 1, 9/8, 4/3, 25/16, 9/5, 49/24`). So the *tangent* constant of QPL is
`c_n = (n+1)^2/(4n)`, `9/8` at `n = 2`.

### 2.5 A new elementary fact: where `Pi` collapses, exactly

> **(COLLAPSE).** If some atom vanishes, `c_{l0} = 0`, then `eta >= 1/sqrt(2^n - 1)`.
> Equivalently `Pi(eta) > 0` for every `eta < 1/sqrt(2^n - 1)`.

*Proof.* With `c_{l0} = 0` only `n` atoms remain; assign each to one slot and put
`u_j = -conj(v_{l(j), j})`, killing all of them at once, so `h(u) = 0`. But
`|h| = |1 + g| >= 1 - delta = 1 - sqrt(2^n-1) eta` everywhere on the torus. ∎
(Checked: `|h|` at that point is `<= 4.3e-15` on random data, `n = 1..4`.)

`1/sqrt(2^n-1) = 0.57735` at `n = 2` and `0.37796` at `n = 3`. `P2` measured `Pi` hitting `0`
"between `0.550` and `0.607`" at `n = 2` and "between `0.363` and `0.451`" at `n = 3`:
**the collapse point is exactly `1/sqrt(2^n - 1)`**, and no product-based argument can reach
past it. (That the bound is attained is `P2`'s numerics, not proved here.)

---

## 3. The master inequality: junk costs at most `3 eta ||m|| + 3 Cdel ||G|| eta^2 - Delta`

This section is exact and elementary; it is the frame into which everything else fits.
Throughout `n` is arbitrary; `N = n+1`, `C_empty = 1`, junk energy `<= eta^2`,
`delta = sqrt(2^n-1) eta < 1`.

### 3.1 The scalar log bound (sharp)

> **LEMMA M1.** For complex `g` with `|g| <= delta < 1`,
>
>     log|1 + g|  >=  Re g  -  Cdel |g|^2 ,      Cdel := ( -log(1-delta) - delta ) / delta^2 ,
>
> and the constant is the smallest possible (attained at `g = -delta`).

*Proof.* Put `s = |g|`, `m = cos(arg g)`, and
`F(m) := (s m - (1/2) log(1 + 2 s m + s^2))/s^2`, which is `(Re g - log|1+g|)/|g|^2`. Then
`dF/dm = (s - s/(1 + 2 s m + s^2))/s^2`, which is `>= 0` exactly when `1 + 2sm + s^2 >= 1`,
i.e. `m >= -s/2`. So `F` decreases on `m < -s/2` and increases after; its maximum over
`m in [-1,1]` is at an endpoint, and `F(-1) = (-s - log(1-s))/s^2 > F(+1)`. Finally
`(-s - log(1-s))/s^2 = sum_{k>=2} s^{k-2}/k` is increasing in `s`, so the sup over `|g| <= delta`
is `Cdel`. ∎

*Plain reading: a multiplicative perturbation `1+g` costs at most its real part plus a
quadratic correction; the worst perturbation is the one that shrinks the modulus head-on.*
`Cdel -> 1/2` as `delta -> 0`.

### 3.2 The three moments

Write `h = 1 + g`, `g = sum_{T != empty} C_T u^T`, and let `mu` be the uniform measure on the
`(n+1)!` kill points. Define

    m_T := E_mu[u^T]                          (kill-point moments, closed form in SS2.2)
    G_{TT'} := E_mu[ u^T conj(u^{T'}) ]       (Hermitian, PSD, unit diagonal, trace 2^n - 1)

Then, exactly, `E_mu[g] = sum_{T != empty} C_T m_T` and
`E_mu[|g|^2] = sum_{T,T'} C_T conj(C_{T'}) G_{TT'}`. Cauchy–Schwarz on the first and the
Rayleigh quotient on the second give, with `||C_{!=empty}||_2 <= eta`,

    Re E_mu[g] >= -eta ||m||_2 ,        E_mu[|g|^2] <= ||G||_2 eta^2 <= (2^n - 1) eta^2 .

### 3.3 The master inequality

> **PROPOSITION M2 (proved; every `n`, every `eta` with `delta = sqrt(2^n-1) eta < 1`).**
> For the product deficit `Phi := -log[(n+1)^{n+1} prod_l |c_l|]`,
>
>     Phi  <=  (n+1) [ eta ||m||_2  +  Cdel ||G||_2 eta^2 ]  -  Delta .

*Proof.* `Phi = -(n+1)H - Delta` by (CHAIN). Apply Lemma M1 pointwise at each kill point
(`|g| <= delta` there by Cauchy–Schwarz over the `2^n - 1` unit monomials) and average:
`H >= Re E_mu[g] - Cdel E_mu[|g|^2] >= -eta||m|| - Cdel ||G|| eta^2`. ∎

*Plain reading: the only way junk can push the atom product below its junk-free value is by
being visible at the kill points; how visible it can be is measured by the kill-point moment
vector `m`, and the price of the non-linearity of `log` by the kill-point Gram matrix `G`.
The Hadamard defect `Delta` is a credit that must be spent first.*

**So QPL at `N = n+1` reduces to one purely phase-space question:** how large can `||m||`
be, per unit of `Delta`, among phase configurations that admit junk as small as `eta`?
Write

    eps(v) := min { ||C_{!=empty}||_2 : C_empty = 1 }     (the smallest junk the phases admit)

— every atom vector on the phases `v` has `eta >= eps(v)` — and note that if
`sup { [ (n+1) eta ||m|| - Delta ] / eta^2 : eps(v) <= eta <= eta_0 } = C_1` then

    Phi  <=  [ C_1 + (n+1) Cdel_0 sup||G|| ] eta^2       for all eta <= eta_0 .   (QPL-form)

`eps(v)` has a clean dual description (`GLOBAL_OPT4` Prop D1 in this special case): if
`Lambda(u) = sum_T L_T u^T` is a multi-affine polynomial vanishing at all `n+1` atom points
`v_l`, then `sum_T L_T C_T = 0`, so `|L_empty| <= ||L_{!=empty}||_2 eta`; and at `N = n+1 = 2^n`
(i.e. `n = 2`) that single `Lambda` is unique up to scale and the bound is exact.

---

## 4. `n = 2`: reduction to two finite lemmas

From here on `n = 2`, `nu = 3`, `N = 3`, `delta = sqrt3 eta`. Gauge: rotating slot `j`'s
phases leaves `||m||`, `Delta`, `eps`, `||G||` unchanged, so normalise `x_3 = y_3 = 1`
(`x_l := v_l1`, `y_l := v_l2`).

### 4.1 Everything in closed form

    S_j = sum_l v_lj ,  Q_j = sum_l v_lj^2 ,  Y = sum_l conj(x_l) y_l ,  Z = sum_l conj(x_l y_l)
    m   = ( -conj(S_1)/3 , -conj(S_2)/3 , W/6 ) ,      W := conj(S_1 S_2) - Z
    ||m||^2 = (|S_1|^2 + |S_2|^2)/9 + |W|^2/36
    G   = [[1, (conj(S_1)S_2 - Y)/6, -S_2/3], [*, 1, -S_1/3], [*, *, 1]]     (Hermitian)
    V_j^2 = 27 - 6|S_j|^2 - 3|Q_j|^2 + 2Re(conj(S_j)^2 Q_j) ,  Delta_j = -(1/2)log(V_j^2/27)

(the `G` entries: `G_{1,2} = E[u_1 conj(u_2)] = (conj(S_1)S_2 - Y)/6`,
`G_{1,12} = E[|u_1|^2 conj(u_2)] = conj(m_2) = -S_2/3`, `G_{2,12} = -S_1/3`.)
Since `G` is Hermitian with unit diagonal, `||G||_2 <= ||G||_inf = 1 + max row off-diagonal
sum`, and `||G||_2 <= trace = 3`.

### 4.2 The Möbius picture of `eps`, and the exact dictionary

> **LEMMA M3 (proved).** If the three points `(x_l, y_l)` of `T^2` have all `x_l` distinct and
> all `y_l` distinct, they lie on the graph of a unique Möbius map preserving the unit
> circle, `y = e^{i th}(x - a)/(1 - conj(a) x)` with `|a| != 1`, and then
>
>     eps  =  |a| / sqrt(2 + |a|^2) ,        equivalently   |a|^2 = 2 eps^2/(1 - eps^2) .
>
> In particular `eps = 0` iff `a = 0` iff the configuration is common-phase (`y_l = e^{i th} x_l`),
> and `eps < 1/sqrt3` iff `|a| < 1`.

*Proof.* Three distinct source points determine a unique Möbius map; it maps the circle
through `x_1,x_2,x_3` (namely `T`) to the circle through `y_1,y_2,y_3` (namely `T`), so it
preserves `T` and has the displayed form. Clearing denominators, the multi-affine polynomial
vanishing at the three points is `Lambda(u) = e^{i th} a - e^{i th} u_1 + u_2 - conj(a) u_1u_2`,
so `|L_empty| = |a|` and `||L_{!=empty}||_2 = sqrt(1 + 1 + |a|^2)`. ∎
(Checked to `4.6e-12` on 4000 random configurations, `p2b_lemmas.py`.)

*Plain reading: a three-atom population at `n = 2` is exactly "a Möbius map between the two
slots"; the map's centre `a` is the whole obstruction to being junk-free, and `eps` is a
monotone rescaling of `|a|`.*

### 4.3 LEMMA M4 (cross-slot): the only place `Z` can hide — PROVED

> **LEMMA M4 (proved).** With the notation of Lemma M3 and `|a| < 1`,
>
>     |Z|  <=  ( |Q_1| + |Q_2| )/2  +  sum_{l=1}^{3} 2 sin^2(psi_l)   <=   (|Q_1|+|Q_2|)/2 + 6|a|^2 ,
>     psi_l := arg(1 - a conj(x_l)) .
>
> Hence `|Z| <= (|Q_1|+|Q_2|)/2 + 12 eps^2/(1 - eps^2)`.

*Proof.* Write `x_l = e^{i xi_l}`. Since `|x| = 1`, `1 - conj(a)x = x conj(x - a)`, so
`y_l = e^{i th}(x_l - a)/(x_l conj(x_l - a)) = conj(x_l) e^{i th} e^{2 i arg(x_l - a)}`, and
therefore

    x_l y_l = e^{i th} e^{2 i (xi_l + psi_l)} ,      y_l^2 = e^{2 i th} e^{2 i (xi_l + 2 psi_l)} ,

using `arg(x_l - a) = xi_l + psi_l`. Put `a_l := 2 xi_l`, `s_l := 2 psi_l` and
`f(tau) := sum_l exp(i(a_l + tau s_l))`. Then

    |Z| = |f(1)| ,      |Q_1| = |f(0)| ,      |Q_2| = |f(2)| .

Now `|f(1)| <= |(f(0)+f(2))/2| + |f(1) - (f(0)+f(2))/2| <= (|f(0)|+|f(2)|)/2 + |R|` with

    R = f(1) - (f(0)+f(2))/2 = sum_l e^{i(a_l + s_l)} ( 1 - cos s_l ) ,

so `|R| <= sum_l (1 - cos s_l) = sum_l 2 sin^2(psi_l)`. Finally `1 - a conj(x_l)` lies in the
disc of radius `|a|` about `1`, whose points have `|sin(arg)| <= |a|`, so
`sum_l 2 sin^2 psi_l <= 6|a|^2`. Lemma M3 converts `|a|` to `eps`. ∎

*Plain reading: the cross-slot moment `Z`, the one quantity the Hadamard defect cannot see,
is the MIDPOINT of the two per-slot second power sums along the Möbius flow; a midpoint of a
modulus can exceed the average of the endpoints only by the second-order defect of that flow,
and that defect is exactly the junk parameter `a`. This is the missing coupling that `P2`
§4.4 asked for, in the case `n = 2`.*

(Checked: `max[ |Z| - (|Q_1|+|Q_2|)/2 - sum 2 sin^2 psi_l ] = -2.5e-9` and
`max[ ... - 6|a|^2 ] = -6.1e-9` over `2.4e5` random configurations with `|a| <= 0.9`;
`p2b_assemble.py` check (4).)

### 4.4 LEMMA M5 (per-slot): the statement that makes the constant sharp

> **LEMMA M5 = (PS_rho) — certified, not proved by hand.** For three unit-modulus numbers
> with `S = sum z_l`, `Q = sum z_l^2`, `V^2 = 27 - 6|S|^2 - 3|Q|^2 + 2Re(conj(S)^2 Q)` and
> `Delta_1 = -(1/2) log(V^2/27)`,
>
>     8 |S|^2 + |Q|^2  <=  rho * 36 Delta_1 ,      any rho > rho* = 1.013802266 .
>
> The ratio tends to `1` at the cube-roots configuration (the block's slot) in EVERY
> direction, and its global supremum is `rho* = 1.013802266` (attained at an interior point
> with `|S| = 0.26828`, `|Q| = 0.46458`, `Delta_1 = 0.021687`). So `rho = 1` is FALSE by
> `1.38%`; the assembly uses `rho = 1.15`, which is what §6 certifies.

*Plain reading: a slot's first two power sums cannot both be large unless its phases are far
from Fekete position, and `36 Delta_1` is exactly the right currency — with an unavoidable
`1.4%` surcharge away from the block.*

---

## 5. The assembly, and the theorem

> **THEOREM P2B-1 (QPL at `n = 2`; proved modulo Lemma M5, which is certified in §6).**
> Let `rho = 1.15`, and for `Delta >= 0`, `F >= 0`, `K := 36 rho Delta` put
>
>     B(Delta, F) := min{ 3 , max over U in [0, min(18, K/8)] of
>                          U/9 + ( U + min(6, sqrt(2(K - 8U))) + 2F )^2 / 144 } ,
>     Gb(Delta)   := min{ 3 , 1 + max( (U*/2 + 3)/6 + sqrt(U*)/3 , sqrt(2 U*)/3 ) } ,
>                    U* := min(18, 4.5 rho Delta) ,
>     Cdel(eta)   := ( -log(1 - sqrt3 eta) - sqrt3 eta ) / (3 eta^2) ,
>     F(eta)      := 12 eta^2 / (1 - eta^2) .
>
> Then for every `eta_0 < 1/sqrt3` and every three-atom population with `C_empty = 1` and
> junk energy `<= eta^2`, `eta <= eta_0`,
>
>     27 prod_{l=1}^{3} |c_l|  >=  exp( - C(eta_0) eta^2 ) ,
>     C(eta_0) := sup { [ 3 eta sqrt(B(Delta, F(eta))) + 3 Cdel(eta) Gb(Delta) eta^2 - Delta ] / eta^2
>                       : 0 < eta <= eta_0 , 0 <= Delta <= 3 sqrt3 eta } .

*Proof.* Proposition M2 at `n = 2` reads `Phi <= 3 eta ||m|| + 3 Cdel ||G|| eta^2 - Delta`,
so it suffices to bound `||m||` and `||G||` by functions of `Delta` and `eta`.

*(a) The moment norm.* Exactly, `||m||^2 = U/9 + |W|^2/36` with `U := |S_1|^2 + |S_2|^2` and
`W = conj(S_1 S_2) - Z`. By the triangle inequality and Lemma M4,

    |W| <= |S_1||S_2| + |Z| <= U/2 + (|Q_1| + |Q_2|)/2 + F ,

so `||m||^2 <= U/9 + (U + |Q_1| + |Q_2| + 2F)^2/144`.

*(b) The phase constraints.* Lemma M5 summed over the two slots gives
`8U + |Q_1|^2 + |Q_2|^2 <= 36 rho (Delta_1 + Delta_2) = K`. Hence `U <= K/8` and, by
Cauchy–Schwarz, `|Q_1| + |Q_2| <= sqrt(2(|Q_1|^2+|Q_2|^2)) <= sqrt(2(K - 8U))`. Trivially
`|S_j| <= 3` and `|Q_j| <= 3`, so also `U <= 18` and `|Q_1|+|Q_2| <= 6`. Since the displayed
bound is increasing in `|Q_1|+|Q_2|`, maximising over `U` gives `||m||^2 <= B(Delta, F)`;
and `||m||^2 <= 3` because each `|m_T| <= 1` (an average of unit-modulus numbers).

*(c) The junk correction.* `F <= 6|a|^2 = 12 eps^2/(1 - eps^2) <= 12 eta^2/(1 - eta^2)` by
Lemma M4 and Lemma M3, using `eps(v) <= eta` — the phases must admit the junk the population
actually has. (`eta <= eta_0 < 1/sqrt3` guarantees `|a| < 1`, which Lemma M4 needs.)

*(d) The Gram norm.* `G` is Hermitian with unit diagonal, so `||G||_2 <= ||G||_inf`, i.e.
`||G|| <= 1 + max(|G_12|+|G_13|, |G_12|+|G_23|, |G_13|+|G_23|)` with
`|G_12| <= (|S_1||S_2| + |Y|)/6 <= (U/2 + 3)/6`, `|G_13| = |S_2|/3`, `|G_23| = |S_1|/3`,
`|G_13|+|G_23| <= sqrt(2U)/3`; all are increasing in `U <= U*`. Also `||G||_2 <= trace = 3`.

*(e) The range of `Delta`.* If `Delta > 3 sqrt3 eta` the bracket is negative because
`||m|| <= sqrt3`, so the supremum is unaffected by restricting to `Delta <= 3 sqrt3 eta`. ∎

**Computed values** (`p2b_assemble.py`; the inner `sup` is a two-variable maximisation of
explicit elementary functions on a `220 x 900` grid, refined):

| `eta_0` | `C(eta_0)` | argmax `(eta, Delta)` | chord threshold `3 rho_need/C` | `(JT)` threshold |
|---|---|---|---|---|
| 0.05 | 4.055346 | (0.0500, 0.00347) | 0.18051 (> `eta_0`) | 0.21355/K |
| 0.08 | 4.382085 | (0.0800, 0.00925) | 0.16706 (> `eta_0`) | 0.19763/K |
| 0.10 | 4.610148 | (0.1000, 0.01445) | 0.15879 (> `eta_0`) | 0.18785/K |
| 0.12 | 4.847702 | (0.1200, 0.02150) | 0.15101 (> `eta_0`) | 0.17865/K |
| **0.15** | **5.224738** | (0.1500, 0.03555) | **0.14011** (<= `eta_0`) | **0.16575/K** |
| 0.18 | 5.631333 | (0.1800, 0.05514) | 0.13000 | 0.15379/K |
| 0.20 | 5.922384 | (0.2000, 0.07283) | 0.12361 | 0.14623/K |
| 0.25 | 6.753338 | (0.2500, 0.14739) | 0.10840 | 0.12824/K |
| 0.30 | 7.809470 | (0.3000, 0.29824) | 0.09374 | 0.11089/K |

(computed with `rho = 1.15`, the value the certificate of §6 actually establishes. `rho` enters
`C` only through the `||m||` half, roughly linearly: at the minimal admissible
`rho = 1.0139` the same computation gives `5.0362` at `eta_0 = 0.15`, i.e. `3.5%` smaller.
`rho = 1.15` was chosen because the branch and bound of §6 terminates with it: the box width
needed at distance `d` from the tight point scales like `d (sqrt rho - 1)/(sqrt rho + 1)`,
which is `d/135` at `rho = 1.03` and `d/29` at `rho = 1.15`.)

As `eta_0 -> 0` the `||m||` half of `C` tends to `9 rho/8 = 1.2938` (the sharp tangent
constant `c_2 = 9/8` of Prop P2-4, inflated by Lemma M5's surcharge `rho`), and the
remainder half to `3 * (1/2) * 3/2 = 2.25`; so the method's floor is `C -> 3.54`, a factor
`3.1` above the true curvature. **That factor, not the structure, is what is lost.**

### 5.1 Checks (`p2b_assemble.py`)

| check | statement | worst value over random admissible configurations |
|---|---|---|
| (4) | `|Z| - (|Q_1|+|Q_2|)/2 - sum_l 2 sin^2 psi_l <= 0` | `-7.1e-13 / -1.8e-11 / -1.0e-11 / -2.5e-9` at `|a| <= 0.05/0.2/0.5/0.9` (2.4e5 draws) |
| (4') | `... - 6|a|^2 <= 0` | `-8.2e-12 / -1.1e-10 / -3.0e-11 / -6.1e-9` |
| (3) | `Phi <= 3 eta||m|| + 3 Cdel ||G|| eta^2 - Delta` | min slack `+0.00186 / +0.00220 / +0.01015 / +0.01487` at `eta <= 0.05/0.1/0.2/0.3` |
| (6) | `||m||^2 <= B(Delta, F)` | max excess `-9.9e-4 / -7.0e-4 / -2.0e-3` at `eta <= 0.05/0.1/0.2` |
| end | `27 prod|c_l| >= exp(-C eta^2)` | min slack `+0.00848 / +0.02210 / +0.03654` at `eta_0 = 0.05/0.1/0.2` |

(the admissible configurations are generated through the Möbius picture of Lemma M3, so they
really do sample the small-junk region rather than being rejected out of it.)

---

## 6. The certificate for Lemma M5 (`p2b_cert_ps.py`)

Lemma M5 is the one statement of §5 that is not proved by hand. It is a **two-real-variable**
statement (rotate the slot so that `z_3 = 1`; then `(a_1, a_2) in [0, 2pi)^2`), and it is
certified in two pieces. Take `rho = 1.15` (comfortably above the measured supremum
`rho* = 1.013802266`; the slack is what makes the branch and bound terminate, see §5's note).

**(L) The local piece, at the two tight points.** `R := 36 rho Delta_1 - 8|S|^2 - |Q|^2`
vanishes only at the two cube-root configurations `B = (2pi/3, 4pi/3)`, `B' = (4pi/3, 2pi/3)`.
There, use `-log(1-u) >= u` with `u = 1 - V^2/27 in [0,1]` (valid because `V^2 <= 27` is
Hadamard), i.e. `Delta_1 >= (1/2)(1 - V^2/27)`, which gives `R >= Psi` with

    Psi := 18 rho (1 - V^2/27) - 8|S|^2 - |Q|^2
         = (6 - 12/rho)|S|^2 + (3 - 3/(2rho))|Q|^2 - 2 Re(conj(S)^2 Q) ,

a **trigonometric polynomial** whose frequencies lie in `{-2,...,2}^2`. Its exact Fourier
coefficients are read off term by term (`|S|^2 = sum_{l,m} e^{i(a_l - a_m)}`,
`|Q|^2 = sum_{l,m} e^{2i(a_l-a_m)}`, `2Re(conj(S)^2 Q) = sum_{a,b,c}[e^{i(2a_c - a_a - a_b)} + c.c.]`),
which gives, for a perturbation `s` of the base point,

    Psi(base + s) = (1/2) s^T H s + R_3 ,   |R_3| <= (M_3/6)||s||^3 ,
    M_3 := sum_k |gam_k| ||k||_2^3 = 205.7222 ,   lam_min(H) = 2.347826087 ,

(`Psi(base) = 0` and `grad Psi(base) = 0` to `4.5e-15`), hence `Psi >= 0` on the closed ball
of radius `r = 3 lam_min/M_3 = 0.03423781` about each of `B, B'`. *(A sampled check inside
the two balls gives `min Psi = 6.02e-08 >= 0`.)*

Note `lam_min(H) = 36 rho * (1 - 1/rho) * ... > 0` **only because `rho > 1`**: at `rho = 1` the
Hessian is exactly singular in every direction, which is the analytic reason Lemma M5 fails at
`rho = 1` and the reason the whole method's asymptotic constant is `9 rho/8` rather than `9/8`.

**(G) The global piece.** Branch and bound over `[0, 2pi]^2` in mpmath interval arithmetic
(`iv.cos`, `iv.sin`, `iv.log`, outward rounded, `dps = 20`), with the evaluation

    V^2 = 27 - 6|S|^2 - 3|Q|^2 + 2Re(conj(S)^2 Q)   (exact identity, §2.3)
    Delta_1 >= -(1/2) log( min(27, sup V^2) / 27 )
    R >= 36 rho * that  -  sup(8|S|^2 + |Q|^2) ,

a box being accepted when this lower bound is `>= 0`, or when it lies inside one of the two
certified balls. (Where `sup V^2 <= 0` the box is accepted immediately: `V = 0` means two
phases coincide and `Delta_1 = +inf`.)

**RESULT: CERTIFIED.** The branch and bound terminates: **129 179 boxes** processed, `64 319`
accepted by their interval lower bound, `189` accepted as lying inside a certified ball, no box
left. (The atom-swap symmetry `(a_1,a_2) -> (a_2,a_1)` is used to skip the strict half
`a_2 > a_1`; the tolerance below which a box would be reported as a failure is `1e-6` and is
never reached.) **Lemma M5 holds at `rho = 1.15` for every configuration.**

---

## 7. What this buys the programme

Write `Cl2_r|_{N=3}(eta) := min{ sum_l |c_l|^r : C_empty = 1, junk energy <= eta^2 }` over
three-atom populations, `kappa0 = 3^{(1-r)/r}`, `A(eta) = Cl2_r^{1/r}`,
`rho_crit(eta) = [1 - A(eta)/kappa0]/eta` (the function both inherited criteria constrain).

> **COROLLARY P2B-2 (from Theorem P2B-1, by `P2`'s Corollary P2-5).** For `eta <= eta_0`,
>
>     Cl2_r|_{N=3}(eta) >= 3^{1-r} exp( - r C(eta_0) eta^2 / 3 ) ,
>     rho_crit(eta)     <= C(eta_0) eta / 3 .
>
> Consequently, on the pure single-frequency class:
> * **(JT)** (`GLOBAL_OPT4` §6.1) holds on a `K`-frequency support for every junk profile with
>   `eta_k <= min( eta_0 , sqrt3/(2 K C(eta_0)) )`;
> * **(CH)** (`P3A` §2.3, `K`-free) holds for every junk level up to
>   `min( eta_0 , 3 rho_need/C(eta_0) )`, `rho_need = 0.2440169` at `nu = 3, q = 2`.

The best self-consistent cell is `eta_0 = 0.15`, `C = 5.2247`:

| criterion | inherited **proved** threshold | **P2B proved** threshold | `P2`'s conditional (measured `Pi`) |
|---|---|---|---|
| (CH), `K`-free | `0` | **`0.1401`** | `0.5395` |
| (JT), `K = 2` | `0` | **`0.0829`** | `0.511` |
| (JT), `K = 3` | `0` | **`0.0553`** | `0.426` |
| (JT), `K = 4` | `0` | **`0.0414`** | `0.369` |
| (JT), `K = 6` | `0` | **`0.0276`** | `0.284` |

(the `(JT)` entries are `sqrt3/(2 K C)`, all below `eta_0 = 0.15`; the `K = 1` entry is capped
by `eta_0` itself.) *Plain reading: the inherited proved minorant of the junk-cost curve was
linear and therefore yielded a junk threshold of exactly zero; this one is quadratic and
yields a positive, explicit threshold — about a factor `3.9` below the measured truth.*

### 7.1 The `N > 3` axis (`P2`'s unfinished F2-2), partially closed

> **PROPOSITION P2B-3 (proved; `n = 2`, `r <= 0.75`, i.e. `q <= 2.25`).**
> Let `c_1, ..., c_N` (`N > 3`) satisfy `C_empty = 1`, junk energy `<= eta^2`, and let
> `tau := sum_{l not in S}|c_l|` be the mass outside the three largest atoms. If
> `tau <= tau_max(eta) := (eta_0 - eta)/(sqrt3 + eta_0)` then
>
>     sum_l |c_l|^r  >=  3^{1-r}(1-tau)^r exp(-r C(eta_0) eta'(tau)^2/3) + tau^r
>                    >=  3^{1-r} exp(-r C(eta_0) eta^2/3) ,
>     eta'(tau) := (eta + sqrt3 tau)/(1 - tau) .

*Proof.* Deleting the tail changes each `C_T` by at most `tau`, so the restricted three-atom
population has `|C'_empty| >= 1 - tau` and `||C'_{!=empty}||_2 <= eta + sqrt3 tau`; its
normalised junk is therefore at most `eta'(tau)`, which is `<= eta_0` exactly when
`tau <= tau_max`. Theorem P2B-1 on the normalised triple plus AM–GM gives the first summand,
and `sum_{l not in S}|c_l|^r >= (sum_{l not in S}|c_l|)^r = tau^r` (subadditivity of `t^r`,
`r <= 1`) the second. The final inequality is the statement that the displayed function of
`tau` is minimised at `tau = 0`, verified in `p2b_ntail.py` on a `2e5`-point grid for
`r in {0.4, 0.5, 2/3, 0.75}` and `eta in {0, .02, .05, .08, .10, .12, .14}`: the ratio to the
`tau = 0` value is `1.000000000000` in every one of the 28 cells. ∎

(At `r = 0.9` and `r = 1` the worst ratio is `0.9866` and `0.9646`, i.e. the reduction as
stated fails there; it is not needed, since `q = 2` at `nu = 3` is `r = 2/3`, and `r > 0.75`
means `q > 2.25`.)

**What is still missing on this axis:** the case `tau > tau_max`, where the restricted triple's
junk exceeds `eta_0` and Theorem P2B-1 no longer applies. The crude fallback
`sum_l|c_l|^r >= (1-tau)^r + tau^r` is *not* enough (at `r = 2/3, tau = 1/2` it gives `1.26`
against the target `1.41`), so this range needs the inherited peeling/`P1` machinery
(`P1A`/`P1B`/`P1C`/`ADVERSARY`) robustified to non-zero junk. **P2's F2-2 is therefore
half-closed, not closed.**

---

## 8. The chain-frame route (`ELEGANT.md`) evaluated for the junk problem

A sibling agent's chain-frame proof closes **P1** (the junk-free sharp constant) for every `N`,
`n`, `r`. The coordinator asked whether it also gives a quadratic junk minorant. I checked the
structural claims (`p2b_elegant.py`, output `FINAL_RUNS/P2B/elegant.json`).

**Confirmed, and sharper than `ELEGANT` §6 states.** With `M/A = I + L' + U'` (`L'` strictly
lower = the interval junk totals over `A`, `U'` strictly upper = the uncontrolled mirror
moments over `A`):

* `E_1(M/A) = D` **exactly** — both `L'` and `U'` have zero diagonal
  (max error `1.1e-14` over 1600 random configurations, `n = 1..4`).
* `E_2(M/A) = binom(D,2) - tr(L' U')` **exactly** — the first junk correction is *bilinear*,
  not linear (max error `1.8e-12`). Proof: `E_2 = (tr^2 - tr(X^2))/2`, `tr(X) = D`, and
  `tr(L'^2) = tr(U'^2) = 0` because a product of two strictly-lower (resp. strictly-upper)
  matrices is strictly lower (resp. upper).
* `E_k(I + U') = binom(D,k)` **exactly for every `k`** (measured error `0.000e+00`), so in the
  multilinear expansion every term that deviates from `binom(D,k)` carries at least one `L'`
  factor. Combining with the permutation argument (a permutation that uses a strictly-lower
  entry must use a strictly-upper one to return), **every correction is at least bilinear in
  `(L', U')`.** This is the coordinator's observation, and it is correct.

**But the route does not become quadratic in `eta`, for a concrete reason: `||U'||` is not
`O(eta)`.** `U'` is a function of the phases (and of the atoms), not of the junk:

* `ELEGANT`'s own `n = 1` witness `v = (1, i)`, `c = ((1-i)/2, (1+i)/2)` is **exactly
  junk-free** (`C_{1} = 0`) and has `||U'||_2 = sqrt2 = 1.414214`.
* At `n = 2`, on the exactly junk-free common-phase family `x = (e^{i th}, e^{-i th}, 1)`,
  `||U'||_2` reaches `7.23` (at `th = 0.05`, where `sum_l|c_l| = 800.3`), and grows without
  bound as the three phases collide.

So `|E_k - binom(D,k)| <= poly(||U'||) ||L'||` is **linear in `eta`** with an `O(1)`
(indeed unbounded) prefactor, exactly the `Sigma`-dependence `ELEGANT` §6 flags as "the
worry". To get a quadratic law one must trade `||L'|| ||U'||` against the Gram-determinant
credit `sum_{|L|=k} (1 - G_L) prod_{l in L}|d_l|` — which is **structurally the same coupling
lemma** that §§4–5 of this note prove for the kill-point route (`3 eta ||m|| <= Delta + C eta^2`,
with `Delta` the Hadamard defect playing the role of the Gram credit). Both routes stand or
fall on a coupling between an uncontrolled phase-space moment and the geometric defect.

**Verdict.** The chain-frame route is the better *packaging* — it is uniform in `N`, which
would also settle §7.1's missing `tau > tau_max` range — but it is **not** closer to a proof
of QPL than the kill-point route: it needs the same lemma, in `binom(D,2)` variables rather
than `2` (the per-slot statement M5) plus one elementary identity (M4). I did not switch. A
worthwhile follow-up is to redo §§4–5 in the chain frame: the per-slot Lemma M5 should have a
clean analogue as a statement about `1 - G_L` for two-element `L`.

---

## 9. Failures and things I did not do

**F2B-1. `sup ||m||^2/Delta` as the coupling constant: too crude, and the fix matters.**
My first formulation asked for `Theta = sup{||m||^2/Delta : eps(v) <= eta_0}`, which is
`+inf`: the reflection configuration `v_l = (w^{-l}, w^{+l})` has `Delta = 0` and
`||m|| = 1/2`. The repair is not to bound the ratio but to carry `eta` through:
`sup{[3 eta||m|| - Delta]/eta^2 : eps(v) <= eta <= eta_0}` is finite because `eps = 1` at the
reflection point excludes it for `eta_0 < 1`. **Recorded because the ratio formulation looks
like the natural one and is vacuous.**

**F2B-2. Replacing `Delta` by its polynomial minorant `D := (1/2) sum_j (1 - V_j^2/27)`
everywhere: fails above `eta_0 ~ 0.2`.** `D` is exact to second order at the block (so it is
the right object for the *local* certificate), but it saturates at `1` while `Delta -> inf` at
degenerate phase configurations. Measured: `sup [3 eta||m|| - D]/eta^2` jumps from `1.2541`
at `eta_0 = 0.16` to `2.5000` at `eta_0 = 0.20` and `4.7846` at `eta_0 = 0.25`, the jump
coming entirely from the fully-degenerate configuration `x_l = y_l = 1` (`||m|| = sqrt3`,
`D = 1`, `Delta = inf`). Everything in §5 therefore uses `Delta`, with `D` reserved for §6(L).

**F2B-3. Lemma M5 at `rho = 1`. REFUTED, by `1.38%`.** `8|S|^2 + |Q|^2 <= 36 Delta_1` is
exactly tight to second order at the cube roots in **every** direction, so it looked like it
might hold globally; it does not. Global max of the ratio `1.013802266` at
`(a_1,a_2) = (4.33783, 2.39248)`, i.e. `|S| = 0.268277`, `|Q| = 0.464582`,
`Delta_1 = 0.021687`. This `1.38%` is the whole reason P2B's asymptotic constant is
`9 rho/8` and not the sharp `c_2 = 9/8`.

**F2B-4. Not done: `n >= 3`.** Every ingredient except Lemma M4 and Lemma M5 is stated and
proved for general `n` (§§2–3). Lemma M4 uses that `n = 2` has exactly `2^n = n+1+1` sign
patterns, so that the compatibility condition is a *single* multi-affine polynomial and the
Möbius picture applies; at `n >= 3` the null space of the `2^n x (n+1)` matrix has dimension
`2^n - n - 1 >= 4`, and I have no replacement. Lemma M5's analogue at `n >= 3` is a
`2n`-variable certificate.

**F2B-5. Not done: the constant.** The method's floor is `C -> 9 rho/8 + 3 Cdel_0 ||G||_block
= 1.294 + 2.25 = 3.54` as `eta_0 -> 0`, against the true curvature `c_2 = 9/8`. The `2.25`
comes from bounding the second-order loss of `log|1+g|` by `Cdel ||G|| eta^2` with
`||G||_block = 3/2` — Cauchy–Schwarz on the linear term and the Rayleigh bound on the
quadratic term cannot both be tight at the same junk vector, and I did not exploit that. A
third-order treatment (`H >= E[Re g] - (1/2)Re E[g^2] - rho_3(delta)`, with
`||N||_block = 1/sqrt2`, `N_{TT'} = E_mu[u^T u^{T'}]`) trades `2.25` for about `2.0` at
`eta_0 = 0.15`; not worth the extra certificate surface.

**Inherited dead routes** (`P2` F2-1 chord/concavity by mixing, F2-3 `l_2` peeling, F2-4
improving `R >= 1 - delta`, F2-5 Jensen over kill points; and `GOPT_STATE`'s list): none
re-attempted, none changed. `P2`'s Theorem P2-2 is **not** used at `n = 2` — the exact
identity (VAND2) of §2.3 replaces it and is a factor `3` sharper there.

---
