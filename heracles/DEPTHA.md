# DEPTH-A: the aimed-output cost theorem at aiming depth 1/2, every width (nu = 3)

Eighth pass. Continues `GOPT_STATE{,2,3}.md`, `ELEGANT.md`, `ELEGANT2.md`, `P3C.md`, `P3A.md`.
Local CPU, fp64, no pods, $0. New files only: `DEPTHA.md`, `deptha_*.py`, `FINAL_RUNS/DEPTHA/`.
Nothing inherited edited, no commits, no background jobs at hand-back.

**Status: PROVEN for nu = 3 (n = 2), every q in [2,3) tested, every number of atoms N, any junk,
at depth A = 1/2 — modulo one fp64 two-variable cell certificate (279 cells at q = 2, margin
>= 0.27%, corner handled by hand).** This is (DEPTH-A) in exactly the form THEOREM P3C-1 consumes
(singleton partition, `Re`-version), so with P3A's Riesz bound it gives (T) for every pure network on
every dissociate support at nu = 3. Not done: n >= 3, impure neurons, interval-arithmetic hardening.

---

## 0. Symbols (plain reading with each)

| symbol | meaning | plain reading |
|---|---|---|
| `n = 2`, `D = nu = 3` | operands, homogeneity degree | two numbers are added |
| `r = q/nu in [2/3,1)` | penalty ratio | `q = 2` weight decay is `r = 2/3` |
| `c_l in C`, `v_lj in T` (`l = 1..N`, `j = 1,2`) | atom amplitude, slot phase | one neuron's contribution at this frequency; `N` is ARBITRARY |
| `x_l = |c_l|`, `S = sum_l x_l`, `cost = sum_l x_l^r` | moduli, their sum, the weight-decay price | |
| `C_T = sum_l c_l prod_{j in T} v_lj` | pattern amplitudes | `A0 := C_empty` clean; `C_1, C_2, C_12` junk |
| `h(t) = A0 + C_1 t_1 + C_2 t_2 + C_12 t_1 t_2` | generating polynomial | what the feature outputs under a probe `t` |
| `m_A = min_{|t_1|=|t_2|=A} Re h(t)` | aimed output at depth `A` | the least the feature delivers to a probe that can only aim to depth `A` (= min over the closed polydisc, ELEGANT2 Lemma A2) |
| `a0 = |A0|`, `mu = m_A/a0`, `d = 1 - mu` | clean modulus, relative aimed output, relative aiming loss | |
| `w = sqrt(S^2 - a0^2)/a0` | **phase-misalignment excess** | how much modulus the atoms waste by not pointing along `A0`; `w = 0` iff all `c_l` are aligned |
| `m_l = (1, v_l1, v_l1 v_l2)`, `M = sum_l c_l m_l m_l^*` | chain vector, chain matrix (ELEGANT.md) | diagonal `A0`; below-diagonal entries are the junk `L_1 = C_1, L_2 = C_2, L_12 = C_12`; above-diagonal are the **mirror** moments `U_I = sum_l c_l conj(v_l^I)` |
| `E_k(M)` | sum of principal `k`-minors | coefficients of `det(I + sM)` |
| `e_k(x)` | elementary symmetric function | |
| `kappa0 = 3^{(1-r)/r}` | block exchange rate | `sqrt 3` at `q = 2` |

---

## 1. Statement

> **THEOREM DA (depth-1/2 aimed-output cost; n = 2, every N, any junk).** Let `v_lj` be unimodular,
> `c_l` complex, `N` arbitrary, `A = 1/2`, `r = 2/3`. Then
>
>     sum_l |c_l|^r  >=  3^{1-r} (m_A)_+^r ,      m_A = min_{|t_1| = |t_2| = 1/2} Re h(t) .
>
> Equality at the junk-free 3-block. *Plain reading: a two-operand feature, carried by any number of
> neurons with any junk, never costs less than the clean 3-block delivering what a half-depth probe
> leaves of it.* The same certificate passes for every tested `r in {0.67, 0.70, ..., 0.99}` (all
> `q in [2,3)` sampled) and for every `A in [0.47, 1]` at `r = 2/3` (the true all-`N` threshold is
> `0.4284`, P3C F6-1, so the method loses only `0.04` in depth).
>
> **COROLLARY DA-1 (torus, settles C1 in Re-form at n = 2).** Same statement with `A = 1`
> (`m = min_{T^2} Re h`) for every `N`, certified at `r in {0.6, 2/3, 0.75, 0.9}`; fails to certify at
> `r <= 0.55` (there the claim needs the combinatorial ">= 3 atoms" rigidity, which this method lacks).

## 2. Proof

Normalise `a0 = 1` (the statement is homogeneous of degree `r`; if `A0 = 0` then `m_A <= Re A0 = 0`).
Assume `m_A > 0`. Five lemmas, each valid for every `N`.

**D1 (single-pattern aiming).** `|C_T| <= d / A^{|T|}`. *Proof.* Average `Re h(t)` over the phases of
`t` on the fibre where `t^T` has phase `phi`: all characters except `1` and `t^T` average out, giving
`Re A0 + A^{|T|} Re(C_T e^{i phi})`; the minimum is at most the smallest fibre average,
`m_A <= Re A0 - A^{|T|}|C_T| <= 1 - A^{|T|}|C_T|`. ∎ At `A = 1/2`: `|C_1|,|C_2| <= 2d`, `|C_12| <= 4d`
(all three simultaneously attainable: `h = 1 - 2d t_1 - 2d t_2 + 4d t_1 t_2`).

**D2 (mirror control — the new ingredient).** `|U_I| <= |C_I| + 2w` for each interval `I`.
*Proof.* Rotate so `A0 = 1`, write `c_l = a_l + i b_l`: `sum a_l = 1`, `sum b_l = 0`. Then
`U_I - conj(C_I) = sum_l (c_l - conj c_l) conj(v_l^I) = 2i sum_l b_l conj(v_l^I)`, so
`|U_I| <= |C_I| + 2 beta`, `beta := sum|b_l|`. And
`S = sum sqrt(a_l^2+b_l^2) >= sqrt((sum|a_l|)^2 + (sum|b_l|)^2) >= sqrt(1 + beta^2)`
(triangle inequality for the vectors `(|a_l|,|b_l|)`), i.e. `beta <= w`. ∎
*Plain reading: ELEGANT2 §6.1 refuted the determinant route because the mirror moments are
"unconstrained". They are not: they differ from the conjugate junk only by the atoms' imaginary
parts, and imaginary parts are paid for in `S = e_1(x)`, which the Bernstein integral sees.*
Also trivially `|U_I| <= S`. Put `u_I := min(S, delta_I + 2w)`, `delta_I := |C_I|`.

**D3 (Cauchy–Binet + Hadamard, ELEGANT.md).** `e_k(x) >= |E_k(M)| / 3^k`, every `N`; and `e_1(x) = S`.

**D4 (minors with junk).** `E_2(M) = 3 - sum_I L_I U_I`,
`E_3(M) = 1 - sum_I L_I U_I + L_1 L_2 U_12 + U_1 U_2 L_12` (3x3 determinant, unit diagonal). Hence
`|E_2| >= 3 - sum_I delta_I u_I =: E2`, `|E_3| >= 1 - sum_I delta_I u_I - delta_1 delta_2 u_12 - u_1 u_2 delta_12 =: E3`,
both decreasing in every `delta_I`, so by D1 worst at `delta_1 = delta_2 = d/A`, `delta_12 = d/A^2`.

**Chain tool.** `prod_l (1 + x_l t) >= P(t) := 1 + S t + (E2)_+ t^2/9 + (E3)_+ t^3/27` for `t >= 0`, and by
the Bernstein representation `x^r = (r sin(pi r)/pi) int_0^inf log(1+xt) t^{-1-r} dt`,
`cost >= chain(d,w) := sum_i Re(y_i^r)` where `P(t) = prod_i (1 + y_i t)` (the identity
`c_r int log(1+yt) t^{-1-r} dt = y^r` continues analytically to `y` off the negative axis; `P` has
non-negative coefficients so no `y_i` is a negative real).

**D5 (depth-A staircase peel).** Sort `x_1 >= x_2 >= ...`. Aim at the largest atom,
`t_j = -A conj(v_{1j})`: atom 1 is multiplied by `(1-A)^2`, every other by at most `(1+A)^2`, so
`mu <= (1-A)^2 x_1 + (1+A)^2 (S - x_1)`, i.e. `S - x_1 >= b_1 := (mu - (1-A)^2 S)/((1+A)^2 - (1-A)^2)`.
Aim slot 1 at atom 1 and slot 2 at atom 2: both get `(1-A)(1+A)`, so
`S - x_1 - x_2 >= b_2 := (mu - (1-A^2) S)/((1+A)^2 - (1-A^2))`. Clip `0 <= b_2 <= b_1 <= S`. Then the sorted
`x` is majorised by `y = (S-b_1, b_1-b_2, b_2)` and `sum x^r` is Schur-concave:
`cost >= peel(d,w) := (S-b_1)^r + (b_1-b_2)^r + b_2^r`. (Lossy at the block — `1.385 < 1.442` — but
strong at large junk, where the chain tool is empty.)

**Trivial tool.** `cost >= S^r`. This settles `S >= kappa0 mu` and all `d >= 1 - 1/kappa0`.

**Assembly.** It remains to show `F(d,w) := max(chain, peel, S^r) >= 3^{1-r}(1-d)^r` on
`0 <= d <= 1 - 1/kappa0`, `0 <= w <= sqrt(kappa0^2-1)`, `S = sqrt(1+w^2)`.

*(i) Corner `R0 = {d <= 0.02, w <= 0.12}`, by hand, every `r in (0,1]`.* Use `u_I <= delta_I + 2w`.
Coefficientwise domination `P(t) >= (1 + mu t/3)^3` needs `S >= mu` (true), `E2 >= 3 mu^2`, `E3 >= mu^3`.
At `A = 1/2`: `sum delta u = 24 d^2 + 16 d w`, so
`E2 - 3mu^2 = d(6 - 27 d - 16 w)` and
`E3 - mu^3 = d(3 - 27 d - 31 d^2 - 16 w - 40 d w - 16 w^2)`; both brackets are decreasing in `d, w` and at
`(0.02, 0.12)` equal `3.54` and `0.2012`. Then Bernstein gives `cost >= 3 (mu/3)^r`. *This is the rigid
part: first-order aiming loss `A delta` beats the second-order damage `delta^2 + delta w`.*

*(ii) Outside `R0`: monotone cell certificate* (`deptha_gridcert.py`). On a cell
`[d0,d1] x [w0,w1]` every tool is bounded below by monotonicity (`e_1 = S(w0)`; `E2, E3, b_1, b_2` at
`(d1, w1)`; total `S(w0)`) and the target above by `T(d0)`; adaptive bisection, tolerance `1e-6`.
Result at `A = 1/2`, `r = 2/3`: **279 cells, 0 failures**. Independent dense scan (`deptha_dense.py`,
700 x 700): `min F/T = 1.00272` at `(d,w) = (0, 0.1214)`, i.e. on the corner's edge; away from the block
the margin is `>= 1.2%` (`deptha_cert4.py`). ∎

## 3. Network corollary (via THEOREM P3C-1, singleton partition)

`R_singleton(pop, A) = kappa0 (m_A)_+ / cost^{1/r} <= 1` for EVERY population is exactly Theorem DA, so
(PFP) holds with `P` = singletons for every network whenever the available depth is `>= 1/2`
(`m_A` is non-increasing in `A`, so any `A* >= 1/2` may be used at depth `1/2`). Hence:

> **COROLLARY DA-2.** `nu = 3`, `q = 2` (and each certified `q in [2,3)`): every gated-chain network of
> **pure** single-frequency neurons, **any width, any junk level, any K, any p**, whose support has
> `A*(p, {2k}) >= 1/2`, satisfies (T): `gamma_q <= 3^{-(nu-q)/q} B Gamma_{nu/q}(p)`, equality at the
> share-weighted block. `A* >= 1/2` is PROVED for dissociate supports (P3A Riesz product) and for
> `K = 1` at every `p >= 3` (`A* = cos(pi/p)`), which removes both the width restriction and the
> `(1-eps_p)^{-1}` blemish of ELEGANT2 Corollary N1. Actually `A* >= 0.47` suffices.
> This replaces P3C's partition mixture, its LP and the measured `A_lam` at `nu = 3`.

On the paper's `Gamma_{3/2}`-optimal supports (P3C §5.2, MEASURED `A*`, not certificates): `p = 71, 97,
113, 127` have `A* = 0.511, 0.496, 0.503, 0.497 >= 0.47` — covered once `A*` is certified, and note
that the `0.47` (rather than `0.5`) is what admits `p = 97, 127`; `p = 59` (`A* = 0.386`) is not covered.

Not covered: supports with `A* < 0.47` (there P3C's mixture threshold `0.304` is still needed and
still unproved); `nu >= 4` (singleton threshold
`0.63 > 1/2`); impure neurons (P3C step 1; not attempted here).

## 4. Numerical record (`FINAL_RUNS/DEPTHA/`)

| script | result |
|---|---|
| `deptha_lemmas.py` (3 seeds, ~9000 populations, N = 2..7) | D1, D2, D5 never violated (worst slack ratios 1.015, 1.007, +0.05); D3 worst `1.0000000000006` (char-poly recheck); theorem ratio min `1.00011` |
| `deptha_e2e.py` (Nelder–Mead adversary MINIMISING true cost / each tool, 300 starts) | chain `1.0000000000016` (equality at block), peel `1.0248`, `S^r` `1.0000` — every tool is a sound lower bound |
| `deptha_gridcert.py 0.5 r` | `r = 2/3`: 279 cells, 0 fail; `r = .67 .. .99`: 264 .. 2 cells, 0 fail |
| `deptha_gridcert.py A 2/3 0.01 0.06` | `A = 0.47, 0.48`: pass; `A = 0.46`: FAILS near `(d,w) = (0.126, ...)` |
| `deptha_gridcert.py 1.0 r` | pass at `r = 0.6, 2/3, 0.75, 0.9`; fail at `0.5, 0.55` |
| `deptha_cert0/1/2/3/4.py`, `deptha_dense.py` | development scans; chain alone closes only `A >= ~0.72`, chain + 1-peel fails at `(d,S) = (0.10, 1.05)`, the 2-peel closes it |

Caveat for a referee: step (ii) is fp64 (`numpy.roots` on cubics, tolerance `1e-6` vs margin `2.7e-3`), not
interval arithmetic. Everything else is by hand.

## 5. Routes that died here (with breaking points)

1. **Depth via splitting `1 + A u v = (1/2) sum_± (1 + u v e^{±i theta})`, `cos theta = A`** — turns depth `A`
   into a torus problem with `2^n N` atoms, but multiplies cost by `2^{n(1-r)}`: worse than trivial.
2. **Bridge from Theorem A (`m >= m_A - E`)**: `h = a + b u_1` has `m_A = a - Ab`, `m = a - b`;
   at `b = 0.5`, `max(a, kappa0 m) = 1.30 < kappa0`. No `r`-free bridge exists.
3. **Product / `e_k` ladder at depth `A`**: FALSE. `h = 1 + u_1` (two atoms) has `m_{1/2} = 1/2 > 0`
   and `e_3 = 0`; consequently (DEPTH-A) fails as `r -> 0` and any proof must use `r` (here: peel + `S^r`).
4. **Any finite family of modulus-linear constraints `sum x_l |Lambda_pi(v_l)| >= m_A`** with
   depth-admissible `Lambda_pi`: `|Lambda_pi(v)| >= (1-A)^n > 0`, so nothing is ever killed and the
   triangle inequality is strict at the block — cannot be tight. Rigidity must come from `E_k(M)`.
5. **Hermitian variants** (`M_x = sum x_l m_l m_l^*`, or the real-part matrix): same cross term
   `delta * sqrt(2 S (S-1))`; no gain over D2.
6. **Three-circles / log-convexity in radius**: `log min|h|` is concave in `log rho`, which bounds
   depth `A` from depth 1, the wrong direction.

## 6. Open

* `n >= 3`: D1, D2, D3, D5 hold verbatim for every `n`; D4 needs `|E_k(I + L + U)|` bounds for the
  `(n+1) x (n+1)` chain matrix (only interval patterns enter). At `nu = 4, q = 2` the singleton
  threshold is `0.63`, so the network use needs the one-block partition too.
* Lower the depth from `0.47` toward `0.4284` (sharper peel with averaged aim measures), or prove
  P3C's mixture at `0.304`, to cover the paper supports with `A* in [0.38, 0.47)`.
* Interval-arithmetic version of step (ii). Impure neurons.
