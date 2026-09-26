# P1 (the sharp constant): a certified hybrid bound, an elementary tail-robust Key Lemma, and three routes killed

Fifth pass on the global-optimality programme. Continues `LEMMA_ATTEMPT.md` (agent 1),
`GLOBAL_OPT.md` (agent 2), `GLOBAL_OPT3.md` (agent 3), `GLOBAL_OPT4.md` (agent 4), and runs
alongside `ADVERSARY.md` (the adversary agent, same day). My target was

> **(P1)** For every `N >= n+1` and every (KL) system, `sum_l |c_l|^r >= (n+1)^{1-r} |A|^r`.

**Status: PARTIAL, with (P1) now CERTIFIED at plain weight decay `q = 2` for `n = 2, 3, 4`.**

**Priority note, stated up front.** `ADVERSARY.md` §6 reaches the same two-variable programme
that I reached independently here (their `G(beta, tau)` is my `Val(s, R)`; their "drop the tail /
peel / tail's own price" are my `A_G` / `Lad` / Lemma C). Theirs was written the same day and I
read it only after my own derivation and code were complete; **the programme should be credited to
them.** What is new here and not in `ADVERSARY.md`:

1. **The certified version they explicitly flagged as missing** ("a certified version would be a
   routine interval branch-and-bound in two variables, which I did not run"). I ran it: §3.3 gives
   a *cell-wise guaranteed* lower bound, not a grid sample, and it closes `q = 2` at
   `n = 2, 3, 4` with `1.1%` to `3.4%` of margin (§3.4).
2. **A rigorous `r`-map** (§3.5): certified `r*(2) >= 0.7969`, `r*(3) >= 0.6133`, `r*(4) >= 0.4234`
   — the first two agree with their measured `0.8433 / 0.6268` to within the conservatism of the
   interval bound, and **`n = 4` is new**. `n = 5` at `q = 2` does **not** close even at grid
   `400 x 400` (ratio `0.9872`), which locates the method's edge.
3. **Lemma T**, an independent and more elementary derivation of the "drop the tail" step
   (bijection identity + Hadamard + AM-GM), which avoids Theorem B and its continuum quantity
   `min_{T^n} Re h` altogether (§2.1).
4. Three routes killed, none of them in any inherited write-up: the **diagonal relaxation**
   (§5.3, exact counterexample), **mixtures of `(n+1)`-atom solutions** (§5.2, dead twice over),
   and the **"equal moduli at every stationary point" rigidity** (§4.3, explicit counterexample).
5. A new exact reformulation of (P1) at stationary points, `sum_l |Lambda(v_l)|^{-sigma} >= n+1`
   (§4.2), and the **Lagrange form** of the first open case (§5.1).

All float64, local CPU, **no pods, $0**. Scripts `heracles/p1c_*.py`, outputs
`heracles/FINAL_RUNS/P1C/`. No background jobs remain.

---

## 0. Symbols, with a plain reading of each

| symbol | meaning | plain reading |
|---|---|---|
| `n` | number of operand slots | how many numbers are added; `nu = n+1` is the homogeneity degree |
| `q in [2, nu]`, `r = q/nu in (0,1]` | penalty exponent and its ratio | `q = 2` is plain weight decay, `q = nu` is Morwani's `L_{2,nu}`; `r` is how cheap it is to split one feature over several neurons (`r = 1` = free) |
| `N` | number of atoms | one atom per neuron carrying this frequency (agent 3 §2: this needs no purity) |
| `v_lj` | unit-modulus phase of atom `l` in slot `j` | the phase of that neuron's `j`-th embedding at this frequency |
| `c_l` | complex amplitude of atom `l` | that neuron's signed clean contribution |
| `A` | the clean total (normalised to `1` throughout) | the feature's Fourier amplitude |
| `(KL)` | `sum_l c_l prod_{j in T} v_lj = A [T empty]` for every `T` subset of `{1..n}` | "`N` atoms carry clean amplitude `A` and cancel all `2^n - 1` junk sign patterns" |
| `F = sum_l \|c_l\|^r` | the cost | what the penalty prices, per frequency |
| `x_1 >= x_2 >= ...` | the sorted moduli `\|c_l\|` | biggest neuron first |
| `S` | the `n+1` **largest** atoms | the "core" |
| `s := x_{n+1}` | the smallest modulus in the core | the core's marginal atom (`ADVERSARY`'s `beta`) |
| `R := sum_{l not in S} x_l` | the **tail mass** | everything parked below the core (`ADVERSARY`'s `tau`) |
| `Lambda(z) = sum_T lam_T prod_{j in T} z_j` | a multi-affine **test polynomial**; *normalised* means `lam_empty = 1` | a probe whose torus average is 1 |
| `D_h(n)` | agent 3's peeling constants (`GLOBAL_OPT3` §3) | how much mass survives after the `h` biggest atoms are killed |
| `sigma := r/(1-r)` | | the exponent in the stationarity identity |

Two standing identities, both exact and re-verified here (`p1c_verify.py` V1, residual `< 3e-13`):

    (KL)  <=>  h(y) := sum_l c_l prod_j (1 + y_j v_lj)  ==  A   for all y in C^n
          <=>  sum_l c_l prod_j (u_j - v_lj)            ==  A prod_j u_j   for all u in C^n.   (dagger)

---

## 1. Exact statements

> **LEMMA T (the KEY LEMMA WITH A TAIL; proved, every `n >= 1`, every `N >= n+1`, every
> `r in (0,1]`).**
> Let `(c_l, v_l)` satisfy (KL). Let `S` be **any** set of `n+1` atoms and `R := sum_{l not in S}|c_l|`.
> Then
>
>     sum_{l in S} |c_l|^r  >=  (n+1)^{1-r} * max(|A| - 2^n R, 0)^r .
>
> Agent 1's Key Lemma is the case `R = 0` (`N = n+1`).
> *Plain reading: the Key Lemma does not need the population to have exactly `n+1` neurons. It
> needs the core to carry almost all of the mass, and it degrades only linearly in the leftover.*
> *Relation to `ADVERSARY` §6 step 1:* the same inequality, derived instead from agent 2's
> Theorem B plus `R_S >= 1 - 2^n tau`. Lemma T uses only (dagger), `|u - v| <= 2`, Hadamard and
> AM-GM, so it needs no continuum minimisation and no Theorem B.

> **LEMMA C (tail cost; proved, elementary; = `ADVERSARY` §6 step 3).** With `S` = the `n+1`
> **largest** atoms and `s := x_{n+1}`,
>
>     sum_{l not in S} |c_l|^r  >=  max( R^r ,  R * s^{r-1} ) ,
>
> and `R > 0` forces `s > 0`.
> *Plain reading: mass parked below the core is charged at least at the marginal price of the
> core's smallest atom — a price that blows up as that atom shrinks. This is what makes one
> argument cover every `N` at once.*

> **THEOREM H (the hybrid bound: assembly proved, value CERTIFIED cell-wise).**
> For every (KL) system with `|A| = 1`, with its own `(R, s)`,
>
>     F  >=  Val(R, s) := max( A_G(R), Lad(R, s) )  +  max( R^r, R s^{r-1} ),
>     A_G(R)   := (n+1)^{1-r} max(1 - 2^n R, 0)^r                                    [Lemma T]
>     Lad(R,s) := min { sum_{l=1}^{n+1} x_l^r : x_1 >= ... >= x_n >= x_{n+1} = s,
>                        sum_{h < l <= n+1} x_l + R >= D_h(n), h = 0..n }             [agent 3]
>
> and hence `F >= Lhyb(n, q) := inf over admissible (R, s) of Val(R, s)`. The infimum is `>=`
> the target `(n+1)^{1-r}` — i.e. **(P1) holds for every `N`** — in the cells of §3.4, the
> infimum being attained only at `R = 0`, where Lemma T *is* the Key Lemma.
> The certificate is **analytic** outside the box `[R1, R2] x [0, 1/(n+1)]` (§3.2) and, inside it,
> a **cell-wise guaranteed** lower bound (§3.3): every ingredient of `Val` is monotone in each
> variable and every ladder vertex is an *affine* function of `(R, s)`, so a lower bound valid on
> each **whole cell** is available in closed form. This is the interval version `ADVERSARY` §6
> flagged as not run.

> **PROPOSITION S (the stationarity picture; proved, and one conjecture refuted).**
> At a local minimum of `F` subject to (KL), after deleting the zero atoms, there is a normalised
> multi-affine `Lambda` with
>
>     (i)   c_l Lambda(v_l) = |c_l|^r / F    for every atom     (so |Lambda(v_l)| = |c_l|^{r-1}/F)
>     (ii)  alpha_lj / Lambda(v_l) is REAL, for every atom l and slot j,
>           where Lambda(z) = alpha_lj + z_j beta_lj with alpha, beta evaluated at v_l's other slots
>
> and (i) alone forces the **value identity**
>
>     F  =  ( sum_l |Lambda(v_l)|^{-sigma} )^{1-r} ,      sigma = r/(1-r),
>
> so that **(P1) at stationary points is exactly `sum_l |Lambda(v_l)|^{-sigma} >= n+1`.**
> If all `|Lambda(v_l)|` were equal, (P1) would follow in two lines (§4.3). **That rigidity is
> FALSE**: `p1c_stat.py` exhibits a stationary branch at `n = 2`, `N = 4`, `q = 2` with moduli
> `(0.336766, 0.336766, 0.163634, 0.163634)`, spread `2.0582`, and
> `sum |Lambda|^{-sigma} = 3.8435` (so `F/target = 1.086096`: no counterexample to (P1), but the
> "equal moduli" route is dead).

> **PROPOSITION L (the Lagrange form of the first open case; proved).**
> At `N = 2^n` the system (KL) is square. Let `Lambda^{(l)}` be the multi-affine Lagrange basis at
> the atoms (`Lambda^{(l)}(v_m) = delta_{lm}`). Then
>
>     c_l = A * Lambda^{(l)}(0)        and      sum_l Lambda^{(l)} == 1   (partition of unity),
>
> so (P1) at `N = 2^n` is: *for any `2^n` points of the torus `T^n`, the multi-affine Lagrange
> basis evaluated at the ORIGIN satisfies* `sum_l |Lambda^{(l)}(0)|^r >= (n+1)^{1-r}`, *given that
> those values sum to 1.* At `n = 1` this is one line (§5.1); `n = 2` is agent 4's six-angle
> determinant problem in coordinate-free form.

> **REFUTATION D (the diagonal relaxation is lossy; exact counterexample).**
> Putting every slot to the same value `y_j = y` turns (KL) into the one-variable identity
> `sum_l d_l p_l(y) == A` with `p_l` monic of degree `n` with **all roots on the unit circle** and
> `|d_l| = |c_l|`. This is a genuine relaxation of (P1) to a classical object, and it is **false**:
> at `n = 2`, `p_1 = y^2 + 1`, `p_2 = y^2 - 1`, `d = (1/2, -1/2)` gives `d_1p_1 + d_2p_2 == 1` with
> cost `2^{1-r}`, a ratio `0.873580` to `3^{1-r}` at `q = 2`. Confirmed by independent global
> search (`p1c_diag.py`): minimum `1.259921 = 2^{1-r}` at both `N = 3` and `N = 4`.

> **REFUTATION M (mixtures of `(n+1)`-atom solutions; dead twice over).**
> (a) *Direction.* `t -> t^r` is **sub**additive for `r <= 1`, so a decomposition
> `c = sum_m c^{(m)}` gives `F(c) <= sum_m F(c^{(m)})` — an upper bound, never the lower bound (P1)
> needs. (b) *Existence.* For `n >= 2` and generic phases **no `(n+1)`-subset supports a (KL)
> solution at all**: at `n = 2` a 3-atom junk-free population must be common-phase
> (`LEMMA_ATTEMPT` §4 rigidity), a positive-codimension condition. The smallest support a generic
> configuration admits is `2^n`, and for those the Key Lemma is exactly what is missing.

---

## 2. The two new lemmas, proved

### 2.1 Lemma T

*Step 1.* (KL) is equivalent to (dagger) (expand `prod_j (u_j - v_lj)` and match the coefficient of
each monomial; agent 1 §3).

*Step 2.* Put `A' := |A| - 2^n R`. If `A' <= 0` the conclusion is trivial, so assume `A' > 0`.
Fix `l0 in S` and a **bijection** `sigma` from the `n` slots onto `S \ {l0}`, and substitute
`u_j := v_{sigma(j), j}` in (dagger). Every atom `l in S`, `l != l0`, equals `sigma(j)` for exactly
one `j`, so its product contains the zero factor `(u_j - v_lj)` and its term vanishes. What is left is

    c_l0 prod_j (u_j - v_{l0,j})  +  sum_{l not in S} c_l prod_j (u_j - v_lj)  =  A prod_j u_j .

All `|u_j| = 1`, so `|prod_j u_j| = 1` and `|prod_j (u_j - v_lj)| <= 2^n`. Hence

    |c_l0| * prod_j d^{(j)}_{sigma(j), l0}  >=  |A| - 2^n R = A' > 0,
    d^{(j)}_{l,m} := |v_lj - v_mj| .                                                          (T2)

*Step 3.* (T2) forces every `d^{(j)}_{l,l0} > 0` for `l in S \ {l0}` (a zero factor would make the
left side `0 >= A' > 0`). So the `n+1` points `v_lj`, `l in S`, are pairwise distinct in every slot.
Take the **geometric mean of (T2) over all `n!` bijections**: a fixed pair `(j, l)` occurs as
`sigma(j) = l` in exactly `(n-1)!` of them, so the geometric mean of `prod_j d^{(j)}_{sigma(j), l0}`
is `(prod_j prod_{l in S, l != l0} d^{(j)}_{l,l0})^{1/n}`, and since a geometric mean of positive
numbers is at least their minimum,

    |c_l0|  >=  A' / ( prod_j prod_{l in S, l != l0} d^{(j)}_{l,l0} )^{1/n} .                 (T3)

*Step 4.* Multiply (T3) over `l0 in S`. The double product runs over all **ordered** pairs of `S`,
i.e. it is the square of the product over unordered pairs, so with
`V_j := prod_{l < m, l,m in S} |v_lj - v_mj|`,

    prod_{l in S} |c_l|  >=  A'^{n+1} ( prod_j V_j^2 )^{-1/n} .

`V_j` is the modulus of the Vandermonde determinant of the `n+1` points `v_lj` on the unit circle;
each of its `n+1` rows has `n+1` entries of modulus 1, hence Euclidean norm `sqrt(n+1)`, so
**Hadamard's inequality** gives `V_j <= (n+1)^{(n+1)/2}`, `prod_j V_j^2 <= (n+1)^{n(n+1)}` and

    prod_{l in S} |c_l|  >=  ( A' / (n+1) )^{n+1} .

*Step 5.* AM–GM: `sum_{l in S} |c_l|^r >= (n+1) (prod_{l in S}|c_l|)^{r/(n+1)} >= (n+1)^{1-r} A'^r`. ∎

**Checked** (`p1c_verify.py` V2): (T2) and the conclusion on random (KL) systems at
`n = 2` (`N = 5,6,7`) and `n = 3` (`N = 9,10`), every one of the `n!(n+1)` bijection instances per
system — **zero violations in 9900 bijection checks and 750 Key-Lemma-with-tail checks**, with
worst slack `-0.174` (not a near miss).

### 2.2 Lemma C

Every tail atom has modulus `t` with `0 <= t <= s` because `S` holds the `n+1` largest. Then
`t^r = t * t^{r-1} >= t * s^{r-1}` (as `r - 1 <= 0`), and summing gives `sum_{tail} t^r >= R s^{r-1}`.
Separately `sum_l t_l^r >= (sum_l t_l)^r = R^r` by subadditivity of `t -> t^r`. And if `R > 0` some
tail atom is positive, so `s > 0`. ∎

---

## 3. Theorem H: why the two tools are complementary, and how far the certificate goes

### 3.1 Why neither tool works alone

Agent 3's peeling ladder constrains only the sorted tail sums, and F3-2 proved it cannot reach
`(n+1)^{1-r}` by itself: its minimising vertex at `n = 2, q = 2` is
`x = (0.350481, 0.350481, 0.299038)` with value `0.999397 * target`. **That vertex has no tail at
all** (`R = 0`) — and at `R = 0` Lemma T *is* the Key Lemma and forbids it outright. Conversely
Lemma T degrades linearly in `R` and is worthless once `2^n R >= 1`, exactly where the ladder is
strong. So the two tools fail in disjoint regions of the single scalar `R`, and their maximum is
what has to be minimised. Without Lemma C the relaxation is vacuous — all the mass can hide in an
unbounded number of tail atoms, giving `1 < (n+1)^{1-r}`.

### 3.2 The certificate outside the box is analytic

* **Small `R`.** With `rho := 2^n R`, `A_G(R) + R^r >= (n+1)^{1-r}` is
  `f(rho) := rho^r 2^{-nr} - (n+1)^{1-r}[1 - (1-rho)^r] >= 0`. Here `f(0) = 0` and
  `f'(rho) = r[ rho^{r-1} 2^{-nr} - (n+1)^{1-r}(1-rho)^{r-1} ]`, whose bracket is **strictly
  decreasing** (`rho^{r-1}` decreases, `(1-rho)^{r-1}` increases). So `f` rises then falls and has a
  unique positive root `rho_1`; with `R1 := rho_1 / 2^n`, `f >= 0` on `[0, R1]`.
  *Plain reading: for a small tail, the Key Lemma's linear loss is beaten by the tail's own cost
  `R^r`, which for `r < 1` dwarfs `R`.*
* **Large `R`.** Every core atom has `x_l >= s`, so by Lemma C `F >= (n+1) s^r + R s^{r-1}`.
  Minimising the right side over `s > 0` gives `s* = (1-r)R/(r(n+1))` and the closed form

      F  >=  K_n(r) R^r ,     K_n(r) := [ r(n+1)/(1-r) ]^{1-r} / r ,

  so `F >= (n+1)^{1-r}` as soon as `R >= R2 := [ (n+1)^{1-r} / K_n(r) ]^{1/r}`. No ladder needed.
* **Large `s`.** If `s > 1/(n+1)` then `F >= (n+1) s^r > (n+1)^{1-r}` at once.

So only `(R, s)` in `[R1, R2] x [0, 1/(n+1)]` can be dangerous.

### 3.3 The certificate inside the box is cell-wise, not sampled

`Lad(R, s)` is the minimum of a **concave** function over a polyhedron, so it sits at a vertex, and
the polyhedron's normals do not depend on `(R, s)` — only its right-hand side does. Hence every
vertex is an **affine** function `x_l(R, s) = p_l + q_l R + w_l s`, and over a grid cell
`[R_lo, R_hi] x [s_lo, s_hi]` each coordinate is minimised at an explicit corner. Since `t -> t^r`
is increasing, that gives a *guaranteed lower bound on `Lad` over the whole cell*. Likewise
`A_G` is decreasing in `R` (use `R_hi`), `R^r` is increasing in `R` (use `R_lo`) and `R s^{r-1}` is
increasing in `R` and decreasing in `s` (use `(R_lo, s_hi)`). A vertex basis is discarded only when
some constraint's affine slack has a **negative maximum** over the cell, i.e. when it is provably
infeasible everywhere on that cell, so the minimum runs over a superset of the true vertices and
the bound stays valid. `p1c_certify.py` does exactly this; its vectorised and naive implementations
agree to `0.0` on 400 random cells.

Because the cell bound is conservative, **a coarser grid gives a weaker but still valid bound: if
it closes, it closes.**

### 3.4 Results at `q = 2` (plain weight decay — the paper's case)

`p1c_certify.py`. "CLOSES" means the guaranteed cell minimum is at or above the target, i.e.
**(P1) holds for every `N` in that cell**.

| `n` | `nu` | `r = 2/nu` | target `(n+1)^{1-r}` | `R1` | `R2` | grid | certified cell minimum | ratio | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 3 | 0.666667 | 1.442249570 | 0.016977 | 0.384900 | `700 x 700` | 1.488587435 | **1.032129** | **CLOSES** |
| 3 | 4 | 0.500000 | 2.000000000 | 0.014692 | 0.250000 | `500 x 500` | 2.068253204 | **1.034127** | **CLOSES** |
| 4 | 5 | 0.400000 | 2.626527804 | 0.008442 | 0.185903 | `500 x 500` | 2.654589598 | **1.010684** | **CLOSES** |
| 5 | 6 | 0.333333 | 3.301927249 | 0.004437 | 0.148148 | `400 x 400` | 3.259666357 | 0.987201 | short by 1.3% |

`n = 2, 3` reproduce `ADVERSARY` §6's `q = 2` result in certified form (their point-sampled margins
were `1.0224` and `1.0225`; the certified ones are `1.0321` and `1.0341`, higher because the
certified sweep starts at `R1` rather than at `eps*/2` and so skips the shallowest part of the
curve). My `R1` and their `eps*` are the same quantity computed independently, and they agree:
`0.016977` vs `1.698e-02` at `n = 2, q = 2`, and `0.014692` vs `1.469e-02` at `n = 3, q = 2`.
**`n = 4` is new.** `n = 5` is the first miss; since the cell bound is conservative, this does not
show the programme itself fails there (§7 item 5).

One further cell, to show the method is not confined to `q = 2`: at `n = 2`, `q = 2.25`
(`r = 0.75`), grid `700 x 700`, the certified minimum is `1.325354986` against target
`1.316074013`, ratio **`1.007052`** — CLOSES.

### 3.5 The certified `r`-map

`p1c_rstar.py` bisects on `r` using the cell-wise bound, so the reported `r*` is a value **below
which the programme is certified to clear the target**, not merely measured to.

| `n` | certified `r* >=` | i.e. `q* >=` | bracket | grid | `ADVERSARY` §6 measured `r*` |
|---|---|---|---|---|---|
| 2 | **0.79688** | 2.3906 | `[0.79688, 0.80208]` | `600 x 600` | 0.8433 |
| 3 | **0.61328** | 2.4531 | `[0.61328, 0.61523]` | `200 x 200` | 0.6268 |
| 4 | **0.42344** | 2.1172 | `[0.42344, 0.42578]` | `110 x 110` | — (not covered) |
| 5 | — | — | `q = 2` (`r = 1/3`) already short at `0.9872` | `400 x 400` | — (not covered) |

Two readings. (a) The certified `r*` tracks the measured one closely at `n = 3` (`0.6133` vs
`0.6268`) and lags a little at `n = 2` (`0.7969` vs `0.8433`); the lag is the price of the interval
bound and shrinks with refinement — at `n = 2`, `r = 0.79167` the bound reads `0.9967` on a
`200 x 200` grid and `1.0008` on `600 x 600`. (b) The method's ceiling is real, not a grid artefact:
`R1 -> 0` as `r -> 1` (at `n = 2`, `q = 2.75`, `R1 = 0` already), so the analytic small-tail region
vanishes and only the ladder is left, which F3-2 proved cannot reach the constant. The open window
`r in (r*, 1)` recorded by `ADVERSARY` stands.

The point-wise profiles (`p1c_margin.py`, which is the `ADVERSARY`-style evaluation) agree about
where the method dies: `n = 3, q = 2.5` (`r = 0.625`) gives `1.002441`, `n = 3, q = 3` (`r = 0.75`)
gives `0.994773`.

### 3.6 Honest scope

* Lemma T, Lemma C and the assembly `F >= Val(R, s)` are **proved**.
* `Lhyb >= (n+1)^{1-r}` is **proved analytically outside the box** and established inside it by a
  finite computation whose every step is a guaranteed *interval* bound, not a sample. What stands
  between this and a closed proof is (i) float64 rounding — the margins are `1.1%` to `3.4%`, i.e.
  some `10^{13}` times the rounding, but this is not interval arithmetic; and (ii) a hand-checkable
  version: `Val(R, s)` is elementary and the ladder vertices are explicit, so a by-hand
  minimisation should be possible (§7 item 1).
* The ladder constants `D_h(n)` are agent 3's. For `n = 2, 3` they are exact algebraic numbers
  (`kappa_1 = 1/2`, `kappa_2 = 3 sqrt3/8`, `kappa_3 = 49/72`); for `n >= 4`, `D_1(n)` uses `S_n`
  computed to fp64, so the `n = 4` cell inherits that caveat.
* `Val` is stated for `|A| = 1` and `N >= n+2`; at `N = n+1` (so `R = 0`) Theorem H reduces to
  agent 1's Key Lemma, which is unconditional.
* The certified `r*` is a **lower bound on the programme's own crossover**, not on the truth: the
  gap between my `r*(2) >= 0.7969` and `ADVERSARY`'s measured `0.8433` is the price of the interval
  bound, and shrinks under refinement (§3.5).

---

## 4. The stationarity picture

### 4.1 The first-order conditions

Fix `M` and minimise `F` over all (KL) systems with at most `M` atoms. For each `M` the minimum is
attained: the phases live on a compact torus and `F <= (n+1)^{1-r}` (the block is feasible) bounds
every `|c_l|`. Delete the zero atoms and write the Lagrangian with multipliers `lam_T`:

    L = sum_l |c_l|^r - Re sum_l c_l Lambda(v_l) + Re lam_empty ,
    Lambda(v) := sum_T conj(lam_T) prod_{j in T} v_j .

Stationarity in `c_l` gives (i) after the normalisation `lam_empty = 1` — which is forced, since
pairing (KL) with `Lambda` gives `sum_l c_l Lambda(v_l) = lam_empty A` while (i) makes the left side
`sum_l |c_l|^r / F = 1`. Stationarity in the phase angle `phi_lj` of `v_lj = e^{i phi_lj}` gives
`Im[c_l v_lj beta_lj] = 0`, which with (i) is (ii).
*Plain reading of (ii): every atom sits at a critical point of `|Lambda|` along every slot circle —
at a local max or a local min of the dual probe.*

**Checked**: `p1c_census.py` finds `|lam_empty| = 1` to `1.9e-6` or better and `res_i <= 2.3e-15`,
`res_ii <= 8.1e-6` at every converged minimum in seven cells (394 converged runs).

### 4.2 The value identity

From (i), `|Lambda(v_l)| = |c_l|^{r-1}/F`, so `|c_l| = (F |Lambda(v_l)|)^{-1/(1-r)}` and

    F = sum_l |c_l|^r  =>  F^{1/(1-r)} = sum_l |Lambda(v_l)|^{-sigma},   sigma = r/(1-r),

i.e. `F = ( sum_l |Lambda(v_l)|^{-sigma} )^{1-r}`. Hence:

> **(P1) at stationary points is exactly `sum_l |Lambda(v_l)|^{-sigma} >= n+1`.**

At the `(n+1)`-block, `Lambda == 1` and the sum is `n+1` — equality. Verified to `8e-8` in every
cell (`p1c_census.py`, column `sum|L|^-sigma`).

Two further exact consequences, the only structure I found on the dual: with the probability
weights `pi_l := |c_l| |Lambda(v_l)|` (they sum to 1 by (i)),

* `E_pi[ t_{l,S} ] = 1` for every subset `S` of `{1..n}`, where
  `t_{l,S} := Lambda(v_l with the slots in S set to 0) / Lambda(v_l)` — because `Lambda` with slots
  zeroed is again multi-affine with constant coefficient 1. By (ii), `t_{l,{j}}` is **real** for
  every `l` and `j`.
* `sum_l pi_l / Lambda(v_l) = A = 1`, so by the triangle inequality
  `sum_l |Lambda(v_l)|^{-(sigma+1)} >= sum_l |Lambda(v_l)|^{-sigma}`.

### 4.3 The rigidity route, and its refutation

If all `|Lambda(v_l)|` on the support were equal, all `|c_l| = m` would be equal, and then

    sum_l c_l = A = 1  =>  N m >= 1,   so   F = N m^r = N^{1-r}(N m)^r >= (n+1)^{1-r}

using only agent 2's Theorem A (`N >= n+1`). So "equal moduli at every stationary point" would prove
(P1) outright, for every `n` and every `r`. **It is false.** Solving the stationarity system
directly (`p1c_stat.py`: the square system (KL) + (i) + (ii) by `least_squares` from 250 random
starts, residual `< 1e-9`, all atoms `> 1e-3`) finds at `n = 2`, `N = 4`, `q = 2` exactly two
distinct full-support branches out of 197 convergent runs:

| branch | moduli | spread | `F` | `F/target` | `sum_l |Lambda(v_l)|^{-sigma}` |
|---|---|---|---|---|---|
| unequal | `(0.336766, 0.336766, 0.163634, 0.163634)` | `2.0582` | `1.566422` | `1.086096` | `3.843496` |
| tensor grid (`Lambda == 1`) | `(1/4, 1/4, 1/4, 1/4)` | `1` | `1.587401` | `1.100642` | `4.000000` |

Both clear (P1) comfortably, but the first kills the rigidity route: what has to be proved is the
weaker `sum |Lambda|^{-sigma} >= n+1`, and equal moduli is not the mechanism.

### 4.4 What the route does prove: `n = 1`, every `N`, in four lines

At `n = 1`, `Lambda(v) = 1 + lam v`. Condition (ii) says `1/Lambda(v_l)` is real, i.e.
`Lambda(v_l) in R`, i.e. `lam v_l in R`, so `v_l in {+u, -u}` with `u = conj(lam)/|lam|`: **at most
two distinct atom positions**, with `Lambda = 1 +- |lam|`. By (i) every `c_l` is real with the sign
of `1/Lambda(v_l)`, and atoms at the same position may be merged (which lowers `F`). So `N <= 2`
effectively, and `sum_l c_l = 1`, `sum_l c_l v_l = 0` force `c = (1/2, 1/2)` at antipodal positions,
`F = 2^{1-r}`. (`lam = 0` gives `Lambda == 1`, all `|c_l|` equal, and §4.3 applies.) This reproves
the `n = 1` case of (P1) for every `N` without the ladder — the method is sound; it is the `n >= 2`
case analysis that does not close.

At `n = 2` the same analysis reduces the dual to four real parameters (`lam_1, lam_2 >= 0` after the
slot-rotation gauge, and `mu := lam_{12} - lam_1 lam_2` complex), and each atom `(a, b)` must satisfy
`1 + lam_1 a` and `1 + lam_2 b` both real multiples of `Lambda(a,b)`. Two sub-cases close by hand:

* **`mu = 0`** (a product dual). Then `Lambda = (1 + lam_1 z_1)(1 + lam_2 z_2)` and (ii) forces
  `lam_1 a` and `lam_2 b` real, so the atoms sit on a `2 x 2` grid; (KL) is then the Hadamard
  transform and puts mass exactly `1/4` on each grid point, giving `F = 4^{1-r} > 3^{1-r}`. This is
  the tensor-grid branch of §4.3.
* **`lam_1 = lam_2 = 0`, `mu != 0`.** Then `Lambda = 1 + mu ab` and (ii) forces
  `mu a_l b_l = +-|mu|`, so `Lambda(v_l) in {1 + |mu|, 1 - |mu|}` and all `c_l` are real.
  `sum_l c_l a_l b_l = 0` splits the mass `1/2` / `1/2` between the two groups, and
  `F = 2^{-r}(n_+^{1-r} + n_-^{1-r})` with `n_+/n_- = ((1+|mu|)/(1-|mu|))^{1/(1-r)}`. The only
  assignments that would undercut `3^{1-r}` are `(n_+, n_-) = (1,1)` and `(2,1)`, and both are
  **infeasible**: `sum_l c_l a_l = 0` with positive `c_l` forces the small group's atoms to coincide
  with the antipode of the large one, which puts them in the same `+-` group — a contradiction.

The remaining sub-case (`lam_1 lam_2 != 0`, `mu != 0`) is a finite system — for each dual parameter
the admissible atom positions solve three real equations in `(theta, rho, tau)`, where
`1 + lam_1 a = rho e^{i theta}` and `1 + lam_2 b = tau e^{i theta}` — but its feasibility analysis is
not done. **This is where the stationarity route stops.**

---

## 5. Structure of the solution set

### 5.1 The Lagrange form (Proposition L)

Let `X` be the `N x 2^n` matrix `X_{l,T} = prod_{j in T} v_lj`, so (KL) is `X^T c = A e_empty`, and a
multi-affine `Lambda` with coefficient vector `lam` has `(X lam)_l = Lambda(v_l)`. At `N = 2^n` and
generic phases `X` is invertible, so `c = A X^{-T} e_empty`, i.e. `c_l = A (X^{-1})_{empty, l}`.
The Lagrange basis `Lambda^{(l)}` has coefficient vector `X^{-1} e_l`, whose constant coefficient —
which is `Lambda^{(l)}(0)` — is exactly `(X^{-1})_{empty, l}`. Hence `c_l = A Lambda^{(l)}(0)`. And
`sum_l Lambda^{(l)} == 1` because the constant function `1` interpolates `1` at all `2^n` points and
multi-affine interpolation there is unique; evaluating at `0` recovers `sum_l c_l = A`, the clean
constraint. Checked to `1.1e-11` (`p1c_verify.py` V4).

*What it buys.* At `n = 1` the basis is explicit, `Lambda^{(1)}(0) = a_2/(a_2 - a_1)`, so
`|c_1| = |c_2| = 1/|a_1 - a_2| >= 1/2` and `F >= 2 * 2^{-r}` — the `n = 1` Key Lemma in one line,
with the `2` coming from `|a_1 - a_2| <= 2`, i.e. from Hadamard at `N = 2`. At `n = 2` it is agent
4's six-angle determinant problem written invariantly; the equality manifold is "three points forming
the block, the fourth anywhere" (then `Lambda^{(4)}(0) = 0`).

### 5.2 Why mixtures cannot work (Refutation M)

My brief asked whether an `N`-atom solution is a mixture of `(n+1)`-atom solutions, so that
concavity plus the Key Lemma would finish. Two independent obstructions.

* **Wrong direction.** For `r <= 1`, `|z_1 + z_2|^r <= |z_1|^r + |z_2|^r`. So any decomposition
  `c = sum_m c^{(m)}` gives `F(c) <= sum_m F(c^{(m)})`. Mixtures produce **upper** bounds on the
  cost. Super-additivity — which a lower bound would need — is false for every `r < 1`, and false in
  precisely the way that matters: splitting one atom into several of the same total is what `r < 1`
  makes expensive, so the cost of a mixture lies *below* the sum of the costs.
* **No such mixtures exist.** For `n >= 2` and generic phases, no `(n+1)`-subset of the atoms
  supports a (KL) solution at all. At `n = 2` the 3-atom junk-free populations form the
  common-phase variety `b_l / a_l = const` (`LEMMA_ATTEMPT` §4), of positive codimension; three
  atoms drawn from a generic configuration do not lie on it. The smallest support available
  generically is `2^n` (there (KL) is square and `c` is unique, Proposition L) — and the Key Lemma
  is not known for `2^n > n+1`.

A decomposition into `2^n`-atom solutions *does* exist generically (the affine hull of the
`C(N, 2^n)` square solutions is the whole solution set), but by the first bullet it is useless.

### 5.3 The diagonal relaxation (Refutation D)

Putting `y_j = y` in `h(y) == A` and writing `p_l(y) := prod_j (y + conj(v_lj))`,
`d_l := c_l prod_j v_lj` (so `|d_l| = |c_l|`) gives

    (DIAG)     sum_l d_l p_l(y)  ==  A       for every y in C,

with every `p_l` **monic of degree `n` with all roots on the unit circle**. This keeps only the
`n+1` "diagonal" combinations of the `2^n` constraints, so it is a relaxation of (P1) to a classical
one-variable object — and at `n = 1` it *is* (P1). It is false for `n >= 2`. The counterexample uses
two atoms:

    p_1(y) = y^2 + 1  (roots  +- i),      p_2(y) = y^2 - 1  (roots  +- 1),
    d = (1/2, -1/2)   =>   d_1 p_1 + d_2 p_2 == 1,   cost = 2 * (1/2)^r = 2^{1-r}.

At `q = 2` that is `1.259921` against the target `3^{1-r} = 1.442250`, a ratio of `0.873580`.
Mapping back, `v_1 = (-i, i)`, `v_2 = (1, -1)`, `c = (1/2, 1/2)`, and indeed
`C_{\{1\}} = (1-i)/2 != 0`: the configuration satisfies the diagonal constraints and violates the
off-diagonal ones. An independent global search reproduces exactly `2^{1-r}` at `N = 3` and `N = 4`
(`p1c_diag.py`).

**Why this matters for the programme.** The Key Lemma's proof substitutes `u_j = v_{sigma(j), j}`
for a *bijection* `sigma` — it uses the slots asymmetrically. Refutation D says that is not an
artefact: any relaxation that forgets which phase sits in which slot loses the constant `n+1` and
gets `2` instead. **The `(n+1)` is genuinely an `n`-slot Fekete/Hadamard constant, not a
one-variable one.**

---

## 6. Numerical record (all float64, local CPU, no pods, `heracles/FINAL_RUNS/P1C/`)

| script | what it does | result |
|---|---|---|
| `p1c_core.py` | the shared library: (KL) algebra, the SLSQP minimiser with `eps`-annealing, the dual solve for `Lambda` from condition (i), and the stationarity diagnostics | — |
| `p1c_diag.py` | global search over the DIAGONAL relaxation (phases + null space, Nelder-Mead with `eps` annealed to `1e-16`), `n = 2`, `N = 3, 4`, 20 starts | minimum `1.259921 = 2^{1-r}` at **both** `N`, i.e. ratio `0.873581` to `3^{1-r}` — the relaxation is REFUTED, and the optimiser lands on the hand counterexample (two atoms at `1/2`) |
| `p1c_verify.py` V1 | the dictionary `(KL) <=> h == A <=> (dagger)`, `n = 1,2,3`, `N` up to 10, 150 systems each | max residual `2.97e-13` |
| `p1c_verify.py` V2 | **Lemma T**: the bijection inequality (T2) and the Key-Lemma-with-a-tail conclusion, on random (KL) systems, `n = 2` (`N = 5,6,7`), `n = 3` (`N = 9,10`) | **zero violations in 9900 bijection checks and 750 conclusion checks**; worst slack `-0.174` (not a near miss) |
| `p1c_verify.py` V3 | the assembled bound `F >= Val(R, s)` on the same systems | worst `Val - F = -0.055` (must be `<= 0`) |
| `p1c_verify.py` V4 | **Proposition L**: `c_l = Lambda^{(l)}(0)` and `sum_l Lambda^{(l)} == 1` at `N = 2^n`, `n = 1,2,3` | max error `1.08e-11` / `1.15e-13` |
| `p1c_verify.py` V5 | the diagonal counterexample in closed form | `d_1p_1 + d_2p_2 == 1` exactly; all four roots of modulus `1.0`; ratio `0.873580` (`q=2`), `0.934655` (`q=2.5`) |
| `p1c_census.py` | 34-60 random-start SLSQP minimisations of `F` subject to (KL) per cell; 7 cells: `n = 2` (`N = 4,5,6` at `q = 2`; `N = 5` at `q = 2.5`), `n = 3` (`N = 5,6,7` at `q = 2`) | min ratio `1.0000000` to `1.0000075` — **never below 1** in any of the 7 cells; the argmin is the `(n+1)`-block every time; `\|lam_empty\| = 1` to `1.9e-6`; `res_i <= 2.3e-15`, `res_ii <= 8.1e-6`; `sum_l \|Lambda(v_l)\|^{-sigma} = n+1` to `8e-8` |
| `p1c_stat.py` | the square stationarity system (KL) + (i) + (ii) solved directly by `least_squares`, 250 starts, `n = 2`, `N = 4`, `q = 2`, full-support solutions only | 197 converged, residual `< 1e-9`; exactly **2** distinct branches: `F/target = 1.086096` (moduli spread `2.0582`) and `1.100642` (the tensor grid). **Rigidity refuted**; both clear (P1) |
| `p1c_hybrid.py`, `p1c_hybrid2.py` | the hybrid programme, first with SLSQP and then with exact vertex enumeration of `Lad` | `n = 2, q = 2`: `Lhyb = 1.442249570` exactly `= target`, infimum attained only at `R = 0` |
| `p1c_margin.py` | the point-wise profile `m(R) = min_s Val(R,s)/target`, with the safety margin over `[R1, R2]` | `n = 2`: `1.0337` (`q = 2`), `1.0095` (`q = 2.25`), `1.0031` (`q = 2.5`) |
| `p1c_certify.py` | the **cell-wise guaranteed** bound (§3.3) | table of §3.4; the vectorised and naive implementations agree to `0.0` on 400 random cells |
| `p1c_rstar.py` | bisection on `r` using the certified bound | table of §3.5: certified `r*(2) >= 0.79688`, `r*(3) >= 0.61328`, `r*(4) >= 0.42344`; `n = 5` short already at `q = 2` |

---

## 7. What remains open, ranked

1. **Close the box analytically.** The certificate has `1.1–3.4%` of margin at `q = 2`, but
   `Val(R, s)` is an elementary function of two variables and the ladder vertices are explicit
   affine formulas, so a by-hand minimisation should be within reach. That would turn Theorem H
   into a closed theorem at `q = 2` for `n = 2, 3, 4`.
2. **Improve the `2^n` in Lemma T** (also `ADVERSARY` §6's suggestion (ii)). The factor comes from
   `|u_j - v_lj| <= 2` applied to each tail atom. For a *fixed* tail atom the bound is attained only
   when all `n` factors align, which cannot happen for every one of the `n!` bijections at once; and
   for two distinct core points `a, b` on the circle, `max_z |z-a||z-b| = 2(1 + |cos alpha|) < 4`
   where `2 alpha` is their angular separation. Turning that into a configuration-free gain would
   widen `R1`, raise `r*`, and might reach `n = 5`.
3. **`sum_l |Lambda(v_l)|^{-sigma} >= n+1`** (§4.2) is the sharpest reformulation of (P1) at
   stationary points, and §4.3 shows it is strictly weaker than equal moduli. With agent 4's
   Theorem O ruling out uniform averaged probes and agent 3's F3-2 capping sup-norm peeling, this
   and the `n = 2` dual case analysis of §4.4 are the two live structural handles.
4. **The `n = 2, N = 4` case in Lagrange form** (§5.1) remains the cleanest single open statement:
   for four points of `T^2`, the bilinear Lagrange basis evaluated at the origin satisfies
   `sum_l |Lambda^{(l)}(0)|^r >= 3^{1-r}` given `sum_l Lambda^{(l)}(0) = 1`.
5. **`n = 5` at `q = 2`** is the first cell the hybrid misses (`0.9872` at grid `400 x 400`). Since
   the cell bound is conservative, it is not known whether the programme itself fails there or only
   my certificate; `ADVERSARY`-style point sampling would settle which.
