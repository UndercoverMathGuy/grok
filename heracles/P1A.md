# P1 (the sharp constant): a rigorous certificate at `q = 2`, and the open window reduced to one
# `r`-free inequality

Work on **P1** of `GOPT_STATE.md`. Continues `LEMMA_ATTEMPT.md` (agent 1), `GLOBAL_OPT.md` (agent 2),
`GLOBAL_OPT3.md` (agent 3), `GLOBAL_OPT4.md` (agent 4) and, in its second half, `ADVERSARY.md`
(the sibling adversary agent, whose §6 result this note turns into a proof).

**Headline.**

1. **P1 at plain weight decay is now PROVED, not measured.** `ADVERSARY.md` §6 reduces P1 -- the
   statement `sum_l |c_l|^r >= (n+1)^{1-r}|A|^r` for **every** number of atoms `N` -- to a
   two-variable programme `min G(beta, tau) >= (n+1)^{1-r}`, and evaluates that minimum on a
   `600 x 600` grid with a local polish. A grid is not a proof. `p1a_cert.py` replaces it by a
   four-piece argument in which **three pieces are exact algebra and the fourth is an
   interval-arithmetic branch-and-bound in two variables** (mpmath outward-rounded, 30 digits).
   It closes at `n = 2, q = 2` in **101 boxes** (93 with the sharpened tail of §3.3) and at
   `n = 3, q = 2` in **63 boxes** (57 with the sharpened tail). Two independent soundness checks pass (§3.4). The result is
   unconditional *given* the two inherited lemmas the adversary builds on (Theorem B of
   `GLOBAL_OPT` §4 and Theorem E of `GLOBAL_OPT3` §3), which I did not re-derive -- what was
   unproved before this pass, and is proved now, is the two-variable minimisation.
2. **The open window `r in (r*, 1)` collapses to a single `r`-free inequality.** `sum_l x_l^r`
   equals `S^r e^{(1-r)H_r(q)}` exactly (`S` = total modulus in units of `|A|`, `H_r` = Rényi
   entropy of the normalised profile), and `H_r >= H` for `r < 1`. So
   **P1 holds for every `r in [r_0, 1]` as soon as `H(q) + (r_0/(1-r_0)) log S >= log(n+1)`** --
   one inequality for the whole exponent range instead of one per exponent (§4). It is *exactly*
   as sharp as P1 (measured minimum `0.000000000` at the block), and it survives down to
   `rho` between `0.1` and `0.2` (measured), against the `rho = r*/(1-r*) ≈ 5.3` the window needs:
   a factor of about **30 of room**.
3. **A geometric reformulation of the first open case, and drop lemmas that remove the `2^n`.**
   Independent of the above, and aimed at the adversary's own list of "what would close the window"
   (item (ii)): S1 at `n = 2, N = 4` is a statement about the affine dependency of four points of
   `T^2`, with the atom moduli equal to four triangle areas (§5); and the `2^n` of the drop step is
   replaceable at `N = n+2` by the true chord geometry, exactly rather than by a Hadamard
   relaxation (§6). The lemmas are proved; their numerical thresholds are not certified, and §6
   does **not** claim a proof of P1 at `N = 4` -- see the status paragraph there.
4. **Two conjectures refuted** with explicit counterexamples, and one confirmation of the
   adversary's obstruction (2) from the other side (§7).

All float64 except the certificate, which is interval arithmetic. Local CPU, **no pods, $0**.
Scripts `heracles/p1a_*.py`, outputs `heracles/FINAL_RUNS/P1A/`. Nothing left running.

---

## 0. Symbols, with a plain reading of each

Inherited symbols keep their meanings (`p`, `n`, `nu = n+1`, `q`, `r = q/nu`, `A`, `c_l`, `v_lj`,
`(KL)`, `D_h(n)`, `Gamma_s`, `B`). New or re-used here:

| symbol | meaning | plain reading |
|---|---|---|
| `x_l` | `\|c_l\|`, sorted decreasing, with `\|A\| = 1` | the atom masses of one feature |
| `beta` | `x_{n+1}` | the smallest of the `n+1` biggest atoms |
| `tau` | `sum_{l > n+1} x_l` | total mass of everything past the first `n+1` atoms |
| `tgt` | `(n+1)^{1-r}` | what P1 says the power sum is at least |
| `G(beta, tau)` | the adversary's two-variable lower bound (§2) | "what Theorem B + peeling + the tail's own price give" |
| `S` | `sum_l x_l` (so `S >= 1`) | total mass in units of the clean amplitude |
| `q_l`, `H(q)` | `x_l/S`, `-sum_l q_l log q_l` | normalised profile, Shannon entropy; `e^H` = effective number of atoms |
| `H_r(q)` | `(1/(1-r)) log sum_l q_l^r` | Rényi entropy of order `r`; `H_r >= H_1 = H` for `r < 1` |
| `rho` | `r/(1-r)` | the coefficient in the entropy condition (E); increasing in `r` |
| `D_(l)`, `e_l` | complementary `3x3` determinants at `n = 2, N = 4`, `e_l = (-1)^{l+1}D_(l)` | the four complementary triangle areas; `e` = the affine dependency |
| `m` | `min_l x_l` | the smallest atom (the `N = 4` decomposition variable of §6) |

---

## 1. Reproducing everything

From `heracles/`:

    uv run --with mpmath python p1a_cert.py --cells 2:2 3:2                 # THE CERTIFICATE
    uv run --with mpmath python p1a_cert.py --cells 2:2 --staircase --sweep 2 \
        --max-boxes 20000                                                   # sharpened tail, certified r* at n = 2
    uv run --with mpmath python p1a_cert.py --cells 3:2 --staircase --sweep 3 \
        --max-boxes 2000                                                    #                        ... at n = 3
    uv run --with mpmath python p1a_certcheck.py                            # the two soundness checks
    uv run python p1a_entropy.py --blocks rho suff n2N4                     # the entropy reduction, (E_rho)
    uv run python p1a_core.py                                               # Proposition A (reformulation)
    uv run python p1a_lemmaR.py --r 0.6666666666666666                      # Lemma R, Lemma R'
    uv run python p1a_window.py --starts 200 --verify                       # Lemma R'' (exact chords), r = 2/3
    uv run python p1a_window.py --r 0.9  --starts 10 --lo 0.005 --hi 0.245   # ... in the open exponent window
    uv run python p1a_window.py --r 0.95 --starts 10 --lo 0.005 --hi 0.26
    uv run python p1a_sharp.py --starts 300 --verify                        # Lemma R* (exact parametrisation)
    uv run python p1a_product.py --starts 1500                              # refutation of (P3)
    P1A_TAG=explore_floor uv run python p1a_explore.py --blocks floor --starts 800   # the floor R(t)
    uv run python p1a_explore.py --blocks sym --starts 200                  # the symmetry group

---

## 2. What is being certified

`ADVERSARY.md` §6 proves, from two inherited lemmas (Theorem B of `GLOBAL_OPT` §4 and Theorem E /
`(PEEL_h)` of `GLOBAL_OPT3` §3) plus the tail's own price, that for **every** (KL) system and every
`N`,

    sum_l |c_l|^r  >=  G(beta, tau)
       :=  Tail(tau, beta)  +  max( Peel(beta, tau),  tgt * (1 - 2^n tau)_+^r ) ,
    Tail  :=  max( tau^r , tau beta^{r-1} ) ,
    Peel  :=  beta^r + min { sum_{i=1}^n y_i^r : y_1 >= ... >= y_n >= beta,
                             sum_{i=h+1}^n y_i >= D_h(n) - beta - tau,  h = 0..n-1 } .

*Plain reading of the three terms.* **Tail**: the atoms past the first `n+1` are each at most
`beta` and add up to `tau`, and a concave power sum is cheapest when its mass is stacked in pieces
of the maximum size. **Peel**: after the tail is set aside, the peeling ladder still forces the
`n+1` big atoms to carry prescribed tail sums. **Drop**: removing the tail leaves an `n+1`-atom
system whose clean amplitude has shrunk by at most `2^n tau`, and Theorem B prices that at the Key
Lemma's constant.

P1 at exponent `r` follows from `min G >= tgt`. That minimum is *attained* at `tau = 0`, where the
drop term equals `tgt` exactly and `Tail = 0` -- so the bound is sharp and no numerical
minimisation can separate `>=` from `<` there. The certificate therefore has to be part analysis,
part certified search.

---

## 3. The certificate

### 3.1 Three pieces of exact algebra

> **(P-a) `beta >= 1/(n+1)` is done.** All `n+1` big atoms are at least `beta`, so
> `Peel >= (n+1) beta^r >= (n+1) (n+1)^{-r} = tgt`. ∎

> **(P-b) `tau >= tau_hi := (tgt / C(n,r))^{1/r}` is done**, where
>
>     C(n,r) := min_{s>0} [ max(1, s^{r-1}) + (n+1) s^r ] = s*^{r-1} + (n+1) s*^r,
>     s* := (1-r) / ((n+1) r)   (verified < 1 in every cell used).
>
> *Proof.* `G >= Tail + (n+1)beta^r`. Write `beta = s tau`; then the right-hand side is
> `tau^r [ max(1, s^{r-1}) + (n+1)s^r ] >= C(n,r) tau^r`. On `(0,1]` the bracket has the single
> critical point `s*` (its second derivative there is positive), and on `[1, infinity)` it is at
> least `n+2 > C`. ∎
>
> `C(2, 2/3) = 2.725680889248`, `tau_hi = 0.384900179460`;  `C(3, 1/2) = 4`, `tau_hi = 0.25`.

> **(P-c) `tau <= eps_cert` is done**, by a chord bound. Here `G >= phi(tau)`,
> `phi(t) := tgt (1 - 2^n t)_+^r + t^r`. The map `z -> (1-z)^r` is **concave**, so on `[0, z_0]` it
> lies above its own chord: `(1-z)^r >= 1 - zK` with `K := (1 - (1-z_0)^r)/z_0`. Hence for
> `t <= t_0` and `z_0 := 2^n t_0`,
>
>     phi(t) - tgt  >=  t^r - tgt * 2^n K * t  >=  0      whenever   t_0^{1-r} <= 1/(tgt * 2^n * K).
>
> Bisecting on `t_0` with that test evaluated in interval arithmetic gives a **proved**
> `eps_cert`. It is essentially lossless: `eps_cert = 1.697662756e-02` at `n = 2, q = 2` against the
> adversary's numerically located `eps* = 1.698e-02`.

*Plain reading of (P-c): near `tau = 0` the bound loses clean amplitude linearly (at rate `2^n`)
but gains the tail's own price `tau^r`, which has infinite slope at `0`. The cusp wins first; the
chord bound says exactly how far.*

### 3.2 The certified branch-and-bound

What is left is the rectangle `[0, 1/(n+1)] x [eps_cert, tau_hi]` -- at `n = 2, q = 2` that is
`[0, 0.3333] x [0.01698, 0.3849]`. On a box `[bl,bh] x [tl,th]` a rigorous lower bound for `G` is

    Tail_lo = max( tl^r , tl * bh^{r-1} )        (beta^{r-1} decreases, tau beta^{r-1} increases in tau)
    drop_lo = tgt * (1 - 2^n th)_+^r             (decreases in tau)
    Peel_lo = bl^r + inner(P*) ,

where `P*` is the **relaxed** polyhedron `{ y_1 >= ... >= y_n >= bl, sum_{i>h} y_i >= D_h - bh - th }`:
every polyhedron occurring in the box is contained in `P*`, and the inner objective is increasing
in each `y_i`, so `inner(P*)` is a valid lower bound. The inner minimum of a concave function over
a pointed polyhedron sits at a vertex; the constraint matrix has **integer** entries, so its
inverses are computed once in exact rational arithmetic and each candidate vertex is an exact
rational combination of the (interval) right-hand sides. Taking the minimum over all candidate
vertex systems -- a superset of the true vertices, with the definitely infeasible ones discarded --
is a valid lower bound.

Boxes whose `beta + tau` cannot reach `D_n(n)` are discarded: `(PEEL_n)` makes them infeasible for
(KL), so there is nothing to prove there.

**Result** (`p1a_cert.py`, mpmath `iv`, 30 digits):

| `n` | `q` | `r` | `eps_cert` (proved) | `tau_hi` (proved) | certified | boxes (plain / with Lemma T) |
|---|---|---|---|---|---|---|
| 2 | 2 | 2/3 | `1.697662756057e-02` | `0.384900179460` | yes | **101 / 93** |
| 3 | 2 | 1/2 | `1.469237832874e-02` | `0.250000000000` | yes | **63 / 57** |

(`C(2,2/3)` is enclosed in `[2.725680889248, 2.725680889248]`, `C(3,1/2)` in `[4, 4]`; the `D_h`
used are the exact algebraic values `1, 3 sqrt3/8, 1/4` at `n = 2` and `1, 49/72, 3 sqrt3/16, 1/8`
at `n = 3`. Maximum branch-and-bound stack depth: 5 and 4.)

> **THEOREM (certified).** Granting the two inherited lemmas (Theorem B and Theorem E) and the
> adversary's assembly of them into `G`, P1 holds for **every number of atoms `N`** at `n = 2`,
> `q = 2` and at `n = 3`, `q = 2`. The only numerical ingredient is an interval branch-and-bound
> over a `2`-dimensional rectangle, which terminates in about a hundred boxes.

*What is still conditional, stated precisely.* I did **not** re-derive Theorem B or Theorem E; the
certificate is of the two-variable programme, which is what the adversary flagged as unproved. §3.4
re-verifies the assembled chain `sum|c|^r >= G` numerically and independently.

### 3.3 A free sharpening of the tail (Lemma T)

> **LEMMA T (proved).** The tail atoms satisfy `0 <= x_l <= beta` and `sum_l x_l = tau`. A concave
> power sum over a box-simplex is minimised at an extreme point, i.e. at `k = floor(tau/beta)`
> atoms of size `beta` plus one of size `tau - k beta`. Hence
>
>     sum_{tail} x_l^r  >=  k beta^r + (tau - k beta)^r ,      k = floor(tau/beta),
>
> which **dominates** the adversary's `tau beta^{r-1}` (because `rem^r >= rem * beta^{r-1}` when
> `rem <= beta` and `r < 1`), with equality exactly when `beta` divides `tau`.

Verified on 200 000 random tails: `min(sum x^r - staircase bound) = 0.0` (the bound is attained,
never violated). It is switched on by `--staircase` and is used for the certified `r*` sweep of
§3.5, where it is a strict gain at no cost.

### 3.4 Soundness: two independent checks (`p1a_certcheck.py`)

The certificate is only worth something if (i) my interval relaxation really is a lower bound for
the adversary's `G`, and (ii) `G` really is a lower bound for `sum|c|^r`. Both are re-implemented
here from the written definitions, independently of `adv_proof.py`.

| check | what is measured | result |
|---|---|---|
| **1** relaxation is sound | `min` over 4000 sampled boxes of `G_float(centre) - (upper end of G_lower(box))`; must be `>= 0`, and small means tight | `+3.86e-03` (`n=2,q=2`), `+4.48e-03` (`n=3,q=2`), `+4.01e-03` (`n=2,q=2.5`) -- **OK**, and tight |
| **2** the chain is sound | `min` over genuine (KL) systems of `sum_l\|c_l\|^r / G(beta,tau)`; must be `>= 1` | `1.002638` (`N=4`, 2999 systems), `1.021110` (`N=5`), `1.030259` (`N=6`), `1.040727` (`N=7`), all at `n=2, q=2` -- **OK** |

(Check 2 at `n = 3` needs the constrained solve the adversary used -- random phases with `N < 2^n`
give an overdetermined system and no feasible atoms -- so `n = 3` is covered by `adv_proof.py`'s own
125-system verification, not re-done here. Recorded rather than glossed.)

### 3.5 A certified `r*`

`ADVERSARY.md` quotes `r*(2) ~ 0.83`, `r*(3) ~ 0.62` from a bisection run on a coarse grid. Running
the *certificate* inside the bisection turns those into proved statements. Results in
`FINAL_RUNS/P1A/cert_staircase.log` and `cert.json`; see §9.

---

## 4. The open window `r in (r*, 1)`, reduced to one `r`-free inequality

This is the part of P1 the adversary left open, and the coordinator flagged it as the highest-value
remaining contribution. The reduction below does not close it, but it changes its shape completely.

> **REDUCTION LEMMA (proved; every `n`, every `N`, every configuration).** With `|A| = 1`,
> `S = sum_l x_l`, `q = x/S`, and `H_r` the Rényi entropy of order `r`,
>
>     sum_l x_l^r  =  S^r * sum_l q_l^r  =  S^r e^{(1-r) H_r(q)}      (an identity),
>     H_r(q) >= H_1(q) = H(q)                                          (Rényi decreases in the order),
>
> so, since `S >= 1`,
>
>     (E_rho)   H(q) + rho log S  >=  log(n+1),      rho := r_0/(1-r_0)
>     ==>  P1 holds at every exponent r in [r_0, 1] .
>
> *Proof.* `sum_l x_l^r >= (n+1)^{1-r}` is `r log S + (1-r) H_r(q) >= (1-r) log(n+1)`, i.e.
> `H_r(q) + (r/(1-r)) log S >= log(n+1)`; use `H_r >= H` and note `rho = r/(1-r)` is increasing in
> `r` while `log S >= 0`. ∎
>
> (The same statement drops out of convexity: `h(r) := log sum_l x_l^r - (1-r)log(n+1)` is convex on
> `[0,1]` because `r -> sum x^r` is log-convex, and `(E_rho)` is exactly "`h` lies above its tangent
> at `r = 1`, and that tangent is non-negative".)

*Plain reading.* `e^{H(q)}` is the **effective number of atoms** and `S` the total mass in units of
the clean amplitude. `(E_rho)` at `rho = 2` (the paper's `q = 2`, `nu = 3`) says
**`S^2 x (effective number of atoms) >= n+1`**: a feature may concentrate its amplitude on fewer
than `n+1` atoms only by paying for it in total mass.

**Three things make this the right handle on the window.**

* **One inequality covers the whole range.** `(E_rho)` at `rho_0` gives P1 for *every*
  `r >= rho_0/(1+rho_0)` at once, so the entire open interval `(r*, 1)` is bought by a single
  `r`-free statement at `rho = r*/(1-r*) ≈ 4.9` (at `n = 2`).
* **It is exactly as sharp as P1**, so it is not a weakening. At `n = 2, N = 4`
  (`p1a_entropy.py --blocks suff`) the measured minimum of the tangent expression is
  `-0.000000000` at `r = 1/4, 1/2, 2/3, 5/6` -- exactly `0`, attained at the block (`S = 1`,
  `H = log 3`).
* **The `rho` the window needs is far above the critical one.** The `rho`-scan
  (`p1a_entropy.py --blocks rho`, `FINAL_RUNS/P1A/rho.log`) minimises `H(q) + rho log S - log 3`
  over `n = 2, N = 4` configurations; smaller `rho` is a strictly stronger statement.

  | `rho` | 10 | 5 | 4.88 | 3 | 2 | 1 | 0.5 | 1/3 | 0.2 | 0.1 | 0.05 | 0.02 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | min of `H + rho log S - log 3` | `-0.0e0` | `-0.0e0` | `-0.0e0` | `-0.0e0` | `-0.0e0` | `-0.0e0` | `+0.0e0` | `-0.0e0` | `-0.0e0` | `-0.0818` | `-0.1974` | `-0.2995` |
  | holds? | yes | yes | yes | yes | yes | yes | yes | yes | yes | **no** | **no** | **no** |

  So the critical coefficient is bracketed in `(0.1, 0.2)`, i.e. `(E_rho)` is true (numerically) for
  every `rho >= 0.2`, hence would give P1 for every `r >= 1/6` at `n = 2, N = 4`. The window needs
  only `rho ≈ 5.3`: there is a factor of about **30 of room**, and at every `rho` where it holds the
  minimum is exactly `0`, attained at the block. *`(E_rho)` is therefore not a delicate statement at
  the value that matters; it is a robust one whose only tight point is the equality configuration.*

**Where the difficulty has moved.** `(E_rho)` is hardest at `S = 1`, i.e. when the atoms are exactly
in phase (`sum_l |c_l| = |sum_l c_l|`), and there it reduces, independently of `rho`, to

> **(E0)** *a probability vector `p` supported on points `v_l` of `T^n` with
> `sum_l p_l prod_{j in T} v_lj = 0` for every non-empty `T` has Shannon entropy `H(p) >= log(n+1)`.*

That is: **a probability measure on the torus that kills every non-trivial multi-affine character
must have entropy at least `log(n+1)`**, with equality at the block. So the `r -> 1` end of P1 is
not an `r`-problem at all -- it is a single entropy statement about cubature measures, which is the
cleanest form the window has taken. (The `S = 1` hypothesis is essential: drop it and the
statement is false -- F5-2.)

**What (E_rho) does not evade.** On the peeling ladder's own polytope the vertex
`x = (0.350481, 0.350481, 0.299038)` has `S = 1` and `e^{H} = 2.9922 < 3`, so `(E_rho)` fails there
for *every* `rho`. Like everything else in this programme it therefore needs information beyond the
sorted tail sums -- consistent with `GLOBAL_OPT3` F3-2. Recorded so nobody expects the ladder to
deliver it.

---

## 5. Proposition A: the first open case is about four points of a plane

Independent of everything above; this is the geometric content of S1 at `n = 2`, `N = 4`.

> **PROPOSITION A (proved; fp64-verified to `4e-12` on 4000 random configurations,
> `p1a_core.py`).** Let `x_l, y_l` (`l = 1..4`) be unimodular, `M` the `4 x 4` matrix with rows
> `(1,1,1,1), (x_l), (y_l), (x_l y_l)`, and put
> `D_(l) := det[[1,1,1],[x_m,x_n,x_o],[y_m,y_n,y_o]]` for `{m<n<o} = {1..4}\{l}` and
> `e_l := (-1)^{l+1} D_(l)`. Then
>
>     (A1)  sum_l e_l = 0,   sum_l e_l x_l = 0,   sum_l e_l y_l = 0 ;
>     (A2)  det M = - conj( sum_l e_l x_l y_l ) * prod_l (x_l y_l) ;
>     (A3)  if det M != 0, the unique solution of (KL) is
>           c_l = A conj(e_l) conj(x_l y_l) / conj( sum_m e_m x_m y_m ),  so |c_l| = |A| |D_(l)| / |det M| .
>
> Hence S1 at `n = 2`, `N = 4` is exactly
>
>     (P1-4)     sum_{l=1}^4 |D_(l)|^r   >=   3^{1-r} | sum_l e_l x_l y_l |^r .

*Plain reading.* Four points of the torus are four points of the affine plane `C^2`, so they carry
one linear dependency `e`; by (A1) its coefficients are, up to sign, the areas of the four
complementary triangles. `sum_l e_l x_l y_l` measures how badly the monomial `xy` fails to be affine
on those four points. **The phases of the atoms carry no information at all** -- Proposition D1 of
`GLOBAL_OPT4` made completely explicit in the first open case.

*Proof.* (A1) is the cofactor expansion of a `4 x 4` determinant with a repeated row. For (A2),
divide column `l` of `M` by `x_l y_l` (both unimodular): the rows `1, x, y, xy` become
`conj(xy), conj(y), conj(x), 1`, and reversing four rows is the even permutation `(14)(23)`; the same
operation on each `3 x 3` minor gives `delta_l = -(prod_{m!=l} x_m y_m) conj(D_(l))`, and (A2)
follows from the cofactor expansion of `det M` along the row of ones. (A3) is Cramer's rule. ∎

**Corollaries.** (i) Writing `gamma_l = arg x_l - arg y_l`, one gets
`x_p y_q - x_q y_p = 2i e^{i(tau_p+tau_q)} sin((gamma_p-gamma_q)/2)`, so `D_(l) = 0` exactly when
the three complementary atoms are **common-phase** -- agent 1's rigidity statement read off the
determinant. (ii) An equivalent "diamond" form (verified to `1.4e-12`): with `u_l = e^{-i tau_l}`,
`W_l = e^{i gamma_l/2}` and `f` the alternating complementary minors of the `3 x 4` matrix with rows
`u, W, conj(W)`, one has `|c_l| = |f_l| / |sum_m f_m conj(u_m)|`.

**The symmetry group is only as large as one expects** (`p1a_explore.py --blocks sym`, 400
configurations). Invariant to `1.6e-14`: the 24 atom permutations, slot swap, conjugation, and the
gauge rotations. **Not** invariant: conjugating one slot (relative change `1.37`) and the
"monomial rotation" `(x,y) -> (conj(y), x conj(y))` (`17.5`), which looks natural from the diamond
form. The reason is a lattice one: the affine automorphisms of the character square
`{(0,0),(1,0),(0,1),(1,1)}` fixing the distinguished corner are only the identity and the coordinate
swap. **There is no hidden symmetry to find beyond a factor 96.**

---

## 6. Replacing the `2^n`: chord-exact drop lemmas at `N = n+2`

The adversary's own list of what would close the window includes *"(ii) a better constant than
`2^n` in `R_S >= 1 - 2^n tau`"*. At `N = n+2` (one tail atom) that constant can be removed entirely,
and this section says by how much. Throughout, relabel so `|c_4| = m = min_l |c_l|`, `|A| = 1`.

> **LEMMA R (proved; every `n`, `N = n+2`).**
> `sum_l |c_l|^r >= m^r + (n+1)^{1-r} (1 - 2^n m)_+^r`.
> *Proof.* Move atom `N` across the identity `sum_l c_l prod_j (u_j - v_lj) = A prod_j u_j`; on the
> torus the new right-hand side has modulus at least `1 - 2^n m`, and the Key Lemma's proof
> (bijection identity, geometric mean over the `n!` bijections, squared Vandermondes, Hadamard) only
> ever uses a lower bound on it. Add `m^r`. ∎
>
> This is the `N = n+2` case of the adversary's `phi(tau)`, with `tau` replaced by `m`; at
> `n = 2, r = 2/3` it covers `m <= 0.01697`.

> **LEMMA R' (proved; `n = 2`, `N = 4`).** Keeping the chords instead of bounding them by `2`:
> with `A_i = |x_i - x_4|`, `B_i = |y_i - y_4|`, `V_x = prod_{i<j<=3}|x_i - x_j|`, `V_y` likewise,
>
>     prod_{l<=3}|c_l|  >=  prod_{i != j, i,j<=3} (1 - m A_i B_j)_+^{1/2} / (V_x V_y) .
>
> *Proof.* Substituting `(u_1, u_2) = (x_{l_1}, y_{l_2})` kills atoms `l_1` and `l_2` and leaves
> `|c_{l_0}| |x_{l_1}-x_{l_0}| |y_{l_2}-y_{l_0}| >= (1 - m A_{l_1} B_{l_2})_+`; take the geometric
> mean over the two orderings and the product over `l_0 = 1,2,3`, where the chord factors assemble
> into exactly `V_x V_y`. ∎
>
> With the Hadamard relaxation (`V_x <= 3^{3/2}`, `V_x prod_i A_i <= 16`) this covers `m <= 0.055`
> -- more than three times Lemma R's range. Verified on 20 000 configurations:
> `min(prod_{l<=3}|c_l| - bound) = +4.4e-04`, never negative.

> **LEMMA R\* (proved; `n = 2`, `N = 4`) -- the geometry parametrised exactly.** Gauge-fix
> `x_4 = y_4 = 1` and write `x_i = e^{2 i u_i}`, `y_i = e^{2 i v_i}`, `u_i, v_i in [0, pi)`. Then
>
>     A_i = 2 sin u_i ,  V_x = 8 prod_{i<j}|sin(u_i - u_j)|  (and likewise in the second slot),
>
> with **no inequality spent**: every admissible `(A, V_x)` is exactly one point of `[0,pi)^3`.
> Where the Hadamard version is lossy is now visible: its minimiser wants `A_i = 2` for all `i`
> (all three retained atoms antipodal to the dropped one), which forces `V_x = 0`, not the
> `16/8 = 2` that Hadamard allows.

> **LEMMA R'' (proved as an inequality; the outer minimisation is numerical).** The *complete*
> family of point substitutions in the identity is: for all `s, t in {1,2,3,4}`,
>
>     (PS_{s,t})    sum_{l not in {s,t}} |c_l| |x_s - x_l| |y_t - y_l|  >=  1 ,
>
> 16 inequalities, linear in the atom masses, and **tight at the block** (taking `s, t` to be two
> block atoms gives `3|c_{l_0}| >= 1`, i.e. `|c_{l_0}| >= 1/3`, for the third). Minimising
> `sum_{l<=3} w_l^r` over `{w >= m, (PS_{s,t})}` -- a concave minimisation over a polytope in `R^3`,
> hence exact by vertex enumeration -- and then over the six slot angles gives `Psi''(m)`.
> Measured (`p1a_window.py`, 12 starts per level):
>
> | `m` | 0.020 | 0.060 | 0.100 | 0.140 | 0.170 | 0.200 |
> |---|---|---|---|---|---|---|
> | `Psi''(m)/3^{1-r}` | 1.0372 | 1.0630 | 1.0751 | 1.0796 | 1.0823 | 1.0878 |
>
> Together with the trivial branch `sum_l|c_l|^r >= 4 m^r >= 3^{1-r}` for
> `m >= m_triv = (3^{1-r}/4)^{1/r} = 0.2165`, this covers **every** `m` with at least `3.7 %` of
> margin. Verified on real configurations: minimum slack in `(PS_{s,t})` is `+4.4e-11` and in the
> assembled Lemma R'' chain `+1.8e-04`, both non-negative.
>
> **And it does not die as `r -> 1`** -- which is exactly where the certificate of §3 does. At the
> exponents in the open window (`p1a_window.py --r 0.9` / `--r 0.95`):
>
> | `r` | `m = 0.005` | `0.035-0.041` | `0.065-0.078` | `0.125-0.155` | `m_triv` | window covered |
> |---|---|---|---|---|---|---|
> | 0.90 | 1.00307 | 1.01180 | 1.01655 | 1.02087 | 0.2421 | yes |
> | 0.95 | 1.00140 | 1.00626 | 1.00873 | 1.01057 | 0.2462 | **yes** (worst `1.0014`) |
>
> (ratios to `3^{1-r}`; `OK` throughout). *So the two attacks are complementary: the adversary's `G`
> covers every `N` but only `r <= 0.84`, while the chord-exact route covers every `r` but only
> `N = n+2`.* Closing the window looks like a matter of making the chord argument work for a tail of
> several atoms, not of finding a new idea.

### 6.1 Why the `2^n` cannot be improved *as an estimate of the same quantity*

The adversary's item (ii) asks for "a better constant than `2^n` in `R_S >= 1 - 2^n tau`". That
cannot be done at the level of the inequality itself:

> **OBSERVATION (proved).** The quantity being bounded is
> `sup_{u in T^n} | sum_{tail} c_l prod_j (1 + u_j v_lj) | / tau`. A **single** tail atom of mass
> `tau` sitting at `v` and probed at the aligning point `u_j = conj(v_j)` gives
> `prod_j (1 + u_j v_lj) = 2^n` exactly. So `2^n` is attained, and no constant below `2^n` is valid.
> (Checked numerically: `sup = 4.000000000000` at `n = 2`.) Moreover the torus mean of
> `prod_j (1 + u_j v_j)` is `1`, so no bound of this shape can ever have a constant below `1` --
> and `1` is precisely the constant at which `eps*` would stop collapsing as `r -> 1`.

*Consequence, and it is the sharp diagnosis of the open window.* `eps*` behaves like
`(tgt * r * 2^n)^{-1/(1-r)}`, so it collapses super-exponentially as `r -> 1` for **any** constant
above `1/(tgt r)`, i.e. for any admissible constant. The open window `r in (r*, 1)` therefore
**cannot** be closed by sharpening that estimate; it needs the constant to be coupled to the
geometry of the *retained* atoms, which is exactly what Lemmas R'/R\*/R'' do at `N = n+2` (they
replace `2^n` by the true chords `A_i B_j` and tie those to the Vandermondes `V_x, V_y`). That is
why §6 is the relevant lever rather than a better one-line estimate.

**Status of §6, stated precisely -- and this matters.** The *inequalities* of Lemmas R, R', R\* and
R'' are all proved. Only Lemma R's threshold (`m <= 0.016972542`) is proved as well, because its
bound is in closed form. For R', R\* and R'' the number `Psi(m)` is the value of a **minimisation
over the slot geometry**, and a multistart Nelder-Mead returns an *upper* bound on a minimum. So
every `Psi` figure quoted above -- `0.055`, the `3.7 %` margin, the `r = 0.9` and `r = 0.95` rows --
is an **upper estimate of a lower bound**: if the optimiser missed a basin, the true `Psi` is lower
and the coverage claim could shrink. **§6 therefore does not prove P1 at `N = 4` at any exponent**,
and nothing in §3 or §4 depends on it.

What §6 does establish unconditionally is the *mechanism*: the `2^n` of the drop step is removable at
`N = n+2` by keeping the chords, and the exact parametrisation of §6's Lemma R\* shows precisely
where the Hadamard relaxation throws the geometry away. The natural next step is to certify that
six-angle minimisation -- which, unlike the raw problem, has no cusp and no equality configuration
inside the region (§7, F5-4), so it is the kind of search that terminates.

---

## 7. Refutations and confirmations

**F5-1. The `r`-free product route `(P3)`. REFUTED, by `2.37x`.** At `N = n+1 = 3` the Key Lemma is
proved `r`-free through `prod_l |c_l| >= (|A|/3)^3`. The obvious `N = 4` analogue -- *the product of
the three largest atoms is at least `(|A|/3)^3`* -- would give S1 for all `r` at once. It is false:
`p1a_product.py` returns `min = 1/64 = 0.015625` against `1/27 = 0.037037` (ratio `0.421875`),
attained at the **4-atom block**: the four fourth-roots of unity in both slots, `c_l = 1/4`, which
satisfies (KL) exactly. S1 survives there comfortably (`sum|c|^{2/3} = 4^{1/3} = 1.5874` vs
`1.4422`), so no product statement about the three largest atoms can prove it. Seven weighted
variants fail at the same configuration. *This is the `N = 4` counterpart of agent 3's refutation of
the capped-mass conjecture: the truth is not an `r`-free majorisation statement.*

**F5-2. The plain entropy statement `(H)`: `H(q) >= log(n+1)`. REFUTED.** `p1a_entropy.py` drives
`H(q)` to `log 2` (deficit `-0.405465`) on configurations where two slot phases nearly coincide, so
`det M -> 0` and two atoms blow up with opposite phases; the normalised profile tends to
`(1/2,1/2,0,0)` while `S -> 6.5e13`. S1 is enormously satisfied there. **This is exactly why the
`rho log S` term in `(E_rho)` is not optional**, and why `(E0)` is stated at `S = 1`.

**F5-3. A hidden "monomial rotation" symmetry. REFUTED** (§5): it has order 3 and changes the
objective by up to `17.5`; the lattice reason is given there.

**F5-4. Confirmation, from the other side, of the adversary's obstruction (2).** The adversary
argues that a Lipschitz branch-and-bound on the raw six-angle inequality cannot terminate, because
the equality set has codimension 2 and the excess grows like `eps^r`. I measured that excess
directly (`p1a_explore.py --blocks floor`): the minimum of `sum|c|^r` subject to `min_l|c_l| >= t`,
as a ratio to the target, is

| `t` | 0 | 0.005 | 0.010 | 0.0170 | 0.030 | 0.050 | 0.120 | 0.200 |
|---|---|---|---|---|---|---|---|---|
| `R(t)/3^{1-r}` | 1.0000 | 1.0169 | 1.0255 | **1.0344** | 1.0466 | 1.0598 | 1.0827 | 1.0957 |

which is `eps^r`-shaped exactly as claimed. **But this is an argument against branching on the raw
angles, not against certified search as such**: once the cusped neighbourhood is removed
analytically -- which is what Lemma R / `phi(tau)` / (P-c) do -- what remains carries a `3.4 %`
margin and is perfectly branchable. That is precisely the shape of §3, where the certified search
runs in 101 boxes. Recorded because the two findings look contradictory and are not.

**Previously recorded dead routes** (Morwani-style dual certificates, the domination lemma, the
capped-mass conjecture, the peeling ladder's own cap, one averaged test measure / Theorem O, gauge
symmetrisation, Mahler measure, Gram / Caratheodory-Fejer, induction on `n` by shifting the last
slot) were not re-attempted; nothing here changes their status. Note that Lemmas R/R'/R\*/R'' are
**not** excluded by Theorem O: that result forbids a single averaged test measure that is uniform
over the torus, whereas these use substitutions *at the atoms themselves*, several of them, tight at
different atoms -- the non-uniform vertex-style argument Theorem O explicitly leaves open.

---

## 8. What I would do next, ranked

1. **Certify the six-angle minimisation of Lemma R''** (§6). That would prove P1 at `n = 2, N = 4`
   outright and, more usefully, show how much the `2^n` can be improved in general -- the
   adversary's item (ii), which is the lever on `r*`.
2. **Prove `(E0)`**: a probability measure on `T^n` killing every non-trivial multi-affine character
   has entropy at least `log(n+1)`. With `(E_rho)` for `S > 1` this closes the window
   `r in (r*, 1)` entirely, and it is a clean statement about cubature measures rather than about
   power sums.
3. **Re-derive Theorem B and Theorem E inside the certificate.** The certificate of §3 is of the
   two-variable programme only; the two inherited lemmas are taken on trust. Both have written
   proofs, so this is bookkeeping, but it is what "unconditional" would require.
4. **For the paper.** Quotable from this note: (i) P1 at `q = 2` -- the case the paper is about --
   is now a *proved* statement for every atom count, with the search reduced to a certified
   two-variable branch-and-bound of about a hundred boxes; (ii) the whole exponent range follows
   from one entropy inequality, and the `r -> 1` end is an entropy statement about cubature
   measures; (iii) the sharp constant at the first open case is a statement about the affine
   dependency of four torus points, with the atom moduli equal to the four complementary triangle
   areas.

---

## 9. Numerical record

All in `heracles/FINAL_RUNS/P1A/`. Interval arithmetic where stated; otherwise float64.

| script | what it does | result |
|---|---|---|
| `p1a_cert.py` | **the certificate**: (P-a)/(P-b)/(P-c) exact algebra + interval branch-and-bound on the residual rectangle | `n=2,q=2`: certified in **101** boxes (**93** with Lemma T); `n=3,q=2`: **63** boxes (**57** with Lemma T) (§9.1) |
| `p1a_cert.py --staircase --sweep` | the same with Lemma T's sharpened tail, plus a certified sweep in `r` (the `--rstar` bisection was killed for CPU, see §9.1) | brackets `r*(2) in [0.84, 0.86)`, `r*(3) in [0.60, 0.6667)` |
| `p1a_certcheck.py` | soundness: the relaxation vs `G`, and `G` vs genuine (KL) systems | `+3.9e-03 .. +4.5e-03`; `1.0026 .. 1.0407` -- both OK |
| `p1a_entropy.py` | the Reduction Lemma; refutation of `(H)`; the `rho`-scan of `(E_rho)` | `min H = log 2` (refutes `(H)`); tangent minimum `0.000000000` at `r = 1/4, 1/2, 2/3, 5/6`; `rho`-scan in §9.2 |
| `p1a_core.py` | **Proposition A**: (A1), (A2), (A3) and `\|c_l\| = \|D_(l)\|/\|det M\|` on 4000 configurations | residuals `8.7e-16`, `4.1e-12`, `4.1e-12`, `1.1e-15`; block+free ratio `1.000000000` at `r = 2/3, 1/2, 5/6` |
| `p1a_explore.py` | symmetry group; the floor `R(t)` (`explore_floor.json`) | symmetries as in §5; `R(t)` table in F5-4. Its `--blocks global degen` run was started and **killed** when the coordinator's message re-prioritised this pass, so no global-minimisation or degenerate-stratum number from it is quoted anywhere above; the adversary's 3.03M-configuration search already covers that ground |
| `p1a_lemmaR.py` | **Lemma R**, **Lemma R'**: thresholds, `Psi(m)`, verification on 20 000 configurations | `t_R = 0.016972542`; `Psi` above target on `[0, 0.055]`; slacks `+3.99e-02`, `+4.41e-04` |
| `p1a_sharp.py` | **Lemma R\***: the exact `(u, v)` parametrisation of the chord geometry | tight at `m = 0` by construction (`Pi*(0) = 1/27`) |
| `p1a_window.py` | **Lemma R''**: the 16 point substitutions, exact chords, inner vertex enumeration | table in §6; whole range covered with `>= 3.7 %` margin |
| `p1a_product.py` | refutation of `(P3)` and of seven weighted variants | `1/64` vs `1/27` at the 4-atom block |

### 9.1 The certificate and the certified `r`-sweep

`cert.json`, `cert_staircase.log`, `cert_rstar.log`, `cert_n3sweep.log`. The certificate is run as a whole (all four
pieces) at each `r`; "PROVED" means the interval branch-and-bound closed the residual rectangle.

At `n = 2` (with Lemma T's staircase tail, box budget 20 000):

| `r` | 0.35 | 0.40 | 0.50 | 0.60 | **2/3** | 0.70 | 0.75 | 0.78 | 0.80 | 0.82 | **0.84** | 0.86 | 0.88 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `q = 3r` | 1.05 | 1.20 | 1.50 | 1.80 | **2.00** | 2.10 | 2.25 | 2.34 | 2.40 | 2.46 | **2.52** | 2.58 | 2.64 |
| certificate | P | P | P | P | **P** | P | P | P | P | P | **P** | fails | fails |

So the **certified** bracket is `r*(2) in [0.84, 0.86)`, i.e. `q* in [2.52, 2.58)`. The adversary's
own sweep confirmed `r = 0.82` good and `r = 0.85` short, with a grid bisection estimating `0.8433`;
the certificate both *proves* the lower end and pushes it from `0.82` to `0.84`, the gain coming
from Lemma T. Everything the paper uses at `n = 2` (`q = 2`) sits comfortably inside.

At `n = 3` (`cert_n3sweep.log`, box budget 2 000):

| `r` | 0.35 | 0.40 | **0.50** | **0.60** | 2/3 | 0.70 | 0.75 | ... | 0.95 |
|---|---|---|---|---|---|---|---|---|---|
| `q = 4r` | 1.40 | 1.60 | **2.00** | **2.40** | 2.67 | 2.80 | 3.00 | | 3.80 |
| certificate | P | P | **P** | **P** | fails | fails | fails | fails | fails |

so `r*(3) in [0.60, 0.6667)`, exactly reproducing the adversary's numerically confirmed bracket --
but proved.

Two honest caveats. (i) A "fails" entry is *conservative*: it means the certificate did not close
within the box budget, not that P1 is false there (nothing in this programme has ever produced a
counterexample to P1). A successful cell closes in under 100 boxes, so the budget is not what
decides the brackets. (ii) The `--rstar` bisection was started and **killed for CPU** after the
`n = 2` sweep: each *failing* evaluation has to exhaust the budget, which at 30-digit interval
arithmetic costs minutes, and the sweep already brackets `r*` to `0.02`. So the numbers quoted are
sweep brackets, not bisected values.

### 9.2 The `rho`-scan of `(E_rho)`

`rho.log`, `entropy.json`. Table in §4. Critical coefficient bracketed in `(0.1, 0.2)`; at every
`rho` where it holds the minimum is exactly `0`, attained at the block (`S = 1`, `H = log 3`).
