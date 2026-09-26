# S1 and S2: an exact dual form for the sharp constant, a proved barrier, and the junk relief is quadratic

Fourth pass. Continues `heracles/LEMMA_ATTEMPT.md` (agent 1), `heracles/GLOBAL_OPT.md` (agent 2),
`heracles/GLOBAL_OPT3.md` (agent 3). Target, unchanged, for the gated chain and `2 <= q <= nu`:

    gamma_q(theta)  <=  (n+1)^{-(nu-q)/q} * B * Gamma_{nu/q}(p),      B = 2 (2 p (n+1))^{-nu/2}.      (T)

The two statements I was asked to settle:

* **(S1) SHARP CONSTANT.** For every `N >= n+1` and every (KL) system, `sum_l |c_l|^r >= (n+1)^{1-r}|A|^r`.
* **(S2) JUNK CANNOT PAY.** Remove the same-`|k|`-junk-free hypothesis (H) from agent 3's Theorem D.

**Status: S1 OPEN (reformulated exactly, and one whole family of proof strategies proved impossible).
S2 PARTIAL, with the inherited obstruction REFUTED as a statement about the truth** — the cost relief
that junk buys is `Theta(J^2)`, not `Theta(J)`; its leading coefficient is now known in closed form and
certified in exact arithmetic at `nu = 3`; and with that coefficient the margin/cost trade, which
previously failed for *every* junk level on a `K >= 2` support, now closes below an explicit junk level
`eta_cross(K)` — about a third of a feature's amplitude at the paper's feature counts (§6). Nothing here
contradicts (T); every numerical attack on (T) failed again, now on `K = 2, 3, 4` frequency supports
with junk deliberately forced on.

All float64 (plus one exact-arithmetic certificate), local CPU, **no pods, $0**. Scripts
`heracles/gopt4_*.py`, outputs `heracles/FINAL_RUNS/GOPT4/`.

---

## 0. Symbols, with a plain reading of each

Inherited symbols keep their meaning (`p`, `n`, `nu = n+1`, `q`, `r = q/nu`, `e_ij`, `w_i`, `P_q`,
`gamma_q`, `G_k`, `c_ik`, `v_ijk`, `J_{k,T}`, `Gamma_s`, `B`). New or re-used here:

| symbol | meaning | plain reading |
|---|---|---|
| `(KL)` | `sum_{l=1}^N c_l prod_{j in T} v_lj = A [T empty]` for every `T subset {1..n}` | "`N` atoms carry clean amplitude `A` and cancel all `2^n - 1` junk sign patterns" |
| `h(u) = sum_l c_l prod_j (1 + u_j v_lj)` | the population's generating polynomial | (KL) `<=>` `h == A`; the constant coefficient is clean, the other `2^n - 1` are junk |
| `Lambda(z) = sum_T lambda_T prod_{j in T} z_j` | a **test polynomial**: multi-affine, `lambda_empty = 1` | a "normalised probe"; its constant coefficient is its average over the torus |
| `K` | number of frequencies actually used | the feature count `K*` of the paper |
| `w_kD = 1 - cos(2 pi k D / p)` | the programme's weight | how much frequency `k` contributes to the gap at offset `D` |
| `C_T^{(k)}` | frequency `k`'s sign-pattern amplitude | `C_empty^{(k)} = G_k` is clean, the rest is junk |
| `eta_k = sqrt(sum_{T != empty} |C_T^{(k)}|^2) / |G_k|` | **relative junk** at frequency `k`, `l_2` measure | "what fraction of this feature's amplitude is junk" |
| `rho_k = max_{T != empty} |C_T^{(k)}| / |G_k|` | the same in the `max` measure | `eta_k/sqrt(2^n-1) <= rho_k <= eta_k` |
| `Cl2_r(eta)`, `Cmax_r(rho)` | the **junk-relief functions**, defined in §5 | the cheapest a feature with that much junk can be |
| `beta_{n,q}` | curvature of the relief at `eta = 0` | how fast junk starts to pay |
| `M(Lambda)` | Mahler measure `exp( integral over T^n of log|Lambda| )` | the geometric-mean size of a test polynomial |

---

## 1. Exact statements

> **PROPOSITION D1 (proved; every `n`, `N`, every configuration).**
> Fix `v_1..v_N` in `T^n` and weights `x_1..x_N > 0`. Then
>
>     min { max_l |c_l| / x_l : (KL) holds with A = 1 }
>       *  min { sum_l x_l |Lambda(v_l)| : Lambda multi-affine, lambda_empty = 1 }  =  1.
>
> Consequently a modulus profile `x` is realisable by some feasible `c` (`|c_l| <= x_l` for all `l`)
> **if and only if**
>
>     sum_l x_l |Lambda(v_l)|  >=  |A|     for every normalised multi-affine Lambda,      (D)
>
> and therefore **S1 is exactly equivalent to the real-variable statement**
>
>     (S1-dual)  x >= 0 and (D)  ==>  sum_l x_l^r >= (n+1)^{1-r} |A|^r .
>
> *Plain reading: the phases of the atoms carry no information at all for S1 — the whole problem is
> about the `N` numbers `|Lambda(v_l)|` as `Lambda` ranges over normalised test polynomials.*

> **LEMMA M (proved; every `n`).** For multi-affine `Lambda` on `T^n`, `M(Lambda) >= |lambda_empty|`.
> *Plain reading: a test polynomial cannot be small on average (in the geometric-mean sense) unless its
> own average is small.*

> **THEOREM O (a proved barrier; every `n >= 1`, every `r in (0,1]`).**
> For every probability measure `pi` on normalised multi-affine polynomials,
>
>     sup over v in T^n of  E_pi |Lambda(v)|^r   >=  1.
>
> Hence the entire **"one averaged test measure" method** — take (D), raise it to the power `r`, use
> `(sum y_l)^r <= sum y_l^r`, and average over `pi` — can never prove `sum_l x_l^r >= kappa^{-1}|A|^r`
> with `kappa^{-1} > 1`. It therefore **cannot prove S1 for any `n` and any `r < 1`**.
>
> *Scope, stated precisely.* What is ruled out is any argument of that shape whose control on
> `E_pi |Lambda(.)|^r` is **uniform over the torus**, i.e. that does not use where the atoms actually
> are. (A `pi` chosen after seeing `v_1..v_N` and bounded only at those `N` points is not excluded —
> but then one is back to a non-uniform, atom-by-atom argument, which is the point of the corollary
> below and of §7 F4-3.)
>
> *Corollary (the slot recursion is capped).* The exact recursion
> `sum_l |c_l|^r |1 + w v_ln|^r >= n^{1-r}|A|^r` (valid for every `w in C`, by induction on `n`) yields,
> under any measure on `w`, at best `sum_l |c_l|^r >= n^{1-r}|A|^r` — never the factor `((n+1)/n)^{1-r}`
> that (T) needs. This makes `GLOBAL_OPT` F-3 / `GLOBAL_OPT3` F3-3 final rather than "not yet made to work".

> **THEOREM J (the junk-relief curvature; the value of the second-order programme is exact, and
> certified in exact arithmetic at `n = 2`).**
> Let
>
>     Cl2_r(eta) := min { sum_l |c_l|^r : C_empty = 1, sum_{T != empty} |C_T|^2 <= eta^2 },
>
> the minimum over every number of atoms and all unit phases. `Cl2_r(0) = (n+1)^{1-r}` is the Key Lemma.
> The second-order programme at the `(n+1)`-block has value exactly `lambda_n(r)/(n+1)`, with
>
>     beta_{n,q} = lambda_n(r) / (n+1),
>     lambda_n(r) = r (1 - r) / [ 1 - ((n-1)/(n+1))^2 (1 - r) ],        r = q/(n+1),
>
> so that `Cl2_r(eta) = (n+1)^{1-r} (1 - beta_{n,q} eta^2 + O(eta^4))`. Values:
>
> | `nu` | `q` | `r` | `beta` exact | decimal |
> |---|---|---|---|---|
> | 3 | 2 | 2/3 | **1/13** | 0.0769230769 |
> | 3 | 7/3 | 7/9 | 14/237 | 0.0590717300 |
> | 3 | 5/2 | 5/6 | 5/106 | 0.0471698113 |
> | 4 | 2 | 1/2 | **1/14** | 0.0714285714 |
> | 4 | 3 | 3/4 | 1/20 | 0.0500000000 |
> | 5 | 2 | 2/5 | 3/49 | 0.0612244898 |
> | any | `nu` | 1 | 0 | 0 |
>
> **The inherited S2 obstruction — "junk of size `J` buys `Theta(J)` of cost relief" — is therefore
> false as a statement about the truth.** It is an artefact of the three available *proofs* (agent 1's
> `(C')`, agent 3's robust peeling, agent 2's Theorem B all degrade linearly). The truth is quadratic,
> and `beta -> 0` as `q -> nu`, i.e. **junk buys strictly nothing at Morwani's exponent.**

> **THE TRADE, QUANTIFIED (§6).** Combining the measured `Cl2_r` with Lemma Q's margin penalty
> gives, for pure-neuron networks on any support with any junk,
>
>     gamma_q / B  <=  V(p),   V(p) := max over x >= 0, eta >= 0 of
>        min_{D != 0} [ sum_k x_k w_kD - (2^n - 1)^{-1/2} max_k eta_k x_k sqrt(w_kD/2) ]
>        subject to sum_k Cl2_r(eta_k) x_k^r <= 1 ,
>
> and `V(p)` is compared with the target `(n+1)^{-(1-r)/r} Gamma_{nu/q}(p)`. Results in §6.

**What is NOT claimed.** Theorem J certifies the *second-order programme*. That this programme governs
`Cl2_r` near `0` needs the minimiser to stay near the block as `eta -> 0`, which follows from the Key
Lemma's equality case for `N = n+1` and is conditional on S1 for `N > n+1`. And `Cl2_r` itself, away
from `0`, is a measured global minimum (SLSQP, `N = n+1, n+2`, 24 starts per level), not a proved lower
bound. §6's conclusion is conditional on that, and I say so again there.

---

## 2. Reproducing everything

Run from `heracles/` (the scripts import one another). Total about 2 hours on one core; the last three
are the expensive ones and can be cut down with fewer `--starts`.

    uv run python gopt4_relief.py --starts 24                    # the relief functions Cmax, Cl2
    uv run python gopt4_secord.py                                # beta as an eigenvalue + closed form
    uv run --with sympy python gopt4_exact.py                    # exact certificate for beta, n = 2
    uv run python gopt4_dual.py --trials 25                      # Proposition D1
    uv run python gopt4_s1.py --blocks MO                        # Lemma M and Theorem O
    uv run python gopt4_s1.py --blocks C --starts 500            # n = 2, N = 4 (the first open case)
    uv run python gopt4_cross.py --cells 2:2 3:2 --starts 14     # the junk threshold (JT)
    uv run python gopt4_trade.py --p 11 13 17 19 23 --starts 30  # (T) on junk-carrying networks
    uv run python gopt4_curve.py --p 11 --starts 10              # the forced-junk curve
    uv run python gopt4_bound.py --p 11 13 --starts 120          # the unrelaxed V(p)

---

## 3. Proposition D1: S1 is a statement about `|Lambda(v_l)|`, not about phases

**Why this is worth having.** Agent 3's Theorem E proves S1 up to a factor `F` by *peeling*: choose a
test polynomial `Lambda` vanishing at `h` chosen atoms and use

    |A| = |sum_l c_l Lambda(v_l)|  <=  sum_{l not in S} |c_l| |Lambda(v_l)|  <=  ||Lambda||_inf sum_{l not in S} |c_l|.

Agent 3 then proved (F3-2) that the best constants obtainable this way are strictly below the block's,
so the ladder *cannot* be sharp. Proposition D1 locates the loss exactly: **the only lossy step in the
whole method is the last one, `|Lambda(v_l)| <= ||Lambda||_inf`.** Keeping `|Lambda(v_l)|` loses
nothing at all.

**Proof.** Write `X` for the `N x 2^n` matrix `X_{l,T} = prod_{j in T} v_lj`, so (KL) reads
`X^T c = e_empty`. With `D = diag(x)` and `d := D^{-1} c`,

    PRIM := min { ||d||_inf : (X^T D) d = e_empty }.

This is a feasible convex programme with finite value, so Lagrangian duality is exact:

    PRIM = max_lambda [ Re<lambda, e_empty> + min_d ( ||d||_inf - Re<(X^T D)^* lambda, d> ) ],

and the inner minimum is `0` when `||(X^T D)^* lambda||_1 <= 1` and `-infinity` otherwise. Now
`((X^T D)^* lambda)_l = x_l * sum_T conj(v_l^T) lambda_T = x_l Lambda(conj(v_l))`, so
`||(X^T D)^* lambda||_1 = sum_l x_l |Lambda(conj(v_l))|`, and

    PRIM = max { Re lambda_empty : sum_l x_l |Lambda(conj(v_l))| <= 1 } = 1 / DUAL

by homogeneity in `lambda` (the family `{|Lambda(conj(v_l))|}` and `{|Lambda(v_l)|}` coincide, since
`lambda` is free complex: replace `Lambda(z)` by `Lambda(conj(z))`). `PRIM <= 1` is exactly the
existence of a feasible `c` with `|c_l| <= x_l`, which is (D). Finally `sum_l |c_l|^r` is monotone in
the moduli, so minimising it over feasible `c` is the same as minimising `sum_l x_l^r` over realisable
profiles. ∎

*Weak duality, which is all §6 uses as a guard, is one line: for any feasible `c` and any normalised
`Lambda`, `|A| = |sum_l c_l Lambda(v_l)| <= max_l(|c_l|/x_l) * sum_l x_l |Lambda(v_l)|`.*

**Numerical check (`gopt4_dual.py`).** `PRIM * DUAL - 1` lies in `[1.7e-08, 2.7e-05]` over 64 random
`(n, N, x, v)` with `n = 1,2,3` and `N` up to `2^n + 2` — always `>= 0`, as weak duality forces, and
`0` to solver accuracy. Both minimisations are convex and are solved by smoothing
(`|w| -> sqrt(|w|^2+eta)`, `max -> log-sum-exp`) with `eta` driven to `1e-14` and L-BFGS restarts.

> **A correction to my own first pass.** I first checked D1 with Nelder-Mead/Powell on the raw
> non-smooth objectives and got `PRIM*DUAL` up to `2.47`. That is not a counterexample to D1: weak
> duality only forces `PRIM*DUAL >= 1`, and a *too large* product is exactly what an unconverged
> minimisation produces. Nelder-Mead does not converge on these in `2(2^n - 1)` dimensions, and both
> problems are convex, so the smoothed L-BFGS version above is the one to trust. The bad run is left in
> `FINAL_RUNS/GOPT4/s1_D.json`; the correct one is `FINAL_RUNS/GOPT4/dual*.json`. Recorded so nobody
> re-runs the broken version and reads it as evidence against D1.

**Also recorded, because it bit me:** for `N < 2^n` the system (KL) is *overdetermined*, so a random
phase configuration has **no** feasible `c` at all; junk-free populations with `N = n+1 < 2^n` atoms
live on a positive-codimension variety in phase space (at `n = 2` this is the common-phase family of
`LEMMA_ATTEMPT` §4). At `n = 2`, `N = 4 = 2^n` is exactly the square case: the solution is unique, which
is what makes §4.1 possible.

---

## 4. Theorem O: a whole family of proofs of S1 is impossible

**Lemma M.** Write `Lambda(z) = alpha(z_{-n}) + z_n beta(z_{-n})` with `alpha, beta` multi-affine in
`n - 1` variables. Jensen's formula in one variable gives, for each fixed `z_{-n}`,

    integral over |z_n| = 1 of log|Lambda| dz_n  =  log max(|alpha|, |beta|)  >=  log|alpha|.

Integrating over the remaining variables, `log M(Lambda) >= log M(alpha)`; and `alpha` is the
multi-affine polynomial in `n-1` variables whose constant coefficient is still `lambda_empty`. Induction
down to `n = 0` gives `M(Lambda) >= |lambda_empty|`. ∎

*(Check: `gopt4_s1.py` block M, 300 random `Lambda` at `n = 1..3`, `64^n` grid — minimum ratio
`0.999984`, i.e. `>= 1` to the accuracy of the log-integral quadrature.)*

**Theorem O.** For a probability measure `pi` on normalised multi-affine `Lambda`, Jensen (`t -> t^r`
concave, or simply AM–GM) gives for each `Lambda`

    E_{v ~ Haar} |Lambda(v)|^r  >=  exp( r E_v log|Lambda(v)| )  =  M(Lambda)^r  >=  1,

so `E_v E_pi |Lambda(v)|^r >= 1` by Fubini, and hence `sup_v E_pi |Lambda(v)|^r >= 1`. ∎

**Why this closes a family.** Every attempt so far of the shape "use the master inequality once,
`r`-subadditively, under an averaged probe" runs:

    |A|^r <= ( sum_l x_l |Lambda(v_l)| )^r <= sum_l x_l^r |Lambda(v_l)|^r,    then average over pi,
    |A|^r <= ( sup_v E_pi |Lambda(v)|^r ) * sum_l x_l^r .

Theorem O says the bracket is `>= 1` for **every** `pi`, so this can only ever reproduce
`sum_l x_l^r >= |A|^r`. Special cases already recorded as dead — the product family
`Lambda = prod_j (1 + lambda_j z_j)` (Lemma P's shape), the last-slot induction of `GLOBAL_OPT` F-3 and
`GLOBAL_OPT3` F3-3, and Mahler/geometric-mean routes — are all instances. **A proof of S1 must use the
constraints non-uniformly across atoms**, i.e. several different `Lambda` tight at different atoms.
That is precisely the vertex/ladder structure of Theorem E, and F3-2 says that structure is capped
*when it is fed sup-norms*; D1 says it is not capped when it is fed `|Lambda(v_l)|`. So the live gap is
a single, sharply-posed object.

*(Check: `gopt4_s1.py` block O, 200 random finite `pi` (up to 5 atoms), random `n in {1,2,3}`, random
`r in [0.2, 1]`, `96^n` grid — minimum of `sup_v E_pi|Lambda(v)|^r` is `1.031486`, never below 1.)*

### 4.1 The first genuinely open case of S1 is a determinant inequality

S1 is proved at `N = n+1` (Key Lemma) and, for **every** `N`, at `n = 1`: there
`D_1(1) = kappa_1 = 1/2` is the block's own value, so agent 3's ladder is already sharp — the ladder
programme `min{ sum_l x_l^r : x_1 >= x_2 >= ... >= 0, sum_l x_l >= 1, sum_{l>1} x_l >= 1/2 }` is a
concave minimisation over a pointed polyhedron, and every vertex (`(1/2,1/2,0,...)`, `(1/3,1/3,1/3)`,
`(1/2,1/4,1/4)`, ...) has value `>= 2^{1-r}`, attained at `(1/2, 1/2, 0, ...)`. The first open case is therefore `n = 2`,
`N = 4`, and it is **exactly square**: (KL) is `2^n = 4` complex equations in `4` unknowns, so for a
generic configuration `c` is *unique* and given by Cramer's rule. Writing `a_l, b_l` for the two slot
phases and

    M = [ 1     1     1     1   ;  a_1   a_2   a_3   a_4 ;
          b_1   b_2   b_3   b_4 ;  a_1 b_1 ... a_4 b_4 ],      delta_l := det( M with row 1 and column l deleted ),

one has `c_l = (-1)^{l+1} delta_l / det M` and `det M = sum_l (-1)^{l+1} delta_l`, so

> **S1 at `n = 2`, `N = 4` is the closed statement:** for any four points of `T^2`,
>
>     sum_{l=1}^4 |delta_l|^r  >=  3^{1-r} | sum_{l=1}^4 (-1)^{l+1} delta_l |^r ,
>
> `delta_l` the four `3 x 3` minors of the `3 x 4` matrix `(a ; b ; a b)`.

After the gauge `a_l -> lambda_1 a_l`, `b_l -> lambda_2 b_l` (which changes nothing: rows 2–4 of `M`
are set to zero by the system) and fixing `a_1 = b_1 = 1`, this is a **six-angle** problem. The
equality cases include every configuration in which three of the points form the block and the fourth
is anywhere (then `c = (1/3, 1/3, 1/3, 0)` is the unique solution). Global search
(`gopt4_s1.py` block C, Nelder-Mead from many random starts):

| `q` | `r` | global min of `sum_l |c_l|^r` | `3^{1-r}` | ratio |
|---|---|---|---|---|
| 2 | 2/3 | 1.442250932683 | 1.442249570307 | **1.000000944618** |
| 5/2 | 5/6 | 1.200936955176 | 1.200936955176 | **1.000000000000** |

i.e. the minimum is the target and is **never** undercut, to the accuracy Nelder-Mead reaches on a
non-smooth objective (`1e-6`; the optimiser lands on the block plus a fourth atom of modulus `< 1e-5`).
This is not a proof, but it says the open case is a six-angle determinant inequality with a known,
degenerate, equality manifold — the most concrete form S1 has taken so far.

For `N >= 5` at `n = 2` the system is underdetermined and the extra freedom is a null space; block G
of the same script would re-verify `N = n+2 .. n+4` at `n = 2, 3` by direct SLSQP, but it was not
run to completion here — agent 1's `lemma_split.py` (200-600 SLSQP restarts, `N` up to `n+5`,
min `= (n+1)^{1-r}` to 8 digits) and agent 2's `gopt_nlp.py` (50 cells, ratio
`1.000000000000`) already cover exactly that ground.

---

## 5. Theorem J: the junk relief is quadratic, with an exact coefficient

### 5.1 What was believed, and what is true

The inherited S2 obstruction (`GLOBAL_OPT3` §5, `LEMMA_ATTEMPT` §6 G3) is a comparison of two rates.
Robust peeling degrades to `sum_{l not in S} |c_l| >= 2^{-h}|A| - (1 - 2^{-h}) J`, so junk of size `J`
looked as though it bought `Theta(J)` of cost relief, against a margin penalty that Lemma Q / Lemma C3
only deliver at `O(J/p)`. On a `K >= 2` support, where the clean margin is `Theta(1)` rather than
`O(p^-2)`, that trade fails for **every** `J`, which is why G3 looked hopeless.

The rate is wrong. Define, for a single frequency, normalised to `C_empty = 1`:

    Cmax_r(rho) := min { sum_l |c_l|^r : C_empty = 1, max_{T != empty}|C_T| <= rho },
    Cl2_r(eta)  := min { sum_l |c_l|^r : C_empty = 1, sum_{T != empty}|C_T|^2 <= eta^2 },

both over every `N` and all unit phases. `gopt4_relief.py` measures them (SLSQP with analytic Jacobians,
`N = n+1` and `n+2`, block and random starts). Log–log slopes of the deficit `1 - C/C(0)` on the four
smallest junk levels:

| cell | `Cmax` slopes | `Cl2` slopes |
|---|---|---|
| `nu=3, q=2` | 2.0008, 2.0017, 2.0038 | 2.0004, 2.0010, 2.0022 |
| `nu=3, q=2.5` | 2.0006, 2.0014, 2.0031 | 2.0004, 2.0008, 2.0018 |
| `nu=4, q=2` | 2.0019, 2.0043, 2.0096 | 2.0000, 2.0000, 2.0001 |
| `nu=4, q=3` | 2.0014, 2.0032, 2.0071 | 2.0000, 2.0000, 2.0001 |

**The relief is exactly quadratic.** (The proved bounds are all linear: agent 1's `(C')` gives
`Cmax_r(rho) >= (n+1)^{1-r}(1 - (2^n-1) rho)^r`, agent 2's Theorem B gives
`(n+1)^{1-r}(1 - Theta(rho))^r`, agent 3's robust peeling the same. At `nu = 3, q = 2, rho = 0.1` the
proved bound is `(1 - 3*0.1)^{2/3} = 0.7884` of `Cmax_r(0)`; the truth is `0.9975470`. The proved bound
hits zero at `rho = 1/3`, where the truth is still `0.9613`.)

### 5.2 Why no odd power can appear, and the exact coefficient

**The structural reason.** Rotating slot `j`'s phases, `v_lj -> e^{i psi_j} v_lj`, leaves every `|c_l|`
untouched and multiplies `C_T` by `exp(i sum_{j in T} psi_j)`. So the function

    F(zeta) := min { sum_l |c_l|^r : C_empty = 1, C_T = zeta_T for T != empty }

is invariant under an `n`-torus acting **non-trivially on every coordinate** `zeta_T`, `T != empty`.
So `F` is a function of the moduli `(|zeta_T|)_T` alone, and in particular it has no linear term in
`zeta`: the only first-order behaviour symmetry leaves open is a conical one,
`F(zeta) = F(0) - G((|zeta_T|)_T) + o(||zeta||)` with `G` positively homogeneous of degree `1`. The
measured log–log slopes (`2.0000`–`2.0096`) exclude that, and the second-order computation below
supplies the quadratic that replaces it.

**The exact coefficient.** Expand at the block `c_l = (1 + eps_l)/(n+1)`, `v_lj = w^{-l} e^{i delta_lj}`
with `w = exp(2 pi i/(n+1))`, `eps_l = x_l + i y_l`:

    sum_l |c_l|^r = (n+1)^{1-r} [ 1 + (1/(n+1)) ( (r(r-1)/2) sum_l x_l^2 + (r/2) sum_l y_l^2 ) + O(3) ],
    C_empty = 1  <=>  sum_l eps_l = 0 + O(2),
    C_T = (1/(n+1)) sum_l w^{-l|T|} ( eps_l + i sum_{j in T} delta_lj ) + O(2).

Two things make this a genuine curvature. `r(r-1)/2 < 0` for `r < 1`: **unequal moduli are what buys
cost**, at second order. And the `delta_lj` — the phases of the embeddings — **cost nothing at all**, so
they are the free currency with which the junk constraint is paid. Hence

    beta_{n,q} = (1/(n+1)) * max { (r(1-r)/2) sum_l x_l^2 - (r/2) sum_l y_l^2 :
                                   sum_l eps_l = 0, sum_{T != empty}|C_T|^2 <= 1 },

a homogeneous quadratic-over-quadratic programme, equal to the least `lambda >= 0` with
`P - lambda Q <= 0` on the `(2(n+1) + n(n+1) - 2)`-dimensional space of admissible perturbations
(`gopt4_secord.py` solves it by bisection on the largest eigenvalue; kernel directions of `Q` are the
junk-free perturbations, on which the Key Lemma forces `P <= 0`, so the value is finite).

**The closed form.** The eigenvalue is
`lambda_n(r) = r(1-r) / [1 - ((n-1)/(n+1))^2 (1-r)]`, `beta = lambda_n(r)/(n+1)`. Verified against the
bisection in **70 cells** (`n = 1..7`, `r = 0.15, 0.2, 1/3, 0.4, 0.5, 2/3, 0.75, 5/6, 0.9, 1`): ratio
`1.000000000` in every cell, zero mismatches at tolerance `1e-6`.

**Exact certificate at `nu = 3`** (`gopt4_exact.py`, sympy, exact arithmetic over `Q(w)`): for
`q = 2, 7/3, 5/2` the matrix `lambda* Q - P` is shown positive semidefinite by verifying that every
coefficient of `det(t I + lambda* Q - P)` is `>= 0` (for a real symmetric matrix this is equivalent to
PSD), and `det(lambda* Q - P) = 0` with a kernel vector `u` satisfying `u^T Q u > 0` (`169/3`,
`6241/48`, `2809/12` respectively), so no smaller `lambda` works. Hence
`lambda_2(2/3) = 3/13`, `lambda_2(7/9) = 14/79`, `lambda_2(5/6) = 15/106` **exactly**.

**Agreement with the constrained optimiser** (`gopt4_relief.py` at `eta = 10^-3`, which is where the
`O(eta^4)` term is still `~3e-4` relative):

| cell | `beta` from Theorem J | `beta` measured | ratio |
|---|---|---|---|
| `nu=3, q=2` | 0.076923077 | 0.076946 | 0.999702 |
| `nu=3, q=2.5` | 0.047169811 | 0.047181 | 0.999763 |
| `nu=4, q=2` | 0.071428571 | 0.071429 | 0.999994 |
| `nu=4, q=3` | 0.050000000 | 0.050000 | 1.000000 |

**Two readings worth keeping for the paper.** `beta_{n,nu} = 0`: at Morwani's exponent `q = nu` junk
buys *nothing* at second order, and the relief function collapses to `Cl2_1(eta) = 1` for every `eta`
(at `r = 1` the cost is exactly `sum_l |c_l| >= |C_empty|`, with no junk term at all). That is at least
consistent with the repo's standing observation that `L_{2,nu}`-trained nets come out clean while `l_2`
nets carry a large junk fraction, though the two junk measures are not the same quantity and I have not
checked the correspondence. And `beta` is largest at intermediate `r`, peaking near
`r ~ 1/2`, i.e. plain weight decay at `nu = 4` is where junk is most tempting.

### 5.3 The relief functions away from `0`

`nu = 3, q = 2` (`(n+1)^{1-r} = 1.442249570307`), relative values `C/C(0)`:

| `rho` or `eta` | 0.01 | 0.05 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|---|
| `Cmax` | 0.9999768 | 0.9994066 | 0.9975470 | 0.9893224 | 0.9727691 | 0.9384572 | 0.8724540 |
| `Cl2` | 0.9999923 | 0.9998046 | 0.9992050 | 0.9966920 | 0.9921938 | 0.9852818 | 0.9752207 |

`Cl2_r` is bounded below by `1` for every `eta` (since `sum_l |c_l|^r >= (sum_l |c_l|)^r >= |C_empty|^r`),
and `Cl2_r(sqrt(2^n-1)) = 1` is attained by a single atom (`N = 1`: `h = c_1 prod_j (1 + u_j v_1j)`,
all `2^n` coefficients equal). So the relief function runs from `(n+1)^{1-r}` down to exactly `1`, and
at `nu = 3, q = 2` the *total* margin headroom junk can ever buy is `(1.4422)^{1/r} = 1.7321`, a factor
`sqrt(3)` — which is exactly the factor by which a single junk-carrying neuron beats the block on cost
and loses on margin.

*(One optimiser artefact to note: in `relief.log`, cell `nu=4, q=2`, `mode=max` at `rho = 0.3, 0.4` the
reported deficit is `0.000000` — SLSQP failed to leave the block from 24 starts there, so the reported
`Cmax` is an over-estimate at those two points. It does not touch the small-`eta` fit or any `l_2`
number used later.)*

---

## 6. The trade, at the network level

### 6.1 An explicit junk threshold

For pure single-frequency neurons (any number per frequency, arbitrary phases, junk allowed) the two
ingredients are exact:

* **cost**, frequency by frequency (different frequencies never mix, `LEMMA_ATTEMPT` §2 Step 2):
  `sum_{k,l} |c_lk|^r >= sum_k Cl2_r(eta_k) |G_k|^r`;
* **margin**, Lemma Q at mode `m_{k,T}` symmetrised over `+-D`, using `|sin(theta_kD/2)| = sqrt(w_kD/2)`
  and `max_T |C_T^{(k)}| >= eta_k |G_k| / sqrt(2^n - 1)`:

      2^n margin <= min_{D != 0} [ sum_k |G_k| w_kD - (2^n-1)^{-1/2} max_k eta_k |G_k| sqrt(w_kD/2) ].

With `sum_{k,l}|c_lk|^r` normalised to `1` one has exactly `gamma_q = B * Lambda`. Write
`x_k := |G_k|`, `phi(eta) := (Cl2_r(0)/Cl2_r(eta))^{1/r} >= 1` (the margin headroom junk buys) and
`xtilde_k := x_k / phi(eta_k)`, so that `sum_k Cl2_r(0) xtilde_k^r <= 1` and hence
`min_D sum_k xtilde_k w_kD <= (n+1)^{-(1-r)/r} Gamma_{nu/q}(p)` by the definition of `Gamma`. Spending
one inequality, `max_k a_k >= (1/K) sum_k a_k` over the `K` frequencies actually used,

    Lambda  <=  min_D sum_k xtilde_k phi(eta_k) [ w_kD - eta_k (2 K^2 (2^n-1))^{-1/2} sqrt(w_kD) ] ,

so (T) follows on this class as soon as, for every `k` and every `w in (0, 2]`,
`phi(eta_k)[ w - eta_k (2K^2(2^n-1))^{-1/2} sqrt(w) ] <= w`. The bracket is worst at `w = 2`, giving the
**explicit junk threshold**

    (JT)        eta / ( 2 K sqrt(2^n - 1) )   >=   1 - ( Cl2_r(eta) / Cl2_r(0) )^{1/r}
                for every eta <= eta_cap   ==>   (T) holds on this class.

*Plain reading: the left side is what one junk-carrying frequency loses on the margin, diluted by the
`K` frequencies it has to share the single Fourier probe with; the right side is what the whole support
gains on the cost. **The left side is linear in `eta` and, by Theorem J, the right side is quadratic**,
so (JT) holds up to an explicit crossing point.* Under the old, proved, LINEAR relief bound the right
side would be linear too, with slope `~ (2^n - 1)/r` against the left's `1/(2K sqrt(2^n-1))` — a factor
of order `K (2^n-1)^{3/2}/r` the wrong way. **That, and nothing else, is why the junk trade used to look
lost for every junk level.**

Note that (JT) does not involve `p` except through `K`, and that the only inequality spent beyond the
two ingredients is `max >= mean`, so the threshold below is conservative.

**The threshold** (`gopt4_cross.py`; the relief curve is re-measured on a 41-point grid over
`[0, sqrt(2^n - 1)]`, so `eta_cross` is quantised to about `0.044` at `nu = 3`). At `nu = 3`, `q = 2`:

| `K` | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|
| `eta_cross` | **all `eta`** | 0.672 | 0.584 | 0.452 | 0.407 | 0.319 | 0.275 | 0.187 |
| as a fraction of the largest possible junk `sqrt 3` | 1 | 0.388 | 0.337 | 0.261 | 0.235 | 0.184 | 0.159 | 0.108 |

At `nu = 4`, `q = 2` (`beta = 1/14`, measured small-`eta` slope `0.142873` against the predicted
`beta/r = 0.142857`), the dilution factor `2 K sqrt(2^n - 1)` is worse by `sqrt(7/3)`:

| `K` | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|
| `eta_cross` | 0.551 | 0.483 | 0.348 | 0.280 | 0.213 | 0.145 | 0.145 | 0.078 |
| as a fraction of `sqrt 7` | 0.208 | 0.183 | 0.132 | 0.106 | 0.080 | 0.055 | 0.055 | 0.029 |


The paper's feature counts are `K* = 4` to `6` at `nu = 3`, so **junk up to roughly a third of a
feature's amplitude cannot pay** (conditionally on the measured `Cl2_r` being the true minimum — see
"What is NOT claimed" in §1), on any support, at any prime. At `K = 1` the condition holds
for every `eta` at all, which re-derives agent 2's single-frequency result from the relief function
alone. As an independent check of Theorem J, the measured small-`eta` slope of the right-hand side of
(JT) is `0.115735` against the predicted `beta/r = 0.115385`.

For comparison, `gopt4_bound.py` evaluates the *unrelaxed* version

    V(p) := max over x >= 0, eta >= 0 of  min_{D != 0} [ sum_k x_k w_kD
              - (2^n-1)^{-1/2} max_k eta_k x_k sqrt(w_kD/2) ]   s.t.  sum_k Cl2_r(eta_k) x_k^r <= 1,

over all `(p-1)/2` frequencies at once with the junk levels unconstrained. With junk free to reach its
maximum `eta = sqrt(2^n-1)` it returns `V/target = 1.304226` at `p = 11` and `1.301838` at `p = 13`
(argmax at `eta_k = sqrt 3` on every frequency): **this pair of tools does not price junk at the very
top of its range**, and the residual factor is about `1.30`. What closes there is not the pair of tools
but the direct searches of §6.2, and, for `K = 1`, agent 2's Theorem B/C, which are exact with junk.

### 6.2 Direct tests on junk-carrying networks

`gopt4_trade.py` maximises the exact `ratio = Lambda / [(n+1)^{-(1-r)/r} Gamma]` over pure-neuron
networks with `n + 2` atoms per frequency on the programme's optimal support, from block, perturbed-block
and random starts (annealed soft-min ascent, then an exact-margin polish). **Anything above `1` refutes
(T).**

| `p` | `Gamma_{3/2}(p)` | optimal support | `K` | block ratio | **best ratio found** |
|---|---|---|---|---|---|
| 11 | 0.535725926 | {2, 5} | 2 | 1.000000000000 | **1.000000000000** |
| 13 | 0.497772067 | {1, 4, 6} | 3 | 1.000000000000 | **1.000000000000** |
| 17 | 0.448340560 | {2, 5, 8} | 3 | 1.000000000000 | **1.000000000000** |
| 19 | 0.442148593 | {1, 5, 8} | 3 | 1.000000000000 | **1.000000000000** |
| 23 | 0.402931495 | {1, 5, 8, 11} | 4 | 1.000000000000 | **1.000000000000** |

(`p = 29, 31` were started and abandoned for CPU: the exact-margin polish over `4 x 4 = 16`
atoms is a 96-dimensional Nelder-Mead and did not finish. The `p = 29` block ratio had already
come out at `1.000000000000`.)

Two independent things are worth noting. The block attains `1.000000000000` at every prime — an
independent check of the whole `gamma_q = B Lambda` bookkeeping against `lemma_verify.py` check B. And
the free optimiser, given junk-carrying starts, always comes **back** to the junk-free block: junk is
not a local direction of improvement at any of these `K`.

`gopt4_curve.py` does the same with the relative junk *forced* to a prescribed level `rho` by a penalty,
which is the only way to see the trade-off curve (left free, the optimiser simply returns to the block):

`p = 11`, `K = 2`, four atoms per frequency, 10 starts per level (soft-min ascent, then an exact-ratio
Nelder-Mead polish with the junk level pinned):

| forced `rho` | 0 | 0.02 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 0.8 |
|---|---|---|---|---|---|---|---|---|
| best `ratio` | **1.000000000000** | 0.972825 | 0.894047 | 0.816325 | 0.697472 | 0.575509 | 0.336063 | -0.000438 |

**The curve falls away from `1` immediately and monotonically.** These are lower bounds on the true
maximum at each junk level (a constrained non-convex ascent), so the honest reading is: junk of any
prescribed size costs a lot of margin at the network level and is never seen to pay. At `p = 23` the
same run was abandoned — the exact-margin polish in 64 dimensions is too slow to be worth the CPU, and
the free search of §6.2 already covers that cell.

---

## 7. Approaches tried here that FAILED, and exactly where

**F4-1. Proving S1 from the dual reformulation by a single dual measure. REFUTED, in full generality,
by Theorem O.** This is the strongest negative result of this pass: it is not that a particular measure
fails, it is that `sup_v E_pi |Lambda(v)|^r >= 1` for every `pi`, because the Mahler measure of a
normalised multi-affine polynomial is at least `1`. It kills, at one stroke, the product family, the
last-slot induction (`GLOBAL_OPT` F-3, `GLOBAL_OPT3` F3-3), and any future variant of the same shape.

**F4-2. Bootstrapping S1 from a smaller exponent. Dead, and instructive.** Proposition D1's programme
suggested reading `Phi_r(V) = min{ sum x^r : (D) }` as an inf over `c` at the exponent `rho = r/(1+r)`;
I spent time on that before realising the duality is `max`, not `sum` — the correct dual of
`sum_l x_l |Lambda(v_l)|` is `min_c max_l |c_l|/x_l`, not `min_c sum_l |c_l|/x_l`. With the correct
duality the relaxation is *exactly* the original problem (that is the content of D1), so there is no
self-improving inequality `F(r) >= F(r/(1+r))^{1+r}`. Recorded because the wrong version looked like a
bootstrap and is not one.

**F4-3. Improving the peeling ladder with the exact `|Lambda(v_l)|`.** D1 says this is the right thing
to do, and at the block it works perfectly: at `n = 2`, killing atom `l_0` with
`Lambda(z) = 1 - (1/2) conj(w)^{l_0} z_1 - (1/2) conj(w)^{2 l_0} z_1 z_2` gives `|Lambda(v_l)| = 3/2`
at the two survivors, hence `x_1 + x_2 >= 2/3` — the block's value, not agent 3's `D_1 = 0.6495`; and
the two-point kill gives `|Lambda| = 3`, hence `x_l >= 1/3`, and those three constraints already pin
`sum x^r = 3^{1-r}` exactly. **But** turning this into a proof needs a bound on `max_{l not in S}
|Lambda(v_l)|` that is uniform over configurations, and there is none: the surviving atoms can sit
wherever they like, including at `Lambda`'s sup-norm points. Making that pay requires using the fact
that those atoms must *also* satisfy their own kill constraints — a coupling the ladder discards. I did
not find a way to exploit it.

**F4-4. Hadamard/Vandermonde for `N > n+1`.** The `N`-point Vandermonde bound
`prod_{l<m}|v_l - v_m| <= N^{N/2}` is still true for every `N`, so the geometric mean of the pairwise
chords is at most `N^{1/(N-1)} -> 1`: more atoms must crowd. Combined with the exact slot recursion
`sum_{l != m} |c_l|^r (d^{(j)}_{ml})^r >= n^{1-r}|A|^r` this looked like a mechanism forcing the cost up
with `N`. It is not quantitatively enough: at `n = 2, r = 2/3` the recursion at the block is already
`10%` slack (`0.9615` available versus `0.8736` needed, exactly `GLOBAL_OPT` F-3's number), and every
way I combined it with the crowding bound lost more than that.

**F4-5. A variance ("fourth-moment") junk lemma to beat Lemma Q. Dead by a factor 2.** For any `D`,
writing `Y = Phi_D - mean(Phi_D)`, the elementary bound `Var(Y) = E Y^2 <= ||Y||_inf E|Y| <= 2
||Y||_inf * (mean - min)` gives

    margin  <=  mean(Phi_D)  -  ( sum_{m != 0} |psihat_D(m)|^2 ) / ( 2 sum_{m != 0} |psihat_D(m)| ),

which sums over *all* junk modes instead of taking the largest. But
`sum a_i^2 / sum a_i <= max a_i`, so this is always at most **half** of Lemma Q — never better. To beat
Lemma Q by the `sqrt(L)` that many modes should give, one needs `mean - min >= c * sd`, i.e. a genuine
hypercontractivity estimate for the degree-`n` multilinear polynomial `Phi_D` on `Z_p^n`. The coordinate
functions there have `||g||_4/||g||_2` as large as `K^{1/4}`, so there is no universal constant; I
could not make this rigorous and did not use it. **This is the single most promising remaining route to
S2** (see §8).

**Previously recorded dead routes**: Morwani-style dual certificate, the domination lemma, gauge
symmetrisation, grid rotation, Gram/Carathéodory–Fejér, Mahler/geometric mean for `N > n+1`, the
capped-mass conjecture `(C')` (refuted by agent 3 at `0.98285239`), the peeling ladder's own cap
(F3-2). None re-attempted; nothing here changes their status, and Theorem O now explains *why* several
of them were bound to fail.

---

## 8. What remains open, ranked

1. **S1 proper.** Prove `L_r(n) = (n+1)^{1-r}`, i.e. (S1-dual) of Proposition D1. After Theorem O the
   shape of any proof is constrained: it must be a *non-uniform* argument over atoms (several test
   polynomials, each tight at a different atom), i.e. a vertex argument on the convex body
   `{x >= 0 : (D)}`. The concrete open sub-case is `n = 2, N = 4`, where the atom vector is the unique
   solution of a `4 x 4` Cramer system and S1 becomes the closed determinant inequality
   `sum_l |delta_l|^r >= 3^{1-r} |sum_l (-1)^{l+1} delta_l|^r` for the four `3 x 3` minors of
   `(a ; b ; ab)`. It is a six-angle problem after fixing the gauge `a_1 = b_1 = 1` (§4.1).
2. **The lower bound on `Cl2_r`.** Theorem J gives the curvature at the block exactly; §6's conclusion
   needs `Cl2_r(eta) >= (n+1)^{1-r}(1 - beta_eff(eta) eta^2)` as a *theorem*, with `beta_eff` the
   measured envelope. The proved bounds are linear and useless here. A proof would most likely come
   from a junk-robust version of the Key Lemma that prices junk through `min_{T^n} Re h` **and** the
   torus symmetry, rather than through `|A| - (2^n - 1)J`.
3. **A `Theta(J)` margin penalty on a `K >= 2` support.** Lemma Q gives `Theta(J sin(pi/p))`, i.e.
   `O(J/p)`; the variance route (F4-5) gives at most half of Lemma Q; hypercontractivity would give
   `Theta(J)` with the `l_2` junk norm summed over frequencies but needs a degree-`n` `(2,4)` estimate
   on `Z_p^n` that is uniform in the support size `K`. With such an estimate S2 would close outright
   rather than numerically.
4. **For the paper, unchanged from agent 3 but strengthened.** The safe claim is Theorem D on junk-free
   networks of any width with the factor `F(n,q)` (`1.0009` at `nu = 3`), plus agent 2's Theorems A/B/C
   for a single frequency with junk. What this pass adds that is quotable: (i) the `n+1` in (T) is a
   Fekete/Hadamard constant (agent 1) *and* the whole sharp-constant question is a statement about test
   polynomials evaluated at the atoms, with a proved barrier explaining why the soft proofs all fail
   (D1 + Theorem O); (ii) **junk is a second-order temptation with an exactly computable curvature
   `beta = lambda_n(r)/(n+1)` that vanishes at `q = nu`** — a clean, checkable statement that explains
   the observed `l_2`-versus-`L_{2,nu}` junk asymmetry and converts G3 from "the trade always fails"
   into "the trade fails only above an explicit junk level".

---

## 9. Numerical record (all float64 unless stated, local CPU, no pods, `heracles/FINAL_RUNS/GOPT4/`)

| script | what it does | result |
|---|---|---|
| `gopt4_dual.py` | **Proposition D1**: `PRIM * DUAL` on random `(n, N, x, v)`, `n = 1,2,3`, `N` up to `2^n + 2`; both minimisations convex, solved by smoothing + L-BFGS | `PRIM*DUAL - 1` in `[1.7e-08, 2.7e-05]`, never negative (weak duality), zero to solver accuracy |
| `gopt4_s1.py` block M | **Lemma M**: Mahler measure of a normalised multi-affine `Lambda` vs `1`, 300 random `Lambda`, `n = 1..3`, `64^n` log-integral grid | min ratio `0.999984` (quadrature error), never below `1` |
| `gopt4_s1.py` block O | **Theorem O**: `sup_v E_pi \|Lambda(v)\|^r` for 200 random finite `pi`, random `n in {1,2,3}`, random `r in [0.2,1]`, `96^n` grid | min `1.031486`, never below `1` |
| `gopt4_s1.py` block C | **`n = 2, N = 4`** (the first open case of S1) via the exact `4 x 4` Cramer solution, global Nelder-Mead search over the 6 gauge-fixed angles, 500 (`q=2`) / 300 (`q=5/2`) starts | min/target `= 1.000000944618` (`q=2`), `1.000000000000` (`q=5/2`) — never undercut |
| `gopt4_junk.py` | the first pass at `Cmax_r`, `nu = 3`, `q = 2`, `N = 3, 4`, 40 starts (superseded by `gopt4_relief.py`, kept because it is the run in which the quadratic rate first showed up) | deficit/`rho^2` flat at `0.2309` over `rho = 1e-4 .. 3e-2` |
| `gopt4_relief.py` | `Cmax_r(rho)`, `Cl2_r(eta)` on 11 junk levels, `nu = 3` (`q = 2, 2.5`) and `nu = 4` (`q = 2, 3`), `N = n+1, n+2`, 24 starts | log–log slopes of the deficit `2.0000`–`2.0096`: **the relief is quadratic**; tables in §5.1/§5.3 |
| `gopt4_secord.py` | **Theorem J**: `beta` as the least `lambda` with `P - lambda Q <= 0` (bisection on the largest eigenvalue), and the closed form, `n = 1..7`, `r = 0.15 .. 1` | closed form `lambda_n(r) = r(1-r)/[1 - ((n-1)/(n+1))^2 (1-r)]` matches the eigenvalue in **70/70 cells**, ratio `1.000000000`; matches the constrained optimiser to `3e-4` or better |
| `gopt4_exact.py` | **exact-arithmetic certificate** for Theorem J at `n = 2`, `q = 2, 7/3, 5/2` (sympy) | `lambda* Q - P` PSD (all coefficients of `det(tI + lambda*Q - P)` are `>= 0`), `det(lambda*Q - P) = 0`, kernel vector with `u^T Q u = 169/3, 6241/48, 2809/12 > 0`. Hence `3/13`, `14/79`, `15/106` **exactly** |
| `gopt4_cross.py` | the explicit junk threshold (JT) as a function of `K`, `nu = 3` and `nu = 4` at `q = 2` | tables in §6.1; the measured small-`eta` slope reproduces `beta/r` from Theorem J to `3e-3` and `1e-4` |
| `gopt4_bound.py` | the unrelaxed `V(p)` over all `(p-1)/2` frequencies with free junk | `V/target = 1.304226` (`p=11`), `1.301838` (`p=13`), argmax at maximal junk — the tool pair leaves a residual factor `~1.30` at the very top of the junk range |
| `gopt4_trade.py` | **(T) itself on junk-carrying pure-neuron networks**: exact margin over the whole `p^n x (p-1)` table, `n+2` atoms per frequency, block / perturbed / random starts, annealed soft-min ascent plus exact polish, `p = 11..23` (`K = 2, 3, 4`) | best ratio `1.000000000000` in all 5 primes — **nothing exceeded `1`** |
| `gopt4_curve.py` | the same with the relative junk forced to a prescribed level, `p = 11`, `rho = 0 .. 0.8` | ratio falls monotonically `1.000000 -> -0.000438`; table in §6.2 |
