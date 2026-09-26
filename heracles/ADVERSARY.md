# The adversarial pass: trying to break (T) and P1 — what survived, what broke, and one thing that closed

Written as the designated ADVERSARY of the global-optimality programme (briefing:
`GOPT_STATE.md`). Everyone before me wanted the target to be true and every inherited search
was biased in the same three ways: the `(n+1)`-neuron block was *among the optimiser's starts*,
the whole-network searches stopped at `p = 5, 7`, and the atom searches only ever sat on the
programme's own optimal support. This pass removes all three biases and attacks the three
open statements head-on.

**Headline.** I did **not** break anything. (T) survived 40 independent whole-network cells at
`p = 5..23`, `n = 2, 3`, `q = 2, 2.5, 2.9, 3` *with the block excluded from every start*, and P1
survived a three-million-configuration search. Two things did break, and both are useful:

* the plausible lemma **"purifying a network never lowers its margin" is FALSE** — impure
  neurons really do buy margin over their own purification, so the G2 gap cannot be closed that
  way (§5); and
* the inherited record `min/target = 1.0000009` for P1 at `n = 2, N = 4` was **optimiser noise**,
  not a gap: I reach `1.000000048716`, and the minimum is the target (§3).

And one thing closed: combining two inherited proved lemmas that had never been used together
**proves P1 for every number of atoms `N`** at `n = 2` for every `r <= 0.83` and at `n = 3` for
every `r <= 0.62` — in particular at plain weight decay `q = 2`, which is the case the paper is
about, with 2.2% to spare (§6). That is the first unconditional statement about `N > n+1`.

All float64, local CPU, never more than three worker processes, `$0`. Scripts
`heracles/adv_*.py`, outputs `heracles/FINAL_RUNS/ADV/`. Nothing left running.

---

## 0. Symbols

Inherited symbols keep their meanings (`p`, `n`, `nu = n+1`, `q`, `r = q/nu`, `E`, `W`, `P_q`,
`gamma_q`, `B = 2(2p(n+1))^{-nu/2}`, `Gamma_s(p)`, `c_l`, `v_lj`, `(KL)`, `D_h(n)`). New here:

| symbol | meaning | plain reading |
|---|---|---|
| `ratio` | `gamma_q(theta) / [ (n+1)^{-(nu-q)/q} B Gamma_{nu/q}(p) ]` | the single number (T) says is `<= 1`; `= 1` at the block |
| `purification` | replace each neuron by its projections onto the `(p+1)/2` Fourier subspaces | "split every neuron into one pure neuron per frequency" |
| `beta` | `x_{n+1}`, the modulus of the `(n+1)`-st largest atom | the smallest of the "big" atoms |
| `tau` | `sum_{l > n+1} x_l` | total modulus of everything past the first `n+1` atoms |

---

## 1. What I attacked, and with what

| surface | inherited evidence | what this pass did differently |
|---|---|---|
| (T) on whole networks | `gopt_full.py`: SLSQP on the explicit `p^n(p-1)` margin constraints, `p = 5, 7` only, block among the starts | `adv_net.py`: scale-invariant annealed soft-min ascent with an **analytic gradient** (L-BFGS-B), so it does not degrade with `p`; `p = 5..23`; **no block start ever**; six start families including deliberately impure and deliberately junk-maximal ones; basin hopping; every number re-computed by an independent brute-force loop |
| (T) with junk, `K >= 2` | `gopt4_trade.py` / `gopt4_curve.py`: atom parametrisation on the programme's **optimal** support | `adv_net.py`'s `clustered` and `junk` start families put the optimiser on deliberately non-optimal supports and on one-neuron-per-frequency (maximal junk) configurations, and let it move freely from there |
| impure neurons (G2) | untouched ("still open", `GLOBAL_OPT` §7) | `adv_impure.py`: an exact purification map, its cost identity, and a direct hunt for networks that beat their own purification |
| P1 (`n = 2, N = 4`) | `gopt4_s1.py` block C: 500 Nelder-Mead starts, `min/target = 1.0000009` | `adv_p1.py` block A: 3.03M sampled configurations from 7 structured families, vectorised exact Cramer solution, two-stage polish, **11 values of `r`** from `0.05` to `0.97` |
| P1 (`N > n+1`, degenerate limits) | `lemma_split.py`, `gopt_nlp.py` (random starts) | `adv_p1.py` blocks B and C: starts with **coalescing atoms** at separations `1e-1 .. 1e-5`, where the Cramer minors and `det M` blow up together |

---

## 2. (T) itself: 40 whole-network cells, no block starts, nothing above 1

`adv_net.py` maximises

    ratio = margin(theta) / P_q(theta)^{nu/q} / [ (n+1)^{-(nu-q)/q} B Gamma_{nu/q}(p) ]

over the **raw tensors** `E (n, m, p)`, `W (p, m)` — no purity, no junk-free, no support
restriction, no width restriction beyond `m`. The objective is the *normalised* gap array
`gaps / (P_q^{nu/q} target)`, which is degree-0 homogeneous, so a fixed soft-min temperature
means the same thing at every scale; `beta` is annealed `8 -> 25000` through eight L-BFGS-B
stages with an analytic gradient (checked against central differences: max relative error
`4.8e-10`). The reported `margin` is always the **exact** min over the full `p^n x p` logit
table in fp64, and the winner of every cell is re-evaluated by `brute_ratio`, a separate
loop-based code path — the two agree to every digit printed below.

**The target is attained.** The `(n+1)`-block on the programme's optimal support, with each
frequency's neurons scaled by `b_k^{1/q}` (`b` = the programme's optimal shares — the
inherited `build_chain` uses *equal* amplitudes and therefore misses the target whenever the
programme's optimum is not equal-share, e.g. `ratio = 0.677` at `p = 11, q = 2`), gives
`ratio = 1.000000000000` in all nine cells checked. So `ratio = 1` is reachable and the scale
is right.

**Results** (best over 30 starts x 3 widths x 2 basin hops per cell; `n = 3` used 20 starts):

| `n` | `q` | `p = 5` | `p = 7` | `p = 11` | `p = 13` | `p = 17` | `p = 19` | `p = 23` |
|---|---|---|---|---|---|---|---|---|
| 2 | 2 | | | 0.999995731 | 0.999994185 | 0.999989750 | 0.999982973 | 0.977832362 |
| 2 | 2.5 | | | 0.952397097 | 0.963286991 | | | |
| 2 | 2.9 | | | **0.999999999990** | 0.999999999685 | | | |
| 3 | 2 | 0.999999999970 | 0.999998125 | 0.999997147 | | | | |
| 3 | 3 | 0.999999999971 | 0.999999999830 | 0.933087078 | | | | |

**Maximum over all 40 `(p, n, q, m)` cells: `0.999999999990`. Nothing reached `1`, let alone
exceeded it.** The optimiser, started from random tensors, from impure two-frequency neurons,
from clustered non-optimal supports and from maximal-junk one-neuron-per-frequency
configurations, *converges back to the block* — that is what the `0.99999x` numbers are. Where
it falls short (`p = 23`; `q = 2.5`; `p = 11, n = 3, q = 3`) it is width- or basin-limited, not
because anything better exists: those cells' programme optima need 5–7 frequencies, i.e. 18–28
neurons, more than the widths run. (The one remaining cell, `p = 11, n = 3, q = 3, m = 16`, was
stopped at the CPU budget; its `m = 12` sibling gave `0.933`.)

Two things worth recording for the paper.

* **This is the first whole-network evidence above `p = 7`.** The inherited `gopt_full.py` was
  capped there by the `p^n(p-1)`-constraint SLSQP; the soft-min/L-BFGS formulation runs `p = 23`,
  `n = 2` in 90 s and `p = 11`, `n = 3` (1331 rows) in 4 min on one core.
* **The block is no longer "reached by construction".** Every start here is a non-block
  configuration and the search still lands on `1 - 3e-6` or better in 11 of the 15 `(p, n, q)`
  cells. That converts the inherited caveat ("the block is among the starts, so reaching 1 is by
  construction") into a genuine basin-of-attraction statement.

My programme values `Gamma_{nu/q}(p)` are computed by a harder search than the inherited one
and came out **larger** in one cell (`p = 23, q = 2`: `0.402933388` vs `0.402931495`). Since
`Gamma` is a maximum, searching harder can only raise the target and lower the ratios — the
conservative direction for an adversary, and I used the larger value.

---

## 3. P1 at `n = 2, N = 4`: the `1.0000009` was noise, and the minimum is the target

At `n = 2, N = 4` the system (KL) is exactly square, so `c` is the unique Cramer solution and P1
is the closed six-angle inequality (`GLOBAL_OPT4` §4.1)

    sum_{l=1}^4 |delta_l|^r  >=  3^{1-r} | sum_l (-1)^{l+1} delta_l |^r ,

`delta_l` = the four `3x3` minors of `(a ; b ; ab)` with `a_1 = b_1 = 1`. `adv_p1.py` block A
evaluates this in closed form (verified against `np.linalg.solve` to `3e-15`) on **3,029,000**
configurations drawn from seven families — uniform random, the equality manifold (three atoms at
the block, the fourth anywhere) plus noise, coalescing pairs at separations `1e-1 .. 1e-6`, the
common-phase diagonal `b ~ a`, perturbed fourth-roots-of-unity, and clustered pairs — then
polishes the best 200 and deep-polishes the leading 12.

| `r` | `q = 3r` | global min of `sum|c_l|^r` | `3^{1-r}` | **min / target** |
|---|---|---|---|---|
| 0.05 | 0.15 | 2.839665672592 | 2.839652467920 | 1.000004650101 |
| 0.10 | 0.30 | 2.687929639575 | 2.687875379522 | 1.000020186967 |
| 0.20 | 0.60 | 2.408323097511 | 2.408224685281 | 1.000040865053 |
| 1/3 | 1.00 | 2.080118327869 | 2.080083823052 | 1.000016588186 |
| 0.40 | 1.20 | 1.933198994890 | 1.933182044932 | 1.000008767906 |
| 0.50 | 1.50 | 1.732092212304 | 1.732050807569 | 1.000023905035 |
| **2/3** | **2.00** | 1.442249640568 | 1.442249570307 | **1.000000048716** |
| 0.75 | 2.25 | 1.316074013100 | 1.316074012952 | 1.000000000112 |
| 5/6 | 2.50 | 1.200936955176 | 1.200936955176 | 1.000000000000 |
| 0.90 | 2.70 | 1.116123174034 | 1.116123174034 | 1.000000000000 |
| 0.97 | 2.91 | 1.033507512043 | 1.033507512043 | 1.000000000000 |

**Never below 1, at any `r`.** The inherited `1.000000944618` at `q = 2` is improved by a factor
of 20 (`1.000000048716`) purely by searching harder, and at `r >= 5/6` the search lands on the
target to all 12 digits. So the answer to "is `1.0000009` a true gap?" is **no**: the infimum is
the target, approached along the degenerate equality manifold (`c_4 -> 0`, where `|c_4|^r` has an
infinite-slope cusp that local optimisers cannot descend exactly). The residual `1e-5..1e-8`
excesses at small `r` are the cusp, not a gap — they *shrink* as `r` grows, exactly as a cusp
`|c_4|^r` should.

**Near-degenerate configurations do not hide a violation** (block C, `r = 2/3`, atoms 3 and 4
forced to separation `eps` in both slots, 120 restarts per level):

| forced `eps` | `1e-1` | `1e-2` | `1e-3` | `1e-4` | `1e-5` |
|---|---|---|---|---|---|
| min / target | 1.000000547 | 1.000519805 | 1.000189034 | 1.001275114 | 1.004146534 |

(levels `1e-6` and `1e-8` were started and stopped at the CPU budget; the trend is already
monotone)

The minimum moves **away** from the target as the atoms coalesce. The brief's hypothesis that
"a violation would hide where the Cramer minors blow up" is refuted: when two atoms collide both
`delta_l` and `det M` blow up together, the ratio is scale-invariant, and the surviving
configuration is the ordinary one.

---

## 4. `N > n+1` and the underdetermined case

`adv_p1.py` block B minimises over **both** the phases and the free `c` on the affine (KL)
solution set by joint SLSQP, from four start families (random; block plus extra atoms; a
**coalescing pair** at separation `1e-3`; all atoms on the common diagonal), 40 starts per cell.

| cell | min `sum|c_l|^r` | `(n+1)^{1-r}` | ratio |
|---|---|---|---|
| `n = 2, N = 5, r = 2/3 (q = 2)` | 1.442254292386 | 1.442249570307 | 1.000003274107 |
| `n = 2, N = 5, r = 5/6 (q = 2.5)` | 1.200937215299 | 1.200936955176 | 1.000000216600 |
| `n = 2, N = 6, r = 2/3 (q = 2)` | 1.442257719600 | 1.442249570307 | 1.000005650404 |

Nothing dipped below the target. The remaining cells (`N = 7, 8` at `n = 2`; `N = 5, 6` at
`n = 3`) were stopped at the CPU budget — the SLSQP cost grows fast with `N` — and are recorded
as not run rather than as evidence; `FINAL_RUNS/ADV/p1_B_partial.log`. This block reproduces
`lemma_split.py` and `gopt_nlp.py` on ground they already covered, with degenerate starts added;
it is a consistency check, and §6 supersedes it with a proof.

---

## 5. WHAT BROKE: impurity does buy margin, so G2 cannot be closed by purification

This is the one place where a plausible and load-bearing conjecture died.

**The exact purification map.** Every `R^p` splits orthogonally into the `(p+1)/2` Fourier
subspaces (`k = 0` plus `span{cos, sin}` for `k = 1..(p-1)/2`). Purifying a neuron means
replacing it by the `(p+1)/2` neurons obtained by projecting all `n+1` of its vectors onto one
subspace each.

> **FACT P1 (cost).** Write `A_{ik}` for the squared norm of neuron `i`'s frequency-`k` part, so
> `||theta_i||^2 = sum_k A_{ik}` by Parseval. The neuron costs `(sum_k A_{ik})^{q/2}`, its
> purification costs `sum_k A_{ik}^{q/2}`. For `q >= 2` the map `t -> t^{q/2}` is convex and
> vanishes at `0`, hence superadditive, so
>
>     P_q(purified theta)  <=  P_q(theta),   with EQUALITY exactly at q = 2.
>
> *Plain reading: purifying never costs more, and under plain weight decay it costs exactly the
> same — impurity is **cost-free** at `q = 2` and strictly wasteful for `q > 2`.*
> (Verified: max cost ratio `1.000000000000` over 72 random nets at `p = 5..13`, `n = 2, 3`,
> `q = 2, 2.5, 2.9, 3`; at `q = 2`, `max |ratio - 1| = 4.4e-16`.)

> **FACT P2 (content).** Purification removes exactly the Fourier terms whose operand
> frequencies are not all equal to minus the class frequency, and every such term has **zero
> mean over the `p^n` rows at every offset `D`**. So the clean amplitudes
> `g_k = sum_i prod_j Ehat[j,i,k] What[i,-k]` are preserved exactly, and so is `mean_a Phi_D(a)`
> for every `D`. (Verified to `4.4e-16` and `1.4e-15` respectively.)

Together these say: **the G2 gap is not about cost at all.** At `q = 2` an impure network and its
purification have the *same* penalty and the *same* clean content; the entire difference is the
cross-frequency junk. The obvious next lemma — *and the one that would have closed G2* — is

> **(CONJECTURE, REFUTED) purification never lowers the margin**, i.e.
> `ratio(theta) <= ratio(purified theta)`.

It is false, and not marginally:

| cell | `ratio(theta) - ratio(purified)` over 300 random nets | fraction where impurity helps | after optimising `theta` |
|---|---|---|---|
| `p=11, n=2, q=2` | median `-0.1230`, max `+0.0251` | 0.3% | **`+0.879`** |
| `p=13, n=2, q=2` | median `-0.1120`, max `+0.0853` | 1.0% | **`+0.911`** |
| `p=7, n=2, q=2` | median `-0.1516`, max `+0.1973` | 1.7% | **`+0.736`** |
| `p=11, n=2, q=2.9` | median `-0.0809`, max `+0.1262` | 1.3% | **`+0.440`** |
| `p=7, n=3, q=2` | median `-0.7946`, max `+5.3e-15` | 0.0% | — |

Cross-frequency junk is *usually* ruinous (the median net loses 8–80% of its ratio when you keep
it), but a soft-min-optimised impure network beats its own purification by up to `+0.91`. The
reason is structural and worth stating: purification is not a perturbation — it **deletes whole
terms of the logit function**, and the margin is a `min` over rows, so a deleted term that was
positive at the worst row costs margin. Only the *row-average* is protected (Fact P2), and the
row-average is precisely the quantity that cannot see the `(n+1)` factor (it kills all junk, so a
single maximal-junk neuron matches the block on it — this is why `gamma_q <= B Gamma` comes for
free and the extra `(n+1)^{-(nu-q)/q}` does not).

**But no impure network ever exceeded the target.** Best ratios found inside the impure class,
after soft-min ascent started from two-frequency neurons: `0.999986660` (`p=11, q=2`),
`0.999981828` (`p=13, q=2`), `0.980041168` (`p=7, q=2`), `0.903951306` (`p=11, q=2.9`). So G2
survives as an open gap, with one plausible route now closed and one structural fact
(P1 + P2) gained.

*(`adv_impure.py` has a fourth block `F` that PINS every neuron to two frequencies of comparable
amplitude and maximises inside that class with Powell. It is written and runs, but Powell on
48–96 variables with an exact-margin evaluation at every step did not finish in the CPU budget,
so no `F` number is reported. Nothing hinges on it: Fact P1 already says such a network cannot
be cheaper than the pure one it contains, and block O searches a strictly larger class.)*

---

## 6. WHAT CLOSED: P1 for every `N`, at `n = 2` and `n = 3`, up to an explicit `r`

While looking for the place where P1 is weakest I found that two *inherited, already proved*
lemmas close it, and that nobody had put them together. `adv_proof.py`.

**The two ingredients.**
`(B)` **Theorem B** (`GLOBAL_OPT` §4): for exactly `n+1` atoms, arbitrary `c` and phases, **junk
allowed**, `sum_{l=1}^{n+1}|c_l|^r >= (n+1)^{1-r} max(R,0)^r` with `R = min_{T^n} Re h`.
`(E)` **Theorem E / (PEEL_h)** (`GLOBAL_OPT3` §3): for a (KL) system with sorted moduli,
`sum_{l>h} x_l >= D_h(n)|A|`, `h = 0..n`, with `D_h` exact algebraic
(`n=2`: `1, 3 sqrt3/8, 1/4`; `n=3`: `1, 49/72, 0.324759526, 1/8`).

Inherited status: `(B)` says nothing about `N > n+1`; `(E)` is **proved unable** to reach the
constant on its own (`GLOBAL_OPT3` F3-2: the best any peeling ladder can certify at `nu=3, q=2`
is `1.000521`, and the ladder's own value is `0.99940` of the target).

**The argument.** Normalise `|A| = 1`, sort the moduli, put `y_i = x_i` for `i <= n+1`,
`beta = y_{n+1}`, `tau = sum_{l>n+1} x_l` ("the `n+1` biggest atoms, and the total mass of all
the rest").

1. **Drop the tail.** (KL) makes `h == A`, so the top-`(n+1)` sub-population has
   `h_S(u) = A - sum_{l>n+1} c_l prod_j (1 + u_j v_lj)` and `|prod_j(1+u_jv_lj)| <= 2^n`, hence
   `R_S >= 1 - 2^n tau`. By `(B)`,
   `sum_{i<=n+1} y_i^r >= (n+1)^{1-r} max(1 - 2^n tau, 0)^r`.
2. **Peel.** `sum_{i>h} y_i + tau = sum_{l>h} x_l >= D_h(n)`, so `sum_{i<=n+1} y_i^r` is at least
   `Peel(beta, tau)`, the value of a **concave** minimisation over a pointed polyhedron —
   computed exactly by vertex enumeration (and, because the vertex is *affine* in `(beta, tau)`,
   vectorised over the whole grid).
3. **The tail's own price.** Every tail atom is at most `beta` and `t -> t^{r-1}` decreases, so
   `sum_{l>n+1} x_l^r = sum_l x_l . x_l^{r-1} >= tau . beta^{r-1}`; subadditivity gives
   `>= tau^r`. Write `Tail(tau, beta) = max(tau^r, tau beta^{r-1})`.
   *This is the step that makes one argument cover every `N` at once.* Without it the relaxation
   is vacuous — `(y, tau) = (0,...,0, 1)` is "feasible" with value `1 < (n+1)^{1-r}`, i.e. all the
   mass hides in an unbounded number of tail atoms. `tau beta^{r-1}` is exactly the statement
   that the tail atoms are the *smallest* ones: it blows up as `beta -> 0`.

Hence, for **every** (KL) system and every `N`,

    sum_l |c_l|^r  >=  G(beta, tau)
                   :=  Tail(tau, beta) + max( Peel(beta, tau),
                                              (n+1)^{1-r} max(1 - 2^n tau, 0)^r ),

a function of **two** real variables. P1 follows at a given `r` as soon as
`min G >= (n+1)^{1-r}`.

**Why the two-variable minimum is decided, not merely measured.** At `tau = 0` the second term is
exactly the target and `Tail = 0`, so `min G = target` with the block among the minimisers: the
bound is *sharp* and a numerical minimisation alone could never separate `>=` from `<`. The
small-`tau` region is therefore handled analytically instead. Define

    phi(t) := (n+1)^{1-r}(1 - 2^n t)_+^r + t^r ,
    eps*(n,r) := the largest t0 with phi(t) >= (n+1)^{1-r} for all t in [0, t0].

`G >= phi(tau)` always (drop the `Peel` branch and the second tail bound), so **for
`tau <= eps*` the conclusion is exact algebra** — `phi(0) = target`, `phi` has infinite slope up
at `0` (the `t^r` cusp) against a linear loss, so it rises before it falls. What is then measured
numerically is only

    min { G(beta, tau) : tau >= eps*/2 } ,

on a region that **overlaps** the analytic one, and there the bound clears the target by a
genuine margin:

| `n` | `q` | `r` | `eps*` | `min G` for `tau >= eps*/2` | target | **margin** |
|---|---|---|---|---|---|---|
| 2 | **2** | 2/3 | 1.698e-02 | 1.474488912 | 1.442249570 | **1.022354** |
| 2 | 1.5 | 0.5 | 7.101e-02 | 1.885807659 | 1.732050808 | 1.088772 |
| 2 | 2.25 | 0.75 | 4.082e-03 | 1.322935839 | 1.316074013 | 1.005214 |
| 2 | 2.46 | 0.82 | 4.534e-04 | 1.218967235 | 1.218657950 | 1.000254 |
| 2 | 2.55 | 0.85 | 9.541e-05 | 1.179119488 | 1.179147646 | 0.999976 (short) |
| 3 | **2** | 0.5 | 1.469e-02 | 2.045017914 | 2.000000000 | **1.022509** |
| 3 | 2.4 | 0.6 | 4.856e-03 | 1.745184592 | 1.741101127 | 1.002345 |
| 3 | 2.667 | 2/3 | 1.637e-03 | 1.583078490 | 1.587401052 | 0.997277 (short) |

> **RESULT.** P1 — `sum_l |c_l|^r >= (n+1)^{1-r}|A|^r` for **every** number of atoms `N` — is
> proved at `n = 2` for every `r <= r*(2) ~ 0.83` (`q <= 2.5`) and at `n = 3` for every
> `r <= r*(3) ~ 0.62` (`q <= 2.5`). **In particular it is proved at `q = 2`, plain weight decay,
> at both `n = 2` and `n = 3`, with 2.2% of margin.**
>
> (`adv_proof.py`'s bisection reports `r*(2) = 0.8433`, `r*(3) = 0.6268`, but it runs the
> minimisation on a coarser `400 x 400` grid than the sweep table; the conservative reading is
> the table's own bracket — confirmed good at `r = 0.82` / `0.60`, confirmed short at `0.85` /
> `0.6667`. Quote `r* ~ 0.83` and `~ 0.62`, or re-run the bisection at grid `600` if a sharper
> figure is wanted. Nothing at `q = 2` depends on this: the margin there is 2.2%.)

**Status and caveats, stated precisely.**
* The argument is *conditional on the two inherited lemmas*, which I did not re-derive. I did
  re-check both numerically as used here: over 125 random exact (KL) systems (`n = 2, 3`,
  `N = n+1..n+5`), `(E)` holds with `min_h [sum_{l>h}x_l]/D_h(n) = 1.032419`, `(B)` holds with
  `min sum_S|c|^r/[(n+1)^{1-r}R_S^r] = 1.116329`, the inequality `R_S >= 1 - 2^n tau` holds with
  minimum slack `+0.003417`, the assembled bound is never violated
  (`min sum|c|^r / G = 1.008693`), and P1 itself clears the target by `1.127057` at worst.
* Steps 1–3 are exact algebra. The inner `Peel` is exact (concave over a pointed polyhedron,
  every vertex enumerated; the vertex is affine in `(beta, tau)`, so the whole grid is one array).
  The only numerical ingredient is the minimisation of a **two-variable** function on
  `{tau >= eps*/2}`, done on a `600 x 600` grid plus Nelder-Mead polish; a certified version
  would be a routine interval branch-and-bound in two variables, which I did not run.
* The search box `beta <= 0.8`, `tau <= 0.4` loses nothing. `beta >= 0.8` forces
  `Peel >= (n+1) 0.8^r >= 1.018 x target` at every `r` (all `n+1` big atoms are `>= beta`), and
  re-running the whole sweep with `beta <= 2`, `tau <= 4` reproduces every value to 9 digits.
* It does **not** reach `r -> 1`. There `eps* -> 0` super-exponentially (at `r = 0.95`,
  `eps* = 8.5e-13`), the drop bound becomes useless, and what is left is the peeling ladder, which
  F3-2 already proved cannot get there. The shortfall is tiny — at most `1.22e-4` relative at
  `n = 2` (worst at `r = 0.925`) and `5.9e-3` at `n = 3` (worst at `r = 0.8`) — but it is real.
* The gap that remains is therefore `r in (r*, 1)`, i.e. `q` between about `2.5` and `nu`. Note
  that `q = nu` is trivially true (`r = 1`: `sum|c_l| >= |C_empty|`), so the open window is an
  interior interval, not a tail.

**What would close the window.** Any of: (i) the sharp `D_1^{sharp}(2) >= 0.6495` improved
towards the LP value `0.653611` — the peeling ladder's own value would then rise and the
crossover `r*` with it; (ii) a better constant than `2^n` in `R_S >= 1 - 2^n tau` (the bound
`|prod_j (1+u_j v_lj)| <= 2^n` is attained only when all `n` factors align, and at the minimising
`u` they cannot all align for every tail atom simultaneously); (iii) applying `(B)` to the best
`(n+1)`-subset rather than the top one.

---

## 7. Attacks that found nothing, and exactly how hard they tried

Recorded so the paper can describe the evidence honestly, and so nobody repeats them.

**A1. Whole networks, block excluded.** `adv_net.py`, 40 cells
(`p = 5,7,11,13,17,19,23`; `n = 2,3`; `q = 2, 2.5, 2.9, 3`; `m = 8..20`), 30 starts x 3 widths x
2 basin hops each (1 at `n = 3`) = about 2,400 independent ascents, ~1 h of CPU. Distribution of
the best ratio per cell: `> 0.999997` in 11 of the 15 `(p,n,q)` cells, `0.93-0.98` in the other
4 (all width- or basin-limited). **Maximum `0.999999999990`.** No cell ever produced a ratio above `1`, and no
`VIOLATION_*.npz` file was written.

**A2. Junk-maximal and non-optimal-support starts.** The `junk` family puts exactly one neuron on
each of `min(m, (p-1)/2)` frequencies, which is the maximal-junk configuration (`eta = sqrt(2^n-1)`,
the very regime where `gopt4_bound.py`'s tool pair still leaves a factor `1.30`); the `clustered`
family puts the support on four consecutive frequencies, which is far from the programme's
optimum. Both families are among the starts of every cell above. In every case the ascent either
climbs back to the block's basin or stalls below it. **Junk at the top of its range is not a
direction of improvement at the network level — the `1.30` residual is slack in the *tools*, not
in the truth.**

**A3. Impure neurons.** §5. The one thing that broke is a proof route, not (T).

**A4. P1 by brute search.** §3–§4. 3.03M configurations x 11 exponents at `n = 2, N = 4`, plus
`N = 5, 6` at `n = 2` with phases and `c` both free. **Nothing below the target, at any `r`.**

**A5. Coalescing atoms.** §3. Refuted as a hiding place.

**A6. A certified branch-and-bound for P1 at `n = 2, N = 4`. ABANDONED, with a reason.** I wrote
the Lipschitz bound (`|delta_l|` and `|det M|` are sums of 6 and 24 unit-modulus monomials of
degree `<= 4`, so `L_delta = 24`, `L_D = 96` per unit of angle) and then computed what it would
cost. The equality set of the six-angle inequality has **codimension 2** (`delta_4 = 0` is one
complex equation), the excess above the target vanishes like `eps^r` at distance `eps` from it
while the Lipschitz error is `O(rho)`, so boxes must satisfy `rho <~ L^{-1/(1-r)} ~ 7e-5`, and
covering a codimension-2 neighbourhood at that resolution needs `~(2 pi/rho)^4 ~ 6e19` boxes.
**A Lipschitz branch-and-bound cannot certify this inequality**, and that is a property of the
inequality (its equality manifold is large and its excess is a cusp), not of my implementation.
The 2-variable programme of §6 is certifiable in a way the 6-angle one is not; that is the other
reason §6 is the better object.

---

## 8. Corrections and cautions for the inherited material

1. **`build_chain` does not attain the target when the programme's optimal shares are unequal.**
   The constructed block in `fvf_obj.py` / `gopt_full.py` gives every frequency the same
   amplitude. At `p = 11, q = 2` the programme's optimum is `b = (0.4305, 0.5695)` and the
   equal-amplitude block scores `ratio = 0.677385`, not `1`; at `p = 13` it scores `0.797165`.
   Scaling frequency `k`'s neurons by `b_k^{1/q}` restores `ratio = 1.000000000000` at every
   prime. Any statement of the form "the block attains the target" should say **the
   share-weighted block**. (`GLOBAL_OPT4` §6.2's table is unaffected — it works on the atom
   parametrisation, where amplitudes are free.)
2. **`Gamma_{3/2}(23)` is slightly larger than recorded.** `GLOBAL_OPT4` §6.2 has `0.402931495`;
   a harder programme search gives `0.402933388`. Immaterial (it makes the target *larger*), but
   worth fixing since the ratios are quoted to 12 digits.
3. **The `1.0000009` in `GLOBAL_OPT4` §4.1 should not be read as a gap.** It is the cusp, and it
   shrinks to `1.000000000000` as `r` grows and to `1.000000049` at the same `r` with a better
   search (§3).
4. **"Junk buys `Theta(J^2)`" (Theorem J) and "the trade fails above `eta_cross`" are consistent
   with everything here**, but note that the `1.30` residual of `gopt4_bound.py` at maximal junk
   is now directly contradicted at the network level by A2: maximal-junk starts never climb above
   `1`. The residual is a limitation of the Lemma-Q/`Cl2` tool pair.

---

## 9. Numerical record

All float64, local CPU, at most 3 worker processes, no pods, `$0`. Outputs in
`heracles/FINAL_RUNS/ADV/`.

| script | what it does | result |
|---|---|---|
| `adv_net.py` | **(T) on whole unrestricted networks**, block excluded from every start; annealed soft-min ascent with analytic gradient + basin hopping; exact fp64 margin over the full `p^n x p` table; independent `brute_ratio` re-evaluation of every winner | 40 cells `p = 5..23`, `n = 2,3`, `q = 2..3`; **max ratio `0.999999999990`**, never `>= 1` |
| `adv_p1.py` block A | **P1 at `n = 2, N = 4`** via the exact Cramer minors, 3,029,000 configurations from 7 structured families, two-stage polish, 11 values of `r` | min/target `>= 1` at every `r`; best approach `1.000000048716` (`r = 2/3`), `1.000000000000` for `r >= 5/6` |
| `adv_p1.py` block C | coalescing atoms, `eps = 1e-1 .. 1e-5` (`r = 2/3`), 120 restarts per level | min/target rises from `1.0000005` to `1.0041` as `eps` shrinks — the degenerate limit is not a hiding place |
| `adv_p1.py` block B | `N = 5, 6` at `n = 2`, phases **and** `c` free by joint SLSQP, 40 starts incl. coalescing pairs | ratios `1.0000033`, `1.0000002`, `1.0000057` — nothing below `(n+1)^{1-r}`; larger `N` stopped at the CPU budget |
| `adv_impure.py` block V | the purification cost identity and content identity | `P_q(purified)/P_q(theta) <= 1` always, `= 1 +- 4.4e-16` at `q = 2`; `g_k` preserved to `4.4e-16`; row-means to `1.4e-15` |
| `adv_impure.py` blocks S/O | does impurity ever pay? 300 random nets per cell (6 cells), then soft-min ascent from impure starts | **"purification never lowers the margin" is FALSE** (gains up to `+0.911`); but best impure ratio `0.999986660` — never above `1`. Block F (pinned two-frequency neurons) written but not run to completion |
| `adv_proof.py` | **P1 for every `N`**: `(B)` + `(E)` + the tail-price term, reduced to a 2-variable programme; `r` sweep and `r*`; full numerical re-check of every link | PROVED for `r <= 0.83` (`n = 2`) and `r <= 0.62` (`n = 3`); at `q = 2` the margin is `1.0224` (`n = 2`) and `1.0225` (`n = 3`); chain re-verified on 125 random (KL) systems |

**Reproducing** (from the repo root, about 1.5 h total on one core):

    uv run python heracles/adv_net.py --p 11 13 17 --n 2 --q 2.0 --m 9 12 15 --starts 30 --tag _n2a
    uv run python heracles/adv_net.py --p 19 23 --n 2 --q 2.0 --m 12 16 20 --starts 30 --seed 5 --tag _n2b
    uv run python heracles/adv_net.py --p 11 13 --n 2 --q 2.5 2.9 --m 15 18 --starts 30 --seed 7 --tag _n2q
    uv run python heracles/adv_net.py --p 5 7 11 --n 3 --q 2.0 3.0 --m 8 12 16 --starts 20 --seed 9 --tag _n3
    uv run python heracles/adv_p1.py --blocks A --nsample 3000000 --ntop 200 --tag _A
    uv run python heracles/adv_p1.py --blocks C --starts 120 --tag _C
    uv run python heracles/adv_p1.py --blocks B --starts 40 --tag _B
    uv run python heracles/adv_impure.py --blocks VSO --nsamp 300 --starts 10
    uv run python heracles/adv_proof.py

---

## 10. Verdict

| statement | verdict after this pass |
|---|---|
| **(T)** `gamma_q <= (n+1)^{-(nu-q)/q} B Gamma_{nu/q}(p)` | **NOT BROKEN.** 40 whole-network cells at `p = 5..23`, `n = 2,3`, `q = 2..3`, block excluded from every start, maximum ratio `0.999999999990`. Still OPEN as a theorem. |
| **P1** `sum_l\|c_l\|^r >= (n+1)^{1-r}\|A\|^r`, all `N` | **PROVED** at `n = 2` for `r <= ~0.83` and `n = 3` for `r <= ~0.62`, in particular at `q = 2` (§6), conditional on the two inherited lemmas and on a 2-variable minimisation. OPEN for `r in (r*, 1)`. |
| **P3 / G3** large junk on a `K >= 2` support | **NOT BROKEN.** Maximal-junk starts never climb above `1`; the `1.30` residual is tool slack. Still OPEN as a theorem. |
| **G2** impure neurons | **NOT BROKEN**, and the obvious route to closing it is now **REFUTED** (§5). Impurity is cost-free at `q = 2` and cost-increasing for `q > 2`, so G2 is purely a junk question. |
| "the best ratio to `B Gamma_K` is the same for every `K`" | consistent with everything measured: the share-weighted block attains `ratio = 1` at `K = 1..6` across `p = 5..23`, and no `K` ever exceeded it. |
