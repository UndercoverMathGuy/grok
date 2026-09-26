# P1 (the sharp constant): an independent converging proof, exact peeling constants, and a duality attack on the `r -> 1` window

Fifth pass on the global-optimality programme. Continues `LEMMA_ATTEMPT.md` (agent 1),
`GLOBAL_OPT.md` (agent 2), `GLOBAL_OPT3.md` (agent 3), `GLOBAL_OPT4.md` (agent 4); briefing
`GOPT_STATE.md`. Run in parallel with the adversary agent (`ADVERSARY.md`). My problem was **P1**:

>     (P1)   sum_{l=1}^N |c_l|^r  >=  (n+1)^{1-r} |A|^r     for every N >= n+1,

for every configuration satisfying the atom system

>     (KL)   sum_{l=1}^N c_l prod_{j in T} v_lj  =  A [T empty]     for every T subset of {1..n},

`v_lj` of unit modulus, `r = q/nu in (0,1]`, `nu = n+1`.

---

## Status, and the honest credit

**Convergence, not priority.** Working independently and in parallel, the adversary agent and I
arrived at *the same* proof architecture for P1: bound the `n+1` largest atoms by a junk-robust Key
Lemma in which the rest of the profile plays the role of junk, add the peeling ladder, and — the step
neither earlier pass took — **charge the tail for its own cost**, which is `Theta(tail^r)` against a
relief that is only `Theta(tail)`. Their `ADVERSARY.md` §6 states it as a two-variable programme
`G(beta, tau)`; mine (§4–§6 below) is a two-branch covering argument with the same ingredients.
Their version is stronger where it overlaps (they keep `beta = x_{n+1}` and so price a spread tail
better, reaching `r <= 0.83` at `n = 2`). **I record my derivation because it is an independent
check of a load-bearing claim, and because three things in it are not in theirs:**

1. **A correction to `GLOBAL_OPT3`'s peeling constants, and the `nu = 5` cell.**
   `D_h(n)` should be a max over `d_1 + ... + d_h <= n`, not `= n`. This gives
   `D_1(n) = kappa_3 = 49/72` for **every** `n >= 3` (against `kappa_4 = 0.678905` at `n = 4`,
   `kappa_5 = 0.666728` at `n = 5`) and makes **every constant in the whole programme exact and
   algebraic** — only `kappa_1 = 1/2`, `kappa_2 = 3 sqrt3/8`, `kappa_3 = 49/72` ever appear, so agent
   3's rigour caveat about numerically-evaluated `S_n` for `n >= 4` disappears. **Honest scope:** I
   first claimed this correction is what closes `nu = 5`; it is not. Re-running the cell with agent
   3's original constants gives the same `L_r(4,0) = 2.526617550` (the `h = 1` constraint is slack at
   the minimising vertex), so `nu = 5` closes either way — it closes because nobody had run that cell,
   not because of the correction. What the correction does buy is exactness, plus a small rise at
   `n = 5` (`L_r(5,0)`: `3.105667 -> 3.106914`, not enough to close `nu = 6`). §5.
2. **A reformulation** that makes P1 a classical extremal problem: (KL) says exactly that `z = c/A`
   is a **complex cubature rule on `T^n`, exact for every multi-affine polynomial**, and P1 bounds
   its `l_r` cost from below. §3.
3. **A duality attack on the open `r -> 1` window**, which is where the coordinator asked for one.
   Near `r = 1` every configuration with `sum_l |c_l|` bounded away from `|A|` is handled by
   subadditivity alone, so the window collapses onto **aligned** configurations, where the rule is a
   genuine *probability* cubature rule. For those I prove, by an explicit kernel test polynomial,
   `max_l x_l <= A_d = (1 - m_d)/(2^d - m_d)`, which is **exactly `1/(n+1)` at `n = 1` and `n = 2`** —
   so **P1 holds for every `r in (0,1]` on the aligned class at `n <= 2`** — and a stability version
   that shrinks the remaining window to the explicit sliver `0 < sum_l|c_l| - |A| < ~6(1-r)^2`. §7.

Plus two exact refutations: **P1-prod is false** (the sign rule), and the multiplicative structure
of (KL) is **essential** (four arbitrary unimodular constraint rows give `0.925 < 1`). §8.

All float64, local CPU, **no pods, $0**. Scripts `heracles/p1b_*.py`, outputs
`heracles/FINAL_RUNS/P1B/`. No background jobs remain.

---

## 0. Symbols, with a plain reading of each

| symbol | meaning | plain reading |
|---|---|---|
| `n` | number of operand slots | `nu = n+1` is the homogeneity degree |
| `q in [2, nu]`, `r = q/nu in (0,1]` | penalty exponent, and its ratio | `q = 2` is plain weight decay, `q = nu` is Morwani's `L_{2,nu}` |
| `v_l = (v_l1,...,v_ln) in T^n` | atom `l`'s phase data | which way each of its `n` embeddings points at this frequency |
| `c_l in C`, `x_l = \|c_l\|` | atom `l`'s complex amplitude and its modulus | how much clean signal it carries |
| `A` | the clean total (normalised to `1`) | the feature's Fourier coefficient `G_k` |
| `B` | the `n+1` largest atoms | "the feature proper" |
| `E = sum_{l>n+1} x_l` | the **tail** | all modulus mass outside those `n+1` atoms |
| `S = sum_l x_l`, `delta = S - 1` | total mass, and its excess | `delta = 0` iff all `c_l` share one phase |
| `d^{(j)}_{lm} = \|v_lj - v_mj\|` | slot-`j` chordal distance, in `[0,2]` | how far apart two atoms point in slot `j` |
| `V_j = prod_{l<m in B} d^{(j)}_{lm}` | slot-`j` Vandermonde of `B` | how spread out the feature's atoms are in slot `j` |
| `D_h(n)`, `kappa_d = (1-2^{-d})/S_d` | the peeling constants | after discarding the `h` biggest atoms, `D_h\|A\|` of mass survives |
| `K_J(z) = prod_{j in J}(1 + z_j conj(v_{0j}))` | the reproducing kernel at atom `0` on slots `J` | the test polynomial used in §7; mean `1`, value `2^{\|J\|}` at `v_0` |
| `m_d = min_{T^d} Re prod_{j}(1+w_j)` | how negative that kernel's real part gets | `0, -1/2, -2, -5.545, -13.5, -30.84` for `d = 1..6` |

---

## 1. Exact statements

> **LEMMA 0 (the kill identity).** (KL) holds **iff** for every `u in C^n`,
> `sum_l c_l prod_j (u_j - v_lj) = A prod_j u_j`.
> *Plain reading: a junk-free population is one whose evaluation polynomial is the pure monomial.
> Putting `u_j = v_mj` switches atom `m` off.*

> **PROPOSITION B1 (branch 1; proved, every `n >= 1`, every `N`, every `r in (0,1]`).**
> With `A = 1`,   `sum_l x_l^r >= (n+1)^{1-r}(1 - 2^n E)_+^r + E^r`.
> *Plain reading: the `n+1` biggest atoms must still do the Key Lemma's job and the tail can relieve
> them by at most `2^n E`, but the tail is itself part of the cost being bounded and it costs `E^r`.
> Since `r < 1`, `E^r` beats the linear relief near `E = 0`.* Threshold
> `eps0(n,r) := sup{ e : g_{n,r} >= 0 on [0,e] }`, `g_{n,r}(E) = E^r - (n+1)^{1-r}[1-(1-2^nE)^r]`.
> (The adversary's step 1 + step 3 with `Tail = tau^r`; independently derived, §4.)

> **PROPOSITION B2 (branch 2; proved).** The peeling ladder `sum_{l>h} x_l >= D_h(n)|A|` (`h = 0..n`),
> with the **corrected** `D_h(n) = max{ prod_i kappa_{d_i} : d_i >= 1, sum_i d_i <= n }`, plus the
> single extra constraint `E >= delta`, gives `sum_l x_l^r >= L_r(n, delta)`, the exact vertex value of
> a concave minimisation over a pointed polyhedron. Threshold
> `eps1(n,r) := inf{ delta : L_r(n, delta) >= (n+1)^{1-r} }`.

> **THEOREM P1 (`eps1 <= eps0` ==> P1 for every `N`).** This holds in

| `nu` | `n` | `q` | `r` | target | `L_r(n,0)` | `eps1` | `eps0` | overlap |
|---|---|---|---|---|---|---|---|---|
| 3 | 2 | 2 | 2/3 | 1.442249570 | 1.441380164 | 2.683e-05 | 1.698e-02 | **633x** |
| 4 | 3 | 2 | 1/2 | 2.000000000 | 1.969854379 | 9.958e-04 | 1.469e-02 | **14.7x** |
| 5 | 4 | 2 | 2/5 | 2.626527804 | 2.526617550 | 3.850e-03 | 8.442e-03 | **2.19x** |
| 3 | 2 | 5/2 | 5/6 | 1.200936955 | 1.200485938 | 1.322e-04 | 2.429e-04 | **1.84x** |

> so **P1 is proved, for every number of atoms, at `nu = 3, 4, 5` with plain weight decay `q = 2`
> (and at `nu = 3`, `q = 5/2`)** — the paper's whole range. Bisecting in `r` gives the full window
> per `nu` (`p1b_proof.py`, `FINAL_RUNS/P1B/rstar.json`):
>
> | `nu` | `n` | two-branch closes for | i.e. `q <=` | adversary's `r*` |
> |---|---|---|---|---|
> | 3 | 2 | `r <= 0.84333` | 2.5300 | `~0.83` |
> | 4 | 3 | `r <= 0.62678` | 2.5071 | `~0.62` |
> | 5 | 4 | `r <= 0.43487` | 2.1744 | not reported |
> | 6 | 5 | fails already at `q = 2` | — | not reported |
>
> The `n = 2` and `n = 3` windows agree with `ADVERSARY.md` §6 to the digits they report — an
> independent confirmation from a differently-assembled argument. `nu = 5` (`n = 4`) is new here.

> **COROLLARY (what this does for the paper).** `GLOBAL_OPT3` Theorem D is
> `gamma_q <= L_r(n)^{-1/r} B Gamma_{nu/q}(p)` for any junk-free gated chain, with
> `F(n,q) = ((n+1)^{1-r}/L_r(n))^{1/r}` the only slack: `1.000905 / 1.030841 / 1.101809` at
> `nu = 3/4/5`. P1 replaces `L_r(n)` by `(n+1)^{1-r}`, so **`F = 1`**:
>
>     at nu = 3, 4, 5 and q = 2, every gated chain of any width, with arbitrary (impure) neurons and
>     any number of frequencies, whose same-|k| junk vanishes, satisfies
>         gamma_q <= (n+1)^{-(nu-q)/q} B Gamma_{nu/q}(p)     exactly,
>     with the (n+1)-neuron common-phase block attaining it.
>
> Agent 2's gap **G1** and agent 3's ranked item 1 close. Junk (**P3**) is then the only gap left
> between the present theorems and (T). *(Caveat inherited from `ADVERSARY.md` §5: the block attains
> the target only when frequency `k`'s neurons are scaled by `b_k^{1/q}` — the share-weighted block.
> That is a statement about the attaining construction, not about the upper bound proved here.)*

> **PROPOSITION A1 (the dual bound on aligned configurations; proved, every `n`).**
> If all `c_l` share a phase — equivalently `S = |A|`, equivalently `z = c/A` is a **probability**
> multi-affine cubature rule — then for every `d <= n`
>
>     max_l x_l  <=  A_d := (1 - m_d)/(2^d - m_d) ,
>     m_d = min over T^d of Re prod_{j=1}^d (1 + w_j)  =  0, -1/2, -2, -5.545085, -13.5, -30.842421 .
>
> `A_1 = 1/2` and `A_2 = 1/3` are **exactly `1/(n+1)`**, and a bound `x_l <= kappa` for every `l`
> gives `sum_l x_l^r = sum_l x_l x_l^{r-1} >= (sum_l x_l) kappa^{r-1}` (because `r - 1 < 0`), which
> with `sum_l x_l = 1` and `kappa = 1/(n+1)` is exactly `(n+1)^{1-r}`. So
> **P1 holds for every `r in (0,1]` on the aligned class at `n = 1` and `n = 2`.** `A_3 = 3/10 > 1/4`,
> so `n >= 3` needs a better certificate. *(Checked: over `~10^4` feasible probability cubature rules
> the largest weight is `0.500000000` at `n = 1`, `0.333333333` at `n = 2`, `0.250000000` at `n = 3` —
> the bound is attained at `n <= 2`, and the truth at `n = 3` looks like `1/4`, i.e. `A1` is not sharp
> there but `max_l x_l <= 1/(n+1)` appears to be TRUE for every `n`.)*

> **PROPOSITION A2 (stability; proved at `n = 2`).** For any (KL) system at `n = 2`, with
> `delta = S - 1`,
>
>     max_l x_l  <=  1/3  +  [ 8.5 delta + (3 sqrt3/2) sqrt(2(1+delta) delta) ] / 4.5 ,
>
> and consequently, since `x_l <= kappa` for all `l` gives `sum_l x_l^r >= S kappa^{r-1}`,
>
>     P1 holds at n = 2, for every r, whenever  S >= (3 max_l x_l)^{1-r} ,
>     which the display above guarantees as soon as  delta >= ~6 (1-r)^2 .
>
> *(Checked: `23 961` configurations in the reproducible block, `99 771` in a larger inline run, `N = 4..7`, **zero violations** in both, minimum slack `+0.0033` / `+0.0012`.)*
> Combined with the trivial branch `cost >= S^r` (which covers `delta >= 3^{(1-r)/r} - 1`) and with A1
> (`delta = 0`), **the `r -> 1` window at `n = 2` now survives only on the sliver
> `0 < sum_l |c_l| - |A| < ~6(1-r)^2`** — at `r = 0.99` that is `delta < 6.3e-4` against the trivial
> branch's `1.1e-2`.

---

## 2. Reproducing everything

From `heracles/`, `uv run python ...` (about 30 min on one core):

    uv run python p1b_proof.py  --cells 2:2 3:2 4:2 5:2 6:2 2:2.5 3:3 4:3 --trials 3000
    uv run python p1b_verify.py --trials 500
    uv run python p1b_align.py  --blocks MBA --trials 250
    uv run python p1b_search.py --blocks HGR --trials 3000
    uv run python p1b_search.py --blocks C   --starts 500      # slow: the 6-angle global search
    uv run python p1b_bound.py  --starts 200 --trials 4000     # the refined branch 1 at n = 2

`p1b_core.py` holds the shared definitions (moment matrix, rule, block, sign family).

---

## 3. The reformulation: (KL) is a complex cubature rule; P1 bounds its `l_r` cost

Normalise `A = 1` and put `z_l := c_l`. (KL) says `sum_l z_l prod_{j in T} v_lj = [T empty]`, and the
right-hand side is exactly the Haar integral of `prod_{j in T} z_j` over `T^n`. By linearity, for
**every** multi-affine `Lambda`,

    sum_l z_l Lambda(v_l)  =  integral over T^n of Lambda dHaar .

> **(KL) says precisely that the nodes `v_l` with complex weights `z_l` form a cubature rule for Haar
> measure on the torus that is exact on multi-affine polynomials.** P1 says the `l_r` cost of such a
> rule is at least `(n+1)^{1-r}`.

Readings worth keeping. At `r -> 0` this is a node-count statement (`N >= n+1`, agent 2's Theorem A);
at `r = 1` it is `sum_l |z_l| >= 1`, the total variation of the rule, trivial. P1 is the exact
interpolation. In Hill-number language, with `p_l = x_l/S`, P1 reads
`S^r (Hill_r(p))^{1-r} >= (n+1)^{1-r}`; the pure diversity statement `Hill_r >= n+1` is **false**
(that is what agent 3's refuted domination lemma already implies), so the total mass `S >= 1` must be
used — and both branches, and §7, do use it.

Two exact families (both verified to `0.0e+00`, `p1b_search.py` block H):

* the **block** — `n+1` nodes `v_lj = w^{-l}`, `w = exp(2 pi i/(n+1))`, weights `1/(n+1)`, cost
  `(n+1)^{1-r}`: the conjectured optimum;
* the **sign rule** — `2^n` nodes `v_lj = +-1` (all sign vectors), weights `2^{-n}`, cost
  `2^{n(1-r)}`: the classical product rule, `1.100642` of target at `n = 2, r = 2/3`, and the
  counterexample of §8.1.

---

## 4. Proof of Proposition B1

**Lemma 0.** Expand `prod_j (u_j - v_lj) = sum_T (-1)^{|T|}(prod_{j in T} v_lj)(prod_{j not in T} u_j)`
and exchange the sums: the coefficient of `prod_{j not in T} u_j` is `(-1)^{|T|} sum_l c_l prod_{j in T} v_lj`
on the left and `A[T empty]` on the right. The `2^n` monomials are linearly independent, so the
polynomial identity is equivalent to (KL). ∎ *(Checked to `5.2e-13`, `n = 1..5`; V1.)*

**Proof of B1.** Take `A = 1`. If `theta := 1 - 2^n E <= 0` the claim reduces to
`sum_l x_l^r >= sum_{l not in B} x_l^r >= (sum_{l not in B} x_l)^r = E^r`, by subadditivity of
`t -> t^r` for `0 < r <= 1`. So assume `theta > 0`.

Fix `l0 in B` and a **bijection** `sigma` from the `n` slots onto the `n` atoms of `B \ {l0}`. Put
`u_j := v_{sigma(j), j} in T`. Every `m in B\{l0}` equals `sigma(j)` for exactly one `j`, so its
`j`-th factor vanishes; Lemma 0 then gives, using `|prod_j u_j| = 1` and `d^{(j)}_{lm} <= 2`,

    1 <= x_{l0} prod_j d^{(j)}_{sigma(j),l0} + sum_{l not in B} x_l prod_j d^{(j)}_{sigma(j),l}
      <= x_{l0} prod_j d^{(j)}_{sigma(j),l0} + 2^n E ,
    hence  x_{l0} * prod_j d^{(j)}_{sigma(j), l0}  >=  theta  >  0 .                        (B1.1)

*(Checked: the minimum of the middle expression over all `(l0, sigma)` and all sampled systems is
`1.000000000` — tight, never violated; V2.)*

**Geometric mean over the `n!` bijections.** For fixed `l0` and slot `j`, each `m in B\{l0}` occurs as
`sigma(j) = m` in exactly `(n-1)!` bijections, so
`prod_sigma prod_j d^{(j)}_{sigma(j),l0} = prod_j prod_{m != l0} (d^{(j)}_{m,l0})^{(n-1)!}`, and the
`n!`-th root of (B1.1) gives

    x_{l0} >= theta * prod_j [ prod_{m in B, m != l0} d^{(j)}_{m,l0} ]^{-1/n} .              (B1.2)

*(The exponent `(n-1)!/n! = 1/n` is agent 2's `GLOBAL_OPT` §4 step (ii); checked as a log identity to
`5.5e-15` at `n = 1..5`; V3.)*

**Product over `l0`, then Hadamard.** Multiplying (B1.2) over the `n+1` choices of `l0`, the inner
double product runs over every **ordered** pair of distinct elements of `B`, i.e. it is `V_j^2`:

    prod_{l in B} x_l  >=  theta^{n+1} prod_j V_j^{-2/n} .

`V_j` is the modulus of the Vandermonde determinant of `n+1` unimodular numbers, whose matrix has rows
of norm `sqrt(n+1)`, so Hadamard gives `V_j <= (n+1)^{(n+1)/2)}` and `prod_j V_j^{2/n} <= (n+1)^{n+1}`:

    prod_{l in B} x_l  >=  theta^{n+1} / (n+1)^{n+1} .                                       (B1.3)

*(Checked: `max_j V_j/(n+1)^{(n+1)/2} = 0.99999999` — Hadamard attained, never exceeded; and the
minimum of the ratio in (B1.3) is `1.0000`/`1.007`/`1.134`/`2.46` at `n = 2/3/4/5`; V4, V5.)*

**AM–GM plus the tail's own cost.** `sum_{l in B} x_l^r >= (n+1)(prod_{l in B} x_l)^{r/(n+1)} >=
(n+1)^{1-r} theta^r`, and `sum_{l not in B} x_l^r >= E^r`. Adding gives B1. ∎

*Equality.* At `E = 0` this is the Key Lemma and its equality case (all `|c_l|` equal, each slot's
phases a rotated copy of the `(n+1)`-st roots of unity: the common-phase block) is inherited; for
`0 < E <= eps0` the inequality is strict.

*Why this is more than `(C')`.* The first term alone is agent 1's `(C')` with the tail as junk, and on
its own it is useless: linear relief against an `O(1)` loss. What makes it bite is that the tail is not
free — it is part of the cost being bounded. Every earlier pass priced junk against the **margin**
(`Lemma Q`, `O(J/p)`); pricing the tail against the **cost** is the trade that works. This is the point
the adversary agent reached independently.

---

## 5. Branch 2, and the correction to the peeling constants

**The ladder.** For a set `S` of `h <= n` atoms, choose disjoint slot groups `J_1..J_h`,
`sum_i |J_i| <= n`, and take `Lambda = prod_i Lambda_{v_{l_i}}(z_{J_i})` with
`Lambda_v(z_J) = [1 - prod_{j in J}(1 + z_j conj(v_j))/2]/(1 - 2^{-|J|})`. This is multi-affine on
`T^n`, has constant coefficient `1`, vanishes at each `v_{l_i}`, and has sup-norm
`prod_i 1/kappa_{|J_i|}`. Pairing with (KL) and taking `S` = the `h` largest atoms gives (PEEL_h).

**The correction.** `GLOBAL_OPT3` §3 defined `D_h(n)` as a max over compositions with
`sum_i d_i = n`. But `Lambda_v(z_J)` may be built on a **proper** subset `J` of the slots: it is still
multi-affine in all `n` variables, still normalised, still vanishes at `v` (only `z_J = v_J` is needed),
and its sup-norm is unchanged. So the correct definition is `sum_i d_i <= n`. Since
`kappa_1 < kappa_2 < kappa_3 > kappa_4 > ...`, replacing any part `d_i >= 4` by `3` both raises
`kappa_{d_i}` and frees a slot, so **only `d_i in {1,2,3}` is ever optimal** and every `D_h(n)` is an
exact algebraic number in `kappa_1 = 1/2`, `kappa_2 = 3 sqrt3/8`, `kappa_3 = 49/72`:

| `n` | `D_0` | `D_1` | `D_2` | `D_3` | `D_4` | `D_5` |
|---|---|---|---|---|---|---|
| 2 | 1 | 0.649519053 | 0.25 | | | |
| 3 | 1 | 0.680555556 | 0.324759526 | 0.125 | | |
| 4 | 1 | **0.680555556** | 0.421875 | 0.162379763 | 0.0625 | |
| 5 | 1 | **0.680555556** | 0.442033800 | 0.210937500 | 0.081189882 | 0.03125 |

(bold = improved on agent 3's `kappa_4 = 0.678905`, `kappa_5 = 0.666728`). Consequence: agent 3's
rigour caveat for `n >= 4` disappears — every constant is exact. The effect on `L_r` is small, and at
`n = 4` it is nil (the `h = 1` constraint is slack at the minimising vertex: `L_r(4,0) = 2.526617550`
with either set of constants); at `n = 5` it lifts `L_r(5,0)` from `3.105667` to `3.106914`, which
moves `eps1` from `1.7813e-2` to `1.7543e-2` — still short of `eps0 = 4.44e-3`.
*(The ladder itself checked directly on sampled (KL) systems: `min (tail mass)/D_h = 1.0000001`
(`n=2`), `1.000061` (`n=3`), `1.0015` (`n=4`), `1.0041` (`n=5`) — tight, never violated; V7.)*

**The programme.** `L_r(n, delta) := min{ sum_{l=1}^N x_l^r : x_1 >= ... >= x_N >= 0,
sum_{l>h} x_l >= D_h(n) (h = 0..n), sum_{l>n+1} x_l >= delta }`. The objective is concave, the feasible
set a pointed polyhedron whose recession directions increase the objective, so the minimum is at a
**vertex** and every vertex is enumerated exactly. The value is independent of `N` for `N >= n+2`
(checked to 9 decimals for `N = n+2 .. n+6`). `L_r(n,0)` reproduces agent 3's numbers where the
constants are unchanged, and its minimising vertex uses exactly `n+1` atoms with **no tail** — which is
why a forced tail is so expensive (`delta^r` bought against `O(delta)` saved) and why `eps1` is tiny.

**An `N`-free variant, for the record.** Splitting `sum_l x_l^r >= sum_{l<=n+1} x_l^r + E^r` and noting
that `(x_1..x_{n+1})` obeys `sum_{h<l<=n+1} x_l >= D_h(n) - E` gives `cost >= M_r(n,E) + E^r` with
`M_r` an `(n+1)`-variable vertex programme; combining it with B1 at each tail size,

    cost  >=  Phi(E) := max( M_r(n,E), (n+1)^{1-r}(1 - 2^n E)_+^r ) + E^r ,

which is `>= (n+1)^{1-r}` on the whole of `[0, eps1]` in the four closing cells (minimum margin
attained at `E = 0`, where it is exactly `0` — the block). `Phi` does **not** by itself reach `nu = 6`
(margin `-3.6e-2`); the adversary's `beta`-refined tail price is what is needed there.

---

## 6. Where it fails, and that the failure is in the proof

Failing cells, with the uncovered window `(eps0, eps1)` searched directly for a counterexample
(`p1b_search.py` block G: block perturbations with `N = 2^n` atoms, so the rule is unique):

| `nu` | `n` | `q` | `r` | `eps1` | `eps0` | worst `cost/target` **inside the window** |
|---|---|---|---|---|---|---|
| 6 | 5 | 2 | 1/3 | 1.754e-02 | 4.437e-03 | 1.407241 |
| 4 | 3 | 3 | 3/4 | 7.111e-03 | 1.928e-04 | 1.002040 |
| 5 | 4 | 3 | 3/5 | 3.703e-02 | 6.965e-04 | 1.029127 |

P1 is comfortably true there. **The single lossy step is `prod_j d^{(j)}_{sigma(j),l} <= 2^n` for a
tail atom**: the kill point is made of `B`'s own phases, and a tail atom can be antipodal to all of them
only if `B`'s atoms coincide — which sends `V_j` to `0` and destroys Hadamard. So the true constant is
strictly better than `2^n`, with a quantitative trade-off. At `n = 2` I computed the refined version
exactly (`p1b_bound.py`): keeping individual distances gives

    prod_{m in B} x_m >= Gplus^{1/2}/(V_1 V_2),
    Gplus = prod over the 6 ordered pairs (m1 != m2) of (1 - E A^{(1)}_{m1} A^{(2)}_{m2})_+ ,

`A^{(j)}_m = |v_mj - v_{tail,j}|`, hence `cost >= 3 maxW(E)^{-r/6} + E^r` with `W = (V_1V_2)^2/Gplus`
purely geometric. A 6-angle global search gives `maxW(0.104) = 3077` against `729` at `E = 0` and pushes
`eps0` from `0.0170` to `0.115` — a factor **6.8**, i.e. an effective tail constant near `2.1` instead of
`4`. This is precisely the adversary's wish-list item (ii), now quantified. **The obstruction is the
certificate, not the mathematics**: `maxW` is a `2n`-angle global maximum with only a `~5%` margin, and
the summed relaxations that would make it analytic are *worse* than `2^n` at large `n` (I checked:
`Prod(1-t_i) >= 1 - Sum t_i` costs a factor `(n+1)!` in the dilution and gives an effective constant
`217` at `n = 5`). Either a closed-form Hadamard-type trade-off for the individual `A^{(j)}_m` — the
available one, `V_j prod_m A^{(j)}_m <= (n+2)^{(n+2)/2}`, bounds their *product*, which is the wrong
combination — or an interval branch-and-bound in `2n` variables.

---

## 7. The `r -> 1` window, by duality

**The reduction.** `cost >= (sum_l x_l)^r = S^r`, so P1 is immediate whenever
`S >= (n+1)^{(1-r)/r}`. That threshold tends to `1` as `r -> 1`, so near `r = 1` **only configurations
with `S` close to `|A|` survive** — i.e. with the `c_l` almost aligned in phase. At `S = |A|` exactly,
the triangle inequality is an equality, every `c_l` has the same phase, and after a global rotation
`x_l = c_l >= 0` with `sum_l x_l = 1`: the rule is a genuine **probability** cubature rule.

**Proposition A1.** Take the kernel at the heaviest atom `v_0` on a slot set `J`, `|J| = d`:
`K_J(z) = prod_{j in J}(1 + z_j conj(v_{0j}))`. Its Haar mean is `1` and `K_J(v_0) = 2^d`, so the
exact pairing of §3 gives `sum_l x_l K_J(v_l) = 1`. Taking real parts, using `x_l >= 0` and
`Re K_J >= m_d` on the torus,

    1 = x_0 2^d + sum_{l != 0} x_l Re K_J(v_l)  >=  x_0 2^d + (1 - x_0) m_d ,

so `x_0 (2^d - m_d) <= 1 - m_d`, i.e. `max_l x_l <= A_d = (1 - m_d)/(2^d - m_d)`. ∎

`m_d = min over a in [-pi/2, pi/2]^d of 2^d prod_j cos(a_j) cos(sum_j a_j)`, whose global minimum
coincides with the diagonal one in every case computed (`d = 1..6`, `40` multistarts each):

| `d` | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| `m_d` | `0` | `-1/2` | `-2` | `-5.545085` | `-27/2` | `-30.842421` |
| `A_d` | **1/2** | **1/3** | 0.3 | 0.303786 | 0.318681 | 0.335740 |
| `1/(d+1)` | 1/2 | 1/3 | 0.25 | 0.2 | 0.166667 | 0.142857 |

So `A_d = 1/(n+1)` **exactly at `n = 1, 2`**. Since `r - 1 < 0`, a uniform bound `x_l <= kappa`
gives `sum_l x_l^r = sum_l x_l x_l^{r-1} >= (sum_l x_l) kappa^{r-1}`; with `sum_l x_l = 1` and
`kappa = 1/(n+1)` that is exactly `(n+1)^{1-r}` (no Schur-concavity needed):

> **P1 holds for every `r in (0,1]` on the aligned class at `n = 1` and `n = 2`.**

At `n >= 3` the single-kernel certificate gives only `3/10 > 1/4`. But the LP search over probability
cubature rules (maximise one weight subject to `x >= 0`, `sum x = 1` and all non-constant moments zero,
then search over nodes) returns a maximum weight of exactly `0.250000000` at `n = 3` — so
**`max_l x_l <= 1/(n+1)` looks true for every `n`, and a better test polynomial (or a different dual
object) is what is missing.** That conjecture would give the aligned case of P1 at one stroke, for every
`n` and every `r`.

**Proposition A2 (stability at `n = 2`).** Keep the same pairing but let the phases move. Rotate
globally so that `sum_l c_l = 1`; write `c_l = x_l e^{i theta_l}`, so `sum_l x_l cos theta_l = 1`,
`sum_l x_l sin theta_l = 0` and `sum_l x_l (1 - cos theta_l) = delta`. With `y_l = x_l cos theta_l`
(so `sum_l y_l = 1` and `sum_l y_l^- <= delta`), `K_0 = 4` real, `Re K >= -1/2` and
`|Im K| <= 3 sqrt3/2` on `T^2`:

    1 = sum_l y_l Re K_l - sum_l x_l sin(theta_l) Im K_l
      >= 4 y_0 - (1/2)(S - x_0) - 4 delta - (3 sqrt3/2) sum_l x_l |sin theta_l| ,
    y_0 >= x_0 - delta ,   sum_l x_l |sin theta_l| <= sqrt(2 S delta) ,

which rearranges to the display in §1 and, for small `delta`, to
`max_l x_l <= 1/3 + sqrt(2/3) sqrt(delta) + O(delta)`. Then `x_l <= kappa` for all `l` gives
`sum_l x_l^r >= S kappa^{r-1}` directly, so P1 holds as soon as `S >= (3 max_l x_l)^{1-r}`, i.e. as soon
as `delta >= ~6(1-r)^2`. *(Checked: `23 961` + `99 771` configurations, `N = 4..7`, zero violations, min slack `+0.0033`
and `+0.0012`; the two constants `min Re K = -1/2` and `max |Im K| = 3 sqrt3/2 = 2.598076` confirmed on
a `4 x 10^5` torus sample to `2e-5`.)*

Covered `delta` at `n = 2`, per `r`:

| `r` | trivial branch `delta >=` | A2 branch `delta >=` | uncovered sliver |
|---|---|---|---|
| 0.85 | 0.2139 | 0.1959 | `(0, 0.196)` |
| 0.90 | 0.1298 | 0.0815 | `(0, 0.082)` |
| 0.95 | 0.0595 | 0.0184 | `(0, 0.018)` |
| 0.99 | 0.0112 | 0.00063 | `(0, 6.3e-4)` |

with `delta = 0` itself covered by A1. **So the adversary's open window `r in (r*, 1)` is now squeezed
from the other side: inside it, only configurations that are almost exactly phase-aligned
(`0 < sum_l|c_l| - |A| < ~6(1-r)^2`) are left.** Closing it needs either `A_3`-style sharpness at the
level of the *whole profile* rather than the maximum weight, or a phase-stability estimate that is
`O(delta)` rather than `O(sqrt delta)` — the `sqrt` comes from the genuinely first-order term
`sum_l x_l sin(theta_l) Im K_l`, which the exact constraint `sum_l x_l sin theta_l = 0` centres but does
not kill.

---

## 8. Refutations

### 8.1 P1-prod is FALSE (exact counterexample)

The natural one-line route is `(P1-prod)`: *some `n+1` atoms satisfy
`prod |c_l| >= (|A|/(n+1))^{n+1}`*, which gives P1 at once by AM–GM on those atoms. It is **false for
every `n >= 2`**: the sign rule (`N = 2^n` nodes `v_lj = +-1`, weights `2^{-n}`, verified to `0.0e+00`)
has `max over (n+1)-subsets of prod|c_l| = 2^{-n(n+1)} < (n+1)^{-(n+1)}`, since `2^n > n+1`. At
`n = 2`: `1/64 < 1/27`; at `n = 3`: `2.44e-4 < 3.91e-3`. *Plain reading: a rule may spread its mass over
more than `n+1` atoms, and then no `n+1` of them carry a large product — P1 must be a statement about
the whole profile.* (Same moral as agent 3's refutation of `(C')`, on a far simpler witness.)

### 8.2 The multiplicative structure of (KL) is essential

Replace the four constraint rows `(e, a, b, a*b)` of the `n = 2`, `N = 4` system by `(e, a, b, d)` with
`d` an unrelated unimodular vector — same number of rows, same count bound. Over `10^5` random
configurations the minimum of `sum_l |c_l|^r / 3^{1-r}` is `1.034` for the multi-affine rows and
**`0.925`** for the free third row. So **no proof of P1 can work from "four unimodular constraint rows"
alone**: the slot/product structure that Lemma 0 and the bijection kills exploit is load-bearing. This
closes off the obvious generalisation and explains why linear-algebraic arguments blind to the torus
factorisation keep failing.

### 8.3 Where my assigned starting angle went

My brief proposed the true Lagrangian dual for `r < 1` and agent 2's statement "`Lambda_mu == 1`". I
derived the stationarity system and record it, because it weakens agent 2's condition usefully. At a
minimiser of `sum_l |c_l|^r` over (KL) configurations with **all** `c_l != 0` (`v` and `c` both free),
there is a multi-affine `Lambda` with

    Lambda(v_l) = (r/2)|c_l|^{r-2} conj(c_l)   (stationarity in c),
    Im[ c_l v_lj conj(beta_j(v_{l,-j})) ] = 0  (stationarity in the phases),   Lambda = alpha_j + z_j beta_j,

and pairing with (KL) gives `lambda_empty = (r/2) sum_l |c_l|^r`, so
`|Lambda~(v_l)| = |c_l|^{r-1}/cost` for the normalised `Lambda~`. Hence:

> it is enough that **`|Lambda~(v_l)|` be the same at every atom** — not that `Lambda~ == 1`. Then all
> `|c_l|` are equal, `= x` say, and `N x >= |sum_l c_l| = 1` gives `cost = N x^r >= N^{1-r} >=
> (n+1)^{1-r}` by agent 2's Theorem A.

This is strictly weaker than agent 2's condition and is consistent with the sign rule, which is a
genuine non-optimal critical point with `|c_l| = 2^{-n}` and `|Lambda~(v_l)| = 1`. I did not prove it.
The route's real difficulty is that the global minimiser is a **boundary** point (`c_l = 0` for the
extra atoms), so the interior stationarity system never sees it — which is exactly what the two-branch
argument sidesteps, and why I switched. The duality that *did* pay off is §7's, where the test
polynomial is fixed in advance (the reproducing kernel) rather than being the unknown optimal dual.

Also abandoned: agent 4's Theorem O already forbids any single averaged test measure, and the
reverse-Hölder combination over subsets of atoms that my brief suggested reduces, after the product
over subsets, to a *geometric* mean of the kill bounds — which is exactly what B1's step (B1.2) already
does optimally (`max >= geometric mean` is the only available step, and it is tight at the block).

### 8.4 Previously recorded dead routes

Unchanged and not re-attempted: Morwani-style dual certificates; the domination lemma; the capped-mass
conjecture `(C')`; gauge symmetrisation and grid rotation; Mahler / Gram / Carathéodory–Fejér; the
last-slot induction. Note that **B1 is not an instance of Theorem O's forbidden family**: it uses `n+1`
*different* test polynomials, one tight at each atom of `B`, combined multiplicatively — precisely the
"non-uniform, several `Lambda` tight at different atoms" shape agent 4's §4 said a proof would need.

---

## 9. What remains open, ranked

1. **`nu >= 6` and `q >= 3`.** §6: refine `2^n` by the `V_j`-vs-`A^{(j)}` trade-off. `n = 2` shows a
   factor `6.8` is available; `4` would close `nu = 6` at `q = 2`. Needs a certificate for a `2n`-angle
   maximum with a `~5%` margin. Merging with the adversary's `beta`-refined tail price first is the
   cheapest next move.
2. **`max_l x_l <= 1/(n+1)` for probability cubature rules, every `n`.** Proved at `n <= 2` (A1) and
   measured exactly at `n = 3`. It would give the aligned case of P1 for every `n` and every `r` at one
   stroke — and with A2-style stability, the whole `r -> 1` window.
3. **An `O(delta)` phase-stability estimate** to replace A2's `O(sqrt delta)`, which would close the
   remaining sliver at `n = 2` outright.
4. **P3 (junk).** Untouched here; with P1 in hand at `nu = 3,4,5`, it is the only gap left before (T).

---

## 10. Numerical record (float64, local CPU, no pods, `heracles/FINAL_RUNS/P1B/`)

| script | what it does | result |
|---|---|---|
| `p1b_core.py` | shared definitions: moment matrix, rule `z = M^{-1}e_0`, block, sign family | block rule `1/(n+1)`, sign rule `2^{-n}`, exact, `n = 1..5` |
| `p1b_proof.py` | `D_h(n)` (corrected); `L_r(n,delta)` and `M_r(n,eps)` by exact vertex enumeration; the two thresholds; `N`-stability; `Phi` | tables of §1, §5; `L_r(n,0)` matches agent 3 to 9 decimals; `L_r` independent of `N` for `N = n+2..n+6` |
| `p1b_verify.py` | links V1–V8 of the proof on random and block-perturbed (KL) systems, `n = 1..5`, `2 800` systems | **zero violations**. Identities: V1 `<= 1.4e-12`, V3 `<= 2.0e-14`. Inequalities (min ratio, must be `>= 1`): V4 `0.99999951` (Hadamard, from below), V2 `1.0000000`, V5 `1.0000` / `1.0072` / `1.1641` / `2.4591`, V6 `1.00028` / `1.00847` / `1.10060` / `1.42104`, V7 `1.0000000` / `1.000061` / `1.001465` / `1.004131`, V8 (P1 itself) `1.00041` / `1.01530` / `1.12127` / `1.42083` at `n = 2/3/4/5` |
| `p1b_align.py` | `m_d` and `A_d` (multistart + diagonal); Proposition A1 against LP-extremal probability cubature rules | `A_d` table of §7; max weight `0.5 / 0.333333333 / 0.25` at `n = 1/2/3`, A1 never violated |
| `p1b_search.py` H | the sign rule as an exact cubature family, `n = 1..5` | weights `2^{-n}` to `0.0e+00`; **P1-prod refuted** for every `n >= 2` |
| `p1b_search.py` G | the uncovered windows of the failing cells | worst `cost/target` in window `1.407 / 1.002 / 1.029` — P1 true there |
| `p1b_search.py` R | multi-affine rows vs a free third row, `10^5` configurations | `1.034` vs `0.925`: the product structure is essential |
| `p1b_search.py` C | `n = 2`, `N = 4` global search over the 6 gauge-fixed angles, `r = 2/3, 1/2, 5/6` | `min cost/target = 1.000000000000` in all three, minimiser `(1/3, 1/3, 1/3, 0)` — agent 4's `1.0000009` was optimiser noise (as `ADVERSARY.md` §3 also found) |
| `p1b_entropy.py` | `a*(n)` by direct sampling (`120 000` / `16 000` systems) and the near-block scaling of both routes | `a* >= 0.13224` (`n=2`), `0.12351` (`n=3`) against thresholds `5.383` / `1.679`; `max_l x_l - 1/(n+1)` flat over `sqrt(delta)` at `~0.222` |
| `p1b_bound.py` | refined branch 1 at `n = 2`: chain verification and `maxW(E)` by 6-angle global search | chain never violated (4000 systems); `eps0` improves `0.0170 -> 0.115` |
| `p1b_align.py` A | Proposition A2 on (KL) systems at `n = 2`, `N = 4..7` | `23 961` configurations, **zero violations**, min slack `+0.00332`; a larger inline run (`99 771` configurations) gave min slack `+0.00120`; `min Re K = -0.4999988` (`-1/2`) and `max\|Im K\| = 2.598056` (`3 sqrt3/2 = 2.598076`) confirmed on `4 x 10^5` torus points |

---

## 11. ADDENDUM (after P1A / P1C): the `r -> 1` window reduces to ONE constant, with a 40x margin

P1A reduced the window to an `r`-free inequality; P1C certified the two-variable programme. Three
findings here, all new, two of them precise dead ends.

### 11.1 The window is ONE extremal constant, and the needed bound is 40x looser than the truth

P1A's identity `sum_l x_l^r = S^r exp((1-r) H_r) >= S^r exp((1-r) H)` (Renyi `>=` Shannon for
`r <= 1`), `p = x/S`, `S = sum_l |c_l|`, makes the criterion `H(p) + (r/(1-r)) log S >= log(n+1)`
sufficient for P1 at that `r`. **That is exactly one number.** Define

    a*(n)  :=  sup over (KL) systems with S > 1 of  [ log(n+1) - H(x/S) ] / log S .

The criterion holds **exactly** for `r >= a*/(1 + a*)`, so, since the certified arguments already
cover `r <= r*(n)`, the whole window closes as soon as `a*(n) <= r*(n)/(1 - r*(n))`:

| `n` | `r*` (certified) | need `a* <=` | measured `a*` (two independent runs) | **margin** | argmax configuration |
|---|---|---|---|---|---|
| 2 | 0.84333 | **5.383** | `0.13224` (120 000 systems) / `0.12654` (36 000) | **40.7x** / 42.5x | `S = 2.8`, moduli `(1.29, 1.27, 0.26, 0.014)` |
| 3 | 0.62678 | **1.679** | `0.12351` (16 000 systems) / `0.13607` (24 000) | **13.6x** / 12.3x | `S = 9.3`–`49`, two near-cancelling atoms |

(`a*` is measured as a **lower** bound by sampling, so the margin statement is "the truth sits ~40x
below the threshold that has to be proved", not a certificate. The point is that the certificate
needed is 40x looser than the ones P1A/P1C had to build.)

*Plain reading: the `r -> 1` end does not need a sharp estimate at all. The certified two-variable
programmes had `~2%` margin; this one has `4000%` at `n = 2`. A crude certificate would do.* The
extremal configurations are **not** near the block: they are the near-cancelling-pair family (the same
family that refutes the domination lemma), far out at `S = 2.8` and `S = 49`.

One branch is already free: `H >= 0` gives `a <= log(n+1)/log S`, so **everything with
`S >= (n+1)^{(1-r)/r}` is covered outright** (`S >= 1.222` at `n = 2, r = 0.843`). Feeding
Proposition A2 (§7) in for the rest covers `delta = S - 1 >= 0.1472`. **The entire residue of P1 at
`n = 2` is therefore the single statement**

    H(x/S)  >=  log 3  -  5.383 * log S        for  0 < S - 1 < 0.1472 ,

against a measured slope of `0.132`. Since `H` is *quadratically* flat at the uniform point, any
lemma of the form `sum_l (x_l - 1/3)^2 <= C (S-1)` finishes it with room to spare.

### 11.2 DEAD END: the max-weight route is capped, and the `sqrt` is real

My Proposition A2 bounds `max_l x_l <= 1/3 + c sqrt(delta)`. I checked whether the square root is an
artefact of the proof. It is **not**: maximising `max_l x_l - 1/3` over (KL) systems at `n = 2` with
`delta` pinned in dyadic bands gives

| `delta` band | `1e-6` | `1e-5` | `1e-4` | `1e-3` | `1e-2` |
|---|---|---|---|---|---|
| `(max_l x_l - 1/3)/sqrt(delta)` | 0.2214 | 0.2237 | 0.2269 | 0.2372 | 0.2725 |
| `(max_l x_l - 1/3)/delta` | 124.6 | 39.8 | 12.8 | 4.22 | 1.53 |
| `(log 3 - H)/log S` | 0.1100 | 0.1093 | 0.1095 | 0.1150 | 0.1219 |

(third row: the entropy route's deficit is **linear** in `log S` with slope `~0.11`, against the
`5.383` that may be spent — the 40x margin of §11.1, seen locally)

flat in the first row, divergent in the second: the growth is genuinely `Theta(sqrt(delta))` with true
constant `~0.222` (my proved `0.8165` is loose by `3.7x`, but the *rate* is right). The reason is
structural — the block is a critical point of `S` along the 2-parameter family of junk-free 3-atom
configurations, so the moduli move linearly while `delta` moves quadratically.

**Consequence.** Bounding `H` by `-log(max_l p_l)` is a *first-order* bound while `H` is
*quadratically* flat at uniform, so the max-weight route loses a square root and **cannot close the
window even with the sharp constant**: it needs `delta >= 0.444 (1-r)^2` where the truth needs nothing.
This is the exact analogue, on my side of the argument, of P1C's finding that the `2^n` in
`R_S >= 1 - 2^n tau` is attained. *Both first-order routes into the `r -> 1` corner are now closed;
what works there is second-order (entropy curvature), which is why §11.1 has 40x of margin.*

### 11.3 A distinction worth keeping about the `2^n` constant

P1C reports that `R_S >= 1 - 2^n tau` is **attained** (single tail atom probed at `u_j = conj(v_j)`),
so sharpening *that* estimate cannot help. That is a statement about Theorem B's `R_S`, whose probe
`u` ranges freely over the torus. My branch 1 (§4) never uses a free probe: its probe points are
`u_j = v_{sigma(j),j}`, the phases of `B`'s own atoms, and the chain takes a geometric mean over all
`n!` bijections and a product over all `n+1` choices of `l_0`. A tail atom would have to be antipodal
to *different* members of `B` in every slot simultaneously, which collides with the Vandermonde. That
is why the refined `n = 2` computation of §6 does improve the constant from `4` to `~2.1`
(`eps0: 0.0170 -> 0.115`) — verified on 4 000 systems with no violation. **So the two statements are
compatible and the constant is attained only in the free-probe form.** It still does not close the
window (§11.2 explains why nothing first-order can), so this is a correction of scope, not a new route.

### 11.4 What to do next, precisely

1. Prove `sum_l (x_l - 1/(n+1))^2 <= C (S - 1)` near the block, or any bound giving
   `log(n+1) - H(x/S) <= a log S` with `a <= r*/(1-r*)`. The margin is `40x` (`n = 2`) and `13.6x`
   (`n = 3`), so a crude constant suffices — and a rigorous upper bound on `a*` would need only an
   interval sweep at that tolerance, far cheaper than the certificates P1A/P1C had to build.
2. `max_l x_l <= 1/(n+1)` for probability cubature rules at `n >= 3` (§7) remains the clean
   `S = 1` endpoint; proved here at `n <= 2`.
