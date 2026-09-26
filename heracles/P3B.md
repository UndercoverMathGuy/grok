# P3 by the dual measure: the premise is true, the route is provably dead, and the gap has an exact name

Fifth pass, problem **P3**: remove hypothesis (H) from agent 3's Theorem D by showing that a **global
maximiser** of `gamma_q` can be taken with no same-frequency junk. Continues `LEMMA_ATTEMPT.md`
(agent 1), `GLOBAL_OPT.md` (agent 2), `GLOBAL_OPT3.md` (agent 3), `GLOBAL_OPT4.md` (agent 4). Target,
unchanged, for the gated chain and `2 <= q <= nu`:

    gamma_q(theta)  <=  (n+1)^{-(nu-q)/q} * B * Gamma_{nu/q}(p),      B = 2 (2 p (n+1))^{-nu/2}.    (T)

**Status: P3 OPEN. The assigned route is REFUTED — not "not yet made to work", but proved impossible,
with the loss computed exactly.** Three things are new and proved:

1. **Theorem P3-1 (convex hull).** For every `q in [2, nu]`, the convex hull of the set of margin
   vectors reachable under the `q`-penalty budget is *exactly* the set reachable under the `nu`-penalty
   budget. Splitting a neuron is free for a convex relaxation, and free splitting is what `q = nu` means.
2. **Theorem P3-2 (duality barrier).** Consequently, for **every** probability measure `mu` on
   (row, wrong class) pairs, the Lagrangian value `V(mu) := max{ E_mu[Phi] : P_q <= 1 }` is
   *independent of `q`* and is at least `gamma_nu^* = B Gamma_1(p)`, the Morwani (`q = nu`) answer.
   The dual-measure / KKT / min-max family — the route I was assigned, and the natural home for
   "average the dual, not the network" — **cannot prove (T) for any `q < nu`**, missing by at least
   `(n+1)^{(nu-q)/q} * Gamma_1/Gamma_{nu/q}` (`2.449` at `p=5`, `3.000` at `p=7`, `nu=3`, `q=2`).
   *The whole gap between plain weight decay and `L_{2,nu}` is a duality gap of the max-margin problem.*
3. **The premise the route needed is TRUE and useless.** At the unrestricted whole-network maximisers
   (`p = 5, 7`, `n = 2`, `q = 2, 2.5`): the same-`|k|` junk is `<= 9e-16` (so (H) holds exactly at the
   maximiser — P3's own claim, confirmed), **every** (row, wrong class) pair is active, and the
   **uniform** measure on the active set is a valid KKT multiplier (residual `<= 2e-15`). So a symmetric
   optimal `mu*` exists, exactly as the brief hoped. It certifies nothing, because `theta*` is a
   *stationary* point of the Lagrangian and not its *maximiser*: the Lagrangian's maximiser is a single
   junk-carrying neuron, at value `B Gamma_1`.

Two further exact statements are recorded because they close doors:

4. **Lemma G (per-neuron gauge torus)** and **Lemma A (the task's translation group)** — both preserve
   `P_q` and every clean coefficient `G_k` exactly and rotate the junk by a character. Lemma A also
   preserves the margin exactly. Both are verified to `5e-15`.
5. **Corollary A2.** The characters by which translations rotate the junk already exhaust `Z_p^*`, so a
   translation average that kills the junk must be the **uniform** average over all `p` translates, at
   cost factor `p^{1-q/nu}` — worse than gauge symmetrisation's `(n+1)^{1-q/nu}`, and (T) would need `1`.
   This is the inherited "symmetrisation is circular" statement in its exact, `p`-dependent form.

All float64, local CPU, **no pods, $0**. Scripts `heracles/p3b_*.py`, outputs
`heracles/FINAL_RUNS/P3B/`. Nothing here contradicts (T); every numerical probe of (T) again failed
to beat it.

---

## 0. Symbols, with a plain reading of each

Inherited symbols keep their meaning. Repeated here so this file is self-contained.

| symbol | meaning | plain reading |
|---|---|---|
| `p` | odd prime | modulus; the task is `a_1 + ... + a_n = c (mod p)` on all `p^n` rows |
| `n`, `nu = n+1` | operands, homogeneity degree | scaling every weight by `s` scales every logit by `s^nu` |
| `q in [2, nu]`, `r = q/nu in (0,1]` | penalty exponent, and its ratio | `q = 2` is plain weight decay, `q = nu` is Morwani's `L_{2,nu}`; `r = 1` means splitting one feature over several neurons is free |
| `e_ij, w_i in R^p` | neuron `i`'s `j`-th operand embedding, its readout | `z(a,c) = sum_i w_i(c) prod_j e_ij(a_j)` |
| `theta_i = (e_i1,..,e_in,w_i)` | neuron `i` | `\|\|theta_i\|\|^2 = sum_j \|\|e_ij\|\|^2 + \|\|w_i\|\|^2` |
| `P_q = sum_i \|\|theta_i\|\|^q` | the penalty | what weight decay prices |
| `s(a) = sum_j a_j mod p` | the correct class of row `a` | |
| `Phi(a,c) = z(a,s(a)) - z(a,c)` | the gap at the pair `(a,c)`, `c != s(a)` | how safely row `a` beats the wrong class `c` |
| `margin = min_{(a,c)} Phi(a,c)` | the margin | worst-case gap over the whole table |
| `gamma_q = margin / P_q^{nu/q}` | normalised margin | scale-invariant "margin per unit of weight" |
| `H = {1,..,(p-1)/2}` | positive frequencies | `k` and `p-k` are the same cosine |
| `ehat_ij(k) = sum_a e_ij(a) omega^{-ka}`, `omega = e^{2 pi i/p}` | unnormalised DFT | |
| `c_ik = prod_j ehat_ij(k) * conj(what_i(k))` | neuron `i`'s **atom** at frequency `k` | the complex clean amplitude it puts on `k` |
| `v_ijk = conj(ehat_ij(k))/ehat_ij(k)` | its **phase datum** at slot `j` | unit modulus |
| `G_k = sum_i c_ik` | the **clean** Fourier coefficient at `k` | what the level-set average of the logit table sees |
| `J_{k,T} = sum_i c_ik prod_{j in T} v_ijk`, `T` a non-empty subset of `{1..n}` | **same-`\|k\|` junk** | the logit tensor's coefficient with the operand signs in `T` flipped |
| `(H)` | `J_{k,T} = 0` for every `k in H` and every `T != empty` | "no same-frequency junk"; hypothesis of Theorem D |
| `eta_k = sqrt(sum_{T != empty} \|J_{k,T}\|^2) / \|G_k\|` | relative junk at frequency `k` | "what fraction of this feature's amplitude is junk" |
| `Gamma_s(p)` | `max { min_{D != 0} sum_k x_k (1 - cos 2 pi k D/p) : x >= 0, sum_k x_k^{1/s} <= 1 }` | the free-amplitude programme, `s = nu/q`; `Gamma_1` is the flat (Morwani) endpoint |
| `P = { (a,c) : a in Z_p^n, c != s(a) }` | the pair set | `\|P\| = p^n (p-1)` |
| `mu` | a probability measure on `P` | a **dual measure**: a weighting of the constraints |
| `V_q(mu) = max { E_mu[Phi(theta)] : P_q(theta) <= 1 }` | the Lagrangian value at `mu` | the best any network can do against that weighting |
| `sigma(mu)` | injective (spectral) norm of the tensor `T_mu` of §2 | the best a **single neuron** can do against `mu` |
| `gamma_q^*` | `max { gamma_q(theta) }` over all networks, all widths | what (T) is about |

Two conventions: `Gamma_s <= Gamma_1` for every `s >= 1` (if `x >= 0` and `sum_k x_k^{1/s} <= 1` then
each `x_k <= 1`, so `x_k <= x_k^{1/s}` and `sum_k x_k <= 1`: the `l_{1/s}` ball sits inside the `l_1`
ball). And `(n+1)^{-(nu-q)/q} = (n+1)^{-(1-r)/r}`.

---

## 1. Exact statements

> **THEOREM P3-1 (the convex hull of the `q`-budget is the `nu`-budget; proved, every `n`, `p`,
> `q in [2,nu]`).**
> For a network `theta` write `Phi(theta) in R^P` for its vector of pair gaps, `Phi(theta)_{(a,c)} =
> Phi(a,c;theta)`, and
>
>     K_q := { Phi(theta) : P_q(theta) <= 1 }   (all widths allowed).
>
> Then
>
>     conv(K_q)  =  K_nu   for every q in [2, nu].
>
> *Plain reading: mixing two networks in the sense a convex relaxation understands — take a `lambda`
> share of one and a `1-lambda` share of the other — is exactly "put both of them in the net, side by
> side, rescaled". Under the `L_{2,nu}` penalty that side-by-side costs nothing extra, so `K_nu` is
> already convex and mixing cannot leave it. Under any `q < nu` it costs the splitting factor — and
> that cost is invisible to the convex hull. So convexifying the plain-weight-decay problem turns it
> into Morwani's problem.*

> **THEOREM P3-2 (duality barrier; proved, every `n`, `p`, `q in [2,nu]`, every `mu`).**
> With `T_mu` the `(n+1)`-way tensor of §2 and `sigma(mu)` its injective norm,
>
>     (i)   V_q(mu)  =  nu^{-nu/2} sigma(mu)   for every q in [2, nu]   -- INDEPENDENT of q,
>           attained by a network of width 1 (one neuron, its nu factors of equal norm);
>     (ii)  V_q(mu)  >=  gamma_nu^*  =  B Gamma_1(p)   for every mu;
>     (iii) inf over mu of V_q(mu)  =  gamma_nu^*  =  B Gamma_1(p).
>
> Since weak duality gives `gamma_q^* <= V_q(mu)` for every `mu`, and since
> `gamma_nu^* = B Gamma_1 >= B Gamma_{nu/q} = (n+1)^{(nu-q)/q} x [the right-hand side of (T)]`,
> **no choice of dual measure can prove (T) for any `q < nu`.** The shortfall is at least
>
>     (n+1)^{(nu-q)/q} * Gamma_1(p) / Gamma_{nu/q}(p).
>
> *Plain reading: a dual measure replaces "the worst row" by "an average row". Junk is exactly what an
> average row cannot see (it has mean zero over the rows — that is the junk lemma L1). So the relaxed
> problem is always won by the configuration that is cheapest per unit of clean amplitude and pays
> nothing for junk: a single neuron. One neuron is precisely the configuration (T) has to rule out, and
> the relaxation hands it the win at every `q`.*

> **COROLLARY P3-3 (scope of the refutation).** Any argument of the shape
> "`gamma_q^* <= max_theta [ a fixed non-negative weighting of the pair gaps ] / P_q^{nu/q}`" is dead,
> including: averaging the dual measure over the task's symmetry group; taking `mu` supported on the
> maximiser's active set (the KKT multiplier); any finite family `min_i V(mu_i)`; and "stationarity
> against a symmetric `mu*` forces junk to be gradient-orthogonal". Only arguments that keep the `min`
> over rows *non-linearly* — i.e. that use at least two rows in a way a single average cannot express —
> can reach (T). The known proof does this through the per-frequency cost lemma (Key Lemma /
> Theorem E), which is why it needs (H).

> **LEMMA G (the per-neuron gauge torus; proved, verified to `6e-16`).**
> For `phi in R` let `Omega_phi` be the orthogonal map of `R^p` that rotates the Fourier coefficient at
> every `k in H` by `e^{i phi}` (and fixes the DC component). Define
> `R_phi^{(i)}` to act on neuron `i` by `e_ij -> Omega_phi e_ij` (all `j`) and `w_i -> Omega_{n phi} w_i`,
> and as the identity on the other neurons. Then
>
>     P_q is EXACTLY invariant;   every atom c_ik is invariant, hence every G_k is invariant;
>     neuron i's junk contribution  c_ik prod_{j in T} v_ijk  ->  e^{-2 i |T| phi} * itself.
>
> The margin is **not** invariant. *Plain reading: there is an `m`-torus of networks (one angle per
> neuron) with identical cost and identical clean spectrum, on which only the junk phases move.*

> **LEMMA A (the task's translation group; proved, verified to `5e-15`).**
> For `tau in Z_p` let `T_tau` act by `e_ij(.) -> e_ij(. - tau)` (all `i`, `j`) and
> `w_i(.) -> w_i(. - n tau)`. This is the network-side action of the data symmetry
> `(a_1..a_n, c) -> (a_1+tau, .., a_n+tau, c + n tau)`. Then
>
>     margin, P_q and every atom c_ik are EXACTLY invariant,   and   J_{k,T} -> omega^{2 k tau |T|} J_{k,T}.
>
> *Plain reading: the same junk rotation as the gauge, but this time the margin is preserved too — the
> whole `p`-element orbit consists of genuinely equally good networks whose junk phases differ.*

> **COROLLARY A2 (translation averaging is circular, with an exact constant).**
> Let `g_tau >= 0`, `sum_tau g_tau = 1`, and build `eta` = the direct sum of the `p` translates
> `T_tau theta` with neuron scales `s_tau = g_tau^{1/nu}`. Then `G_k(eta) = G_k(theta)`,
> `margin(eta) >= margin(theta)`, `P_q(eta) = C(g) P_q(theta)` with `C(g) = sum_tau g_tau^r`, and the
> junk is multiplied by `ghat(-2k|T|)`. Killing all same-`|k|` junk needs `ghat` to vanish on
> `U := { +- 2 k |T| mod p : k in H, 1 <= |T| <= n }`. But already at `|T| = 1`, `{+-2k : k in H} =
> Z_p \ {0}`, so `U = Z_p \ {0}` for every `n >= 1` and `g` must be the uniform measure. Hence
>
>     C_min = p^{1 - q/nu},     while (T) would need C <= 1  (and C >= 1 always, with equality only
>     for a point mass, because sum_tau g^r >= (sum_tau g)^r = 1 for r <= 1).
>
> Measured (`p3b_routes.py` A2, `n=2`, `q=2`): `1.709976, 1.912931, 2.223980, 2.351335, 2.571282,
> 2.843867` at `p = 5, 7, 11, 13, 17, 23` — i.e. exactly `p^{1/3}`, against the gauge route's
> `3^{1/3} = 1.442250` and the required `1`.

> **PROPOSITION P3-4 (why P3 cannot be purely algebraic).** Suppose one could show: *for every network
> `theta` there is a network `eta` with `P_q(eta) <= P_q(theta)`, `G_k(eta) = G_k(theta)` for all `k`,
> and `(H)`.* Then (T) would follow at once, because Theorem D's cost step uses nothing else. **No such
> `eta` exists.** Take `theta` = one balanced neuron pure at a single frequency, with atom `c != 0`:
> then `P_q(theta) = (2 nu/p)^{q/2} |c|^r` exactly (the AM-GM and Parseval steps of (2.2) are tight) and
> `G_k(theta) = c`, while any junk-free `eta` with the same clean spectrum has
> `P_q(eta) >= L_r(n) (2 nu/p)^{q/2} |c|^r > P_q(theta)` by Theorem E. *So the margin must enter: P3 is
> not a statement about cost and clean spectrum alone, and any proof of it must use the fact that the
> object is a maximiser.* Combined with Theorem P3-2, which says the maximiser property cannot be used
> through a dual measure, this is a sharp fence around what is left.

---

## 2. Proof of Theorems P3-1 and P3-2

Everything rests on one structural fact: **`Phi` is additive over neurons and multilinear inside each
neuron.** Explicitly,

    Phi(a,c; theta) = sum_i F_{a,c}(e_i1, ..., e_in, w_i),
    F_{a,c}(x_1,...,x_n,y) = ( y(s(a)) - y(c) ) * prod_{j=1}^n x_j(a_j),           ... (2.1)

an `(n+1)`-linear form. For a probability measure `mu` on `P` put `M_mu := sum_{(a,c)} mu(a,c) F_{a,c}`,
whose coefficient tensor is

    T_mu[b_1,...,b_n, c']  =  m(b) [c' = s(b)]  -  mu(b, c'),      m(b) := sum_{c} mu(b,c),   ... (2.2)

a `p^{n+1}` array (`b` runs over rows, `c'` over classes), and let

    sigma(mu) := max { M_mu(x_1,..,x_n,y) : ||x_j||_2 = ||y||_2 = 1 }                          ... (2.3)

be its injective (spectral) norm — *the best a single neuron of unit factors can do against `mu`*.

**Proof of Theorem P3-2 (i).** *Upper bound.* By (2.1) and (2.3), for any `theta`,

    E_mu[Phi(theta)] = sum_i M_mu(theta_i) <= sigma(mu) sum_i prod_j ||x_ij||
                    <= sigma(mu) sum_i ( ||theta_i||^2 / nu )^{nu/2}                 [AM-GM on nu factors]
                     = sigma(mu) nu^{-nu/2} sum_i ||theta_i||^nu
                    <= sigma(mu) nu^{-nu/2} ( sum_i ||theta_i||^q )^{nu/q}           [l_nu <= l_q, nu >= q]
                    <= sigma(mu) nu^{-nu/2}.

*Attainment.* Take width 1 and the maximising unit vectors of (2.3) rescaled to norm `nu^{-1/2}` each;
then `||theta_1||^2 = 1`, so `P_q = 1` for every `q`, and the value is `nu^{-nu/2} sigma(mu)`. ∎

Note that both the bound and the maximiser are the same for every `q in [2,nu]`: the only `q`-dependent
step, `l_nu <= l_q`, is an equality at width 1. **`V_q(mu)` does not depend on `q`.**

**Proof of Theorem P3-1.** `K_q subset K_nu`: `P_q(theta) <= 1` implies `P_nu(theta) <= 1` by
`l_nu <= l_q`. `K_nu` is convex: given `theta^1, theta^2` with `P_nu <= 1` and `lambda in [0,1]`, let
`eta` be the direct sum of `lambda^{1/nu} theta^1` and `(1-lambda)^{1/nu} theta^2`. Because `Phi` is
additive over neurons and homogeneous of degree `nu`,

    Phi(eta) = lambda Phi(theta^1) + (1-lambda) Phi(theta^2),
    P_nu(eta) = lambda P_nu(theta^1) + (1-lambda) P_nu(theta^2) <= 1,

so `K_nu` is closed under convex combinations; hence `conv(K_q) subset K_nu`. Conversely let
`P_nu(theta) <= 1` with neurons `theta_i`, and put `lambda_i := ||theta_i||^nu` (so `sum_i lambda_i <= 1`).
The width-1 network `theta_i / ||theta_i||` has `P_q = 1`, i.e. lies in `K_q`, and its `Phi`-vector is
`Phi(theta_i)/||theta_i||^nu`. Then `Phi(theta) = sum_i Phi(theta_i) = sum_i lambda_i *
[Phi(theta_i)/||theta_i||^nu]`, a convex combination (top up with the zero network, which is in `K_q`).
So `K_nu subset conv(K_q)`. ∎

**Proof of Theorem P3-2 (ii), (iii).** By (i), `V_q(mu) = V_nu(mu) = max{ E_mu[Phi] : P_nu(theta) <= 1 }`.
For any `theta` in that set, `E_mu[Phi(theta)] >= min_{(a,c)} Phi(a,c;theta) = margin(theta)`; taking the
max over the set gives `V_q(mu) >= gamma_nu^*`, which is (ii). For (iii), LP duality over the convex
compact `conv(K_q) = K_nu` (Theorem P3-1) gives

    inf_mu max_{v in K_nu} <mu, v>  =  max_{v in K_nu} min_{(a,c)} v_{a,c}  =  gamma_nu^*,

the first equality being the minimax theorem for a bilinear form on (simplex) x (convex compact). ∎

And `gamma_nu^* = B Gamma_1(p)` is the inherited `q = nu` statement: the per-frequency AM-GM bound is
two-sided at `r = 1` (`THEORY_nsum` §3/§8, agent 3's Theorem D with `F = 1`), and the block attains it.

**Why this kills the assigned route, said once more.** The brief's plan was: take a maximiser `theta*`,
take a symmetry-averaged optimal dual measure `mu*`, and read off from stationarity that junk is
useless. Steps 1–2 are fine (§3 shows `mu*` uniform is available). Step 3 cannot work, because
stationarity of `theta*` against `mu*` says `theta*` is a **critical point** of
`theta -> E_{mu*}[Phi(theta)]` on the `P_q`-sphere, and a critical point is not a maximiser: the
maximiser of that Lagrangian is a single neuron at value `B Gamma_1`, which exceeds the target by the
whole factor (T) is trying to prove. There is no way to repair this by a better `mu`, by symmetrising,
or by restricting the support of `mu` — Theorem P3-2 quantifies over all `mu`.

---

## 3. What the maximiser actually looks like (`p3b_max.py`)

Unrestricted whole-network max-margin solutions (raw tensors `E (n,m,p)`, `W (p,m)`, SLSQP with
analytic Jacobians, starts = the constructed block on every ladder support + noise + random; this is
agent 2's `gopt_full.py` search, kept rather than discarded). `n = 2`, width `m = 9`.

| `p` | `q` | `gamma_q`/target | frequencies used | `max_k eta_k` | active pairs | distinct rows | KKT residual | uniform-`mu` residual | deficit `gamma_clean(D) - min_a Phi_D(a)` |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 2   | 1.000000000000 | {1,2}   | 8.52e-16 | 100/100 | 100% | 9.4e-16 | 1.8e-15 | `[5e-18, 1e-17]` |
| 5 | 2.5 | 1.000000000000 | {1,2}   | 8.54e-16 | 100/100 | 100% | 8.3e-16 | 1.8e-15 | `[9e-18, 2e-17]` |
| 7 | 2   | 1.000000000000 | {1,2,3} | 8.97e-16 | 294/294 | 100% | 1.0e-15 | 1.8e-15 | `[2e-18, 5e-18]` |
| 7 | 2.5 | 1.000000000000 | {1,2,3} | 8.59e-16 | 294/294 | 100% | 9.8e-16 | 2.0e-15 | `[4e-18, 1e-17]` |

Reading, line by line.

* **(H) holds exactly at the maximiser.** `eta_k`, the relative same-`|k|` junk, is at machine epsilon
  on every used frequency. This is the empirical content of P3 and it is clean: the maximiser really is
  junk-free, so Theorem D really does apply to it — we just cannot *prove* the hypothesis.
* **Every pair is active.** Not "many": all `p^n (p-1)` of them. Equivalently `Phi(a,c)` is the same
  number for every row and every wrong class, which is the junk-free signature (the deficit column is
  at `1e-17`, i.e. zero). The maximiser's logit table is a pure function of `s(a) - c`.
* **A symmetric dual measure exists.** Since all pairs are active, the uniform measure on `P` is a
  candidate multiplier, and it satisfies the KKT equation
  `sum_{(a,c)} mu(a,c) grad Phi(a,c) = lambda grad P_q`, `lambda = nu*margin/(q*P_q)`, to `2e-15`.
  (A non-negative-least-squares multiplier found independently has residual `1e-15`; its `a`-marginal
  is not uniform — the dual set is a polytope of positive dimension and `nnls` returns a sparse vertex.
  That is why the *uniform* measure is tested directly.)
* So the brief's premise is confirmed and the conclusion still does not follow: with this very `mu`,
  `V(mu) = B Gamma_1`, which at `p = 5, q = 2` is `1.5214515486e-02` against the maximiser's
  `gamma_q = 6.211300e-03`. **The maximiser realises 40.8% of its own Lagrangian's value.** The missing
  factor is `2.449490 = sqrt 6 = 3^{1/2} * Gamma_1/Gamma_{3/2}`.

**Lemma G verified** at the same maximisers (`phi = 0.7231`, neuron 1): `|Delta P_q|/P_q <= 2.2e-16`,
`max_k |Delta G_k|/max|G_k| <= 3.7e-16`, and the junk phase law `-> e^{-2i|T|phi}` holds to `5.7e-16`.

---

## 4. Numerical record of the barrier (`p3b_dual.py`)

`n = 2` (`nu = 3`). `sigma(mu)` by alternating maximisation (80 restarts); the primal
`max { E_mu[Phi] : P_q <= 1 }` by unconstrained L-BFGS on the scale-invariant ratio
`E_mu[Phi]/P_q^{nu/q}` over raw `(E, W)` of width `nu+2`, seeded with `sigma`'s argmax plus 6 random
starts; `inf_mu` by projected subgradient descent on the simplex (legitimate: `mu -> sigma(mu)` is a max
of linear functionals of `mu`, hence convex).

**(B1) `V_q(mu) = nu^{-nu/2} sigma(mu)`, independent of `q`** — 18/18 cells exact:

| `p` | `mu` | `q=2` | `q=2.5` | `q=3` | `nu^{-nu/2} sigma(mu)` |
|---|---|---|---|---|---|
| 5 | uniform | 1.5214515486e-02 | 1.5214515486e-02 | 1.5214515486e-02 | 1.5214515486e-02 |
| 5 | random  | 1.9564983698e-02 | 1.9564983698e-02 | 1.9564983698e-02 | 1.9564983698e-02 |
| 5 | spiky   | 2.2741958705e-02 | 2.2741958705e-02 | 2.2741958705e-02 | 2.2741958705e-02 |
| 7 | uniform | 8.5724083312e-03 | 8.5724083312e-03 | 8.5724083312e-03 | 8.5724083312e-03 |
| 7 | random  | 9.7265307892e-03 | 9.7265307892e-03 | 9.7265307892e-03 | 9.7265307892e-03 |
| 7 | spiky   | 1.3196355446e-02 | 1.3196355446e-02 | 1.3196355446e-02 | 1.3196355446e-02 |

(ratio primal/tensor `= 1.00000000` in every cell; the primal never exceeds the bound, as the proof
requires, and the maximiser found is always of width 1.)

**(B2) `inf_mu V(mu) = B Gamma_1`, and how far that is above (T)'s target:**

| `p` | `inf_mu V(mu)` | `B Gamma_1` | ratio | `q` | `Gamma_{nu/q}` | unrestricted `B Gamma_{nu/q}` | target of (T) | `inf_mu V / target` |
|---|---|---|---|---|---|---|---|---|
| 5 | 1.5214515486e-02 | 1.5214515486e-02 | **1.00000000** | 2 | 0.883883 | 1.075829e-02 | 6.211300e-03 | **2.449490** |
| 5 | | | | 2.5 | 1.088188 | 1.324501e-02 | 1.063232e-02 | 1.430969 |
| 5 | | | | 3 = `nu` | 1.250000 | 1.521452e-02 | 1.521452e-02 | **1.000000** |
| 7 | 8.5724083312e-03 | 8.5724083312e-03 | **1.00000000** | 2 | 0.673575 | 4.949282e-03 | 2.857469e-03 | **3.000000** |
| 7 | | | | 2.5 | 0.936532 | 6.881428e-03 | 5.524009e-03 | 1.551846 |
| 7 | | | | 3 = `nu` | 1.166667 | 8.572408e-03 | 8.572408e-03 | **1.000000** |

The last column is the barrier: a dual-measure proof would need it `<= 1`. It equals `1` exactly at
`q = nu` — where the relaxation is tight, as Theorem P3-1 says it must be — and is `2.449` / `3.000`
at `q = 2`. Note the shortfall exceeds the naive circularity factor `(n+1)^{(nu-q)/q} = 1.7321`,
by the extra `Gamma_1/Gamma_{nu/q}` (`1.414` at `p=5`, `1.732` at `p=7`): the relaxation loses the
splitting factor **and** the sparsity of the programme, because a dense spectrum is also something only
the average can afford.

**Consistency check.** Over 200 random `mu` (Dirichlet with random concentration, plus power tilts),
`min_mu V(mu)/(B Gamma_1) = 1.084` (`p=5`), `1.036` (`p=7`): never below `1`, as Theorem P3-2 (ii)
requires.

---

## 5. The route the barrier leaves open, measured (`p3b_routes.py` B1)

Theorem P3-2 says the `min` over rows must be used non-linearly. The one place in the inherited work
where junk is priced **exactly** rather than bounded is agent 2's Theorem B: at a single frequency,
`sum_l |c_l|^r >= (n+1)^{1-r} R^r` with `R := min over the torus T^n of Re h`, where
`h(u) = sum_l c_l prod_j (1 + u_j v_lj)` is the frequency's generating polynomial. That statement needs
no junk-free hypothesis at all. Extending it from `K = 1` to `K >= 2` requires exactly one new thing:

    (SIM)   min over rows a of  sum_k w_kD Re h^{(k)}( u^{(k)}(a) )   ~=   sum_k w_kD R_k ,
            u^{(k)}(a)_j = omega^{-2 k a_j},   w_kD = 1 - cos(2 pi k D/p),

i.e. **one row must nearly realise every frequency's own torus minimum simultaneously**. The left side
is `>= ` the right side always (each factor is `>= R_k` and `w_kD >= 0`), and `margin <= 2^{-n} min_D`
of the left side (average `Phi_D` over `+-D` to kill the imaginary part). So the only question is the
size of the **simultaneity defect**

    defect := [ min_D min_a sum_k w_kD Re h^{(k)} ] - [ min_D sum_k w_kD R_k ] ,

measured below in units of the junk-free scale `min_D sum_k w_kD |G_k|`. Configurations: `n = 2`,
`N = n+1 = 3` atoms per frequency, the junk-free block perturbed by i.i.d. phase and modulus noise of
scale `js` (so `js = 0` is exactly junk-free); torus minima on a `180^2` grid plus a Nelder-Mead polish;
5 trials per cell, median and worst reported.

| `p` | `K` | `js=0` | `js=0.1` | `js=0.3` | `js=0.6` | `js=1.0` |
|---|---|---|---|---|---|---|
| 11 | 1 | 0.0000 | 0.0036 | 0.0049 | 0.0134 | 0.4634 |
| 11 | 2 | 0.0000 | 0.0177 | 0.0604 | 0.0949 | 0.5062 |
| 11 | 3 | 0.0000 | 0.0274 | 0.0830 | 0.4678 | 0.5579 |
| 13 | 2 | 0.0000 | 0.0269 | 0.0414 | 0.3252 | 0.9738 |
| 13 | 3 | 0.0000 | 0.0429 | 0.1292 | 0.6342 | 1.0504 |
| 17 | 1 | 0.0000 | 0.0009 | 0.0019 | 0.1447 | 0.3203 |
| 17 | 2 | 0.0000 | 0.0310 | 0.0271 | 0.1155 | 0.4535 |
| 17 | 3 | 0.0000 | 0.0156 | 0.0605 | 0.3606 | 1.4560 |
| 23 | 1 | 0.0000 | 0.0007 | 0.0023 | 0.2600 | 0.4185 |
| 23 | 2 | 0.0000 | 0.0069 | 0.0335 | 0.2654 | 2.6117 |
| 23 | 3 | 0.0000 | 0.0231 | 0.0612 | 0.1963 | 0.2712 |

(medians over 5 trials; worst-case values, and the `p = 13, K = 1` and `p = 23` rows, are in
`FINAL_RUNS/P3B/routes_B.json` and `routes_B.log`. Worst cases run about 2-3x the medians at
`js <= 0.3` and up to 4x at `js = 1`.)

Reading: **(SIM) is exact in the junk-free limit (`0.0000` at `js = 0`, to fp64) and degrades
gracefully up to moderate junk (a few per cent at `js <= 0.3`), then breaks down.** The defect grows
with `K` at fixed junk and shrinks with `p` at fixed `K`, which is what the underlying equidistribution
heuristic predicts: the rows reach `(u^{(k)})_{k in S}` only along a cyclic subgroup of `(T^n)^K` of
order `p^n`, whose covering radius is `~ p^{-1/K}`. So route (SIM) is live **in the same regime agent 4
already covers** (junk below about a third of a feature's amplitude, `GLOBAL_OPT4` §6.1) and does not
close the top of the junk range. It is, however, the only route I found that is not blocked by
Theorem P3-2, since it keeps the `min` over rows in an essential, non-averaged way.

---

## 6. Approaches tried here that FAILED, and exactly where

**F5-1. The assigned route (symmetric dual measure + stationarity). REFUTED by Theorem P3-2.** Not "the
symmetry does not exist" — it does (§3: the uniform measure is a valid multiplier, to `2e-15`). The
failure is that the Lagrangian's value at *any* `mu` is the `q = nu` answer. Two sub-ideas died with it:
"stationarity forces junk to be gradient-orthogonal" (true but vacuous: it says `theta*` is critical,
not optimal, for the Lagrangian) and "the saddle-point value with symmetric `mu*` equals the clean-part
value" (true — with `mu` uniform in `a`, `E_mu[Phi]` *is* a function of the clean spectrum alone — but
that value is `B Gamma_{nu/q}`, the already-known unrestricted bound, not the target).

**F5-2. "Find a cost- and clean-spectrum-preserving junk killer." REFUTED by Proposition P3-4.** This
was the cleanest possible formalisation of P3 that avoids pricing junk — and it is false for the
single junk-carrying neuron. It is worth stating because it shows the margin is *logically necessary*
in any proof of P3, which in turn is what makes Theorem P3-2 fatal rather than merely inconvenient.

**F5-3. Translation averaging (Lemma A + weighted direct sum). DEAD, with the exact constant
`p^{1-q/nu}`.** Better-looking than gauge symmetrisation at first sight, because Lemma A preserves the
margin exactly, whereas the gauge does not. Worse in the end: the junk characters `+-2k|T|` already
exhaust `Z_p^*` at `|T| = 1`, so no non-uniform weighting can kill the junk, and the uniform one costs
`p^{1-r}` against gauge's `(n+1)^{1-r}` and the required `1` (Corollary A2). *The general obstruction,
which applies to every averaging scheme of this shape: normalising the clean part to `sum_tau g_tau = 1`
forces `C(g) = sum_tau g_tau^r >= 1` for `r <= 1`, with equality only for a point mass — i.e. only when
no averaging happens. Averaging always costs, and (T) has no room for any cost at all.*

**F5-4. Gauge symmetrisation with unequal amplitudes.** Same computation as F5-3 with `U` replaced by
`{2|T| phi}`: the requirement is `sum_l g_l e^{-2i|T|phi_l} = 0` for `|T| = 1..n`, which the `(n+1)`-th
roots of unity satisfy with `g` uniform, cost `(n+1)^{1-r}`. Any other feasible `g` still has
`C(g) >= 1` with equality only at a point mass, so no reweighting rescues it. Confirms agents 2-4's
"gauge symmetrisation is exactly circular" and shows the circularity is not an artefact of the equal
weights.

**F5-5. `min_i V(mu_i)` over several dual measures, and `mu` adapted to the maximiser's active set.**
Both are still of the form "`gamma_q^* <= V(mu)` for some `mu`", so both are covered by Theorem P3-2;
the active-set version is exactly the one measured in §3 (`V = B Gamma_1`, target realised at 40.8%).

**Previously recorded dead routes**, not re-attempted and unchanged: Morwani-style dual certificate;
the domination lemma; the capped-mass conjecture `(C')`; peeling with sup-norms (`F3-2`'s cap); one
averaged test measure (agent 4's Theorem O); product test measures; the last-slot induction;
Mahler/geometric mean; Gram / Caratheodory-Fejer. Theorem P3-2 is the network-level analogue of agent
4's Theorem O — where Theorem O rules out one averaged *test polynomial* for the atom lemma, P3-2 rules
out one averaged *row measure* for the whole problem, and for the same underlying reason: averaging
cannot see the cost of splitting.

---

## 7. What remains open, ranked

1. **P3 itself.** After Theorem P3-2 the shape of any proof is constrained: it must use the `min` over
   rows non-linearly, and it must use the margin (Proposition P3-4). The only concrete candidate is
   (SIM) of §5, which would upgrade agent 2's Theorem B from `K = 1` to all `K` and thereby remove (H)
   outright. What is missing is a *uniform* lower bound on how well the row lattice
   `{ (omega^{-2k a_j})_{j,k} : a in Z_p^n }` approximates an arbitrary point of `(T^n)^K` — a covering
   -radius statement for a cyclic subgroup, with the covering radius entering multiplied by the
   Lipschitz constant of `sum_k w_kD Re h^{(k)}`. Measured defects (§5) say this is worth a few per cent
   at `js <= 0.3` and fails above; a proof would give (T) with junk exactly on agent 4's
   `eta <= eta_cross` window, and no further.
2. **Close the `q = nu` end differently.** Theorem P3-1 says the convexification of the `q`-problem is
   the `nu`-problem. The *only* way to beat it is a relaxation that is not convex in the
   direct-sum sense — e.g. one that bounds `P_q` from below by something strictly super-additive over
   neurons. The per-frequency cost lemma is precisely such a thing; this is a restatement of "the atom
   lemma is the whole content", and it says any further progress must go through S1/S2, not around them.
3. **For the paper.** Two quotable additions. (a) *The gap between plain weight decay and `L_{2,nu}` is
   a duality gap*: `conv{ Phi(theta) : P_q <= 1 } = { Phi(theta) : P_nu <= 1 }`, so every convex
   relaxation of the plain-weight-decay max-margin problem returns Morwani's dense answer, and the
   sparsity is invisible to it (Theorems P3-1, P3-2). This is a one-paragraph explanation of why the
   sparse result is hard that costs nothing and is checkable. (b) *The maximiser is junk-free and its
   whole table is flat*: at `p = 5, 7` every (row, wrong class) pair is active and the relative
   same-`|k|` junk is `< 1e-15` — the hypothesis (H) of Theorem D is not an artefact of the class, it is
   a property of the true optimum (§3).

---

## 8. Numerical record (all float64, local CPU, no pods, `heracles/FINAL_RUNS/P3B/`)

| script / block | what it does | result |
|---|---|---|
| `p3b_dual.py` B1 | `V_q(mu)` by primal L-BFGS on the scale-invariant ratio vs `nu^{-nu/2} sigma(mu)` by ALS; `p = 5, 7`, `q = 2, 2.5, 3`, `mu` uniform/random/spiky | ratio `1.00000000` in **18/18** cells; the primal maximiser is always a single neuron; `V_q` identical across `q` to 11 digits |
| `p3b_dual.py` B2 | `inf_mu V(mu)` by projected subgradient descent on the simplex (convex), vs `B Gamma_1` and vs (T)'s target | `inf_mu V / (B Gamma_1) = 1.00000000` at `p = 5` and `p = 7`; `inf_mu V /` target `= 2.449490 / 1.430969 / 1.000000` (`p=5`, `q=2/2.5/3`) and `3.000000 / 1.551846 / 1.000000` (`p=7`) |
| `p3b_dual.py` B2-check | `V(mu)/(B Gamma_1)` on 200 random `mu` per prime | min `1.084` (`p=5`), `1.036` (`p=7`) — never below `1` |
| `p3b_max.py` M1 | same-`\|k\|` junk `eta_k` at the unrestricted whole-network maximisers, `p = 5,7`, `q = 2, 2.5`, `m = 9` | `gamma_q`/target `= 1.000000000000` in 4/4 cells; `max_k eta_k <= 9.0e-16` — **(H) holds at the maximiser** |
| `p3b_max.py` M2 | active set; KKT multiplier by `nnls`; the **uniform** measure tested directly as a multiplier | all `p^n(p-1)` pairs active in 4/4 cells; `nnls` residual `<= 1.0e-15`; uniform-`mu` residual `<= 2.0e-15`; deficit `gamma_clean(D) - min_a Phi_D(a) <= 2e-17` |
| `p3b_max.py` M3 | Lemma G (per-neuron gauge): `P_q`, `G_k` invariance and the junk phase law `e^{-2i\|T\|phi}` | `dP_q <= 2.2e-16`, `dG <= 3.7e-16`, phase-law residual `<= 5.7e-16` |
| `p3b_routes.py` A1 | Lemma A (diagonal translation) on random nets, all `tau`, `p = 5, 7` | `d(margin) <= 2.5e-16`, `d(P_q) <= 1.3e-16`, `d(G_k) <= 4.7e-16`, junk-rotation-law residual `<= 4.4e-15` |
| `p3b_routes.py` A2 | `C_min = min{ sum_tau g^r : g >= 0, sum g = 1, ghat\|_U = 0 }`, `n = 2`, `q = 2` | `\|U\| = p-1` at every prime (so `g` must be uniform); `C_min = p^{1/3} = 1.710, 1.913, 2.224, 2.351, 2.571, 2.844` at `p = 5,7,11,13,17,23` vs the `1` that (T) needs |
| `p3b_routes.py` B1 | the simultaneity defect of (SIM), `p = 11,13,17,23`, `K = 1,2,3`, junk scale `0 .. 1`, 5 trials | `0.0000` at zero junk (exact); median `0.0007-0.031` at `js = 0.1`; `0.002-0.061` at `js = 0.3`; `0.013-0.36` at `js = 0.6`; `0.27-2.6` at `js = 1.0` |

One recorded false start, so nobody repeats it: the first version of the primal check in
`p3b_dual.py` packed the readout tensor transposed, so the L-BFGS maximiser wandered and the ratio
primal/tensor came out at `0.31-0.71` instead of `1`. That is an under-converged primal, not evidence
against Theorem P3-2 (i) — the inequality `V_q(mu) <= nu^{-nu/2} sigma(mu)` is proved, so only a ratio
**above** `1` would have been news. The corrected run gives `1.00000000` in 18/18 cells. Those buggy
JSONs were deleted; `dual_p5n2.json` is the corrected run.

Reproduce (about 25 minutes total on one core):

    uv run python heracles/p3b_dual.py --p 5 --n 2 --qs 2.0 2.5 3.0 --steps 250
    uv run python heracles/p3b_dual.py --p 7 --n 2 --qs 2.0 2.5 3.0 --steps 300
    uv run python heracles/p3b_max.py --ps 5 7 --n 2 --qs 2.0 2.5 --ms 9 --restarts 14 --tag _main
    uv run python heracles/p3b_routes.py --block A1A2 --ps 5 7 11 13 17 23 --trials 3 --tag _A
    uv run python heracles/p3b_routes.py --block B1 --ps 11 13 17 23 --trials 5 --tag _B
