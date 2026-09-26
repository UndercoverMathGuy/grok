# Is the (n+1)-block the GLOBAL max-margin network? — the peeling ladder, and why G1 and G2 were the same gap

Third pass. Continues `heracles/LEMMA_ATTEMPT.md` (agent 1) and `heracles/GLOBAL_OPT.md` (agent 2);
target, unchanged, for the gated chain and `2 <= q <= nu`:

    gamma_q(theta)  <=  (n+1)^{-(nu-q)/q} * B * Gamma_{nu/q}(p),      B = 2 (2 p (n+1))^{-nu/2}.      (T)

**Status: PARTIAL, with a strict advance — and (T) is now proved up to an explicit, small, computed
factor on a class that contains everything the paper needs except junk.** Nothing here contradicts (T).

What is new, in one paragraph. The whole per-frequency argument of agents 1 and 2 was carried out inside
the class of *pure* neurons (one frequency each), because their reduction to "atoms" needed purity.
It does not: §2 below shows that **every** gated chain, with arbitrary neurons and arbitrary width, hands
you the same finite atom system at every frequency — the atoms are just the neurons' Fourier data — so
the purity gap **G2 and the count gap G1 are literally the same gap**, namely "the Key Lemma for more
than `n+1` atoms". §3 then proves a new, elementary, unconditional **peeling ladder** for that lemma,
valid for every number of atoms, whose constant is within `0.06%` (nu 3), `1.5%` (nu 4) and `3.8%`
(nu 5) of the conjectured sharp one; feeding it through the chain gives (T) up to a factor
**1.000905** (`nu = 3, q = 2`), **1.030841** (`nu = 4, q = 2`), **1.101809** (`nu = 5, q = 2`),
for every network — any width, no purity, any support — whose *same-frequency* junk vanishes. §5 also
narrows G3: only `2^n - 1` junk coefficients per frequency, out of the `p^{n+1} - p` the logit tensor
has, ever enter the argument. Three further routes to the sharp constant are **refuted** with explicit
counterexamples (§6), including one that also caps the new method: the best factor any peeling ladder
can ever certify at `nu = 3, q = 2` is `1.000521`, against the `1.000905` proved here.

All float64, local CPU, **no pods, $0**. Scripts `heracles/gopt3_*.py`, outputs
`heracles/FINAL_RUNS/GOPT3/`. To reproduce everything (about 20 minutes total, single core):

    uv run python heracles/gopt3_const.py                         # the constants and the factor table
    uv run python heracles/gopt3_peel.py --ns 1 2 3 --trials 150   # (PEEL_h) + Theorem E on random systems
    uv run python heracles/gopt3_chain.py                          # the network reduction + Theorem D
    uv run python heracles/gopt3_dual.py --n 2 --ngrid 8 --G 32 --P 16 --rand 20   # refutation F3-2
    uv run python heracles/gopt3_cap.py --ns 1 2 --extra 2 --starts 120            # refutation F3-1

---

## 0. Symbols, with a plain reading of each

| symbol | meaning | plain reading |
|---|---|---|
| `p` | odd prime | the modulus; the task is `a_1 + ... + a_n = c (mod p)` on all `p^n` rows |
| `n`, `nu = n+1` | operands, homogeneity degree | how many numbers are added; scaling all weights by `s` scales logits by `s^nu` |
| `q in [2, nu]`, `r = q/nu in (0,1]` | penalty exponent and its ratio | `q = 2` is plain weight decay, `q = nu` is Morwani's `L_{2,nu}`; `r` measures how cheap it is to split one feature over several neurons (`r = 1` = free) |
| `e_ij, w_i in R^p` | neuron `i`'s `j`-th operand embedding, its readout | one row of one layer each; `z(a,c) = sum_i w_i(c) prod_j e_ij(a_j)` |
| `P_q = sum_i ||theta_i||^q` | the penalty | `||theta_i||^2 = sum_j ||e_ij||^2 + ||w_i||^2` |
| `gamma_q = margin / P_q^{nu/q}` | normalised margin | scale-invariant "margin per unit of weight" |
| `H = {1..(p-1)/2}` | positive frequencies | `k` and `p-k` are the same cosine |
| `ehat_ij(k) = sum_a e_ij(a) w^{-ka}`, `w = e^{2 pi i/p}` | unnormalised DFT | Parseval: `sum_{k in Z_p} \|ehat(k)\|^2 = p ||e||^2` |
| `c_ik = prod_j ehat_ij(k) conj(what_i(k))` | neuron `i`'s **atom** at frequency `k` | the (complex) amount of clean signal neuron `i` puts on frequency `k` |
| `v_ijk = conj(ehat_ij(k)) / ehat_ij(k)` | its **phase data** | unit modulus; records the phase of the `j`-th embedding at `k` |
| `G_k = sum_i c_ik` | clean Fourier coefficient | `g(D) := mean_a z(a, s(a)-D) = p^{-nu} sum_k G_k w^{kD}` |
| `J_{k,T} = sum_i c_ik prod_{j in T} v_ijk` | **same-\|k\| junk**, `T` a non-empty subset of `{1..n}` | the logit tensor's Fourier coefficient at multi-index `(+-k,...,+-k; -k)` with the signs of `T` flipped |
| `Gamma_s(p)` | free-amplitude programme, `s = nu/q` | `max { min_{D != 0} sum_k x_k (1 - cos 2 pi k D/p) : x >= 0, sum_k x_k^{1/s} <= 1 }` |
| `B = 2 (2 p (n+1))^{-nu/2}` | constant of the existing bound | pure bookkeeping (Parseval, AM-GM, the `+-k` factor 2) |
| `S_d`, `kappa_d`, `D_h(n)`, `L_r(n)` | the new constants | defined in §3–§4 |

---

## 1. Exact statements

> **THEOREM D (the main new result; proved, every `n >= 1`, every `2 <= q <= nu`, every `p`, every width,
> NO purity assumption, any number of frequencies).**
> Let `theta` be any gated-chain network on the full table, and suppose its **same-frequency junk
> vanishes**:
>
>     (H)     J_{k,T} = 0    for every k in H and every non-empty T subset of {1..n}.
>
> Then
>
>     gamma_q(theta)  <=  L_r(n)^{-1/r} * B * Gamma_{nu/q}(p)
>                     =   F(n,q) * (n+1)^{-(nu-q)/q} * B * Gamma_{nu/q}(p),
>     F(n,q) := ( (n+1)^{1-r} / L_r(n) )^{1/r} ,
>
> with `L_r(n)` the explicit ladder constant of §4. Numerically
>
> | `nu` | `q = 2` | `q = 2.5` | `q = 3` | `q = nu` |
> |---|---|---|---|---|
> | 3 (`n=2`) | **1.000905** | 1.000451 | 1 | 1 |
> | 4 (`n=3`) | **1.030841** | 1.022630 | 1.014757 | 1 |
> | 5 (`n=4`) | **1.101809** | 1.082346 | 1.063922 | 1 |
> | 6 (`n=5`) | 1.201816 | 1.170148 | 1.140525 | 1 |
>
> The `(n+1)`-neuron common-phase block on the programme's optimiser satisfies (H) and attains
> `(n+1)^{-(nu-q)/q} B Gamma_{nu/q}(p)` exactly, so on this class the max-margin value is pinned between
> `1` and `F(n,q)` times the conjectured answer. **(T) on this class is exactly the statement
> `L_r(n) = (n+1)^{1-r}`, i.e. `F = 1`.**
>
> *Plain reading: for every junk-free network at all — however wide, however its neurons mix
> frequencies — the best normalised margin is what the `(n+1)`-neuron block gets, to within `0.09%` at
> `nu = 3`.*

The engine is a new finite lemma, which needs neither `p`, nor the task, nor a network:

> **THEOREM E (the peeling ladder; proved, EVERY number of atoms `N`, every `n >= 1`, every
> `r in (0,1]`).**
> Let `c_1..c_N` be complex, `v_lj` (`l = 1..N`, `j = 1..n`) of unit modulus, and
>
>     (KL)    sum_{l=1}^N c_l prod_{j in T} v_lj  =  A [T empty]     for every T subset of {1..n}.
>
> Then, writing `x_1 >= x_2 >= ... ` for the sorted `|c_l|`,
>
>     (PEEL_h)   sum_{l > h} x_l  >=  D_h(n) |A|          for h = 0, 1, ..., n,
>
> with `D_0 = 1` and the explicit constants `D_h(n)` of §3, and consequently
>
>     sum_l |c_l|^r  >=  L_r(n) |A|^r,
>     L_r(n) := min { sum_l x_l^r : x_1 >= x_2 >= ... >= 0, sum_{l>h} x_l >= D_h(n), h = 0..n }.
>
> The open Key Lemma (`sum_l |c_l|^r >= (n+1)^{1-r}|A|^r`, "G1") is the statement
> `L_r(n) = (n+1)^{1-r}`; Theorem E proves it with `L_r(n) / (n+1)^{1-r} = 0.9994 / 0.9849 / 0.9620`
> at `n = 2 / 3 / 4` and `q = 2`.
>
> *Plain reading: `(PEEL_h)` says that after you throw away the `h` biggest neurons of a feature,
> at least a fixed fraction of the clean amplitude is still carried by the rest — for `h` up to the
> number of operand slots. Previous work used ONE value of `h`; the whole ladder `h = 0..n` is what
> nearly closes the gap.*

> **COROLLARY (scope of the junk gap G3).** Of the `p^{n+1} - p` off-diagonal Fourier coefficients of the
> logit tensor, only the `(p-1)(2^n-1)/2` listed in (H) enter the proof at all. Everything the argument
> needs about a network is: its clean spectrum `(G_k)`, its per-neuron frequency masses, and those
> `2^n - 1` numbers per frequency.

---

## 2. The reduction with no purity: G1 and G2 are the same gap

This section is exact algebra; `gopt3_chain.py` checks every identity against the brute-force
`p^n x p` logit table to `2.7e-15`.

**Step 1 (the clean part).** With `ehat_ij(k) = sum_a e_ij(a) w^{-ka}` (`w = e^{2 pi i/p}`),

    z(a,c) = p^{-nu} sum_{k_1..k_n, k_c} [ sum_i prod_j ehat_ij(k_j) what_i(k_c) ] w^{sum_j k_j a_j + k_c c} ,

and averaging over the level set `{s(a) - c = D}` kills every multi-index except `k_1 = ... = k_n = -k_c`:

    g(D) := mean_a z(a, s(a) - D)  =  p^{-nu} sum_{k in Z_p} G_k w^{kD},
    G_k = sum_i prod_j ehat_ij(k) conj(what_i(k))  =  sum_i c_ik .

**Step 2 (the atoms are the Fourier data).** Put `t_ik := |c_ik| = prod_j |ehat_ij(k)| * |what_i(k)|` and
`v_ijk := conj(ehat_ij(k))/ehat_ij(k)` (unit modulus whenever `ehat_ij(k) != 0`; if it is `0` then
`c_ik = 0` and the atom is absent). The logit tensor's coefficient at the multi-index that flips the sign
of the operands in `T` and takes `k_c = -k` is exactly

    sum_i prod_{j in T} ehat_ij(-k) prod_{j not in T} ehat_ij(k) what_i(-k)  =  sum_i c_ik prod_{j in T} v_ijk  =  J_{k,T}.

So **the (KL) system of Theorem E is the statement "the same-\|k\| junk vanishes", with `A = G_k` and one
atom per neuron** — for arbitrary neurons. Agents 1 and 2 obtained this system only for pure neurons,
where the number of atoms at a frequency is the number of neurons sitting on it (and their Key Lemma
needed that number to be `n+1`). In general the number of atoms at each frequency is the **width**.
Hence:

> **G1 (`N > n+1` atoms) and G2 (impure neurons) are one and the same gap.** Removing purity costs
> nothing except allowing `N` to be arbitrary; conversely, the Key Lemma for all `N` removes purity for
> free. This is why "prove G1" was always worth more than it looked: it buys G2 as well, at every `K`.

**Step 3 (margin).** `L1` (orbit averaging, full table) gives `margin <= margin(g)`, and averaging the
bound over `D` and `-D` (the phase lemma) gives, for every network,

    margin  <=  (2 p^{-nu}) min_{D != 0} sum_{k in H} |G_k| (1 - cos(2 pi k D/p)).      ... (2.1)

Checked: `max margin / RHS = 1.000000` over random and block-perturbed nets, never above (`gopt3_chain.py`
V2; equality is attained by the block, so the step is sharp).

**Step 4 (cost, impure allowed).** With `m_ik := sum_j |ehat_ij(k)|^2 + |what_i(k)|^2`, AM-GM over the
`nu` factors gives `t_ik <= (m_ik/nu)^{nu/2}`, and since `q/nu <= 1 <= q/2`,

    sum_{k in H} t_ik^r  <=  nu^{-q/2} sum_{k in H} m_ik^{q/2}  <=  nu^{-q/2} ( sum_{k in H} m_ik )^{q/2}
                         <=  (p / (2 nu))^{q/2} ||theta_i||^q .                          ... (2.2)

(The last step is Parseval with the `+-k` factor `1/2`.) This is the only place `q >= 2` is used; it is
exactly the per-neuron budget of `PROOF_AUDIT` F8, and it **already tolerates impure neurons**. Checked:
`max LHS/RHS = 1.000000`, attained by a balanced single-frequency neuron (`gopt3_chain.py` V3).

**Step 5 (assembly).** Under (H), Theorem E at each frequency gives `sum_i t_ik^r >= L_r(n) |G_k|^r`, so
`sum_{k in H} |G_k|^r <= L_r(n)^{-1} (p/(2nu))^{q/2} P_q`. Feeding that into (2.1) via the definition of
`Gamma_s` (`min_D sum_k x_k g_k(D) <= Gamma_s ||x||_r`, `s = 1/r`) and using
`(2 p^{-nu}) (p/(2nu))^{nu/2} = 2 (2 p nu)^{-nu/2} = B` yields Theorem D. Without Step 5's Theorem E —
i.e. with the trivial `|G_k|^r <= sum_i t_ik^r` — the same chain is precisely the known unrestricted
bound `gamma_q <= B Gamma_{nu/q}`. **So `L_r(n)` is exactly the quantity that separates the old
one-sided theorem from (T).**

---

## 3. The peeling ladder (proof of Theorem E)

**The dual pairing.** For any multi-affine `Lambda(z) = sum_T lambda_T prod_{j in T} z_j`, (KL) gives the
exact identity

    sum_l c_l Lambda(v_l)  =  lambda_empty * A .                                          ... (3.1)

*Plain reading: a "test polynomial" applied to the atoms reads off the clean amplitude, weighted by the
test polynomial's own average over the torus (its constant coefficient is that average).*

**(PEEL_h).** Let `S` be any set of `h <= n` atoms. Give each `l_i in S` its own operand slot `j_i` and take

    Lambda(z) = prod_{i=1}^h ( 1 - z_{j_i} conj(v_{l_i, j_i}) ) / 2 .

Then `Lambda(v_{l_i}) = 0`, `lambda_empty = 2^{-h}`, and `|Lambda| <= 1` on the whole torus, so (3.1)
gives `2^{-h}|A| <= sum_{l not in S} |c_l|`. Taking `S` = the `h` largest atoms:

    sum_{l > h} x_l  >=  2^{-h} |A| ,   h = 0, 1, ..., n.                                 (PEEL_h, crude)

**Better test polynomials.** `2^{-h}` is what a *product* of `h` one-slot kills gives. The multi-affine
space on `T^n` has the reproducing kernel `K(z,v) = prod_j (1 + z_j conj(v_j))` for Haar measure, and

    Lambda_v(z) := [ 1 - prod_{j=1}^d (1 + z_j conj(v_j))/2 ] / (1 - 2^{-d})

is multi-affine in `d` slots, has constant coefficient `1`, and vanishes at `v`. Its sup-norm is

    S_d := max_{|z| = 1} | 1 - ((1+z)/2)^d | ,

because on `T^d` we may write `prod_j (1 + z_j)/2 = m e^{i phi}` with `m = prod_j cos(psi_j)`,
`phi = sum_j psi_j`, `z_j = e^{2 i psi_j}`; then `|1 - u|^2 = 1 - 2 m cos(phi) + m^2` and at any critical
point with `m != 0`,

    d/d psi_j = 2 m [ tan(psi_j) (cos phi - m) + sin phi ] = 0

forces the same `tan(psi_j)` for every `j`, hence `psi_j in {psi, psi + pi}`; flipping `k` of them
multiplies `m` by `(-1)^k` and shifts `phi` by `k pi`, leaving `m e^{i phi}` unchanged. So the sup is the
diagonal one-variable maximum. (`m = 0` gives `|1-u| = 1 <= S_d`.) Hence, writing
`kappa_d := (1 - 2^{-d}) / S_d`, and splitting the `n` slots into `h` disjoint groups of sizes
`d_1 + ... + d_h = n`, the product `Lambda = prod_i Lambda_{v_{l_i}}(z_{J_i})` is multi-affine with
constant coefficient `1`, vanishes at all `h` points, and has sup-norm `prod_i 1/kappa_{d_i}`. Therefore

    (PEEL_h)   sum_{l > h} x_l  >=  D_h(n) |A| ,    D_h(n) := max over d_1+...+d_h = n of prod_i kappa_{d_i}.

| `d` | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| `S_d` | `1` | `2/sqrt(3) = 1.154700538` | `9/7 = 1.285714286` | `1.380900987` | `1.452990673` | `1.509590371` |
| `kappa_d` | `1/2` | `3 sqrt(3)/8 = 0.649519053` | `49/72 = 0.680555556` | `0.678904577` | `0.666728299` | `0.652080868` |

(`S_2 = 2/sqrt3` and `S_3 = 9/7` are exact: with `c = cos^2 psi`, `|1-u|^2` equals `1 + 2c - 3c^2` and
`1 + 6c^2 - 7c^3`, maximal at `c = 1/3` and `c = 4/7`, values `4/3` and `81/49`. `S_4, S_5, S_6` are
algebraic — `S_4^2 = max_c (1 - 15c^4 + 16c^3 - 2c^2)` at `c = (6 + sqrt 21)/15` — and are given here to
fp64.)

**Rigour note.** For `n = 2` and `n = 3` — the cases the paper uses — only `kappa_1 = 1/2`,
`kappa_2 = 3 sqrt(3)/8` and `kappa_3 = 49/72` are needed, all three in exact closed form, so the
constants `D_h(2)`, `D_h(3)` and hence `L_r(2)`, `L_r(3)` and the factors `1.000905` / `1.030841` are
exact algebraic numbers. For `n >= 4` the constant `D_1(n)` uses `S_n`, evaluated numerically (a
one-variable maximisation on a `4e6`-point grid followed by a bounded polish, `xatol 1e-15`); a
referee-proof version there needs either the exact algebraic value or a rounded-down `kappa_n`.

| `n` | `D_1` | `D_2` | `D_3` | `D_4` | `D_5` | best splits |
|---|---|---|---|---|---|---|
| 2 | 0.649519053 | 0.25 | | | | `(2)`, `(1,1)` |
| 3 | 0.680555556 | 0.324759526 | 0.125 | | | `(3)`, `(1,2)`, `(1,1,1)` |
| 4 | 0.678904577 | 0.421875 | 0.162379763 | 0.0625 | | `(4)`, `(2,2)`, `(1,1,2)`, `(1,1,1,1)` |
| 5 | 0.666728299 | 0.442033800 | 0.210937500 | 0.081189882 | 0.03125 | `(5)`, `(2,3)`, `(1,2,2)`, `(1,1,1,2)`, `(1,1,1,1,1)` |

For `h = 1` the kernel polynomial is **optimal**: an independent LP over all multi-affine `Lambda`
(`gopt3_dual.py`) returns `0.649521` at `n = 2` against the closed form `3 sqrt(3)/8 = 0.6495190528`.
For `h = 2` at `n = 2` the LP gives `0.2599` against our `0.25`, so a little is still on the table.

---

## 4. From the ladder to `L_r(n)`

`(PEEL_h)` constrains only the *sorted tail sums* of the atom moduli, so

    sum_l |c_l|^r  >=  L_r(n) |A|^r,
    L_r(n) = min { sum_l x_l^r : x_1 >= x_2 >= ... >= 0,  sum_{l>h} x_l >= D_h(n),  h = 0..n }.

This is a small, `r`-dependent, finite programme. It is solved **exactly**, not by a local optimiser:
`sum_l x_l^r` is *concave* on `R_+^N` and the feasible set is a pointed polyhedron, so the minimum sits
at a vertex, and `L_r_vertex` in `gopt3_peel.py` enumerates every vertex (each `N`-subset of the
`N + n + 1` defining inequalities that gives an invertible system and a feasible point), `N <= 8`.
SLSQP from 120 random starts agrees to `< 1e-9` in every cell. The minimising vertex is *not* the block:
at `n = 2` it is `x = (0.350481, 0.350481, 0.299038)` (the `h = 0` and `h = 1` constraints tight, `h = 2`
slack), at `n = 3` it is `(0.319444, 0.319444, 0.236111, 0.125)`. Because the block
(`x_l = |A|/(n+1)`) makes the crude `(PEEL_h)` slack and the kernel ones nearly tight, `L_r(n)` lands
just below the conjectured `(n+1)^{1-r}`:

| `nu` | `q` | `r` | `L_r` (kernel `D`) | `(n+1)^{1-r}` | ratio | **factor `F` in Theorem D** | (`F` with the crude `2^{-h}`) |
|---|---|---|---|---|---|---|---|
| 3 | 2 | 0.6667 | 1.44138016 | 1.44224957 | 0.999397 | **1.000905** | 1.019646 |
| 3 | 2.5 | 0.8333 | 1.20048594 | 1.20093696 | 0.999624 | 1.000451 | 1.009830 |
| 4 | 2 | 0.5 | 1.96985438 | 2 | 0.984927 | **1.030841** | 1.091640 |
| 4 | 2.5 | 0.625 | 1.65843527 | 1.68179283 | 0.986112 | 1.022630 | 1.067923 |
| 4 | 3 | 0.75 | 1.39876048 | 1.41421356 | 0.989073 | 1.014757 | 1.044682 |
| 5 | 2 | 0.4 | 2.52661755 | 2.62652780 | 0.961961 | **1.101809** | 1.218082 |
| 6 | 2 | 0.3333 | 3.10566686 | 3.30192725 | 0.940562 | 1.201816 | — |

At `q = nu` (`r = 1`) the ladder gives `L_1 = 1` exactly, which is the classical `sum|c_l| >= |A|`, and
`F = 1`: the theorem is two-sided there, as it must be.

**Verification.** `(PEEL_h)` and `sum|c|^r >= L_r|A|^r` were checked on random feasible (KL)
configurations with `N` from `n+1` to `n+7` at `n = 1,2,3` (least-squares solution plus random
null-space components, residual `< 1e-8`): the minimum observed `[tail mass]/2^{-h}` is
`1.013, 1.403, 1.145` at `n = 2` and `1.043, 1.728, 2.802, 4.308` at `n = 3` — never below `1`; and
`min sum|c|^r/(n+1)^{1-r} = 1.126` (`n=2`), `1.417` (`n=3`) at `q = 2`. Theorem D itself was checked on
gauge-symmetrised networks (§8): `gamma_q` never exceeded the certified bound, and a margin ascent
restricted to the junk-free class reached `0.9991` of it, i.e. exactly `1/F` — the block.

---

## 5. What this changes about G1, G2, G3

* **G1** — unchanged as a sharp statement (the Key Lemma for `N > n+1` is still open), but it is no
  longer an all-or-nothing gap: Theorem E gives it with a constant that is `0.06% / 1.5% / 3.8%` short
  at `n = 2 / 3 / 4`. Part of that slack is recoverable inside the method — the `D_h(n)` for `h >= 2`
  are not optimal (at `n = 2, h = 2` the LP optimum is `0.2599` against our `0.25`) — but not all of it:
  F3-2 shows that even with the *sharp* peeling numbers the ladder stops strictly below
  `(n+1)^{1-r}`, because `(PEEL_1)` is already slack at the block (`D_1^{sharp} = 0.6495 < 2/3`).
  Closing G1 therefore needs one ingredient that the sorted tail sums do not see.
* **G2 — closed, as a gap in its own right.** §2 shows the atom system is exact for arbitrary neurons;
  there is nothing left to prove about impurity. (Agent 1's ranked item 2, "close G2 by an
  `O(sqrt(delta))` stability estimate", is unnecessary — no perturbation argument is needed.)
* **G3 — narrowed, not closed.** Theorem D covers *every* `K`, so the "multi-frequency" half of G3 is
  gone; what is left is exactly junk, and only the `2^n - 1` same-frequency coefficients per frequency.
  The obstruction recorded by agents 1 and 2 survives in the new language: junk enters the cost bound
  through (3.1), which degrades to

      sum_{l not in S} |c_l|  >=  2^{-h}|A| - (1 - 2^{-h}) max_{T != empty} |J_{k,T}|

  (the test polynomial's non-constant coefficients pick up the junk), i.e. junk of size `J` buys `O(J)`
  of cost relief at every frequency, while the only known margin penalty for it (Lemma Q / Lemma C3)
  is `O(J/p)` relative to a clean margin that is `Theta(1)` on a `K >= 2` support. The trade therefore
  still fails on spread supports, exactly as recorded. Nothing here makes it work; but the target is
  now a statement about `(p-1)(2^n-1)/2` numbers rather than about the whole logit tensor.

---

## 6. Approaches tried here that FAILED, and exactly where

**F3-1. The capped-mass conjecture (C'). REFUTED, by 1.7%.** The cleanest `r`-free route to the Key
Lemma is

    (C')   for every theta > 0:   sum_l min(|c_l|, theta) >= min( (n+1) theta, |A| ),

which implies the Key Lemma for every `r` at once via `x^r >= t^r min(x/t, 1)` at `t = |A|/(n+1)` (both
steps are equalities at the block, so (C') is exactly as tight as (T)). **It is false.** `gopt3_cap.py`
(SLSQP over `c` and the phases; the violating configuration is a non-obvious local optimum — 25 starts
per cell do not find it, 60 and 120 do) gives, at `n = 2`, `A = 1`,

    min over feasible configurations of sum_l min(|c_l|, 1/3)  =  0.98285239   <   1,

identically at `N = 3, 4, 5` (so it is not an `N > n+1` effect — it already fails where the Key Lemma is
*proved*). At `theta = 0.2/(n+1), 0.5/(n+1), 0.8/(n+1), 1.3/(n+1), 2/(n+1)` the ratio is exactly `1`, and
at `n = 1` it is `1` everywhere. So the truth is a genuine *profile* statement: writing
`G(theta) = sum_l min(x_l, theta)` and `N(t) = #{l : x_l >= t}`, the layer-cake identity
`sum_l x_l^r = r(1-r) int_0^inf t^{r-2} G(t) dt` shows the Key Lemma needs
`int_0^inf t^{r-2} [G - G_block] dt >= 0` and NOT `G >= G_block` pointwise — the profile dips below the
block's exactly at the critical scale `theta = |A|/(n+1)` and must make it up at smaller `theta`.

**F3-2. Sharp peeling numbers by convex duality. REFUTED as a route to the sharp constant.** Fixing the
`h` excluded atoms, the minimal remaining mass is an atomic norm whose dual is the multi-affine sup-norm,
so

    P_h := min over configurations of [mass outside h chosen atoms]
         = min over v_1..v_h in T^n of max{ |Lambda_empty| : ||Lambda||_{L^inf(T^n)} <= 1, Lambda(v_i) = 0 } = D_h(n)^{sharp}.

If `D_h^{sharp}` were the block's value `(n+1-h)/(n+1)` the Key Lemma would follow exactly. It is not.
`gopt3_dual.py` computes `min ||Lambda||_inf` by an LP on the epigraph with `|Lambda| <= t` imposed only
on a finite torus grid and a finite set of phases; **dropping constraints lowers the minimum**, so the LP
value `t_lp` satisfies `t_lp <= min||Lambda||_inf` and therefore `1/t_lp >= D_h(v)` is an **upper
bound**, while the exact sup-norm `t_true` of the LP's own solution gives the matching lower bound
`1/t_true <= D_h(v)`. At `n = 2` (grid `32^2`, 16 phases; sup-norms re-measured on `200^2`):

    D_1(v) in [0.649521, 0.653611]  for v = (1,1)      -- vs the block's 2/3 = 0.666667,
    D_2(v) in [0.259892, 0.259892]  for v = (1,1), (e^{-i pi/4}, e^{-i pi/4})  -- vs 1/3.

Both brackets lie strictly below the block's values, so `(P_h)` is false and the peeling ladder *cannot*
reach `(n+1)^{1-r}` by itself. Quantitatively: feeding the sharp *upper* bounds
`D_1 <= 0.653611, D_2 <= 0.259892` into the ladder programme gives `L_r <= 1.44174873` at `q = 2`, i.e.
the **best factor any peeling ladder can ever certify at `nu = 3, q = 2` is `1.000521`**, against our
`1.000905` and the target `1`. So a little under half of the residual is recoverable inside the method
and the rest is not. (The relaxation is lossy in an identifiable way: it lets the `h` excluded atoms be
arbitrarily small, whereas the sorted statement excludes the *largest* ones.)

**F3-3. Induction on `n` by shifting the last slot. REFUTED (independent rediscovery of `GLOBAL_OPT`
F-3).** Expanding (KL) in the last slot gives, for every `lambda in C`,
`sum_l c_l (1 + lambda v_{ln}) prod_{j<n}(1 - v_lj w_j) = A`, so the `(n-1)`-slot case yields
`sum_l x_l^r |1 + lambda v_{ln}|^r >= n^{1-r}|A|^r` for all `lambda`. Gaining the factor
`((n+1)/n)^{1-r}` needs a probability measure `mu` on `lambda` with `E_mu |1 + lambda v|^r <=
(n/(n+1))^{1-r} < 1` for every unit `v`; averaging that over `v` and using Jensen on `log|1+lambda v|`
(whose mean over `v` is `log max(1,|lambda|) >= 0`) forces `mean_v E_mu |1+lambda v|^r >= 1`. **No such
measure exists**, for any `n`. Adapting `mu` to the actual atoms is exactly `GLOBAL_OPT` F-3 and fails
at the block by the same 10% (`0.9615` vs `0.8736` at `n=2, r=2/3`).

**F3-4. The modulus relaxation `sum_l x_l prod_j |u_j - v_lj| >= |A|`.** Tight at the block (it *is* the
bijection identity), but as an LP in `x` alone it has the same defect as F3-2 and gives nothing beyond
the peeling ladder.

**Previously recorded dead routes confirmed, not re-attempted:** Morwani-style dual certificate (provably
cannot certify any constant `< 1` for `q < nu`), the domination lemma `sum|c| >= (n+1) max|c|`
(refuted by null configurations), gauge symmetrisation (exactly circular — see the remark below),
Mahler measure / geometric mean, Gram / Caratheodory–Fejer, grid rotation.

**A useful reading of the circularity.** Averaging a network over the `n+1` gauge phases
`pi l/(n+1)` (rotating every positive-frequency Fourier coefficient of each embedding by `e^{i phi}` and
of the readout by `e^{i n phi}` — an orthogonal map on each vector) kills exactly the same-`|k|` junk and
multiplies `P_q` by `(n+1)^{1-r}`. So the *sup* of `gamma_q` over gauge-symmetrised networks is
`(n+1)^{-(nu-q)/q}` times the sup over all networks — and applying the unrestricted bound
`gamma_q <= B Gamma_s` to the pre-symmetrised net gives **exactly** the target (T) for the symmetrised
one. This is why (a) the parametrisation `theta -> symmetrise(theta)` is the natural one for numerical
attacks on (T) (§8), (b) the equality case of the unrestricted bound — one pure balanced neuron per
frequency — symmetrises to precisely the `(n+1)`-block, and (c) symmetrisation can never *prove* (T),
since it controls the minimum over a gauge orbit while (T) is about the maximum.

---

## 7. What remains open, ranked

1. **The last `0.06% / 1.5% / 3.8%` (G1 sharp).** Concretely: prove `L_r(n) = (n+1)^{1-r}`. Since F3-2
   shows the peeling ladder alone cannot get there, a proof must use information the sorted tail sums
   throw away. The most promising remaining handle is agent 2's dual-measure statement, which in the
   present language reads: at a minimiser of `sum|c_l|^r` there is a multi-affine `Lambda` with
   `Lambda(v_l) = (r/2)|c_l|^{r-2} conj(c_l)`; and — this is new — stationarity in the *phases* forces
   `Im[ beta_j(v_{l,-j}) v_{lj} conj(alpha_j(v_{l,-j})) ] = 0` for every atom and slot, where
   `Lambda = alpha_j + beta_j z_j`. If `Lambda` is constant then all `|c_l|` are equal, all `c_l` share a
   phase, `sum_l c_l = A` forces `|c_l| = |A|/N`, and Theorem A (`N >= n+1`) finishes exactly. So G1 is
   still "the optimal dual test polynomial is constant", now with an extra rigidity equation available.
2. **Improve `D_h(n)` for `h >= 2`.** Our product-of-kernels construction gives `0.25` at `n = 2, h = 2`
   against a sharp value of `0.2599`, and `0.3248` at `n = 3, h = 2`. Closing that recovers the factor
   from `1.000905` to at best `1.000521` at `nu = 3, q = 2` (F3-2) — worth doing, and unlike item 1 it is
   an *existence* problem: construct, for every configuration of `h` points of `T^n`, a multi-affine
   `Lambda` vanishing there with a good average-to-peak ratio. The obvious candidate is the `h`-point
   reproducing-kernel polynomial `Lambda = 1 - sum_i mu_i K(.,v_i)/2^n` with the `mu_i` solving
   `Lambda(v_i) = 0`; what is missing is a sup-norm bound uniform over configurations. Measured
   worst-case values of exactly that two-point polynomial (`n = 2`: 3000 random pairs on a `120^2` grid;
   `n = 3`: 600 pairs on `40^3`) are `D_2 >= 0.252882` at `n = 2` (vs our proved `0.25`: no change to the
   factor) and `D_2 >= 0.405227` at `n = 3` (vs `0.324760`), which would move the `nu = 4, q = 2` factor
   from `1.030841` to `1.027425`. So the two-point kernel is worth proving at `n >= 3` and worth little
   at `n = 2`.
3. **G3 (junk).** Unchanged in difficulty but reduced in scope (§5). The live ingredients are the robust
   peeling inequality of §5 and Lemma Q / Lemma C3; the missing one is still a junk penalty that is
   `Theta(J)` rather than `O(J/p)` on a `K >= 2` support.
4. **For the paper.** The safe, and now much stronger, claim is **Theorem D**: on junk-free networks of
   any width and with no structural assumption on the neurons, the `(n+1)`-block is the max-margin
   solution to within a factor `1.0009` at `nu = 3` (`1.031` at `nu = 4`), together with agent 2's
   Theorems A/B/C for the single-frequency case *with* junk. That is a genuine two-sided statement on a
   class that contains the paper's constructions, and it removes the purity hypothesis that made the
   earlier version look narrow.

---

## 8. Numerical record (all float64, local CPU, no pods, `heracles/FINAL_RUNS/GOPT3/`)

| script | what it does | result |
|---|---|---|
| `gopt3_chain.py` V1 | the §2 dictionary (clean part; `J_{k,T}` vs the logit tensor's Fourier coefficients) against brute force, `p = 5..13`, `n = 2,3`, random *and* block-perturbed nets | relative error `2.7e-15` |
| `gopt3_chain.py` V2 | the margin bound (2.1) | `max margin/bound = 1.000000`, attained by the block, never exceeded |
| `gopt3_chain.py` V3 | the per-neuron budget (2.2), `q = 2, 2.5, nu` | `max = 1.000000`, attained by a balanced single-frequency neuron, never exceeded |
| `gopt3_chain.py` V4 | **Theorem D** on gauge-symmetrised (hence junk-free, impure, width `(n+1)m`) networks: sampled nets, block perturbations, and an L-BFGS margin ascent inside the class; `p = 7,11,13` (`n=2`), `p = 7` (`n=3`), `q = 2, 2.5` | same-`\|k\|` junk residual `<= 1e-14`; nothing exceeded the certified bound; the ascent reaches `0.9991 = 1/F` at `nu 3` and `0.970` at `nu 4`; the `(n+1)`-block attains the target `1.00000` |
| `gopt3_peel.py` | `(PEEL_h)` and `sum\|c\|^r >= L_r\|A\|^r` on random feasible (KL) configurations, `n = 1,2,3`, `N = n+1 .. n+7`; also `L_r_vertex` | min `[tail mass]/2^{-h}` `= 1.013/1.403/1.145` (`n=2`), `1.043/1.728/2.802/4.308` (`n=3`); min `sum\|c\|^r/(n+1)^{1-r} = 1.126` (`n=2`), `1.417` (`n=3`) — never below 1 |
| `gopt3_const.py` | `S_d`, `kappa_d`, `D_h(n)`, `L_r(n)`, `F(n,q)` to fp64; `L_r` by **exact vertex enumeration**, cross-checked against SLSQP | table of §3–§4; the two methods agree to `< 1e-9` in all 16 cells; `S_2 = 2/sqrt3`, `S_3 = 9/7`, `kappa_2 = 3 sqrt3/8`, `kappa_3 = 49/72` confirmed to `1e-15` |
| `gopt3_dual.py` | the sharp peeling numbers `D_h^{sharp}`, bracketed above by the LP relaxation and below by the exact sup-norm of the LP's solution | `n=1`: `D_1 = 0.5` exactly (= the block's `1/2`); `n=2`: `D_1 in [0.649521, 0.653611]` (`< 2/3`), `D_2 in [0.259892, 0.259892]` (`< 1/3`) — **the refutation of (P_h), F3-2** |
| `gopt3_cap.py` | the capped-mass conjecture (C'), `n = 1,2`, `N = n+1..n+3`, `theta` on a grid around `\|A\|/(n+1)` | **refuted**: `0.98285239` vs `1` at `n = 2`, `theta = 1/3`, for every `N` tested; `1.000000` at every other `theta` and at `n = 1` |
