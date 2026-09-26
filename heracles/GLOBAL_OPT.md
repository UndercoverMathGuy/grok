# Is the (n+1)-block the GLOBAL max-margin network? — a new reduction, three new theorems, and the honest gap list

Continues `heracles/LEMMA_ATTEMPT.md` (yesterday) and `heracles/THEORY_nsum.md` §8 / `heracles/PROOF_AUDIT.md` F1.
Target set by the coordinator, for the gated chain and `2 <= q < nu`:

    gamma_q(theta)  <=  (n+1)^{-(nu-q)/q} * B * Gamma_{nu/q}(p),     B = 2 (2 p (n+1))^{-nu/2}.      (T)

**Status: PARTIAL, with a strict advance.** Nothing here contradicts (T); every numerical attack failed.
What is new:

* a **new exact reduction** — the whole margin of a one-frequency population (junk included) is one
  min of one multi-affine polynomial, with no inequality spent (§2);
* **Theorem A** — `N <= n` neurons at a frequency force margin `<= 0`, *with junk allowed*
  (yesterday's Theorem 1b needed junk-free);
* **Theorem B** — the single-frequency case of (T), exactly, for `N = n+1` neurons, **with junk
  allowed, for every n and every `2 <= q <= nu`, with no condition on `p`**, in the continuum, and with
  the equality case characterised. Yesterday's Theorem 2 proved the same conclusion at finite `p` but
  only under `2(2^n − 1) sin(pi/p) <= 1` (`p >= 19 / 47 / 97` for `n = 2 / 3 / 4`), and needed two
  separate ingredients (a junk-free Key Lemma plus Lemma Q, traded off against each other). The new proof
  is one step, needs no trade-off, and **recovers yesterday's threshold as a one-liner** (§5, Lemma C3);
* **Theorem C** — an explicit finite-`p` version covering the primes *below* that threshold, where
  nothing was proved before: `(T)` at a single frequency up to a factor `1 + O(p^{-2})` (§5);
* **the G1 gap is reduced to a single clean statement about the dual measure** (§6), and is settled
  numerically to 14 digits by a proper global NLP search (§10) rather than by heuristic ascent;
* **Lemma P** (§7), a new *product-structured* junk tool, complementary to (not containing) yesterday's
  Lemma Q, aimed at G3;
* **the first genuinely global numerical test of (T) itself** (`gopt_full.py`): the max-margin problem
  over *whole unrestricted networks* — raw tensors `E (n,m,p)`, `W (p,m)`, no purity, no junk-free, no
  support restriction, i.e. all three gaps at once — is also a smooth NLP. At `p = 5, 7`, `n = 2`,
  `q = 2, 2.5`, widths `6/9/12`, and at `p = 5`, `n = 3` (`nu = 4`), `q = 2, 3`, widths `8/12`, it returns
  `gamma_q / target = 1.000000000000` (max over all 16 cells `1 − 1.6e−15`, never above) whenever the
  width can hold the optimal support. Previous evidence was
  annealed ascent reaching `0.98` at `p = 7` and `0.42` at `p = 11`. (Caveat: the block is among the
  starts, so *reaching* `1` is by construction; the content is that **nothing exceeded it** from 12
  random restarts plus 2 noisy block restarts per ladder support per cell, and that `Gamma_s` is exact
  here because the programme search covers all `nf` frequencies at `p = 5, 7`.)

Everything is float64, local CPU, no pods. Scripts `heracles/gopt_*.py`, outputs
`heracles/FINAL_RUNS/GOPT/`.

---

## 0. Symbols, with a plain reading of each

| symbol | meaning | plain reading |
|---|---|---|
| `p` | odd prime | the modulus; the task is `a_1 + ... + a_n = c (mod p)` on all `p^n` rows |
| `n` | number of operands | how many numbers are added |
| `nu = n+1` | homogeneity degree | scaling every weight by `s` scales every logit by `s^nu` |
| `q in [2, nu]` | penalty exponent | `q = 2` is plain weight decay, `q = nu` is Morwani's `L_{2,nu}` |
| `r = q/nu in (0,1]` | the ratio that runs everything | how cheap it is to split one feature over several neurons; `r = 1` = free |
| `e_ij, w_i in R^p` | neuron `i`'s `j`-th operand embedding, and its readout | one row of one layer each |
| `z(a, c)` | logit of class `c` on row `a` | `sum_i w_i(c) prod_j e_ij(a_j)` |
| `P_q = sum_i ‖theta_i‖^q` | the penalty | what weight decay prices; `‖theta_i‖² = sum_j ‖e_ij‖² + ‖w_i‖²` |
| `margin` | `min` over rows and wrong classes of `z_correct − z_wrong` | worst-case gap to a mistake |
| `gamma_q = margin / P_q^{nu/q}` | normalised margin | scale-invariant "margin per unit of weight" |
| `M_1 = 1 − cos(2 pi / p)` | the `K = 1` programme value | best clean margin a single frequency can buy, per unit amplitude |
| `Gamma_s(p)` | free-amplitude programme, `s = nu/q` | `max { min_{D≠0} sum_k x_k (1 − cos 2 pi k D / p) : x >= 0, sum_k x_k^{1/s} <= 1 }`; `Gamma_s >= M_1` always |
| `B = 2 (2 p (n+1))^{-nu/2}` | constant of the existing upper bound | pure bookkeeping (Parseval, AM-GM, the `±k` factor 2) |
| `mu_p` | the `p`-th roots of unity | the values the task's Fourier phases can actually take |
| `T^n` | the `n`-torus (each coordinate on the unit circle) | the continuum idealisation of `mu_p^n` |

Two conventions: `theta_x := 2 pi k x / p` for a frequency `k`; `T` always denotes a subset of the
operand indices `{1..n}` (a *sign pattern*: which operands have their Fourier sign flipped).

---

## 1. Exact statements

Fix a frequency `k` and consider a population of `N` neurons that are **pure single-frequency cosines**,

    e_lj(a) = r_lj cos(theta_a + alpha_lj)  (j = 1..n),     w_l(c) = rho_l cos(theta_c + beta_l),

and put (the *atom data* of the population)

    c_l  := (prod_j r_lj) rho_l e^{i psi_l},   psi_l := sum_j alpha_lj − beta_l      (l = 1..N),
    v_lj := e^{−2 i alpha_lj}                  (unit modulus),
    h(u) := sum_{l=1}^N c_l prod_{j=1}^n (1 + u_j v_lj)        (a multi-affine polynomial).

Plain reading: `c_l` is neuron `l`'s signed clean amplitude, `v_lj` records the phase of its `j`-th
embedding, and `h` is a generating function for the `2^n` Fourier sign-patterns the population emits at
this frequency: the constant coefficient of `h` is the **clean** amplitude, the other `2^n − 1`
coefficients are the **junk**. `h` constant `<=>` junk-free.

> **THEOREM A (count floor, junk allowed; proved, every `n >= 1`).**
> If a frequency is carried by `N <= n` pure neurons then `min_{u in T^n} Re h(u) <= 0` (continuum), and
> at finite `p` the network's margin obeys `2^n margin <= M_1 · 2^{n−1}(1 − cos(pi/p)) ‖c‖_1`, which is
> strictly below the target of (T) for every `p >= 5` (`n = 2`), `p >= 11` (`n = 3`), `p >= 23` (`n = 4`)
> at `q = 2`. *Plain reading: with `n` operand slots you can shoot down `n` neurons at once, so with at
> most `n` neurons you shoot them all down and nothing is left to carry the signal.* (Yesterday's
> Theorem 1b was the junk-free special case.) Numerically the finite-`p` sup is exactly `0`
> (`gopt_nlp.py --Ns 1 2`: max ratio `−1e−15` at `N = 2`, `−3.9` to `−29` at `N = 1`, `p = 5..13`), so
> the `O(p^{-2})` slack above is slack in the proof, not in the truth.

> **THEOREM B (single frequency, `N = n+1`, JUNK ALLOWED; proved, every `n >= 1`, every `2 <= q <= nu`,
> arbitrary phases and amplitudes, no condition on `p`).**
> Let `R := min_{u in T^n} Re h(u)` and `r = q/nu`. Then
>
>     sum_{l=1}^{n+1} |c_l|^r  >=  (n+1)^{1−r} max(R, 0)^r ,
>
> equivalently `R <= (n+1)^{−(1−r)/r} ‖c‖_r`. Consequently, for every network all of whose neurons are
> pure at one frequency `k` and number exactly `n+1`,
>
>     gamma_q  <=  (n+1)^{−(nu−q)/q} · B · M_1  <=  (n+1)^{−(nu−q)/q} · B · Gamma_{nu/q}(p)
>
> in the continuum sense of §5 (exactly, with `T^n` in place of `mu_p^n` in the margin), with equality
> iff the population is the `(n+1)`-neuron common-phase block of `THEORY_nsum` §8.

> **THEOREM C (the same at finite `p`, explicit).**
> With `eps := 1 − cos(pi/p)` and `eta_n(eps) := 2^n [ (1 + eps/2)^n − 1 + eps/2 ]`, for the same class,
>
>     gamma_q  <=  (n+1)^{−(nu−q)/q} · B · M_1 · [ 1 + (n+1)^{(1−r)/r} eta_n(eps) ].
>
> The bracket is `1 + O(p^{-2})`: `1.0102` at `n = 2, q = 2, p = 71`; `1.0040` at `p = 113`; `1.0002` at
> `p = 509` (table in §5). **Where it matters** is `p` below the Theorem-2 threshold: for
> `tan(pi/p) <= 1/(2(2^n − 1))` — i.e. `p >= 23 / 47 / 97` for `n = 2 / 3 / 4`, essentially yesterday's
> `p >= 19 / 47 / 97` — the bracket can be replaced by **exactly 1** (Lemma C3 below, which recovers
> yesterday's Theorem 2 in one line inside the new framework). So: (T) on single-frequency supports is
> **exact for all `p` above the threshold**, and holds within `1 + O(p^{-2})` for every `p`.

> **LEMMA P (product test measures; proved, NO restriction on the network at all).**
> For every gated-chain network, every `D != 0` and every **product** probability measure
> `w = w_1 x ... x w_n` on `Z_p^n`,
>
>     margin  <=  sum_{k=1}^{(p−1)/2} 2 (1 − cos(2 pi k D / p)) Re X_k(w),
>     X_k(w)  =  sum_i what_i(k) prod_j S_ij(k),   S_ij(k) = E_{w_j}[ e^{2 pi i k a_j / p} e_ij(a_j) ].
>
> `w_j` uniform gives the classical clean bound; a tilted `w_j` makes junk visible, and because the
> operands are independent coordinates **every frequency gets its own evaluation point inside the unit
> disc**. It is complementary to yesterday's Lemma Q (whose single-mode density on `Z_p^n` is not a
> product measure; neither lemma contains the other — both are instances of "min <= any average") and is
> the tool §7 proposes for G3.

**Scope, stated bluntly.** Theorems A–C are *single-frequency* statements about *pure* neurons. They
close, for `K = 1`, both the junk gap (yesterday's G3 at `K = 1`) and yesterday's `p`-condition. They do
**not** prove (T). The three gaps G1 (more than `n+1` neurons per frequency), G2 (impure neurons) and G3
(junk on a support of `K >= 2` frequencies) are discussed in §6–§7; G1 is reduced to one clean statement
and settled numerically, G2 and G3 remain open.

---

## 2. The new reduction: the margin of a one-frequency population, exactly

This is an identity, not an inequality — nothing is spent here. Verified to `1.1e-15` against the
brute-force `p^n x p` logit table (`gopt_verify.py` check F).

**Step 1.** Expand the product of cosines. With `u_j := e^{−2 i theta_{a_j}}`,

    z(a, c) = 2^{−n} Re[ e^{i (theta_s − theta_c)} h(u) ],      theta_s = sum_j theta_{a_j}.

*Proof.* Each neuron contributes `2^{−(n+1)} sum_{eps in {±1}^n, delta = ±1} exp(i[sum_j eps_j(theta_{a_j}
+ alpha_lj) + delta(theta_c + beta_l)])`. Pair `(eps, delta)` with `(−eps, −delta)` and keep `delta = −1`;
writing `T = {j : eps_j = −1}` the exponent is `theta_s − theta_c − 2 sum_{j in T} theta_{a_j}` and the
amplitude is `c_l prod_{j in T}(u_j v_lj)`. Summing over `l` and `T` gives `h(u)`. ∎

**Step 2.** Put `c = s(a) − D`, so `theta_s − theta_c = theta_D`:

    Phi_D(a) := z(a, s(a)) − z(a, s(a) − D) = 2^{−n} Re[ (1 − e^{i theta_D}) h(u(a)) ].

**Step 3.** `margin` is the min of `Phi_D(a)` over all rows `a` and all `D != 0`. Because `2k` is
invertible mod `p`, `u(a)` ranges over **all** of `mu_p^n` as `a` ranges over `Z_p^n`, independently of
`D`. Hence, exactly,

    2^n · margin  =  min over u in mu_p^n, D = 1..p−1  of  Re[ (1 − e^{2 pi i D / p}) h(u) ]
                  =  min over u in mu_p^n  of  [ M_1 Re h(u) − sin(2 pi / p) |Im h(u)| ]   ... (2.1)

the second line because `min over D` of `(1 − cos th_D) X + sin(th_D) Y` is attained, when
`|arg(X + iY)| <= pi/p`, at `D = ±1`, and is `<=` that value always (take `D = ∓1`). Call the left side
`Lambda`.

**Step 4 (cost).** Rebalancing a neuron's `n+1` factors at fixed product is free and makes
`‖theta_l‖² = (n+1)(p/2)|c_l|^{2/(n+1)}`, so `P_q = ((n+1) p / 2)^{q/2} sum_l |c_l|^r`. With
`2^{−n} (2/((n+1)p))^{nu/2} = B`, the single-frequency case of (T) is **exactly**

    Lambda  <=  M_1 · (n+1)^{−(1−r)/r} · ‖c‖_r ,     ‖c‖_r := (sum_l |c_l|^r)^{1/r}.      ... (2.2)

The `(n+1)`-block is `c_l = A/(n+1)`, `v_lj = e^{−2 pi i l /(n+1)}`; then `h == A` (verified to `1.6e-15`,
`gopt_atom.py block`) and (2.2) is an equality at every `p` and every `n` tested (ratio `1.000000000000`).

*Why this reduction is the advance:* yesterday's route bounded the **clean amplitude** `C_empty` and then
had to pay separately for junk (Lemma Q), and the two prices did not match — the junk relief in the cost
bound, `(2^n − 1) J`, always exceeded the margin damage, `J/2`, that a single Fourier mode can extract.
Here the *whole* table, junk included, is one polynomial and one min; junk is never priced separately.

---

## 3. Proof of Theorem A

Let `N <= n`. Choose an injection `sigma : {1..N} -> {1..n}` and set `u_{sigma(l)} := −conj(v_{l,sigma(l)})`
for each atom `l` (other slots arbitrary). Then atom `l`'s factor at slot `sigma(l)` is
`1 − conj(v_{l,sigma(l)}) v_{l,sigma(l)} = 0`, so **every** term of `h` vanishes and `h(u) = 0`. Hence
`min_{T^n} Re h <= 0`. ∎

At finite `p` the kill points need not be grid points, so run the same choice through Lemma C1 of §5
(`t_j := cos(pi/p) · (−conj(v_{sigma(l),j}))`): every atom then keeps a factor of modulus
`eps = 1 − cos(pi/p)` at its own slot and at most `2` at the others, so
`|h(t)| <= eps 2^{n−1} ‖c‖_1` and `Lambda / M_1 <= eps 2^{n−1} ‖c‖_1`. Since `‖c‖_1 <= ‖c‖_r`, this is
below the target `(n+1)^{−(1−r)/r} ‖c‖_r` as soon as `2^{n−1}(1 − cos(pi/p)) <= (n+1)^{−(1−r)/r}`, i.e.
`p >= 5 / 5 / 11 / 23` for `(n,q) = (2,2) / (2,2.5) / (3,2) / (4,2)`. ∎

Verified (`gopt_verify.py` B): over 200 random configurations per cell, `max R_torus/‖c‖_1` is
`−8.0e−06` (`n=1, N=1`), `−0.199` (`n=2, N=2`), `−0.768` (`n=3, N=3`), `−1.91` (`n=4, N=4`) — never
positive.

---

## 4. Proof of Theorem B

Assume `R := min_{T^n} Re h > 0` (else there is nothing to prove). Write `d^{(j)}_{lm} := |v_lj − v_mj|`.

**(i) The bijection bound.** Fix an atom `l0` and a **bijection** `sigma` from the `n` slots onto the `n`
atoms other than `l0`. Put `u_j := −conj(v_{sigma(j), j})`. Every atom `l != l0` equals `sigma(j)` for
exactly one `j`, so its product contains a zero factor; only `l0` survives:

    R  <=  Re h(u)  <=  |h(u)|  =  |c_l0| prod_j |1 − conj(v_{sigma(j)j}) v_{l0 j}|
                                 =  |c_l0| prod_j d^{(j)}_{sigma(j), l0} .               ... (4.1)

(Checked: `max R / (bijection bound)` over random junk-carrying configurations is `0.9998, 0.9971,
0.9910, 0.9816` at `n = 1..4` — never `> 1`; `gopt_verify.py` C.)

**(ii) Geometric mean over the `n!` bijections.** A pair `(j, l)` with `l != l0` occurs as `sigma(j) = l`
in exactly `(n−1)!` bijections, so the geometric mean of the right side of (4.1) over `sigma` is
`|c_l0| (prod_j prod_{l != l0} d^{(j)}_{l,l0})^{1/n}`. Since a min is at most a geometric mean,

    R  <=  |c_l0| ( prod_j prod_{l != l0} d^{(j)}_{l, l0} )^{1/n}.

**(iii) Product over `l0`, then Hadamard.** Multiplying over `l0 = 1..n+1`, the double product runs over
all *ordered* pairs, i.e. it is the square of the product over unordered pairs:

    R^{n+1}  <=  prod_l |c_l| · ( prod_j V_j² )^{1/n},     V_j := prod_{l<m} |v_lj − v_mj|.

`V_j` is the modulus of the Vandermonde determinant of the `n+1` points `v_1j, ..., v_{n+1,j}` on the unit
circle; each of its `n+1` rows has `n+1` entries of modulus 1, so Hadamard's inequality gives
`V_j <= (n+1)^{(n+1)/2}`, with equality iff the rows are orthogonal, i.e. iff the `v_lj` are a rotation of
the `(n+1)`-st roots of unity. Hence `(prod_j V_j²)^{1/n} <= (n+1)^{n+1}` and

    prod_l |c_l|  >=  ( R / (n+1) )^{n+1}.                                                ... (4.2)

(Checked: `max (R/(n+1))^{n+1} / prod|c_l|` is `0.997, 0.983, 0.942, 0.863` at `n = 1..4`, and
`max V_j / (n+1)^{(n+1)/2} = 1.0000000` — Hadamard is attained, never exceeded.)

**(iv) AM–GM.** For `0 < r <= 1`,

    sum_l |c_l|^r  >=  (n+1) ( prod_l |c_l| )^{r/(n+1)}  >=  (n+1) ( R/(n+1) )^r  =  (n+1)^{1−r} R^r. ∎

(Checked: `max (n+1)^{1−r} R^r / sum|c|^r` over `r in {0.25, 0.4, 0.5, 2/3, 0.8, 1}` is
`0.9996, 0.9986, 0.9963, 0.9927` at `n = 1..4`.)

**Equality.** AM–GM forces all `|c_l|` equal and Hadamard forces each slot's phases to be a rotated copy
of the `(n+1)`-st roots of unity; feeding that back into `h` fixes the relative rotations and gives the
common-phase block. *The constant `(n+1)` in (T) is therefore an extremal-Vandermonde (Fekete) constant:
the `n+1` neurons of one feature must sit at the phases that maximise the Vandermonde product on the
circle, and those are the roots of unity.*

**What is new relative to yesterday.** The same three moves (bijection, geometric mean, Hadamard) were
used yesterday, but applied to the *clean amplitude* `A = C_empty` of a population **assumed junk-free**.
Applying them instead to `R = min Re h` costs nothing and needs no assumption: the polynomial identity
`sum_l c_l prod_j (u_j − v_lj) = A prod_j u_j` that encoded junk-freeness is replaced by the trivial
observation that a kill point makes *every* term vanish, junk and clean alike. That is why the
`2(2^n − 1) sin(pi/p) <= 1` hypothesis of yesterday's Theorem 2 disappears.

---

## 5. Theorem C: the finite-`p` version

(2.1) is a min over the **grid** `mu_p^n`, and `min_{grid} Re h >= min_{T^n} Re h`, so Theorem B does not
by itself give a finite-`p` statement. Two lemmas close most of the gap.

**Lemma C1 (peeling).** For a multi-affine `h` and `rho_0 := cos(pi/p)`,

    min_{u in mu_p^n} Re h(u)  <=  min_{t in (rho_0 T)^n} Re h(t).

*Proof.* Induction on `n`. Write `h = A(u_{-1}) + u_1 B(u_{-1})`. For any `W`, the `p` numbers
`Re(u_1 W)`, `u_1 in mu_p`, include one whose angle is within `pi/p` of `pi`, so
`min_{u_1 in mu_p} Re(u_1 W) <= −cos(pi/p)|W|`, i.e. `min_{u_1 in mu_p} Re h <= Re A − rho_0 |B|
= min_{|t_1| = rho_0} Re h(t_1, u_{-1})`. Minima commute, so
`min_{mu_p^n} Re h <= min_{|t_1| = rho_0} min_{mu_p^{n−1}} Re h(t_1, ·)`, and `h(t_1, ·)` is multi-affine
in `n−1` variables. ∎ *Plain reading: a `p`-point grid on the circle can always be aimed to within `pi/p`
of any direction, which is as good as an exact evaluation at radius `cos(pi/p)`.*

**Lemma C2 (the two extreme classes are worth a factor `cot(pi/p)` on the imaginary part).** Applying
Lemma C1 to `zeta_D h` for `D = 1` and `D = p−1` separately and then taking, for one and the same `t`,
the smaller of the two:

    Lambda / M_1  <=  min_{t in (rho_0 T)^n} [ Re h(t) − cot(pi/p) |Im h(t)| ].            ... (5.1)

(Using `min(x,y) = (x+y)/2 − |x−y|/2` with `x + y = 2 M_1 Re h(t)`, `x − y = 2 sin(2 pi/p) Im h(t)`.)

**Proof of Theorem C.** Drop the (non-negative) `cot` term and run §4 with `t_j := rho_0 ·
(−conj(v_{sigma(j)j}))`. Now atom `sigma(j)`'s factor at slot `j` is `1 − rho_0 = eps` rather than `0`, and
the other moduli satisfy `|1 − rho_0 conj(v_{sigma(j)j}) v_lj| <= d^{(j)}_{sigma(j)l} + eps`. Hence

    Lambda/M_1  <=  |c_l0| prod_j ( d^{(j)}_{sigma(j)l0} + eps )  +  eps 2^{n−1} sum_{l != l0} |c_l|
                <=  |c_l0| prod_j d^{(j)}_{sigma(j)l0}  +  eta_n(eps) ‖c‖_1 ,

with `eta_n(eps) = 2^n[(1+eps/2)^n − 1 + eps/2]`. Steps (ii)–(iv) of §4 then apply to
`Delta' := max(Lambda/M_1 − eta_n(eps)‖c‖_1, 0)`, giving
`sum_l |c_l|^r >= (n+1)^{1−r}(Delta')^r`, and `‖c‖_1 <= ‖c‖_r` for `r <= 1` finishes. ∎

**Lemma C3 (the junk penalty, sharp; recovers yesterday's Theorem 2 in one line).** Let
`J := max_{T != empty} |C_T|` be the largest junk coefficient and `kappa = cot(pi/p)`. Apply the bound
`Lambda/M_1 <= Re E_w[h] − kappa |Im E_w[h]|` (valid for every probability measure `w` on the grid, by
(2.1) and Jensen) to the single-mode density

    w(u) = p^{-n} ( 1 + 2 Re[ lambda conj(u^T) ] ),   |lambda| = 1/2,   T the argmax of |C_T|,

which is non-negative and has `E_w[u^{T'}] = [T' empty] + lambda [T' = T]`, so `E_w[h] = C_empty +
lambda C_T`. Choosing the phase of `lambda` to align `lambda C_T` with `i · sign(Im C_empty)` gives

    Lambda / M_1  <=  Re C_empty − cot(pi/p) ( |Im C_empty| + J/2 )  <=  A − cot(pi/p) J / 2.   ... (5.2)

Feeding (5.2) into the perturbed cost bound `sum_l |c_l|^r >= (n+1)^{1−r}(A − (2^n − 1)J)^r`
(`LEMMA_ATTEMPT` (C'), `N = n+1`) shows that (T) holds **exactly**, with no bracket, as soon as

    cot(pi/p) / 2  >=  2^n − 1,    i.e.   p >= 23 / 47 / 97  for  n = 2 / 3 / 4,

(and if `A − (2^n − 1)J < 0` the same inequality makes `Lambda <= 0`). This is yesterday's Theorem 2
threshold `2(2^n − 1) sin(pi/p) <= 1` (`p >= 19 / 47 / 97`) up to the `cos(pi/p)` between `sin` and `tan`
— obtained here in four lines from the reduction, with no separate phase lemma and no trade-off argument.
*Plain reading: junk of size `J` costs the margin `cot(pi/p) J/2`, which is large because `cot(pi/p) ~ p/pi`;
it buys back at most `(2^n − 1)J` of clean amplitude in the cost; the trade is losing once `p` exceeds
`2 pi (2^n − 1)`.* Theorem C is what covers the remaining, small primes.

Explicit values of the bracket `1 + (n+1)^{(1−r)/r} eta_n(eps)`:

| `n, q` | `p = 11` | `p = 23` | `p = 71` | `p = 113` | `p = 509` |
|---|---|---|---|---|---|
| `n=2, q=2` | 1.4238 | 1.0969 | **1.0102** | **1.0040** | 1.0002 |
| `n=2, q=2.5` | 1.3048 | 1.0697 | 1.0073 | 1.0029 | 1.0001 |
| `n=3, q=2` | 3.6321 | 1.5982 | 1.0627 | 1.0247 | 1.0012 |
| `n=4, q=2` | 19.56 | 5.189 | 1.4380 | 1.1729 | 1.0085 |

Only the entries *below* Lemma C3's threshold are needed: at `p >= 23 / 47 / 97` (`n = 2 / 3 / 4`) the
bracket is replaced by exactly `1`. So the live cells of this table are `n=2: p = 11`; `n=3: p = 11, 23`;
`n=4: p = 11, 23, 71` — and there the bracket is genuinely poor (`1.4` to `19.6`), which is the honest
statement of what is still missing at small `p` and large `n`.

**Is the finite-`p` statement exactly true (bracket = 1) everywhere?** Numerically, yes — and not only at
a single frequency: `gopt_full.py` reaches exactly the target of (T) over whole unrestricted networks
(§10). (2.1)–(2.2) is a *smooth NLP* — every constraint is linear in `c` for fixed phases — so it can be
searched properly rather than by annealed ascent:

    maximise t  s.t.  (1 − cos th_D) Re h(u) + sin(th_D) Im h(u) >= t  for all (u, D),  sum_l |c_l|^r = 1.

`gopt_nlp.py` (SLSQP, analytic Jacobians, ~70 starts per cell including block-seeded and random) returns
`max ratio = 1.000000000000` in **all 50 cells tried** (`p = 5,7,11,13`; `n = 2,3`; `q = 2, 2.5, 3`;
`N = 1..6`), the worst deviation over all of them being `1 − 3.9e−13`, with the optimiser landing on
`n+1` equal-modulus atoms and sending any extras to `0.00000`.
So the `O(p^{-2})` bracket is slack in the *proof*, not in the truth. Closing it is open (§9, item 2).

---

## 6. G1 (more than `n+1` neurons per frequency): reduced to one statement, and settled numerically

The bijection dies at `N > n+1`: with `n` slots one can zero at most `n` atoms, leaving `>= 2` alive with
unknown relative phase.

**What I can prove.** At a maximiser of `R` subject to `sum_l |c_l|^r = 1` there is a probability measure
`mu` on the argmin set `{u : Re h(u) = R}` with, for every atom,

    E_mu[K_l] = R |c_l|^{r−2} conj(c_l),      K_l(u) := prod_j (1 + u_j v_lj),

(multiply by `c_l` and sum to identify the multiplier: `sum_l c_l E_mu[K_l] = E_mu[h]`, whose real part is
`R`). Define the **dual function** `Lambda_mu(x) := E_mu[ prod_j (1 + u_j x_j) ]`, a multi-affine function
with `Lambda_mu(0) = 1` and `Lambda_mu(v_l) = E_mu[K_l]`. Then `|Lambda_mu(v_l)| = R |c_l|^{r−1}` and

    sum_{l=1}^N ( R / |Lambda_mu(v_l)| )^{s} = 1,     s := r/(1−r) > 0,                    ... (6.1)

so `N <= ( ‖Lambda_mu‖_inf / R )^{s}`, a finite bound (e.g. `N <= 48` at `n = 2, q = 2`), but not `n+1`.

**The reduction.** By the maximum principle `‖Lambda_mu‖_inf >= |Lambda_mu(0)| = 1`, with equality iff
`Lambda_mu == 1`, i.e. iff `E_mu[u^T] = 0` for every non-empty `T` — *the dual measure is itself
junk-free*. If the optimal `mu` can be chosen junk-free, then `E_mu[K_l] = 1` for every `l`, so by (6.1)
all `|c_l|` are equal, `|c_l| = N^{−1/r}` and `R = N^{−(1−r)/r}`; this decreases in `N`, and Theorem A
forbids `N <= n`, so the optimum is `N = n+1` with `R = (n+1)^{−(1−r)/r}` — exactly Theorem B, for every
`N`. At the block this holds with `mu` = uniform on `T^n` (`Lambda_mu == 1`).

> **G1 is therefore equivalent to: at a maximiser, the optimal dual measure can be taken with vanishing
> junk moments.** I could not prove it; it is the single clean statement that would close G1 for all `N`.

**Numerical status of G1 (much stronger than yesterday's).**
* `gopt_nlp.py` (proper NLP global search on the *exact finite-`p` margin*, not a proxy): `N = 3, 4, 5, 6`
  at `n = 2` (`p = 5, 7, 11, 13`; `q = 2, 2.5`; 32 cells) and `N = 4, 5, 6` at `n = 3` (`p = 5, 7`;
  `q = 2, 3`; 12 cells) all return ratio `1.000000000000`, extras exactly `0`.
* `gopt_atom.py torus` (annealed ascent on the continuum `R/‖c‖_r`, `N` up to `n+4`): max ratio `0.9944`
  (`n=2`), `0.9827` (`n=3`); extras `~1e−3` and falling.
* `gopt_local.py` (start *at* the block, anneal hard, 5 noise scales x 2–3 seeds, `p = 5..23`): never
  escapes — best escape value `0.9970` (`p=5`), `0.9955` (`p=7`), `0.9904` (`p=11`), `0.9928` (`p=17`) on
  the true margin, and `1 − 2.5e−8` (i.e. the block itself, to optimiser precision) on `min Re h`.

---

## 7. G2 and G3, and the new tool for them

**G2 (impure neurons) — untouched, still open.** Theorems A–C assume each neuron is a pure
single-frequency cosine in all `n+1` of its vectors. That is exactly the equality class of the per-neuron
AM-GM and Parseval steps of the existing upper bound, so a maximiser must be *near* it; "near" is not
"in", and there is still no stability estimate. Outside the class only `gamma_q <= B Gamma_{nu/q}` holds.
The reduction of §2 does not survive impurity: `h` exists only because a pure population's table is a
single multi-affine polynomial in `u`.

**G3 (junk on a `K >= 2` support) — still open; Lemma P is the new handle.** The obstruction is sharp and
worth recording precisely. Write `Psi_D(a) := (1/2)[Phi_D(a) + Phi_{−D}(a)]`. For a pure multi-frequency
population,

    Psi_D(a) = sum_k 2^{−n} (1 − cos(2 pi k D / p)) Re h^{(k)}( u^{(k)}(a) ),

one polynomial per frequency — but the evaluation points `u^{(k)}(a)` are all functions of the *same* row
`a`, so one cannot reach each frequency's own kill point simultaneously, and `min_a` of a sum is not the
sum of the `min_a`'s. Lemma P (§1) is the tool that re-parameterises this: a **product** measure `w = ⊗_j w_j` gives

    margin <= sum_k 2 (1 − cos(2 pi k D/p)) Re h^{(k)}(t(k)),     t_j(k) = hat{w_j}(2k) in the unit disc,

i.e. each frequency is evaluated at its own point, at the price that the points lie strictly inside the
disc and are tied together by the requirement that `(hat{w_j}(2k))_k` be the Fourier coefficients of one
probability measure on `Z_p`. Verified to `3.9e−15` on random nets, with the inequality never violated
(`gopt_prodmeas.py`). What kills the naive use of it: since `h^{(k)}` is multi-affine, the reachable set
of moment vectors is the convex hull of the grid points, so `min over product measures` of the bound is
*exactly* `min over rows a` — no free lunch. Lemma P is therefore a re-parameterisation that makes the
multi-frequency problem tractable-looking, not yet a solution; the missing ingredient is a lower bound on
how much of each frequency's individual deficit can be realised at a common `a`.

---

## 8. Approaches that failed, and exactly where

**F-1. Bounding the clean amplitude and pricing junk separately — works, but only above a threshold in
`p` (and my first version of this entry was wrong; recorded so the error is not repeated).**
The single-mode route (Lemma C3 / yesterday's Lemma Q) gives `Lambda/M_1 <= Re C_empty − cot(pi/p) J/2`
and the perturbed cost bound gives `sum|c_l|^r >= (n+1)^{1−r}(A − (2^n − 1)J)^r`, so the two match iff
`cot(pi/p)/2 >= 2^n − 1`. **I first wrote this comparison having dropped the `cot(pi/p)` factor**, and
concluded (wrongly) that the route was dead for every `n`; it is not — it is exactly yesterday's
Theorem 2, valid for `p >= 19 / 47 / 97`. What *is* true is that it dies as `p` shrinks (the junk penalty
is `O(J/p) · p = O(J)` with a constant `1/2`, the cost relief is `(2^n − 1)J`), and that it gives nothing
on a `K >= 2` support, where `M_1` is replaced by an `O(1)` coverage and the same `O(1/p)`-vs-`O(1)`
mismatch reappears (G3). §2's reduction is what removes the trade-off entirely in the continuum.

**F-2. Mahler measure / geometric mean.** `Re h >= R > 0` on the torus gives `M(h) := exp E[log|h|] >= R`,
and `M` is multiplicative, which looked like the natural home for `prod_l |c_l|`. Dead: for `N = 1`,
`M(h) = |c_1|` while the target needs `<= (n+1)^{−(1−r)/r}|c_1| < |c_1|`. The geometric mean cannot see
that a single atom has a zero on the torus; only the min can. Likewise `M(h) <= ‖h‖_{L²} = (sum_T
|C_T|²)^{1/2}` fails at `N = 1` by `2^{n/2}` — the `L²` norm *rewards* junk as extra energy.

**F-3. Induction on `n` by replacing `|B|` with `Re(e^{i phi} B)`.** Writing `h = A + u_1 B`,
`min_{T^n} Re h = min_{u'} [Re A − |B|] <= min_{u'} min_phi Re[A − e^{i phi}B]`, whose coefficient vector is
`c_l(1 − e^{i phi} v_{l1})`; the `(n−1)`-slot hypothesis then needs
`min_phi sum_l |c_l|^r |1 − e^{i phi} v_{l1}|^r <= ((n+1)/n)^{r−1} sum_l |c_l|^r`. At the block
(`n = 2, r = 2/3`) the left side is `0.9615` and the right side `0.8736`: **the induction is false as
stated**, by 10%. Averaging over `phi` instead of minimising is worse (`A_r = 1.1316` vs `0.8736`).

**F-4. Gauge symmetrisation.** Confirmed exactly circular, as yesterday. Concretely: averaging the network
over the `n+1` gauge phases produces a junk-free `theta_bar` with `P_q(theta_bar) = (n+1)^{1−q/nu}
P_q(theta)` and the same clean part; feeding the *unrestricted* bound into it yields
`gamma_q(theta) <= (n+1)^{(nu−q)/q} B Gamma_s` — worse than the existing bound by the square of the target
factor. The gauge orbit is a set of legitimate networks with *equal cost and different margins*; the
target is an upper bound on the sup over the orbit while symmetrisation controls only the min.

**F-5. Rotating the grid (gauge on the phases).** `v_lj -> lambda_j v_lj` is cost-preserving and turns
`min over mu_p^n` into `min over the coset lambda·mu_p^n`; the min over `lambda` is the torus min, which
Theorem B bounds. But the given network corresponds to one fixed `lambda`, and what (T) needs is the
**sup** over `lambda`, not the min. This is precisely the `O(p^{-2})` gap of Theorem C.

**F-6. Morwani-style dual certificate**, **the domination lemma `sum|c| >= (n+1) max|c|`**, and
**Gram/Carathéodory–Fejér**: all three remain dead exactly as recorded in `LEMMA_ATTEMPT.md` §7 (a–d);
nothing here changes their status. In particular the first is *provably* unable to certify any constant
below 1 for `q < nu`.

---

## 9. What remains open, ranked

1. **G1 for all `N`** — reduce to: *the optimal dual measure can be chosen with vanishing junk moments*
   (§6). Everything else in that argument is proved, and it would upgrade Theorem B to every `N`.
2. **The small-`p` regime (`p` below Lemma C3's threshold `cot(pi/p) >= 2(2^n − 1)`)** — there only
   Theorem C's bracket is available, and it is poor (`1.42` at `n=2, p=11`; `19.6` at `n=4, p=11`), while
   the numerics say the truth is exactly `1`. Two concrete handles, both from §5:
   (a) the `cot(pi/p)|Im h(t)|` term of (5.1) is `~ p/pi` times the imaginary part, and must be shown to
   pay for the `O(p^{-2})` residue that a partial kill leaves; (b) **Lemma C4 (rigidity of realness):**
   on the grid `mean_u |Im h(u) − Im C_empty|² = (1/2) sum_{T != empty} |C_T|²` exactly (verified to `7.8e−16`,
   `gopt_c3.py`), and since the characters `1_T` and `−1_{T'}` coincide only
   for `T = T' = empty` when `p >= 3`, **`Im h == 0` on `mu_p^n` if and only if the population is exactly
   junk-free with a real clean amplitude**. So the hard case is junk that is small *and* almost exactly
   real at the near-kill points; a second-order expansion of the cost around the block in the junk
   directions is what is missing.
3. **G2 (impure neurons)** — quantify: if the per-neuron AM-GM and Parseval slacks are `<= delta`, the
   neuron is `O(sqrt(delta))`-close to pure; then redo §2 with an `O(sqrt(delta))` perturbation of `h`.
4. **G3 (`K >= 2` with junk)** — the only live route is Lemma P plus a statement about how much of each
   frequency's deficit is simultaneously realisable at one row `a`.
5. For the paper, the safe and now considerably stronger claim is: **Theorem A + Theorem B + Theorem C on
   single-frequency supports, with the Fekete/Hadamard reading of the constant `n+1`**, plus the
   whole-network NLP evidence of §10 for the rest (which is the first evidence that speaks to (T) itself
   rather than to a sub-class). This is strictly more than `PROOF_AUDIT` F1's "CONJECTURE" and more than
   yesterday's `C(n+1)`-class theorem, because junk is no longer assumed away and `p` is unconstrained.

---

## 10. Numerical record (all float64, local CPU, no pods, `heracles/FINAL_RUNS/GOPT/`)

| script | what it does | result |
|---|---|---|
| `gopt_atom.py block` | builds the `(n+1)`-block, checks `h == const` and (2.2) | `h` constant to `1.6e−15`; `margin/M_1 = 1.000000000000` at `n = 1..4`, `p = 7,11,23,41` |
| `gopt_atom.py torus` | anneals `R_torus/‖c‖_r`, `n = 2,3`, `q = 2, 2.5, 3`, `N = n−1 .. n+4` | max ratio `0.9944` / `0.9827`; `N <= n` gives `R <= 0`; extras `-> 0` |
| `gopt_atom.py grid` | same on `mu_p^n`, `p = 7,11,17,23` | max ratio `0.953` (heuristic optimiser, below the block) |
| `gopt_local.py` | local escape from the block, 5 noise scales x 2–3 seeds, `p = 5..23`, both objectives | never escapes: best `0.9970 / 0.9955 / 0.9904 / 0.9928` at `p = 5/7/11/17` (true margin), `1 − 2.5e−8` (`min Re h`) |
| **`gopt_nlp.py`** | **the exact single-frequency problem as a smooth NLP** (SLSQP, analytic Jacobians, ~70 starts) `p = 5,7,11,13`, `n = 2,3`, `q = 2, 2.5, 3`, `N = 3..6` (50 cells with `--Ns 1 2`) | **max ratio `1.000000000000` in every cell** (worst `1 − 3.9e−13`), `n+1` equal atoms, extras exactly `0.00000` |
| `gopt_verify.py` A–E | every step of §3–§4 on random junk-carrying configurations, `n = 1..4` | kill residual `<= 7.8e−16`; `R/`bijection bound `<= 0.9998/0.9971/0.9910/0.9816`; `(R/(n+1))^{n+1}/prod|c| <= 0.997/0.983/0.942/0.863`; `V_j/(n+1)^{(n+1)/2} <= 1.0000000`; key inequality `<= 0.9996/0.9986/0.9963/0.9927` |
| `gopt_verify.py` F | the §2 reduction against the brute-force `p^n x p` table, `p = 5..13`, `n = 2,3,4` | relative error `<= 1.1e−15`; the `(2.1)` upper bound never violated |
| **`gopt_full.py`** | **(T) itself: max-margin over whole unrestricted networks as a smooth NLP** (raw `E`, `W`; no purity / junk-free / support restriction), `p = 5, 7`, `n = 2`, `q = 2, 2.5`, widths `6, 9, 12`; `p = 5`, `n = 3` (`nu = 4`), `q = 2, 3`, widths `8, 12`; starts = block on every ladder support + noise + random | **`gamma_q/target = 1.000000000000` in every cell with enough width** (max over all 16 cells `1 − 1.6e−15`, never `> 1`); `m = 6` at `p = 7` falls to `0.84 / 0.74` because the width cannot hold the optimal support |
| `gopt_c3.py` | Lemma C3 / inequality (5.2) on random and block-perturbed configurations, `p = 5..47`, `n = 2,3,4`, 240 trials per cell; plus the Lemma C4 identity | (5.2) never violated (closest approach `1.8e−2` at `p=7, n=2`); C4 identity `mean|Im h − Im C_0|² = ½Σ_{T≠∅}|C_T|²` exact to `7.8e−16` |
| `gopt_nlp.py --Ns 1 2` | Theorem A at finite `p`, `p = 5..13` | max ratio `−1e−15` (`N = 2 = n`), `−3.9 .. −29` (`N = 1`): the sup is exactly `0` |
| `gopt_prodmeas.py` | Lemma P identity + inequality on random nets, `p = 5..13`, `n = 2,3,4`, uniform / tilted / mixture measures | identity relative error `<= 3.9e−15`; `E_w[Psi_D] − margin >= 2.75` in every cell (never negative) |
