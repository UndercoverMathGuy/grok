# The (n+1)-block constant: what is now proved, what is refuted, what is still open

Target of this note (set by the coordinator): for the gated chain and 2 ≤ q < ν, prove, refute, or pin
the constant in

    gamma_q(theta)  <=  (n+1)^{-(nu-q)/q} * B * Gamma_{nu/q}(p),      B = 2 (2 p (n+1))^{-nu/2}.      (T)

**Status: PARTIAL — proved on an explicitly stated class, with the crux (a new finite lemma) proved in
full generality, and with two of the three remaining gaps isolated and one of my own earlier proof
routes refuted.** Nothing here contradicts (T); every numerical attack failed to beat it.

Everything below is float64. Scripts: `heracles/lemma_split.py`, `lemma_family3.py`, `lemma_key.py`,
`lemma_verify.py`, `lemma_ascent.py`. Outputs: `heracles/FINAL_RUNS/LEMMA/`.

---

## 0. Symbols, with a plain reading of each

| symbol | meaning | plain reading |
|---|---|---|
| `p` | odd prime | size of the modulus; the task is `a_1 + ... + a_n = c (mod p)` on all `p^n` rows |
| `n` | number of operands | how many numbers are added |
| `nu = n+1` | homogeneity degree | multiplying every weight by `s` multiplies every logit by `s^nu` |
| `q` in `[2, nu]` | penalty exponent | `q = 2` is plain weight decay, `q = nu` is Morwani's `L_{2,nu}` |
| `r = q/nu` in `(0,1]` | the ratio that runs the whole argument | "how cheap it is to split one feature over several neurons"; `r = 1` means free |
| `e_ij` in `R^p` | neuron `i`'s embedding of operand `j` | one row of one linear layer |
| `w_i` in `R^p` | neuron `i`'s readout | the column of the unembedding this neuron writes to |
| `z(a, c)` | logit of class `c` on row `a` | `sum_i w_i(c) prod_j e_ij(a_j)` |
| `\|theta_i\|^2` | `sum_j \|e_ij\|^2 + \|w_i\|^2` | all of neuron `i`'s weights, squared and added |
| `P_q = sum_i \|theta_i\|^q` | the penalty | what weight decay actually prices |
| `margin` | `min` over all rows and all wrong classes of `z_correct - z_wrong` | the worst-case gap to a mistake |
| `gamma_q = margin / P_q^{nu/q}` | normalised margin | scale-invariant "margin per unit of weight" |
| `g(D)` | mean over rows of `z(row, s(row) - D)`, `s(a) = sum_j a_j` | the part of the logit table that depends only on `(sum of inputs) - (class)`: the *clean* part |
| `nf = (p-1)/2` | number of positive frequencies | `k` and `p-k` are the same cosine |
| `Gamma_s(p)` | `max { min_{D != 0} sum_k x_k (1 - cos(2 pi k D / p)) : x >= 0, sum_k x_k^{1/s} <= 1 }` | best achievable clean margin when features are priced with exponent `1/s`; `s = nu/q` |
| `B = 2 (2 p (n+1))^{-nu/2}` | the constant of the existing upper bound | pure bookkeeping (Parseval, AM-GM, the factor 2 from `+k`/`-k`) |

Two conventions used throughout. `theta_j := 2 pi k a_j / p`, `theta_c := 2 pi k c / p` for a frequency
`k`; and `T` always denotes a subset of the operand indices `{1..n}` (a "sign pattern": the set of
operands whose Fourier sign is flipped).

---

## 1. Exact statement of what is proved

Let **`C(n+1)`** be the class of gated-chain networks such that

* **(pure)** every neuron is a single-frequency cosine in all `n+1` of its vectors:
  `e_ij(a) = r_ij cos(2 pi k_i a / p + alpha_ij)` and `w_i(c) = rho_i cos(2 pi k_i c / p + beta_i)`;
* **(count)** each frequency that is present is carried by exactly `n+1` neurons.

(Plain reading: every neuron is a "pure feature" at one frequency — this is exactly the equality class
of the per-neuron AM-GM and Parseval steps of the existing upper bound, i.e. the only place the bound
could conceivably be attained — and each feature is built out of `n+1` such neurons.)

> **THEOREM 1 (junk-free case; proved, every `n >= 1`, every `2 <= q <= nu`, every `p`).**
> If `theta` is in `C(n+1)` and is exactly junk-free (its logit table is a function of `s(a) - c`
> alone), then
>
>     gamma_q(theta)  <=  (n+1)^{-(nu-q)/q} * B * Gamma_{nu/q}(p),
>
> with equality if and only if `theta` is the `(n+1)`-neuron common-phase block of `THEORY_nsum` §8,
> placed on a maximiser of the programme `Gamma_{nu/q}(p)` — up to the gauge freedoms (global rotation
> of each frequency, permutation of neurons, per-neuron rebalancing).

> **THEOREM 1b (count lower bound; proved, unconditional inside the pure class).**
> A pure, junk-free population at one frequency with non-zero clean amplitude needs **at least `n+1`
> neurons**. With `n` or fewer it is impossible, for any phases and amplitudes.

> **THEOREM 2 (junk allowed, single-frequency support; proved for `2 (2^n - 1) sin(pi/p) <= 1`).**
> If `theta` is in `C(n+1)`, all of its neurons sit at one frequency, and
> `2 (2^n - 1) sin(pi/p) <= 1` (i.e. `p >= 19` for `n = 2`, `p >= 47` for `n = 3`, `p >= 97` for
> `n = 4`), then (T) holds — junk is allowed and is priced correctly.

> **LEMMA Q (quantitative junk lemma; proved, NO restriction on the network at all).**
> For every gated-chain network, every `D != 0` and every non-zero Fourier mode `m` in `Z_p^n`,
>
>     margin(theta)  <=  mean_a[phi_D(a)]  -  |psi_D(m)|,
>     phi_D(a) := z(a, s(a)) - z(a, s(a) - D),     psi_D(m) := mean_a[ phi_D(a) e^{-2 pi i m.a / p} ].
>
> The classical junk lemma (`THEORY_nsum` L1) is the case `m = 0` with the second term dropped. Plain
> reading: the margin is not merely below the *average* gap on a level set, it is below that average by
> at least the size of any single Fourier ripple of the gap across the level set.

The crux of Theorems 1 and 2 is a finite complex-analysis lemma that involves neither `p`, nor the
task, nor a network:

> **KEY LEMMA (proved, every `n >= 1`, arbitrary phases).**
> Let `c_1, ..., c_{n+1}` be complex numbers and `v_lj` (`l = 1..n+1`, `j = 1..n`) unit-modulus complex
> numbers with
>
>     sum_{l=1}^{n+1} c_l prod_{j in T} v_lj  =  A * [T empty]     for every T subset of {1..n}.
>
> Then `prod_l |c_l| >= (|A|/(n+1))^{n+1}`, and consequently, for every `r` in `(0,1]`,
>
>     sum_{l=1}^{n+1} |c_l|^r  >=  (n+1)^{1-r} |A|^r,
>
> with equality iff the `v_lj` are, for each `j`, a rotation of the `(n+1)`-st roots of unity that is
> the *same* rotation pattern across `j`, and all `|c_l| = |A|/(n+1)`.
> Plain reading: to cancel all `2^n - 1` junk sign-patterns while keeping clean amplitude `A`, you must
> spread that amplitude over `n+1` neurons *and* you cannot spread it unevenly to save on cost.

---

## 2. The reduction: from the network to a finite programme

This section is exact algebra; it is checked to `2e-15` by `lemma_verify.py` check A.

**Step 1 (rebalancing is free).** Neuron `i` contributes the rank-one tensor
`(x)_j e_ij  (x)  w_i`. Rescaling its `n+1` factors at fixed product leaves the network function
unchanged and minimises `sum_j ||e_ij||^2 + ||w_i||^2` when all `n+1` norms are equal. So WLOG every
neuron is balanced, and then

    ||theta_i||^2 = (n+1) * (prod_j ||e_ij|| * ||w_i||)^{2/(n+1)}.

**Step 2 (one frequency at a time).** In the pure class a neuron at frequency `k` emits Fourier modes
only inside `{+k, -k}^{n+1}`. Since `p` is prime and `k != +-k'`, different frequencies never collide.
The network therefore splits into independent per-frequency populations.

**Step 3 (atoms).** For a population at frequency `k`, put

    c_l := (prod_j r_lj) * rho_l * e^{i psi_l},   psi_l := sum_j alpha_lj - beta_l,
    v_lj := e^{-2 i alpha_lj}       (unit modulus),
    C_T  := sum_l c_l prod_{j in T} v_lj        for T subset of {1..n}.

Reading: `c_l` is neuron `l`'s signed clean amplitude; `v_lj` records the phase of its `j`-th
embedding; `C_T` is the amplitude of the sign pattern that flips exactly the operands in `T`.
Expanding the product of cosines gives exactly

    z = 2^{-(n+1)} sum_{T} [ C_T exp(i(sum_{j not in T} theta_j - sum_{j in T} theta_j - theta_c)) + c.c. ],

so `C_empty` is the **clean** amplitude (`|ghat(k)| = 2^{-(n+1)} |C_empty|`) and the `2^n - 1` other
`C_T` are the **junk**. Junk-free at this frequency means `C_T = 0` for all `T != empty`.

**Step 4 (cost).** With the balanced choice, `||theta_l||^2 = (n+1)(p/2) |c_l|^{2/(n+1)}`, so

    P_q  =  ((n+1) p / 2)^{q/2} * sum_l |c_l|^{q/nu}  =  ((n+1) p / 2)^{q/2} * sum_l |c_l|^r.

**Step 5 (the programme).** Putting `x_k := 2^{-n} |C_empty^{(k)}|`, the junk lemma + the phase lemma
give `margin <= min_{D != 0} sum_k x_k (1 - cos(2 pi k D / p))`, while Step 4 gives
`P_q >= ((n+1)p/2)^{q/2} 2^{nr} sum_k (sum_{l at k} |c_l|^r)`. Feeding the Key Lemma
(`sum_{l at k} |c_l|^r >= (n+1)^{1-r} |C_empty^{(k)}|^r`) into this and using the definition of
`Gamma_s` with `s = nu/q = 1/r`, and the identity

    2^{-n} * (2/((n+1) p))^{nu/2} = 2 (2 p (n+1))^{-nu/2} = B,

yields exactly (T). So **Theorem 1 is the Key Lemma plus bookkeeping**, and the constant `(n+1)` in (T)
is the constant `(n+1)` of the Key Lemma.

---

## 3. Theorem 1b: fewer than `n+1` neurons per frequency is impossible

All `2^n` constraints at one frequency are equivalent to the single polynomial identity

    sum_l c_l prod_{j=1}^n (u_j - v_lj)  =  A prod_{j=1}^n u_j     for all u in C^n.        (†)

*Proof of the equivalence.* Expand `prod_j (u_j - v_lj) = sum_T (-1)^{|T|} (prod_{j not in T} u_j)
prod_{j in T} v_lj` and match coefficients of the monomials `prod_{j not in T} u_j`; the coefficient of
`prod_j u_j` is `C_empty` and the others are `+-C_T`. ∎

Now suppose there are `N <= n` atoms. Pick a surjection `sigma: {1..n} -> {1..N}` and set
`u_j := v_{sigma(j), j}`. Every atom `l` equals `sigma(j)` for some `j`, so every product on the left
contains a zero factor and the left side vanishes; the right side is `A prod_j v_{sigma(j),j}`, of
modulus `|A|`. Hence `A = 0`. ∎

Plain reading: with `n` slots you can "shoot down" at most `n` neurons at a time; if there are at most
`n` neurons you can shoot them all down at once, and then there is nothing left to carry the clean
signal.

---

## 4. Proof of the KEY LEMMA

Let `N = n+1`. Fix an atom `l0` and let `sigma` be any **bijection** from the `n` operand slots onto
the `n` atoms other than `l0`. Put `u_j := v_{sigma(j), j}` in (†). Every atom `l != l0` is `sigma(j)`
for exactly one `j`, so its product contains the zero factor `(u_j - v_lj)`; only `l0` survives:

    c_{l0} * prod_j (v_{sigma(j),j} - v_{l0,j})  =  A * prod_j v_{sigma(j),j}.

Taking moduli (`|v| = 1`) gives the **bijection identity**

    |c_{l0}| * prod_{j=1}^n d^{(j)}_{sigma(j), l0}  =  |A|,     d^{(j)}_{l,m} := |v_lj - v_mj|.     (S2)

In particular no factor vanishes (else `A = 0`), so within each slot `j` the `n+1` points
`v_1j, ..., v_{n+1,j}` of the unit circle are pairwise distinct, and every `c_l != 0`.

Now take the **geometric mean of (S2) over all `n!` bijections `sigma`**. A fixed pair `(j, l)` with
`l != l0` occurs as `sigma(j) = l` in exactly `(n-1)!` bijections, so

    prod_sigma prod_j d^{(j)}_{sigma(j), l0}  =  prod_j prod_{l != l0} (d^{(j)}_{l,l0})^{(n-1)!},

and since the geometric mean of positive numbers is at least their minimum,

    |c_{l0}|  >=  |A| / ( prod_j prod_{l != l0} d^{(j)}_{l,l0} )^{1/n}.                            (S3)

Multiply (S3) over `l0 = 1..n+1`. The double product over `(l0, l != l0)` runs over all *ordered* pairs,
i.e. it is the square of the product over unordered pairs:

    prod_{l0} |c_{l0}|  >=  |A|^{n+1} * ( prod_{j=1}^n V_j^2 )^{-1/n},     V_j := prod_{l<m} |v_lj - v_mj|.

Finally, `V_j` is the modulus of the Vandermonde determinant of the `n+1` points `v_lj` of the unit
circle; each row `(v_1j^t, ..., v_{n+1,j}^t)` of that matrix has `n+1` entries of modulus 1 and hence
Euclidean norm `sqrt(n+1)`, so **Hadamard's inequality** gives

    V_j  <=  (n+1)^{(n+1)/2},        with equality iff the rows are orthogonal,
                                     i.e. iff the v_lj are a rotation of the (n+1)-st roots of unity.  (S4)

Therefore `prod_j V_j^2 <= (n+1)^{n(n+1)}` and

    prod_l |c_l|  >=  |A|^{n+1} (n+1)^{-(n+1)}  =  ( |A| / (n+1) )^{n+1}.

By AM–GM, for `0 < r <= 1`,

    sum_l |c_l|^r  >=  (n+1) ( prod_l |c_l| )^{r/(n+1)}  >=  (n+1) (|A|/(n+1))^r  =  (n+1)^{1-r} |A|^r. ∎

*Equality.* AM–GM forces all `|c_l|` equal, and (S4) forces each slot's phases to be a rotated copy of
the `(n+1)`-st roots of unity; feeding that back into (†) fixes the relative rotations and gives exactly
the common-phase block `alpha_lj = pi l /(n+1)`, `beta_l = -n pi l /(n+1)` of `THEORY_nsum` §8.

**Why the `(n+1)` is really "Fekete points on a circle".** The only inequality with any content is
Hadamard/(S4): the `n+1` neurons of one feature must sit at phases that maximise the Vandermonde
product, and on the unit circle the maximisers are the roots of unity with product `(n+1)^{(n+1)/2}`.
The constant `(n+1)^{-(nu-q)/q}` in (T) is `(n+1)^{-(1-r)/r}` with `(n+1)` coming from that extremal
Vandermonde. This is a pleasing statement for the paper: *the optimal feature is a Fekete
configuration.*

**A structural by-product (rigidity, `n = 2`, proved and separately verified).** At `n = 2, N = 3`,
(S2) applied to the two bijections available for each `l0` gives three relations that force
`|y_l - y_m| = kappa |x_l - x_m|` for all pairs, where `x_l = v_l1`, `y_l = v_l2`. Three distinct points
of a circle are non-collinear, and similar triangles inscribed in the *same* unit circle must be
congruent, so `kappa = 1` and the `y` configuration is a rotation or a reflection of the `x`
configuration. The reflection case gives `x_l y_l = e^{i phi}` for all `l` and hence
`0 = C_{\{1,2\}} = e^{i phi} A != 0`; so **every junk-free 3-atom population at `n = 2` is
common-phase** (`y_l = e^{i phi} x_l`). `lemma_verify.py` check C: 177 solutions found from random
starts, maximum deviation from common phase `4.3e-10`. The same argument generalises: for `N = n+1` all
`n` slots carry congruent configurations, so junk-free populations are common-phase for every `n`.
(The Key Lemma proof above does not need this, but it explains *why* the answer is a one-parameter
family.)

---

## 5. Proof of LEMMA Q, and Theorem 2

**Lemma Q.** Fix `D != 0` and a mode `m != 0`. For any probability weights `w(a)` on `Z_p^n`,
`margin <= sum_a w(a) phi_D(a)`, because every `phi_D(a)` is a gap to a *wrong* class (`s(a) - D !=
s(a)`). Take

    w(a) = p^{-n} (1 + lambda e^{2 pi i m.a/p} + conj(lambda) e^{-2 pi i m.a/p}),   |lambda| = 1/2.

Then `w >= 0` (the bracket is `1 + 2 Re(...)` with `|2 lambda| = 1`) and `sum_a w(a) = 1`, and

    sum_a w(a) phi_D(a) = mean(phi_D) + 2 Re( lambda * conj(psi_D(m)) ).

Choosing `lambda = -(1/2) psi_D(m)/|psi_D(m)|` makes the second term `-|psi_D(m)|`. ∎
(Checked on random nets, `lemma_verify.py` check E: no violation.)

**What Lemma Q says in the pure class.** Expanding (Step 3) and collecting the level-set gap `phi_D`,
each sign pattern `T != empty` produces its own mode `m_T` (the vector with entry `2k` on `T` and `0`
elsewhere; these are distinct and non-zero because `p` is an odd prime and `4k != 0`), with

    mean(phi_D) = 2^{-n} Re[C_empty (1 - e^{i theta_D})],      |psi_D(m_T)| = 2^{-n} |C_T| |sin(theta_D/2)|.

Averaging the resulting bound over `D` and `-D` (the standard phase-lemma trick, which is needed
because the pointwise phase inequality is false) gives, for every `D != 0` and every `T != empty`,

    margin  <=  2^{-n} [ |C_empty| (1 - cos theta_D)  -  |C_T| |sin(theta_D/2)| ].            (Q-pure)

(Verified to `1e-15` against the true tables, `lemma_verify.py` check A and check F.)

**Theorem 2.** One frequency `k`, `N = n+1` neurons, `A := |C_empty|`, `J := max_{T != empty} |C_T|`.
Since `k D` runs over all non-zero residues, we may take `D` with `|sin(theta_D/2)| = sin(pi/p) =: x0`,
and then `1 - cos theta_D = 2 x0^2`, so (Q-pure) gives

    margin  <=  2^{-n} ( 2 A x0^2 - J x0 ).                                                     (M)

For the cost, run the Key Lemma proof with junk present: in (†) the right-hand side becomes
`sum_T (-1)^{|T|} C_T prod_{j not in T} u_j`, of modulus at least `A - (2^n - 1) J`, so (S2) becomes an
inequality with `|A|` replaced by `A' := A - (2^n - 1) J` and the whole chain goes through:

    sum_l |c_l|^r  >=  (n+1)^{1-r} (A')^r        whenever A' >= 0.                              (C')

If `A' <= 0` then `J >= A/(2^n - 1)` and (M) gives `margin <= 2^{-n} A x0 (2 x0 - 1/(2^n-1)) <= 0`
under the hypothesis `2 (2^n - 1) x0 <= 1`; so `gamma_q <= 0` and (T) is trivially true. If `A' > 0`,
compare (M) with what a junk-free population of clean amplitude `A'` would be allowed
(`2^{-n} A' * 2 x0^2`):

    2 A x0^2 - J x0  <=  2 A' x0^2  <=>  J x0 >= 2 (2^n - 1) J x0^2  <=>  2 (2^n - 1) sin(pi/p) <= 1.

So under that hypothesis the pair (margin, cost) is dominated by the junk-free pair with amplitude `A'`,
and Theorem 1's computation applies verbatim. ∎

Plain reading: the clean margin at the hardest `D` is quadratically small in `sin(pi/p)`, whereas junk
damages the margin only linearly in `sin(pi/p)`. For a single-frequency feature, junk is therefore a
losing trade as soon as `p` is moderately large, and the bound reverts to the junk-free one.
`lemma_verify.py` check F shows the crossover empirically at `p = 11, 23, 41`: the true margin of a
`3`-block with junk ratio `J/|C_empty|` turns negative well before `(Q-pure)`'s ceiling does.

---

## 6. What is NOT proved (the honest gap list)

**G1 — more than `n+1` neurons per frequency.** The Key Lemma is proved for `N = n+1` exactly. The
bijection trick dies at `N > n+1`: with `n` slots one can zero at most `n` atoms, leaving `N - n >= 2`
of them, and the identity degrades to the useless `|A| <= 2^n sum_{l in R} |c_l|`.
*Numerical status (strong):* `lemma_split.py` minimises `sum_l |c_l|^r` over the full `2^n`-constraint
system with **free** phases, `N` up to `n+5`, 200 (and, for `n = 2`, 600) SLSQP restarts each. The
minimum is `(n+1)^{1-r}` to 8 digits at every `(n, q, N)` tested (`n = 1,2,3`; `q = 2, 2.5, 3`;
`r = 0.4..1`), and the minimiser always puts `n+1` equal atoms down (`0.3333...` at `n=2`,
`0.2500...` at `n=3`) and sends the rest to exactly `0.0000`. `lemma_family3.py` walks the *entire*
2-parameter family of 3-atom solutions at `n = 2` and finds the minimum exactly at the symmetric point.
So G1 is a proof gap, not a plausible counterexample.

**G2 — impure neurons.** Theorems 1/2 assume each neuron lives at one frequency with all `n+1` factors
concentrated there. That is exactly the equality class of the per-neuron AM-GM and of Parseval, so a
maximiser of (T) must be *near* it; but "near" is not "in", and I have no stability estimate. Outside
the class only the old bound `gamma_q <= B Gamma_{nu/q}` is available.

**G3 — junk on a multi-frequency support.** Theorem 2 is single-frequency. The trade in §5 works
because the clean margin of a one-frequency feature is `2 A sin^2(pi/p)`, i.e. `O(p^{-2})`, while the
junk penalty is `O(p^{-1})`. With `K >= 2` frequencies on a coverage-optimal support the clean margin
is `Theta(1)` in `p`, so Lemma Q's `O(sin(pi/p))` penalty no longer beats the `O(J M_K)` cost relief
that junk buys. Concretely the sufficient condition becomes
`2 (2^n - 1) sum_{k in S} |sin(pi k D'/p)| <= 1`, which fails for spread supports. A sharper junk lemma
(several modes at once, or the exact `min_a` rather than a single Fourier test function) would be
needed; Lemma Q as stated is the best single-mode version (within the family of Fourier test densities,
putting all the weight on the largest mode is optimal, so no gain is available there).

---

## 7. Approaches that failed, and exactly where

**F-a. Morwani-style dual certificate (tensor spectral norm). Provably cannot work for `q < nu`.**
For any probability measure `pi` on (row, wrong class) pairs, `margin <= <z, Lambda_pi>`, and bounding
each balanced neuron separately gives `gamma_q <= nu^{-nu/2} ||Lambda_pi||_sigma`, where
`||.||_sigma` is the injective (spectral) norm of the `(n+1)`-way tensor. The only lossy step is
`sum_i tau_i^nu <= (sum_i tau_i^q)^{nu/q}`, which is an equality **only at width 1**. For the
`(n+1)`-block on `K` frequencies (`N = (n+1)K` equal neurons) it is loose by exactly
`N^{(nu-q)/q}` — a factor `sqrt(18) = 4.24` at `nu=3, q=2, K=6`. So no certificate of this shape can
ever certify a constant below 1, which is why the recorded attempt came out at `1.009` of a value
already known to be unattainable. At `q = nu` the step is an identity and the certificate is exactly
Morwani's theorem. **This route is closed, not merely hard.**

**F-b. The "domination lemma" route. REFUTED (my own conjecture, killed by an explicit family).**
I first reduced the Key Lemma to `sum_l |c_l| >= (n+1) max_l |c_l|` (which, with the elementary
`sum a^r >= (sum a) (max a)^{r-1}`, implies it). This is TRUE at `n = 1` and FALSE for `n >= 2`. The
counterexample: the constraint system is linear, so one may add any *null* configuration
(all `C_T = 0` including `T = empty`). At `n = 2`, take all first-slot phases equal, `x_l = x`, and
`y_1 = 1, y_2 = e^{i delta}, y_3 = e^{-i delta}` with `c` proportional to
`(y_2 - y_3, y_3 - y_1, y_1 - y_2)`; this is null, with masses `(2 sin delta, 2 sin(delta/2),
2 sin(delta/2))`, so `max/sum -> 1/2` as `delta -> 0`. Adding a large multiple of it to the `3`-block
drives `sum|c|/max|c|` to `2 < 3`. (The objective `sum |c|^r` still rises, which is why the Key Lemma
survives.) Moreover **no linear functional can prove domination**: for `h(v) = sum_T c_T prod_{j in T}
v_j` on the torus one has `||h||_inf = sum_T |c_T|` when the phases are free (direct computation at
`n = 2`: `sup_{v1} [ |c_0 + c_1 v_1| + |c_2 + c_{12} v_1| ]`), so the best ratio
`|h(1)| / ||h||_inf` is `1`, never `n`.

**F-c. Gauge symmetrisation. Exactly circular — and this is informative.**
There is a genuine gauge group: rotating every positive-frequency Fourier coefficient of each embedding
by `e^{i phi}` and of the readout by `e^{i n phi}` is an orthogonal map on each vector, so it preserves
`P_q` exactly and preserves the clean part `g` exactly. Averaging a network over the `n+1` phases
`phi_l = pi l/(n+1)` kills all junk of a pure network and costs exactly `(n+1)^{1-q/nu}` in `P_q`.
Feeding that into the existing bound reproduces the existing bound *identically*: the symmetrisation
factor `(n+1)^{1-q/nu}` raised to `nu/q` is precisely the target factor `(n+1)^{(nu-q)/q}`. So the
target inequality is *equivalent* to the statement "the gauge average is the cheapest way to remove
junk, and removing junk is never a loss" — which is why it needs the Key Lemma and Lemma Q rather than
a soft argument.

**F-d. Gram / Carathéodory–Fejér.** Writing the constraints as `U f = A e_empty` with `U` a
`2^n x N` matrix of unit-modulus entries gives `f = A G^{-1} 1`, `G_{lm} = prod_j (1 + conj(v_lj) v_mj)`,
with the consistency condition `1^T G^{-1} 1 = 1`; the claim becomes `prod_l |(G^{-1}1)_l| >=
(n+1)^{-(n+1)}` with `sum_l (G^{-1}1)_l = 1`, i.e. exactly the *reverse* of AM–GM, so it is a rigidity
statement and no Gram-matrix estimate I tried (trace, `lambda_max`, Toeplitz positivity) is tight
enough: at `n = 2` the `lambda_max` route is loose by a factor 4. The Carathéodory–Fejér variant
(`|A| <= 2 (sum_l a_l - |sum_l a_l z_l|)` at `n = 2`) has slack exactly 2 at the optimum.

**F-e. Partial bound for `N > n+1`.** Removing `n` atoms via (†) gives `|A| <= 2^n sum_{l not removed}
|c_l|`; combined with `sum a^r >= a_min^{r-1} sum a` this yields
`sum_l |c_l|^r >= 2^{-nr} |A|^r r^{-1} (n r/(1-r))^{1-r}`, which at `n=2, r=2/3` is `0.945 |A|^{2/3}`
against the needed `1.442 |A|^{2/3}` — i.e. `0.655` of the conjectured constant, and after inversion
that is *worse* than the trivial bound `B Gamma`. Recorded so nobody redoes it.

---

## 8. Numerical record (all float64, local CPU, no pods)

| script | what it does | result |
|---|---|---|
| `lemma_split.py` | minimise `sum_l |c_l|^r` over the full `2^n`-constraint system, free phases, `N = 2..n+5`, `n = 1..4`, `q in {2, 2.5, 3}`, 200–600 SLSQP restarts | min `= (n+1)^{1-r}` to 8 digits wherever `N >= n+1`; infeasible for `N < n+1` (Theorem 1b); minimiser always `n+1` equal atoms, extras exactly 0 |
| `lemma_family3.py` | walks the entire 2-parameter family of 3-atom junk-free configurations at `n=2` (49 427 solutions) | `min sum|c|^r / 3^{1-r} = 1.00013` at grid resolution, argmin at `sigma = (2.111, 4.172)` vs the exact `(2pi/3, 4pi/3)`, `|c| = (0.3302, 0.3350, 0.3350)` vs `1/3` |
| `lemma_key.py` | every step (S1)–(S5) of the Key Lemma proof, `n = 1..6` | constraint residuals `< 1.2e-11`; bijection identity (S2) residual `< 5e-11`; `prod|c|/(A/(n+1))^{n+1} >= 1.000` always; `sum|c|^r/((n+1)^{1-r}A^r) >= 1.000` at `r = 0.2, 0.5, 2/3, 0.8, 1`; `V_j/(n+1)^{(n+1)/2} <= 1` always |
| `lemma_verify.py` A | network `<->` atom programme dictionary, `p = 11,13`, `n = 2,3` | max error `3.2e-15` |
| `lemma_verify.py` B | `(n+1)`-block attains `(n+1)^{-(nu-q)/q} B Gamma` exactly, `n = 2,3,4`, `q = 2, 2.5`, 1/2/3-frequency supports | ratio `1.0000000000` in all 18 cells |
| `lemma_verify.py` C | rigidity at `n = 2, N = 3` | 177 solutions, all common-phase to `4.3e-10` |
| `lemma_verify.py` D | Hadamard/Fekete `prod_{l<m}|z_l - z_m| <= N^{N/2}` | random search never exceeds; roots of unity attain exactly, `N = 2..6` |
| `lemma_verify.py` E | Lemma Q on random nets | no violation |
| `lemma_verify.py` F | junk budget vs `2 sin(pi/p)`, `p = 11, 23, 41` | (Q-pure) holds in all 18 cells; its ceiling changes sign at `J/|C_empty| ~ 2 sin(pi/p)` as predicted, and the true margin changes sign earlier |
| `lemma_ascent.py` | direct annealed maximisation of `gamma_q` over networks: `p = 7, 11, 13`, `n = 2`, `q = 2`, widths `3..15`, 8–12 seeds each, random + warm starts | **nothing beat (T).** Best value from a *random* start: `0.9796` of the conjectured sup at `p = 7` (`ascent_p7.log`), `0.418` at `p = 11`, `0.162` at `p = 13` — the annealed optimiser degrades badly as `p` grows, so only `p = 7` is a strong test |
| `lemma_ascent.py --local` | ascent started *at* the exact `(n+1)`-block (noise `1e-6`, cold temperature): does the optimum leak upward? | `p = 7`: returns exactly `1.00000000` of the conjectured sup, i.e. the block is a local maximum and 3000 Adam steps + polish find no escape. `p = 11` drifts *down* to `0.418` (optimiser, not geometry) |

---

## 9. What I would do next (ranked)

1. **Close G1.** The natural statement is: *the minimum of `sum_l |c_l|^r` over the constraint system is
   attained at `N = n+1`.* A promising line is the first-order condition derived here: at an interior
   minimiser there is a multilinear dual `Lambda(v) = sum_T conj(lambda_T) prod_{j in T} v_j` with
   `|Lambda(v_l)| = r |c_l|^{r-1}` at every atom and `sum_l |c_l|^r = conj(lambda_empty) A / r`; since
   `Lambda` is bounded on the torus this already forces every atom to be bounded *below*, which is
   incompatible with the numerically observed "extras go to zero" unless `N = n+1`. Making that
   rigorous is the shortest path to a clean, unrestricted theorem for pure junk-free nets.
2. **Close G2 by stability.** Quantify: if the per-neuron AM-GM and Parseval slacks are `<= eps`, the
   neuron is `O(sqrt(eps))`-close to pure; then re-run §2 with an `O(sqrt(eps))` perturbation.
3. **Sharpen Lemma Q for G3** by using a positive test density with several modes and the true
   `min_a` (the single-mode version is optimal only within the one-mode family).
4. For the paper, the safe claim is Theorem 1 + Theorem 1b + the Fekete reading, stated on `C(n+1)`,
   with the `lemma_split`/`lemma_ascent` numerics carrying the rest. That is strictly more than
   `PROOF_AUDIT` F1's "CONJECTURE", and the Fekete/Hadamard mechanism is a genuinely new and quotable
   reason for the number `n+1`.
