# Impure neurons: the cross-frequency-blind certificate family, its exact limit, and what is left

Scope: n = 2 operands, q = 2 (plain weight decay), so nu = 3, r = q/nu = 2/3. Full modular-addition table
mod an odd prime p. Local CPU only. Scripts and logs are in this directory; nothing outside it was changed.

## 0. Symbols

| symbol | meaning | plain reading |
|---|---|---|
| `Z(x1,x2,c)` | the network's logit tensor | logit of class `c` on input `(x1,x2)` |
| `gap(x,D)` | `z_y(x) - z_{y+D}(x)`, `y = x1+x2`, `D != 0` | how far the right class beats the wrong class `y+D` on row `x` |
| margin `m` | `min over x, D of gap(x,D)` | the worst row/wrong-class pair |
| `Zhat(a,b,c')` | Fourier coefficient of `Z` at slot frequencies `(a,b,c')` | one "component" of the logits |
| row mode `(u,v)` | `(a+c', b+c')` | how that component varies across rows; `(0,0)` means it is the same on every row |
| clean | `a = b = -c'` | row mode `(0,0)`; this is the feature itself |
| same-`|k|` junk | `|a| = |b| = |c'| = k` but not clean | the leftover sign patterns of a feature; row modes `(2k,0), (0,2k), (2k,2k)` |
| cross-frequency part `X` | every component whose three frequencies are not all equal in `|.|` | what IMPURE neurons (mixed-frequency embeddings) add on top of per-frequency atoms |
| test distribution `pi(x,D)` | a probability on (row, wrong offset) pairs | any such `pi` gives the bound `m <= E_pi gap` |
| `D/2` | `D * 2^{-1} mod p` | half the offset, well defined since `p` is odd |

Why impurity is the hard part: splitting each neuron into its per-frequency atoms and applying Hölder
charges the network only for its *diagonal* Fourier mass, so in that accounting `X` looks **free**. If
`X` really were free it could cancel every junk dip, and the cheap `sqrt 3` ceiling would be sharp.
Any proof has to either never look at `X`, or charge for it. This note settles how far the first
option goes.

## 1. Theorem (the complete cross-frequency-blind family)

A component `(a,b,c')` enters `E_pi gap` with weight

    F(u,v; c') = pihat(u,v,0) - pihat(u,v,c'),       (u,v) = (a+c', b+c'),

where `pihat(u,v,d) = sum_{x,D} pi(x,D) e(-u x1 - v x2 - d D)` is the joint Fourier transform.

**Theorem 1.** `E_pi gap` is independent of `X` for every logit tensor if and only if `pi` is a mixture of

* (clean) row-uniform distributions with arbitrary offset weights `omega(D)`, and
* (type T, for `T` in `{1}, {2}, {1,2}`) the **half-offset line densities**: `D` uniform on `Z_p \ {0}`,
  and the row chosen so that `x1 + D/2` (type {1}), `x2 + D/2` (type {2}) or `x1 + x2 + D/2`
  (type {1,2}) is distributed by an arbitrary density `rho` on `Z_p`, with the other coordinate uniform.

For such `pi`, with `J_{k,T}` the type-`T` junk coefficient of frequency `k` and `G_k` the clean one:

    E_pi gap = (1-lam) sum_D omega(D) Clean(D)
             + lam * p/(p-1) * sum_{k != 0} [ G_k - J_{k,T} rhohat(2k) ],

where `Clean(D) = sum_k G_k (1 - e(kD))` and `lam` is the weight of the type-`T` part. Mixing all three
types adds nothing beyond separate mixtures.

*Proof.* Blindness means `F(u,v;c') = 0` for every component in `X`. Fix a row mode `(u,v) != (0,0)`. A
same-`|k|` component `(u-d, v-d, d)` exists only if `u, v` are both in `{0, 2d}`, so a mode carries at
most one non-`X` readout frequency `d(u,v)`, and it carries one only for modes `(2d,0), (0,2d), (2d,2d)`.
Blindness forces `pihat(u,v,.)` to be constant off that one frequency. Transforming back in `D`, the
row-mode-`(u,v)` part of `pi(.,D)` equals `beta_{uv} e(d D)` for every `D != 0`, with a magnitude that
does not depend on `D`, and it vanishes for every other mode. Hence
`pi(x,D) = p^{-2}[omega(D) + f_1(x1 + D/2) + f_2(x2 + D/2) + f_3(x1 + x2 + D/2)]`, where each
`f_T` has mean zero. For fixed `D` the three arguments range independently, so nonnegativity
constrains only the sum of the three minima, and `E_pi gap` is linear in the `f_T`. The extreme points
are therefore single-type point aims, which is the mixture statement. The displayed formula is a direct
evaluation: the uniform `D`-average turns `e(-(a+c')D/2) - e((c'-a)D/2)` into `p[1{a = -c'} - 1{a = c'}]`,
which keeps only clean and type-`T` junk. ∎

**Check** (`xblind_check.py`): random full tensors at `p = 5, 7, 11`, all three types, random `rho`. The
value on the full tensor equals the value on its same-`|k|` projection to `1e-16`. A product density used
as a control is not blind: the two values differ at the first digit.

**Consequence.** Any margin certificate built from this family applies verbatim to impure networks,
because the per-frequency atoms (Hölder split) are all it sees.

## 2. The family is provably too weak on its own

The aimed part is forced to use **uniform** offset weights, so its baseline is the average gap over all
`D`. That exceeds the worst `D`, which is the thing the ansatz optimizes. Relaxation R maximizes the best
bound from this family over pure networks, normalized by `P^{3/2}` (`xblind_relax.py`, p = 11, 18
seeds). The table shows the bound divided by the ansatz value, with the true margin of the same network:

| neurons per frequency | best relaxed bound / ansatz | true margin / ansatz of that net |
|---|---|---|
| 1 | 0.787 | -3.09 |
| 2 | **1.300** | -0.97 |
| 3 | 1.299 | -0.96 |

So the blind family alone certifies at most about `1.30x` the ansatz at `p = 11`. That is better than the
old unrestricted `sqrt 3 = 1.732`, but it is not `1`. Networks with 2–3 junky atoms per frequency
satisfy every blind constraint while having negative true margins. Their junk dips land on
wrong-class/row pairs that only product-type (X-visible) densities can see.

## 3. Real impure networks cannot use that room

The optimal skeletons from §2 were given 30 extra free neurons (mixed frequencies allowed, every weight
free) and the TRUE margin was maximized (`repair.py`, `repair_diag.py`, p = 11):

| skeleton | relaxed bound | true margin at start | repaired true margin / ansatz |
|---|---|---|---|
| 1 | 1.221 | -1.33 | 0.9992 |
| 2 | 1.299 | -0.96 | 0.9952 |
| 4 | 1.235 | -1.20 | 0.9994 |
| 7 | 1.220 | -1.06 | 0.9993 |
| 10 | 1.300 | -0.97 | 0.9951 |
| 14 | 1.225 | -0.76 | 0.9951 |

Anatomy of the repaired nets (energy shares of the centered logit tensor):

| skeleton | before: X / junk / support | after: X / junk / support |
|---|---|---|
| 2 | 0.000 / 0.584 / {1,2,3,5} | **0.0000 / 0.0000** / {1,2,5} |
| 10 | 0.000 / 0.583 / {1,3,4,5} | **0.0000 / 0.0000** / {1,3,5} |

The optimizer never builds `X` to cancel junk. It deletes the junk and ends as a pure junk-free block on
a near-optimal support. (`{1,2,5}` and `{1,3,5}` are dilation images of each other; the certified
optimum at `p = 11` is `{1,3}` with `K = 2`, and these three-frequency supports sit at `0.995`.)

## 4. What is left, stated as one lemma

Everything above says that impurity can matter only through the product-type (X-visible) certificates,
and that in practice `X` is never worth building. The accounting shows why it costs something. A
neuron's cross terms come either from

1. **Hölder slack**: mass on frequencies not shared by all three slots. For example `u` at `a`, while `v`
   and `w` sit at `c' != a`. This pays its full cost `a_i^r` and earns no diagonal atom at all.
2. **Hölder-tight spreading**: proportional profiles over a frequency set `F`. This is free in the
   budget, but that one neuron is then a single atom at every frequency in `F`. It deposits full-size
   junk (`|J_T| = |G|`) at each of them, and that junk must itself be cancelled. Producing `X` at the
   junk mode `(2k, 2k')` also needs arithmetic coincidences in `F` (e.g. `g = 2k - f`).

> **Charging lemma (open).** For the product aiming densities of P3C (singleton partition, depth `A`),
> the `X`-visible part of `E_pi gap` is at most what the same cost would buy as clean amplitude.
> Case 1 is paid by the slack. Case 2 is paid by the junk it deposits at the frequencies in `F`.

With that lemma the impure case reduces to P3C's pure-class inequality (PFP) plus aiming depth. Without
it, §1–§2 give a rigorous reduction and a numerical ceiling: every network, impure included, is below
`sup R` times the ansatz, where `sup R` is a pure-class quantity measured at about `1.30` at `p = 11`.
Certifying `sup R` rigorously would already be the first impure-inclusive ceiling below `sqrt 3`.
