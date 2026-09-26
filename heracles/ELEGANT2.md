# Junk, elegantly: the aimed-output cost theorem

Seventh pass on the global-optimality programme. Continues `GOPT_STATE.md`, `GOPT_STATE2.md`,
`GOPT_STATE3.md` and the write-ups they point to (`ELEGANT.md`, `P3C.md`, `P2B.md`, `P3A.md`,
`P3B.md`, `P2.md`, `GLOBAL_OPT4.md`, `VERIFY.md`, `THEORY_nsum.md`). Brief: find an **elegant**
proof that junk cannot help, replacing the partial, computer-assisted machinery
(`Cl2_r(eta)` + Lemma Q + the 129 179-box certificate of `P2B`, the aiming depths `A*` of `P3C`).

Local CPU, fp64, no pods, `$0`. New files only: `ELEGANT2.md`, `eleg2_core.py`, `eleg2_probe.py`,
`eleg2_spec.py`, `eleg2_thmA.py`, `eleg2_net.py`, `eleg2_genN.py`, `eleg2_routes.py`,
`eleg2_extra.py`, `eleg2_det.py`; outputs in `FINAL_RUNS/ELEG2/`. No background jobs remain at
hand-back. Nothing inherited was edited; no commits.

---

## 1. Headline

**The junk parameter `eta` is the wrong variable, and so is the clean amplitude `A`.** The right
one is the feature's **aimed output**

    hmin  :=  min over u in T^n of |h(u)| ,       h(u) := sum_l c_l prod_j (1 + u_j v_lj) ,

"the least this feature can be made to deliver by probing the operands with a phase vector `u`".
In that variable the whole of the junk problem at one frequency becomes a single inequality that
contains `P1` as its junk-free case, and it is **proved outright at `N = n+1` atoms** in four lines,
sharp, with no `eta`, no case split, no certificate and no constant to lose:

> **THEOREM A (Aimed-Output Cost).** *Let `n >= 1`, let `v_lj` (`l = 1..n+1`, `j = 1..n`) be complex
> numbers of modulus 1 and `c_1, ..., c_{n+1}` complex. Then for every `r in (0,1]`*
>
>     sum_{l=1}^{n+1} |c_l|^r  >=  (n+1)^{1-r} * hmin^r ,       hmin = min_{u in T^n} |h(u)| .
>
> *In particular, since `|h| >= Re h`, the same holds with `hmin` replaced by
> `(min_{u in T^n} Re h(u))_+`.*
>
> *Plain reading: a feature carried by `n+1` neurons costs at least `(n+1)^{1-r}` times its
> **worst-case** output, whatever junk it carries. Junk-free means `h` is the constant `A`, and
> then the statement is exactly the Key Lemma / `P1` at `N = n+1`. Any junk makes `h` non-constant,
> and a non-constant multi-affine polynomial can be aimed at: what it loses at the aiming point is
> at least what its cheapness gains.*

Proof: the kill-point/Hadamard chain identity of `P2` §2.1 says
`log[(n+1)^{n+1} prod_l |c_l|] = (n+1) H + Delta` with `H` the average of `log|h|` over the
`(n+1)!` kill points and `Delta >= 0` the Hadamard defect of the slot phases. Every kill point is a
torus point, so `|h| >= hmin` there, so `H >= log hmin`; drop `Delta >= 0`; take AM–GM. Written out
in §3.

**What this replaces.** For `N = n+1` it retires, in one statement: `P2`'s Theorem P2-1 and QPL;
`P2B`'s Lemma M3 (Möbius), Lemma M4 (cross-slot), Lemma M5 (`rho = 1.15`, 129 179 interval boxes),
Proposition M2, Theorem P2B-1, Corollary P2B-2 and Proposition P2B-3 — i.e. the entire quadratic
junk-cost-curve programme at `n = 2` **and** its unproved extension to `n >= 3` (`P2B` F2B-4).
It is not a sharpening of `Cl2_r(eta)`: it says the `eta`-parametrisation was the wrong reduction.

**What it does not do.** Two gaps remain, both stated exactly, both with hard numerics:

* **`N > n+1`** — the ladder `e_k(|c|) >= binom(n+1,k) (hmin/(n+1))^k` (Conjecture **C2**, §5) is
  true in every test and tight, but the determinantal route that closed the junk-free case in
  `ELEGANT.md` is **REFUTED here** (§6.1): the chain matrix's determinant can be made arbitrarily
  small compared with `hmin^{n+1}` (infimum `0` at `n = 1` and `n = 2`), and its spectrum and
  numerical range leave the half-plane `Re >= hmin` altogether.
* **aiming depth `A < 1`** — `hmin` is a minimum over the *full* torus, whereas a network on
  `Z_p^n` can only aim to depth `A*(p,S) < 1` (`P3C` §3.2). §4 converts Theorem A into a
  network statement at a cost of `(1 - c(n,q)/p^2)^{-r}`, which is the `1 + O(p^{-2})` blemish of
  Theorem C — now, however, for **every** `n`, **every** `q`, and **with junk**.

Everything below is written so that each numbered check can be re-run from `FINAL_RUNS/ELEG2/`.

---

## 2. Symbols, with a plain reading of each

Inherited symbols keep their meaning. The ones this note leans on:

| symbol | meaning | plain reading |
|---|---|---|
| `n` | number of operand slots | how many numbers are added |
| `D = nu = n+1` | homogeneity degree = chain dimension | scaling all weights by `s` scales logits by `s^nu` |
| `q in [2,nu]`, `r = q/nu in (0,1]` | penalty exponent, its ratio | `q = 2` is plain weight decay; `r < 1` is why splitting a feature over neurons costs |
| `N` | atoms at one frequency | one atom per neuron that touches this frequency (no purity needed) |
| `c_l in C`, `v_lj in T` | atom `l`'s amplitude and slot-`j` phase | that neuron's signed contribution, and the phase of its `j`-th embedding |
| `C_T = sum_l c_l prod_{j in T} v_lj` | sign-pattern amplitude | `T = empty` is the clean cosine-of-the-sum; the other `2^n - 1` are junk |
| `A = C_empty` | clean amplitude | the feature's useful Fourier coefficient |
| `h(u) = sum_l c_l prod_j (1 + u_j v_lj)` | the feature's generating polynomial | `h = sum_T C_T u^T`; **junk-free iff `h` is constant** |
| `hmin = min_{u in T^n} |h(u)|` | the **aimed output** | the least this feature delivers when a probe is aimed at it |
| `m = min_{u in T^n} Re h(u)` | the signed aimed output | `hmin >= m`, so Theorem A in `hmin` is the stronger statement |
| `m_A = min_{|u_j| = A} Re h(u)` | aimed output at **depth `A`** | what a probe that can only aim to depth `A` can take away |
| kill point `u^{l0,sigma}` | `u_j = -conj(v_{sigma(j),j})`, `sigma` a bijection slots -> atoms `!= l0` | the torus point that annihilates every atom except `l0` |
| `V_j = prod_{a<b} |v_aj - v_bj|` | slot `j`'s Vandermonde product | how spread out slot `j`'s phases are; maximal at roots of unity |
| `Delta = D log D - (1/n) sum_j log V_j^2` | **Hadamard defect**, `>= 0` | how far the slot phases are from rotated `D`-th roots of unity |
| `H` | mean of `log|h|` over the `D!` kill points | what the surviving atom sees there |
| `e_k(x)` | `k`-th elementary symmetric function | — |
| `kappa0 = (n+1)^{(1-r)/r}` | `= Cl2_r(0)^{1/r}` | how much clean amplitude one unit of cost buys at zero junk |

Normalisation: statements are homogeneous of degree `r` in `c` and gauge-invariant under
`v_lj -> e^{i psi_j} v_lj` combined with `u_j -> e^{-i psi_j} u_j`, so nothing is lost by scaling.

---

## 3. Theorem A, proved

### 3.1 Two degeneracy lemmas

> **LEMMA A0 (collapse).** *For any number of atoms: if `hmin > 0` then (i) **at least `n+1`** of
> the `c_l` are non-zero; and if moreover `N = n+1`, then (ii) in every slot `j` the `n+1` phases
> `v_{1j}, ..., v_{n+1,j}` are pairwise distinct.*
>
> *Plain reading: a feature that cannot be aimed down to zero needs at least `n+1` neurons, and if
> it has exactly `n+1` then no two of them may agree in any one operand.*

*Proof.* (i) If at most `n` atoms have `c_l != 0`, choose an injection `sigma` from a set of slots
onto them and put `u_j = -conj(v_{sigma(j),j})` on those slots (`u_j = 1` elsewhere). Then atom
`sigma(j)` has the factor `1 + u_j v_{sigma(j),j} = 1 - |v_{sigma(j),j}|^2 = 0`, so every non-zero
atom is annihilated and `h(u) = 0`, i.e. `hmin = 0`. *(This is agent 2's Theorem A, and is the
`r -> 0` limit of Theorem A.)*

(ii) Suppose `v_aj = v_bj` with `a != b`. Take `l0 := b` and any bijection `sigma` from the slots
onto the `n` atoms `!= b` with `sigma(j) = a`. At `u^{l0,sigma}` every atom `!= b` is annihilated by
its own slot, and `b` is annihilated too because `u_j = -conj(v_aj) = -conj(v_bj)`. So `h(u) = 0`. ∎

### 3.2 The chain identity

> **LEMMA A1 (chain identity; `P2` §2.1, re-derived).** Let `N = n+1 = D` and suppose all slot
> phases are distinct in each slot (Lemma A0). Then
>
>     log[ D^D prod_{l=1}^{D} |c_l| ]  =  D * H  +  Delta ,
>     H     = mean over the D! kill points of log|h| ,
>     Delta = D log D - (1/n) sum_{j=1}^n log V_j^2   >=  0 .
>
> *Plain reading: the product of the atom moduli is exactly what the `D!` kill points see, corrected
> by how far the slot phases are from Fekete (roots-of-unity) position. With no junk `h` is the
> constant `A`, `H = log|A|`, and the identity IS the Key Lemma.*

*Proof.* At `u = u^{l0,sigma}` every atom `l != l0` is annihilated (the factor at slot
`j = sigma^{-1}(l)` vanishes), so

    h(u^{l0,sigma}) = c_{l0} prod_{j=1}^n ( 1 - conj(v_{sigma(j),j}) v_{l0 j} ) ,

and for unit-modulus `a, b` one has `|1 - conj(a) b| = |conj(a)| |a - b| = |a - b|`, so

    log|h(u^{l0,sigma})| = log|c_{l0}| + sum_j log|v_{sigma(j),j} - v_{l0 j}| .

Fix `l0` and average over the `n!` bijections `sigma`: a given pair `(j,l)` with `l != l0` occurs as
`sigma(j) = l` in exactly `(n-1)!` of them, i.e. with frequency `1/n`, so the average is
`log|c_{l0}| + (1/n) sum_j sum_{l != l0} log|v_{lj} - v_{l0 j}|`. Now average over the `D` choices
of `l0`; the ordered double sum over `(l0, l != l0)` is twice the unordered one, i.e.
`log V_j^2`. Hence

    D * H = sum_l log|c_l| + (1/n) sum_j log V_j^2 .

Add `D log D` to both sides and move the Vandermonde term across:
`log[D^D prod_l |c_l|] = D H + D log D - (1/n) sum_j log V_j^2 = D H + Delta`.

`Delta >= 0` is Hadamard's inequality: `V_j^2 = |det W|^2` with `W_{a,l} = v_{lj}^a`
(`a = 0..n`), a matrix each of whose rows has Euclidean norm `sqrt D`, so `|det W|^2 <= D^D`,
`log V_j^2 <= D log D`, and averaging over the `n` slots gives `Delta >= 0`. ∎

### 3.3 Proof of Theorem A

If `hmin = 0` there is nothing to prove. So let `hmin > 0`; by Lemma A0 all `c_l != 0` and all slot
phases are distinct in each slot, so Lemma A1 applies (both `H` and `Delta` are finite). Every kill point lies on `T^n`, so `|h| >= hmin` there and therefore `H >= log hmin`.
Dropping `Delta >= 0`,

    log[ D^D prod_l |c_l| ]  =  D H + Delta  >=  D log hmin ,     i.e.  prod_l |c_l| >= (hmin/D)^D .

By AM–GM applied to the `D` non-negative numbers `|c_l|^r`,

    sum_l |c_l|^r  >=  D ( prod_l |c_l|^r )^{1/D}  =  D ( prod_l |c_l| )^{r/D}
                   >=  D (hmin/D)^r  =  D^{1-r} hmin^r .  ∎

*(The `|h| >= Re h` remark is immediate: `hmin >= (min Re h)_+`.)*

### 3.4 Equality

Equality forces, simultaneously, (a) all `|c_l|` equal — AM–GM; (b) `Delta = 0`, i.e. in every slot
the `n+1` phases are a rotation of the `(n+1)`-st roots of unity (Hadamard's equality case: the rows
of `W` are orthogonal, which for a Vandermonde on the unit circle means the nodes are a rotated
regular `(n+1)`-gon); (c) `|h| = hmin` at **all** `(n+1)!` kill points. The
share-weighted common-phase block satisfies all three with `h` constant — but so does the **pure
junk** feature `h(u) = lambda u^T`, which is the block's image under a genuine symmetry of the
problem:

> **SLOT-CONJUGATION TWIST.** For any `T subset [n]`, the map
> `(c_l, v_lj) |-> (c_l prod_{j in T} v_lj , conj(v_lj) for j in T, v_lj otherwise)`
> leaves every `|c_l|` unchanged and sends `h(u)` to `u^T h(u~)`, where `u~_j = conj(u_j)` for
> `j in T` and `u_j` otherwise. Since `u |-> u~` is a bijection of `T^n` and `|u^T| = 1`, it leaves
> `hmin` unchanged as well.
>
> *Proof.* `1 + u_j conj(v_lj) = conj(v_lj)(v_lj + u_j) = conj(v_lj) u_j (1 + v_lj conj(u_j))` for
> `|u_j| = 1`, and the prefactor `prod_{j in T} v_lj` cancels the `conj(v_lj)`'s. ∎
>
> Applied to the block (`h == A`) with `T = {1}` it gives `h = A u_1`: a feature that is **all
> junk** and still exactly optimal. *(Checked: `n = 1`, `c = (1/2,-1/2)`, `v = (1,-1)` gives
> `h = u`, `hmin = 1`, cost `2^{1-r}` — equality in Theorem A.)*

So the equality class is the block's orbit under this `2^n`-element group together with the usual
gauge; whether it is exactly that is not needed for Theorem A and is left open. Numerically, every
adversarial minimiser found has `n+1` atoms of (near-)equal modulus (§8.1).

*This twist also explains why no statement in the programme can be phrased in terms of `|A|` alone:
it moves all of a feature's mass from `C_empty` to `C_T` at zero cost.*

### 3.5 What Theorem A is, in the language of the existing programme

* It is exactly `P3C`'s criterion **(PFP) with the singleton partition at aiming depth `A = 1`**,
  proved, for `N = n+1` populations — `P3C` §7 item 1 conjectured that "at `A = 1` it *is* P1" and
  left it open; Theorem A is that statement, with junk, at `N = n+1`.
* It contains `ELEGANT.md` Theorem 1 at `N = n+1` (junk-free `=> h == A`, `hmin = |A|`).
* It contains agent 2's Theorem A ("`N <= n` ⇒ margin `<= 0`") as the case `r -> 0` of Lemma A0(i).
* At `r = 1` it says `sum_l |c_l| >= hmin`, which is *not* the triangle inequality (the naive bound
  goes the wrong way, `|h(u)| <= 2^n sum_l |c_l|`); it is the `k = 1` rung, proved for every `n` and
  every `N` as Lemma A5 in §4.

---

## 4. Two structural lemmas that come free (and replace Lemma Q)

> **LEMMA A2 (polydisc extension).** *If `Re h >= m` on the torus `T^n` then `Re h >= m` on the
> whole closed polydisc `D^n`. In particular, if `m > 0` then `h` has no zero in `D^n`: **a feature
> with positive aimed output is a stable multi-affine polynomial**.*

*Proof.* `h` is affine in each variable separately. Fix `u_2, ..., u_n in T` and write
`h = a + b u_1`; then `min_{|u_1| <= 1} Re(a + b u_1) = Re a - |b| = min_{|u_1| = 1} Re h`, so the
bound on the circle already holds on the disc. Now fix `u_1 in D` and `u_3, ..., u_n in T`: `h` is
affine in `u_2` and, by the previous step, `Re h >= m` for `u_2 in T`; hence `Re h >= m` for
`u_2 in D`. Iterating over the `n` slots gives the claim. ∎

> **LEMMA A3 (junk is bounded by the aiming loss).** *`|C_T| <= 2 (Re A - m)` for every non-empty
> `T`, where `m = min_{T^n} Re h` and `A = C_empty`.*
>
> *Plain reading: no junk coefficient can exceed twice the amount a probe can take off the clean
> amplitude. This is the `Cl2_r` bookkeeping run backwards, and it is one line.*

*Proof.* `g := h - m` has `Re g >= 0` on `T^n`. As a function on `T^n`, `Re g = (g + conj g)/2` is
non-negative, and its Fourier coefficients are: `Re A - m` at the zero multi-index, `C_T / 2` at the
multi-index `1_T` (`T != empty`, coming from `g`), and `conj(C_T)/2` at `-1_T` (from `conj g`) —
`T` and `-T` never collide for `T != empty`. A non-negative function `f` on a compact abelian group
satisfies `|f-hat(xi)| <= f-hat(0)`, so `|C_T|/2 <= Re A - m`. ∎

*(Lemma A3 is the clean form of `Lemma Q` / the "Ripple Bound" of `GOPT_STATE`: it says the margin
penalty is at least half the largest junk coefficient, with no test-measure construction.)*

> **LEMMA A5 (the aimed output never exceeds the largest coefficient; every `n`, every `N`).**
>
>     hmin  =  min_{u in T^n} |h(u)|   <=   max_{T subset [n]} |C_T|   <=   sum_l |c_l| .
>
> *Plain reading: a feature can always be aimed down to at most its own largest Fourier
> coefficient. This is the `k = 1` rung of Conjecture C2 (§6), proved outright.*

*Proof.* Induction on `n`. For `n = 0`, `h = A = C_empty` and there is nothing to prove. For
`n >= 1` split off the last slot: `h(u) = h_0(u') + u_n h_1(u')` with `u' = (u_1..u_{n-1})`,
`h_0 = sum_l c_l prod_{j<n}(1 + u_j v_lj)` (coefficients `C_T`, `T subset [n-1]`) and
`h_1 = sum_l c_l v_{ln} prod_{j<n}(1 + u_j v_lj)` (coefficients `C_{T u {n}}`). For fixed `u'`,
`min_{|u_n| = 1}|h| = | |h_0(u')| - |h_1(u')| |`, so

    hmin  =  min_{u' in T^{n-1}} | |h_0(u')| - |h_1(u')| | .

The continuous function `|h_0| - |h_1|` on the connected set `T^{n-1}` either vanishes somewhere —
and then `hmin = 0` — or has a constant sign. If `|h_0| >= |h_1|` throughout, evaluate at a point
`u'_0` minimising `|h_0|`: `hmin <= |h_0(u'_0)| - |h_1(u'_0)| <= |h_0(u'_0)| = hmin(h_0)`, and the
induction hypothesis for `h_0` gives `hmin(h_0) <= max_{T subset [n-1]}|C_T|`. If `|h_1| >= |h_0|`
throughout, the same with `h_1` gives `hmin <= max_{T subset [n-1]}|C_{T u {n}}|`. Either way
`hmin <= max_T |C_T|`, and `|C_T| <= sum_l |c_l|` always. ∎

*Equality is attained (§8 measures `max_T|C_T| / hmin = 1.000000000` exactly), e.g. by
`h(u) = lambda u^T`, which is the "pure junk" feature; and by the block, where `h == A`.*

---

## 5. The network corollary at `K = 1`, with junk

Theorem A lives on the **full torus**. A network on `Z_p^n` can only aim with a **non-negative
density on `Z_p^n`**, and for a single frequency `k` the achievable moment vectors are exactly
`(prod_{j in T} t_j)_T` with `t_j` in the convex hull of the `p`-th roots of unity — a regular
`p`-gon of inradius `A_p := cos(pi/p)` (`P3A` §4.1, `P3C` §3.2). So the object the margin sees is

    m_A  :=  min over |t_j| = A_p  of  Re h(t)      (the aimed output at depth A_p),

and `m_A >= m` (Lemma A2), i.e. the grid is *weaker* than the torus. The gap is `O(p^{-2})`:

> **LEMMA A4 (depth bridge).** `m_A <= m + E`, `E := sum_{T != empty} (1 - A_p^{|T|}) |C_T|
> <= (2^n - (1 + A_p)^n) * sum_l |c_l|`.

*Proof.* Let `u*` attain `m` on `T^n`. The point `t = A_p u*` has `|t_j| = A_p`, so it is admissible
and `m_A <= Re h(t)`. Now
`h(t) - h(u*) = sum_{T != empty} (A_p^{|T|} - 1) C_T (u*)^T`, so
`Re h(t) <= Re h(u*) + sum_{T != empty}(1 - A_p^{|T|})|C_T| = m + E`.
Each `|C_T| <= sum_l|c_l|`, and `sum_{T != empty}(1 - A_p^{|T|}) = sum_{k=1}^n binom(n,k)(1-A_p^k)
= 2^n - (1+A_p)^n`. ∎

> **COROLLARY N1 (`K = 1`, any junk, width `<= n+1`).** *Let `p` be an odd prime, `A_p = cos(pi/p)`,
> `q in [2, nu]`, `r = q/nu`, `kappa0 = (n+1)^{(nu-q)/q}`, and*
>
>     eps_p  :=  kappa0 * ( 2^n - (1 + A_p)^n )    <  1 .
>
> *Then every gated-chain network on the full `p^n` table whose neurons are pure cosines at one
> frequency `k`, of width at most `n+1`, satisfies*
>
>     gamma_q  <=  (1 - eps_p)^{-1} * (n+1)^{-(nu-q)/q} * B * Gamma_{nu/q}(p) ,
>     B = 2 (2 p (n+1))^{-nu/2} ,
>
> *with no hypothesis on its junk. With Conjecture C1 (§6) the width restriction disappears.*

*Proof.* Write `S := sum_l |c_l|` and `cost := sum_l |c_l|^r`.

*(a) Margin.* `P3A`'s profile identity gives `2^n margin = min_{a, D != 0} Re[W_{kD} h(u(a))]`,
`W_{kD} = 1 - omega^{-kD}`. Replacing the point mass at the worst row by the product density with
block depths `t_j` (legal: each slot's density on `Z_p` realises any `t_j` with `|t_j| = A_p`) and
averaging the pair `+-D` with `kD = +-1` gives `2^n margin <= M_1 * m_A`, `M_1 = 1 - cos(2 pi/p)`.
*(Checked against brute-force logit tables, §8 (N1),(N2).)*

*(b) Cost.* If `m_A <= 0` the bound of (a) makes `margin <= 0` and there is nothing to prove, so let
`m_A > 0`, and write `S = sum_l |c_l|`. Two branches.
*Branch 1, `S >= kappa0 m_A`:* by subadditivity of `t -> t^r` (`r <= 1`),
`cost >= S^r >= kappa0^r m_A^r`, and `kappa0^r = (n+1)^{1-r}`.
*Branch 2, `S < kappa0 m_A`:* then `E <= (2^n - (1+A_p)^n) S < eps_p m_A` by Lemma A4, so
`m >= m_A - E > (1 - eps_p) m_A > 0`. Since `m > 0`, Lemma A0(i) forces at least `n+1` non-zero
atoms; the width is at most `n+1`, so it is exactly `n+1` and Theorem A applies:
`cost >= (n+1)^{1-r} m^r >= (n+1)^{1-r}(1-eps_p)^r m_A^r`.
Either way `cost^{1/r} >= kappa0 (1 - eps_p) m_A`.

*(c) Budget.* `P_q >= ((n+1) p/2)^{q/2} cost` (`LEMMA_ATTEMPT` §2 Step 4), and
`2^{-n}((n+1)p/2)^{-nu/2} = B`. Hence

    gamma_q = margin / P_q^{nu/q} <= B * (2^n margin) / cost^{1/r}
            <= B M_1 m_A / (kappa0 (1-eps_p) m_A) = (1-eps_p)^{-1} (n+1)^{-(nu-q)/q} B M_1 ,

and `Gamma_{nu/q}(p) >= M_1` (put the whole share on the single frequency). ∎

**The threshold `eps_p < 1` in numbers** (`q = 2`):

| `nu` | `n` | `q` | `kappa0` | smallest `p` with `eps_p < 1` | `eps_p` at `p = 23` | at `p = 113` |
|---|---|---|---|---|---|---|
| 3 | 2 | 2 | 1.732051 | **7** | 0.06438 | 0.002677 |
| 3 | 2 | 2.5 | 1.245731 | **5** | 0.04630 | 0.001925 |
| 4 | 3 | 2 | 4 | **17** | 0.44500 | 0.018546 |
| 4 | 3 | 3 | 1.587401 | **11** | 0.17660 | 0.007360 |
| 5 | 4 | 2 | 11.180340 | **43** | 3.30909 (vacuous) | 0.138218 |

*(computed in `FINAL_RUNS/ELEG2/epsp.json`)*

*(For comparison, agent 2's Theorem C — the previous `K = 1`-with-junk statement — needed
`p >= 23 / 47 / 97` at `nu = 3/4/5`, was proved only at `q = 2`, and rested on a case analysis.
Corollary N1 holds at every `q in [2,nu]`, at every `n`, and is three lines given Theorem A.)*

**Where `(1 - eps_p)^{-1}` comes from and why it is not removable by this argument.** `m_A > m` is
a genuine inequality: the grid cannot aim as deeply as the torus. The equality case is unaffected
(at the block `h` is constant, so `m_A = m = |A|` for every depth, and §8 (N4) records
`gamma_q / target = 1.000000000000` there), but a *sharp* `K = 1` statement needs Theorem A at depth
`A < 1`, which is exactly the open problem of §7.

---

## 6. `N > n+1`: the ladder, and why the determinant route dies

> **CONJECTURE C2 (the symmetric ladder).** *For every `N`, every `n`, with `D = n+1` and
> `hmin = min_{T^n}|h|`,*
>
>     e_k(|c_1|, ..., |c_N|)  >=  binom(D, k) (hmin/D)^k ,      k = 1, ..., D .

> **CONJECTURE C1.** `sum_l |c_l|^r >= (n+1)^{1-r} hmin^r` *for every `N` and every `r in (0,1]`.*

`C2 => C1`, exactly as in `ELEGANT.md`: with `x_l := D|c_l|/hmin`, `C2` says `e_k(x) >= binom(D,k)`
for `k <= D`, hence `prod_l (1 + x_l t) >= (1+t)^D` for `t >= 0`, and integrating the logarithm
against `t^{-1-r} dt` through the Bernstein representation
`x^r = (r sin(pi r)/pi) int_0^inf log(1+xt) t^{-1-r} dt` gives `sum_l x_l^r >= D`.
Theorem A is `C2` at `N = D` and `k = D` (plus AM–GM). `C1` at `r = 1` is `k = 1`;
`C1` as `r -> 0` is Lemma A0(i) generalised ("at least `n+1` atoms").

> **THEOREM B (`n = 1`, every `N`).** *At `n = 1` (`D = 2`), `C2` and hence `C1` hold for every
> number of atoms `N` and every `r in (0,1]`:* `sum_l |c_l|^r >= 2^{1-r} hmin^r`.

*Proof.* Write `x_l = |c_l|`, `S = sum_l x_l`, `A = sum_l c_l`, `C_1 = sum_l c_l v_l`, so
`h(u) = A + C_1 u` and `hmin = | |A| - |C_1| |`.

*Rung `k = 1`.* `S >= |A|` and `S >= |C_1|`, so `S >= max(|A|,|C_1|) >= hmin = binom(2,1)(hmin/2)`.

*Peeling.* For each `l`, evaluate at `u = -conj(v_l)`, which annihilates atom `l`:
`h(u) = sum_{l' != l} c_{l'}(1 - conj(v_l) v_{l'})`, and `|1 - conj(a)b| <= 2` for unit `a,b`, so
`hmin <= 2 sum_{l' != l} x_{l'}`, i.e. `S - x_l >= hmin/2` for **every** `l`; in particular for
`l = argmax`.

*Rung `k = 2`.* `e_2 = (S^2 - sum_l x_l^2)/2 >= (S^2 - x_max S)/2 = S (S - x_max)/2
>= hmin * (hmin/2)/2 = (hmin/2)^2 = binom(2,2)(hmin/2)^2`.

*Conclusion.* `prod_l (1 + x_l t) = sum_k e_k t^k >= 1 + 2(hmin/2)t + (hmin/2)^2 t^2
= (1 + (hmin/2) t)^2` for `t >= 0`; take logarithms, integrate against the Bernstein measure, and
get `sum_l x_l^r >= 2 (hmin/2)^r = 2^{1-r} hmin^r`. ∎

**Why this does not generalise, exactly.** The peeling step loses a factor `2^n`: a kill point
annihilates the atoms it targets but multiplies each survivor by at most
`prod_j |1 + u_j v_lj| <= 2^n`, so it only gives `sum_{l not in J} |c_l| >= hmin/2^n` for a targeted
set `J` of at most `n` atoms. The block, however, leaves exactly `hmin/(n+1)` behind. At `n = 1`
the two agree, `2^n = 2 = n+1`, and that coincidence is the whole reason the argument is sharp
there. From `n = 2` the crude peel is short by `2^n/(n+1) = 4/3`, and even the *sharp* peeling
constants do not close the gap — this is `ELEGANT.md`'s "peeling cannot be sharp"
(`D_1(2) in [0.6495, 0.6536] < 2/3`).

**And the other natural induction loses the `+1` too.** Lemma A5's slot-peeling shows that one of
the two `(n-1)`-slot populations `h_0, h_1` — which have the **same** atom moduli `x_l` — satisfies
`hmin(h_branch) >= hmin(h)`. So an induction on `n` delivers `e_k(x) >= binom(n,k)(hmin/n)^k`,
whereas the target is `binom(n+1,k)(hmin/(n+1))^k`. The ratio is
`[(n+1-k)/(n+1)] * [(n+1)/n]^k`, which is `1` at `k = 1` (so the induction *does* give Lemma A5)
and `< 1` from `k = 2` — at `n = 2, k = 2` it is `(1/3)(3/2)^2 = 3/4`. **The `+1` is bought only by
the kill-point/Hadamard mechanism, and that mechanism needs `N = n+1` exactly**: with more atoms no
torus point isolates a single amplitude.

**Status for `n >= 2`, `N > n+1`: OPEN, and numerically tight.**
`eleg2_probe.py` / `eleg2_spec.py` / `eleg2_genN.py`,
`n = 1..4`, `N` up to `2(n+1)`, `r` over ten exponents, random + adversarial: the minimum of
`C1` is `1.000000009` and of `C2` is `1.000000179`, never below `1`, and every minimiser has
`n+1` atoms of equal modulus (§8).

### 6.1 REFUTED: the determinant route (this is the one that closed the junk-free case)

`ELEGANT.md` proved `C2` in the junk-free case from the chain matrix `M = sum_l c_l m_l m_l^*`:
Cauchy–Binet plus Hadamard give `|E_k(M)| <= D^k e_k(|c|)`, and junk-freeness makes `M` triangular
with constant diagonal `A`, so `E_k(M) = binom(D,k) A^k`. The obvious robust version is

    (C3)   |det(I + s M)| >= (1 + s m)^D   for all s >= 0 ,      equivalently, at s -> infinity,
           |det M| >= m^D ,

which with Cauchy–Binet and Bernstein would give `C1` for every `N` in two lines. **It is false.**
At `n = 1`, `D = 2`, the chain matrix is

    M = [[A, Cbar_1], [C_1, A]] ,   C_1 = sum_l c_l v_l ,  Cbar_1 = sum_l c_l conj(v_l) ,
    det M = A^2 - C_1 Cbar_1 ,      m = Re A - |C_1| ,

and the **mirror** moment `Cbar_1` is not constrained by anything (`ELEGANT.md` §5.2). A direct
search (`eleg2_det.py`, starts from random *junk-free* populations so that `hmin > 0` is guaranteed)
drives `|det M| / hmin^2` **to 0** already at `N = 3`, and the whole `s`-family down to
`0.0104 .. 0.125`; the same happens at `n = 2` (`D = 3`, `N = 3, 4`: infimum `0`, `s`-family `0.2122` and `0.3606`).
The two terms of `det M` cancel while `hmin` stays `O(1)`. A recorded witness at `N = 4`, `n = 1`:

    c = (-0.332186-0.356499i, 0.867622+0.028669i, 0.990393-0.138278i, -0.600629+0.473790i)
    v = (-0.999776-0.021151i, 0.642281-0.766469i, -0.477696-0.878525i, 0.891389-0.453239i)
    A = 0.925200+0.007681i,  C_1 = -0.011441-0.392617i,  Cbar_1 = -0.226826+2.119056i
    m = 0.532417 ,  det M = 0.021364-0.050599i ,  |det M| / m^2 = 0.193758 .

**So `C2` is true but the chain matrix does not certify it.** This is the precise sense in which
junk is harder than the junk-free case: triangularity is an all-or-nothing property, and there is no
"approximately triangular" version, because `U'` (the mirror block) is `O(1)` even at zero junk
(`P2B` §8 measured `||U'||` up to `7.23` on *exactly junk-free* families).

### 6.2 REFUTED: spectrum and numerical range

`C3` would follow from `Re lambda >= m` for every eigenvalue of `M` (then
`|1 + s lambda| >= 1 + s Re lambda >= 1 + sm`), which in turn would follow from
`Herm(M) = (M + M^*)/2 >= m I`. Both are false, and badly:

| `n` | `min Re(spec M) / m` | `min lambda_min(Herm M) / m` |
|---|---|---|
| 1 | `-350.71` | `-413.95` |
| 2 | `-179.04` | `-1176.74` |
| 3 | `+0.2189` | `-1506.89` |

*(worst over `eleg2_spec.py` and `eleg2_routes.py`, random populations with `m > 0`; §8.4.
Even at `n = 3`, where the spectrum happened to stay to the right of `0`, it is a factor `4.6` to
the left of `m`.)* The reason is structural and worth stating, because it
also explains why the ELEGANT proof cannot be perturbed: `Herm(M) = sum_l Re(c_l) m_l m_l^*` only
sees the real parts of the amplitudes, and `psi^* M psi / ||psi||^2` for `psi` the chain vector of
`w in T^n` equals `A + (1/D) sum_{intervals I} [ w^I Cbar_I + conj(w)^I C_I ]` — the **mirror**
moments `Cbar_I` enter the quadratic form on exactly the same footing as the junk `C_I`, and they
are unconstrained.

### 6.3 REFUTED: the diagonal restriction

The obvious way to make everything one-variable is to restrict to `u_1 = ... = u_n = z`, turning `h`
into the degree-`n` polynomial `H(z) = sum_l c_l prod_j (1 + z v_lj) = sum_k (sum_{|T| = k} C_T) z^k`
and `hmin` into `min_{|z| = 1}|H(z)|`. This is a *stronger* statement (the diagonal minimum is at
least the torus minimum), and it is **false already in the junk-free direction**: "junk-free on the
diagonal" is only the `n` conditions `sum_{|T| = k} C_T = 0`, which `n` atoms can satisfy, giving
cost `n^{1-r}` (at `n = 2, r = 2/3`: `2^{1/3} = 1.2599`) against the target `(n+1)^{1-r} = 1.4422`.
Measured in §8. So the diagonal carries strictly too little information, at every `n >= 2`.

### 6.4 REFUTED (again, and now precisely): gauge symmetrisation

`GOPT_STATE` lists "gauge symmetrisation is circular"; here is the exact accounting, which also
answers the question whether replacing `sum|c|^r` by `E_k(M)` rescues it (it does not). Average the
population over the `D` rotations `v_lj -> omega^s v_lj`, `omega = e^{2 pi i/D}`, giving `DN` atoms
`c_l/D`. Every junk pattern dies (`C_T -> C_T * (1/D) sum_s omega^{s|T|} = 0` because
`1 <= |T| <= n < D`), so the averaged population is exactly junk-free and `P1` applies to it. But
its cost is

    sum_{s,l} |c_l/D|^r  =  D * D^{-r} * sum_l |c_l|^r  =  D^{1-r} * cost ,

which is **exactly** the factor to be proved, so `P1` on the average returns `cost >= |A|^r` — the
triangle inequality. Running it through the `E_k` route changes nothing: the averaged chain matrix
is `A I` so `E_k = binom(D,k)A^k`, and Cauchy–Binet then bounds `e_k` of the `DN`-atom multiset
`{|c_l|/D, each D times}`, whose generating product is `prod_l (1 + |c_l|t/|A|)^D >= (1+t)^D`, i.e.
`prod_l (1 + |c_l|t/|A|) >= 1 + t` — the triangle inequality again. Measured: junk of the average
`<= 1e-15`, cost ratio `= D^{1-r}` to `1e-15` (§8). **The symmetrisation is circular at every level
of the argument, not just at the `sum|c|^r` level.**

### 6.5 What Theorem A does *not* improve: the `eta` curve

Theorem A implies, for `N = n+1`,
`Cl2_r(eta) >= (n+1)^{1-r}(1 - sqrt(2^n - 1) eta)_+^r` (since `hmin >= |A| - sum_{T != empty}|C_T|`
and Cauchy–Schwarz), which is **exactly** `P2`'s Theorem P2-1 — the *linear* minorant whose induced
junk threshold is `0`. It does **not** reproduce `P2B`'s quadratic `exp(-1.74 r eta^2)`.
This is not a defect: it is the statement that `eta` throws away the phase information that decides
the trade. In the variable `hmin` the same populations obey a *sharp* inequality with the block as
the equality case, and no curve is needed. `P3C` reached the same conclusion from the other side
(its criterion never mentions `eta`); Theorem A is the proof that the `eta`-free formulation is the
right one at `N = n+1`.

---

## 7. The one thing that would close `K >= 2`: Theorem A at depth `A < 1`

`P3C`'s THEOREM P3C-1 reduces `(T)` for every pure network, every `p`, every `K`, every junk level
to the single-population inequality `R_P(pop, A*) <= 1` for a mixture of partitions. For the
**singleton** partition that inequality is literally

    (DEPTH-A)     sum_l |c_l|^r  >=  (n+1)^{1-r} (m_A)_+^r ,     m_A = min_{|u_j| = A} Re h(u) .

Theorem A is (DEPTH-A) at `A = 1`. The thresholds that matter:

| cell | needed `A` (singleton only, `P3C` F6-1) | available `A*` |
|---|---|---|
| `nu = 3, q = 2` | `0.428372` (binding at the two-atom family `M_2`) | `>= 1/2` on dissociate supports (`P3A` §4.1, proved) |
| `nu = 4, q = 2` | `0.629961` (binding at `M_3`) | `>= 1/2` — **not enough**, whence `P3C`'s mixture |

So **(DEPTH-A) at `A = 1/2` would close `nu = 3, q = 2` for every `p`, every `K`, every junk level
and every width on dissociate supports, with the singleton density alone** — replacing `P3C`'s
partition mixture, its LP, and its measured `A_lam`. The kill-point proof does not survive the
deformation: at depth `A` the point `u_j = -A conj(v_{sigma(j),j})` multiplies atom `sigma(j)` by
`(1-A)` instead of annihilating it, so the linear system stops being diagonal and the chain identity
has no analogue. Measured thresholds for `N = n+1` populations are in §8.

---

## 8. Numerical record

fp64, local CPU, no pods, `$0`. Scripts `heracles/eleg2_*.py`, outputs `heracles/FINAL_RUNS/ELEG2/`.

    uv run --with scipy python eleg2_probe.py 3 25     # first probe of C1/C2/C3 + identities
    uv run --with scipy python eleg2_spec.py 4 120     # spectrum / numerical range / C1 / C2 / C3
    uv run --with scipy python eleg2_thmA.py           # Theorem A: identities + adversary
    uv run --with scipy python eleg2_net.py --p 7 11 13 --n 2 --qs 2.0 2.5 --trials 30
    uv run --with scipy python eleg2_routes.py         # the refuted routes
    uv run --with scipy python eleg2_genN.py --mode genN
    uv run --with scipy python eleg2_genN.py --mode depth
    uv run --with scipy python eleg2_extra.py          # Theorem B ingredients, Lemma A5, R1 redone
    uv run --with scipy python eleg2_det.py --ns 1 2 3 --Ns 3 4 5 6   # the determinant refutation

Every statement below is a worst case over the whole sweep it describes; `>= 1` is the required
direction for a conjecture, `~ 0` for an identity.

### 8.1 Theorem A's ingredients (`eleg2_thmA.py`, `FINAL_RUNS/ELEG2/thmA.json`, `thmA.log`)

`n = 1..5`, `N = n+1`, three samplers (`generic`, `junkfree`, `nearjf`), 30/30/12/5 populations
per cell:

| check | worst value |
|---|---|
| (V1) chain identity `log[D^D prod|c_l|] - (D H + Delta)` (Lemma A1) | `9.97e-13` |
| (V3) `min over kill points of |h|` minus `hmin` (must be `>= 0`; this is the proof's only step) | `+7.78e-16` |
| (V4) `sum_l|c_l|^r / ((n+1)^{1-r} hmin^r)` over nine exponents `r in [0.05, 1]` | `1.000000123` |
| (V6) `hmin` by coordinate descent vs a dense torus grid (`G = 400` at `n=1`, `60` at `n=2`) | agree or descent lower (max gap `0.0482`, always on the descent's side) |
| (V8) collapse: `hmin` after zeroing an atom, or after making two atoms share a slot phase (Lemma A0) | `3.79e-06` (i.e. `0`) |

Adversarial (Nelder–Mead on the ratio itself, 5 restarts, block-perturbed and free starts):

| `n` | `r` | version | min ratio | `|c|` at the minimiser (normalised) |
|---|---|---|---|---|
| 1 | 1/3 | `hmin` | `1.000000000001` | `(1.000, 1.000)` |
| 1 | 2/3 | `hmin` | `1.000000000000` | `(1.000, 1.000)` |
| 2 | 1/3 | `hmin` | `1.002658731966` | `(0.950, 0.899, 1.000)` |
| 2 | 2/3 | `hmin` | `1.008814704063` | `(0.901, 0.894, 1.000)` |
| 2 | 0.9 | `hmin` | `1.000492240809` | `(0.995, 1.000, 0.974)` |
| 3 | 1/3 | `hmin` | `1.002441991651` | `(1.000, 0.994, 0.928, 0.932)` |
| 3 | 2/3 | `hmin` | `1.054097045615` | `(1.000, 0.822, 0.663, 0.821)` |
| 1–3 | 1/3, 2/3, 0.9 | `(min Re h)_+` | `1.000000216 .. 1.157040` | equal moduli |

**Never below 1, and the minimiser always has `n+1` atoms of (near-)equal modulus** — Theorem A's
equality case.

### 8.2 The network corollary (`eleg2_net.py`, `net_n2.json`)

`p = 7, 11, 13`; `n = 2`; `q = 2, 2.5`; 25 networks per cell (perturbed blocks and free random pure
networks with `N = 1..3` neurons at a random frequency `k`), every margin computed by brute force
over all `p^n` rows and all `p` classes:

| check | worst |
|---|---|
| (N1) profile identity `2^n margin == min_{a,D} Re[W_{kD} h(u(a))]`, relative | `1.08e-13` |
| (N2) aiming step `2^n margin <= M_1 m_A`, `A = cos(pi/p)` | `1.0000000000` (equality at the block) |
| (N4) `gamma_q / [(n+1)^{-(nu-q)/q} B Gamma]` — the **unrelaxed** target | `1.0000000000` |
| (N4) `gamma_q` / relaxed target `(1-eps_p)^{-1} x target` | `0.8563` |

So `(T)` itself is never violated by these junk-carrying `K = 1` networks (it is *attained* by the
block), and Corollary N1's relaxed bound has the slack the `eps_p` table predicts.

### 8.3 The general-`N` conjectures (`eleg2_probe.py`, `eleg2_spec.py`, `eleg2_genN.py`)

* `eleg2_probe.py`, `n = 1..3`, `N in {D-1, D, D+1, D+3, 2D}`, three samplers, 25 populations per
  cell, `r` over eight exponents: `min C1 = 1.0000061`, `min C2 = 1.0001221`,
  chain identity `8.9e-16`, kill-point check `-2.2e-16`.
* `eleg2_spec.py`, `n = 1..3`, 1 515 populations with `hmin > 0`: `min C1 = 1.000000009`,
  `min C2 = 1.000000179`.
* `eleg2_genN.py --mode genN`, adversarial, `n = 2` (`N = 4,5,6,8`) and `n = 3` (`N = 4,5`):
  every minimum `>= 1` — `n=2`: `1.0127 / 1.0048 / 1.0071` (`N=4`, `r=1/3` / `r=2/3` / `C2`),
  `1.0408 / 1.0411 / 1.0012` (`N=5`), `1.0005 / 1.0179` (`N=6`), `1.0343 / 1.0088` (`N=8`);
  `n=3`: `1.0475 / 1.2810` (`N=4`), `1.0230` (`N=5`) — and **the optimiser always drives the
  surplus atoms to `0`** (the support at the minimiser is `n+1`, e.g. `n=2, N=6, r=1/3`:
  `|c|/max = (0.977, 1.000, 0.994, 0, 0, 0)`). That is the strongest structural evidence that
  `N > n+1` never helps.
* Lemma A5 measured (`eleg2_extra.py` E2): `min max_T|C_T| / hmin = 1.000000000` **exactly** at
  `n = 1, 2, 3` — the lemma is tight — and `min (sum_l|c_l|)/hmin = 1.000014 / 1.0664 / 1.2218`.
* Theorem B's three ingredients (`eleg2_extra.py` E1, `n = 1`, `N` up to 12, 2 400 populations):
  `S/hmin >= 1.1024`, `(S - x_max)/(hmin/2) >= 1.0132`, `e_2/(hmin/2)^2 >= 1.2073`, and the
  theorem itself `>= 1.0047`.

### 8.3b The slot-conjugation twist and the depth thresholds

* The twist of §3.4, checked on `n = 1,2,3`, `N` up to `2n+2`, all `2^n` subsets `T`, random torus
  points: the identity `h'(u) = u^T h(u~)` holds to **`1.5e-14`** relative
  (`FINAL_RUNS/ELEG2/twist.json`), so `hmin` is exactly preserved (the `0.028` in the same file is
  coordinate-descent noise in the two independent `hmin` solves, not a discrepancy — the identity
  makes the two minima equal by a change of variable).
* The pure-junk equality case: `n = 1`, `c = (1/2, -1/2)`, `v = (1, -1)` gives `C_empty = 0`,
  `C_{1} = 1`, `h = u`, `hmin = 1`, and `sum|c_l|^r = 2^{1-r} = ` the target at
  `r = 0.2, 0.5, 2/3, 0.9` to full precision. **A feature that is 100% junk can be exactly
  optimal** — which is why no theorem here can be stated in terms of `|A|`.
* Aiming-depth threshold (`eleg2_genN.py --mode depth`, bisection): for `N = n+1` populations at
  `n = 2`, `r = 2/3` (`nu = 3, q = 2`) the singleton-partition threshold is
  `A_1 in [0.2727, 0.2875]` — comfortably below the `A* >= 1/2` that `P3A`'s Riesz product proves on
  dissociate supports, and below `P3C`'s all-`N` mixture threshold `A_lam = 0.3034..0.3044`. So
  (DEPTH-A) at `A = 1/2` is not close to the edge at `N = n+1`; what is missing is a proof, not
  room.

### 8.3c Run completeness

`eleg2_probe.py`, `eleg2_spec.py`, `eleg2_thmA.py`, `eleg2_net.py`, `eleg2_routes.py` and the twist
check ran to completion and wrote their `*.json`. `eleg2_det.py`, `eleg2_extra.py`,
`eleg2_genN.py` (both modes) were stopped at hand-back after the cells reported above; for those the
`*.log` files in `FINAL_RUNS/ELEG2/` are the record, and every number quoted here appears in them.
The stopped cells were the larger ones (`eleg2_det.py` at `n = 2, N >= 5` and `n = 3`;
`eleg2_extra.py` E3 at `n = 3, N >= 5`; `eleg2_genN.py` at `n = 3, N >= 6` and the depth bisection
beyond `n = 2, N = 3`). No background process remains.

### 8.4 The refuted routes (`eleg2_routes.py`, `routes.json`, `routes.log`)

| route | measured |
|---|---|
| **R1** `|det M| >= hmin^D` | infimum **`0`** at `n = 1` (`N = 3,4,5,6`) and at `n = 2` (`N = 3,4`), from junk-free starts (`eleg2_det.py`); the full `s`-family `|det(I+sM)|/(1+s hmin)^D` is driven to `0.0104 / 0.1253 / 0.0926 / 0.0010` (`n=1`, `N=3..6`) and `0.2122 / 0.3606` (`n=2`, `N=3,4`). At `n = 1, N = 2` the minimum is exactly `1.000000000`, i.e. the statement *is* true there (that is the `N = D` case, where Theorem A's mechanism applies). A block-biased search (`eleg2_extra.py` E3) returns `1.000005 .. 1.00055` at `n = 2`, `N = 3..6` — a reminder that this refutation needs starts that are not near the block. |
| **R2** `Re spec(M) >= m` | `min Re(spec)/m = -350.71 / -9.31 / +0.886` at `n = 1/2/3` |
| **R3** `Herm(M) >= m I` | `lambda_min(Herm)/m = -413.95 / -139.78 / -1506.89` at `n = 1/2/3` |
| **R4** diagonal restriction | `n = 2, r = 2/3`: a **two**-atom population makes `H(z)` constant at cost `1.259921050 = 2^{1/3}`, against the target `3^{1/3} = 1.442249570` — the diagonal is short by `12.6%`, exactly as predicted |
| **R5** gauge symmetrisation | junk of the `D`-fold rotation average `9.67e-16` (it really is junk-free) and `cost_avg / (D^{1-r} cost) - 1 = 4.44e-16` (it really does cost exactly the factor to be proved) — circular to machine precision |
| **R6** what Theorem A gives on the `eta` axis | `Cl2_r(eta) >= 3^{1-r}(1-sqrt3 eta)^r` = `1.4087 / 1.3577 / 1.2705 / 1.0862 / 0.8846` at `eta = 0.02/0.05/0.1/0.2/0.3` — `P2` Theorem P2-1 exactly, i.e. no improvement of the curve |

---

## 9. Status

**PROVEN (written out in full, every step fp64-checked).**

1. **THEOREM A** (§3): for `N = n+1` atoms with arbitrary junk, `sum_l |c_l|^r >= (n+1)^{1-r}
   hmin^r`, `hmin = min_{T^n}|h|`, every `n`, every `r in (0,1]`. Four lines from the chain
   identity. Contains `P1`'s Key Lemma (`h` constant) and agent 2's Theorem A (`r -> 0`).
2. **LEMMA A0** (collapse), **LEMMA A1** (chain identity, re-derived), **LEMMA A2** (a feature with
   positive aimed output is non-vanishing on the whole polydisc), **LEMMA A3** (`|C_T| <=
   2(Re A - m)`: junk is bounded by twice the aiming loss — an `eta`-free replacement for Lemma Q),
   **LEMMA A5** (`hmin <= max_T|C_T|`, every `n` and `N`: the `k = 1` rung of Conjecture C2).
3. **THEOREM B** (§6): Conjectures C1 and C2 in full at `n = 1`, every `N`, every `r`.
4. **COROLLARY N1** (§5): every pure `K = 1` network of width `<= n+1` on the full `p^n` table, with
   arbitrary junk, satisfies `(T)` up to `(1 - eps_p)^{-1}`, `eps_p = (n+1)^{(nu-q)/q}(2^n -
   (1+cos(pi/p))^n)`; `eps_p < 1` from `p >= 7 / 17 / 43` at `nu = 3/4/5`, `q = 2`, and
   `eps_p -> 0` like `p^{-2}`. This is at every `q in [2,nu]` and every `n`, where the inherited
   Theorem C was `q = 2` and `p >= 23/47/97` by case analysis.

**REFUTED, with exact breaking points (§6.1–6.5).** The determinant form `|det(I+sM)| >= (1+sm)^D`
(and so the direct robustification of `ELEGANT.md`'s Cauchy–Binet proof); the spectral form
`Re spec(M) >= m`; the numerical-range form `Herm(M) >= m I`; the diagonal restriction
(short by `2^{1/3}` vs `3^{1/3}` at `n = 2`); gauge symmetrisation (circular *also* at the `E_k`
level, not only at `sum|c|^r`). Recorded: Theorem A does **not** improve the `eta`-parametrised
junk-cost curve `Cl2_r` — it replaces the variable.

**OPEN, stated precisely.**

* **(C2 / C1) `N > n+1` at `n >= 2`.** Numerically tight and the minimiser always collapses to
  `n+1` atoms, but every determinantal certificate is refuted and the slot-peeling induction of
  Lemma A5 gains only `n`, not `n+1` (the exact shortfall is computed in §6). This is the same
  `N`-uniformity problem that `ELEGANT.md` solved by triangularity; triangularity has no
  approximate version.
* **(DEPTH-A) Theorem A at aiming depth `A < 1`** (§7). `A = 1/2` would close `nu = 3, q = 2` for
  every `p`, every `K`, every junk level on dissociate supports **with the singleton density
  alone**, replacing `P3C`'s partition mixture and its measured `A_lam`. The kill-point proof does
  not deform: at depth `A` the kill point multiplies the targeted atom by `(1-A)` instead of `0`.
* Impure neurons with junk (unchanged; `P3C` F6-5).

**What a referee should re-run.** Nothing in §3–§5 is computer-assisted: Lemmas A0–A5, Theorem A,
Theorem B and Corollary N1 are proofs. §8 is confirmation, not evidence for a step. The only
numerically-supported claims are the two OPEN conjectures and the five refutations, and the
refutations are one-sided (a measured violation is a genuine violation).


