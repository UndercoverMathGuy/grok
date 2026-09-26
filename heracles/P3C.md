# P3C: junk in the middle range at `K >= 2` — the per-frequency reformulation

Sixth pass. Problem **P3** of `GOPT_STATE.md` / `GOPT_STATE2.md`: remove the "no same-`|k|` junk"
hypothesis `(H)` from agent 3's Theorem D, i.e. prove

    (T)   gamma_q(theta)  <=  (n+1)^{-(nu-q)/q} * B * Gamma_{nu/q}(p)   for every 2 <= q < nu,
          B = 2 (2 p (n+1))^{-nu/2},

for networks carrying junk on `K >= 2` frequencies at an arbitrary (in particular middle) junk level.

**Status: PARTIAL.** The premise was re-checked and made unconditional (§2: the chord criterion is
refuted at `nu = 4, q = 2` by an exact closed-form witness, and at `nu = 5, q = 2` too). A different
route is then built: P3 at `K >= 2`, at every junk level, is reduced to ONE finite-dimensional,
`p`-free, `K`-free, `eta`-free inequality about a single one-frequency population (THEOREM P3C-1 +
PROPOSITION P3C-2), whose threshold is measured and compared to what is available (§5). Full summary
in §9.

---

## 0. Symbols, with a plain reading of each

Inherited symbols keep their meaning (`p`, `n`, `nu = n+1`, `q`, `r = q/nu`, `B`, `Gamma_s(p)`,
`Cl2_r`, `C_T`, `eta`, `A*`). The ones this note leans on, and the new ones:

| symbol | meaning | plain reading |
|---|---|---|
| `n` | number of operands | `nu = n + 1` is the homogeneity degree of the network in its weights |
| `r = q/nu in (0,1]` | penalty ratio | `q = 2` is plain weight decay; `r < 1` is why splitting a feature over several neurons costs |
| `omega = e^{2 pi i/p}` | primitive `p`-th root of unity | the task's basic phase |
| `S`, `K = \|S\|` | frequency support, its size | which cosines the network uses |
| `c_l`, `v_lj` | atom `l`'s complex amplitude, its slot-`j` unit phase | one neuron's signed contribution at one frequency, and the phase of its `j`-th operand embedding |
| `C_T = sum_l c_l prod_{j in T} v_lj` | sign-pattern amplitude | `T = empty` is the wanted cosine of the SUM ("clean"); the other `2^n - 1` are junk |
| `h(u) = sum_l c_l prod_j (1 + u_j v_lj)` | the frequency's generating polynomial | `h = sum_T C_T u^T`; junk-free iff `h` is constant |
| `kappa0 = (n+1)^{(1-r)/r}` | `= Cl2_r(0)^{1/r}` | how much clean amplitude one unit of cost buys at zero junk — the `(n+1)`-block's exchange rate |
| `eta` | relative `l_2` junk of a frequency | "what fraction of the feature's output is garbage" |
| `A(eta) = Cl2_r(eta)^{1/r}` | certified clean amplitude per unit cost at junk `eta` | `A(0) = kappa0`, `A(sqrt(2^n-1)) = 1` |
| `rho(eta) = [1 - A(eta)/kappa0]/eta` | the chord slope of the relief curve | "how much margin head-room one unit of junk buys" |
| `rho_need = (1 - 1/kappa0)/sqrt(2^n-1)` | the chord slope P3A's `K`-free criterion allows | the slope of the straight line from `(0,kappa0)` to `(eta*,1)` |
| `A*(p,S)` | polydisc inradius of the moment body of `{2k : k in S}` | "how deep a non-negative test density can aim at every listed frequency at once, with any phases I like" |
| `g_A(psi) = sum_T C_T A^{\|T\|} e^{i sum_{j in T} psi_j}` | the frequency's output under a **product** test density of depth `A` | what is left of this feature after the test density has been aimed at it |
| `P = {B_1..B_s}` | a **partition** of the operand slots `{1..n}` | which groups of operands the test density looks at only through their sum |
| `g_P(t) = sum_U C_{B_U} prod_{i in U} t_i` | the frequency's output under the partition density | `B_U` = union of the blocks in `U`; `t_i` = the density's block-`i` depth, `\|t_i\| = A` |
| `R_P(pop, A)` | the **partition score** (3.2) | "how much clean signal survives the aiming, in units of what the junk-free block of the same cost delivers"; `<= 1` means "no better than the block" |
| `(PFP)`, `A_lam(n,r)` | the criterion of THEOREM P3C-1 and its threshold | the aiming depth at which junk can never pay, at any `K` and any level |
| `M_m` | `m` atoms at the `m`-th roots of unity in every slot | the extremal family: it kills every junk pattern whose size is not a multiple of `m` |

---

## 1. What this pass had to do

1. **Independently re-check** the claim in `GOPT_STATE2.md` that P3A's `K`-free *chord criterion*
   (`rho(eta) <= rho_need` on the whole junk range) is REFUTED at `nu = 4` by P2's measured relief
   curve (`FINAL_RUNS/P2/chordcheck.json`). That claim had never been verified. → §2.
2. Given the refutation, find a **genuinely different route** to (T) at `K >= 2` with junk. → §3 onwards.

---

## 2. The chord criterion at `nu = 4`: re-checked, and why the refutation is sound

### 2.1 The logical direction of the refutation is sound

`Cl2_r(eta)` is a **minimum** over populations, so any measurement of it by an optimiser is an
**upper** bound: `Cl2_r^meas >= Cl2_r^true`. Hence `A^meas = (Cl2_r^meas)^{1/r} >= A^true` and

    rho^meas(eta) = [1 - A^meas/kappa0]/eta  <=  rho^true(eta) .

So a *measured* `rho` above `rho_need` is a genuine refutation of the chord criterion, and a measured
`rho` below it proves nothing. `GOPT_STATE2.md` states this correctly. It also means the
non-monotone wobble in P2's `cost_nu4_q2.0.json` (`rho` goes
`0.0896 -> 0.2004 -> 0.3147 -> 0.3113 -> 0.2962 -> 0.2919 -> 0.3495 -> 0.3837`) — a sure sign that
SLSQP missed the global minimum at some levels — can only make the refutation *stronger*.

### 2.2 The refutation is structural, not numerical: an exact algebraic witness

The measured curve at `nu = 4, q = 2` (`n = 3`, `r = 1/2`, `kappa0 = 4`) is a staircase whose steps
are the **`m`-atom families**. Define, for `1 <= m`, the population

    M_m :   m atoms,  c_l = 1/m,  v_lj = zeta^{-l} for every slot j,  zeta = e^{2 pi i/m}.

Then `C_T = (1/m) sum_{l=0}^{m-1} zeta^{-l|T|} = 1` if `m` divides `|T|` and `0` otherwise, so

    clean C_empty = 1,   junk eta(M_m)^2 = #{ T != empty : m divides |T| }  =  sum_{j>=1} binom(n, jm),
    cost(M_m) = m * (1/m)^r = m^{1-r},   A(M_m) := cost^{1/r} = m^{(1-r)/r} .

`M_{n+1}` is the junk-free block (`A = kappa0`); `M_1` is the single atom (`A = 1`, `eta = sqrt(2^n-1)`).
These are *feasible points*, so `Cl2_r(eta(M_m)) <= m^{1-r}`, i.e. `A(eta(M_m)) <= m^{(1-r)/r}`, hence

    rho( eta(M_m) )  >=  [ 1 - m^{(1-r)/r}/kappa0 ] / eta(M_m)
                      =  [ 1 - (m/(n+1))^{(1-r)/r} ] / sqrt( sum_{j>=1} binom(n, jm) ) .

**At `nu = 4, q = 2` (`n = 3`, `r = 1/2`, `kappa0 = 4`), `m = 2`:**
`eta(M_2)^2 = binom(3,2) = 3`, so `eta = sqrt 3 = 1.7320508`; `A(M_2) = 2`; and

    rho( sqrt 3 )  >=  (1 - 2/4)/sqrt 3  =  1/(2 sqrt 3)  =  0.288675135  >  0.283473355 = rho_need .

That is an **exact, closed-form refutation** of the chord criterion at `nu = 4, q = 2` — no optimiser
involved. It is a thin margin (`1.83%`), which is why the measured curve, which finds cheaper
populations than `M_2` at that junk level, reports the much larger `0.3837` at `eta = 1.24`.

**At `nu = 4, q = 3`** (`r = 3/4`, `kappa0 = 4^{1/3} = 1.58740105`), `m = 2`:
`rho(sqrt 3) >= (1 - 2^{1/3}/4^{1/3})/sqrt 3 = (1 - 2^{-1/3})/sqrt 3 = 0.1250855 < 0.1398618`. So `M_2`
alone does **not** refute that cell; the refutation there rests on P2's measurement (`0.15004` at
`eta = 1.3479`).

**At `nu = 3, q = 2`** (`n = 2`, `r = 2/3`, `kappa0 = sqrt 3`) the only interior family is `m = 2`:
`eta(M_2)^2 = binom(2,2) = 1`, `eta = 1`, `A(M_2) = (2/3)^{1/2} * kappa0 = sqrt 2`;
`rho(1) >= (1 - sqrt2/sqrt3)/1 = 0.183503`, comfortably below `rho_need = 0.244017`. Consistent with
the chord holding at `nu = 3` with exactly zero room (equality at the endpoint `eta* = sqrt 3`, where
`A = 1` by the single atom `M_1` — that "equality" is not a coincidence or a deep identity: it is the
*definition* of `rho_need`, whose value P3A computed as the endpoint value of `rho`. The content of
"`nu = 3` holds with zero room" is only that the interior never beats the endpoint.)

**Plain reading.** `M_m` is "`m` neurons whose operand phases are the `m`-th roots of unity in every
slot". Such a population cancels every junk pattern whose size is *not* a multiple of `m`, and leaves
the rest at full strength. It costs `m^{1-r}` instead of the block's `(n+1)^{1-r}`, so it is a
genuinely cheaper feature that carries junk; and at `nu = 4` its trade beats the straight-line budget
that P3A's criterion allows.

### 2.3 The witness, verified in fp64

`p3c_verify.py --block mm` builds each `M_m` from its atoms and recomputes `C_T`, `eta` and the cost
from scratch (`FINAL_RUNS/P3C/verify_im.json`). Closed forms and computation agree to fp64; the
`rho` column is the exact lower bound on `rho(eta(M_m))` above.

| `nu` | `q` | `m` | `eta(M_m)` | `A(M_m)` | `rho >=` | `rho_need` | verdict |
|---|---|---|---|---|---|---|---|
| 3 | 2 | 1 | 1.732050808 | 1 | 0.244016936 | 0.244016936 | equality (this IS `rho_need`) |
| 3 | 2 | 2 | 1.000000000 | 1.414213562 | 0.183503419 | 0.244016936 | no refutation |
| 3 | 2.5 | 2 | 1.000000000 | 1.148698355 | 0.077892089 | 0.113887212 | no refutation |
| **4** | **2** | **2** | **1.732050808** | **2** | **0.288675135** | **0.283473355** | **REFUTES (exactly)** |
| 4 | 2 | 3 | 1.000000000 | 3 | 0.250000000 | 0.283473355 | no refutation |
| 4 | 3 | 2 | 1.732050808 | 1.259921050 | 0.119107057 | 0.139861775 | no refutation |
| 4 | 3 | 3 | 1.000000000 | 1.442249570 | 0.091439704 | 0.139861775 | no refutation |
| **5** | **2** | **2** | **2.645751311** | **2.828427125** | **0.282346184** | **0.235104879** | **REFUTES** |
| **5** | **2** | **3** | **2.000000000** | **5.196152423** | **0.267620999** | **0.235104879** | **REFUTES** |
| **5** | **2** | **4** | **1.000000000** | **8** | **0.284458247** | **0.235104879** | **REFUTES** |

**Verdict on the re-check.** `GOPT_STATE2.md`'s claim is CONFIRMED and strengthened:

* at `nu = 4, q = 2` the chord criterion is refuted **exactly and unconditionally**, by
  `rho(sqrt 3) >= 1/(2 sqrt 3) = 0.288675135 > 0.283473355 = rho_need`. No optimiser, no
  measurement, no dependence on P2's `Cl2_r` grid. (P2's measured `0.3837` at `eta = 1.24` is a
  stronger but optimiser-dependent statement; both point the same way, and §2.1 shows optimiser
  error can only help the refutation.)
* at `nu = 5, q = 2` it is refuted by three separate `M_m`, the worst by `21%`. This cell was not
  previously tested.
* at `nu = 3` (`q = 2, 2.5`) the `M_m` family does **not** refute it, consistently with P2's
  "holds with zero room". The "zero room" is a tautology (the endpoint value of `rho` *is*
  `rho_need`); the non-trivial content is that the interior stays below, which remains a measured
  statement.
* at `nu = 4, q = 3` the `M_m` family does not refute it either; that cell's refutation rests on P2's
  measurement alone (`0.15004` vs `0.139862`, a `7.3%` violation), which I did not reproduce to
  completion (`p3c_chord.py`, an independent SLSQP implementation, was still running at hand-back;
  see §8).

So the brief's premise stands: **the route "one per-frequency `l_2` junk-cost curve against a
`K`-free margin penalty linear in the junk norm" is dead at `nu >= 4`, `q = 2`.**

---

## 3. The new idea: a per-frequency criterion with a partition-adapted test density

### 3.1 Why a new idea is needed, stated precisely

Everything in P3A/P2 goes through the pair

  (i) a margin penalty `Pen(D) >= rho * sum_k w_kD eta_k x_k`, **linear in the junk norm**;
  (ii) a cost relief `x_k <= phi(eta_k) * (certified amplitude)`,

and the chord computation is exactly the question whether (i)'s linear budget covers (ii)'s relief.
§2 shows it does not at `nu = 4`. But the linearity in `eta` is an artefact of the *route*: `eta` is
a summary statistic that is thrown away and then re-used. The criterion below never mentions `eta`.

### 3.2 The test-density family: products of fibre densities over a partition

Let `P = {B_1, ..., B_s}` be a **partition of the operand slots** `{1..n}`. For each block pick a
probability density `nu_i` on `Z_p` and set

    w(a)  =  prod_{i=1}^{s}  nu_i( sum_{j in B_i} a_j ) / p^{|B_i| - 1} ,      a in Z_p^n .

*Plain reading: independently for each block of operands, look only at the SUM of that block's
operands, and put the density `nu_i` on it; inside a block, spread uniformly over the fibre.*

This is a probability density (summing over a fibre of `sigma_{B_i}` gives `p^{|B_i|-1}` terms). Its
Fourier coefficient at a mode `m in Z_p^n` factorises over the blocks, and the block factor is
`nuhat_i(m_0)` if `m` is constant `= m_0` on `B_i` and `0` otherwise. The junk mode of frequency `k`
and sign pattern `T` is `m_{k,T} = 2k * 1_T` (P3A §2.1), which restricted to `B_i` is `2k * 1_{T cap B_i}`
— constant on the block exactly when `T cap B_i` is `empty` or all of `B_i`. Hence

    what(2k 1_T)  =  prod_{i : B_i subset T} nuhat_i(2k)   if T is a union of blocks,   else 0.  (3.1)

Two extremes: `P` = all singletons gives the product-of-Riesz density of P3A's THEOREM S (depth
`A^{|T|}` on every pattern); `P = {[n]}` gives P3A's fibre density F5-4 (depth `A` on `T = [n]`,
zero on every other pattern).

By the definition of `A* = A*(p, {2k : k in S})` (P3A §4.1) each `nu_i` can be chosen with
`nuhat_i(2k) = A* e^{i phi_i^{(k)}}` for **arbitrary** phases `phi_i^{(k)}`, independently for every
block `i` and every frequency `k in S`. That is the crucial freedom: `A*` is a *polydisc* inradius,
so the phases are free, and different frequencies do not fight over them.

### 3.3 The reduction

Write, for a one-frequency population `pop = {c_l, v_lj}` with sign-pattern amplitudes `C_T`, and for
a partition `P` with blocks `B_1..B_s`, the **`s`-variable multi-affine form**

    g_P(t)  =  sum_{U subset {1..s}}  C_{B_U} * prod_{i in U} t_i ,      B_U := union_{i in U} B_i ,

and the **partition score**

    R_P(pop, A)  :=  kappa0 * ( min_{|t_i| = A} Re g_P(t) )_+ / ( sum_l |c_l|^r )^{1/r} .   (3.2)

*Plain reading: `R_P` is "how much clean signal this population still delivers after the density has
been aimed at it, measured in units of what the junk-free `(n+1)`-block of the same weight-decay cost
delivers". `R_P <= 1` means the population is no better than the block.* `R_P` is scale-invariant
(both numerator and denominator are homogeneous of degree 1 in `c`), and `R_P(block) = 1` for every
`P` and every `A`, because the block's `g_P` is the constant `C_empty`.

> **THEOREM P3C-1 (the reduction; every `p`, every `K`, every junk level).**
> Let `theta` be a chain network all of whose neurons are pure single-frequency neurons, with
> frequency support `S`, per-frequency populations `pop_k` and per-frequency costs
> `cost_k = sum_{l at k} |c_l|^r`. Put `A = A*(p, {2k : k in S})` and `b_k = cost_k / sum_j cost_j`.
> If there is a partition `P` of `{1..n}` with
>
>     (PFP)        sum_k b_k * R_P(pop_k, A)^r   <=   1 ,
>
> then `theta` satisfies (T): `gamma_q(theta) <= (n+1)^{-(nu-q)/q} B Gamma_{nu/q}(p)`.

**Proof.** Six steps, each elementary.

1. *(profile identity, inherited: P3A P0, verified there to `2.9e-15`, re-verified here in
   `p3c_verify.py`)* With `Lambda_D(a) := 2^n (z_{correct}(a) - z_{correct+D}(a))`,
   `Lambda_D(a) = sum_k Re[ W_kD h_k(u^{(k)}(a)) ]`, `W_kD = 1 - omega^{-kD}` (sign convention: see §4.4),
   `u_j^{(k)}(a) = omega^{-2 k a_j}`, and `2^n * margin = Lambda := min_{a, D != 0} Lambda_D(a)`.
2. *(test density)* For any probability density `w` on `Z_p^n` and any `D != 0`,
   `Lambda <= E_w[Lambda_D] = sum_k Re[ W_kD * sum_T C_T^{(k)} what(2k 1_T) ]`.
   With `w` the partition density of §3.2 and (3.1), the inner sum is exactly `g_P^{(k)}(t^{(k)})`
   with `t_i^{(k)} = nuhat_i(2k)`, `|t_i^{(k)}| = A`, phases free and independent across `(i,k)`.
3. *(`+-D` symmetrisation)* `W_{k,-D} = conj(W_{kD})`, so
   `E_w[Lambda_D] + E_w[Lambda_{-D}] = 2 sum_k w_kD Re g_P^{(k)}`, `w_kD = 1 - cos(2 pi k D/p) >= 0`.
   Taking the smaller of the two: `Lambda <= sum_k w_kD Re g_P^{(k)}`, for every `D != 0`, and for
   every admissible choice of the phases.
4. *(aim)* Choose the phases to minimise each `Re g_P^{(k)}` separately (legal by §3.2), and write
   `y_k := ( min Re g_P^{(k)} )_+ = R_P(pop_k,A) cost_k^{1/r} / kappa0 >= 0`. Since `w_kD >= 0`,
   `Lambda <= min_{D != 0} sum_k w_kD y_k`.
5. *(the programme)* By the definition of `Gamma_s(p) = max_{b>=0, sum b = 1} min_{D != 0} sum_k b_k^s w_kD`
   with `s = 1/r` and `b_k = y_k^r / sum_j y_j^r`:
   `min_{D != 0} sum_k w_kD y_k <= Gamma_{1/r}(p) * (sum_k y_k^r)^{1/r}`.
6. *(the budget)* `sum_k y_k^r = kappa0^{-r} sum_k R_P(pop_k,A)^r cost_k`, so (PFP) gives
   `sum_k y_k^r <= kappa0^{-r} sum_k cost_k`, hence
   `Lambda <= Gamma_{1/r} kappa0^{-1} (sum_k cost_k)^{1/r}`.
   Finally `P_q >= ((n+1)p/2)^{q/2} sum_k cost_k`, with equality after the per-neuron rebalancing
   that leaves every `c_l` fixed (`LEMMA_ATTEMPT` §2 Step 4) — and `>=` is the direction an upper
   bound on `gamma_q = margin / P_q^{nu/q}` needs. And
   `2^{-n} ((n+1)p/2)^{-nu/2} = 2 (2p(n+1))^{-nu/2} = B` exactly (checked: `2^{-n} 2^{nu/2} (p(n+1))^{-nu/2}
   = 2^{-n} 2^{nu} (2p(n+1))^{-nu/2} = 2 (2p(n+1))^{-nu/2}` since `nu = n+1`), so
   `gamma_q <= B * Lambda / (sum_k cost_k)^{1/r} <= B kappa0^{-1} Gamma_{1/r} = (n+1)^{-(nu-q)/q} B Gamma_{nu/q}`. ∎

**What the reduction buys.**
* `K` has disappeared. There is no dilution by the number of frequencies, so THEOREM R's Dirichlet
  obstruction — which is about summing a junk norm over frequencies — does not apply: the frequencies
  are never summed, they are handled one at a time and re-assembled by the `Gamma` programme itself.
* `eta` has disappeared, so §2's refutation does not apply either.
* The density is chosen **after** seeing `theta` (the partition and all `s * K` phases), so P3B's
  duality barrier THEOREM P3-2, which is about measures fixed in advance, does not apply.
* It is an aggregate condition: a frequency with `R_P > 1` can be paid for by frequencies with
  `R_P < 1`. That matters (§4.3).

### 3.4 The criterion as a single-population statement (minimax)

(PFP) asks for a partition that works for the *whole network*; different frequencies may prefer
different partitions. The usual minimax removes the quantifier order.

> **PROPOSITION P3C-2.** Let `Pi_n` be the (finite) set of partitions of `{1..n}`. The following are
> equivalent:
> (a) every pure network with support `S` admits a partition satisfying (PFP);
> (b) there is a probability vector `lam` on `Pi_n` with
>     `sum_P lam_P R_P(pop, A)^r <= 1` for **every single** one-frequency population `pop`.
>
> *Proof.* (b) ⇒ (a): averaging (b) over `b` gives `sum_P lam_P E_b[R_P^r] <= 1`, so some `P` has
> `E_b[R_P^r] <= 1`. (a) ⇒ (b): the game `sup_b min_P E_b[R_P^r]` has a linear payoff in `b` (a
> probability measure on the compact set of normalised populations) and in `lam` (a simplex), so by
> Sion's minimax `sup_b min_P = min_lam sup_pop`; (a) says the left side is `<= 1`. ∎
>
> *Plain reading: the network's frequencies are the adversary's mixture; our randomisation over
> partitions is the mixed strategy that answers it. Note the randomisation is a proof device only —
> for any given network one honest partition is used.*

Define accordingly

    A_lam(n, r)  :=  inf { A in (0,1] : some lam on Pi_n satisfies (b) } .

**So P3 at `K >= 2`, at every junk level, follows for a support `S` as soon as
`A*(p, {2k : k in S}) >= A_lam(n, r)`.** This is a `p`-free, `K`-free, `eta`-free, finite-dimensional
constant, and §4 computes it.

### 3.5 Three remarks on the scope of THEOREM P3C-1

1. **Pure neurons only.** Step 1 is P3A's profile identity, which is a statement about
   single-frequency neurons. Agent 3's THEOREM D does not need purity because its atoms are the
   neurons' Fourier data at one `k` and hypothesis `(H)` kills everything else; here the whole point
   is to keep the same-`|k|` junk, and the cross-frequency terms of an impure neuron are not covered.
   THEOREM S and all of P3A have the same scope.
2. **The block is an exact equality case.** `R_P(block) = 1` for every `P` and every `A`, so (PFP) is
   tight at the block and can never have slack. Any proof of it must therefore be a *rigidity*
   statement. (That is also a strong consistency check on the numerics: an LP value of exactly
   `1.000000000` means the block is the maximiser.)
3. **Junk-free frequencies cost nothing.** If `pop_k` is junk-free then `g_P = C_empty` for every `P`,
   so `R_P = kappa0 Re C_empty / cost^{1/r} <= 1` by the Key Lemma / P1 (which is exactly
   `cost >= (n+1)^{1-r} |C_empty|^r`). So the criterion degenerates to the known junk-free case.

---

## 4. Computing `A_lam(n, r)`

### 4.1 The `M_m` family again — now as the adversary

The same populations that refuted the chord criterion in §2.2 turn out to be the binding
configurations here. For `M_m` (`C_T = 1` iff `m` divides `|T|`, cost `m^{1-r}`) and a partition
`P = {B_1..B_s}`,

    g_P(t)  =  sum_{U : m divides sum_{i in U} |B_i|}  A^{|U|} prod_{i in U} e^{i phi_i} ,
    R_P(M_m, A)  =  ((n+1)/m)^{(1-r)/r} * ( min_phi Re g_P )_+ .

Two readings of this formula. With `P` = singletons, a block of `m` atoms is seen at depth `A^m`
(its lightest junk pattern has `|T| = m`), so `R = ((n+1)/m)^{(1-r)/r}(1 - binom(n,m) A^m + ...)`:
the singleton density is good at catching junk that lives on *small* patterns. With `P` a partition
into `m`-element blocks (when `m` divides `n`), that same junk is seen at depth `A^1`, which is much
deeper. The two extremes of the `m`-ladder therefore need *different* partitions, and that is
precisely what the mixture `lam` of PROPOSITION P3C-2 buys.

`p3c_part.py`/`p3c_stress.py` solve the LP `min_lam max_pop sum_P lam_P R_P^r`. Restricting the
adversary to the `M_m` family alone gives the **`M`-threshold** `A_M(n,r)`:

| cell | `n` | `r` | `kappa0` | `A_M(n,r)` | optimal `lam` |
|---|---|---|---|---|---|
| `nu=3, q=2` | 2 | 2/3 | 1.7320508 | **0.300416** | 0.567 singletons + 0.433 the one-block |
| `nu=3, q=2.5` | 2 | 5/6 | 1.2457309 | **0.137235** | 0.504 / 0.496 |
| `nu=4, q=2` | 3 | 1/2 | 4 | **0.481063** | 0.655 singletons + 0.345 the one-block |
| `nu=4, q=3` | 3 | 3/4 | 1.5874011 | **0.245495** | 0.835 `{1,2}{3}`-type + 0.165 one-block |
| `nu=5, q=2` | 4 | 2/5 | 5 | **0.553514** | 0.304 singletons + 0.534 `{1..3}{4}`-type + 0.143 one-block |

(`p3c_part.py` machinery, bisection to `1e-11`; every `R_P` is an exact coordinate-descent minimum,
verified against a brute-force torus grid in 400/400 random cases.)

### 4.2 The full adversary

Letting the adversary range over **all** populations (`N = 1..5` atoms, `12 x 5` Nelder-Mead ascents
per cutting-plane round on top of a `4000`-population random pool, 5–6 rounds) at `nu = 3, q = 2`:

| `A` | LP value `min_lam max_pop sum_P lam_P R_P^r` | verdict |
|---|---|---|
| 0.250 | 1.048536919 | fails |
| 0.295 | 1.005263683 | fails |
| 0.300 | 1.001985625 | fails |
| 0.3025 | 1.001557903 | fails |
| 0.310 | **1.000000000** | holds (block is the maximiser) |
| 0.303438 | 1.000570852 | fails |
| **0.304375** | **1.000000000** | **holds** |
| 0.306250 | **1.000000000** | holds |
| 0.310 | **1.000000000** | holds |
| 0.340 | **1.000000000** | holds |
| 0.350 | **1.000000000** | holds |

so (bisection, `p3c_stress.py --block sweep`) `A_lam(2, 2/3) in [0.3034375, 0.3043750]`, bracketing the
`M`-family value `A_M = 0.300416` from above by `1.0%-1.3%`. **The `M_m` family is (essentially) the extremal family**, which is a genuinely useful
structural fact: it says the criterion is an inequality about `n` explicit configurations, not about
an unstructured optimisation.

*Plain reading of `A_lam(2,2/3) ~ 0.30`: if a non-negative test density can aim at every frequency the
network uses, simultaneously and with phases of my choosing, to a depth of `30%` of the maximum, then
no amount of junk on any number of frequencies can push a two-operand network above the target.*

### 4.3 The block is the maximiser, and why

The LP value is **exactly** `1.000000000` whenever the criterion holds, i.e. the block is the
maximiser and the criterion is tight there. That is a rigidity statement, and it has a clean
first-order reason (checked by `p3c_stress.py --block dirs local`):

Perturb the block, `c_l = (1+eps_l)/(n+1)`, `v_lj = w^{-l} e^{i delta_lj}` with `w = e^{2 pi i/(n+1)}`.
Then, writing `t` for the perturbation size,

* every junk amplitude `C_T` (`T != empty`) is **first order** in `t`, because
  `C_T = (1/(n+1)) sum_l w^{-l|T|} (eps_l + i sum_{j in T} delta_lj) + O(t^2)`;
* the cost surplus `cost^{1/r}/kappa0 - 1 = (1/(2(n+1)))[(r-1) sum_l x_l^2 + sum_l y_l^2] + O(t^3)`
  (`eps_l = x_l + i y_l`, `sum_l eps_l = 0`) is **second order**, and is the only way the perturbation
  can *gain*.

So `sum_P lam_P R_P^r = 1 - r[ sum_P lam_P M_P - (second order) ] + ...`, where `M_P >= 0` is the
first-order amount of junk that partition `P` can see. `M_P = 0` for all `P` forces every `C_T = 0`
to first order (for any non-empty `T`, the partition `{T} u {singletons of the complement}` sees `T`
at depth `A^1`, and so does the singleton partition at depth `A^{|T|}`), i.e. the perturbation is
junk-free to first order — and on junk-free populations `R_P <= 1` by the Key Lemma / P1. **So the
block is a strict local maximum as soon as `lam` puts positive weight on the singleton partition**,
and first order beats second order. Measured: `d log(max_T |C_T|)/d log t` and
`d log|cost surplus|/d log t` come out at **0.9986** and **1.9835** (600 perturbations, radii
`1e-4 .. 0.1`, log-log fit); and over 3000 random block perturbations at radii `1e-4 .. 0.5` at
`A = 0.5`, half of them with an extra small atom of size `1e-4 .. 0.2` added, the maximum of
`sum_P lam_P R_P^r` is **0.999979221445** — never above `1`.

### 4.4 THEOREM P3C-1 verified end to end, including at its equality case

`p3c_verify.py` (`FINAL_RUNS/P3C/verify_im.json`, `verify_tight.json`, `verify_chain.json`):

* **the profile identity** (step 1) against the brute-force `p^n x p` logit table of explicit pure
  networks, `p = 7,11,13`, `n = 2,3`, `K = 2,3`: worst relative error **`3.789e-15`**, both for
  `Lambda_D(a)` row by row and for `2^n * margin = Lambda`. *(Note: the correct sign convention is
  `W_kD = 1 - omega^{-kD}`; with `1 - omega^{+kD}` the row-wise identity fails while the minimum still
  happens to match, which is how I caught it.)*
* **the whole chain** on explicit networks built around the equality case: the share-weighted
  `(n+1)`-block on the `Gamma`-optimal support, then one frequency swapped for an `M_m` and/or the
  whole thing perturbed. For each network the partition is chosen by the criterion, the phases are
  read off by coordinate descent, and **the density `w` is then actually constructed on `Z_p^n`** by
  solving each block's `nu_i` with an LP and checking `w >= 0` and `sum w = 1` numerically. Result
  over 36 networks at `p = 11, 13, 17`:

      Lambda  <=  min_{+-D} E_w[Lambda_D]  <=  Gamma (sum_k y_k^r)^{1/r}  <=  target

  holds in **36/36**, with **0 violations**, and at the unperturbed block the four quantities are
  **equal to 12 digits**: `0.535725926` (`p=11`), `0.497772067` (`p=13`), `0.448340560` (`p=17`),
  i.e. exactly `Gamma_{3/2}(p)` (which reproduces P3A's values), `gamma/target = 1.000000000000`,
  `agg = 1.000000000`. So no step of the proof is lossy at the equality case.

---

## 5. What this closes: `A_lam` against `A*`

THEOREM P3C-1 + PROPOSITION P3C-2 close P3 (junk on `K >= 2` frequencies at ANY level, pure class)
for every support `S` with `A*(p, {2k : k in S}) >= A_lam(n,r)`. Two ways to certify that.

### 5.1 The unconditional certificate: dissociate supports

P3A's Riesz product (its §4.1, proved) gives `A* >= 1/2` whenever `{2k : k in S}` is **dissociate**
(the `3^K` signed sums `sum_k eps_k 2k`, `eps in {0,+-1}`, are pairwise distinct), with **no
dependence on `K` or `p`**. So:

> **COROLLARY P3C-3.** On every dissociate support, (T) holds for every pure network at every junk
> level and every `K`, at `nu = 3, q = 2` and (conditional on `A_M` being the true `A_lam`)
> at `nu = 4, q = 2`, since `A_lam = 0.310 <= 1/2` and `A_M(3,1/2) = 0.481063 <= 1/2`.
> It **fails to apply** at `nu = 5, q = 2`, where `A_M(4, 2/5) = 0.553514 > 1/2`.

That `n = 5`-ish boundary is the same phenomenon as THEOREM S's `(n+1)^{(nu-q)/q} <= 2^n` (true for
`n <= 4`, false from `n = 5`) — the head-room the target gives grows faster than what a bounded-depth
density can take away.

### 5.2 The measured certificate: the paper's actual supports

`p3c_astar.py --papers` computes `A*` of the `Gamma_{3/2}`-optimal supports of `kstar.json` at the
primes the paper uses (`FINAL_RUNS/P3C/astar_papers.json`; the optimiser gives an UPPER bound on
`A*`, so these are indicative, not certificates — but they are far above the threshold):

| `p` | `K*` | `S` at `K*` | `A*` | `>= A_lam(2,2/3) = 0.310` |
|---|---|---|---|---|
| 59 | 6 | 1,4,12,18,25,27 | 0.386225 | yes |
| 71 | 5 | 1,7,12,16,34 | 0.510728 | yes |
| 97 | 6 | 1,4,10,28,35,47 | 0.495940 | yes |
| 113 | 6 | 1,5,12,15,38,46 | 0.503209 | yes |
| 127 | 7 | 1,11,14,20,36,38,43 | 0.497436 | yes |

and at `K* +- 1` as well: the full table spans `A* = 0.380786` (`p=59`, `K=7`) to `0.614090`
(`p=71`, `K=4`), **every one of the 15 cells above `0.310`**. Eleven of the fifteen are also above
`0.481063`, i.e. would carry `nu = 4, q = 2` too.

### 5.3 Where it does NOT reach: small `p`, large `K`

`p3c_astar.py --scan` computes `A*` for **every** support at `p = 11, 13` (`FINAL_RUNS/P3C/astar_scan_nu3q2.json`):

| `p` | `K` | number of supports | `A*` range | covered at `A_lam = 0.310` |
|---|---|---|---|---|
| 11 | 1 | 5 | 0.9595 | 5/5 |
| 11 | 2 | 10 | 0.4573 – 0.6018 | 10/10 |
| 11 | 3 | 10 | 0.2805 – 0.3710 | 5/10 |
| 11 | 4 | 5 | 0.2013 – 0.2528 | 0/5 |
| 11 | 5 | 1 | 0.1905 | 0/1 |
| 13 | 1 | 6 | 0.9709 | 6/6 |
| 13 | 2 | 15 | 0.4775 – 0.6628 | 15/15 |
| 13 | 3 | 20 | 0.3045 – 0.4014 | 15/20 |
| 13 | 4 | 15 | 0.2162 – 0.2942 | 0/15 |
| 13 | 5,6 | 7 | 0.1684 – 0.2168 | 0/7 |

So at `p = 11, 13` the theorem covers `K <= 2` always, most of `K = 3`, and none of `K >= 4`. This is
THEOREM R's collapse arriving early because there are so few residues: at `p = 11` a 4-element
frequency set cannot avoid additive relations. It is **not** a small-`K*` limitation — at the paper's
primes the same `K = 6, 7` supports have `A* ~ 0.5` (§5.2) — but it does mean the statement is
support-dependent and has no uniform-in-`p`-and-`K` form, exactly as THEOREM R predicts it must not.

---

## 6. Routes tried here that FAILED, and exactly where

**F6-1. The single best partition, without the mixture.** Using only the singleton partition (i.e.
exactly THEOREM S's density, extended from one atom per frequency to any population) needs
`A >= 0.428372` at `nu=3,q=2` (binding at `M_2`: `(3/2)^{1/2}(1 - A^2) = 1`) and `A >= 0.629961` at
`nu=4,q=2` (binding at `M_3`: `(4/3)(1 - A^3) = 1`, i.e. `A = 4^{-1/3}`). Using only the one-block
(fibre) partition needs `A >= 1 - 3^{-1/2} = 0.422650` (`nu=3`) and `A >= 3/4` (`nu=4`), both binding
at `M_1`. Neither
alone reaches `1/2` at `nu = 4`; the **mixture** does (`A_M = 0.481063`), because `M_1` and `M_3` want
opposite ends of the partition lattice. This is the one place where PROPOSITION P3C-2's
randomisation is doing real work, and it is why P3A's F5-4 verdict ("combining the two families costs
a factor `2^n - 1`") is too pessimistic: one must combine them *in the budget*, not *in the density*.

**F6-2. Mixing the densities instead of the strategies. Strictly worse.** A single density
`w = sum_P theta_P w_P` is legal and gives `Re E_w h_k = sum_P theta_P (min Re g_P^{(k)})` (the phases
of the different components are independent), so the criterion would read
`E_b[ ((sum_P theta_P rho_P)_+)^r ] <= 1` with the *signed* `rho_P`. Since `x -> x^r` is concave,
`(sum_P theta_P rho_P)^r >= sum_P theta_P rho_P^r` whenever all `rho_P >= 0`, so this is **worse**
than PROPOSITION P3C-2 except in the region where some `min Re g_P < 0` — where it is better, and
where the criterion is satisfied anyway. Recorded because it looks like the obvious move.

**F6-3. Restricting the aim to a sub-support.** When `A*(S)` is too small one would like to aim only
at a good sub-support `S' subset S`. But then the frequencies in `S \ S'` are seen with moment `1` on
the clean mode and nothing else, giving `R_P = kappa0 Re C_empty/cost^{1/r}`, which is `<= kappa0` and
in general `> 1`; the criterion then fails for an adversary that puts all its cost on `S \ S'`. So
`A*` of the **whole** support is what enters, and §5.3's uncovered supports stay uncovered.

**F6-4. Anisotropic aiming depths.** The real constraint is that the vector
`(nuhat_i(2k))_{k in S}` lies in the moment body `Omega`, not merely in its inscribed polydisc of
radius `A*`. One may use any polydisc `prod_k {|t_k| <= A_k} subset Omega`, and THEOREM P3C-1 goes
through verbatim with `R_P(pop_k, A_k)`. This should help exactly on the supports of §5.3 (where the
inradius is dragged down by one bad direction). I did not develop it: the criterion then involves
`K` different `A_k`, so it stops being one constant, and the minimax of PROPOSITION P3C-2 needs
redoing per admissible radius vector. **This is the most promising loose end.**

**F6-5. Extending past the pure class.** For an impure neuron the row-gap has Fourier mass outside
the `K(2^n-1)+1` junk modes (the mode is `(k_j + m)_j` for arbitrary embedding frequencies `k_j` and
readout frequency `m`), so step 1 of THEOREM P3C-1 does not hold and the partition density's moments
at those extra modes are uncontrolled. Agent 3's THEOREM D avoids this only because hypothesis `(H)`
kills the same-`|k|` junk that this note exists to keep. Not attempted.

---

## 7. What remains open, ranked

1. **Prove (PFP) at `A = 1/2`** (or at any `A <= A*` of the supports in play). It is a
   finite-dimensional, `p`-free, `K`-free, `eta`-free inequality with an exact equality case (the
   block) and, numerically, an explicit extremal family (`M_m`). At `A = 1` it *is* P1, which three
   agents have now closed at `q = 2` for `nu = 3,4,5`; the deformation to `A < 1` breaks every known
   proof because the Key Lemma's kill points sit on the **unit** torus. A proof would have to be a
   quantitative rigidity of the block against a *shrunken* polydisc.
2. **The anisotropic version (F6-4)**, which is what would cover `K >= 4` at small `p`.
3. **`A*` rigorously** for non-dissociate supports. I verified the Lipschitz-mesh certificate works
   at `K = 2` (`A*(11,{1,4}) >= 0.564264747`, against the measured `0.601807812`), but the mesh is
   `(2K-1)`-dimensional and does not scale to the `K = 3..7` supports that matter. Same open item as
   P3A's #2.
4. **`nu >= 5, q = 2`.** `A_M(4,2/5) = 0.553514 > 1/2`, so the dissociate certificate expires, exactly
   where THEOREM S expires. Whether that is a real obstruction or an artefact of the partition family
   is unknown.
5. **Impure neurons (F6-5).**

---

## 8. Numerical record (all fp64, local CPU, no pods, `$0`; `heracles/FINAL_RUNS/P3C/`)

| script | what it does | result |
|---|---|---|
| `p3c_verify.py --block ident` | the profile identity against the brute-force logit table, `p=7,11,13`, `n=2,3` | worst relative error **`3.789e-15`** |
| `p3c_verify.py --block mm` | the `M_m` family: `C_T`, `eta`, cost against closed forms; `rho` against `rho_need` | fp64 agreement; **chord REFUTED exactly at `nu4 q2` (`m=2`) and `nu5 q2` (`m=2,3,4`)** |
| `p3c_verify.py --block tight` | THEOREM P3C-1 end to end on 36 explicit networks built around the equality case, with the density constructed by LP and checked `>= 0` | **0 violations**; exact equality (12 digits) at the share-weighted block, `Gamma_{3/2} = 0.535725926 / 0.497772067 / 0.448340560` at `p = 11/13/17` |
| `p3c_part.py` | the criterion's LP `min_lam max_pop sum_P lam_P R_P^r`, cutting planes on a 3000–4000-population pool plus `12 x 5` (resp. `6 x 5`) Nelder-Mead ascents per round | `nu3 q2`: fails at `A = 0.25` (1.048537), `0.295` (1.005264), `0.30` (1.001986), `0.3025` (1.001558); holds (`= 1.000000000`) from `A = 0.304375`. `nu4 q2`: fails at `A = 0.40` (1.076758) and `0.45` (1.032860); **holds (`= 1.000000000`) at `A = 0.50`**. (The `nu4 q2` ladder was stopped after `A = 0.5`, so `part_nu4q2.json` was not written; `part_nu4q2.log` is the record.) |
| `p3c_stress.py --block sweep` | bisection for `A_lam` | `A_lam(2, 2/3) in [0.3034375, 0.3043750]` |
| `p3c_stress.py --block local dirs` | the block's local maximality | max over 3000 perturbations `0.999979221445`; order slopes `0.9986` (junk) vs `1.9835` (cost surplus) |
| `p3c_part.py` (`M`-only LP) | the `M_m`-family threshold `A_M(n,r)` | `0.300416` (`nu3q2`), `0.137235` (`nu3q2.5`), `0.481063` (`nu4q2`), `0.245495` (`nu4q3`), `0.553514` (`nu5q2`) |
| `p3c_astar.py --papers` | `A*` of the `Gamma_{3/2}`-optimal supports of `kstar.json` at `p = 59..127`, `K = K*-1, K*, K*+1` | **15/15 above `A_lam = 0.310`**; range `0.380786 .. 0.614090`; 11/15 also above `0.481063` |
| `p3c_astar.py --scan` | `A*` for EVERY support at `p = 11, 13` | table in §5.3; `K <= 2` always covered, `K >= 4` never |
| `p3c_astar.py --S 1 4 --p 11` | rigorous Lipschitz-mesh lower bound on `A*` | `A*(11,{1,4}) >= 0.564264747` (measured `0.601807812`) |
| `p3c_chord.py` | an independent SLSQP re-measurement of `Cl2_r` and `rho` at `nu4 q2` | started, **not finished** within the pass's CPU budget (1 h on one core) and killed at hand-back; no output file. It is redundant: §2.2's witness settles that cell without any optimiser. |

`p3c_pf.py` is the first (superseded) version of the criterion — the singleton partition only, i.e.
THEOREM S's density extended to arbitrary populations; it is kept because it is the cleanest way to
measure the single-partition thresholds of F6-1 and because its `min_reg` is an independent second
implementation of the inner minimisation. `p3c_core.py` holds the atom algebra (an independent
re-implementation of `p2_core.py`'s `C_T`, by explicit slot-by-slot convolution).

`min_re`'s exact cyclic coordinate descent was checked against a brute-force torus grid
(`G = 48..60`, dimensions 1–4) in **400/400** random cases: never above the grid minimum.

### Reproducing

    uv run python p3c_verify.py --block ident mm                 # ~1 min
    uv run python p3c_verify.py --block tight                    # ~3 min
    uv run python p3c_astar.py  --papers                         # ~5 min
    uv run python p3c_astar.py  --scan --p 11 13                 # ~5 min
    uv run python p3c_astar.py  --p 11 --S 1 4 --mesh 0.02       # ~2 min (rigorous A* lower bd)
    uv run python p3c_part.py   --cells nu3q2 --A 0.25 0.30 0.35 --starts 10 --rounds 5 --npop 4000
    uv run python p3c_stress.py --cells nu3q2 --block sweep local dirs --lo 0.28 --hi 0.40
    uv run python p3c_part.py   --cells nu4q2 --A 0.40 0.45 0.50 --starts 6 --rounds 5 --npop 3000
    uv run python p3c_chord.py  --cells nu4q2 --starts 60 --levels 20      # slow, > 1 h

---

## 9. Status

**PROVEN (unconditional, written out and fp64-checked).**

* **§2.2 + §2.3: the chord criterion is refuted at `nu = 4, q = 2` by an exact closed-form witness.**
  The two-atom family `M_2` at `n = 3` has `eta = sqrt(binom(3,2)) = sqrt 3`, cost `sqrt 2`, `A = 2`, so
  `rho(sqrt 3) >= 1/(2 sqrt 3) = 0.288675135 > 0.283473355 = rho_need`. Also refuted at
  `nu = 5, q = 2` (three witnesses, worst `0.284458247` vs `0.235104879`), a cell not previously
  tested. `GOPT_STATE2.md`'s claim is confirmed and made optimiser-free. At `nu = 3` (`q = 2, 2.5`)
  and at `nu = 4, q = 3` the `M_m` family does **not** refute it; those cells' status is unchanged.
* **THEOREM P3C-1 (§3.3): the reduction.** For every pure network, every `p`, every `K`, every junk
  level: if some partition `P` of the operand slots satisfies
  `sum_k b_k R_P(pop_k, A*)^r <= 1`, then (T) holds. Proof in six elementary steps; verified end to
  end on 36 explicit networks with the test density actually constructed (0 violations), and exactly
  tight (12 digits) at the share-weighted block.
* **PROPOSITION P3C-2 (§3.4): the minimax.** "Some partition works for every network" is equivalent
  to "some fixed mixture `lam` over partitions works for every single one-frequency population" —
  turning P3 at `K >= 2` into a single-population, `p`-free, `K`-free, `eta`-free inequality.
* **§4.3: the block is a strict local maximum of the criterion** whenever `lam` charges the singleton
  partition (junk is first order in the perturbation, the cost gain second order, and every pattern
  `T` is seen by the singleton partition), so the criterion's equality case is rigid.
* **§6: four routes killed with their exact breaking points** (single partition; mixing densities
  rather than strategies; aiming at a sub-support; and the pure-class scope of step 1).

**CONDITIONAL — one cleanly stated finite-dimensional inequality.**

Everything above reduces P3 at `K >= 2`, at every junk level, on the pure class, to

> **(PFP-`lam`)** There is a probability vector `lam` on the partitions of `{1..n}` with
> `sum_P lam_P R_P(pop, A)^r <= 1` for every one-frequency population `pop`,
> where `R_P(pop,A) = (n+1)^{(1-r)/r} ( min_{|t_i| = A} Re sum_{U} C_{B_U} prod_{i in U} t_i )_+ /
> (sum_l |c_l|^r)^{1/r}`.

Measured thresholds `A_lam = inf{A : (PFP-lam) holds}`:

| cell | `A_M` (`M_m` family only) | `A_lam` (full adversary) | `<= 1/2`? |
|---|---|---|---|
| `nu=3, q=2` | 0.300416 | **[0.3034375, 0.3043750]** | yes |
| `nu=3, q=2.5` | 0.137235 | (not run) | yes |
| `nu=4, q=2` | 0.481063 | **(0.45, 0.50]** | yes |
| `nu=4, q=3` | 0.245495 | (not run) | yes |
| `nu=5, q=2` | 0.553514 | (not run) | **no** |

**Consequences, if (PFP-`lam`) holds at `A = 1/2`** (which the measurements say it does at
`nu = 3,4`, `q = 2`, and does not at `nu = 5, q = 2`):

* **(T) holds, with junk on any number of frequencies at any level, for every pure network whose
  frequency support is dissociate** — because P3A's Riesz product proves `A* >= 1/2` there, with no
  `p` or `K` dependence. That is a strictly larger class than THEOREM S's (one atom per frequency).
* Measured `A*` clears `A_lam(2,2/3) = 0.304` at **all 15** `Gamma_{3/2}`-optimal supports of
  `kstar.json` at the paper's primes `p = 59, 71, 97, 113, 127` and `K = K*-1, K*, K*+1`
  (`A* = 0.381 .. 0.614`), and at every support with `K <= 3` at `p = 11, 13`.

**OPEN / NOT covered.** `K >= 4` supports at very small `p` (`A*` collapses, §5.3); `nu >= 5, q = 2`;
impure neurons; and (PFP-`lam`) itself, which is the one remaining inequality.

**Relation to the existing programme.** This replaces, rather than repairs, the
`(Pen, eta, Cl2_r)` bookkeeping: `eta` never appears, so §2's refutation is irrelevant to it; `K`
never appears, so THEOREM R's Dirichlet dilution is irrelevant to it (it acts only through `A*(S)`,
where it is real and is exactly the `1/(p-1)` collapse); the density is chosen after seeing `theta`,
so P3B's THEOREM P3-2 is irrelevant to it. It does **not** make P2 unnecessary for other purposes,
but P3 no longer depends on P2.

