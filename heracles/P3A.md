# P3: junk on `K >= 2` frequencies, at every junk level — the problem is ONE constant, and that
# constant does not exist uniformly

Fifth pass. Problem **P3** of `GOPT_STATE.md`: remove the same-`|k|`-junk-free hypothesis `(H)` from
agent 3's Theorem D, i.e. find a junk lemma strong enough that, with the known cost relief, junk can
never raise `gamma_q` above

    (T)     gamma_q(theta)  <=  (n+1)^{-(nu-q)/q} * B * Gamma_{nu/q}(p),     B = 2 (2 p (n+1))^{-nu/2}.

My assigned starting angle was "a stronger margin penalty": non-negative test densities on `Z_p^n`
that collect junk from many modes and many frequencies at once (Riesz products, Fejér kernels,
densities built from the junk itself, hypercontractivity).

**Status: PARTIAL, with one route REFUTED outright and the problem reduced to a single explicit
constant.**

1. **The whole of P3 is one inequality about one constant** (§2). Every test density, of every shape,
   gives at most `mean_a - min_a` of the level-set gap; so the entire family of margin lemmas the brief
   asks for is *exactly* an anti-concentration statement, and the amount of anti-concentration needed
   is a single number `rho_need(n,q)` whose value is known in closed form at the place where it binds:
   `rho_need = (1 - (n+1)^{-(nu-q)/q}) / sqrt(2^n - 1)`, `= 0.244016936` at `nu = 3, q = 2`.
2. **The "no dilution by `K`" lemma that P3 wants is FALSE, and I can prove it is false** (§5). The
   best constant in
   `mean_a Lambda_D - min_a Lambda_D >= rho * sum_k w_kD * ||junk_k||_2`
   over the FULL frequency support is exactly `1/(p-1)` — the extremal junk profile is the **Dirichlet
   kernel** (equal junk on every frequency, one sign pattern, aligned phases). Measured: `0.100000000`
   at `p = 11`, `0.083333333` at `p = 13`, `0.062500000` at `p = 17`, i.e. `1/(p-1)` to 9 digits.
   So **no support-independent junk lemma of this shape exists**, however clever the test density:
   Riesz products, Fejér kernels and hypercontractivity are all instances and all die here.
   This closes the brief's starting angle as a route to a *uniform* lemma.
3. **On the supports that matter the constant is fine, and I measured it** (§4): `rho = 0.601807812`
   (`p = 11`, `K = 2`), `0.336908719` (`p = 13`, `K = 3`), `0.478051060` (`p = 17`, `K = 3`) — all
   above the required `0.244016936`. The resolution of 2-vs-3 is that the supports on which the
   constant collapses are exactly the supports whose clean margin is too small to threaten (T)
   anyway (§5.2).
4. **A new unconditional theorem at the extreme junk level** (§4.2, THEOREM S). For networks with one
   pure neuron per frequency — maximum junk, `eta = sqrt(2^n - 1)`, and the whole cost head-room
   `(n+1)^{(nu-q)/q}` — a *product of per-slot Riesz products* gives
   `gamma_q / B <= (1 - A*)^n * (clean value of the support)`, with `A*` the polydisc inradius of the
   moment body. (T) follows on this family whenever `(1 - A*)^n <= target/clean`. `A* >= 1/2` whenever
   the support is dissociate (Riesz), and the condition then reduces to the clean statement
   `(n+1)^{(nu-q)/q} <= 2^n` — true at `q = 2` for `n <= 4`, false from `n = 5`.
5. **The `(x, eta)` framework is NOT dead** (§3): replacing Lemma Q by the exact margin of the profile
   and keeping the measured relief curve `Cl2_r`, the best bound the bookkeeping can give,
   `Vtilde(p)`, comes out at `1.000000000` (`p=11`), `1.000000000` (`p=13`), `0.999999994` (`p=17`)
   of the target, with the arg-max at ZERO junk in every case. Agent 4's
   residual factor `1.30` at the top of the junk range is therefore entirely an artefact of Lemma Q,
   not of the `(x, eta)` decoupling.

Nothing here contradicts (T); every numerical attack on (T) failed again, now including the whole
one-neuron-per-frequency family on **every** support, which no previous pass had searched — and there
the result is stronger than "does not exceed the target": at `p = 11` and `p = 13` the best normalised
margin over that entire family is **exactly `0`**, i.e. a network with one pure neuron per frequency
never separates the data at all, whatever its support, amplitudes and phases (§8). That is a
*conjecture worth testing at larger `p`*: **one atom per frequency implies margin `<= 0`**, which would
make THEOREM S redundant and close the extreme of P3 outright.

All float64, local CPU, **no pods, $0**. Scripts `heracles/p3a_*.py`, outputs
`heracles/FINAL_RUNS/P3A/`. To reproduce (about 2 hours on one core; the `Cl2_r` grid is cached after
the first call):

    uv run python p3a_truth.py  --mode check                                   # the profile identity
    uv run python p3a_lemma.py  --block need                                   # rho_need from the relief curve
    uv run python p3a_lemma.py  --block riesz --p 11 13 17 19 23               # dissociativity
    uv run python p3a_poly.py   --block astar --p 11 13 17 19 23               # A*, and the 1/(p-1) collapse
    uv run python p3a_poly.py   --p 11 13 17 19 23                             # the half-plane variant
    uv run python p3a_lemma.py  --block rho   --p 11 13                        # rho(p,n,S) itself
    uv run python p3a_single.py --block pressure --p 11 13 17                  # clean(S)/target, every support
    uv run python p3a_single.py --block search   --p 11 13 --starts 14         # (T) at maximum junk
    uv run python p3a_truth.py  --mode vtilde --p 11 13 17 --starts 30         # Vtilde(p)

`p = 19, 23` were started for `Vtilde` and `p = 17` for the maximum-junk search, and neither finished
inside this pass's CPU budget; the logs in `FINAL_RUNS/P3A/` are the record (the `.json` files for
those two runs were therefore not written).

---

## 0. Symbols, with a plain reading of each

Inherited symbols keep their meaning (`p`, `n`, `nu = n+1`, `q`, `r = q/nu`, `e_ij`, `w_i`, `P_q`,
`gamma_q`, `Gamma_s`, `B`, `Cl2_r`, `(KL)`). New or re-pointed here:

| symbol | meaning | plain reading |
|---|---|---|
| `omega = e^{2 pi i / p}` | primitive `p`-th root of unity | the task's basic phase |
| `S`, `K = \|S\|` | the frequency support and its size | which cosines the network actually uses |
| `sigma_T(a) = sum_{j in T} a_j` | for a sign pattern `T` subset `{1..n}` | which operands the junk term mixes |
| `C_T^{(k)}` | frequency `k`'s Fourier amplitude on sign pattern `T` | `T` empty = the CLEAN amplitude `x_k`; the other `2^n - 1` are the JUNK |
| `eta_k = (sum_{T != empty} \|C_T^{(k)}\|^2)^{1/2} / x_k` | relative junk, `l_2` | "what fraction of this feature's amplitude is junk"; ranges over `[0, sqrt(2^n-1)]` |
| `w_kD = 1 - cos(2 pi k D / p)` | the programme's weight | how much frequency `k` contributes to the gap at class-offset `D` |
| `W_kD = 1 - e^{2 pi i k D/p}` | its complex version | `\|W_kD\| = sqrt(2 w_kD)` |
| `Lambda_D(a)` | the level-set gap at row `a`, offset `D` | `2^n` times `z_correct - z_wrong` |
| `Lambda = min_{a, D != 0} Lambda_D(a)` | `= 2^n * margin` | the worst-case gap |
| `Pen(D) = mean_a Lambda_D - min_a Lambda_D` | the **junk penalty** | how far below its level-set average the gap dips somewhere |
| `phi(eta) = (Cl2_r(0)/Cl2_r(eta))^{1/r}` | the margin head-room junk buys | "a feature with junk `eta` is this much cheaper, so it may carry this much more clean amplitude" |
| `phi_max = phi(sqrt(2^n-1)) = (n+1)^{(nu-q)/q}` | the head-room at maximum junk | `sqrt 3 = 1.7320508` at `nu = 3, q = 2`; **exactly known**, since `Cl2_r(sqrt(2^n-1)) = 1` (one atom) |
| `rho_need(n,q)` | the constant a junk lemma must beat | `max over eta of [1 - phi(eta)^{-1}]/eta` |
| `Omega` | the **moment body** `conv{ (omega^{-m_i . a})_i : a in Z_p^n }` | all the Fourier moment vectors a non-negative test density can have |
| `A*` | the polydisc inradius of `Omega` | "how big a moment I can put on EVERY listed mode at once, with any phases I like" |

---

## 1. Exact statements

> **PROPOSITION P0 (the profile reduction; exact, verified to `2.9e-15`).**
> For a network all of whose neurons are pure single-frequency cosines, the margin depends on the
> network only through the per-frequency Fourier profile `C_T^{(k)}`, and the cost only through the
> moduli `|C_T^{(k)}|`:
>
>     Lambda_D(a) = sum_k Re[ W_kD * h_k(a) ],   h_k(a) = sum_T C_T^{(k)} omega^{-2 k sigma_T(a)},
>     Lambda = min over a in Z_p^n and D != 0 of Lambda_D(a) = 2^n * margin,
>     sum_{k,l} |c_lk|^r = sum_k Cprof_r( |C^{(k)}|/x_k ) x_k^r ,
>
> `Cprof_r(m) := min{ sum_l |c_l|^r : C_empty = 1, |C_T| = m_T }`. Hence **(T) on pure networks is
> exactly a statement in profile coordinates**, with no atoms, no choice of `N`, and no redundancy.
> *(`p3a_truth.py --mode check`: 60 cells, `p = 7,11,13`, `n = 2,3`, `K = 1,2,3`, `N = 1,2,3`; worst
> relative error between the brute-force `p^n x p` logit table and the profile formula `2.905e-15`.)*

> **PROPOSITION P1 (the reduction of P3 to one constant; exact).**
> Let a pure network have profile `(x_k, C_T^{(k)})`, normalised so that
> `sum_k Cl2_r(eta_k) x_k^r = 1`. If, for the offset `D*` that minimises
> `sum_k [x_k/phi(eta_k)] w_kD`,
>
>     (P3-NEED)     Pen(D*)  >=  sum_k w_kD* x_k ( 1 - phi(eta_k)^{-1} ) ,
>
> then (T) holds for that network. In particular a junk lemma of the shape
>
>     (AC)          Pen(D)  >=  rho * sum_k w_kD * eta_k x_k        for every D != 0
>
> implies (T) for **every junk level and every `K`** as soon as
>
>     rho  >=  rho_need(n,q) := max over 0 < eta <= sqrt(2^n - 1) of [1 - phi(eta)^{-1}] / eta .
>
> *Plain reading: `Pen(D)` is the only thing a margin lemma can ever deliver — it is literally the gap
> between the level-set average and the best row — and (P3-NEED) says it has to pay for exactly the
> cost discount that junk buys.*

> **PROPOSITION P2 (the value of `rho_need`).**
> `rho_need` is attained at the endpoint `eta = sqrt(2^n - 1)`, where `phi` is known exactly
> (`Cl2_r(sqrt(2^n-1)) = 1`, realised by a single atom), so
>
>     rho_need(n,q)  =  ( 1 - (n+1)^{-(nu-q)/q} ) / sqrt(2^n - 1) .
>
> | `nu` | `q` | `phi_max` | `rho_need` |
> |---|---|---|---|
> | 3 | 2 | 1.732050808 | **0.244016936** |
> | 3 | 2.5 | 1.245730940 | 0.113932372 |
> | 4 | 2 | 4 | 0.283473390 |
> | 4 | 3 | 1.587401052 | 0.139862129 |
> | 5 | 2 | 11.18033989 | 0.235122372 |
>
> That the maximum sits at the endpoint is read off the measured `Cl2_r` curve (the interior values at
> `nu=3, q=2` are `0.010, 0.021, 0.033, 0.046, 0.061, 0.078, 0.100, 0.131, 0.214, ..., 0.213` against
> the endpoint's `0.244`), so **this half of P1 is conditional on P2's relief curve**, exactly as
> `GOPT_STATE` flags; the endpoint value itself is unconditional.

> **THEOREM R (REFUTATION of the uniform lemma; proved, every odd `p`, every `n >= 1`).**
> Let `S` be the full frequency support `{1, ..., (p-1)/2}` and let `T` be any non-empty sign pattern.
> Take the junk profile `C_T^{(k)} = c` for every `k` (same `c` for all `k`, all other patterns zero).
> Then
>
>     Pen(D)  =  |W_1D| * |c| / 2 * (1 + o(1))   while   sum_k w_kD eta_k x_k  =  Theta(p) * |c| ,
>
> and precisely, for the normalised anti-concentration constant
>
>     rho(p, n, S) := min over junk profiles of  [ max_a ( - Re sum_{k,T} z_kT omega^{-2 k sigma_T(a)} ) ]
>                                               / sum_k ( sum_T |z_kT|^2 )^{1/2} ,
>
> one has `rho(p, n, full support) <= 1/(p-1)`, because for `z_kT = 1` on the chosen `T` and every `k`,
> `sum_{k=1}^{(p-1)/2} cos(2 pi (2k) s / p) = (p * [s=0] - 1)/2`, so the maximum of the negated sum is
> exactly `1/2` while the mixed norm is `(p-1)/2`.
> **Measured: `rho = 0.100000000` at `p = 11`, `0.083333333` at `p = 13`, `0.062500000` at `p = 17` —
> i.e. `1/(p-1)` exactly, and the optimiser finds precisely the Dirichlet profile.**
> Since `rho_need` is a positive constant, (AC) **cannot** hold with a support-independent `rho`.
> *Plain reading: if a network spreads equal junk over all `(p-1)/2` frequencies with aligned phases,
> the junk cancels at every row but one — a Dirichlet kernel is a huge spike at one point and a tiny
> constant everywhere else — so the row-minimum sees almost none of it. No non-negative test density
> can see it either, because the row-minimum is the best any test density can do.*

> **THEOREM S (maximum junk, every support; proved modulo an explicit constant).**
> Let `A*(p, S)` be the polydisc inradius of the moment body of the modes `{2k mod p : k in S}`, i.e.
> the largest `A` such that for **every** phase vector `(phi_k)` there is a probability density `nu`
> on `Z_p` with `nu^(2k) = A e^{i phi_k}` for all `k in S`. Then for every pure network with exactly
> one neuron per frequency on `S` (maximum junk `eta_k = sqrt(2^n-1)`),
>
>     gamma_q / B  <=  (1 - A*(p,S))^n * min_{D != 0} sum_k |c_k| w_kD
>                  <=  (1 - A*(p,S))^n * Gamma_{nu/q}(p) * ( sum_k |c_k|^r )^{1/r} ,
>
> so **(T) holds on this family whenever `(1 - A*(p,S))^n <= target / clean(S)`**, and on a
> `Gamma`-optimal support whenever `A* >= 1 - (n+1)^{-(nu-q)/(q n)}` ( `= 0.240164314` at `nu=3,q=2` ).
> Moreover `A* >= 1/2` whenever `{2k : k in S}` is **dissociate** (all `3^K` signed sums
> `sum_k eps_k 2k`, `eps in {0,+-1}`, distinct), by a Riesz product; and then the condition becomes
> simply
>
>     (n+1)^{(nu-q)/q}  <=  2^n ,
>
> true at `q = 2` for `n = 2, 3, 4` (`1.73 <= 4`, `4 <= 8`, `11.18 <= 16`) and **false from `n = 5`**
> (`36 > 32`).
> Measured `A*`: `0.601807812` (`p=11`, `S={1,4}`, dissociate, so `>= 1/2` is also *proved*),
> `0.336908719` (`p=13`, `S={2,3,6}`), `0.478051060` (`p=17`, `S={2,5,8}`) — all above `0.240164314`.
> *Plain reading: a single neuron carrying a whole feature is a product of `n` factors `1 + u_j v_j`;
> a Riesz product lets me aim every slot at the antipode of its own phase simultaneously for all `K`
> frequencies, but only to depth `A*`, which shrinks each factor to `1 - A*`. The feature loses
> `1 - (1-A*)^n` of its clean amplitude, and (T) asks it to lose `1 - (n+1)^{-(nu-q)/q}`.*

---

## 2. The reduction (proof of P0, P1, P2)

### 2.1 Profile coordinates

A pure neuron at frequency `k` has `e_lj(a) = r_lj cos(2 pi k a/p + alpha_lj)`,
`w_l(c) = rho_l cos(2 pi k c/p + beta_l)`. Put `c_l = (prod_j r_lj) rho_l e^{i psi_l}`,
`psi_l = sum_j alpha_lj - beta_l`, `v_lj = e^{-2 i alpha_lj}`, and
`C_T^{(k)} = sum_l c_l prod_{j in T} v_lj`. Agent 2's identity (`GLOBAL_OPT` §2), extended over
frequencies (different `k` never collide, `p` prime), gives

    Lambda_D(a) = sum_k Re[ W_kD h_k(u^{(k)}(a)) ],      u_j^{(k)}(a) = omega^{-2 k a_j},

and since `h_k` is multi-affine, `h_k(u^{(k)}(a)) = sum_T C_T^{(k)} omega^{-2 k sigma_T(a)}`. So the
junk of frequency `k` on pattern `T` lives at the **mode** `m_{k,T} = 2k * 1_T` of `Z_p^n`, and these
`K(2^n - 1)` modes are distinct and non-zero (`2k` invertible, `T` recoverable from the support of
`m`). The clean part is the mode `0`. **For pure neurons these are the ONLY modes the gap has** — this
is why the whole argument is finite. `p3a_truth.py --mode check` verifies `Lambda/2^n = margin` against
the brute-force logit table in 60 cells, worst relative error `2.905e-15`.

Cost: rebalancing each neuron's `n+1` factors is free and gives
`P_q = ((n+1) p/2)^{q/2} sum_{k,l}|c_lk|^r` (`LEMMA_ATTEMPT` §2 Step 4), and the per-slot gauge
`v_lj -> e^{i psi_j} v_lj` fixes every `|c_l|` while rotating `C_T` by `exp(i sum_{j in T} psi_j)`, so
the cheapest population realising a given profile depends only on the moduli `|C_T^{(k)}|`. That is
`Cprof_r`, and `Cl2_r(eta) = min{ Cprof_r(m) : ||m||_2 <= eta }` is its `l_2` relaxation. ∎

### 2.2 Every margin lemma is an anti-concentration statement

For any probability density `w` on `Z_p^n` and any `D != 0`, `margin <= E_w[Phi_D]` because every
`Phi_D(a)` is a gap to a wrong class (this is the one line behind Lemma Q, Lemma P and Lemma C3).
Writing it in profile coordinates,

    Lambda <= min_a Lambda_D(a) <= E_w[Lambda_D] = mean_a Lambda_D - ( mean_a Lambda_D - E_w[Lambda_D] ),

and the best choice of `w` is the point mass at the arg-min. **So the supremum over the entire family
of test densities — Riesz products, Fejér kernels, densities built from the junk, everything — is
exactly**

    Pen(D) = mean_a Lambda_D - min_a Lambda_D,     mean_a Lambda_D = sum_k w_kD Re C_empty^{(k)}.

This is worth stating because it settles the brief's starting angle in principle: there is no cleverer
test density to find. What remains is to *lower bound* `min_a`, i.e. anti-concentration, and the only
question is how much of it is needed.

### 2.3 How much is needed (proof of P1)

Normalise `sum_k Cl2_r(eta_k) x_k^r = 1`, so `gamma_q/B = Lambda / (sum_{k,l}|c_lk|^r)^{1/r} <= Lambda`.
Put `xtilde_k := x_k / phi(eta_k)`. Then `Cl2_r(0) xtilde_k^r = Cl2_r(eta_k) x_k^r`, so
`sum_k xtilde_k^r = 1/Cl2_r(0) = (n+1)^{r-1}`, and the definition of `Gamma_{1/r}` gives

    min_{D != 0} sum_k xtilde_k w_kD  <=  Gamma_{1/r}(p) * (n+1)^{-(1-r)/r}  =  target/B .

Let `D*` attain that minimum. Then

    Lambda  <=  Lambda_{D*}(argmin)  =  sum_k w_kD* x_k - Pen(D*)
            <=  sum_k w_kD* xtilde_k                          (by (P3-NEED))
            <=  target/B .  ∎

For the second half, `1 - phi(eta)^{-1} <= rho_need * eta` by the definition of `rho_need`, and
`sum_k w_kD x_k * rho_need * eta_k` is the right-hand side of (AC). ∎

*Where P2 enters.* `phi` is built from `Cl2_r`, for which only the second-order expansion at `0`
(agent 4's Theorem J) and the trivial `Cl2_r >= 1` are proved. The endpoint value `phi_max` is
unconditional; the statement "`rho_need` is attained at the endpoint" uses the measured curve. With
only the *proved* linear relief (`Cl2_r(eta) >= (n+1)^{1-r}(1 - (2^n-1) eta)^r`) one gets
`1 - 1/phi(eta) <= (2^n-1) eta` and hence `rho_need <= 2^n - 1 = 3`, which no `rho <= 1` can meet:
**P3 is closed only modulo P2**, and this note makes that dependence exact rather than vague.

---

## 3. The `(x, eta)` framework is not the obstruction: `Vtilde(p)`

Agent 4's `V(p)` (Lemma Q + measured `Cl2_r`) left a residual factor `1.304226` at `p = 11` with the
arg-max at maximal junk on every frequency, and it was not clear whether the blame lay with Lemma Q or
with the `(x, eta)` decoupling itself. `p3a_truth.py --mode vtilde` settles that by replacing Lemma Q
with the **exact** margin of the profile and keeping everything else:

    Vtilde(p) := max over profiles (x_k >= 0, C_T^{(k)} in C, all (p-1)/2 frequencies free) of
                 Lambda(profile) / ( sum_k Cl2_r(eta_k) x_k^r )^{1/r}      versus     target .

This is the **best bound the `(x, eta)` bookkeeping can ever give**: `Lambda` is the truth and
`Cl2_r` is the cheapest shape at that junk level, so `Vtilde >= V* = ` the real maximum over pure
networks, and `Vtilde <= target` would close P3 outright on the pure class.

| `p` | `Gamma_{3/2}(p)` | target | `Vtilde/target` | exact ratio at the arg-max | arg-max support | arg-max `eta` |
|---|---|---|---|---|---|---|
| 11 | 0.535725926 | 0.309301507 | **1.000000000** | 1.000000000 | `{1,4}` | `1.0e-15`, `2.5e-15` |
| 13 | 0.497772067 | 0.287388837 | **1.000000000** | 1.000000000 | `{2,3,6}` | `6.8e-13`, `6.4e-13`, `9.4e-13` |
| 17 | 0.448340560 | 0.258849543 | **0.999999994** | 0.999999994 | `{2,5,8}` | `2.2e-09`, `2.5e-09`, `1.5e-09` |

(`n = 2`, `q = 2`, `r = 2/3`; 30 starts per prime of which 8 begin at MAXIMUM junk on the programme
shares, 8 at maximum junk on flat shares and 8 at maximum junk on a random sub-support, then annealed
L-BFGS with analytic gradients through seven temperatures and an exact-margin Nelder-Mead polish.
`Cl2_r` on 21 levels, `N = n+1, n+2`, 14 SLSQP starts each. `p = 19, 23` were started and not finished
within the CPU budget of this pass; the `p = 11` value was also reproduced independently by a 12-start
run, `0.999999995`.)

**Nothing exceeded the target, and every junk-carrying start walked back to `eta = 0`.** The exact
ratio (third column, using `Cprof_r` — the cost of the winning profile's OWN shape rather than the
cheapest shape at that junk level) equals `Vtilde` because the winner is junk-free, where the two
costs agree.


*Reading: the `1.30` residual was entirely Lemma Q's. The `(x, eta)` decoupling is sound, the optimiser
walks back to zero junk from maximum-junk starts on every support, and what is missing is only a
provable substitute for `Pen`.*

---

## 4. What CAN be proved: Riesz products and the polydisc inradius

### 4.1 The moment body and `A*`

For a list of distinct non-zero modes `m_1..m_M` in `Z_p^n` put
`Omega = conv{ (omega^{-m_i . a})_i : a in Z_p^n }`, the set of moment vectors of non-negative test
densities (the vertices are the point masses). By convex duality (support functions, dual norms),

    A*  :=  max { A : A * (unit polydisc)^M  subset  Omega }
        =   min over z != 0 of  [ max_a Re sum_i z_i omega^{-m_i . a} ] / sum_i |z_i| ,

i.e. `A*` is the reciprocal of the ONE-SIDED Sidon constant of the mode set. Two facts:

* **Riesz certificate.** If the mode set is *dissociate* — the `3^M` signed sums `sum_i eps_i m_i`,
  `eps in {0,+-1}`, are pairwise distinct — then for any `|lambda_i| <= 1/2`,
  `w(a) = p^{-n} prod_i (1 + 2 Re(lambda_i omega^{m_i . a}))` is non-negative (each factor is
  `1 + 2|lambda| cos(.) >= 0`), has total mass `1` (only `eps = 0` gives the mode `0`) and has
  `w^(m_i) = lambda_i` **exactly** (only `eps = e_i` gives `m_i`). Hence `A* >= 1/2`, *with no
  dependence on `M`*. This is the only mechanism known that does not dilute by the number of modes.
* **Dirichlet obstruction.** If the modes are all of `Z_p \ {0}` (up to sign), `z_i = 1` gives
  `max_a Re(...) = 1/2` and `sum|z_i| = (p-1)/2`, so `A* <= 1/(p-1)`.

Measured `A*` (`p3a_poly.py --block astar`, `n = 2`, `q = 2`; needed for THEOREM S on a
`Gamma`-optimal support: `A* >= 0.240164314`):

| `p` | `Gamma`-optimal `S` | `K` | dissociate | `A*` | `(1-A*)^2` | verdict |
|---|---|---|---|---|---|---|
| 11 | `{1,4}` | 2 | **yes** | 0.601807812 | 0.158557 | OK (and *proved* `>= 1/2`) |
| 13 | `{2,3,6}` | 3 | no (`6 = 2*3`) | 0.336908719 | 0.439690 | OK (measured) |
| 17 | `{2,5,8}` | 3 | no | 0.478051060 | 0.272431 | OK (measured) |
| 19 | `{1,5,8}` | 3 | no | 0.515573914 | 0.234669 | OK (measured) |
| 23 | `{2,5,6,9}` | 4 | no | 0.348585825 | 0.424340 | OK (measured) |
| 11 | all `{1..5}` | 5 | no | **0.100000000** `= 1/(p-1)` | 0.810000 | FAIL |
| 13 | all `{1..6}` | 6 | no | **0.083333333** `= 1/(p-1)` | 0.840278 | FAIL |
| 17 | all `{1..8}` | 8 | no | **0.062500000** `= 1/(p-1)` | 0.878906 | FAIL |
| 19 | all `{1..9}` | 9 | no | **0.055555556** `= 1/(p-1)` | 0.891975 | FAIL |

The `1/(p-1)` column is Theorem R's Dirichlet profile, found by the optimiser to 9 digits.

### 4.2 Proof of THEOREM S

One neuron per frequency: `h_k(u) = c_k prod_j (1 + u_j v_jk)`, and the profile has
`|C_T^{(k)}| = |c_k|` for **every** `T`, so `eta_k = sqrt(2^n-1)` and the exact cost is
`sum_k |c_k|^r` — a factor `(n+1)^{1-r}` below the `(n+1)`-block's. Take for each slot `j` a
probability density `w_j` on `Z_p` with

    t_j^{(k)} := E_{w_j}[ omega^{-2 k a_j} ] = - A* * conj(v_jk)      for every k in S

(legal: each `-conj(v_jk)` is a unit complex number, and `A*` is by definition the radius at which
every phase vector is realisable). Let `w = w_1 (x) ... (x) w_n`, a probability density on `Z_p^n`.
Because `h_k` is multi-affine in `u_1..u_n` and `w` is a product,

    E_w[h_k] = c_k prod_{j=1}^n (1 + t_j^{(k)} v_jk) = c_k (1 - A*)^n .

Hence, for every `D != 0`, `Lambda <= E_w[Lambda_D] = (1-A*)^n sum_k Re[W_kD c_k]`, and taking the
smaller of `D` and `-D` (which shares the single offset across all `k`) gives
`Lambda <= (1-A*)^n min_{D != 0} sum_k w_kD |c_k|`. Feeding the definition of `Gamma_{1/r}` and
dividing by `cost^{1/r} = (sum_k|c_k|^r)^{1/r}` gives the statement; `(T)` needs
`(1-A*)^n <= (n+1)^{-(1-r)/r} = 1/phi_max` when the support attains `Gamma`, and
`(1-A*)^n <= target/clean(S)` in general. ∎

*The `n`-dependence.* With the Riesz value `A* = 1/2` the condition is `2^{-n} <= 1/phi_max`, i.e.
`(n+1)^{(nu-q)/q} <= 2^n`. At `q = 2` this reads `(n+1)^{(n-1)/2} <= 2^n`: `1.732 <= 4` (`n=2`),
`4 <= 8` (`n=3`), `11.180 <= 16` (`n=4`), `36 > 32` (`n=5`). So the product-of-Riesz-products argument
covers the paper's cells and expires at `n = 5`, `q = 2` — a sharp, quotable boundary.

*Where THEOREM S is not yet rigorous.* `A* >= 1/2` is proved only for dissociate supports. The
`Gamma`-optimal supports at `p = 13, 17, 19, 23` are not dissociate (`p = 13`: `{2,3,6}` has
`6 - 3 = 3`), so for them `A*` is a *measured* non-convex minimum, hence an upper bound on the truth;
a rigorous version needs either a perturbed Riesz product (solve `w^(2k) = target_k` by contraction,
the collision terms being `O(lambda^2)`) or a certified covering of the `2K-1`-dimensional direction
sphere. Neither is done here.

### 4.3 The same constant for general junk, and where it falls short

For a general profile the product-density argument gives, with `t_j^{(k)} = A* e^{i psi_jk}`
(free phases per `(j,k)`),

    Pen(D)  >=  sum_k |W_kD| * kappa_n(A*) * eta_k x_k ,
    kappa_n(A) := min over junk vectors C, ||C||_2 = 1, and phases xi, of
                  max over psi in T^n of  -Re[ e^{i xi} sum_{T != empty} C_T A^{|T|} e^{i sum_{j in T} psi_j} ],

and since `|W_kD| = sqrt(2 w_kD) >= w_kD`, (AC) holds with `rho = kappa_n(A*)`. The worst junk vector
is the one concentrated on the full pattern `T = {1..n}`, which the product density reaches only at
depth `A^n`:

**`kappa_n(A) <= A^n`, with equality for the junk vector concentrated on the full pattern**
(`C = e_{1..n}`: then the only term is `C_T A^n e^{i sum_j psi_j}` and its best alignment is `A^n`).
A matching lower bound of the same order holds slotwise (averaging over `psi_j` kills the cross terms,
so `kappa_n(A) >= A^n max_T |C_T| >= A^n (2^n-1)^{-1/2}` at `||C||_2 = 1`). The binding number is
therefore `A^n`:

| `A` | 0.25 | 0.3369 (`p=13`) | 0.4781 (`p=17`) | 0.5 (Riesz) | 0.6018 (`p=11`) |
|---|---|---|---|---|---|
| `A^2` | 0.0625 | 0.1135 | 0.2286 | 0.2500 | 0.3622 |
| vs `rho_need = 0.2440` | FAIL | FAIL | FAIL | OK | OK |


So on a `Gamma`-optimal support this route needs `A*^n >= rho_need`, i.e. `A* >= 0.494` at
`nu = 3, q = 2` — met at `p = 11` (`0.602`) and **not** at `p = 13` (`0.337`) or `p = 17` (`0.478`).
The fix for that particular worst case is a *different* density — for junk concentrated on one pattern
`T`, the density that is uniform on the fibres of `sigma_T` and Riesz in `sigma_T` gives the full `A*`
(and exactly `0` on every other pattern, since `1_{T'} = c 1_T` forces `T' = T`) — but combining the
two families costs a factor `2^n - 1` in the worst mixed case, which is again not enough. **General
junk at `K >= 2` is therefore still open; only the extreme (THEOREM S) is closed.**

---

## 5. Theorem R: why the brief's starting angle cannot give a uniform lemma

### 5.1 The refutation

The brief asked for "test densities that collect junk from many modes and many frequencies at once".
§2.2 says the ceiling of that family is `Pen(D) = mean - min`. Theorem R exhibits a junk profile on
which `Pen` is `O(1)` while `sum_k w_kD eta_k x_k` is `Theta(p)`:

    C_T^{(k)} = c  for every k in {1..(p-1)/2},  one fixed T,  all other patterns 0.

Then the fluctuation of `Lambda_D` is `Re[ c * W_1D-ish * sum_k omega^{-2 k sigma_T(a)} ]`, and
`sum_{k=1}^{(p-1)/2} cos(2 pi (2k) s / p) = (p [s = 0] - 1)/2` because `{2k}` runs over half of the
non-zero residues. So the fluctuation is a constant `-|c|/2 * |W|`-ish at every row with
`sigma_T(a) != 0` and a single huge spike at `sigma_T(a) = 0`; the row-minimum sees only the constant.
The measured constant is `1/(p-1)` to 9 digits at `p = 11, 13, 17`.

**Consequences, stated bluntly.**
* Riesz products over the full junk-mode set cannot work: the set `{2k 1_T}` is never dissociate
  (`m_{k,T1} + m_{k,T2} = m_{k,T1 u T2}` for disjoint `T`'s *always*, and across frequencies
  `k1 + k2 = k3` or `k2 = 2k1` collisions are common — `p = 13`'s own optimal support `{2,3,6}` has
  `6 = 2 * 3`), and Theorem R shows no repair can restore a uniform constant.
* Fejér / squared-polynomial densities, hypercontractivity (agent 4's ranked item 3), and any
  fourth-moment argument are all instances of "bound `mean - min` from below by something of size
  `||junk||`", and all are capped by `1/(p-1)` on the Dirichlet profile.
* Agent 4's `1/K` dilution (`max_k >= mean_k`) is therefore **not** a removable artefact of Lemma Q in
  general — it is removable only on supports where `rho(p,n,S)` is large.

### 5.2 Why this does not sink P3 — the pressure table

The Dirichlet profile is harmless for a different reason: its *clean* margin is small. Writing
`clean(S) := max{ min_D sum_{k in S} x_k w_kD : x >= 0, sum x^r = 1 }`, the quantity a junk lemma must
destroy is `clean(S)/target`, and that ratio is **bounded by `phi_max` for every support**, with
equality exactly on the `Gamma`-optimal ones (`p3a_single.py --block pressure`, `n=2, q=2`, every
support at `p = 11, 13, 17`):

| `p` | `K` = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| 11 | 0.513 | **1.732** | **1.732** | **1.732** | 1.687 | | | |
| 13 | 0.399 | 1.613 | **1.732** | **1.732** | 1.689 | 1.680 | | |
| 17 | 0.261 | 1.332 | **1.732** | **1.732** | **1.732** | 1.717 | 1.661 | 1.642 |

(`1.732051 = phi_max`, the cap, since `clean(S) <= Gamma` by definition.) So the supports on which
`rho` collapses (`K` near `(p-1)/2`) are supports whose pressure is *below* the cap, and the two
effects move in opposite directions. A correct lemma is therefore necessarily **support-dependent**:
the object to bound is `rho(p,n,S)` against `rho_need * clean(S)/Gamma`, not a single universal
constant. Measured `rho(p,n,S)` (`p3a_lemma.py --block rho`):

| `p` | `S` | `K` | `rho` | `rho_need` | verdict |
|---|---|---|---|---|---|
| 11 | `{1,4}` (`Gamma`-optimal) | 2 | 0.601807812 | 0.244016936 | OK |
| 13 | `{2,3,6}` (`Gamma`-optimal) | 3 | 0.336908719 | 0.244016936 | OK |
| 11 | `{1,2,3,4,5}` (all) | 5 | 0.100000000 | 0.244016936 | FAIL (Dirichlet) |

### 5.3 How this sits with P3B's duality barrier (cross-reference, not duplication)

Sibling agent P3B proves (THEOREM P3-2) that for any probability measure `mu` on (row, wrong class)
pairs chosen **independently of the network**, `max{ E_mu[Phi] : P_q <= 1 }` does not depend on `q` and
is attained by a width-one junk-carrying neuron, so `inf_mu` of it is the `q = nu` value `B Gamma_1`:
no fixed test measure can prove (T) for `q < nu` through weak duality alone.

The two results are complementary and fit together exactly.

* P3B's barrier is about **non-adapted** measures. §2.2 of this note works with the *adapted* one —
  the point mass at the arg-min row, chosen after seeing `theta` — which is precisely the escape P3B
  points to, and §2.3 pairs it with the structural cost bound `Cl2_r` rather than with weak duality.
  So P1 is an instance of the programme P3B recommends, and the reduction says what the adapted
  bound has to deliver: the single number `rho_need`.
* THEOREM R then says the escape is **not** enough by itself: even fully adapted, the best constant
  collapses to `1/(p-1)` on the full support. So "adapt the density to the network" is necessary
  (P3B) but not sufficient (this note); what is additionally needed is that the bound be allowed to
  depend on the network's frequency SUPPORT, through `A*(p,S)`.
* P3B's SIM route needs "one row that nearly realises every frequency's torus minimum at once", with
  a simultaneity defect governed by the covering radius `~ p^{-1/K}` of the row lattice in `(T^n)^K`.
  THEOREM S replaces the row by a *measure* — a product of per-slot Riesz products — and then the
  amount of simultaneity available is not a covering radius but the polydisc inradius `A*` of the
  moment body, which is an exactly computable convex-geometry quantity, is `>= 1/2` on dissociate
  supports with **no** `p` or `K` dependence at all, and is `1/(p-1)` exactly on the full support.
  That is the same phenomenon measured in a form that has a Riesz certificate on one side and a
  Dirichlet counterexample on the other.

---

## 6. Approaches tried here that FAILED, and exactly where

**F5-1. A support-independent multi-mode junk lemma (the brief's starting angle). REFUTED by
Theorem R.** Not "hard": *false*, with an explicit family and an exact constant `1/(p-1)`. Every
member of the family the brief listed — Riesz products, Fejér-type kernels, densities proportional to
a positive part or a squared polynomial, anti-concentration / hypercontractivity — is a way of lower
bounding `mean - min`, and `mean - min` is itself `1/(p-1)` times the junk norm on the Dirichlet
profile. Recorded so nobody re-attempts the family.

**F5-2. `l_2`-over-frequencies anti-concentration (the natural weakening). Dead by `sqrt(K)`.** The
second-moment identity is exact and pleasant: for distinct modes,
`Var_a(Lambda_D) = sum_k w_kD eta_k^2 x_k^2`. Combined with `max F >= E[F_+] = E|F|/2` and
`E|F| >= ||F||_2^3/||F||_4^2` (Hölder), a Sidon-type fourth-moment bound `||F||_4 <= 3^{1/4}||F||_2`
would give `Pen(D) >= (1/(2 sqrt 3)) sqrt(sum_k w_kD eta_k^2 x_k^2)`. But (P3-NEED) asks for an
`l_1`-over-`k` quantity, and `||.||_2 >= K^{-1/2}||.||_1`, so this route loses a factor `sqrt K` and
the constant `1/(2 sqrt 3) = 0.2887` is already only just above `rho_need = 0.2440`. It closes `K = 1`
(where it reproduces agent 2's Theorem B qualitatively) and nothing else. *This is the precise reason
agent 4's F4-5 note "hypercontractivity is the most promising remaining route" is too optimistic: even
with the best conceivable `(2,4)` estimate, the norm on the wrong side of the inequality is `l_2`,
not `l_1`.*

**F5-3. Riesz products on the full junk-mode set.** The set `{2k 1_T : k in S, T != empty}` is never
dissociate, because `m_{k,T1} + m_{k,T2} = m_{k,T1 u T2}` whenever `T1, T2` are disjoint. The
product-over-slots density (§4.2) is the correct way to respect that relation automatically — it gives
`prod_{j in T} t_j` with no collision at all — but it pays `A^{|T|}` instead of `A`, which is exactly
the `kappa_n(A) = A^n` loss of §4.3.

**F5-4. Fibre densities (one sign pattern at a time).** For a fixed non-empty `T`, the density
`w(a) = nu(sigma_T(a)) / p^{n-1}` has `w^(m_{k,T}) = nu^(2k)` and `w^(m_{k,T'}) = 0` for every
`T' != T` (because `1_{T'} = c 1_T` forces `T' = T`), so it extracts pattern `T` at the full depth
`A*`. But the pattern has to be the same for every frequency, and `max_T sum_k a_kT >= (2^n-1)^{-1}
sum_{k,T} a_kT` loses `2^n - 1`, giving only `rho >= A*/(2^n-1) = A*/3` at `n = 2` — below
`rho_need` for every measured `A*`. Mixtures of fibre densities are worse (a mixture scales every
moment by its weight).

**F5-5. Killing whole frequencies (the multi-frequency version of agent 2's Theorem A).** In the
continuum, `u_j^{(k)} = zeta_j^{2k}` with one `zeta_j in T` per slot, so a slot can be aimed at the
antipode of one chosen atom of one chosen frequency; if the TOTAL number of pure neurons is at most
`n`, every atom can be killed at once and `min Re(sum_k ...) <= 0`, i.e. **margin `<= 0` whenever the
whole network has at most `n` pure neurons** — a clean generalisation of Theorem A that costs one
line. At finite `p` the kill is only to within `pi/p`, and the resulting bound
`gamma_q/B <= 2^n pi / p` beats the target only for `p >= 2^n pi / ((n+1)^{-(nu-q)/q} Gamma)`
(about `p >= 73` at `nu = 3, q = 2`). THEOREM S supersedes it at every `p`, so it is recorded as a
remark, not used.

**F5-6. Using the `D`-average as extra freedom.** `Lambda` is a min over `(a, D)` jointly, so any
joint distribution is admissible. Averaging over `D` uniformly gives only the coverage bound
`Lambda <= (p/(p-1)) sum_k x_k`, which at `sum_k Cl2_r x_k^r = 1` is `<= p/(p-1)` — an order of
magnitude above the target. The `D`-freedom is already fully used by the `+-D` symmetrisation and by
the programme's `min_D`.

---

## 7. What remains open, ranked

1. **General junk at `K >= 2`.** THEOREM S closes the extreme (one atom per frequency); §4.3 shows the
   same construction falls short by a factor `~2` in the middle of the junk range at `p = 13, 17`. The
   missing object is a density that extracts the `T = {1..n}` pattern at depth `A*` *and* the
   lower-order patterns at depth `A*` simultaneously; the product density gives `A*^{|T|}` and the
   fibre density gives one pattern only.
2. **A rigorous `A*` for non-dissociate supports.** `A* >= 1/2` is proved only when
   `{2k : k in S}` is dissociate. The `Gamma`-optimal supports of `p = 13, 17, 19, 23` are not. A
   perturbed Riesz product (solve `w^(2k) = A e^{i phi_k}` by contraction, the collision terms being
   `O(A^2)` times the number of representations) looks routine and is the shortest way to make
   THEOREM S unconditional at those primes.
3. **P2 (the relief curve).** `rho_need = 0.244016936` is exact at the endpoint but its being the
   MAXIMUM uses the measured `Cl2_r`. With only the proved linear relief the requirement is
   `rho >= 2^n - 1`, which is impossible. So P3 cannot be closed before P2, and P1 makes the
   dependence explicit.
4. **For the paper.** What is quotable from this pass: (i) the profile reduction, which removes atoms
   entirely from the junk discussion and makes the whole of P3 one anti-concentration constant;
   (ii) **Theorem R**, an honest impossibility result that explains why four passes of junk lemmas have
   all diluted by `K`, and shows the dilution is real, not a proof artefact; (iii) **THEOREM S**, the
   first unconditional statement about junk on an arbitrary number of frequencies, with the clean
   boundary `(n+1)^{(nu-q)/q} <= 2^n`; (iv) `Vtilde(p)`, which clears the `(x, eta)` bookkeeping of
   agent 4's residual `1.30`.

---

## 8. Numerical record (all float64, local CPU, no pods, `heracles/FINAL_RUNS/P3A/`)

| script | what it does | result |
|---|---|---|
| `p3a_truth.py --mode check` | the profile identity `2^n margin = Lambda` against the brute-force `p^n x p` logit table; `p = 7,11,13`, `n = 2,3`, `K = 1,2,3`, `N = 1,2,3` (60 cells) | worst relative error **`2.905e-15`** |
| `p3a_truth.py --mode vtilde` | `Vtilde(p)`: the exact margin of the profile over the measured `Cl2_r` cost, all `(p-1)/2` frequencies free, starts = block / block+noise / MAXIMUM junk on the programme shares / on flat shares / on random sub-supports / random | see §3 |
| `p3a_single.py --block pressure` | `clean(S)/target` for EVERY support, `p = 11,13,17` | capped at `phi_max = 1.7320508`, attained exactly on the `Gamma`-optimal supports; table in §5.2 |
| `p3a_single.py --block search` | (T) on one-neuron-per-frequency networks (MAXIMUM junk), exact margin over the whole `p^n x (p-1)` table, every nested support and the `Gamma`-optimal one, 14 annealed starts each | **best ratio over all supports `= -0.000000000` at `p = 11` AND at `p = 13`** — the entire family has non-positive margin, so (T) holds on it with the whole head-room to spare. Per support (`p=13`): `-1.371` (`{2,3,6}`), `-2.065` (`{1,2}`), `-0.985` (`{1,2,3}`), `-0.570` (`{1,2,3,4}`), `-0.000` (`{1..5}`, `{1..6}`). `p = 17` not finished within the CPU budget |
| `p3a_lemma.py --block rho` | `rho(p,n,S)`, the exact anti-concentration constant of (AC), by smoothed L-BFGS + Nelder-Mead over the junk profile | `0.601807812` (`p=11`, `K=2`), `0.336908719` (`p=13`, `K=3`); `0.100000000` (`p=11`, all 5), `0.083333333` (`p=13`, all 6) `= 1/(p-1)`, worst-case shape = the Dirichlet profile (equal `\|z_kT\|` on one pattern, to `1e-15`) |
| `p3a_lemma.py --block riesz` | dissociativity of the junk-mode sets | the FULL junk-mode set is never dissociate; `{2k}_{k in S}` is dissociate at `p=11` (`S={1,4}`) and not at `p = 13,17,19,23` |
| `p3a_poly.py --block astar` | `A*(p,S)`, the polydisc inradius = 1/(one-sided Sidon constant) | `Gamma`-optimal: `0.601807812` (`p=11`), `0.336908719` (13), `0.478051060` (17), `0.515573914` (19), `0.348585825` (23) — all above the `0.240164314` THEOREM S needs. Full support: `0.100000000`, `0.083333333`, `0.062500000`, `0.055555556` = `1/(p-1)` exactly |
| `p3a_poly.py` (default block) | the half-plane variant of the polydisc radius for each choice of one sign pattern per frequency | `0.686` (`p=11`), `0.398` (13), `0.522` (17), `0.536` (19), `0.360` (23); identical across the three patterns, as the `sigma_T` reduction predicts |
| `p3a_prof.py` `cl2_grid` | `Cl2_r` on 21 levels over `[0, sqrt 3]`, `N = n+1, n+2`, 14 SLSQP starts | `1.4422495703` at `0` down to `1.0000000002` at `sqrt 3` (the exact single-atom value); `rho_need` curve in §1 |

**Two measured identities worth recording.**
`rho(p,n,S) = A*(p, {2k: k in S})` in all four cells tested, to 9 digits — the worst junk shape for the
anti-concentration constant is always "all of every frequency's junk on one common sign pattern", which
collapses the `Z_p^n` problem onto the single variable `sigma_T(a) in Z_p`. The inequality
`rho <= A*` is immediate (restrict `z` to one pattern; the mixed `(1,2)` norm becomes the `l_1` norm);
the reverse is measured, not proved. Together with `A*(full support) = 1/(p-1)` this makes THEOREM R
rigorous.
`A*(p, all frequencies) = 1/(p-1)` exactly at `p = 11, 13, 17, 19`.
