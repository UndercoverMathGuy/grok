# VERIFY — hostile referee pass on the global-optimality proof chain (2026-09-21)

Independent audit of the claim the ICLR paper wants to state as a theorem. Every link was
**re-derived by hand from the write-ups' own definitions** and **re-computed with fresh code**
(`heracles/verify_*.py`, outputs `heracles/FINAL_RUNS/VERIFY/`); none of the authors' scripts was
imported or called. All float64 unless stated; `mpmath` at 40 digits where tightness matters.
Local CPU only, no pods, no background jobs left.

## THE CLAIM UNDER AUDIT

> For the gated chain `z(a_1..a_n, c) = sum_i W[c,i] prod_j E[j,i,a_j]` on the **full table** of
> n-operand modular addition mod a prime `p`, at plain weight decay `q = 2` and `nu = n+1` in
> `{3,4,5}`, every network satisfying
>
>     (H)   J_{k,T} := sum_i c_ik prod_{j in T} v_ijk = 0   for every positive frequency k
>                                                           and every non-empty T subset of {1..n}
>
> ("no same-|k| junk") satisfies
>
>     (T)   gamma_2  <=  (n+1)^{-(nu-2)/2} * B * Gamma_{nu/2}(p),     B = 2 (2 p (n+1))^{-nu/2},
>
> with equality at the share-weighted `(n+1)`-neuron common-phase block — hence `K*` is exactly the
> feature count of the max-margin network **in that class**.

Symbols: `gamma_q = margin / P_q^{nu/q}`; `margin` = min over all `p^n` rows and all wrong classes of
(correct logit − wrong logit); `P_q = sum_i ||theta_i||^q` with `||theta_i||^2 = sum_j ||E[j,i,:]||^2
+ ||W[:,i]||^2`; `c_ik = prod_j ehat_ij(k) conj(what_i(k))` is neuron `i`'s **atom** at frequency `k`
(unnormalised DFT), `v_ijk = conj(ehat_ij(k))/ehat_ij(k)` its phase datum, `G_k = sum_i c_ik` the
clean Fourier coefficient; `Gamma_s(p) = max{ min_{D != 0} sum_k x_k (1 - cos 2 pi k D/p) :
x >= 0, sum_k x_k^{1/s} <= 1 }`, `s = nu/q`.

---

## VERDICT TABLE

| # | link | verdict | severity of the worst issue |
|---|---|---|---|
| 1 | junk lemma L1 + phase lemma L2 + assembly L3 + per-frequency AM-GM + budget (THEORY_nsum, PROOF_AUDIT) | **VERIFIED** (re-derived + brute-force checked) | cosmetic (full-table scope must be stated) |
| 2 | Key Lemma, `N = n+1` (LEMMA_ATTEMPT §4) | **VERIFIED** (re-derived; one unstated non-degeneracy step) | cosmetic |
| 3 | Theorem B + the exact margin identity (2.1) (GLOBAL_OPT §2, §4) | **VERIFIED** as stated; **but it is a CONTINUUM statement** and the finite-`p` version is only Theorem C (`1+O(p^-2)`) | repairable — see L3-b (the claim as written is finite-`p` and does **not** follow from Theorem B alone at a single frequency) |
| 4 | Theorem E peeling, `kappa_1,2,3`, `D_h(n)`, P1B's `sum d_i <= n` correction | **VERIFIED**; P1B's correction is right and strictly improves `D_1(n)` for `n >= 4` | cosmetic (GLOBAL_OPT3's table is valid but suboptimal; its "numerical `S_n`" caveat dissolves) |
| 5 | Theorem D — whole network with impure neurons to per-frequency atoms, assembly into `Gamma` | **VERIFIED** (algebra re-derived line by line; `(H)` is used exactly as stated; cross-frequency terms genuinely absent) | none found |
| 6 | P1 (Key Lemma for every `N`) at `q = 2`, `nu = 3,4,5` | **VERIFIED as an argument**; both thresholds reproduced to every reported digit; and **newly CERTIFIED here** (closed-form branch 1 + exact-rational vertex enumeration + interval evaluation, §6.7) at `nu = 3,4,5` — which the inherited material did **not** achieve at `nu = 5` | repairable (R2, R3, R4) |
| 7 | equality case (share-weighted block attains) | **VERIFIED, and it is PROVED algebra, not numerics**: for the share-weighted block the two sides of (T) are *identically* equal for every `(p,n,q,S,b)`, so equality holds iff `b` maximises the programme (§ "LINK 7 (final)"). Confirmed at 50 digits (`|ratio − 1| ~ 1e-50`) by two independent implementations | repairable (R7: the paper's `K*` is a different object from the theorem's) |

**Bottom line.** **I could not break the claim.** Every inequality in the chain re-derives correctly,
there is no circularity, and no fresh numerical attack beat it — including two ascents inside the
`(H)` class with the block excluded from every start (best `0.9999999998`), and a from-scratch
search for a (KL) counterexample to P1 over `N = n+1 .. 2^n+3` whose minimum is exactly
`1.0000000000`. Moreover the audit **strengthens** the result: P1 at `q = 2`, `nu = 3,4,5` is now
certified in exact + interval arithmetic depending only on elementary lemmas and three exact
algebraic constants (§6.7) — the inherited package had no certificate at `nu = 5` at all.

What must change in the paper: one provenance/scope correction (R1), three genuinely unproved or
unstated steps with their repairs (R2, R3, R7), and bookkeeping (R4–R6, C1–C10).

*(Sections are appended in the order the work finished, so LINK 5, LINK 7 and X3 each have a
"(continued)" or "(final)" part further down; every part is labelled.)*

---

## LINK 1 — L1, L2, L3, per-frequency AM-GM, the `q`-budget

**What is claimed** (THEORY_nsum §1, §3, §6; GLOBAL_OPT3 §2 steps 1, 3, 4):
`margin <= margin(g)` with `g(D) = mean_a z(a, s(a)-D)` (L1);
`margin(g) <= min_{D != 0} sum_k 2|ghat(k)| (1 - cos 2 pi k D/p)` (L2);
`ghat(k) = sum_i prod_j ehat_ij(k) conj(what_i(k))` (L3); and the per-neuron budget
`sum_{k in H} |c_ik|^r <= (p/(2 nu))^{q/2} ||theta_i||^q` for `2 <= q <= nu`.

**Re-derivation.** L1: the translations `(a_1..a_n, c) -> (a_1+t_1, ..., a_n+t_n, c + sum_j t_j)`
act transitively on each level set `{s(a) - c = D}` and map rows to rows *of the full table*; the
margin is a min over the orbit and a min is at most a mean. **The full table is load-bearing** — with
a training fraction `< 1` the orbit leaves the training set and the step fails (PROOF_AUDIT F6 says
this; the theorem statement must carry it). L3 is the elementary fact that the `(k,..,k)` Fourier
coefficient of a product of one-variable functions is the product of their `k`-th coefficients — for
the gated chain there is no other monomial at all. Budget: AM-GM over the `nu` non-negative numbers
`|ehat_i1(k)|^2, ..., |ehat_in(k)|^2, |what_i(k)|^2` gives `|c_ik| <= (m_ik/nu)^{nu/2}`; then
`sum_k |c_ik|^r <= nu^{-q/2} sum_k m_ik^{q/2} <= nu^{-q/2} (sum_k m_ik)^{q/2}` (superadditivity of
`t^{q/2}`, needs `q >= 2`) `<= (p/(2nu))^{q/2} ||theta_i||^q` (Parseval, with the factor `1/2` for
the `+-k` pairing). Each step re-derived; the `q >= 2` requirement is exactly where it is claimed.

**Fresh checks** (`verify_network.py`, brute-force `p^n x p` logit tables, random *impure*
multi-frequency nets, plus the adversarial "one balanced neuron carrying every frequency"):

* dictionary `ghat(k) = p^{-n} G_k` against the brute-force clean part: max relative error
  **1.8e-14** over `p = 5..13`, `n = 2,3,4`.
* margin bound (L1 + L2 assembled): `margin/bound <= 0` on every random net (they have negative
  margins); the bound is exercised properly in link 5 below, where an optimised `(H)` net reaches
  `0.99999999983` of it.
* budget (2.2): `max lhs/rhs = 0.9667` at `q = 2`, `0.820` at `q = 2.5`, `0.700` at `q = nu` over
  `p = 5..13`, `n = 2,3,4` — never above 1, and the earlier claim that it is *sharp* at `q = 2`
  (attained by a balanced single-frequency neuron) is consistent with what I see.
* the bookkeeping identity `2 p^{-nu} (p/(2nu))^{nu/2} = 2 (2 p nu)^{-nu/2} = B`: relative error
  **2.2e-16**.

**VERDICT: VERIFIED.** One cosmetic requirement: every statement of the theorem must say
*full-table training*, because L1 is false otherwise.

---

## LINK 2 — the Key Lemma (`N = n+1`)

**Statement.** If `c_1..c_{n+1}` in `C` and `v_lj` on the unit circle satisfy
`(KL) sum_l c_l prod_{j in T} v_lj = A [T empty]` for every `T` subset of `{1..n}`, then
`prod_l |c_l| >= (|A|/(n+1))^{n+1}` and hence `sum_l |c_l|^r >= (n+1)^{1-r}|A|^r` for `r in (0,1]`.

**Re-derivation** (my own, following the four moves): (KL) is equivalent to the polynomial identity
`sum_l c_l prod_j (u_j - v_lj) = A prod_j u_j` (match the `2^n` monomial coefficients — the monomials
are linearly independent). Fix `l0`; for a **bijection** `sigma` from the `n` slots onto the other
`n` atoms put `u_j := v_{sigma(j),j}`; every other atom's product contains a zero factor, so
`|c_{l0}| prod_j d^{(j)}_{sigma(j), l0} = |A|` with `d^{(j)}_{lm} = |v_lj - v_mj|`. Geometric mean
over the `n!` bijections (each pair `(j,l)` occurs `(n-1)!` times, exponent `(n-1)!/n! = 1/n`);
product over the `n+1` choices of `l0` turns the double product into `prod_j V_j^2` with
`V_j = prod_{l<m}|v_lj - v_mj|`; Hadamard on the Vandermonde matrix of `n+1` unimodular points
(rows of Euclidean norm `sqrt(n+1)`) gives `V_j <= (n+1)^{(n+1)/2}`; AM-GM finishes.

**Issues found.** (i) The step "no factor vanishes, so all `v_lj` in a slot are distinct and every
`c_l != 0`" is used but only parenthetically stated in LEMMA_ATTEMPT (it follows from the bijection
identity with `A != 0`, because the *product* over `j` must be non-zero for **every** bijection, and
every pair `(j, m)` occurs in some bijection). Without it the division in the geometric-mean step is
illegal. **Cosmetic**, but it must appear in the paper. (ii) The equality case as stated ("iff the
`v_lj` are, for each `j`, a rotation of the `(n+1)`-st roots of unity that is the same rotation
pattern across `j`") is the Hadamard + AM-GM equality case plus feeding back into (KL); I checked the
"same rotation pattern" claim only at `n = 2` (LEMMA_ATTEMPT's rigidity argument, which I re-derived
and agree with) — for general `n` it is asserted, not proved, in the write-ups. It is not load-bearing
for (T) (only the inequality is), but the paper should not claim it for general `n` without proof.

**Fresh checks.** `verify_p1_stress.py` block (e) builds exact (KL) systems from scratch
(least-squares + null-space sampling, residual `< 1e-8`) in six adversarial families — random,
coalescing phases, the sign rule with noise, block perturbations, near-cancelling pairs, grids — and
minimises `sum|c_l|^r`; results in §6 below.

**VERDICT: VERIFIED** (severity of the issues: cosmetic).

---

## LINK 3 — Theorem B and the exact margin identity

**The identity (GLOBAL_OPT §2).** For a population of pure single-frequency neurons at one
frequency, with `h(u) = sum_l c_l prod_j (1 + u_j v_lj)`,

    2^n * margin = min over u in mu_p^n of [ M_1 Re h(u) - sin(2 pi/p) |Im h(u)| ],   M_1 = 1 - cos(2 pi/p).

I re-derived it: expanding the product of cosines gives
`z(a,c) = 2^{-n} Re[e^{i(theta_s - theta_c)} h(u)]` with `u_j = e^{-2i theta_{a_j}}`; putting
`c = s(a) - D` gives `Phi_D(a) = 2^{-n} Re[(1 - e^{i theta_D}) h(u(a))]`; because `2k` is invertible
mod `p`, `u(a)` sweeps all of `mu_p^n` independently of `D`; and
`min_D [(1-cos th_D) X + sin(th_D) Y] <= M_1 X - sin(2 pi/p)|Y|` with equality when
`|arg(X+iY)| <= pi/p`. **The second line of (2.1) is an inequality in general, an equality only
under that argument condition** — GLOBAL_OPT states this correctly ("`<=` that value always").

**Theorem B.** `sum_{l=1}^{n+1}|c_l|^r >= (n+1)^{1-r} max(R,0)^r`, `R = min_{u in T^n} Re h(u)`,
junk allowed. Re-derived: at the kill point `u_j = -conj(v_{sigma(j),j})` every atom other than `l0`
dies, so `R <= |h(u)| = |c_{l0}| prod_j d^{(j)}_{sigma(j),l0}`; then the same three moves as link 2.
Correct.

### L3-b. THE SCOPING ISSUE — severity: repairable, but it must be fixed in the paper

`R` is a minimum over the **continuous torus** `T^n`, while the margin is a minimum over the
**grid** `mu_p^n`, and `min_grid >= min_torus`. So Theorem B does **not** by itself give a
finite-`p` single-frequency statement; GLOBAL_OPT is explicit about this ("in the continuum sense of
§5") and supplies Theorem C, whose bracket is `1 + (n+1)^{(1-r)/r} eta_n(eps)` — i.e. `1.0102` at
`n=2, p=71`, `1.0040` at `p=113`, and `1.42` / `3.63` / `19.6` at `p = 11` for `n = 2/3/4`. The
bracket is replaced by exactly `1` only above Lemma C3's threshold `cot(pi/p)/2 >= 2^n - 1`, i.e.
`p >= 23 / 47 / 97` for `n = 2/3/4`.

**Why it matters for the claim under audit.** The claim as written is a finite-`p` statement with no
condition on `p`. Theorem B cannot supply it. **However**, the claim's actual proof route does not go
through Theorem B at all: link 5 (Theorem D) + link 6 (P1) is a finite-`p` argument from beginning to
end, because it prices the *clean amplitude* `G_k` through `(H)` rather than the continuum quantity
`min_{T^n} Re h`. So the repair is a statement of provenance, not new mathematics:

> **Repair.** State (T) on the `(H)` class as Theorem D + P1 (links 5, 6) and cite Theorems A/B/C
> only for the *single-frequency, junk-allowed* sub-result. Do **not** let a reader infer that
> Theorem B contributes to the `(H)`-class theorem; it does not, and its continuum hypothesis would
> otherwise leak a `1+O(p^{-2})` bracket into a statement that is exact.

A second, smaller provenance issue: ADVERSARY §6's version of P1 *does* use Theorem B (its "drop the
tail" step is `R_S >= 1 - 2^n tau` plus Theorem B), so **ADVERSARY's route inherits the continuum
quantity**; P1C's Lemma T and P1B's Proposition B1 do not (they use only the polynomial identity,
`|u - v| <= 2`, Hadamard and AM-GM). The paper should quote **Lemma T / B1**, not ADVERSARY's
assembly, since only the former is a finite, elementary, hand-checkable statement.

**VERDICT: VERIFIED as stated; the claim must not be routed through it (repairable).**

---

## LINK 4 — Theorem E, the peeling constants, and P1B's correction

**Re-derivation.** For any multi-affine `Lambda(z) = sum_T lam_T prod_{j in T} z_j`, (KL) gives
`sum_l c_l Lambda(v_l) = lam_empty A`. Take `S` = the `h` largest atoms, give each of them a
*disjoint* group `J_i` of slots, and put
`Lambda = prod_i [1 - prod_{j in J_i}(1 + z_j conj(v_{l_i,j}))/2] / (1 - 2^{-|J_i|})`. Each factor has
constant coefficient `1`, vanishes at `v_{l_i}` and has sup-norm `S_{|J_i|}/(1 - 2^{-|J_i|}) =
1/kappa_{|J_i|}`. Hence `|A| <= (prod_i 1/kappa_{d_i}) sum_{l>h} x_l`, i.e.
`(PEEL_h) sum_{l>h} x_l >= D_h(n)|A|` with `D_h(n) = max{prod_i kappa_{d_i} : d_i >= 1, sum_i d_i <= n}`.
Then `sum_l |c_l|^r >= L_r(n) := min{ sum x_l^r : x sorted, (PEEL_h) h = 0..n }`, a concave
minimisation over a pointed polyhedron whose recession cone raises the objective — so the minimum is
at a vertex and vertex enumeration is exact.

**The `S_d` diagonal reduction is correctly proved.** On `T^d`, `prod_j (1+z_j)/2 = m e^{i phi}`
with `z_j = e^{2 i psi_j}`, `m = prod_j cos psi_j`, `phi = sum_j psi_j`; the critical-point equation
`tan(psi_j)(cos phi - m) + sin phi = 0` forces a common `tan psi_j`, and sign flips leave
`m e^{i phi}` invariant, so the sup is the diagonal one. The degenerate branch `cos phi = m` forces
`sin phi = 0` and `|m| = 1`, i.e. `u = 1`, which is harmless. I re-derived this and confirmed it
numerically by a **full `d`-dimensional** multistart search (`verify_constants.py`): full-torus
maximum minus diagonal maximum is `<= 4.4e-16` for `d = 1..6`.

**Fresh numbers** (`verify_constants.py`, 40-digit):

| `d` | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| `S_d` | 1 | 1.154700538379251 | 1.285714285714286 | 1.380900986983999 | 1.452990673410724 | 1.509590370786036 |
| `kappa_d` | 0.5 | 0.649519052838329 | 0.680555555555556 | 0.678904576676114 | 0.666728298899520 | 0.652080868459330 |

`|S_2 - 2/sqrt3| = 0`, `|S_3 - 9/7| = 0`, `|kappa_2 - 3 sqrt3/8| = 0`, `|kappa_3 - 49/72| = 0` at 40
digits. GLOBAL_OPT3's closed form `S_4^2 = max_c (1 - 15c^4 + 16c^3 - 2c^2)` at `c = (6+sqrt21)/15`
reproduces `S_4` to 0.

**P1B's correction is CORRECT and strictly improves the constants.** `Lambda_v(z_J)` may be built on a
*proper* subset `J` of the slots — it is still multi-affine in all `n` variables, still normalised,
still vanishes at `v` — so the composition may satisfy `sum_i d_i <= n`, not `= n`. Since
`kappa_1 < kappa_2 < kappa_3 > kappa_4 > ...` (max at `d = 3`), only parts in `{1,2,3}` are ever
optimal and every `D_h(n)` is exact algebraic in `kappa_1 = 1/2`, `kappa_2 = 3 sqrt3/8`,
`kappa_3 = 49/72`. My recomputation of both conventions:

| `n` | `D_1` (`<= n`, P1B) | `D_1` (`= n`, GLOBAL_OPT3) | other `D_h` |
|---|---|---|---|
| 2 | 0.649519053 | 0.649519053 | `D_2 = 0.25` (same) |
| 3 | 0.680555556 | 0.680555556 | `D_2 = 0.324759526`, `D_3 = 0.125` (same) |
| 4 | **0.680555556** | 0.678904577 | `0.421875, 0.162379763, 0.0625` (same) |
| 5 | **0.680555556** | 0.666728299 | `0.442033800, 0.210937500, 0.081189882, 0.03125` (same) |
| 6 | **0.680555556** | 0.652080868 | same |

**Consequences.** (a) GLOBAL_OPT3's `D_h` table is *valid* (a max over a subset of the admissible
compositions is still a lower bound), just not optimal; nothing downstream of it is wrong.
(b) GLOBAL_OPT3's stated "rigour note" — that `n >= 4` needs an exact `S_n` or a rounded-down
`kappa_n` — **dissolves**: under the corrected definition no `kappa_d` with `d >= 4` ever appears,
so all constants at every `n` are exact algebraic. P1C's §3.6 caveat for its `n = 4` cell is
therefore removable, and I confirm below that the `n = 4` threshold is unchanged either way.
(c) I reproduce GLOBAL_OPT3 §4's `L_r` and `F` exactly:

| `nu` | `q` | `L_r` (mine) | `(n+1)^{1-r}` | `F` (mine) | `F` (GLOBAL_OPT3) | minimising vertex (mine) |
|---|---|---|---|---|---|---|
| 3 | 2 | 1.441380164 | 1.442249570 | 1.000905 | 1.000905 | `(0.350481, 0.350481, 0.299038)` |
| 4 | 2 | 1.969854379 | 2 | 1.030841 | 1.030841 | `(0.319444, 0.319444, 0.236111, 0.125)` |
| 5 | 2 | 2.526617550 | 2.626527804 | 1.101809 | 1.101809 | `(0.318630, 0.259495, 0.259495, 0.099880, 0.0625)` |
| 6 | 2 | 3.106914154 | 3.301927249 | 1.200369 | 1.201816 (`= n` convention: I get 3.105666856) | — |

**VERDICT: VERIFIED.** Issues: GLOBAL_OPT3's `D_h` definition is suboptimal (cosmetic, with the
repair already written down in P1B); the two write-ups disagree on `D_1(n >= 4)` and on
`F(nu=6, q=2)` (`1.200369` vs `1.201816`) and the paper must pick one convention.

---

## LINK 5 — Theorem D: whole networks, impure neurons, assembly into `Gamma`

**Re-derivation, line by line.**
*Step 1–2 (dictionary).* `g(D) = mean_a z(a, s(a)-D) = p^{-nu} sum_k G_k w^{kD}` with
`G_k = sum_i c_ik`; and the logit tensor's coefficient at the multi-index that flips the operands of
`T` with `k_c = -k` is `sum_i prod_{j in T} ehat_ij(-k) prod_{j not in T} ehat_ij(k) what_i(-k)`,
which equals `sum_i c_ik prod_{j in T} v_ijk = J_{k,T}` because `e` and `w` are **real**
(`ehat(-k) = conj(ehat(k))`). So `(H)` is literally "the atoms at frequency `k` satisfy (KL) with
`A = G_k`", **with one atom per neuron and no purity assumption**. I verified this dictionary against
the brute-force logit table to `1.8e-14`.
*Step 3.* `margin <= 2 p^{-nu} min_{D != 0} sum_{k in H} |G_k| (1 - cos 2 pi k D/p)`: L1 + L2, link 1.
*Step 4.* the per-neuron budget, link 1.
*Step 5.* Under `(H)`, Theorem E (or P1) at each frequency gives `sum_i |c_ik|^r >= L_r |G_k|^r`;
summing over `k` and using step 4, `sum_k |G_k|^r <= L_r^{-1}(p/(2nu))^{q/2} P_q`. Then by the
definition of `Gamma_s` with `s = 1/r`,
`min_D sum_k |G_k| g_k(D) <= Gamma_s (sum_k |G_k|^r)^{1/r}`, and
`gamma_q <= 2 p^{-nu}(p/(2nu))^{nu/2} L_r^{-1/r} Gamma_s = L_r^{-1/r} B Gamma_s`. With
`F = ((n+1)^{1-r}/L_r)^{1/r}` and `(1-r)/r = (nu-q)/q` this is exactly Theorem D. **I re-derived
every exponent; the assembly is correct.**

**Two things I specifically checked because the brief asked.**
1. **Is `(H)` used exactly as stated?** Yes, and only in step 5, and only per frequency. Nothing
   anywhere requires cross-frequency junk to vanish.
2. **Are cross-frequency terms genuinely absent?** Yes. They are killed *unconditionally* by L1
   (orbit averaging: every Fourier mode whose operand frequencies are not all `-k_c` has zero mean on
   every level set), not by `(H)`. This is what makes the theorem hold for impure neurons of
   arbitrary width, and it is also why ADVERSARY §5's finding — impure nets can beat their own
   purification on margin — does not threaten it.

**A structural remark the paper should make (not an error).** The `(H)` class is thin: it is
`(p-1)(2^n-1)/2` complex conditions. So Theorem D is *not* a statement about the unconstrained
max-margin network; P3 (junk) is the gap, and P3A records that the "chord criterion", the registered
sufficient condition for removing `(H)`, is **refuted at `nu = 4`** and knife-edge at `nu = 3`. The
claim's own wording ("in that class") is correct and must be kept.

**Fresh numerics — an adversarial ascent inside the `(H)` class** (`verify_ascent_H.py`; my own
torch/L-BFGS annealed soft-min ascent, built on a *gauge-symmetrisation* parametrisation that makes
`(H)` exact by construction while leaving neurons impure and cross-frequency junk intact; **the block
is excluded from every start** — starts are random, junk-maximal one-neuron-per-frequency,
deliberately two-frequency impure, and clustered-support; every reported number recomputed by exact
brute force over all `p^n` rows, and `Gamma_s` solved by my own max-min NLP with 120 restarts plus
an exhaustive equal-amplitude floor):

| `p` | `n` | `q` | `Gamma_s` | max same-`|k|` junk | best `gamma_q / [(n+1)^{-(nu-q)/q} B Gamma]` |
|---|---|---|---|---|---|
| 7 | 2 | 2 | 0.67357531 | 4.2e-07 | **0.9999999998250851** |
| 11 | 2 | 2 | 0.53572593 | 8.2e-11 | 0.9999998461025299 |
| 13 | 2 | 2 | 0.49777207 | 1.5e-10 | 0.9995382083177125 |
| 11 | 2 | 2.5 | 0.79725763 | 7.3e-10 | 0.9523998075238203 |
| 5 | 3 | 2 | 0.69098301 | 4.4e-16 | **0.9999999999517954** |
| 7 | 3 | 2 | 0.47012486 | 6.7e-07 | 0.9999997713606937 |

Nothing exceeded 1 — including in the `nu = 4` cells, where the *proved* Theorem D only guarantees
`1.030841`, so the `F = 1` claim is being genuinely tested and survives.

**VERDICT: VERIFIED.** No issue found in the derivation.

---

## LINK 6 — P1: the Key Lemma for EVERY number of atoms, at `q = 2`, `nu = 3,4,5`

This is the load-bearing link: Theorem D alone gives `F = 1.000905 / 1.030841 / 1.101809`, and the
claim under audit is the `F = 1` version, which is exactly

>   **(P1)**  `sum_{l=1}^N |c_l|^r >= (n+1)^{1-r}|A|^r` for every `N` and every (KL) system.

### 6.1 The three versions, and which one the paper should quote

`ADVERSARY.md` §6, `P1B.md` and `P1C.md` give the same architecture. They are **not** circular —
each cites only GLOBAL_OPT/GLOBAL_OPT3, never each other's results — but they differ in what they
depend on:

| version | "drop the tail" step | depends on | hand-checkable? |
|---|---|---|---|
| ADVERSARY §6 | `R_S >= 1 - 2^n tau` + **Theorem B** | Theorem B (a **continuum** `min_{T^n} Re h`), Theorem E | partly |
| P1C Lemma T | polynomial identity + `|u-v| <= 2` + Hadamard + AM-GM | Theorem E only | **yes** |
| P1B Prop B1 | identical to Lemma T, derived independently | Theorem E only | **yes** |

**Recommendation (and a correction of provenance):** quote **Prop B1 / Lemma T**. ADVERSARY's route
drags the continuum quantity of Theorem B into a finite-`p` theorem for no reason (see L3-b).

**Naming clash that must be fixed:** P1A's "Lemma T" (a tail *staircase* price) and P1C's "Lemma T"
(the tail-robust Key Lemma) are different statements with the same name.

### 6.2 My own re-derivation of Proposition B1 / Lemma T

Normalise `A = 1`. Let `B` be the `n+1` largest atoms and `E = sum_{l not in B} x_l`. If
`theta := 1 - 2^n E <= 0` then `sum_l x_l^r >= sum_{l not in B} x_l^r >= E^r` and the claim is the
trivial branch. Otherwise, for `l0 in B` and a bijection `sigma` from the `n` slots onto `B\{l0}`,
substitute `u_j = v_{sigma(j),j}` into `sum_l c_l prod_j(u_j - v_lj) = A prod_j u_j`: the other core
atoms die, so

    1 <= x_{l0} prod_j d^{(j)}_{sigma(j),l0} + sum_{l not in B} x_l prod_j |u_j - v_lj|
      <= x_{l0} prod_j d^{(j)}_{sigma(j),l0} + 2^n E,

i.e. `x_{l0} prod_j d^{(j)}_{sigma(j),l0} >= theta > 0`. **This forces every `d^{(j)}_{m,l0} > 0`**
(every pair `(j,m)` occurs in some bijection and the product must be positive) — again an unstated
but necessary non-degeneracy step. Geometric mean over the `n!` bijections, product over the `n+1`
choices of `l0`, Hadamard `V_j <= (n+1)^{(n+1)/2}` and AM-GM give
`sum_{l in B} x_l^r >= (n+1)^{1-r} theta^r`; adding `sum_{l not in B} x_l^r >= E^r` (subadditivity)
gives **B1**. I re-derived every step; it is correct.

Branch 2 is Theorem E (link 4) plus the single extra constraint `sum_{l>n+1} x_l >= delta`, giving
`L_r(n, delta)`.

### 6.3 The covering argument, and my own recomputation of both thresholds

`eps0(n,r) := sup{t : phi >= target on [0,t]}` with `phi(E) = (n+1)^{1-r}(1-2^nE)_+^r + E^r`;
`eps1(n,r) := inf{delta : L_r(n,delta) >= target}`. Since `L_r(n, .)` is non-decreasing, branch 1
covers `[0, eps0]` and branch 2 covers `[eps1, inf)`, so **`eps1 <= eps0` proves P1 for every `N`**.
The logic is sound.

`verify_p1_twobranch.py` recomputes both from scratch (40-digit `mpmath` for `eps0`, my own exact
vertex enumeration of the concave programme for `L_r`):

| cell | target `(n+1)^{1-r}` | `eps0` (mine) | `eps0` (P1B/P1C) | `eps1` (mine) | `eps1` (P1B) | overlap | verdict |
|---|---|---|---|---|---|---|---|
| `nu=3, q=2` | 1.442249570 | 1.697663e-02 | 1.698e-02 / 0.016977 | 2.682700e-05 | 2.683e-05 | **632.8x** | P1 PROVED |
| `nu=4, q=2` | 2.000000000 | 1.469238e-02 | 1.469e-02 / 0.014692 | 9.958286e-04 | 9.958e-04 | **14.75x** | P1 PROVED |
| `nu=5, q=2` | 2.626527804 | 8.441542e-03 | 8.442e-03 / 0.008442 | 3.850495e-03 | 3.850e-03 | **2.19x** | P1 PROVED |
| `nu=6, q=2` | 3.301927249 | 4.437280e-03 | 4.437e-03 | 1.754303e-02 | 1.754e-02 | 0.25x | fails (as reported) |
| `nu=3, q=2.5` | 1.200936955 | 2.428819e-04 | 2.429e-04 | 1.322067e-04 | 1.322e-04 | 1.84x | P1 PROVED |
| `nu=4, q=2.5` | 1.681792831 | 3.373656e-03 | — | 3.238220e-03 | — | 1.04x | P1 PROVED (new here) |

**Every reported digit reproduces.** `verify_p1_stress.py` (a) confirms the bisection is legitimate:
`{E : phi(E) >= target}` is genuinely a prefix interval `[0, eps0]` on `[0, 2^{-n}]` (200 001-point
scan; the scan crossing agrees with the bisected `eps0` to 5 digits). (b) confirms P1B's "honest
scope" note: `eps1` is **identical** under both peeling conventions at `n = 2,3,4` (the `h = 1`
constraint is slack at the minimising vertex), and differs only at `n = 5`, where neither convention
closes — so the corrected `D_h` is *not* what closes `nu = 5`, exactly as P1B says. Also: since the
corrected constants make everything exact algebraic, **P1C's fp64 caveat about `S_4` in its `n = 4`
cell is removable and the `nu = 5` result is exact.**

### 6.4 ISSUE 6-a. The `N`-independence of `L_r` is asserted from numerics — and as stated it is FALSE

P1B §5 states "the value is independent of `N` for `N >= n+2` (checked to 9 decimals for
`N = n+2 .. n+6`)". This is used to justify computing `L_r` at one finite `N`. **It is false for
large `delta`.** My enumeration (`verify_p1_stress.py` (c), `N` up to `n+8`):

    nu=3, q=2, delta = 0.3:   L_r over N = 4..10:  1.7925618986, then 1.6804559585 from N = 5 on
    nu=4, q=2, delta = 0.3:   L_r over N = 5..11:  2.7386127875, then 2.4364916731 from N = 6 on

i.e. `N = n+2` is strictly too small once `delta > x_{n+1}`. Severity: **repairable, and harmless as
used** — but the write-up's justification is wrong and must be replaced. The repair is one line:

> **Repair.** `L_r(n, delta)` depends on the tail only through its total `delta` and the bound
> `x_l <= x_{n+1}`; the cheapest way to carry mass `delta` in atoms of size `<= s := x_{n+1}` uses
> `ceil(delta/s)` atoms (subadditivity of `t^r`), so `N = n+1+ceil(delta/s)` suffices. The peeling
> constraint at `h = n` gives `x_{n+1} + delta >= D_n(n) = 2^{-n}`, so for `delta <= 2^{-n-1}` we
> have `s >= 2^{-n-1} >= delta` and **`N = n+2` is exactly enough**. Since every `eps1` above is
> `< 2^{-n-1}`, and `L_r(n, .)` is non-decreasing in `delta` (as an infimum over `N` of
> non-decreasing functions), the covering argument is unaffected.

`verify_p1_stress.py` (c)/(d) confirm both halves: `L_r` is `N`-independent from `N = n+2` for
`delta <= 0.1`, and the cheapest tail of mass `E` with atoms `<= s` uses `ceil(E/s)` atoms.

### 6.5 ISSUE 6-b. The equality case of P1 is never stated, but the claim needs it

The claim says "with equality at the share-weighted block". Theorem D + P1 give only an inequality;
the equality case for `N > n+1` appears in none of the write-ups. It **does** follow, and the paper
should say so:

> For `E in (0, eps0]`, branch 1 is *strict* (`phi(E) > target` on the open interval); for
> `E > eps0 >= eps1`, branch 2 gives `L_r(n,E) >= L_r(n,eps0) > target` (e.g. `1.4949 > 1.4422` at
> `nu=3, q=2`, `E = 0.02`). So equality in P1 forces `E = 0`, i.e. at most `n+1` non-zero atoms;
> Theorem A forbids `<= n`; so exactly `n+1`, and the Key Lemma's equality case (Hadamard + AM-GM)
> forces the common-phase block. Equality in (T) then additionally forces the `q >= 2` budget step
> and Parseval to be tight (one balanced single-frequency neuron per atom), L2 to be tight (every
> `ghat(k)` real and non-negative), and `(|G_k|)` to be a maximiser of `Gamma_{nu/q}(p)`.

Severity: **repairable** (the argument is three lines), but it is currently a hole between "P1" and
"hence `K*` is exactly the feature count".

### 6.6 Fresh assault on P1 itself

`verify_p1_assault.py` builds exact (KL) systems from scratch (moment matrix, least-squares solve,
null-space minimisation of `sum|c_l|^r` by SLSQP/Nelder-Mead) over six adversarial phase families —
uniform random, block-on-the-core-plus-free-tail, the exact sign rule (`2^n` tensor grid), `m`-th
root tensor grids, coalescing atoms at separations down to `1e-6`, and perturbed sign rules —
sweeping `N` from `n+1` to `2^n+3`. A structural point the earlier passes gloss over and this sweep
makes concrete: **(KL) is `2^n` complex equations, so for generic phases no solution exists until
`N >= 2^n`**; at `n >= 3` the range `n+2 <= N <= 2^n - 1` is empty for generic phases, and the real
content of P1 begins at `N = 2^n`.

Result (`FINAL_RUNS/VERIFY/p1_assault.{log,json}`): **the minimum of
`sum_l |c_l|^r / (n+1)^{1-r}` is exactly `1.0000000000` and never below, in every cell**:

    nu=3, q=2:  N = 3..7   -> 1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 1.0740700222
    nu=4, q=2:  N = 4..11  -> 1.0000000000, 1.0000000030, ..., 1.0000000187, 1.4361006420
    nu=5, q=2:  N = 5..19  -> 1.0000000000, 1.0000001014, ..., 1.0000018297, 1.6849282075

The infimum is attained (by the block, `N = n+1`), and it is the target — consistent with
ADVERSARY §3's finding that GLOBAL_OPT4's `1.0000009` was optimiser noise, and reached here from a
completely independent code path.

**VERDICT for link 6: VERIFIED as an argument** (both branches re-derived, both thresholds
reproduced to every reported digit, the covering logic is sound), with two repairable gaps (6-a, 6-b)
and one provenance correction (6.1).

### 6.7 A NEW, FULLY RIGOROUS certificate for P1 at `q = 2`, `nu = 3, 4, 5` (`verify_p1_rigorous.py`)

The inherited certificates each fall short of "rigorous at the paper's cells":

* **P1A** — genuine mpmath-interval branch-and-bound, but it **stops at `n = 3`** (its `kappa(d)`
  raises for `d >= 4`), so `nu = 5` is **not** covered by any interval certificate, and it is
  *conditional on Theorem B*, a continuum statement (L3-b).
* **P1C** — cell-wise guaranteed but **float64**, by its own §3.6.
* **P1B / ADVERSARY** — float64 minimisation / grid.

So as the material stands, the paper **cannot** say "certified at `nu = 3,4,5`". I closed that gap.
Two replacements:

**(i) Branch 1 in closed form — no computer at all.** `phi(E) := tgt (1-2^nE)^r + E^r >= tgt` holds
on `[0, E_a]` with

    c := (tgt * r * 2^n)^{-1/(1-r)},        E_a := c / (1 + 2^n c),     tgt = (n+1)^{1-r}.

*Proof.* `phi(E) >= tgt` is `E^r >= tgt[1 - (1-2^nE)^r]`. The derivative of `u -> 1-(1-u)^r` is
`r(1-u)^{r-1}`, increasing on `[0,1)`, so by the mean value theorem `1-(1-u)^r <= u r(1-u)^{r-1}`.
With `u = 2^n E` it suffices that `E^r >= tgt r 2^n E (1-2^nE)^{r-1}`, i.e. (as `r-1 < 0`)
`E/(1-2^nE) <= c`, i.e. `E <= E_a`. ∎ Three lines, hand-checkable.

    n = 2: c = 9/512 = 0.017578125 exactly,  E_a = 0.016423357664...   (bisected eps0 = 0.016977)
    n = 3: c = 1/64  = 0.015625 exactly,     E_a = 0.013888888...      (bisected eps0 = 0.014692)
    n = 4: c = 0.0090656032,                 E_a = 0.0079172141        (bisected eps0 = 0.0084415)

`E_a` recovers 94–97% of the numerically bisected `eps0`, which is far more than the covering
argument needs.

**(ii) Branch 2 in exact arithmetic.** Every `D_h(n)` is replaced by a **rational lower bound**
(weakening a `>=` constraint only lowers `L_r`, so the result stays a valid lower bound), every
vertex of the polyhedron is found by **exact `Fraction` Gaussian elimination**, and `sum_l x_l^r` is
evaluated in **mpmath interval arithmetic** at 60 digits. Using `N = n+2 .. n+4` (justified by the
repair in 6.4, since every `delta` below is far under `D_n(n) = 2^{-n}`):

| cell | rational `delta` used | `delta <= E_a` | certified `L_r(n,delta) >=` | target | margin | verdict |
|---|---|---|---|---|---|---|
| `nu=3, q=2` | 16094889/10^9 | yes | 1.4889339659838589647 | 1.4422495703074084 | **1.0323692** | **P1 CERTIFIED for every N** |
| `nu=4, q=2` | 13611110/10^9 | yes | 2.0667173919014296679 | 2 | **1.0333587** | **P1 CERTIFIED for every N** |
| `nu=5, q=2` | 7758868/10^9 | yes | 2.6527750952866688827 | 2.6265278044037675 | **1.0099932** | **P1 CERTIFIED for every N** |

(The `nu=3` and `nu=4` margins agree with P1C's float64 `1.032129` / `1.034127`; the `nu=5` margin
`1.009993` agrees with P1C's `1.010684`.) **This makes P1 at `q = 2`, `nu = 3,4,5` a certified
theorem depending only on: Proposition B1 / Lemma T (elementary), Theorem E with `kappa_1 = 1/2`,
`kappa_2 = 3 sqrt3/8`, `kappa_3 = 49/72` (exact, and only these three under P1B's corrected `D_h`),
and a finite exact-rational vertex enumeration.** No continuum quantity, no Theorem B, no float64.

### 6.8 Sub-referee findings on P1A's certificate (delegated audit, cross-checked)

A second referee independently re-ran and re-implemented `p1a_cert.py` / `p1a_certcheck.py`
(scripts `verify_p1a_recert.py`, `verify_p1a_cover.py`, `verify_p1a_links.py`, `verify_p1a_mono.py`;
outputs in `FINAL_RUNS/VERIFY/`). Summary, which I have cross-checked against my own numbers:

* **The certificate stands.** Independently re-certified with the exponent carried as an *exact
  rational interval* rather than a float: `n = 2` in 125 boxes, `n = 3` in 81 boxes; the `r`-sweep
  reproduces P1A's brackets exactly (`r*(2) in [0.84, 0.86)`, `r*(3) in [0.60, 2/3)`). Box leaves
  tile the rectangle; the four regions (P-a) `beta >= 1/(n+1)`, (P-b) large `tau`, (P-c) small
  `tau`, (D) the branch-and-bound cover `[0,inf)^2` with no silent exclusion; `tau -> 0`,
  `beta -> 0`, `beta -> inf`, `tau -> inf` all handled. All seven asserted monotonicities verified.
  No circularity: the citation graph is a DAG and P1A is a leaf.
* **SCOPE GAP (repairable, matters):** P1A covers `n = 2, 3` only. At `nu = 5` the inherited P1
  rests on P1C (float64) and P1B alone. `p1a_cert.py --cells 4:2` raises
  `ValueError("only n <= 3 is covered by exact kappa values")`. **This is exactly the gap 6.7
  closes**, and the repair the sub-referee proposes (P1B's corrected `D_1(4) = 49/72`) is the one
  used there.
* **S1 (repairable, no effect on the verdict):** `float()` is applied to interval *lower* endpoints
  at four sites in `p1a_cert.py` (`inner_lower`, `tail_staircase`, `G_lower`'s drop term,
  `run_cell`'s `tau_hi`). `float()` rounds to nearest and can round **up**, so a "rigorous lower
  bound" can be inflated by one ulp and `tau_hi`'s coverage can fall `~1e-17` short. Repair:
  `iv.mpf([x,x])`. Margins are `2e-2`, so nothing changes.
* **S2 (cosmetic):** the exponent is the float `q/(n+1.0)`, so what is certified is the inequality
  at `fl(2/3)`, not at `2/3`, with no continuity argument. Fixed in the sub-referee's re-run.
* **S3/S4 (cosmetic):** `certifies()` swallows every exception as a clean "fails"; `p1a_certcheck.py`
  CHECK 1 exercises the `staircase=True` variant while the headline 101/63-box run uses
  `staircase=False`; "max stack" is mislabelled as branch-and-bound depth.
* **Under-advertised dependency (cosmetic but must be named):** Theorem E's **top rung `h = n`**
  enters only as a box discard, yet it carries 16.4% of the rectangle at `n = 2` and 6.2% at
  `n = 3`, and **at `n = 3` the certificate fails without it.**
* The sub-referee's own stress tests (5400 (KL) systems per cell, `N = 2^n .. 2^n+12`, zero-modulus
  padding to `N = 60`, shared phase vectors, coalescing to `1e-6`): Theorem E holds with
  `min_h [sum_{l>h}x_l]/D_h = 1.000000000`; the drop step's slack is `>= +4.7e-07`; the assembled
  chain's `min sum|c|^r/G = 1.000000000`; P1 itself never below `1.0039`. **No violation anywhere.**

---

## LINK 5 (continued) — a second, stronger attack on the `(H)` class

`verify_ascent_H.py` and the inherited `gopt3_chain.py` V4 both explore `(H)` through **gauge
symmetrisation**, i.e. through the image of the map `theta -> average of theta over the n+1 gauge
phases`. **That map is not surjective onto `(H)`** — an `(H)` network need not be anybody's gauge
average — so all the inherited numerical evidence for Theorem D covers only a proper subset of the
hypothesis class. I therefore ran a second search over the *whole* class
(`verify_ascent_H2.py`): raw weights, no structure, with the junk
`sum_{k,T} |J_{k,T}|^2 / sum_k |G_k|^2` driven to zero by a penalty annealed from `1e1` to `1e9`,
the block excluded from every start, starts including junk-maximal one-neuron-per-frequency,
three-frequencies-per-neuron impure, and clustered-support families; only runs whose final relative
junk is `< 1e-6` are counted, and the ratio is the exact brute-force margin of the final weights.

| `p` | `n` | `q` | width | final relative junk | best `gamma_q / [(n+1)^{-(nu-q)/q} B Gamma]` |
|---|---|---|---|---|---|
| 7 | 2 | 2 | 6 | 9.8e-07 | 0.5485200234 |
| 11 | 2 | 2 | 9 | 9.7e-10 | **0.9999979120** |
| 13 | 2 | 2 | 9 | 3.4e-07 | 0.4218345726 |
| 11 | 2 | 2.5 | 9 | 1.5e-07 | 0.8242882822 |
| 5 | 3 | 2 | 8 | 2.7e-13 | **0.9999989694** |

Nothing above 1. (The low cells are optimiser failures under the hard junk penalty, not evidence.)
The `nu = 4` cell matters most: there the *proved* Theorem D allows `1.030841`, and the search still
stops at `0.9999989694`, so the `F = 1` claim is being tested with real headroom and survives.
Together with the gauge-symmetrised ascent this is the first evidence for Theorem D / the claim that
is not confined to the gauge-average sub-class.

---

## CROSS-CUTTING FINDINGS

### X1. Constant and convention mismatches between the write-ups (must be resolved before submission)

| quantity | GLOBAL_OPT3 | P1B | my recomputation | comment |
|---|---|---|---|---|
| `D_h(n)` definition | max over `sum d_i = n` | max over `sum d_i <= n` | P1B is right | changes `D_1(n)` for `n >= 4` only |
| `D_1(4)` | 0.678904577 (`= kappa_4`) | **0.680555556** (`= 49/72`) | 0.680555556 | GLOBAL_OPT3's value is valid but weaker |
| `D_1(5)` | 0.666728299 | **0.680555556** | 0.680555556 | idem |
| `F(nu=6, q=2)` | 1.201816 | (implied 1.200369) | **1.200369** | direct consequence of the above |
| `L_r(5, 0)` | 3.105666856 | 3.106914 | 3.106914154 | idem |
| `eps0 / R1 / eps*` | — | 1.698e-02 / 1.469e-02 / 8.442e-03 / 4.437e-03 | 1.697663e-02 / 1.469238e-02 / 8.441542e-03 / 4.437280e-03 | agree; ADVERSARY's `eps*` and P1C's `R1` are the same number and agree |
| `r*(2)` | — | 0.84333 (P1B), `~0.83` (ADVERSARY), `>= 0.79688` (P1C), `[0.84, 0.86)` (P1A) | — | mutually consistent; the spread is rigour, not disagreement |
| "Lemma T" | — | — | — | **name clash**: P1A's Lemma T (tail staircase) vs P1C's Lemma T (tail-robust Key Lemma) |

### X2. What is computer-assisted, and what a referee must re-run

**Nothing in the chain needs a numerically-evaluated constant any more.** With P1B's corrected `D_h`,
only `kappa_1 = 1/2`, `kappa_2 = 3 sqrt3/8`, `kappa_3 = 49/72` ever appear, all exact, and their
derivation (the diagonal reduction of `S_d`) is a proof, not a computation.

Hand-checkable with no machine: L1, L2, L3, the per-neuron AM-GM and Parseval budget, the `B`
identity, the Key Lemma, Lemma T / Proposition B1, Lemma C, Theorem E's test-polynomial
construction, the `S_d` diagonal reduction, branch 1's closed-form `E_a` (§6.7), and
"concave minimum at a vertex".

**Computer-assisted, and exactly what must be re-run:**

1. **`L_r(n, delta) >= (n+1)^{1-r}` at one rational `delta <= E_a`** — a finite exact-rational
   vertex enumeration (at most `C(N+n+2, N) <= 924` linear solves for `n <= 4`, `N <= n+4`) plus an
   interval evaluation of `sum_l x_l^r`. `uv run python heracles/verify_p1_rigorous.py`, ~2 min,
   needs only `mpmath`. This is now the **only** computer-assisted step in the `(H)`-class theorem.
2. **`Gamma_s(p)`** — a nonconvex max-min. Everything about `K*` is downstream of a *global*
   optimum of this programme, and no certificate for it exists anywhere in the package. See X3.
3. (Optional, for the wider `r`-map rather than `q = 2`.) P1A's branch-and-bound
   (`uv run --with mpmath python heracles/p1a_cert.py --cells 2:2 3:2`, 0.3 s) and the sub-referee's
   independent re-implementation (`verify_p1a_recert.py`, 41 s).

### X3. The last clause of the claim ("hence `K*` is exactly the feature count") is the weakest part

The theorem pins a **value**. To pin a **count** you need, in order:

1. the equality case of P1 for `N > n+1` — **not stated anywhere**; the three-line argument is in
   §6.5 above;
2. the equality cases of L2 (every `ghat(k)` real and non-negative), of the per-neuron AM-GM and
   Parseval (one balanced single-frequency neuron per atom), and of `min_D sum x_k g_k(D) <=
   Gamma_s ||x||_r` (the clean spectrum is a maximiser of `Gamma_s`);
3. **uniqueness of the support size of the `Gamma_s` maximiser** — otherwise "the feature count" is
   not well defined;
4. **`K*` as the paper computes it must equal that support size.** `kstar.py` computes
   `argmax_K M_K(p) K^{-s}`, the **equal-amplitude** restriction. `PROOF_AUDIT` F2 already found that
   the free-amplitude programme is 2–10% larger and that the two argmaxes *differ in some cells*
   (`p=47, s=2`: 4 vs 3; `p=23/31, s=1.25`: 7 vs 6), while adjacent ladder rungs at the paper's
   primes differ by only 1–3%. **This is unresolved and is a correctness issue for the count
   statement, not a presentational one.** `verify_gamma.py` re-measures both.

### X4. Scope statements the paper must carry

* **Full-table training.** L1 fails at training fraction `< 1` (PROOF_AUDIT F6). Every heracles /
  nudial / S5 / Z cohort at `frac 0.3–0.8` is outside the theorem.
* **The `(H)` class.** `(H)` is `(p-1)(2^n-1)/2` complex conditions; it is a hypothesis, not a
  derived property of max-margin networks. P3A records that the registered sufficient condition for
  removing it (the chord criterion) is **refuted at `nu = 4`** and knife-edge at `nu = 3`, and P3B
  proves a duality barrier. The claim's "in that class" is doing real work and must stay.
* **`nu <= 5`, `q = 2`.** P1 fails at `nu = 6` (`eps1 = 1.754e-02 > eps0 = 4.437e-03`, confirmed
  independently here) and at `q >= 2.5` for `nu >= 4`, so the theorem is `F = 1` only in the stated
  window; outside it, Theorem D's `F > 1` is what is available.
* **Wei et al. Thm 4.1** connects this to training only for the **global minimiser** as `lambda -> 0`.
  Nothing here says gradient descent finds it.

---

## REPRODUCTION

All scripts are new and self-contained; run from `heracles/` with `uv run python`. Local CPU,
single core, no pods.

| script | what it establishes | time |
|---|---|---|
| `verify_constants.py` | `S_d` (diagonal + full-torus), `kappa_d`, both `D_h` conventions, the exact algebraic checks | ~5 min |
| `verify_p1_twobranch.py` | `eps0`, `eps1`, `L_r(n,delta)` by my own exact vertex enumeration, `F(n,q)` | ~2 min |
| `verify_p1_stress.py` | shape of `phi`; `eps1` under both conventions; `N`-dependence of `L_r`; the tail mechanism | ~10 min |
| **`verify_p1_rigorous.py`** | **the certified P1 at `q = 2`, `nu = 3,4,5`: closed-form branch 1 + exact-rational vertex enumeration + interval evaluation** | ~2 min |
| `verify_p1_assault.py` | direct search for a (KL) counterexample to P1, `N` up to `2^n+3` | ~20 min |
| `verify_network.py` | the Fourier dictionary, L1+L2, the budget, the `B` identity, `Gamma_s` solver | ~3 min |
| `verify_ascent_H.py` | margin ascent inside `(H)` via gauge symmetrisation, block excluded | ~8 min |
| `verify_ascent_H2.py` | margin ascent over the FULL `(H)` class by junk penalty, block excluded | ~15 min |
| `verify_gamma.py`, `verify_gamma2.py` | `Gamma_s(p)`, its maximiser's support size, vs `kstar.py`'s ladder `K*` | ~30 min |
| (sub-referee) `verify_p1a_recert.py`, `verify_p1a_cover.py`, `verify_p1a_links.py`, `verify_p1a_mono.py` | independent re-implementation and coverage/monotonicity audit of P1A's interval certificate | ~10 min |
| (sub-referee) `verify_eq_*.py` | the equality case and the share-weighted block (§7) | — |

Outputs: `heracles/FINAL_RUNS/VERIFY/*.json`, `*.log`.

---

## ISSUE LIST (by severity)

### FATAL
**None found.** No inequality in the chain is wrong, no step is circular, and no numerical attack
of mine (or of the sub-referees') beat the claim.

### REPAIRABLE (each with the repair)

| id | issue | repair |
|---|---|---|
| **R1** | **Provenance / scope of Theorem B (link 3).** Theorem B bounds `R = min over the CONTINUOUS torus of Re h`, while the margin is a min over the grid `mu_p^n`. It therefore cannot deliver a finite-`p` statement without Theorem C's `1 + O(p^{-2})` bracket (`1.42` at `n=2, p=11`, `19.6` at `n=4, p=11`), and ADVERSARY §6's version of P1 inherits that continuum quantity. | Route the `(H)`-class theorem through **Theorem D (link 5) + Prop B1 / Lemma T (link 6)** only; these are finite-`p` throughout. Cite Theorems A/B/C for the single-frequency, junk-allowed sub-result and nothing else. |
| **R2** | **`N`-independence of `L_r` is asserted from numerics and is FALSE as stated** (P1B §5: "independent of `N` for `N >= n+2`"). I exhibit `L_r(2, 0.3)` dropping from `1.79256` at `N = 4` to `1.68046` from `N = 5`. | Replace by: the tail's cheapest form uses `ceil(delta/x_{n+1})` atoms, and `(PEEL_n)` gives `x_{n+1} >= 2^{-n} - delta`, so for `delta <= 2^{-n-1}` (true of every `eps1` and every `delta` used) **`N = n+2` is exactly enough**; monotonicity in `delta` does the rest. |
| **R3** | **The equality case of P1 for `N > n+1` is nowhere stated**, yet the claim's "with equality at the block" and the whole `K*` conclusion need it. | Three lines (§6.5): branch 1 is strict on `(0, eps0]` and branch 2 gives `L_r(n,E) >= L_r(n,eps0) > target` beyond, so equality forces `E = 0`, hence exactly `n+1` atoms (Theorem A), hence the Key Lemma equality case. |
| **R4** | **No interval certificate for P1 at `nu = 5`.** P1A's branch-and-bound stops at `n = 3` (`p1a_cert.py --cells 4:2` raises); P1C's `n = 4` cell is float64 and inherits a numerically-evaluated `S_4`. | `verify_p1_rigorous.py` (§6.7): closed-form branch 1 + exact-rational vertex enumeration + mpmath interval evaluation certifies `nu = 3, 4, 5` at `q = 2` with margins `1.0324 / 1.0334 / 1.0100`, using only `kappa_1, kappa_2, kappa_3`. |
| **R5** | **`float()` applied to interval lower endpoints** at four sites in `p1a_cert.py` (rounds to nearest, so can round *up* by one ulp; `tau_hi` coverage can fall `~1e-17` short). | `iv.mpf([x,x])`. No effect on the verdict (margins `2e-2`). |
| **R6** | **P1A certifies the inequality at the float `q/(n+1.0)`, not at the rational `r`**, with no continuity argument. | Carry `r` as an exact rational interval (done in the sub-referee's re-implementation). |
| **R7** | **`K*` as the paper computes it is not the theorem's `K*`.** `kstar.py` maximises the EQUAL-amplitude `M_K K^{-s}`; the theorem's equality case pins the support size of the FREE-amplitude `Gamma_s` maximiser. Measured (`verify_gamma2.py`): at `p = 11, nu = 3, q = 2` the free support size is **2** while the ladder `K*` is **5**. | Define `K*` as the support size of the `Gamma_s` maximiser, compute it, and present `M_K K^{-s}` as its equal-amplitude approximation with the measured discrepancy (this is `PROOF_AUDIT` F2, still unactioned). |

### COSMETIC (but each must be fixed in the paper)

* **C1** Key Lemma / Lemma T: the non-degeneracy step ("every `d^{(j)}_{m,l0} > 0`, so all `c_l != 0`
  and the division is legal") is used without being stated.
* **C2** The Key Lemma's equality case "the same rotation pattern across slots" is proved only at
  `n = 2` (LEMMA_ATTEMPT's rigidity argument); at general `n` it is asserted.
* **C3** `GLOBAL_OPT3`'s `D_h(n)` uses `sum d_i = n`; P1B's `sum d_i <= n` is correct and strictly
  better for `n >= 4`. Consequences: `D_1(4)`, `D_1(5)`, `D_1(6)` change; `F(nu=6,q=2)` becomes
  `1.200369` not `1.201816`; and **GLOBAL_OPT3's rigour caveat about numerically-evaluated `S_n` for
  `n >= 4` disappears entirely** (only `kappa_1, kappa_2, kappa_3` ever appear).
* **C4** Name clash: P1A's "Lemma T" (tail staircase) vs P1C's "Lemma T" (tail-robust Key Lemma).
* **C5** Theorem E's **top rung `h = n`** is a separate, unadvertised dependency of the certificates
  (it carries 16.4% of P1A's rectangle at `n = 2`, 6.2% at `n = 3`, and the `n = 3` certificate
  **fails without it**). Name it in the theorem statement.
* **C6** `p1a_cert.py`'s `certifies()` swallows every exception as a clean "fails"; `p1a_certcheck.py`
  CHECK 1 exercises a different code path from the headline run; "max stack" is mislabelled.
* **C7** "Full-table training" must appear in the theorem statement (L1 is false otherwise).
* **C8** The gauge-symmetrisation parametrisation used by `gopt3_chain.py` V4 for all the inherited
  Theorem-D numerics is **not surjective onto `(H)`**; say so, or cite the full-class search
  (`verify_ascent_H2.py`) instead.

---

## X3 (continued) — measured: `Gamma_s`'s maximiser vs `kstar.py`'s ladder `K*`

`verify_gamma2.py`. `Gamma_K` (the best value on supports of size exactly `K`) is **non-decreasing**
in `K`, because a size-`K` optimum may put zero weight on a frequency; so "the maximiser's support
size" is the **smallest** `K` attaining `Gamma_s`, not `argmax_K`. (My first pass reported the
argmax and was confounded by that; the corrected statistic is below.) `K*(ladder)` is `kstar.py`'s
`argmax_K M_K(p) K^{-s}`.

| `p` | `nu` | `q` | `Gamma_s` | support size of the maximiser | `K*` (equal-amplitude ladder) | optimal shares |
|---|---|---|---|---|---|---|
| 11 | 3 | 2 | 0.535725926 | **2** | **5** | 0.569455, 0.430545 |
| 11 | 4 | 2 | 0.380195988 | 2 | 2 | 0.552239, 0.447761 |
| 11 | 5 | 2 | 0.269428382 | 2 | 2 | 0.541846, 0.458154 |
| 23 | 3 | 2 | 0.402933388 | 4 | 4 | 0.285663, 0.281496, 0.280976, 0.151866 |
| 23 | 4 | 2 | 0.221009424 | 3 | 3 | 0.384463, 0.329375, 0.286163 |
| 23 | 5 | 2 | 0.150640096 | 2 | 2 | 0.504839, 0.495161 |
| 31 | 3 | 2 | 0.370093692 | 4 | 4 | 0.269851, 0.255517, 0.254425, 0.220207 |
| 31 | 4 | 2 | 0.197402197 | 3 | 3 | (not recorded) |
| 31 | 5 | 2 | 0.114146400 | 3 | 3 | (not recorded) |

(`p = 11, 23` and `p = 31, nu = 3` from `gamma2.log`; the `p = 31, nu = 4, 5` rows from the
`verify_gamma.py` sweep in `gamma_small.log`, whose per-`K` free values give the smallest `K`
attaining the maximum — `K = 3` and `K = 3` — but do not record the shares there. That run was
stopped once the answer was unambiguous.)

Readings.

* `Gamma_{3/2}(23) = 0.402933388` — **my value agrees with ADVERSARY's to all nine reported digits**,
  from an independent solver, and with the sub-referee's third implementation
  (`0.4029333881190096`, support `{1,2,7,10}`).
* The two notions of `K*` **agree at `p = 23` and `p = 31` in all six cells** and **disagree at
  `p = 11`, `nu = 3` (2 vs 5)**. `p = 11` is the known-degenerate cell (`PROOF_AUDIT` F2 noted that
  `Gamma_free` there sits on a 2-set while the ladder argmax is 5 = dense). This does not refute the
  theorem, but it does mean **the paper's `K*` and the theorem's `K*` are different objects that
  happen to agree in most cells**, and the paper must say which one it is asserting.
* **The optimal shares are genuinely unequal in every cell** (spread 15%–47%). This is the whole
  content of ADVERSARY's code correction: the equal-amplitude `build_chain` cannot attain (T) —
  only the share-weighted block can.
* Caveat, and it is the sharpest one in the package: **`Gamma_s` is a nonconvex max-min and every
  number in this table is a computer-assisted global optimisation with no certificate.** Three
  independent implementations agree, which is reassuring but is not a proof.

---

## LINK 7 — the equality case: does the share-weighted block attain (T)?

Delegated to a second referee with instructions to build everything from scratch (own block
builder, own brute-force margin over all `p^n` rows and all wrong classes, own `Gamma_s` solver,
`mpmath` where float64 is not enough) and never to call the authors' functions. I cross-checked its
`Gamma_s` values against mine: identical.

**Result (`FINAL_RUNS/VERIFY/eq_block.json`, `eq_block_extra.json`).** With the `(n+1)`-neuron
common-phase block placed on the `Gamma_s` maximiser and frequency `k`'s neurons scaled by
`b_k^{1/q}`:

(In this table "share spread" is `max b_k − min b_k` over the searched support, which may include a
zero share; in the X3 table above it is the relative spread `(max − min)/max` over the *non-zero*
shares. Both say the same thing: the optimal shares are unequal.)

| cell | `Gamma_s` | share spread | **`gamma_q` / RHS of (T), share-weighted** | `gamma_q` / RHS, equal-amplitude | max same-`|k|` junk |
|---|---|---|---|---|---|
| `p=11, n=2, q=2` | 0.535725926 | 0.5695 | **0.9999999999999964** | 0.6773851716 | 7.6e-17 |
| `p=13, n=2, q=2` | 0.497772067 | 0.0918 | **0.9999999999999946** | 0.8872957240 | 1.4e-16 |
| `p=23, n=2, q=2` | 0.402933388 | 0.1338 | **0.9999999999999906** | 0.8620139790 | 7.6e-17 |
| `p=11, n=3, q=2` | 0.380195988 | 0.1045 | **0.9999999999999909** | 0.8555201433 | 3.9e-17 |
| `p=11, n=2, q=2.5` | 0.797257630 | 2.5e-16 | 0.9999999999999980 | 0.9999999999999986 | 3.1e-17 |
| `p=13, n=2, q=2.5` | 0.757062712 | 0.0 | 0.9999999999999966 | (equal shares: same) | 6.9e-17 |

**Verdicts.**

* **The equality case is VERIFIED**: the share-weighted block attains (T) to 15–16 digits in every
  cell, and is exactly junk-free (`~1e-17`), so it does satisfy `(H)`.
* **ADVERSARY's code correction is REAL.** `build_chain` with equal amplitudes reaches only
  `0.677` of the target at `p = 11` (reproduced exactly), and `0.86–0.89` at `p = 13, 23`. The
  deficit is entirely the share weighting: it vanishes precisely in the cells where the `Gamma_s`
  optimum happens to have equal shares (`q = 2.5` at `p = 11, 13`, spread `0` to `2.5e-16`), which
  is why `LEMMA_ATTEMPT`'s "check B: ratio 1.0000000000 in all 18 cells" did not catch it —
  **those checks were not against `Gamma_s`**, and `THEORY_nsum` §8's formula is stated correctly as
  "programme value of `(S, x)`", i.e. of the block's own amplitudes, not of the programme's optimum.
  Any place in the write-ups that reads the §8 constant as "attains `B Gamma_s` times
  `(n+1)^{-(nu-q)/q}`" is wrong unless the shares are equal. **This must be fixed in the paper's
  construction section.**
* `Gamma_{3/2}(23) = 0.402933388` as ADVERSARY reports; three independent solvers agree
  (`0.4029333881190096`, support `{1, 2, 7, 10}`, shares `0.1519, 0.2857, 0.2810, 0.2815`).

**Three further results from that pass, all of which I regard as strengthening the claim.**

1. **The equality is exact, not numerical.** Re-evaluated in `mpmath` at 36 digits
   (`eq_mp.json`): at `p = 11, n = 2, q = 2`, `gamma_q = 0.00115370859748209288436514314867464524`
   and the RHS of (T) is the *same* number, `|ratio − 1| = 2.5e-50`. Same at `p = 13`. So the block
   attains (T) identically, not to within optimiser tolerance.
2. **Local escape fails.** Perturbing away from the share-weighted block and re-optimising
   (`eq_escape.json`) never improves on it: `best_after_escape = block_ratio` to the last digit at
   `p = 11, 13, 17` (`n = 2`) and `p = 11` (`n = 3`).
3. **The equality case is rigid in the way the Key Lemma predicts** (`eq_phase_variants.json`): the
   documented phases `pi l/(n+1)` give junk `7.6e-17` and ratio `0.9999999999999964`, and so does any
   global rotation (a genuine gauge freedom); but `pi l^2/(n+1)` leaves junk `0.093` and ratio
   `−1.37`, and keeping the documented phases while making the moduli within a block **unequal**
   leaves junk `0.047` and ratio `0.095`. Equal moduli and roots-of-unity phases are both necessary —
   exactly the Hadamard + AM-GM equality case.
   **Correction to an earlier draft of this section (mine):** I first wrote that the variant
   `2 pi l/(n+1)` "also works". That is true only at `p = 11, n = 2`, i.e. when `n+1` is **odd**. The
   sub-referee's wider sweep shows it **fails at `nu = 4`** (junk `7.6e-2`, ratio `−4.72`). The
   correct statement, which the paper must use, is that the phase set has to be **a rotation of the
   `(n+1)`-st roots of unity**, not merely `n+1` equally spaced phases. (Issue **C9** below.)
4. A `q`-sweep at `p = 11` (`eq_qsweep.json`) shows the share-weighted block attaining (T) at
   `q = 2, 2.125, 2.25, ...` with ratio `0.999999999999998` throughout, while the equal-amplitude
   block ranges from `0.859` to `1.000` depending on whether the programme's optimum happens to be
   flat at that `q` — further confirmation that the deficit is entirely the share weighting.

---

## WHAT TO DO BEFORE 2026-09-25, IN PRIORITY ORDER

1. **Restate the theorem as Theorem D + P1 only** (R1). Drop Theorem B from the `(H)`-class chain;
   it is a continuum statement and contributes nothing there. Keep Theorems A/B/C as the
   single-frequency-with-junk result, clearly separated.
2. **Quote the certified P1.** Use `verify_p1_rigorous.py`'s statement: at `q = 2`, `nu = 3,4,5`,
   P1 holds for every `N`, by (i) the closed-form branch-1 threshold `E_a = c/(1+2^n c)`,
   `c = (tgt r 2^n)^{-1/(1-r)}` — three lines, no computer — and (ii) one exact-rational vertex
   enumeration of a polyhedron with at most `n+4` variables, evaluated in interval arithmetic.
   This is the only computer-assisted step left in the `(H)`-class theorem, and it is small enough
   to put in an appendix.
3. **Add the three missing lines** (R3): the equality case of P1, hence of (T).
4. **Fix `N`-independence** (R2) with the `ceil(delta/x_{n+1})` + `(PEEL_n)` argument.
5. **Decide what `K*` means** (R7) and say so. If it is the `Gamma_s` maximiser's support size, then
   `kstar.py`'s ladder is an approximation to it that differs at `p = 11, nu = 3`; if it is the
   ladder argmax, then the theorem does not pin it. Either is defensible; silence is not.
6. **Fix the construction section** (link 7): the attaining network is the **share-weighted** block,
   `b_k^{1/q}` per frequency; the equal-amplitude `build_chain` reaches only `0.68–0.89` of the
   target whenever the programme's shares are unequal.
7. **Adopt P1B's `sum d_i <= n`** (C3), which makes every constant in the package exact algebraic and
   removes GLOBAL_OPT3's own rigour caveat. Update `D_1(4), D_1(5), D_1(6)` and `F(nu=6, q=2)`.
8. Bookkeeping: name the `h = n` rung as a hypothesis of Theorem E (C5); fix the Lemma-T name clash
   (C4); add the non-degeneracy sentence (C1); scope to full-table training (C7); state that the
   gauge-symmetrised numerics do not cover all of `(H)` (C8), or cite `verify_ascent_H2.py`.
9. Do **not** claim the Key Lemma's "same rotation pattern across slots" equality case for general
   `n` (C2) — it is proved at `n = 2` only, and nothing needs it.
10. State the block's phase rule as "**a rotation of the `(n+1)`-st roots of unity**" (C9);
    `2 pi l/(n+1)` is wrong at even `nu`. Fix `Gamma_{3/2}(23)` in `GLOBAL_OPT4` §6.2 (C10), and
    re-label `lemma_verify.py` check B, which does not test what its caption says (E1).
11. Quote the equality case as the **algebraic identity** of "LINK 7 (final)", not as a numerical
    check: the two sides of (T) coincide identically for the share-weighted block on *any* support
    and *any* shares, so the only content is the optimality of `b`. That is a one-paragraph proof
    and it is strictly better than the "ratio = 1.0000000000" tables currently in the write-ups.

---

## LINK 7 (continued) — how large the share-weighting effect is, and a last adversarial ascent

**How large is the share-weighting effect, across primes** (`eq_equalamp.log`): the best the
*equal-amplitude* block can do, maximised over ALL supports, against the target `(T)`:

| cell | `Gamma_s` | equal-amp on the true optimal support | best equal-amp over every support |
|---|---|---|---|
| `p=11, nu=3` | 0.535725926 | 0.858637997 | 0.918258631 (on `S = {1,2,3,4,5}`) |
| `p=13, nu=3` | 0.497772067 | 0.887295724 | 0.931449236 (on `S = {2,3}`) |
| `p=17, nu=3` | 0.448340560 | 0.813622540 | 0.897538579 |
| `p=19, nu=3` | 0.442148593 | 0.950834946 | 0.950834946 |
| `p=23, nu=3` | 0.402933388 | 0.862013979 | 0.928704486 |
| `p=11, nu=4` | 0.380195988 | 0.855520143 | 0.855520143 |
| `p=13, nu=4` | 0.327849643 | 1.000000000 | 1.000000000 (shares happen to be equal) |

So the equal-amplitude construction loses **5–19%** of the target in every cell but the degenerate
ones, and the `0.677` figure ADVERSARY quotes at `p = 11` is the value when a zero-share frequency
is included in the equal-amplitude support. The share-weighted block loses nothing.

**A fresh adversarial ascent over UNRESTRICTED networks** (`verify_eq_ascent.py`, block excluded from
every start, exact brute-force margin recomputed for every reported number, start families
`gauss` / `impure` / `sparsecos` / `clustered`). Note this is over *all* gated chains, with junk
allowed — i.e. a test of (T) itself, not only of the `(H)` class:

| `p` | width `m` | best `gamma_2` / RHS of (T) | winning start family |
|---|---|---|---|
| 11 | 3 | 0.2963198984 | impure |
| 11 | 6 | 0.9999990698 | gauss |
| 11 | 9 | 0.9999991420 | impure |
| 11 | 12 | **0.9999993346** | gauss |
| 13 | 3 | 0.2301124498 | impure |
| 13 | 6 | 0.5620762663 | gauss |
| 13 | 9 | 0.9314491320 | sparsecos |
| 13 | 12 | 0.9999992292 | clustered |

Maximum over all cells `0.9999993346`. **(T) not broken.** (Widths below `(n+1)K*` cannot hold the
optimal support, which is why `m = 3` and `m = 6` fall short; that is a capacity effect, not
evidence.)

---

## PROVENANCE NOTE ON THE DELEGATED WORK

Two parts of this audit were delegated to sub-referees with instructions to build everything from
scratch and never call the authors' functions:

* **P1A's interval certificate** (§6.8). Its narrative report is reproduced there; its scripts are
  `verify_p1a_recert.py`, `verify_p1a_cover.py`, `verify_p1a_links.py`, `verify_p1a_mono.py` and its
  outputs `FINAL_RUNS/VERIFY/p1a_*.{json,log}`. **Disclosure:** re-running
  `p1a_cert.py --cells 2:2 3:2` rewrote `FINAL_RUNS/P1A/cert.json`; the script writes it
  unconditionally and the run is deterministic, so the content is unchanged and only the mtime moved.
* **The equality case** (§7). Its scripts are `verify_eq_block.py`, `verify_eq_gamma.py`,
  `verify_eq_mp.py`, `verify_eq_identity.py`, `verify_eq_escape.py`, `verify_eq_ascent.py` and its
  outputs `FINAL_RUNS/VERIFY/eq_*.{json,log}`. **Every number quoted in §7 above was read by me
  directly from those output files and cross-checked against my own independent implementation**
  (`verify_gamma2.py`, `verify_network.py`): `Gamma_{3/2}(11) = 0.535725926` on support `{1,3}` with
  shares `0.4305 / 0.5695`, and `Gamma_{3/2}(23) = 0.4029333881190096` on support `{1,2,7,10}`, agree
  to every digit between the two implementations.

No existing file in the repository was edited by this audit, nothing was committed, and no
background job of mine remains running.

---

## LINK 7 (final) — the equality is ALGEBRA, not a numerical coincidence

The delegated referee's full report arrived after the rest of this document was written. Its central
contribution upgrades §7 from "verified to 15 digits" to **proved**, and I have checked the
derivation by hand:

> For the `(n+1)`-neuron common-phase block on any support `S` with any share vector `b`
> (`r_k = b_k^{1/q}`), each of the block's `n+1` vectors at frequency `k` is `r_k cos(k· + phase)`
> with squared norm `r_k^2 p/2`, so `||theta_i||^2 = (n+1) r_k^2 p/2` and
> `P_q = (n+1)((n+1)p/2)^{q/2} sum_k b_k = (n+1)((n+1)p/2)^{q/2}`.
> The junk-free lemma gives `z(a,c) = sum_k (n+1) 2^{-n} r_k^{n+1} cos(2 pi k (s(a)-c)/p)`, so
> `margin = (n+1) 2^{-n} val(S,b)` with `val(S,b) = min_{D != 0} sum_k b_k^{nu/q} g_k(D)`. Hence
> `gamma_q = 2^{-n} (n+1)^{1-nu/q} ((n+1)p/2)^{-nu/2} val` while the RHS of (T) is
> `(n+1)^{1-nu/q} · 2 (2p(n+1))^{-nu/2} · val`, and the two prefactors differ by
> `2^{-n} 2^{nu/2} / (2 · 2^{-nu/2}) = 2^{nu-n-1} = 2^0 = 1`.

So **equality in (T) holds identically, for every `(p, n, q, S, b)`, iff `val(S,b) = Gamma_{nu/q}(p)`**
— i.e. iff the shares maximise the programme. Nothing about the block is approximate; the only
content is the optimality of `b`. This also re-confirms `LEMMA_ATTEMPT` §2 Step 5's constant identity
`2^{-n}(2/((n+1)p))^{nu/2} = 2(2p(n+1))^{-nu/2}` (checked to `4.6e-41` in mpmath, `n = 1..6`,
`p = 5..113`).

**Extra numerical confirmation from that pass**, beyond what §7 already reports:

* `Gamma_{3/2}(23) = 0.402933388119009641338311795593` at 50 digits, support `{1,2,7,10}`, obtained
  by **exhaustive enumeration of all `2^{nf}-1` supports** (187 symmetry classes under `k -> ck`)
  plus multistart epigraph SLSQP plus a 50-digit KKT Newton polish. Equality verified at 50 digits
  in four cells with `|ratio − 1|` of `2.5e-50`, `3.6e-50`, `8.0e-50`, `9.4e-50`; extended to 7 more
  cells and a 27-cell `q`-sweep over `q in [2, nu]`, every ratio in
  `[0.9999999999999909, 0.9999999999999986]`.
* **Full-DFT junk check** (every non-clean mode of the `(n+1)`-dimensional transform, not only the
  same-`|k|` ones of `(H)`): max `|junk|` is `7.6e-17 / 1.4e-16 / 7.6e-17 / 3.9e-17 / 1.1e-17`
  against max clean `0.161 / 0.088 / 0.057 / 0.076`. The block is junk-free in the *strong* sense,
  not merely `(H)`.
* **The programme's optimum always sits at a kink**: the number of active offsets `D` (up to
  `D <-> p-D`) equals `K` exactly in every cell, with all KKT mixing weights strictly positive
  (`p = 23`: `0.3781, 0.0560, 0.2128, 0.3532`). **This does not affect the construction** — the
  block never references `D`; its margin is automatically the `min_D` of the clean expression. So
  the "is the share-weighted block well defined at a kink?" worry is answered: yes.
* **Interiority is not needed**: a zero share simply contributes cost-free, content-free neurons.
* **The maximising support is unique up to `k -> ck`** in all six cells checked, so the support size
  is unambiguous; and the per-`K` ceilings are not ties
  (`p=23`: `0.092 / 0.748 / 0.949 / 1.000 / 0.964` of `Gamma*` at `K = 1..5`).
* **Local escape from the block, including deliberately impurity-injecting and junk-injecting
  perturbations** (240 perturb-and-reascend runs, 5 cells, 3 perturbation kinds x 4 magnitudes
  `1e-4 .. 0.4` x 4 seeds): **max = the block's own value, 0 violations.** This is the load-bearing
  adversarial test, and it probes exactly the direction ADVERSARY §5 said an escape would live in.

### Two further constant errors found (add to X1)

| id | where | issue | severity |
|---|---|---|---|
| **C9** | `THEORY_nsum` §8 and any restatement | the phase rule must be stated as "**a rotation of the `(n+1)`-st roots of unity**". `phi_l = 2 pi l/(n+1)` works only when `n+1` is odd and **fails at `nu = 4`** (junk `7.6e-2`, ratio `−4.72`). | cosmetic, but it is a correctness statement about the construction |
| **C10** | `GLOBAL_OPT4` §6.2 | `Gamma_{3/2}(23)` is quoted as `0.402931495`; the converged value is `0.402933388119009641` (relative deficit `4.7e-6`). The support quoted there, `{1,5,8,11}`, **is** correct — it is the image of `{1,2,7,10}` under `k -> 11k`. | cosmetic |

### Correction to my own §7 attribution of ADVERSARY's numbers

I wrote above that ADVERSARY's `0.677` is "the value when a zero-share frequency is included in the
equal-amplitude support". The sub-referee pins this down exactly: `0.677385172` (`p = 11`) and
`0.797165386` (`p = 13`) are the equal-amplitude ratios on **size-`K*+1`** supports `{1,2,3}` and
`{1,2,3,5}` — an artefact of a `Gamma` solver returning the optimum with one share driven to zero
and the block then being built on the padded support. On the **true optimal support** the
equal-amplitude ratios are `0.858638` and `0.887296`. **So ADVERSARY's headline figure overstates the
bug by about 2x.** The bug itself is real and unchanged: equal amplitudes lose 5–19% whenever the
optimal shares are unequal, and only the share-weighted block attains (T).

Finally, the sub-referee confirms my reading of **E1/`lemma_verify.py` check B**: its `val` is the
programme value *at equal shares on a hand-picked support*, never `Gamma_s(p)`, so its
"ratio 1.0000000000 in all 18 cells" verifies the identity above and not attainment — the label is
misleading, but **`THEORY_nsum` §8 itself is not wrong**, since its identity already carries free
amplitudes. The bug is confined to `build_chain` and to check B's label.
