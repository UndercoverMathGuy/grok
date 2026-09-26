# 2026-09-23: attempt at "(T) for all ν, p, q"

## 1. What the request can mean
The request was to prove that the ansatz support `K*` is the global margin maximizer for all `ν`, `p`, `q`. Taken literally, this is false
in three places, independently of any proof:

| region | what actually happens | source |
|---|---|---|
| `q > ν` | no maximizer: splitting one neuron into `N` copies multiplies `P_q` by `N^{1−q/ν} → 0` at fixed logits, so `sup γ_q = ∞` | THEORY_nsum §5, PROOF_AUDIT F8 |
| `q < 2` | a neuron carrying several frequencies is cheaper (`x^{q/2}` concave across frequencies), so the unrestricted bound's argmax changes (`K*(p, ν/2)` for x^n); `Σ‖θ_i‖^q` is not a norm | THEORY_nsum §5 |
| isolated exponents | two supports of different size tie, e.g. `p=5`, `r_c = log 2/log(5/(2a)) ≈ 0.539` (so `ν=4`, `q≈2.16`): `K=1` and `K=2` are both optimal, and "the" count is undefined | astra ARGUMENT §2.4 |
| `x^n` activation | no attainment result: the best junk-free `x^n` block reaches 0.408 of the bound, the flat solution 0.51–0.57 | PROOF_AUDIT F1, HOLES H5 |

The meaningful target is (T) for the gated chain, full table, `2 ≤ q ≤ ν`, every odd prime `p`, plus rigidity (every maximizer has a
`Γ`-optimal support) away from the tie exponents.

## 2. New result: junk IS locally optimal somewhere, so local analysis cannot give (T)
Route tested: "every local maximizer of `γ_q` satisfies (H)". The global maximizer exists at finite width (astra §4), so this
route together with ELEGANT Theorem 2 would have given (T) for all `ν, p, q ∈ [2, ν]`.

Method (`lmax_junk.py`): true-margin ascent (soft-min, then SLSQP polish of `max t` s.t. every gap `≥ t`, `P_2 = 1`), a KKT residual via
NNLS on the active set, the anatomy of the logit tensor, then escape tests (random perturbations of size `ε`, re-polished).

**p = 5, n = 2, q = 2, width 9, seed 3:** KKT point (residual 4e-9, 56 active constraints) with ratio **0.837046** to the ansatz, with
cross-frequency share X = 0.135 and same-|k| junk share 0.086 (clean 0.649/0.130 on k = 1, 2). Every re-polish from perturbations
`ε = 1e-4 … 3e-2` returns to 0.837046; only `ε = 0.1` escapes to 1.0. The point is saved as `endpoint_p5_s3.npy`.
Because `q < ν`, adding a new neuron at scale `s` changes the margin by `O(s^ν)` and the cost by `Θ(s^q)`. So widening cannot improve it at
first order, and it is a local maximizer over all widths.

Consequence: the landscape of the unrestricted problem has impure, junk-carrying local maxima. Any proof of (T) must be
genuinely global. Classifying critical points or local maxima cannot give it, and neither can the astra local theorem plus an escape argument.

**p = 7 (10 seeds, width 12):** 9 of 10 endpoints are the ansatz (1.000000) or the junk-free support-{1,3}-type ansatz vertex
(0.980055, shares 0.765/0.235). Seed 9 is a junky local max (§5). No value above 1. See `lmax_p7.log`.

## 3. Why "for all p" is uncertain, not just unproved
Every surviving route for junk at `K ≥ 2` (P3C aiming densities + DEPTH-A) needs a lower bound on the aiming depth `A*(S)` (the
inverse one-sided Sidon constant of `2S`) for every support `S` whose programme value is competitive, `Γ(S) ≥ ν^{−(1−r)/r} Γ*`.

- Dense support: junk hides perfectly (Dirichlet profile, dip `1/(p−1)` of its mass) AND coverage is perfect. Only the `K^{1−s}` size
  penalty saves (T) there (densejunk.py: junky/junk-free ratio 0.90 at p = 809, heading to √3; `Γ*/Γ_dense ~ √(p/log p)`).
- Dissociate supports (`K ≲ log_3 p`): `A* ≥ 1/2`, junk cannot hide, and (T) holds on pure nets (HOLES_logic).
- The paper's supports have `K* ≈ 1.3 ln p`, so `3^{K*} ≈ p^{1.43} ≫ p`: they are not dissociate, and `A*` is certified only case by case
  (0.35–0.47, HOLES_aiming).
- Heuristic scaling: coverage loss at the optimum is `Θ(√(K log p))`. A junk profile with free phases on a sparse set can be flattened
  to a Spencer-type `O(√(K log(p/K)))` dip, which is the same order. So whether max-junk networks lose asymptotically depends on
  constants in a support-choice/discrepancy trade-off that nobody controls. The one quantitative mechanism in sight is that flat
  junk needs spikes, and two spikes at `a*, a**` force a bad coverage offset `D = 2(a* − a**)` when the phases are aligned. It does not hold
  in general: the dense Dirichlet case is flat and covers perfectly.

So (T) for every prime is a conjecture with strong small-`p` evidence (no violation found up to p = 31 unrestricted, p = 59 pure,
junk losing by 2.5× at p = 59), not a theorem with a known route.

## 4. What is PROVEN (inherited, unchanged)
- (T) under (H), every `ν`, every `q ∈ [2, ν]`, any width, impure allowed; attaining nets have exactly `K*` features × `ν` neurons.
- (T) unrestricted at `p = 3` (`n = 2, q = 2`; opus55/P3_THEOREM.md, computer-assisted).
- Pure networks, `n = 2, q = 2`, supports with aiming depth `≥ 1/2` (exact rational certificate, HOLES_logic), and depth 0.47 at
  `r = 2/3` (HOLES_aiming, p = 71 support closed).
- Unrestricted strict LOCAL optimality of every globally optimal ansatz, `n = 2, q = 2`, every odd prime (astra LOCAL_THEOREM).
- `q = ν`: two-sided (Morwani endpoint).

## 5. More junky local maxima (p = 7)
`lmax_p7.log`, seed 9: KKT point (residual 8e-9) at **0.750841**, X = 0.190, junk = 0.292, stable to the same escape tests.
So junk-carrying impure local maxima occur at p = 5 and p = 7.

## 6. Pure junk scan vs p (`ptrend.py`, n = 2, q = 2; exact profile identity; balanced cost)
Per frequency, N atoms with free complex amplitudes and slot phases; the optimizer also chooses the support. Ratio = normalized margin
/ ansatz, where the ansatz uses a multistart LOWER bound on Γ*, so every ratio is an over-estimate. N = 3 is the unrestricted pure
problem (control). Its shortfall below 1 measures optimizer bias.

| p | N=1 (max junk) | N=2 | N=3 control | N=2 / control | junk-harmless ceiling for dense N=2 = 1.2247·Γ_dense/Γ* |
|---|---|---|---|---|---|
| 11 | ≤ 0 | 0.476 | 0.986 | 0.48 | 1.125 |
| 23 | ≤ 0 | 0.522 | 0.944 | 0.55 | 0.962 |
| 31 | ≤ 0 | 0.542 | 0.910 | 0.60 | 0.888 |
| 47 | 0.12 | 0.540 | 0.903 | 0.60 | 0.789 |
| 59 | 0.14 | 0.519 (K = 19–23, not dense) | 0.864 | 0.60 | 0.746 |
| 83 | 0.24 | 0.504 (2 seeds, 1500 steps) | 0.882 | 0.57 | 0.686 |

Junk-carrying optimizers always pick the dense support, where junk hides best (Dirichlet), but whose programme value decays like
`√(log p / p)` relative to Γ*. The max-junk dense ceiling `√3·Γ_dense/Γ*` exceeds 1 only for p ≈ 31–67, and there the measured
max-junk value is ≤ 0.12.

## 7. The local theorem looks extendable to every ν and every q ∈ [2, ν)
- Block Hessian (`block_nu_ext.py`, `block_nu_q.py`): on ker J, −H has exactly n zero modes (the phase motions) and every other
  eigenvalue ≥ 2, at ν = 3, 4, 5, 6 (q = 2) and at q = 2.5 and q → ν (ν = 3, 4, 5). The q-penalty adds `−(q−2)θθᵀ/‖θ‖²` per neuron,
  which is negative semidefinite, and within a block it is universal after scaling by stationarity.
- New neurons for q > 2: the penalty Hessian at 0 vanishes, but new neurons enter F with degree ν > q and never interact with other
  neurons in F, so `F_new − μP_new < 0` for small nonzero ones. The sequence argument goes through at higher order.
- Support counting for n operands (sketch): for an input mode u, the admissible readout frequencies are ±S if u has at most one nonzero
  coordinate; if it has two equal nonzero coordinates, a subset of u ± S; if it has two distinct ones, at most 3 values
  ((u_1+u_2)/2 and the u_j/2). So |R_u| ≤ 2K (K = 1 as in astra), and Tao's prime uncertainty applies unchanged.
- Still needed: an analytic proof of the block certificate for all ν (the Z_ν cyclic symmetry should block-diagonalize it), and
  a careful write-up of the n-operand support lemma.

Reading of §6 so far: the best two-atom junky ratio goes 0.476 → 0.522 → 0.542 → 0.540 → 0.519 over p = 11 … 59. It peaks near
p = 31–47 and then falls, tracking the falling dense ceiling. At p = 59 the optimizer leaves the dense support for K ≈ 19–23, and
still reaches only 0.52. The p = 59 Γ lower bound (0.309915) is 1% below the exact 0.313049 (opus55), so these ratios are
overstated by about 1%. No sign of junk approaching the ansatz.

## 8. astra's projection obstruction, taken one step further (2026-09-23 evening)
astra/PROJECTION_OBSTRUCTION.md: the cubic-phase dense family beats the cheapest clean representation of its OWN projection
(by up to ν^{(1−r)/r}, as p → ∞), but a clean net with a different spectrum beats the family. Its proposed target ("every junky net is
beaten by SOME clean net") is (T) itself, because by ELEGANT Thm 2 the best clean net is the ansatz.

**Own-share bound (Theorem A, proved: the existing unrestricted bound without the final max over shares).** For any network
(impure, any width, q ∈ [2, ν]), let U_k = Σ_i m_ik^{q/2} (m_ik = neuron i's Fourier mass at frequency k) and β_k = U_k/Σ U. Then
  γ_q(Z) ≤ B · V_s(β),   V_s(β) = min_{D≠0} Σ_k β_k^s (1 − cos 2πkD/p),   s = ν/q.
Steps: L1 → L2 → L3 + per-(i,k) AM-GM → Σ_i u_ik^s ≤ (Σ_i u_ik)^s (the only place junk-free nets lose ν^{1−s}) → per-neuron
superadditivity of x^{q/2}. The ansatz has value ν^{1−s}·B·Γ*, so Z can beat it only if **V_s(β_Z) > ν^{1−s}Γ\*** ("competitive shares").
Averaging over D gives the cruder form K_eff(β) := (Σβ^s)^{−1/(s−1)} < ν·(p/((p−1)Γ*))^{1/(s−1)}. The obstruction family attains
Theorem A asymptotically, so Theorem A is the sharp spectrum-blind statement. What is left is junk that hides on competitive shares.

**The two windows never overlap for the dense family** (`cubic_family.py`, true full-table margin, n = 2, q = 2; Γ* multistart
lower bound, so ratios to the ansatz are over-estimates; the dense-block closed form matches the true margin to 1e-15):

| p | cubic / own clean rep | cubic / ansatz | random-phase / ansatz | ceilA = best any uniform-share net can do / ansatz |
|---|---|---|---|---|
| 23 | −0.91 | −0.72 | −0.60 | 1.36 |
| 47 | −0.50 | −0.32 | −0.18 | 1.12 |
| 59 | −0.21 | −0.13 | −0.19 | 1.06 |
| 83 | −0.30 | −0.17 | −0.07 | 0.997 |
| 101 | +0.09 | +0.05 | +0.00 | 0.93 |
| 211 | +0.59 | +0.26 | +0.18 | 0.76 |
| 401 | +0.82 | +0.29 | +0.25 | 0.61 |

Junk on dense shares hides (positive margin) only from p ≈ 101. Theorem A already caps every dense-uniform-share net below the
ansatz from p ≈ 83. Below 83 the shares are competitive, but the junk is not hidden. With the true Weil bound (|S| ≤ 2√p), cubic hiding
needs about p ≳ 800, and the ceiling there is about 0.4. Reading: hiding by exponential-sum cancellation needs many frequencies,
and many frequencies means a non-competitive spectrum. Regime B (competitive shares) is where a proof still has to show that junk
cannot hide.

## 9. Adversarial test: junk placed on the best hiding structure, with clean frequencies covering for it (`adv_ap.py`, `run_adv.sh`, `adv_scan.log`)
Pure n = 2, q = 2. The frequencies of an AP {d, …, Ld} each get ONE atom: forced full junk, the cheapest possible, with chirp or random junk phases
(flat junk hides best on an AP). Every other frequency gets 3 atoms, starts as clean blocks on the Γ-optimal support, and may cover the
AP's bad offsets. The AP's share σ of the penalty budget is pinned; everything else is optimised (2 seeds × 1500 Adam steps, soft-min
margin). Ratio 1 = ansatz, Γ = multistart lower bound.

| p | (L, d) | σ = 0.02 | 0.08 | 0.2 | 0.4 |
|---|---|---|---|---|---|
| 31 | (5,1) / (5,5) | 0.88 / 0.93 | 0.79 / 0.84 | 0.61 / 0.65 | 0.33 / 0.31 |
| 31 | (9,1) / (9,5) | 0.91 / 0.94 | 0.84 / 0.85 | 0.68 / 0.68 | 0.40 / 0.40 |
| 59 | (7,1) / (7,7) | 0.92 / 0.89 | 0.82 / 0.79 | 0.63 / 0.61 | 0.33 / 0.29 |
| 59 | (13,1) / (13,7) | 0.89 / 0.86 | 0.80 / 0.80 | 0.66 / 0.65 | 0.39 / 0.38 |

(L = 13 at p = 31 is excluded because the AP swallows 13 of the 15 frequencies, so no cover is possible.) Unpinned, the optimiser drives the AP share to 0.5%.
In every configuration the ratio falls monotonically and steeply with the junk share, about −1.5 per unit of share near σ = 0. No sign of junk
paying, even when it is placed adversarially on its best hiding structure with free cover.

Proof status of the remaining lemma. With the junk-damage route of P3C (aiming depth A* ≥ A_cert), P3C F6-3 shows the aim must cover the
WHOLE support. A naive hybrid (aim at a core of large-share frequencies, bound the tail crudely as in Theorem A) needs the tail threshold
τ ≤ (Γ*/(2κ0))^{1/(s−1)} ≈ 0.006 at ν3q2, i.e. a core of up to about 170 frequencies. That is no restriction at paper primes. So the
additive-combinatorics lemma is not yet correctly posed: it must be a statement about whole supports carrying competitive shares,
including thin tails.
