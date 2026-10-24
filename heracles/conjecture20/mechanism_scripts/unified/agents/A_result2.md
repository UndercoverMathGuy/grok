# Two-sided line lemma: `sigma2 = (1/2) osc_w P / Sum u`

All work is in `/tmp/claude-0/-home-user-grok/512ae18c-6d12-5fa4-88c0-000434907593/scratchpad/unifiedA/two/`. Nothing under `/home/user/grok` was modified.

Definitions.
* `P(w) = Sum_l u_l cos(2 pi l w/p + th_l)` and `sigma2(S) = inf_{u >= 0, th} (1/2) osc_w P / Sum u`.
* Coverage `t` and `kappa = K_eff/ln p` are as in `result.md`, computed with the weights `u`.
* The application only needs `(1/2)osc >= c Sum u_l s_l(delta0)` with `s_l <= 1`, so `sigma2 >= c` is sufficient: `c12 = 0.1835` in the pair case, `0.289` for a single slot (Theorem A).

## 0. Verdict

1. **(Q1) Open, and nothing contradicts it.** No construction comes near the needed constants inside the window `kappa <= 2.5`.
   * Constructions force any valid `f(t,kappa)` to satisfy `f(t,kappa) <= (sqrt2 + o(1)) / sqrt((1-t) kappa ln kappa)` (Theorem 1′, rigorous).
   * So the Sudakov shape `c sqrt(t/kappa)` can hold at best up to a factor `sqrt(log kappa)`.
   * The one-sided Fejer counterexamples of `result.md` do not transfer: their oscillation is about `1/2` of `Sum u`, because they are spikes.
2. **(Q2) Getting below the needed constants takes `kappa` far above the window.**
   * Asymptotically (`p -> infinity`), with optimised two-sided-flat blocks and exact large-deviation rates, the `kappa` needed is:

     | target | t = 0.3 | t = 0.4 | t = 0.5 | t = 0.6 |
     |---|---|---|---|---|
     | `sigma2 <= 0.18` | 11.6 | 14.6 | 19.1 | 22.4 |
     | `sigma2 <= 0.289` | 5.35 | 6.9 | 8.9 | 10.4 |
     | `sigma2 <= 0.42` | 3.5 | 4.1 | 5.1 | 6.0 |

     These are smaller than the guessed 60, because optimised blocks reach `sigma_b ~ 0.91/sqrt(K_b)` rather than `C/sqrt(q)` with a large `C`.
   * The rigorous Rudin–Shapiro and Bennett version gives `kappa ~ 26` for `sigma2 <= 0.177` at `t = 0.3`.
   * The best explicit finite-p witness below 0.18 is at `p = 100003`: `t = 0.362`, `kappa = 8.34`, `sigma2 = 0.173`.
   * Inside the window (`kappa <= 2.5`, `p = 1009` and `10007`, coverage `t >= 0.3`), the smallest `sigma2` found is 0.36. Adversarial search over weights and phases, from glued, random and programme starts, never went lower.
   * On programme supports, `sigma2` (weights and phases free) is 0.357–0.551, which is 2.1–2.6 times the Parseval floor.
3. **(Q3) Partly.**
   * Relation mass below the independent baseline separates the classes again: every programme support has negative excess, every two-sided-flat glued witness has positive excess. At large `kappa` the excess is tiny (`+0.003`), so the separation is only by sign.
   * The tilt does not reach 0.18 at finite `p` and `kappa >= 1.8`.
   * New phase-uniform moment certificates do clear 0.18 on all programme supports at the weights tested, with values 0.20–0.35.
   * Under RM, the tilt clears 0.18 asymptotically for `kappa < 3.0`, and 0.289 only for `kappa < 1.95`.

## 1. Proved statements

### Theorem 1′ (glued Rudin–Shapiro blocks: upper envelope for any valid `f`)

Setup.
* `q = 2^m`, with `eps_j` the Rudin–Shapiro signs, so `|Sum_{j<=q} eps_j e^{ijx}|^2 <= 2q`.
* `p > 2q` is prime, `a_1..a_N in Z_p^*`, and `P(w) = Sum_b Sum_{j<=q} eps_j cos(2 pi j a_b w/p)`.

Assume no two of the `j a_b` coincide up to sign. For random multipliers this fails with probability at most `N^2 q^2/(p-1)`.

Claims:
* **(i) Flatness.** `(1/2)osc P <= N sqrt(2q)` and `Sum u = Nq`, so `sigma2 <= sqrt(2/q)` for every choice of multipliers.
* **(ii) Coverage.** `cov(delta)/Sum u = 1 - N^{-1} Sum_b D_q(2 pi a_b delta/p)/q`, where `D_q(x) = Sum_{j<=q} cos jx`.
  For random multipliers and fixed `delta`, the variables `y_b = D_q/q` are i.i.d. with `y_b <= 1`, `E y_b <= 0` and `E y_b^2 <= (p/(p-1))/(2q)`.
  Bennett's inequality then gives `Pr[t(delta) <= t] <= exp(-N (1-t)(ln(1 + 2q'(1-t)) - 1))`, with `q' = q(p-1)/p`.
* **(iii) Conclusion.** A union bound over `delta` and over collisions gives, for large `p`, a set with `t(u) >= t`, `K_eff = Nq` and `sigma2 <= sqrt(2/q)`. This holds whenever
  `kappa > q / ((1-t)(ln(1 + 2q(1-t)) - 1)) + o(1)`.
  Consequently `f(t,kappa) <= (sqrt2 + o(1)) / sqrt((1-t) kappa ln kappa)`.

Numbers at `t = 0.3`:

| q | `sigma2 <=` | `kappa` |
|---|---|---|
| 16 | 0.354 | 10.6 |
| 32 | 0.25 | 16.2 |
| 64 | 0.177 | 26.1 |

*Proof.*
* (i) is the Rudin–Shapiro identity, with `|Re z| <= |z|`.
* (ii) uses `Sum_{u in Z_p} D_q(2 pi u/p) = 0` and `Sum_{u in Z_p} D_q^2 = pq/2`, the latter because `q < p/2`. Removing `u = 0`, where `D_q = q`, gives the stated mean and second moment.
* For `y <= 1` the bound `e^{ly} <= 1 + ly + y^2 (e^l - 1 - l)` holds. Take `l = ln(1 + 2q'(1-t))`. ∎

Two flatness constants matter. Parseval/Bhatia–Davis gives the floor `sigma2 >= 1/sqrt(2 K_eff)` (Note, Lemma osc). Measured optimal blocks on `{1..q}` sit at about 1.29 times that floor (§2.1).

### Proposition 2 (phase-uniform two-sided certificates)

Notation.
* `Pi = Prod I_0(lam u_l)`.
* `P_0` is the zero-phase polynomial.
* `T_j` is the `j`-th moment of `Sum u_l cos theta_l` with i.i.d. uniform `theta_l`.
* `M_j = E_delta P_0(delta)^j`, the sum of the moduli of all terms with `Sum +- l = 0 (mod p)`.

For every phase vector:

**(a) Moments.**
* `|E_w P^j - T_j| <= M_j - T_j`.
* Hence `(1/2)osc >= min_c ( lower bound of E (P-c)^m )^{1/m}`, with each `E P^j` replaced by the endpoint of `[T_j - (M_j - T_j), T_j + (M_j - T_j)]` that is unfavourable for that term.
* `m = 2` is Bhatia–Davis. With no relations of length at most 4 one gets `(1/2)osc >= (3/8)^{1/4} ||u||_2`, and about `0.93/sqrt(K_eff)` for spread weights.

**(b) cosh-tilt.**
* Write `A = 2 Pi - E_delta cosh(lam P_0)` (even-length relations only) and `B = E_delta sinh(lam P_0)` (odd-length relations).
* If `A^2 - B^2 >= 1`, then `(1/2)osc >= lam^{-1} arccosh sqrt(A^2 - B^2)`.
* This always beats the one-sided log-tilt `lam^{-1} log(A - B)`.

**(c) One-sided certificates also bound `sigma2`.** These are the tilted mean and the Riesz certificate (Proposition 4 of `result.md`). Each bounds `max P` and `-min P` separately, both uniformly in the phases.

*Proof.*
* (a) Expand `E_w P^j` over signed sequences of frequencies. The terms that cancel formally in `Z^S` give `T_j`; all other terms have modulus at most their zero-phase value. Then use `|P - c| <= (1/2)osc`, where `c` is the midpoint of the range.
* (b) Jacobi–Anger gives `E cosh(lam P) >= A` and `|E sinh(lam P)| <= B`. Then `cosh(lam r) >= E cosh(lam(P - c)) >= min_y [A cosh y - B |sinh y|] = sqrt(A^2 - B^2)`. ∎

### Corollary 3 (under RM)

Assume `Z(lam) <= Pi(1 + gamma e^{E}/p)`. Then (b), with the concavity bounds of `result.md` Corollary 5, gives `sigma2 >= g(x) - O(1/K)` for every `x < x_kappa`. The tilted mean gives asymptotically `A(x_kappa)`:

| `kappa` | 1.3 | 1.6 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|---|
| `A(x_kappa)` | 0.45 | 0.36 | 0.28 | 0.22 | 0.18 |

So RM implies the pair case for `kappa < 3` asymptotically, and the single-slot case for `kappa < 1.95`.

## 2. Numerics (all witnesses recomputed by the independent `verify2.py` → `verify2.out`)

### 2.1 Two-sided-flat blocks on `{1..q}` (`block_opt.out`)

| quantity | value |
|---|---|
| optimised phases, equal weights | `sigma_b = 0.99/sqrt(q)` (q=4: .498, 8: .349, 16: .245, 32: .176, 64: .124) |
| free weights | `sigma_b ≈ 0.91/sqrt(K_b)` |
| Bhatia–Davis floor | `0.707/sqrt(K_b)` |

### 2.2 Asymptotic frontier (`frontier2.out`)

This is the table in the verdict. It uses glued blocks with random multipliers, union bounds over both `w` and `delta`, and exact Cramér rates of the block value and the block coverage. The optimal block is `q = 32` for target 0.18, `q = 8–10` for 0.289 and `q = 3–5` for 0.42.

Random singletons, i.e. random sets, give `sigma2 ≈ 1 - t*(kappa)`. Reaching 0.42 that way needs `kappa ≈ 5.4`, and 0.18 needs `kappa ≈ 30`.

### 2.3 Explicit witnesses

Glued optimised blocks (exact over `Z_p`):

| p | q, N | t | kappa | sigma2 |
|---|---|---|---|---|
| 10007 | q=16, N=4 | 0.551 | 6.95 | 0.237 |
| 100003 | q=16, N=5 | 0.506 | 6.95 | 0.238 |
| **100003** | **q=32, N=3** | **0.362** | **8.34** | **0.173** |
| 10007 | q=8, N=4 | 0.469 | 3.47 | 0.342 |

### 2.4 Inside the window (`scan2_*.out`)

Minimum `sigma2` over weights and phases, coverage `t >= 0.3`:

| p | `kappa0` | glued q=2 / glued q=4 / random starts | best (t) |
|---|---|---|---|
| 1009 | 1.5 | ≥ 0.50 | — |
| 1009 | 2.0 | 0.420–0.455 | — |
| 1009 | 2.5 | 0.361–0.366 | 0.361 (t ≈ 0.52) |
| 10007 | 2.0 | 0.45–0.47 | — |
| 10007 | 2.5 | 0.395–0.411 | 0.395 (t ≈ 0.41) |

The LP hunt (coverage maximised at fixed `(1/2)osc <= sigma0`) was infeasible at `sigma0 = 0.289` from every start at `kappa0 = 2.5`.

### 2.5 Programme supports (`prog2.out`, `run_cert2.out`, `certmin.out`)

`sigma2`, adversarial:

| p | K | `kappa` | `u = y` | `u` free |
|---|---|---|---|---|
| 101 | 7 | 1.44 | 0.526 | 0.451 |
| 401 | 8 | 1.27 | 0.576 | 0.551 |
| 1009 | 11 | 1.47 | 0.537 | 0.486 |
| 2003 | 12 | 1.54 | 0.541 | 0.523 |
| 2003 | 16 | 1.92 | 0.458 | 0.424 |
| 2003 | 20 | 2.49 | 0.382 | **0.357** |
| 10007 | 14 | 1.50 | 0.560 | 0.541 |
| 10007 | 18 | 1.84 | 0.494 | 0.478 |
| 10007 | 23 | 2.43 | 0.421 | 0.408 |
| 30011 | 17 | 1.60 | 0.552 | 0.531 |
| 30011 | 23 | 2.18 | 0.458 | 0.449 |

Best phase-uniform certificate, as the maximum of BD, `mom4/6/8`, cosh, tilted mean and Riesz:
* `u = y`: 0.339, 0.351, 0.308, 0.300, 0.239, 0.204, 0.295, 0.258, 0.205, 0.287, 0.225.
* `u = u*` (the adversary's weights): 0.198–0.347.
* The moment certificates are the strongest at these `p`; the tilts are the weakest, at 0.09–0.23.
* Minimised over `u`: at least 0.321 (p=101), 0.340 (p=401), 0.268 (p=1009) and 0.294 (p=2003, K=12). Larger supports were still running in `two/certmin.out` at the time of writing. This is a local search over `u`, so it is not a proof for all `u`.

### 2.6 Relation excess `eps/rho0` at `x = 0.25, 0.5, 1, 2`

| set | `kappa` | x=0.25 | x=0.5 | x=1 | x=2 |
|---|---|---|---|---|---|
| programme supports (`../rm_compare.out`) | all | −0.97 to −0.31 | <0 | <0 | <0 |
| glued, p=10007, q=4 | 2.6 | +2.55 | +0.15 | +2e-4 | — |
| glued, p=10007, q=8 | 3.5 | +1.46 | — | — | — |
| glued, p=10007, q=16 | 7.0 | +0.0065 | — | — | — |
| glued, p=100003, q=32 | 8.3 | +0.0046 | — | — | — |

Programme supports are negative at every tested `x`. The glued witnesses are positive at `x <= 0.5`.

## 3. Obstruction and remaining sub-claim

* **Two-sided flatness needs long internal relations.** Glued progressions supply them, at the price of coverage: about `q/ln q` coverage cost per unit of `ln p` for a block of size `q`.
* **The p-independence question is still open.** The natural route is a weighted Pisier–Bourgain Sidon theorem: every weight vector with coverage `t` has a proportional "quasi-independent" part in weight, with controlled leakage.
* **For the programme, the cleanest sub-claim** is the two-sided Sidon bound `sigma2 >= c12` on near-optimal supports at `kappa <= 2.5`.
  * Data supports it with a large margin: the minimum found is 0.357.
  * At finite `p` it is certified per support by moment certificates (Proposition 2a), which need only relation sums of length at most 8.
  * Asymptotically it would follow from RM (Corollary 3) for `kappa < 3`.
  * The single-slot constant 0.289 is likewise far from the data (minimum 0.357), but certificates under RM reach it only for `kappa < 1.95`.
