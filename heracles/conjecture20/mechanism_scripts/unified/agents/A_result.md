# The line lemma ("deterministic Sudakov minoration"): result of step 1

Scratch directory: `/tmp/claude-0/-home-user-grok/512ae18c-6d12-5fa4-88c0-000434907593/scratchpad/unifiedA/`.
Nothing under `/home/user/grok` was modified.

## 0. Verdict

* **As stated, the lemma is false at the constants the programme needs.** Its hypotheses are coverage `t` and `K_eff <= kappa ln p` only.
  A rigorous family, *glued Fejer blocks* (Theorem 1), has `t` bounded below, `K_eff = O(ln p)` and `-min P <= Sum v / q` exactly, at every prime.
  Explicit witnesses inside the stated window (`K_eff <= 2.5 ln p`) were checked by an independent script. Examples at `p = 1009`:
  * `sigma <= 0.166 < 0.18` with `t = 0.41`, `kappa = 2.10`;
  * `sigma <= 0.18` with `t = 0.451`, `kappa = 2.18`;
  * `sigma <= 0.42` with `t = 0.562`, `kappa = 1.735`.

  The same holds at `p` up to `100003`.
* **Neither conjectured shape is right.** For fixed `t < 1` the best possible `sigma(t,kappa)` is at most `(3/4+o(1))/((1-t) kappa ln kappa)` (Corollary 2).
  That is far below both `c sqrt(t/kappa)` and the large-deviation form `1 - t*(kappa) ~ kappa^{-1/2}`. Structured supports beat the random model.
* **Still open:** p-independence in the weak sense, `inf_p sigma*(t,kappa;p) > 0` for each fixed `(t,kappa)`. It is neither proved nor refuted.
  Every construction found gives `sigma` of order `1/(kappa log kappa)`.
* **What does hold (proved):**
  * a two-sided Bessel tilt: flatness is equivalent to a heavy lower tail of the coverage profile (Proposition 3);
  * the tilt that uses only minimum coverage is confined to an `O(1/K)` layer at a Laplace packing limit that no known set approaches. This explains the "conflict" noted in the Note.
  * a phase-uniform Riesz-product certificate with exact relation moduli (Proposition 4).
* **Where the obstruction sits (data):**
  * Programme supports are *super-random*: relation mass below the independent-phase baseline at every tested tilt.
  * Flat witnesses exceed that baseline at small tilt.
  * On programme supports, adversarial `sigma` is `0.42 - 0.54` with `v = y`, and `0.417 - 0.48` after optimising over `v` and phases, in every case tried.
  * So `0.18` has a large margin there, while `0.42` is at the edge. At `p = 601`, with `v = y`, adversarial `sigma = 0.4177`.

Notation.
* `V = Sum_l v_l`, `cov(delta) = Sum_l v_l kappa_l(delta)`, `t = min_{delta != 0} cov(delta)/V`.
* `P_psi(w) = Sum_l v_l cos(2 pi l w/p + psi_l)`, `sigma = -min_w P_psi / V`, `E_w` = average over `Z_p`.
* `sigma*(t,kappa;p) = inf { sigma : t(v) >= t, K_eff <= kappa ln p }`. The lemma asks for `inf_p sigma*(t,kappa;p) >= sigma(t,kappa) > 0`.
* `h(x) = x - log I_0(x)` (concave), `g(x) = log I_0(x)/x`, `A(x) = I_1(x)/I_0(x)` (increasing).

---------------------------------------------------------------------------------------------------

## 1. Proved results

### Theorem 1 (glued Fejer blocks)

Setup.
* `F_q(x) = Sum_{|j|<=q} (1-|j|/(q+1)) e^{ijx} = (q+1)^{-1} (sin((q+1)x/2)/sin(x/2))^2 >= 0` is the Fejer kernel.
* `q >= 2` and `p > 2q` is prime. Take multipliers `a_1,...,a_N in Z_p^*` and define
  `P(w) = Sum_{b=1}^N Sum_{j=1}^q (1 - j/(q+1)) cos(2 pi j a_b w / p)`.
* Write `P` in the lemma's normal form: reduce `j a_b` to `{1..(p-1)/2}`. The phase stays 0 because cosine is even. Merge equal frequencies by adding their weights.

Then:
1. `V = Nq/2` and `-min_w P <= N/2`. Hence `sigma <= 1/q`, for every choice of the `a_b`.
2. `K_eff <= N K_b(q)`, where `K_b(q) = 3q(q+1)/(2(2q+1))`.
3. `cov(delta)/V = N^{-1} Sum_b c_q(2 pi a_b delta/p)`, where `c_q(x) = (q+1-F_q(x))/q`, which lies in `[0, (q+1)/q]`.
4. Let `D_q(t) = D( 1 - tq/(q+1) || 1/(q+1) )`, where `D(a||pi) = a ln(a/pi) + (1-a) ln((1-a)/(1-pi))` is the Bernoulli divergence.
   If `N > ln((p-1)/2) / D_q(t)`, then some choice of multipliers gives coverage `t(v) >= t`.

*Proof.*

(1) For each block, `Sum_{j=1}^q (1-j/(q+1)) cos(j x) = (F_q(x)-1)/2 >= -1/2`. Also `Sum_{j=1}^q (1-j/(q+1)) = q/2`.
Merging adds nonnegative weights, so it changes neither `P` nor `V`.

(2) Before merging, `Sum v^2 = N Sum_{i=1}^q (i/(q+1))^2 = N q(2q+1)/(6(q+1))`. Merging can only increase `Sum v^2`, because `(a+b)^2 >= a^2+b^2`.
`V` is unchanged, so `K_eff <= (Nq/2)^2 / (N q(2q+1)/(6(q+1))) = N K_b(q)`.

(3) `kappa_l` depends only on `l` up to sign, so merging does not change `cov`.
Each block contributes `Sum_j (1-j/(q+1))(1-cos(j x_b)) = q/2 - (F_q(x_b)-1)/2 = (q/2) c_q(x_b)`.

(4) Fix `delta != 0` and draw the `a_b` independently and uniformly from `Z_p^*`.
* The `a_b delta` are i.i.d. uniform on `Z_p^*`.
* Put `y_b = F_q(x_b)/q`, which lies in `[0, M]` with `M = (q+1)/q`.
* `F_q` is a trigonometric polynomial of degree `q < p`, so `Sum_{u in Z_p} F_q(2 pi u/p) = p`. Hence `E y_b = (p-q-1)/((p-1)q) < 1/q`, so `pi_p := E y_b / M < 1/(q+1)`.
* By convexity, `E e^{s y} <= 1 - pi_p + pi_p e^{sM}`. The Chernoff bound then gives `Pr[ N^{-1} Sum c_q(x_b) <= t ] = Pr[ Sum y_b >= N(M-t) ] <= exp(-N D(1-t/M || pi_p)) <= exp(-N D_q(t))`.
  The last step holds because `D(a||.)` is decreasing on `(0,a)`.
* `cov(-delta) = cov(delta)`, so a union bound over `delta in {1..(p-1)/2}` gives an expected number of bad `delta` below `((p-1)/2) e^{-N D_q(t)} < 1`. ∎

### Corollary 2 (upper envelope for any valid lemma)

(a) For every `q >= 2`, `t in (0,1)` and prime `p > 2q`, some `(S,v,psi)` satisfies all of:
* `t(v) >= t`;
* `K_eff <= K_b(q)(1 + ln(p/2)/D_q(t))`;
* `sigma <= 1/q`.

Hence any p-uniform lemma must satisfy `sigma(t,kappa) <= 1/q` whenever `kappa > K_b(q)/D_q(t)`.

(b) For fixed `t in (0,1)`, as `kappa -> infinity`, `sigma(t,kappa) <= (3/4 + o(1)) / ((1-t) kappa ln kappa)`.
In particular, for every `c > 0` the bounds `sigma >= c sqrt(t/kappa)` and `sigma >= 1 - t*(kappa)` both fail once `kappa` is large. (Here `1 - t*(kappa) ~ kappa^{-1/2}`.)

(c) As `p -> infinity`, `D_q` can be replaced by the Cramer rate `I_q(t) = sup_s [ -s t - log E e^{-s c_q(U)} ]`, with `U` uniform on `T`.
This uses Chernoff with the exact moment generating function at each `p`, and `I_{q,p} -> I_q`. Table in §2.2.

*Proof of (b).*
* `D(a||pi) >= a ln(a/pi) - 1/e` and `a ln a >= -1/e`. With `a >= 1-t` this gives `D_q(t) >= (1-t) ln(q+1) - 2/e`. Also `K_b(q) <= 3(q+1)/4`.
* So `K_b/D_q <= phi(q) := 3(q+1)/(4[(1-t)ln(q+1) - 2/e])`.
* Take the largest `q` with `phi(q) < kappa`. Then `q+2 >= (4/3) kappa [(1-t) ln(q+2) - 2/e]`, so `q >= (4/3)(1-t) kappa ln kappa (1-o(1))`.
* Apply (a). ∎

### Remark (sharpness inside the class)

Restrict to glued progressions of length `q` whose multiplier set is dissociated, with no relation `j a_b = Sum +- a_b'`.
The Riesz product `Prod_b (1 - cos(2 pi a_b w/p + psi_{1,b}))` on the transversal `{a_b}` has mean 1. It picks up exactly `(1/2) Sum_b v_{1,b}`, which gives `sigma >= 1/(q+1)`.
So the Fejer phases are optimal in this class up to `1/q` versus `1/(q+1)`. Measured: adversarial phases improve the `q = 6` witnesses only from `0.1657` to `0.1640`.

Theorem 1 with `q = 2` has `N` relations `2 a_b - (2a_b) = 0` of length 3, while `t` stays bounded below. So **coverage controls no count of short relations**. In consequence, every moment-method or pair-correlation argument that relies on coverage alone is ruled out.

### Proposition 3 (two-sided Bessel tilt)

Let `Pi(lambda) = Prod_l I_0(lambda v_l)` and `E(lambda) = lambda V - log Pi(lambda)`, and let
`Z(lambda) = E_delta e^{lambda P_0(delta)} = e^{lambda V} E_delta e^{-lambda cov(delta)}`.
For every `lambda > 0` and every phase vector `psi`:

(a) `2 Pi(lambda) - Z(lambda) <= E_w e^{-lambda P_psi(w)} <= Z(lambda)`.

(b) (Laplace packing bound) `Pi <= Z`. With coverage `t`, this gives `e^{-E(lambda)} <= 1/p + e^{-lambda t V}` for all `lambda`.

(c) (Flat implies a heavy coverage tail) If `-min P_psi <= sigma V`, then for all `lambda`:
`E_delta e^{-lambda cov(delta)} >= 2 e^{-E(lambda)} - e^{-lambda(1-sigma)V}`.

(d) (Minimum-coverage tilt is a boundary statement)
* The tilt bound `-min P >= lambda^{-1}[log Pi + log(2 - r)]`, with `r = e^{E}(1/p + e^{-lambda t V})`, has `r >= 1` by (b). It never certifies more than `log Pi(lambda)/(lambda V)`.
* Equal weights, `K = |S|`, `p = e^{K/kappa}`, `x = lambda v`: the condition `r < 2` requires `h(x) < 1/kappa + ln2/K` and `t > 1 - g(x) - ln2/(xK)`. Applying (b) at the point where `e^{-K h} = 2 e^{-K/kappa}` gives `t <= 1 - g(x_kappa) + O(1/K)`, with `h(x_kappa) = 1/kappa`.
* So the minimum-coverage tilt applies only to sets whose coverage is within `O(1/K)` of the Laplace limit `L(kappa) = 1 - g(x_kappa)`, and then it certifies at most `1 - t + O(1/K)`.
* `L(kappa) = 0.76, 0.82, 0.86, 0.89` at `kappa = 1.3, 1.6, 2, 2.5`. Programme supports have `t = 0.42 - 0.69`, so the condition is vacuous in practice.

*Proof.*
* Jacobi–Anger gives `e^{-x cos th} = Sum_k (-1)^k I_k(x) e^{ik th}` with `I_{-k} = I_k >= 0`.
* Orthogonality on `Z_p` then gives `E_w e^{-lambda P_psi} = Sum_{k in H^perp} Prod_l (-1)^{k_l} I_{k_l}(lambda v_l) e^{i k_l psi_l}`, where `H^perp = { k : Sum k_l l = 0 mod p }`.
* The `k = 0` term is `Pi`. The sum of all moduli is `Z`, by the same identity at zero phase.
* This gives (a), and `Z >= Pi`.
* For (b), use `cov(delta) >= tV` for `delta != 0`.
* For (c), combine (a) with `E_w e^{-lambda P} <= e^{lambda sigma V}`.
* For (d), use the concavity of `h`, the monotonicity of `g`, and `x'' - x' = O(1/K)`. ∎

### Proposition 4 (Riesz certificate with exact moduli; phase-uniform)

For `eps in [0,1]^S` let `Q(delta) = Prod_l (1 + eps_l cos(2 pi l delta/p))`. If `E_delta Q < 2` and the numerator below is positive, then for every `psi`:

`-min_w P_psi >= ( Sum_l v_l eps_l - E_delta[ P_0(delta) Q(delta) ] ) / E_delta Q`.

*Proof.*
* Take `R(w) = Prod_l (1 - eps_l cos theta_l(w))`, which is `>= 0`, with `theta_l = 2 pi l w/p + psi_l`. Expand `F = -P_psi R` and `R` on `T^S`.
* `E_w F = Sum_{k in H^perp} F^(k) e^{ik.psi}`. The coefficient `F^(0)` equals `(1/2) Sum v_l eps_l`.
* `|F^(k)| <= G^(k)`, where `G = (Sum v cos th)(Prod (1 + eps cos th))` has nonnegative coefficients and `Sum_{k in H^perp} G^(k) = E_delta[P_0 Q]`.
* Hence `E_w F >= Sum v eps - E_delta[P_0 Q]`. Likewise `0 < 2 - E_delta Q <= E_w R <= E_delta Q`.
* Finally, `-min P >= E_w F / E_w R`. ∎

### Corollary 5 (the conditional statement that does hold; essentially the Note's (RM))

Suppose `Z(lambda) <= Pi(lambda)(1 + gamma e^{E(lambda)}/p)` (RM_gamma) holds at some `lambda`. Put `K = |S| <= kappa ln p` and `x = lambda V/K`. Then

`sigma >= g(x) + ln(1 - gamma e^{K(h(x) - 1/kappa)}) / (xK)`.

This is p-independent, and as `K -> infinity` it tends to `g(x)` for any `x < x_kappa`.
The tilted-mean version, `(2 Pi' - Z')/Z` with `Z'` bounded through the convexity of `Z` (as in the Note), tends to `A(x_kappa)` instead. That limit is not proved here in detail.
*Proof:* Proposition 3(a), plus Jensen. Here `log Pi >= K log I_0(x)` because `log I_0` is convex, and `E <= K h(x)` because `h` is concave. ∎

---------------------------------------------------------------------------------------------------

## 2. Numerical verification

`verify.py` recomputes everything from `witnesses.json` with code it does not share with the search code. It:
* reduces and merges the frequencies;
* recomputes `t` exactly over all `delta` and `sigma` by brute force over all `w`;
* re-evaluates the minimiser at 50-digit precision;
* checks `P >= -N/2` and `V = Nq/2` for the Fejer witnesses.

### 2.1 Explicit witnesses (output of `verify.out`)

| family | p | K | t | kappa | sigma |
|---|---|---|---|---|---|
| Fejer q=6 | 1009 | 18 | 0.4108 | 2.102 | 0.1657 |
| Fejer q=8 | 1009 | 24 | 0.4470 | 2.755 | 0.1244 |
| Fejer q=3 | 1009 | 18 | 0.5338 | 2.231 | 0.3158 |
| Fejer q=6 | 2003 | 18 | 0.3081 | 1.912 | 0.1660 |
| Fejer q=6 | 10007 | 24 | 0.3366 | 2.104 | 0.1655 |
| Fejer q=3 | 10007 | 27 | 0.5246 | 2.513 | 0.3163 |
| Fejer q=6 | 30011 | 36 | 0.4342 | 2.820 | 0.1639 |
| Fejer q=6 | 100003 | 36 | 0.3326 | 2.526 | 0.1655 |
| Fejer q=6 | 100003 | 42 | 0.3965 | 2.947 | 0.1647 |
| Fejer q=3 | 100003 | 36 | 0.4929 | 2.680 | 0.3085 |
| LP-improved (q=6) | 1009 | 17 | 0.4510 | 2.181 | 0.1800 |
| LP-improved (q=7) | 2003 | 19 | 0.4202 | 2.289 | 0.1800 |
| LP-improved (q=6) | 10007 | 23 | 0.3694 | 2.190 | 0.1800 |
| LP-improved (q=2) | 1009 | 13 | 0.5617 | 1.735 | 0.4200 |
| LP-improved (q=2) | 2003 | 15 | 0.5352 | 1.778 | 0.4200 |
| LP-improved (q=2) | 10007 | 18 | 0.4902 | 1.841 | 0.4200 |

"LP-improved" means the weights were obtained by alternating two steps:
* an LP in `v` that maximises coverage subject to `P_psi >= -sigma_0 V` (both constraints are linear in `v` for fixed `psi`);
* adversarial optimisation of the phases.

Comparison with the programme frontier at similar `kappa` and the same `p`:
* At flatness `0.18`, the witnesses reach about 61–69% of the frontier coverage.
* At flatness `0.42`, they reach about 86–95%. For example, at `p = 2003`, `kappa ~ 1.75`, the witness has `t = 0.535` against `0.58–0.60`.

### 2.2 Asymptotic (p -> infinity) envelope of the Fejer family (`tables.out`)

`kappa_LD(q,t) = K_b/I_q(t)` is the threshold above which `sigma <= 1/q` is achievable:

| q | 1/q | t=0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|---|
| 2 | .500 | 1.63 | 2.22 | 3.08 | 4.46 | 6.94 | 12.2 |
| 3 | .333 | 1.95 | 2.60 | 3.55 | 5.06 | 7.74 | 13.3 |
| 6 | .167 | 2.77 | 3.59 | 4.77 | 6.62 | 9.81 | 16.3 |
| 16 | .063 | 5.01 | 6.29 | 8.11 | 10.9 | 15.5 | 24.5 |
| 64 | .016 | 13.4 | 16.4 | 20.5 | 26.6 | 36.4 | 54.6 |

The rigorous Bernoulli version `K_b/D_q` of Theorem 1 is valid for every `p`. It is about 1.4–1.8 times larger; see `tables.out`.

For comparison, random equal-weight sets give `t*(kappa) = 0.34` with `1 - t* = 0.66` at `kappa = 2`.
The tilted-mean cap `A(x_kappa)` takes the values below:

| kappa | 1.3 | 1.6 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|---|
| `A(x_kappa)` | .45 | .36 | .28 | .22 | .18 |

### 2.3 Programme supports (`prog601.out`, `prog_frontier.out`, `riesz_cert.out`, `tilt_compare.out`, `rm_compare.out`)

* **Adversarial `sigma` with `v = y`:**

  | p | 101 | 211 | 401 | 601 | 701 | 809 | 1009 |
  |---|---|---|---|---|---|---|---|
  | `sigma` | .451 | .486 | .540 | **.4177** | .493 | .485 | .498 |

  The `p = 601` witness was verified independently: `t = 0.674`, `kappa = 1.82`.
* **`v` free, with a coverage floor (heuristic search):** the minimum `sigma` was 0.437 (p=1009), 0.431 (2003, K=14), 0.483 (10007, K=16) and 0.417 (10007, K=20).
  The LP hunt found no `v` with `sigma <= 0.30` on any programme support.
* **Relation excess over independent phases.** Write `eps/rho0` with `rho = Z/Pi - 1 = rho0 + eps` and `rho0 = e^E/p`.
  * Programme supports are negative at every tested tilt `x in {0.25, 0.5, 1, 2}`, ranging from `-0.97` to `~0` (super-random).
  * Flat Fejer witnesses are positive at small tilt: up to `+3.6` (p=2003) and `+6.9` (p=10007) at `x = 0.25`.
  * This quantity separates the two classes; `(t, kappa)` does not.
* **Phase-uniform certificates (exact relation data, `v = y`):**

  | kappa | 1.29 | 1.70 | 1.84 | 2.10 | 2.70 |
  |---|---|---|---|---|---|
  | Riesz (Prop. 4) | .339 | .209 | .186 | .148 | .105 |
  | Exact tilted mean | .263 | .178 | .162 | .132 | .097 |

  On the flat witnesses both are at most 0.15, as they must be.

---------------------------------------------------------------------------------------------------

## 3. Obstruction and cleanest remaining sub-claims

**Obstruction (precise).**
* Coverage is the minimum of the profile `delta -> cov(delta)`. Every tilt or moment argument needs its Laplace transform `E_delta e^{-lambda cov}`, equivalently the relation masses.
* Proposition 3(c) shows that flatness forces this transform up. Theorem 1 shows that minimum coverage plus `K_eff = O(ln p)` allow it to be large: glued short progressions with generic multipliers produce a coverage profile with exponentially many `delta` just above `t`, and such sets are flat.
* Packing (coverage) does not imply covering (a deep hole of depth `sigma V`) for cyclic codes on the torus.

**Remaining sub-claims, from cleanest to most useful.**
1. *(Qualitative, open.)* `inf_p sigma*(t,kappa;p) > 0` for all fixed `t > 0` and `kappa < infinity`. Conjecture: `sigma* ≍_t 1/(kappa log kappa)`.
   A further conjecture is that the asymptotic `sigma*` is attained by glued finite blocks: a single-letter formula `inf { sigma_B : K_B / I_B(t) <= kappa }` over finite frequency patterns `B`.
2. *(What the programme can use.)* Replace `(t, kappa)` by the super-random relation hypothesis `RM_gamma` with `gamma <= 1` at tilts `x <= x_kappa`.
   * It holds with room on every programme support tested, and fails on the flat witnesses.
   * Under it, Corollary 5 gives `sigma >= g(x_kappa) - O(1/K)`, and the tilted-mean refinement gives `A(x_kappa) = 0.28` at `kappa = 2` and `0.22` at `2.5`. That is enough for `0.18` asymptotically when `kappa < 3`.
   * At `p <= 10^4` the exact certificates give only 0.10–0.34, because of finite-size losses. The relations case at moderate `p` therefore still needs either a better certificate or a direct proof of `RM_gamma` from near-optimality of the allocation.
3. *(General `n = 2`, constant 0.42.)* Out of reach for any coverage-type line lemma: flat-at-0.42 witnesses reach 86–95% of the frontier coverage, and programme supports themselves sit at `sigma ~ 0.42–0.50` (minimum found 0.4177).
   * Caveat: the application's normalisation `Sum v_l sqrt(kappa_l(delta_0)/2)` may lower the target below 0.42 — check before concluding.

## 4. Files

All of these are in the scratch directory `/tmp/claude-0/-home-user-grok/512ae18c-6d12-5fa4-88c0-000434907593/scratchpad/unifiedA/`.

| file | contents |
|---|---|
| `lib.py` | coverage, `K_eff`, exact `sigma`, adversarial phase optimiser |
| `fejer_blocks.py`, `fejer_large.py` | greedy gluing; output in `fejer_blocks.out`, `fejer_large.out` |
| `blocks_ld.py`, `tables.py` | asymptotic rates; output in `tables.out` |
| `unicorn.py`, `hunt2.py`, `hunt3.py` | LP / phase unicorn hunt; JSON witnesses `hunt3_*.json` |
| `hunt_prog.py`, `prog_frontier.py`, `prog601.py` | programme supports with free `v`, flatness floors and adversarial phases |
| `tilt_compare.py`, `riesz_cert.py`, `rm_compare.py` | exact tilt, Riesz certificate (Prop. 4), relation excess |
| `delsarte*.py`, `capbound.py` | Delsarte-LP attempt to bound the coverage profile; sphere bounds proved too weak (kissing-type clusters) |
| `witnesses.py` → `witnesses.json` | all explicit witnesses |
| `verify.py` → `verify.out` | independent verification |
