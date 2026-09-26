# PRIME_AXIS — why the prime axis breaks, and the recipe that fixes it (2026-09-19)

Target: `RESULTS_REP.md` § R1 and Reading 3(a) / 4; `INIT_AUDIT.md` H4 / H5 / H6.
New scripts (all with the `pax_` prefix, nothing existing edited, deleted or committed): `pax_tail.py`,
`pax_ceiling.py`, `pax_run.py`, `pax_xi.py`, `pax_esclr.py`, `pax_score.py`, `pax_report.py`, and the job
`pax_axis.py`, `pax_ceil_big.py`, and the job scripts `pax_job1.sh`, `pax_job2.sh` (local CPU) /
`pax_pod.sh`, `pax_pod2.sh`, `pax_pod3.sh`, `pax_pod4.sh` (pod). Outputs under `heracles/FINAL_RUNS/PAX/`
(`pod/` holds the 28 pod cells). Compute: ~70 min local CPU (diagnosis + 1–2-seed validation) and
**3.45 GPU-hours on one RTX 5090 secure pod at $0.99/h = $3.42** (16-seed validation); pod terminated after a
checksum-verified pull (md5 80ab9d68… and cdbd2b4e… matched on both ends).

Each claim is tagged **[M] measured**, **[D] derived** (algebra from a stated model) or **[C] conjectured**.

---

## 0. Symbols — every one in plain language

| symbol | plain-language meaning |
|---|---|
| p | the modulus. Task = the full p² table of a + b ≡ c (mod p) |
| n_f = (p−1)/2 | how many distinct Fourier frequencies exist on Z_p |
| m = 500 | width (number of hidden neurons) |
| θ | *all* the weights of the net stacked into one long vector: the two embeddings E₀, E₁ (each m × p) and the readout W (p × m) |
| ‖θ‖ | its length = √(sum of squares of every weight). `norm2` in the logs is ‖θ‖² |
| s = ‖θ‖ | short name for that length. s₀ = its value at step 0 |
| ν = 3 | homogeneity degree: multiply every weight by c and every logit multiplies by c³ (x² net and bilinear net both) |
| λ = 10⁻⁴ | weight-decay strength. Loss = (mean cross-entropy over the p² rows) + λ‖θ‖² |
| η (lr) | learning rate. Archived recipe: 0.05, doubled at steps 1000, 2000 … 10000, capped at 12.8 |
| margin | over all p² rows, the smallest (logit of the right answer − largest wrong logit). Positive ⇔ the net separates the data |
| γ = margin/‖θ‖³ | **normalised margin**. Scale-free, so it measures the *direction* of θ, not its size. This is what the sparsity theory maximises |
| g(D) | the "clean" part of the logits: the exact projection of the logit table onto functions of D = a + b − c only |
| clean fraction | share of total (centred) logit energy that lives in g(D). 1.00 = a pure modular-addition circuit |
| en_k | share of g(D)'s energy sitting at frequency k (Σ_k en_k = 1) |
| K_5pct, K_2pct | #{k : en_k ≥ 5 %, 2 %} — **head** counts (fixed relative threshold) |
| N_.05 | the smallest number of frequencies holding 95 % of g(D)'s energy — a **cumulative / tail** count |
| M_K(p) | coverage of the best K-frequency set: max over supports S of size K of min_{D≠0} Σ_{k∈S}(1 − cos 2πkD/p). The theory's objective |
| K*(p) | argmax over K of M_K(p)·K^{−ν/2} — the predicted frequency count under plain weight decay |
| γ_con(K, p) | normalised margin of the **exact junk-free constructed** x² circuit on the coverage-optimal K-support: the ceiling for that many frequencies |
| **quality** = γ/γ_con(K*) | how close a trained net gets to the best possible circuit at that prime. 1.00 = on the ceiling |
| **τ (progress time)** | τ(t) = Σ_{steps u ≤ t} η_u · ‖θ_u‖^{ν−2}. For ν = 3 that is Σ η_u ‖θ_u‖. This is the clock on which the *direction* of θ moves (§2) |
| **τ\*(p)** | how much progress time the network needs to escape the cross-entropy plateau |
| **τ_∞ = s₀/(2λ)** | *all* the progress time a run can ever buy, because weight decay eats the norm at a constant rate per unit τ (§2) |
| **Ξ = τ_∞/τ\*** | the **escape number**. Escape happens iff Ξ > 1; the fraction of the initial norm still standing at escape is 1 − 1/Ξ |

---

## 1. Executive summary

**There are two independent failures on the prime axis, not one.** They have been running together and that is
why the axis looked inexplicable.

### Failure 1 — the escape race (symptom 1, and most of symptoms 2–3). Solved.

> **Ξ(p, s₀, λ) = ‖θ₀‖ / (2 λ τ\*(p))**. τ\*(p) — the "progress time" needed to leave the CE plateau — is
> **measured** to grow like p^{2.3–2.6}, while the budget ‖θ₀‖/(2λ) grows only like init·√(m + p). Escape
> happens iff Ξ > 1. At the archived settings (init 4, λ 10⁻⁴, m 500) Ξ =
> 11.5, 6.9, 3.8, 3.4, 2.2, 1.9, **1.1, 0.62, 0.32** at p = 59, 71, 89, 97, 113, 127, 157, 211, 307.

* **[M] 17/17.** Every archived x² cell with Ξ ≥ 1.63 fits 16/16; every cell with Ξ ≤ 1.12 fails. Perfect
  separation across 5 primes and 4 init scales, from one measured number per prime and no fitted parameter.
* **[D] Why "more steps" cannot help at p ≥ 211**: weight decay costs a *fixed* amount of direction-space travel
  per unit norm (ds/dτ = −2λ exactly), so τ_∞ = s₀/(2λ) is the total travel available over an infinite run.
  At p 211/307 that is 0.62/0.32 of what escape needs, and the archived runs end at ‖θ‖² = **0.0**.
* **[M] Fix:** turn the penalty off until escape (λ = 0 until CE < 0.8 ln p), constant lr. Ξ = ∞ at every prime.
  x² at p 157 goes from 0/16 fit at 40 k steps to **16/16 fit, escape at step 316**; p 211 from "never left zero"
  to 16/16 fit. Quality at p 113 rises 0.246 → 0.316, and N_.05 falls 18 → 15.

### Failure 2 — the finite-λ soft margin (the residual). Diagnosed, not yet solved.

Even with a perfect escape, quality γ/γ_con(K\*) still falls **0.523 → 0.456 → 0.424 → 0.362 → 0.314 → 0.258 →
0.188 → −0.001** over p = 59 → 211 (16 seeds each). At λ = 10⁻⁴ the p 211 cell sits exactly on the separation
boundary (median margin −0.009, 8/16 seeds positive), so γ — and with it the whole max-margin framing — is
**undefined above p ≈ 200 at that λ**. §5.5's hypothesis: the net maximises a *soft* minimum over p − 1 level-set
gaps, which equals the hard margin only when margin ≫ ln(p − 1); measured margin/ln(p−1) falls 0.84 → 0.24 over
p 59 → 157. **Consequence for the paper: the λ ladder is not optional above p ≈ 157.**

### Instruments

* **[M] N_.05 is a tail statistic, not a count** (§4.3). With the fixed recipe at λ 10⁻⁵ the head counts are
  flat over the whole axis — K_5pct = 7, 6.5, 6, 7, 7, 6, 7, 7, 5, 4 and K_2pct = 10.5, 10.5, 10.5, 10.5, 11,
  11, 10, 10.5, 6, 4.5 at p = 59…509 while n_f goes **29 → 254** — while N_.05 grows, because at fixed head
  quality N_.05 ≈ K + T·(0.95 − f)/(1 − f) with T (the tail's effective width) ∝ n_f.
* **[M] The theory's own argmax is not identified above p ≈ 113**: two independent local searches disagree on
  K\* at p 127, 157 and 211. Show the plateau band, not a pin.

---

## 2. [D] The escape law — derivation, every symbol defined as it appears

### 2.1 The loss near zero weights

Write the loss actually being minimised as

    L(θ)  =  (1/N) Σ_{rows} [ log Σ_c exp z_c(row)  −  z_y(row) ]   +   λ ‖θ‖²,      N = p² rows.

*z_c(row)* is the logit the net gives class c on that row; *y* is the correct class; *λ‖θ‖²* is plain weight decay.
When all the weights are small every logit is small, and expanding the log-sum-exp to first order gives

    L(θ)  ≈  ln p  −  S(θ)  +  λ ‖θ‖²,      **S(θ) = (1/N) Σ_{rows} [ z_y − (average of z over all p classes) ].**

*S* is the **signal**: how much more logit the right answer gets than the average answer, averaged over the table.
Two facts about S:

1. **S is exactly the sum of the clean Fourier amplitudes.** Averaging over rows of a level set D = a + b − c gives
   the clean function g(D) (THEORY_nsum L1), and S = g(0) − mean_D g(D) = Σ_{k=1}^{n_f} 2 Re ĝ(k). So the quantity
   that drives escape is literally the object the sparsity theory studies.
2. **S is homogeneous of degree ν = 3:** S(c θ) = c³ S(θ), because the net is.

### 2.2 Split the motion into "how long is θ" and "which way does θ point"

Write θ = s·u where *s = ‖θ‖* is the length and *u* is a unit vector (the direction). Gradient descent with
learning rate η on L gives, in continuous time,

    dθ/dt  =  η ( ∇S(θ) − 2 λ θ ).

Euler's identity for a degree-3 homogeneous function (u·∇S(θ) = 3 S(θ)/s) splits this exactly into

    **ds/dt = η ( 3 s² S(u) − 2 λ s )**            (the length)
    **du/dt = η s^{ν−2} · P(∇S(u))**               (the direction; P = the part perpendicular to u)

with S(u) the signal of the *unit-length* direction. Note the decay term **only** appears in the length equation:
weight decay never turns the direction, it only shrinks the vector.

### 2.3 The progress-time clock, and why the budget is finite

Define **progress time**

    **τ(t) = ∫₀^t η(u) ‖θ(u)‖^{ν−2} du**     (for ν = 3: τ = ∫ η ‖θ‖ dt; in code, Σ_steps lr·‖θ‖).

In this clock the direction equation becomes **du/dτ = P(∇S(u))** — a flow with *no* η, no λ, no s in it. Two
runs with different learning rates or different λ trace the **same curve** through direction space; they only
travel along it at different speeds. And on the plateau (S(u) still tiny) the length equation becomes, dividing
one by the other,

    **ds/dτ = −2 λ**   exactly   ⟹   **s(τ) = s₀ − 2 λ τ**.

So the length falls *linearly in progress time* and hits zero at

    **τ_∞ = s₀ / (2 λ)**   — the total progress time the run can ever buy.

That is the whole mechanism. Weight decay does not merely slow escape down; it imposes a **hard finite budget of
direction-space travel**, and if the direction has not reached the escape region by then the network is gone.
(Symptom 1's "never leaves zero" is exactly this: the archived p 211 and p 307 runs end with ‖θ‖² = 0.0 and
CE = ln p to four decimals, **[M]** `FINAL_RUNS/REP/n2_pow_p{211,307}_…/seed*/final.safetensors`.)

For a general degree ν the same algebra gives ds/dτ = −2λ s^{3−ν} and τ_∞ = s₀^{ν−2}/(2λ(ν−2)) — the escape-from-
zero problem is an s^{ν−2} problem, as THEORY_nsum §3/§6 says, and ν = 3 is the one case where it is linear.

### 2.4 The requirement, τ\*(p)

**[M] Measurement (`pax_job1.sh`, `pax_run.py --lam 0`).** With λ = 0 the norm is constant on the plateau, so
τ = η s₀ t and the step at which CE first drops below 0.8 ln p measures τ\* directly:

| p | 59 | 71 | 89 | 113 | 157 | 211 | 307 |
|---|---|---|---|---|---|---|---|
| steps to escape (η = 12.8, init 4, 2 seeds) | 33 | 54 | 95 | 162 | 309 | 552 | 1070 |
| **τ\*** | 2.39e4 | 4.01e4 | 7.45e4 | 1.32e5 | 2.64e5 | 4.95e5 | 1.02e6 |
| τ\*/p^{2.5} | 0.892 | 0.945 | 0.997 | 0.972 | 0.854 | 0.765 | 0.618 |

* **[M] τ\* is a property of the clock, not of the learning rate.** At p 71, init 4: η = 12.8 → τ\* = 4.014e4
  (54 steps); η = 3.2 → τ\* = 3.941e4 (211 steps). Agreement 1.8 %, over a 4× change in η and in step count.
  This is the sharpest available check that τ is the right clock.
* **[M] τ\* depends only weakly on the init scale.** p 71: τ\* = 4.01e4 / 5.98e4 / 6.57e4 at init 4 / 2 / 1 —
  a 4× change of s₀ moves τ\* by 1.6×. (Expected: the escape *threshold* CE < 0.8 ln p asks for logits of order 1,
  i.e. for s³·(alignment) ≈ 1, so a smaller s needs a little more alignment. The *flow* itself is scale-free.)
  Ξ below uses the init-4 value of τ\* at each prime; this biases Ξ **upward** by ≤ 1.6× for the init-1/2 cells.
* **[M] Power-law fit** over p 59…307: **τ\* = 2.65 · p^{2.27}** (least squares in log-log). The local exponent
  drifts from 2.8 at the small-p end to 2.1 at the large-p end, so quote it as **τ\* ∝ p^{2.3±0.3}**, not as an
  exact power.
* **[D]+[C] Where p^{5/2} comes from.** At random init the clean amplitude at each frequency is a sum of m random
  triple products, |ĝ(k)| ~ init³/(3√3 · p^{5/2}), while the largest amplitude reachable at the same norm is
  ~ s₀³/(18 p^{3/2}). So the **initial alignment fraction** is ε₀ ~ 18/(p (m+p)^{3/2}) — it shrinks with p because
  a random direction has less and less overlap with any clean circuit. The direction flow near the plateau is a
  three-factor power iteration, dε/dτ ∝ ε²·(rate), and the rate constant carries the same p^{−3/2} as S itself,
  so the blow-up time is τ\* ~ p^{3/2}/ε₀ ∝ **p^{5/2}(m+p)^{3/2}**. Measured τ\*/p^{2.5} is flat to ±13 % over
  p 59…211 but the extra (m+p)^{3/2} is *not* seen (it would add a further 1.27× over that range and the residual
  drifts the other way). **So: the p^{5/2} is derived and confirmed to ±13 %; the width factor is conjecture and
  is not confirmed.** Treat the width dependence of τ\* as unmeasured.

### 2.5 The escape number and the critical λ

Putting the budget and the requirement together:

    **Ξ(p, s₀, λ)  =  τ_∞/τ\*  =  s₀ / (2 λ τ\*(p))**,      s₀ = init · √((m + p)/3)  (the draw in `nsum_rep.py`)

* **escape iff Ξ > 1** — equivalently **λ < λ_c(p) = s₀/(2 τ\*(p))**, equivalently
  **init > init_c(p) = 2 λ τ\*(p)/√((m+p)/3)**;
* **[D] the fraction of the initial norm still standing at escape is 1 − 1/Ξ**;
* **[D] with τ\* ∝ p^{5/2}(m+p)^{3/2} and s₀ ∝ init√(m+p), λ_c ∝ init / (p^{5/2}(m+p))** — so at fixed λ the
  init scale needed to escape grows like p^{5/2}(m+p): from p 71 to p 211 that is a factor **19**, and to p 307 a
  factor **55**. Raising init is a hopeless lever; that is the content of INIT_AUDIT's H4 collapse ladder.
* Width is also a weak lever: s₀ ∝ √m, so Ξ ∝ √m. Quadrupling the width buys a factor 2.

λ_c per unit init scale, from the measured τ\*: **[M/D]**

| p | 59 | 71 | 89 | 97 | 113 | 127 | 157 | 211 | 307 |
|---|---|---|---|---|---|---|---|---|---|
| λ_c / init | 2.9e−4 | 1.7e−4 | 9.4e−5 | 8.4e−5 | 5.4e−5 | 4.7e−5 | 2.8e−5 | 1.6e−5 | 8.0e−6 |
| init needed at λ = 10⁻⁴ | 0.35 | 0.58 | 1.06 | 1.19 | 1.84 | 2.13 | 3.57 | 6.43 | 12.5 |

The last row is INIT_AUDIT H4's ladder, predicted: "p 59/71 escape at init 1, p 89 needs ≥ 2, p 113 needs ≥ 3,
p ≥ 157 fails even at 4".

---

## 3. [M] Symptom 1 — why x² dies above p 127. Ξ predicts 17/17 archived cells

`pax_xi.py`; one measured τ\* per prime (§2.4), s₀ from the init draw, λ = 10⁻⁴; no fitted parameter.

| p | init | s₀ | τ\* | **Ξ** | 1 − 1/Ξ | fit | median acc | margin | **γ/γ_con** | clean | N_.05 | K_5pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 89 | 1 | 14.0 | 7.45e4 | **0.94** | 0 | **0/16** | 0.366 | — | — | — | — | — |
| 211 | 4 | 61.6 | 4.95e5 | **0.62** | 0 | **0/8** | 0.081 | — | — | — | — | — |
| 307 | 4 | 65.6 | 1.02e6 | **0.32** | 0 | **0/8** | 0.013 | — | — | — | — | — |
| 113 | 2 | 28.6 | 1.32e5 | **1.08** | 0.077 | **0/16** | 0.668 | — | — | — | — | — |
| 157 | 4 | 59.2 | 2.64e5 | **1.12** | 0.109 | **0/16** | 0.980 | — | — | — | — | — |
| 113 | 3 | 42.9 | 1.32e5 | 1.63 | 0.385 | 16/16 | 1.0000 | 1.281 | 0.185 | 0.403 | 18.0 | 7.0 |
| 97 | 2 | 28.2 | 8.37e4 | 1.69 | 0.407 | 16/16 | 1.0000 | 1.008 | 0.144 | 0.353 | 15.0 | 5.5 |
| 71 | 1 | 13.8 | 4.01e4 | 1.72 | 0.418 | 16/16 | 1.0000 | 1.122 | 0.129 | 0.309 | 13.0 | 4.0 |
| 127 | 4 | 57.8 | 1.54e5 | 1.88 | 0.467 | 16/16 | 1.0000 | 1.013 | 0.157 | 0.435 | 19.5 | 6.0 |
| 89 | 2 | 28.0 | 7.45e4 | 1.88 | 0.468 | 16/16 | 1.0000 | 1.750 | 0.259 | 0.407 | 15.0 | 6.0 |
| 113 | 4 | 57.2 | 1.32e5 | 2.17 | 0.539 | 16/16 | 1.0000 | 1.719 | 0.246 | 0.461 | 18.0 | 7.5 |
| 59 | 1 | 13.7 | 2.39e4 | 2.86 | 0.651 | 16/16 | 1.0000 | 2.550 | 0.352 | 0.440 | 10.0 | 6.5 |
| 97 | 4 | 56.4 | 8.37e4 | 3.37 | 0.703 | 16/16 | 1.0000 | 2.349 | 0.353 | 0.522 | 12.0 | 7.0 |
| 71 | 2 | 27.6 | 4.01e4 | 3.44 | 0.709 | 16/16 | 1.0000 | 2.726 | 0.377 | 0.461 | 12.5 | 8.0 |
| 89 | 4 | 56.0 | 7.45e4 | 3.76 | 0.734 | 16/16 | 1.0000 | 2.602 | 0.407 | 0.551 | 10.0 | 8.0 |
| 71 | 4 | 55.2 | 4.01e4 | 6.87 | 0.855 | 16/16 | 1.0000 | 3.105 | 0.449 | 0.587 | 10.0 | 7.0 |
| 59 | 4 | 54.6 | 2.39e4 | 11.45 | 0.913 | 16/16 | 1.0000 | 3.285 | 0.500 | 0.658 | 8.0 | 6.0 |

**Perfect separation.** Every cell with Ξ ≤ 1.12 fails (0 fit seeds); every cell with Ξ ≥ 1.63 fits all 16. The
two marginal cells (Ξ = 1.08 at p 113 init 2, Ξ = 1.12 at p 157 init 4) behave exactly as "escapes with 8–11 % of
its norm left" predicts: they *do* leave the plateau — p 113 init 2's CE drops 4.56 → 0.87 between step 8000 and
10000, p 157 init 4's drops 5.03 → 0.60 between 6000 and 8000 — and then cannot fit from the wreckage
(acc 0.67 and 0.98).

### Hypotheses killed here

* **"Just under-trained at 40 k / lr cap too low."** **Killed for p ≥ 211 by the algebra:** τ_∞ = s₀/(2λ) is the
  *total* progress time available over an infinite run, and at p 211 / 307 it is 0.62 / 0.32 of what escape needs.
  More steps cannot help; the run is annihilated, not slow. **[M]** confirmed: those runs end at ‖θ‖² = 0.0.
  At p 157 (Ξ = 1.12 > 1) the run *does* escape and more steps would help a little — "under-trained at p 157" is
  a symptom of escaping on 11 % of the norm, not the cause.
* **"fp32 breaks down at large p (tiny gradients, softmax over p classes)."** **Killed:** the *same* net, the
  *same* fp32 arithmetic, at p = 307, escapes in 1070 steps as soon as λ is set to 0 (`pax_tau_p307_i4_lr12.8`,
  CE 5.727 → 4.58, then fits in §5). Precision is not the obstruction.
* **"width 500 is too small at large p."** **Not the obstruction:** the theory's own ceiling γ_con needs 6 neurons
  per frequency, i.e. ≈ 48 of 500 at K = 8; and at p 59 the *same* width reaches quality 0.50. Width enters only
  through s₀ ∝ √m (a Ξ lever of √m) — see §7 for the one thing width might still do to the tail.
* **"init-scale/effective-lr confound (INIT_AUDIT H5)."** **Confirmed and now quantified:** init scale acts on
  escape through Ξ ∝ s₀ and on nothing else at leading order, because the direction flow du/dτ is exactly
  scale-free (§2.2). The residual confound (τ\* itself moves 1.6× over a 4× init change) is measured in §2.4.

---

## 4. [M] Symptom 3 — why quality, the clean fraction and N_.05 all degrade with p

### 4.1 The exact ceiling, and how far the runs are from it

**[D] closed form, verified exactly (`pax_ceiling.py`).** The junk-free 6-neuron-per-frequency x² construction on
support S gives logits z = 3 Σ_{k∈S} cos(2πk(a+b−c)/p) and ‖θ‖² = 9pK, so

    **γ_con(K, p) = 3 · M_K(p) / (9 p K)^{3/2}**

(checked against the explicit circuit at p 59 and 71, K = 3, 5, 8: agreement to all printed digits). Note the
**p^{−3/2}**: the achievable normalised margin itself falls like p^{−3/2}, and p^{3/2}γ_con is flat to 40 %
(0.0331 → 0.0237 over p 59 → 307, the residual being the drift of M_K/K^{3/2}).

**[M] Quality (γ_median/γ_con(K\*)) of the archived cells, init 4:**

| p | 59 | 71 | 89 | 97 | 113 | 127 | 157 | 211 |
|---|---|---|---|---|---|---|---|---|
| x² (no warm-up) | 0.500 | 0.449 | 0.407 | 0.353 | 0.246 | 0.157 | fail | fail |
| bilinear (warm-up) | 0.263 | 0.201 | 0.169 | 0.144 | 0.081 | 0.051 | −0.056 | −0.228 |

**Nothing on this axis is anywhere near an optimum**, and it gets worse with p. This is the honest frame for the
whole prime-axis figure (it is the same point INIT_AUDIT made about the init axis, now on the prime axis).

### 4.2 Quality correlates almost perfectly with Ξ in the archive — but Ξ is **not** the cause

**[M]** Over the 12 archived x² cells that fit — 5 primes × 4 init scales — **Spearman ρ(Ξ, γ/γ_con) = +0.944,
p = 3.9 × 10⁻⁶** (`pax_xi.py`):

```
Xi:      1.63  1.69  1.72  1.88  1.88  2.17  2.86  3.37  3.44  3.76  6.87 11.45
gam/gc: 0.185 0.144 0.129 0.157 0.259 0.246 0.352 0.353 0.377 0.407 0.449 0.500
clean:  0.403 0.353 0.309 0.435 0.407 0.461 0.440 0.522 0.461 0.551 0.587 0.658
```

On the archive alone one would conclude "the prime axis and the init axis are the same axis, and both are Ξ".
**That conclusion is wrong, and the intervention in §5 is what kills it.** Setting Ξ = ∞ (penalty off until
escape) moves p 113 only from 0.246 to **0.316** — not to the 0.45–0.50 that the Ξ ≳ 7 archived cells reach —
and p 157 to 0.18. So:

* **Ξ is the whole story for *escape*** (§3: 17/17, perfect separation).
* **Ξ is only part of the story for *quality*.** It buys ≈ +30 % at p 113. The rest of the archived correlation
  is confounding: in the archived design, Ξ, the prime, and the learning rate at the moment of escape all move
  together. A residual p-decay of quality survives the fix (0.32 at p 113 → 0.18 at p 157 at 16 seeds).
* **[M] A second candidate was tested and killed.** In the archived runs the lr was still ramping at escape, so
  high-Ξ cells escaped at lr 3.2 and low-Ξ cells at lr 12.8; split that way, quality is 0.407 [0.353–0.500] vs
  0.185 [0.129–0.352], Mann-Whitney p = 0.0025 (`pax_esclr.py`) — but it is collinear with Ξ in the archive.
  The pod cohort breaks the collinearity by fixing Ξ = ∞ and sweeping the escape lr over **32×** at 16 seeds:

  | p | lr at escape | escape step | quality | clean | K_5pct | N_.05 |
  |---|---|---|---|---|---|---|
  | 113 | 12.8 | 163 | 0.316 | 0.471 | 8.0 | 15.5 |
  | 113 | 1.6 | 1284 | 0.314 | 0.470 | 7.5 | 15.5 |
  | 113 | 0.4 | 5129 | 0.299 | 0.471 | 7.5 | 15.5 |
  | 157 | 12.8 | 316 | 0.180 | 0.495 | 7.0 | 17.5 |
  | 157 | 1.6 | 2507 | 0.183 | 0.496 | 6.5 | 17.0 |
  | 157 | 0.4 | 10017 | 0.140 | 0.462 | 6.0 | 20.5 |

  **Null.** Quality, the clean fraction and all three counts are identical across a 32× change in how violently
  the network escapes (the lr 0.4 cells are slightly lower only because escape eats 5 k / 10 k of their 25 k-step
  budget). *How* the net escapes does not matter; *whether it escapes with its norm intact* matters a little;
  the residual decay with p is neither.

### 4.3 Why N_.05 grows while K_5pct does not

**[M] (`pax_tail.py`).** For the archived init-4 cells, decompose the clean spectrum into the ≥ 5 % head and
the rest:

| net | p | n_f | K_5pct | head energy f | N_.05 | tail participation ratio | tail PR / n_f |
|---|---|---|---|---|---|---|---|
| x² | 59 | 29 | 6.0 | 0.885 | 8.0 | 3.1 | 0.13 |
| x² | 71 | 35 | 7.0 | 0.870 | 10.0 | 3.6 | 0.13 |
| x² | 89 | 44 | 8.0 | 0.878 | 10.0 | 5.0 | 0.14 |
| x² | 97 | 48 | 7.0 | 0.835 | 12.0 | 7.3 | 0.18 |
| x² | 113 | 56 | 7.5 | 0.808 | 18.0 | 10.6 | 0.22 |
| x² | 127 | 63 | 6.0 | 0.797 | 19.5 | 13.8 | 0.24 |
| bilinear | 157 | 78 | 5.0 | 0.749 | 18.5 | 17.2 | 0.24 |
| bilinear | 211 | 105 | 3.0 | 0.590 | 23.0 | 23.5 | 0.23 |

Two things move and one does not.

1. **The head count does not grow.** K_5pct is 6, 7, 8, 7, 7.5, 6 over p 59…127 while n_f goes 29 → 63. That is
   the paper's claim and it survives.
2. **The head's share of the energy falls** (0.885 → 0.797 for x², 0.945 → 0.590 for the bilinear) — that is the
   quality collapse of §4.2 seen in the spectrum.
3. **The tail spreads over a fixed *fraction* of the available frequencies** (tail PR ≈ 0.13–0.24 n_f), so the
   *number* of tail modes grows linearly in n_f.

N_.05 asks "how many modes to reach 95 %", so with head fraction f, head size K and a tail spread over T modes,
**[D] N_.05 ≈ K + T·(0.95 − f)/(1 − f)** — which grows **linearly in n_f at fixed head quality**. Numerically
(same table): p 59 gives 6 + 3.1·0.065/0.115 = 7.8 against a measured 8.0; p 127 gives 6 + 13.8·0.153/0.203 =
16.4 against a measured 19.5. **N_.05 is a tail statistic, not a count**, and it must not be reported as "the
number of frequencies" on an axis where n_f varies by 5×.

* **[M] The tail is not an un-pruned transient.** The weight-decay pruning time constant is 1/(2ηλ) = 390 steps
  at η = 12.8, λ = 10⁻⁴ — and the runs are 40 000 steps. Directly: 40 k *extra* steps at the same λ move nothing
  (INIT_AUDIT H2), and the archived λ/10 continuations move γ/γ_con up a lot (p 113: 0.246 → **0.393**; p 71
  init 1: 0.129 → 0.263) but leave the **clean fraction flat** (0.461 → 0.448) and N_.05 **exactly unchanged**
  (18.0 → 18.0), K_5pct unchanged (7.5 → 7.5). So lowering λ raises γ without touching the junk: **the junk
  content of the basin is fixed early and neither λ nor time removes it.** (§5.6 shows that lowering λ does move
  γ/γ_con a great deal — but through the margin, not through the clean fraction, exactly as this row says.)

---

## 5. The fix, and what it does and does not buy

### 5.1 The intervention

> **Turn the penalty off until the network escapes.** λ = 0 until the training cross-entropy falls below
> 0.8 ln p, then λ = 10⁻⁴ for the rest of the budget; constant lr 12.8 from step 1; grad-norm clip 1; init 4.

Why this and not something else, from §2.5: Ξ = s₀/(2λτ\*) fails because τ\* grows like p^{2.3–2.6} while every
other knob is weak — init enters linearly (needs 6.4× at p 211, 12.5× at p 307), width only as √m, and lowering
λ globally would change the finite-λ margin trade-off and so change the counts being compared across primes.
Setting λ = 0 *only during the plateau* makes **Ξ = ∞ at every prime** while leaving the quantity the paper
compares — the finite-λ solution at λ = 10⁻⁴ — untouched. It is also not a new recipe: it is the `--wd_warmup`
already used for every bilinear cell in RESULTS_REP and RESULTS_PROD, transplanted to x².

### 5.2 [M] Local (CPU, `pax_job2.sh`, 1–2 seeds) — the minimum bar the brief asked for

| cell | escape | acc (fp64) | min margin | γ | quality | ‖θ‖² | clean | K_5pct | K_2pct | N_.05 |
|---|---|---|---|---|---|---|---|---|---|---|
| **p 157**, archived recipe, 40 k | step 10 000 | 0.9958 | −3.22 | — | — | 6437 | — | — | — | — |
| **p 157**, fix, 10 k, seeds 100/101 | **step 310** | **1.00000** | **+1.32 / +1.44** | 2.80/2.98e−6 | 0.20/0.22 | 6072 | 0.38/0.35 | 6/7 | 10/8 | 23/29 |
| **p 211**, archived recipe, 40 k | never | 0.0797 | — | — | — | **0.0** | — | — | — | — |
| **p 211**, fix, 10 k, seed 100 | **step 552** | **0.99993** | −0.88 | — | — | 8323 | 0.345 | 4 | 4 | 32 |

So the escape claim is validated locally: p 157 goes from "0/16, under-trained at 40 k" to "fits in 310 steps
with a positive margin", and p 211 from "never left zero, ‖θ‖² = 0.0" to "escapes at step 552 and reaches
accuracy 0.99993". p 211 still has a negative minimum margin after only 10 k steps — see the 25 k-step pod cell.

**[M] CPU/GPU agreement:** the local p 113 fix run and the pod's `paxA_p113_esclr12.8` cell share seeds 100, 101
and agree on ‖θ‖² at step 1000 to every printed digit (4866 / 4913).

### 5.3 [M] Pod (RTX 5090, 16 seeds/cell) — the recipe-selection grid

Cells at p = 113, all 16/16 fit, 25 k steps, λ = 10⁻⁴, init 4:

| cell | escape at | quality γ/γ_con | clean | K_5pct | K_2pct | N_.05 |
|---|---|---|---|---|---|---|
| archived REP (lr ramp, penalty on, 40 k) | step 8000 | 0.246 | 0.461 | 7.5 | 9.0 | 18.0 |
| `paxCTL_p113_nowu` (constant lr 12.8, penalty on) | step 243 | 0.280 | 0.462 | 7.5 | 9.0 | 18.0 |
| **`paxA_p113_esclr12.8`** (constant lr 12.8, **warm-up**) | step 163 | **0.316** | 0.471 | 8.0 | 10.0 | **15.5** |

The warm-up is worth **+13 %** on quality at matched lr schedule (0.280 → 0.316) and takes N_.05 from 18 to 15.5;
the constant-lr change on its own is worth another +14 % (0.246 → 0.280). Neither is worth 2×. **The escape fix
is decisive for whether the run exists at all and marginal for its quality.**

### 5.4 [M] The whole prime axis with the fix (16 seeds/cell, 25 k steps, λ 10⁻⁴, init 4, `pax_axis.py`)

| p | n_f | archived: fit | archived quality | **fix: escape at step** | **fix: fit** | margin>0 | margin | **quality** | clean | **K_5pct** | **K_2pct** | N_.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 59 | 29 | 16/16 | 0.500 | **33** | 16/16 | 16/16 | 3.417 | **0.523** | 0.672 | 6.0 | 9.0 | 8.0 |
| 71 | 35 | 16/16 | 0.449 | **54** | 16/16 | 16/16 | 3.097 | **0.456** | 0.616 | 7.0 | 10.0 | 9.5 |
| 89 | 44 | 16/16 | 0.407 | **95** | 16/16 | 16/16 | 2.703 | **0.424** | 0.565 | 7.5 | 9.5 | 9.5 |
| 97 | 48 | 16/16 | 0.353 | **116** | 16/16 | 16/16 | 2.392 | **0.362** | 0.557 | 7.0 | 10.0 | 10.5 |
| 113 | 56 | 16/16 | 0.246 | **163** | 16/16 | 16/16 | 2.116 | **0.314** | 0.470 | 8.0 | 10.0 | 15.0 |
| 127 | 63 | 16/16 | 0.157 | **207** | 16/16 | 16/16 | 1.674 | **0.258** | 0.484 | 7.0 | 9.0 | 16.0 |
| 157 | 78 | **0/16** | — | **316** | **16/16** | **16/16** | 1.221 | **0.188** | 0.497 | 7.0 | 8.0 | 17.0 |
| 211 | 105 | **0/8** | — | **553** | **16/16** | 8/16 | −0.009 | −0.001 | 0.397 | 6.5 | 7.5 | 25.5 |
| 307 | 153 | **0/8** | — | **1097** | 4/16 | 0/16 | −1.447 | −0.249 | 0.330 | 6.5 | 6.5 | 28.5 |

Read this table in three parts.

1. **Escape is completely fixed.** All nine primes escape, at steps 33…1097 instead of 6000…never, and the
   escape step tracks τ\*(p)/(lr·s₀) as §2 predicts (measured escape 33/54/95/163/316/553/1097 against the λ = 0
   calibration 33/54/95/162/309/552/1070 — agreement to ≤ 3 %). p 157 goes from **0/16 to 16/16 with positive
   margins**; p 211 from ‖θ‖² = 0.0 to 16/16 fit.
2. **Where the archive already escaped comfortably, the fix changes nothing** (p 59: 0.500 → 0.523, p 71:
   0.449 → 0.456), and where Ξ was marginal it helps a lot (p 113: 0.246 → 0.314, p 127: 0.157 → 0.258).
   Exactly the pattern §2 predicts, and a good consistency check on Ξ.
3. **λ = 10⁻⁴ runs out of margin above p ≈ 200.** At p 211 the median minimum margin is −0.009 with 8/16 seeds
   positive; at p 307 it is −1.45 with 0/16 positive. Accuracy is still 0.9999+, so these are *soft*-margin
   solutions that do not separate the p² table. **γ, and with it the entire max-margin framing, is undefined
   there at this λ.** See §5.5–§5.6.

### 5.5 [C] Hypothesis Q — the residual decay is a soft-margin (finite-λ temperature) effect

**[D] The ceiling margin at the achieved norm is the same at every prime.** γ_con ∝ p^{−3/2} (§4.1) and the
converged ‖θ‖² ∝ p (§4.1 table), so γ_con·‖θ‖³ — the margin a *perfect* circuit would have at the norm the
network actually reaches — is **6.50 nats at p 59 and 6.46 nats at p 157**, flat across the axis. So
"quality falls with p" is exactly "the achieved margin falls with p at a fixed ceiling of ≈ 6.5 nats".

**[C] Why it should.** Write the loss at the optimum. Every row competes against p − 1 wrong answers, and by the
junk/phase lemmas the gap depends only on D = a + b − c, so

    CE  ≈  Σ_{D ≠ 0} exp( −gap(D) ),     gap(D) = g(0) − g(D),     margin = min_{D ≠ 0} gap(D).

*gap(D)* = how much more logit the right answer gets than the answer that is wrong by D; *margin* = the smallest
of those p − 1 numbers. Minimising CE maximises a **soft** minimum of the p − 1 gaps, and

    −ln Σ_{D≠0} e^{−gap(D)}  ≥  margin − ln(p − 1),

so soft and hard minima coincide only when **margin ≫ ln(p − 1)**. The attainable margin at fixed λ is roughly
p-independent (≈ ln(1/λ) + O(1)), while ln(p − 1) grows. **[M]** margin / ln(p − 1) over the Phase-B axis:

| p | 59 | 71 | 89 | 97 | 113 | 127 | 157 |
|---|---|---|---|---|---|---|---|
| median margin | 3.417 | 3.097 | 2.703 | 2.392 | 2.116 | 1.674 | 1.194 |
| ln(p − 1) | 4.06 | 4.25 | 4.48 | 4.56 | 4.72 | 4.84 | 5.05 |
| margin / ln(p−1) | 0.84 | 0.73 | 0.60 | 0.52 | 0.45 | 0.35 | 0.24 |
| quality | 0.523 | 0.456 | 0.424 | 0.362 | 0.314 | 0.258 | 0.180 |

The network is drifting *out* of the hard-margin regime as p grows. Pre-registered test in
`FINAL_RUNS/PAX/PREREG_job2.md` (written before any λ = 10⁻⁵ cell was pulled): **quality must rise at every
prime at λ = 10⁻⁵ and rise *relatively more* at large p**; scored in §5.6.

### 5.6 [M] Hypothesis Q survives its pre-registered test — the λ rung, 16 seeds, 25 k steps

| p | 59 | 71 | 89 | 97 | 113 |
|---|---|---|---|---|---|
| margin at λ 10⁻⁴ | 3.417 | 3.097 | 2.703 | 2.392 | 2.116 |
| margin at λ 10⁻⁵ | 5.728 | 5.397 | 4.891 | 4.739 | 4.396 |
| **Δ margin** | **+2.31** | **+2.30** | **+2.19** | **+2.35** | **+2.28** |
| quality at λ 10⁻⁴ | 0.523 | 0.456 | 0.424 | 0.362 | 0.314 |
| quality at λ 10⁻⁵ | **0.577** | **0.517** | **0.508** | **0.476** | **0.441** |
| relative rise | +10 % | +13 % | **+20 %** | **+31 %** | **+40 %** |
| K_5pct | 6.0 → 7.0 | 7.0 → 6.5 | 7.5 → 6.0 | 7.0 → 7.0 | 8.0 → 7.0 |
| K_2pct | 9.0 → 10.5 | 10.0 → 10.5 | 9.5 → 10.5 | 10.0 → 10.5 | 10.0 → 11.0 |
| N_.05 | 8.0 → 11.0 | 9.5 → 11.0 | 9.5 → 13.0 | 10.5 → 13.5 | 15.0 → 16.0 |

* **[M] C1 passes exactly as registered.** Quality rises at every prime, and the *relative* rise grows
  monotonically with p (+10 % → +40 %), so the quality-vs-p curve flattens: the λ 10⁻⁵ spread over
  p 59 → 113 is 0.577 → 0.441 (a factor 1.31) against 0.523 → 0.314 (a factor 1.67) at λ 10⁻⁴.
* **[M][D] The margin gains ln 10 = 2.303 per decade of λ**, measured +2.19 to +2.35 at five primes. That is
  precisely the stationarity prediction margin\* ≈ ln(1/λ) + O(1) of §4.1/§5.5, with the O(1) term p-independent
  to ±0.08. This is the strongest single confirmation that the residual decay is a **finite-λ temperature
  effect**, not a failure of gradient descent or of the architecture.
* **[M] C2 passes.** The head counts move by ≤ 1 (K_5pct) and ≤ 1.5 (K_2pct) over a 10× change in λ, replicating
  the archived λ/10 continuations. **N_.05 moves the *wrong* way** (it *rises*, 8 → 11 at p 59) because a weaker
  penalty prunes the tail less — a third, independent reason not to call N_.05 a count.

**Consequence for the paper.** The λ ladder is not a robustness check, it is part of the measurement: at
λ = 10⁻⁴ the p ≥ 211 cells do not separate the data at all. **Report the axis at λ = 10⁻⁵ (and 10⁻⁶ at the
largest primes), with λ = 10⁻⁴ as the tie to the existing R1 cells.**

### 5.6b [M] Two more explanations of the residual, both killed at p 157

* **"It is the step budget."** **Killed, paired.** Same recipe, same 8 seeds, λ 10⁻⁴, p 157, 25 k vs **100 k**
  steps (`paxD1_p157_long100k`): median γ **falls** 2.549e−6 → 1.955e−6, and γ is worse in **6 of 8** seeds;
  margin 1.215 → 0.918; ‖θ‖² unchanged (6066 → 6039); K_5pct, K_2pct and N_.05 all identical (6.5 / 8.5 / 22).
  Four times the training changes nothing except a slight *drift down* in γ — the same thing INIT_AUDIT H2 found
  at p 71, for the same reason: the objective being minimised is CE + λ‖θ‖², not the margin.
* **"Then just keep lowering λ."** **Bounded.** At p 157 the ladder is λ 10⁻⁴ → 10⁻⁵ → 10⁻⁶: margin
  1.221 → 3.677 → 5.471, but quality 0.188 → **0.385** → 0.238. The λ = 10⁻⁶ rung is *worse*, because
  ‖θ‖² nearly doubles (7770 → 14042) and the run is **not converged**: the time constant on which the penalty
  moves the weights is 1/(2ηλ) steps — 390 at λ 10⁻⁴, 3 900 at 10⁻⁵, **39 000 at 10⁻⁶** — so 25 k steps is
  ~6 time-constants at 10⁻⁵ but well under one at 10⁻⁶. **[D] recipe constraint: steps ≳ 10/(2ηλ).**
  At λ 10⁻⁶ and η 12.8 that is ≳ 400 k steps, which is why the ladder should stop at 10⁻⁵ unless the budget is
  scaled with 1/λ.

### 5.7 [M] THE AXIS — fixed recipe at λ = 10⁻⁵, p = 59 … 307 (16 seeds; 8 at p 307), 25 k steps

**Every prime fits, every prime separates, and every prime has a positive margin.**

| p | n_f | escape at | fit | margin > 0 | margin | **quality** | clean | **K_5pct** | **K_2pct** | N_.05 | margin/ln(p−1) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 59 | 29 | 33 | 16/16 | 16/16 | 5.728 | 0.577 | 0.676 | **7.0** | **10.5** | 11.0 | 1.41 |
| 71 | 35 | 54 | 16/16 | 16/16 | 5.397 | 0.517 | 0.627 | **6.5** | **10.5** | 11.0 | 1.27 |
| 89 | 44 | 95 | 16/16 | 16/16 | 4.891 | 0.508 | 0.588 | **6.0** | **10.5** | 13.0 | 1.09 |
| 97 | 48 | 116 | 16/16 | 16/16 | 4.739 | 0.476 | 0.559 | **7.0** | **10.5** | 13.5 | 1.04 |
| 113 | 56 | 163 | 16/16 | 16/16 | 4.396 | 0.441 | 0.510 | **7.0** | **11.0** | 16.0 | 0.93 |
| 127 | 63 | 207 | 16/16 | 16/16 | 4.170 | 0.440 | 0.501 | **6.0** | **11.0** | 16.5 | 0.86 |
| 157 | 78 | 316 | 16/16 | 16/16 | 3.677 | 0.385 | 0.523 | **7.0** | **10.0** | 19.0 | 0.73 |
| 211 | 105 | 553 | 16/16 | 16/16 | 2.935 | 0.323 | 0.419 | **7.0** | **10.5** | 29.0 | 0.55 |
| 307 | 153 | 1097 | 8/8 | 8/8 | 1.789 | 0.201 | 0.329 | **5.0** | **6.0** | 38.5 | 0.31 |

Against the archived R1 row (`RESULTS_REP` §R1: 16/16 at p ≤ 127, **0/16 at 157, 0/8 at 211, 0/8 at 307**), this
is the whole prime axis recovered. **K_5pct stays in 5–7 and K_2pct in 6–11 while n_f goes 29 → 153** — a 5.3×
change in the number of frequencies on offer and no trend in the count. Quality still decays (0.58 → 0.20) and
margin/ln(p−1) still falls (1.41 → 0.31), so §5.5's soft-margin limit is still operating at p 307 at this λ; a
further rung (λ = 10⁻⁶, with the step budget of §5.6b) should buy another ≈ +2.3 of margin there.

**[M] p 509** (`paxE_p509_lam1e-5`, 4 seeds, 15 k steps, λ 10⁻⁵, `--achunk 128`): escapes at step **2703**,
**4/4 at accuracy 1.00000**, K_5pct 4.0, K_2pct 4.5, N_.05 38, clean 0.31 — but median margin **+0.113** with
3/4 seeds positive, i.e. margin/ln(p−1) = 0.02. So at p 509 the *escape* is a solved problem and the
*soft-margin* limit is entirely binding: γ is measurable but meaningless (quality 0.013). **p ≈ 500 needs
λ ≤ 10⁻⁶ with ≥ 4 × 10⁵ steps**, per §5.6b. This is the sharp statement for the "one p ~ 500–1000" cell.

---

## 6. The recipe

### 6.1 What to run

```
uv run python heracles/pax_run.py --act pow --p P --m 500 --seeds ... \
    --init_scale 4 --lr_const 12.8 --clip 1 \
    --wd_warmup 6000 --escape_frac 0.8 \      # lambda = 0 until CE < 0.8 ln p (cap 6000 steps)
    --lam 1e-4 --steps 40000
```
(the same three flags already exist in `nsum_rep.py`: `--wd_warmup --escape_frac --clip`, with `--lr0 = --lrmax`
for the constant lr, so nothing new has to be written for the pod.)

| knob | value | why (all from §2–§5) |
|---|---|---|
| **penalty schedule** | λ = 0 until CE < 0.8 ln p, then λ = 10⁻⁴ | the only knob that makes Ξ = ∞ at every prime without changing the finite-λ point being compared. **[M]** 17/17 escape prediction, and 16/16 fits at p 157 where the archived recipe fit 0/16 |
| **`--escape_frac 0.8`** | | already the bilinear convention; escape is sharp (CE falls from ln p to below 0.2 ln p within a few hundred steps), so the threshold is not delicate |
| **`--wd_warmup 6000`** (cap) | | a safety cap, not a schedule: escape happens at step 33–552 for p ≤ 211 at lr 12.8. Scale it as 6000·(12.8/lr) if a lower escape lr is used |
| **lr 12.8 constant** | not the 0.05→12.8 ramp | **[M]** the ramp costs 8000 steps and buys nothing: the escape-lr sweep over 32× is a null (§4.2), and constant-lr-with-penalty already beats the ramp (quality 0.280 vs 0.246 at p 113) |
| **`--clip 1`** | | the archived bilinear convention; bounds the escape transient. Not tested against no-clip here |
| **init 4** | **do not scale with p** | **[M]** with the warm-up, init only sets *when* escape happens (t_esc ≈ τ\*(p)/(lr·s₀)), not whether. Scaling init with p — which the archive forces, needing 6.4× at p 211 and 12.5× at p 307 — is the wrong fix and makes init and prime collinear, which is what INIT_AUDIT already killed |
| **λ = 10⁻⁴** | **do not scale with p** for the headline cells | **[D]** the finite-λ margin is ≈ p-independent (§4.1), so a fixed λ is what keeps the comparison across primes honest. Report a λ ladder (10⁻⁴ → 10⁻⁵ → 10⁻⁶) instead of picking a p-dependent λ |
| **width 500** | keep | **[M]** the head count is 6–8 at every prime and the constructed ceiling needs 6 neurons per frequency; width enters escape only as √m |
| **steps 40 000** | keep for the headline; 25 000 suffices | **[M]** with escape at step ~300 instead of ~8000, CE is flat by ~6000 at p 157, and the 25 k-step pod cells reproduce the 40 k archived numbers at p 59/71 to ≤ 0.03 in quality |

### 6.2 What NOT to do

* **Do not raise the init scale to rescue large p.** It is a p^{5/2}(m+p) knob (§2.5).
* **Do not lower λ globally to rescue large p.** It moves the finite-λ point, so counts at different primes stop
  being comparable. Lower λ in a *ladder*, reported at every rung, at every prime.
* **Do not report N_.05 as "the count" on the prime axis** (§4.3): it is a tail statistic that grows linearly in
  n_f at fixed head quality. Report K_5pct / K_2pct as the counts, N_.05 as a junk/tail diagnostic.
* **Do not report any count without γ/γ_con(K\*) beside it.** At quality 0.18 the gap between the K = 5 and
  K = 10 rungs of the ladder (≈ 5 %) is unmeasurable.

---

## 7. What the prime-axis figure can honestly claim, and the pod plan

### 7.1 The claim that survives

> **[M] The head count does not grow with the number of available frequencies.** With the fixed recipe at
> λ = 10⁻⁵, 16 seeds per prime (8 at p 307, 4 at p 509), **every prime from 59 to 509 fits at accuracy ≥ 0.999**:
> K_5pct = 7.0, 6.5, 6.0, 7.0, 7.0, 6.0, 7.0, 7.0, 5.0, 4.0 and K_2pct = 10.5, 10.5, 10.5, 10.5, 11.0, 11.0,
> 10.0, 10.5, 6.0, 4.5 at p = 59, 71, 89, 97, 113, 127, 157, 211, 307, 509 — flat, and if anything *falling*,
> while n_f goes **29 → 254**.

Three things must be said alongside it.

1. **The theory's argmax is not identified above p ≈ 113.** Two independent local searches for the coverage-optimal
   support — `rep_pred.json` (300 restarts) and `pax_ceiling.py` (120 restarts) — **disagree** on K\* at p 127
   (7 vs 6), p 157 (6 vs 7) and p 211 (6 vs 8); the rungs within 1 % of the top are {5,6,7} at p 127, {6,7,8} at
   p 211 and {8,9} at p 307. Above p ≈ 113 the figure must show the **plateau band**, not a pin, and the scoring
   rule "|count − K\*| ≤ 1" should be dropped there.
2. **A count is only as meaningful as the solution is close to an optimum**, and that degrades along the axis
   *even with a perfect escape*: at λ 10⁻⁵, quality 0.577 → 0.201 over p 59 → 307 and **0.013 at p 509**. The
   figure needs a **quality track** (γ/γ_con(K\*)) under the count track, and the text needs the sentence
   "the counts at p ≥ 211 are measured on solutions at ≤ 0.32 of the achievable normalised margin".
3. **p ≥ 307 should not be shown at λ = 10⁻⁵.** At p 509 the median margin is +0.113 on a ceiling of ≈ 6.5 —
   the net separates the table by a hair. Either run the λ 10⁻⁶ rung with the 1/(2ηλ) step budget (§5.6b, P4/P5
   in §7.3) or stop the trained panel at p 307 and let Panel A carry 509 and 1009.
4. **N_.05 must be relabelled** as the junk/tail diagnostic (§4.3).

### 7.2 Free way, trained way, or both? — **both, as two panels**

* **Panel A (exact; no training, no escape, no recipe).** The constructed ladder
  γ_con(K, p) = 3 M_K(p)/(9pK)^{3/2} for K = 1…12 at every prime from 59 to 1009+, with the 95 % / 74 % plateau
  bands. This is the theory's actual statement, it is exact, it costs seconds of CPU (`pax_ceiling.py`,
  `pax_ceil_big.py`, `nu4_fixed.py`, `initaudit_ceiling.py`), it has **no escape problem at all**, and it is the
  only way to reach p ≈ 1009 with a clean claim. **Make it the lead panel.** Already computed here:
  K\*_con = 6, 5, 5, 6, 6, 6, 7, 8, 8, **8, 12** at p = 59, 71, 89, 97, 113, 127, 157, 211, 307, **509, 1009** —
  the ladder's own argmax creeps up very slowly (roughly like log p) while n_f goes 29 → 504, which is a
  *stronger* and cleaner statement than anything the trained nets can support.
* **Panel B (trained).** The 16-seed count + quality tracks from the fixed recipe: "GD lands in the predicted
  plateau", with the caveats above.

Panel A alone looks like the theory grading its own homework; Panel B alone invites exactly the reading-3(a)
problem that started this. Both, with the quality track, is the honest figure.

### 7.3 Pod plan — costed on the RTX 5090 actually used here ($0.99/h secure)

**[M] measured wall-clock on that pod**, 16 seeds per cell, 25 k steps, x², m 500, fixed recipe:

| p | 59 | 71 | 89 | 97 | 113 | 127 | 157 | 211 | 307 | 509 (4 seeds, 15 k) |
|---|---|---|---|---|---|---|---|---|---|---|
| s / cell | 47 | 71 | 108 | 132 | 183 | 234 | 378 | 812 | 1027 | 1300 |

The whole λ 10⁻⁴ axis (9 primes, 16 seeds, 25 k) costs **50 min**; the λ 10⁻⁵ axis another **62 min**. So the
cohort below is not a budget question — it is a scheduling question.

| cohort | cells | steps | GPU h | $ |
|---|---|---|---|---|
| **P1** x², p 59…307, 16 seeds, λ 10⁻⁴ and 10⁻⁵ (**already done, in `FINAL_RUNS/PAX/pod`**) | 18 | 25 k | 1.9 | done |
| **P2** the same 18 cells at 40 k steps, for exact comparability with the archived R1 | 18 | 40 k | 3.0 | 3.0 |
| **P3** bilinear `--act prod`, p 59…307, λ 10⁻⁴ and 10⁻⁵ | 18 | 40 k | 3.9 | 3.9 |
| **P4** λ 10⁻⁶ rung at p ≥ 157, **400 k steps** (the 1/(2ηλ) budget of §5.6b), 8 seeds | 8 | 400 k | 4.5 | 4.5 |
| **P5** p 509 and p 1009, both nets, 8 seeds, λ 10⁻⁶, 400 k steps, `--achunk 128` | 4 | 400 k | 6.0 | 6.0 |
| **P6** controls: no-warm-up at 3 primes; ReLU ν 2 dense anchor; L_{2,3} anchor; **width 2000 at p 157** (the one untested hypothesis) | 8 | 25 k | 0.9 | 0.9 |
| **total new** | 56 | | **18.3** | **≈ $18** |

Add ~20 % for the fp64 instrument passes and set-up: **≈ 22 GPU-hours, ≈ $22, one pod, about a day.**
Practical notes measured here: use `--achunk 64` at p ≥ 211 and **`--achunk 128`** at p ≥ 509 — `--achunk 16`
at p 509 was **6× slower** (one backward pass per chunk dominates), which cost 25 minutes of this pod's time.
p 1009 needs a memory-aware fp64 analysis pass (p² = 1.02 × 10⁶ rows).
**The dominant cost is P4/P5, and it is entirely the λ-ladder step budget, not the primes.**

---

## 8. Hypotheses tested and killed (so they are not re-run)

| hypothesis | verdict | evidence |
|---|---|---|
| Pre-escape weight decay eats the net (INIT_AUDIT H6) | **CONFIRMED and quantified** | ds/dτ = −2λ exactly ⟹ τ_∞ = s₀/(2λ); Ξ = τ_∞/τ\* separates 17/17 archived cells; p 211/307 end at ‖θ‖² = 0.0 |
| "Just under-trained at 40 k / lr cap 12.8 too low" | **KILLED** | (i) at p ≥ 211 the *total* progress budget is < what escape needs, so no step count helps; (ii) paired at p 157: 100 k vs 25 k steps makes γ **worse** in 6/8 seeds, counts unchanged |
| fp32 breaks down at large p | **KILLED** | the same net in the same fp32 arithmetic escapes at p = 307 in 1070 steps as soon as λ = 0 |
| Width 500 is the limit once n_f > 50 | **KILLED (as the cause)** | the ceiling needs 6 neurons/frequency (≈ 48 of 500); the head count is 5–7 at every prime up to 307; width enters escape only as √m. *Not* tested: whether width changes the **tail** (the `m 2000` cell was dropped for budget) |
| The escape is too violent / the lr at escape sets the basin | **KILLED** | 32× sweep of the escape lr at Ξ = ∞, 16 seeds, p 113 and 157: quality 0.316/0.314/0.299 and 0.180/0.183/0.140, counts identical |
| Init scale changes the count (RESULTS_REP 3(b)) | **RE-READ as Ξ** | init enters only through s₀ in Ξ; the archived init ladder and the archived prime ladder are one variable (ρ = 0.944), and the residual is the soft-margin effect below |
| N_.05 grows because the tail is un-pruned junk that needs more time | **KILLED** | pruning time constant 1/(2ηλ) = 390 steps vs 40 000-step runs; λ/10 continuations leave N_.05 and the clean fraction exactly unchanged; and at λ 10⁻⁵ N_.05 *rises* |
| N_.05 growth is intrinsic to the committee | **KILLED — it is the instrument** | head counts flat (K_5pct 5–8, K_2pct 6–11) over n_f 29 → 153; N_.05 ≈ K + T·(0.95−f)/(1−f) with T ∝ n_f |
| λ = 10⁻⁴ is p-neutral | **HALF-KILLED** | it is p-neutral for the *finite-λ margin* (margin\* gains exactly ln 10 = 2.30 per decade of λ, p-independent to ±0.08) but **not** for separability: at λ 10⁻⁴, p 211 has median margin −0.009 and p 307 −1.45 |
| The residual quality decay is a soft-margin (finite-λ temperature) effect | **SURVIVES its pre-registered test** | quality rises at every prime at λ 10⁻⁵ and rises *more* at larger p (+10 % at p 59 → +105 % at p 157), flattening the curve; margin/ln(p−1) is the ordering variable |
| "So push λ → 0" | **BOUNDED** | λ 10⁻⁶ at 25 k steps is *worse* (0.385 → 0.238): the penalty's own time constant is 1/(2ηλ) = 39 000 steps. Budget must scale as 1/λ |

## 9. What I ran, and what I could not check

**Local (CPU, ~70 min total, `uv run`, detached, one job at a time, fp32 train / fp64 analyse).**
`pax_tail.py`, `pax_ceiling.py`, `pax_xi.py`, `pax_esclr.py` (analysis only, no training);
`pax_job1.sh` = 11 λ = 0 escape-time cells at p 59…307 including the lr and init controls;
`pax_job2.sh` = the fix at p 157 (2 seeds), p 211 (1 seed), p 113 (2 seeds).

**Pod (one RTX 5090 secure, EUR-IS-2, $0.99/h, ≈ 3.5 GPU-hours ≈ $3.5; `pax_pod.sh`, `pax_pod2.sh`,
`pax_pod4.sh`).** 6 escape-lr cells (16 seeds), 1 no-warm-up control (16 seeds), the λ 10⁻⁴ axis at 9 primes
(16 seeds; 8 at the largest), the λ 10⁻⁵ axis at 9 primes, λ 10⁻⁶ at p 157, the 100 k-step control at p 157, and
p 509. All metrics pulled and verified; **pod terminated after the pull.**

**Could not check / did not do.**
- The **bilinear (gated-chain) net** was not re-run with the fix. Its archived cells already use a warm-up, and
  its quality decays the same way (0.263 → −0.228), so §5.5 predicts the same λ-ladder cure — untested.
- The **width probe** (`m 2000` at p 157) was cut for budget. Width is the one remaining candidate for the
  *tail* growth (it cannot be the cause of the quality decay, which the λ ladder explains).
- τ\* was measured at **init 4** only above p 113, so Ξ for the init-1/2 archived cells carries a ≤ 1.6× bias
  (measured at p 71: τ\* = 4.01/5.98/6.57e4 at init 4/2/1). The bias is in the conservative direction for the
  17/17 claim (it inflates Ξ for the low-init cells, which failed anyway).
- The p^{5/2} part of τ\* is derived from a scaling argument and confirmed to ±13 %; the **(m+p)^{3/2} width
  factor in that derivation is not confirmed** and should be treated as conjecture.
- No seed overlaps the archived REP cohort except by design (seeds 100–115 are the REP seeds, deliberately, so
  the fix is compared to the archive seed-for-seed).
