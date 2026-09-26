# Consequences of the two-sided bound

2026-09-25. Structural statements that follow from the two-sided margin bound. They come from
`astra/LOGARITHMIC_SCALING.md` and two external-agent notes pasted into the session on 09-25. Each
one has a hand-checked proof sketch and a rough numerical check.

- Checks: `consequences/check_consequences.py` (exact vertex enumeration, p = 7–23) and
  `consequences/check_equivariance.py` (float64 full-batch GD).
- Raw output: `consequences/check_out_small.txt`, `check_out_19_23.txt`, `check_equiv_out.txt`.
- **These are rough checks, not certificates.**

## Symbols

- **Task and model.** The task is c = (a+b) mod p, with p an odd prime and h = (p−1)/2 frequencies
  after folding k ≡ −k. The model is the product model with degree ν = 3 (two operands times a readout).
- **Penalty.** The weight penalty has exponent b; b = 2 is plain weight decay. s = ν/b, so s = 3/2 for
  plain weight decay.
- **Coverage.** Frequency k covers the wrong answer at offset D by cov_k(D) = 1 − cos(2πkD/p).
  Reading: how differently frequency k scores the right answer c and the wrong answer c − D.
- **Share programme.** Choose shares y_k ≥ 0 with budget Σ_k y_k^{1/s} ≤ 1. The worst covered offset
  is m(y) = min over D of Σ_k y_k·cov_k(D). Then Γ_s(p) = max of m(y), and Γ_s(S) is the same with
  y supported on the frequency set S.
  - Reading: spend a fixed "regularisation budget" across frequencies so that the worst-covered
    wrong answer is covered as well as possible. The budget is concave, which makes spreading
    expensive.
- **Budget share.** w_j = y_j^{1/s} is frequency j's share of the budget. K is the number of active
  frequencies.
- **Two-sided bound.** ν^{1−s}·B_p·Γ_s(p) ≤ γ\*(p) ≤ B_p·Γ_s(p), where γ\* is the best normalised
  margin over all networks and B_p = 2(2pν)^{−ν/2}.
  - The upper side holds for every network. The lower side is the exact optimum of the junk-free
    class.
  - At ν = 3, b = 2 the two sides differ by √3.

## Status table

| # | Statement | Proof | Rough check (p = 7…23) |
|---|---|---|---|
| 0 | Γ_s = Θ((log p)^{1−s}); K\* = Θ(log p); γ\* = Θ(p^{−ν/2}(log p)^{1−ν/b}) for all networks | hand-checked (astra) | order-only; constants ~185× loose |
| 1 | Plateaus: optimal support and amplitude ratios are piecewise constant in s | hand-checked | ✓ 2–8 switches over s ∈ [1, 4] |
| 2 | Degeneracy: dilates of an optimum are optimal; orbit ≥ h/gcd(h, K) ≥ (p−1)/(2K) | hand-checked | ✓ orbit = h, exactly one orbit of optima |
| 3 | Ablation: m₋ⱼ ≤ m(1−wⱼ)^s | hand-checked | ✓ actual loss 1.15–1.40× the bound |
| 4 | The dense window shrinks: dense optimum needs s − 1 = O(1/log p) | hand-checked | ✓ (s_c−1)·log p falls 1.06 → 0.50 |
| 5 | Certificate: Γ_s(S) < ν^{1−s}Γ_s(p) ⇒ S cannot carry any global optimum | hand-checked | ✓ but weak at small p |
| 6 | Progressions {a,…,Ka}: Γ_s ≤ 1−cos(2πK/p) ≤ 2π²K²/p² | hand-checked | ✓ 0 violations |
| 7 | Bottlenecks: at least K offsets tie for the worst margin | hand-checked | ✓ exactly K at every p |
| 8 | Phase gauge: a continuous family of block phases gives the same function at the same cost | hand-checked | ✓ table change ≤ 4e−15 |
| 9 | Training equivariance under x → ux | hand-checked (exact arithmetic) | ✓ float drift ≤ 2e−15 over 6000 GD steps |

## 1. Amplitude plateaus

**Statement.** At fixed p, rescale any allocation so its worst margin is 1. The optimum is then a
vertex of the polyhedron {v ≥ 0, Av ≥ 1}, where A is the coverage matrix. That polyhedron does not
depend on s. Changing s only changes which vertex is cheapest. So between finitely many switch
values of s:
- the support and the amplitude ratios v_j / v_i stay fixed;
- the budget shares move by the explicit power rule w_j = v_j^{1/s} / Σ_i v_i^{1/s}.

**Proof.** The cost Σ v^{1/s} is concave, and strictly so along any segment. Its minimum over a
pointed polyhedron, which it never decreases along recession directions, is attained only at
vertices.

**Check.** Plateaus were counted after identifying vertices that differ only by relabelling
frequencies k → uk; the raw argmin flips between such tied vertices.

| p | switches over s ∈ [1, 4] | fewest frequencies reached |
|---|---|---|
| 7 | 2 | 1 |
| 11 | 3 | 1 |
| 13 | 4 | 1 |
| 17 | 7 | 1 |
| 19 | 4 | 2 |
| 23 | 8 | 2 |

The support always starts dense. By s = 4 it has shrunk to one frequency at p ≤ 17 and to two at
p = 19 and 23.

Ratios can also jump while the support stays put. At p = 23, the support {7, 10, 11} spans two
plateaus with different ratios: (.87, .99, 1) up to s = 1.95, then (1, .55, .73) up to s = 2.09.
Those are two different vertices with the same support. So test ratio plateaus directly, not just
support changes.

Example at p = 13:

| s range | support | amplitude ratios |
|---|---|---|
| 1.00–1.28 | all 6 | equal |
| 1.29–1.41 | {3, 4, 5, 6} | .44 / .59 / .92 / 1 |
| 1.41–1.68 | {4, 5, 6} | .82 / .66 / 1 |
| 1.68–3.52 | {4, 6} | 1 / 1 |
| ≥ 3.52 | {6} | 1 |

**Experiment.** Run a fine b-sweep at fixed p. Test whether trained amplitude ratios are piecewise
constant with jumps, against the alternative of continuous drift. The exact-family trainers
(`RESULTS_Q.md`) already have a b dial.

## 2. Degeneracy of frequency identity

**Statement.** Multiplying all frequencies by a unit u mod p and folding signs preserves both the
programme value and the cost. The units modulo ±1 act on the h folded frequencies simply
transitively.
- The stabiliser of a K-set has order dividing gcd(h, K).
- So its orbit has at least h/gcd(h, K) ≥ (p−1)/(2K) distinct supports.
- With K\* = Θ(log p), there are Ω(p/log p) optimal supports.

This is slightly sharper than the (p−1)/(2K) given in the note.

**Check.**
- Every dilate of the optimum is optimal.
- The number of distinct optimal supports is exactly h at p = 11–23, and 1 at p = 7, where the
  optimum is dense.
- So there is one orbit of optima and no other ties.

**Meaning.** The objective cannot prefer an absolute frequency label. It can only prefer
*relationships* between frequencies. So compare runs modulo dilation.

## 3. Ablation lower bound

**Statement.** Take an optimal allocation (budget 1, margin m = Γ_s). Deleting frequency j leaves a
margin m₋ⱼ ≤ m(1 − wⱼ)^s. So the relative loss is at least 1 − (1 − wⱼ)^s, which is ≈ s/K for equal
shares.
- For an allocation within (1 − ε) of the optimum, the relative loss is at least
  1 − (1 − wⱼ)^s/(1 − ε).
- The matching upper bound from astra §6 is: relative loss ≤ 2yⱼ/m.

**Proof.** Rescaling the survivors to restore the budget multiplies their margin by (1 − wⱼ)^{−s}.
The result cannot beat the optimum.

**Check.** The bound holds at every p. The actual loss is 1.15–1.40× the lower bound, so the bound
is fairly tight. Examples at p = 17:

| frequency k | share w | actual loss | lower bound | upper bound 2y/m |
|---|---|---|---|---|
| 2 | .43 | .80 | .57 | — |
| 3 | .30 | .67 | .41 | .72 |

**Scope.** This applies to the clean allocation and its constructed network. Ablations in trained
networks also move interference terms.

## 4. The dense window shrinks toward b = ν

**Statement.**
- At each fixed p the dense optimum (all h frequencies) survives for s up to some s_c(p) > 1.
- For any fixed s > 1, the support is Θ(log p) for large p.
- Quantitatively: at an optimum each active share satisfies y_j^{1/s} ≥ (m/2)^{1/(s−1)} (astra §6),
  and the random-frequency construction gives m ≥ ½(8 log p)^{1−s}.
- Together these give s_c − 1 ≤ log 4 / log(h/(8 log p)) = O(1/log p).

This is an upper bound on the window, not its rate.

**Check.**

| p | s_c | (s_c − 1)·log p |
|---|---|---|
| 7 | 1.545 | 1.06 |
| 11 | 1.342 | 0.82 |
| 13 | 1.284 | 0.73 |
| 17 | 1.216 | 0.61 |
| 19 | 1.192 | 0.57 |
| 23 | 1.158 | 0.50 |

- The window shrinks at least as fast as 1/log p at these p. Small p cannot pin the rate.
- Side note: at p = 7, plain weight decay (s = 1.5) is still inside the dense window.

**Corollary (trivially true).** Near-optimal margin cannot fix an exact feature count. Moving a
fraction ε of the budget onto every unused frequency costs at most a factor (1 − ε)^s of margin. So
thresholded counts and amplitude mass must be reported together.

**Experiment.** A 2-D map over (p, b) of where the dense optimum ends.

## 5. Certificates that exclude a frequency set (any network)

**Statement.** Any network whose clean spectrum lives on S has γ ≤ B_p·Γ_s(S). A junk-free network
reaches ν^{1−s}·B_p·Γ_s(p). Hence, if Γ_s(S) < ν^{1−s}·Γ_s(p), then S cannot carry an unrestricted
global optimum. A certified constructive lower bound can replace Γ_s(p) on the right.

**Check (ν = 3, threshold Γ/√3; fraction of sets of each size that survive).**

| p | size 1 | size 2 | size 3 | size ≥ 3 or 4 |
|---|---|---|---|---|
| 11 | 0/5 | all | all | all |
| 13 | 0/6 | all | all | all |
| 17 | 0/8 | 20/28 | all | all |
| 19 | 0/9 | 27/36 | all | all |
| 23 | 0/11 | 22/55 | 154/165 | all from size 4 |

- **Honest limit:** the √3 slack is large at small p, so the certificate only kills very small
  supports there.
- Its reach grows with p because Γ_s(S) for small sets falls faster than Γ_s(p). This is not yet
  measured at the primes used in training (p = 59–127). Computing Γ_s(S) for small S there is cheap.

## 6. Progressions are bad

**Statement.** If S ⊆ {a, 2a, …, Ka} (mod p, folded), then offset δ = a⁻¹ turns every frequency in
S into one of 1…K. So Γ_s(S) ≤ 1 − cos(2πK/p) ≤ 2π²K²/p², and γ ≤ B_p·2π²K²/p² for any network
supported on S. For K = O(log p) that is a vanishing fraction of the optimum.

**Check.**
- There are 0 violations over all (a, K).
- The best short progression (K ≤ max(2, h/3)) reaches this fraction of Γ:

| p | 7 | 11 | 13 | 17 | 19 | 23 |
|---|---|---|---|---|---|---|
| fraction of Γ | .98 | .71 | .61 | .45 | .68 | .54 |

**Meaning.** This answers "why these frequencies *together*". Good sets must not share a badly
covered offset, even though no absolute label is preferred. Connects to the veto set-level law and
additive depletion in memory.

## 7. Bottleneck equalisation

**Statement.** An optimal allocation on K frequencies is a vertex. So at least K linearly
independent offset constraints tie at the minimum margin, counting D ≡ −D as one. In the realising
network these are K equally strong wrong-answer competitors, and their equations fix the amplitude
ratios.

**Check.** The number of tight offsets equals K exactly at every p (K = 3, 2, 3, 3, 3, 4 at
p = 7…23 for s = 1.5).

**Experiment (trained networks).**
1. Recover the clean spectrum.
2. Find the hardest wrong offsets.
3. Test whether K of them nearly equalise.
4. Test whether their coverage equations predict the measured amplitude ratios.

This is a prediction about the geometry of errors, not the count.

## 8. Phase gauge freedom

**Statement.** Take a block on frequency k with input phases φ₁, φ₂ and readout phase φ₃. Shifting
φ₁ by α, φ₂ by β, and φ₃ by α + β:
- leaves every neuron's clean term unchanged (its phase is φ₁ + φ₂ − φ₃);
- rotates all neurons' junk by a common phase, so a cancelling junk sum stays zero.

The logit table and the cost are therefore unchanged.

**Check.** The table changes by at most 4e−15 for random (α, β); the cost change is 0.

**Meaning.** Variation in learned neuron phases need not mean a different computation.

## 9. Training equivariance under dilation (the intervention)

**Statement.** Relabel every token by x → ux mod p. This is an automorphism of the task. Apply it to
the initialisation **and** to the training split, and to any batch order.
- Full-batch GD and coordinate-wise optimisers (Adam, AdamW) are permutation-equivariant.
- So the whole trajectory is relabelled: frequency k becomes fold(u⁻¹k), while loss, norm, accuracy,
  margin and count are identical.

If the init and split distributions are dilation-symmetric, then Pr(j active) = E[K]/h for every j.
This fixes the marginals only, not independence between frequencies.

**Check (p = 13, width 12, float64 full-batch GD, 6000 steps).**

| training data | relative drift from exact relabelling | control: relabelled init, original split | frequency map |
|---|---|---|---|
| full (frac 1.0) | ≤ 2e−15 | — (full split is already symmetric) | {1,2,3,4,5} → {1,2,4,5,6}, as predicted |
| half (frac 0.5) | ≤ 9e−16 | final distance 1.57 | — |

The half-data control shows the split must be relabelled too.

**Caveats.**
- Real runs are long and chaotic; memory records that assembly is chaotic to sub-threshold init
  swaps. Float reordering, GPU nondeterminism and minibatch order could break exact equivariance in
  practice.
- The claim is exact only in exact arithmetic with every source of randomness transformed.
- The marginal-uniformity prediction is not yet tested on existing runs.

**Experiment.** Run the real trainer twice: init plus relabelled init, with the split relabelled.
Predict the second run's frequencies from the first's. This is a controlled intervention, stronger
than seed–circuit correlations.

## Other leads (unchecked, from the 09-25 brainstorm)

- **Confusion map.** A trained net's runner-up answers should sit at the programme's tight offsets.
  This is the empirical side of §7.
- **Margin audit.** Split γ(net)/ceiling into share error, count error, junk and cross-frequency.
- **Norm law with no fitted constants.** P₂ = (m/γ)^{2/3} with m the Adam-eps margin floor. The same
  quantity is r\* in the flight-time law.
- **Optimiser programmes.** Rerun the programme in the geometry each optimiser implicitly uses
  (AdamW is ℓ∞). This may explain AdamW's K\*+1.
- **Mod multiplication.** Via the discrete log it is the Z_{p−1} programme, with divisor frequencies.
- **Not recommended.** Extracting the unrestricted closed form from the scaling proof. The proof
  bounds the interference advantage by a constant but does not show it vanishes.
