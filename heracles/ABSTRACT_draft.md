# Abstract draft (ICLR 2027, abstract due 2026-09-18) — v1, 2026-09-17

Working title: **How many Fourier frequencies does a modular-addition network use? Weight decay's representation cost decides.**

Networks trained on modular addition express their outputs through a handful of Fourier frequencies, yet the only
maximum-margin theory of the task (Morwani et al., 2024) predicts that every frequency is used. We resolve this by
pricing features the way weight decay actually prices them. For a one-hidden-layer network of homogeneity degree
ν, the ℓ₂ maximum-margin solution on a_1 + … + a_n ≡ c (mod p) uses K*(p, ν) = argmax_K M_K(p)/K^{ν/2}
frequencies, where M_K is the worst-case coverage of the best K-set of frequencies: dense at ν = 2, sparse for
ν ≥ 3, growing like log p. The formula is an unrestricted theorem for the family in which the degree comes from the
activation (x^n on the n-operand sum: Morwani's quadratic network is the ν = 3 member), and for the ReLU network of
Levine's open problem MAIS-O52, where it gives the dense answer and identifies the sparse observations in the
literature as a degree ≥ 3 phenomenon. The count depends on the norm, not on the optimiser: on Morwani's own
recipe, swapping his L_{2,3} penalty for plain weight decay moves the count from all 35 frequencies to the
predicted 5 at p = 71, while the ν = 2 member (Levine's ReLU network) uses every frequency on the same recipe; the
plain-weight-decay solutions carry half their logit energy in uncancelled cross terms at every degree, so the count is
a property of the clean committee, and it is stable under the weight-decay-to-zero continuation.
The same functional predicts one irreducible representation plus a sign patch on S₅ and the divisor-frequency
committees on Z₁₂₀, matching published reverse-engineering. Across [N] trained networks on [primes 59–127, five
groups, degrees 2–6] the measured count is K* or K* + 1; the +1 is a finite-decay tie-break inside a flat plateau
of the functional, not a second mechanism.

Notes for the final pass (v2, 2026-09-18 01:45Z)
- ν 4 (x³ on a + b + c) is NOT claimed empirically: GD fixed points are hierarchical (N_.05 5–13 vs predicted 4),
  junk 43–55%, margin ≈ 0; see RESULTS_NS.md §4. Abstract says "theorem at every degree, verified at ν 3, dense at ν 2".
- Run census for "[N] trained networks": heracles cohorts (~4300 MLP + 500 transformer + 470 S5 + 160 Z) + QM 40 + NS.
- Claims to keep exact: "unrestricted theorem" only for (i) x^n on the n-sum and (ii) ReLU ν 2; ReLU with extra
  linear layers is a class theorem + measured class membership (THEORY_relu / relu_dense_check).
- ν 5 (x⁴ on the 4-sum): theorem holds, GD did not reach the sparse endpoint in 120k steps (dense junk-heavy
  state, negative margin) — say so in the paper, not in the abstract.
- O52: state the answer as "positive (dense) at the posed degree; K*(p, ν) ~ 4 log p/(ν − 2) above it; the
  max-margin endpoint is the object proven, trajectories are measured"; cite the registry, do not cite our own
  future issue post inside the submission.
- 2026-09-17 q dial (RESULTS_Q.md): abstract sentence to add after "The count depends on the norm": "Moving only the
  penalty exponent q on the same network carries the count continuously from the predicted K* at q = 2 to every
  frequency at q = ν, as argmax_K M_K/K^{ν/q} says; the clean fraction rises with q and reaches 1 only at q = ν."
  Do NOT claim q < 2 (bound reverts to l2 argmax, GD diverges). The ν > 3 statement stays as in RESULTS_NS.
- 2026-09-17 ReLU attempt (THEORY_relu.md): do NOT write "provably optimal over all ReLU networks". Write: "for the
  linear-layer ReLU family the bound holds for every network up to an explicit coupling factor L(N) (symmetrisation
  theorem), measured at 1.10 (median) and 1.78 (max) on 432 trained networks". The sentence "verified dense at ν 2 on
  the exact O52 network" stays unrestricted.
- 2026-09-17 gated chain (RESULTS_PROD.md): abstract may now say "a multiplicative depth-n family for which the count
  is a theorem at every depth and every q in [2, ν], measured in the predicted plateau at ν 3, 4, 5 (on the argmax at
  3 and 5)". Do not claim the argmax at ν 4.

---

# v2 — 2026-09-18 (trend framing; supersedes the v1 text above, notes still apply)

Title options
1. **Weight decay counts the features: a zero-parameter law for how many Fourier frequencies a modular-arithmetic network uses**
2. **How many frequencies? The representation cost of weight decay sets the feature count in group-arithmetic networks**
3. **Sparse or dense? One functional, three dials: coverage, depth and norm in networks that learn modular addition**

Abstract (≈ 235 words)

Networks trained on modular addition compute through a handful of Fourier frequencies, yet the only maximum-margin
theory of the task (Morwani et al., 2024) predicts that every frequency is used, and how many are used is a listed
open problem (MAIS-O52). We show the count is set by how the regulariser prices a feature. For a network of
homogeneity degree ν trained with an ℓ_q-type penalty on per-neuron norms, the margin of any network is bounded by a
functional whose maximiser uses K* = argmax_K M_K / K^{ν/q} frequencies, where M_K is the worst-case coverage of the
best K-set: the numerator is the task, the exponent is the architecture and the norm. We prove the bound without
restricting the network for two families — power activations on the n-operand sum, and a multiplicative depth-n chain
(a CP factorisation of the logit tensor) — for every depth and every 2 ≤ q ≤ ν, show that q < 2 provably reverts to
the q = 2 count, and extend it to ReLU networks with linear layers up to an explicit coupling factor measured at 1.10
(median) on 432 trained networks. With no fitted parameter, the functional predicts each dial separately: moving only
q on one network carries the count monotonically from 5 to all 35 frequencies at p = 71, recovering Morwani's dense
solution as the endpoint q = ν; depth moves it through the predicted plateau at ν = 3, 4, 5 and to dense at ν = 2,
the degree at which O52 is posed; and changing the group moves the numerator, giving one irreducible representation
plus a sign patch on S₅ and divisor-frequency committees on Z₁₂₀. Across more than 6,000 trained networks the count
lies in the functional's plateau; the optimiser chooses within it.

Wording guards for v2
- "bounded by a functional whose maximiser uses K*" — deliberately NOT "the max-margin solution uses K*" (attainment is in [1/4, 1]).
- "predicted plateau at ν = 3, 4, 5" — not "argmax"; ν 4 is off-argmax in both families.
- ReLU ν 2 dense: measured + Morwani q = ν theorem; do not write "unrestricted theorem for ReLU ν 2" (trained nets at 0.17 of bound).
- "more than 6,000": nudial ~4,300 + heracles 1,104 + S5 470 + Z 160 + clean cohort 230 + QM/QT/NS/Q/prod ~200. Verify before submit; fall back to "thousands of".
- S5/Z_120 results are ReLU-class nets (coupling-factor theorem), not exact-family nets — abstract wording is compatible; paper must say so.
