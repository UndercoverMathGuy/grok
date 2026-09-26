# SPLIT — the theory / empirical division for the ICLR paper (2026-09-24, deadline 09-25)

Purpose: fix, in one page, what is a theorem, in which class, what the experiments are allowed to claim,
and which sentences of the submitted abstract (ABSTRACT.md, 09-19) no longer match. Nothing below is new
work; every line points at the document that carries it.

## 0. The line

**Theory** characterises the maximum-normalized-margin network *within the symmetric class* (no
same-frequency / same-irrep interference, "junk-free") and shows the feature count K*(task, ν, q) is
decided there by the trade-off between coverage and the regularizer's price per feature.

**Empirics** show that free training with plain weight decay *moves the feature count in the direction
the class optimum predicts* when ν or q is moved, lands within one of K* under a causal count, and
reaches the class optimum exactly only at the q = ν endpoint. Trained networks are **not** in the
symmetric class (every plain-l2 net carries roughly half its logit energy as junk, RESULTS_NS /
THEORY_relu §6), so no empirical sentence may say "attains", "optimal", or "the theorem predicts the
trained network".

Bridge statements (the only places the two columns touch):
1. Symmetrisation bound (THEORY_relu §3, unrestricted): γ(N) ≤ L(N) · γ(N̄) with N̄ junk-free and L
   explicit. This is what lets the theory *bound* a trained net without claiming it is in the class.
2. Objective landscape (LADDER §5.2, FIXED_VS_FREE): a constructed K* net zero-padded into the trained
   parameter space has strictly lower CE + λ‖θ‖² in 6/7 archived cells; at q = ν the trained net matches the
   dense optimum to 3e-6. Trained endpoints are local, not global, minima at q < ν.
3. Wei et al. 2019 Thm 4.1: the λ → 0 global minimiser is the max-margin direction. Dynamics out of scope,
   exactly as in Morwani.

## 1. Theory column — three tiers, do not mix them in the text

| tier | statement | scope | where |
|---|---|---|---|
| **T1 unconditional** | Master bound: γ_q ≤ (Nν)^{-ν/2} Γ(c) for every gated chain on every finite group, every width, q ∈ [2, ν]; Z_p special case | all networks in the family | s5theory §2 (G1), ELEGANT Thm 1 |
| T1 | Own-share / sparsity form of the bound | all networks | GOPT_STATE2, junky-local-max UPDATE |
| T1 | ν = 2 ReLU (no linear layers): dense, unrestricted | all networks | THEORY_relu §2, THEORY_nsum §2 |
| T1 | x^n on n-operand sum, and gated chain at every depth, 2 ≤ q ≤ ν: sparse programme is unrestricted (AM-GM), count = argmax_K M_K/K^{ν/q} | all networks in that family | THEORY_nsum §3, §6 |
| T1 | Complex-weight gated chain on any finite abelian group: exact global optimum, one neuron per character | all networks | s5theory §8 (Thm C) |
| T1 | Symmetrisation: γ(N) ≤ L(N) γ(N̄), L explicit, ≥ 1 | all networks (ReLU + linear layers) | THEORY_relu §3 |
| T1 | Vertex lemma: programme solved exactly by vertex enumeration; count = support of optimal vertex | — | s5theory §3 |
| **T2 symmetric class** | Exact optimum + K* for the ReLU-with-linear-layers family at ν ≥ 3 (decoupled, single-frequency, junk-free) | symmetric class | THEORY_relu §4 |
| T2 | Groups: ceiling attained; each d-dim irrep priced by matmul-tensor decomposition, schoolbook optimal at even ν (κ(d) = d^{ν(1-r)}) | symmetric class | s5theory §4–5 (G2, K2) |
| T2 | S5: exactly ONE irrep (std_sign), no sign patch, even ν; count κ-free at ν = 3, q = 2 | symmetric class | s5theory §6.1 |
| T2 | D_n: count = cyclic count, sign never used | symmetric class | s5theory §6.4 |
| T2 | (n+1)-block optimal in the pure junk-free neuron class (Fekete) | junk-free | LEMMA_ATTEMPT.md |
| **T3 partial / conjecture** | (T): no network beats the symmetric optimum on Z_p, p odd prime | conjecture | GLOBAL_OPT*, GOPT_STATE3 |
| T3 | Proven at p = 3 (all n); explicit endpoint window ν−4/(3(p−1)) ≤ q ≤ ν all p; p = 5 for q ≥ 5/2 | partial | P3_THEOREM.md, astra/EXPLICIT_ENDPOINT.md |
| T3 | Theorem F: pure class, n = 2, q = 2, on supports with certified aiming depth ≥ 0.3476 (incl. p = 113 K* support) | pure class | fable0923/FINDINGS.md |
| T3 | Local theorem: ρ = 0 strict local optimum; junk-free locking beaten at ν 3 (exact grid) | local | THEORY_relu §4–5, astra/LOCAL_THEOREM.md |
| T3 | Negative results that bound the conjecture: junky local max at p = 5 (0.837); (T) FALSE on Z_12, D_3, S4 (1.06–1.62×); cyclic statement only for odd p | — | junky-local-max, s5theory §9 |

Rules of use:
- Every T2 sentence carries the qualifier "among symmetric (interference-free) networks" the first time it appears in each section. Define the class by the invariance, not by the word junk.
- (T) appears once, as a formal Conjecture with T3 listed under it as Propositions and the Z_12 / D_3 / S4 counterexamples stated. That is a programme, not a hole.
- Never: "provably optimal over all networks" (except T1 rows), "the S5 count is proven" (it is the class count), "the theory predicts the S5 sign patch" (the class argmax has none).

## 2. Empirical column — what stays, at what strength

Instrument policy (LADDER §9.9): lead with the loss-ablation count (Nanda protocol), second the spectral
head K_5pct, third the Morwani neuron histogram. N_.05 is a tail statistic, never "the count". Say which
instrument every number comes from.

| keep | claim strength | evidence | where |
|---|---|---|---|
| **E1 degree dial**: at plain weight decay the trained count falls with ν (5 → 4 → 3 direction at p 59/71) | trend | 8 sparse cells, loss-ablation medians within 1 of K* | LADDER §7, RESULTS_NS, RESULTS_PROD |
| **E2 penalty dial**: q from 2 to ν moves the count from K* to dense, monotone | trend + exact endpoint | x² q-sweep 5 → 35 monotone; at q = ν trained = dense optimum to 3e-6 | RESULTS_Q, LADDER §5.2 |
| **E3 collapse**: matched s = ν/q from different (ν, q) gives the same count | trend | matched-s triple 6, 5, 4 | LADDER §4.1b, §7.3 |
| **E4 groups**: S5 8/8 and 32/32 one irrep; A5 Galois pair 32/32; Z_11², Z_120 modes in {K*, K*+1} (3/4 cells) | consistent with T2 argmax; the ladder predicts the margin ORDER of solutions, not which basin a seed picks | s5theory §10 data-consistency, RESULTS_S5, RESULTS_Z |
| **E5 landscape**: trained endpoints are local non-global minima at q < ν; global at q = ν | exact objective comparison | LADDER §5.2, FIXED_VS_FREE |
| **E6 exact evaluation** (training-free): constructed ladders peak at K*, sparse/dense advantage sharpens with ν (1.6×, 4–7×, 12–24×); K*(p) ∼ log p, 69 primes | this is theory illustration, put it in the theory section's figure, not in "experiments" | LADDER §4, §6, PRIME_AXIS |
| **E7 quality gap**: γ_trained / γ_class-optimum = 0.94 at q = ν, 0.39–0.53 at ν 3, 0.10 at ν 4, negative at ν 5 | report as a finding: the higher the degree the further GD finishes from the optimum whose sparsity the theory explains | LADDER §8 F |

| cut (or demoted to appendix) | why |
|---|---|
| Fixed-support / projected training as headline | user decision 09-19: unconventional, width-dependent, fails at ν 4 |
| "normalized margin attained by these networks peaks at the predicted optimum" | trained nets are not in the class; their γ peaks are 0.1–0.5 of the ceiling and the peak location is a band |
| "trained nets are in the class the theorem covers" (THEORY_relu §6 measured L) | L measures decoupling, not junk; every l2 net is ~half junk; say "L is 1.1 on trained nets" only as a bound-tightness remark under bridge 1 |
| K*+1 / AdamW / transformer / Nanda cohorts | different optimizer, partial fraction, not full-table margin; out of the paper |
| Init-scale count dependence | did not survive audit (INIT_AUDIT) |
| Any count from N_.05 | tail statistic |
| Global (T) empirical support beyond "no counterexample at p 17/23/31" | conjecture paragraph only |

Scope sentence for the experiments: full table, full-batch GD, plain l2 weight decay, fp64 margins, one
recipe, 4 seeds per cell, two primes at ν 3 and one at ν 4/5. State it; do not hide it.

## 3. What this does to the submitted abstract

Sentences that no longer match, with the direction of the fix (prose is yours):

1. "Hence we show that an optimal number of features can be calculated for networks based on the task,
   their homogeneity degree and the regularization." → true for the **symmetric class**; add the qualifier
   or say "for networks whose features do not interfere".
2. "we empirically show that the normalized margin attained by these networks peaks at the predicted
   optimum" → false as written (attained margins are 0.1–0.5 of the ceiling, not in the class). Replace with
   the two dials: moving degree or penalty exponent moves the trained count as the optimum moves, and
   the count sits within one of it under a causal count.
3. "matching the result of Morwani et al. for their specific case" → keep and strengthen: at q = ν the
   trained network IS the exact dense optimum (3e-6), the one place theory and training coincide.
4. Add one clause that the abstract currently lacks and reviewers will look for: trained networks with
   plain weight decay carry interference the class optimum does not, sit at strictly higher training
   objective than the class optimum, and the gap widens with degree. That is a finding, and it pre-empts
   "theorem architecture ≠ trained architecture".
5. McCracken / Chughtai citations unchanged. Morwani framing unchanged (their L_{2,ν} result is our q = ν
   endpoint, now exact on every group and not always dense: s5theory §6.5, item 8 — worth one clause).

## 4. Risks after the split

- The reviewer who reads T2 as "assumed the answer". Mitigation: T1 master bound first, then the class as the
  place where the bound is attained, then the symmetrisation bound as the link back to arbitrary nets.
- "Trends only" reads thin. Mitigation: E2's exact endpoint and E5's landscape statement are not trends.
  E4's group data are 32/32-type counts.
- The abstract as submitted promises the attained-margin claim; a reviewer comparing abstract to body will
  notice if it is not rewritten before 09-25.
