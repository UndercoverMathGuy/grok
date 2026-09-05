# Adversarial review of `notes/adversarial_pack_nudial.md` — 2026-08-24

Reviewer: Claude (hostile-review pass requested by author). Method: re-ran V1–V5,
independently re-audited `notes/nudial_summary.json` (4320 rows), re-derived the
§2 math, checked git provenance, and web-verified the five load-bearing citations
(Morwani 2311.07568, Tian 2410.01779, McCracken 2505.18266, He 2602.16849,
Lyu–Li 1906.05890).

## Verdict

The **dial** (C5's within-family monotone trend), the **norm-contrast point**
(C2/C7), and the **M_K numerics** (C3, C4 modulo nits below) replicate and
survive. The weak core is the **+1/optimality complex (C8/C9)** — internally
inconsistent across ν, instrument-dependent, and resting on comparisons against
demonstrably broken baselines — plus **provenance**: the pack's honesty
apparatus (prereg timestamps, ν_eff "committed before") is unverifiable because
everything entered git in a single commit, and the ν_eff generator script is
absent from the repo.

## What replicates cleanly (verified 2026-08-24)

- V1 exact: M_K = 0.0015/0.1875/0.8446/1.8842/2.8943/3.9060 (K≤6 exhaustive),
  K* = 6/4/4/3 for ν=3..6, sparse/dense 2.0/6.5/24.5/97.2×, free-amp argmax
  identical.
- V2 JSON matches §2.4: K*(ν,p) table verbatim, c_4 = 2.12…3.63, ν=6
  runner-up ratios 0.941 (p=113) / 0.996 (p=211).
- V3 reproduces every §2.5 number I checked (dial medians, full margin ledger,
  E placement 8.84/4.19/3.89 and 9.70/3.85, F 1.34/3.74/4.13/4.30, Spearman
  −1.0/−1.0 at m=512).
- V4 refits to β=9.0, tilt=0.30, E[K] pred/obs as printed. V5: 6.61/6.13,
  148 ckpts, K hist 3:20 4:92 5:33 (+6:2, 7:1 unreported).
- 4320 rows confirmed; trainer is genuinely bias-free (exact ν-homogeneity);
  the γ formula is identical across all cohorts (full p² grid, g/‖θ‖^ν);
  TF32 off; MLX-semantics AdamW as documented.
- The p-independence retraction (C4) is handled honestly; W1–W10 are real and
  mostly well-aimed.

## Findings

### A. Provenance and auditability (most damaging structurally)

**A1 (major).** All nudial files — prereg `nudial/README.md`, results, summary
JSON, and `notes/nu_eff_dial.json` — entered git in **one commit** (9cdd40d,
2026-08-23 21:31), after all runs. The docket's "File committed before any
nudial result existed (git history is the timestamp)" is **false as stated**:
git history contains no ordering evidence whatsoever. Only file mtimes
(nu_eff_dial.json 08-22 17:04 vs summary 08-23 01:42) support the ordering, and
mtimes are not tamper-evident. Same for all three prereg rounds: "written
before the corresponding runs" is unverifiable.

**A2 (major).** No script in the repo computes ν_eff. `nu_eff_dial.json` has no
generator; V5 only reads committed medians. C6's measurement (the scaling test,
checkpoint selection, t-range, logit-vs-margin variants, the `weights_only_*`
vs `all_*` split) is unauditable from the pack.

**A3 (moderate).** "Everything the claims rest on is in this one file" is
overstated. Not regenerable from the pack: pre-grok top-K 88–90%, lesion
0.40×/0.39×, weakest-member 0.33–0.44, AdamW ℓ∞ box 0.13–0.63, transformer
25k→50k frozen, tilt-vs-init-scale (−3.14→+0.12), hand-built quad
constructions (`quad_construct.py` is "scratch", not in the manifest), the
transformer "3:74/4:80" split (source cohort absent), and the "free-amp gain
≤2.5%" numbers (V1 prints argmax only, not gains).

### B. The +1 / optimality complex (C5, C8, C9) — the scientific weak core

**B1 (major) — the +1 is instrument-dependent.** relu-4 m=512: k_eff median
4.25 (= K*), K_gap mode 5 (= K*+1), K_gap median and mean 6.0 (= K*+2); K_eps
median 5.5. The docket picks k_eff for the dial, K_gap-mode for the +1, K_eps
for density. Full histogram: {4:4, 5:43, 6:27, 7:9, 8:6, ≥9:7} — only 4% of
runs sit at the "optimum" 4, and as much mass is at ≥6 as at 5.

**B2 (major) — the +1 is ν-inconsistent.** Against the ansatz K* (6/4/4/3),
trained modal K_gap (12/5/4/4) deviates by **+6 / +1 / 0 / +1**. At ν=5 the
trained mode equals K* exactly — no undercount. "+1" is uniform only against
the cohort-D optima (4/3/3) — but C9 itself establishes D as a weak maximizer
whose outputs say nothing about the true optimum. Using D's K to define "the
optimum count" while discarding D's γ is having it both ways.

**B3 (major) — C9's ordering rests on broken baselines.** Verified per-seed:
**J regressed on 32/32 runs** (median 5.14e-6 → 2.06e-6); 3/32 D relu-4 runs
ended with *negative* margins. So D/H/J margins are lossy lower bounds — fine —
but then "trained = optimum" has no independent support; it is "trained beats
our broken maximizers." The GD comparison: only 18/96 (relu-4) and 8/96
(relu-6) grokked, all at wd=3e-4, 10k epochs (GD margin growth is slow —
plausibly unconverged), and the **best GD run (4.56e-6) already exceeds the
AdamW A-cohort median (4.40e-6)**. "7–20× above margin-ascent, 1.6–2× above
GD" is medians-vs-medians under wildly asymmetric selection. W3 concedes this
"in principle"; it is already visible in the pack's own data.

**B4 (major) — wd is not flat, contradicting a killed-explanation line.**
A relu-4 m=512 by wd: γ = 2.73e-6 / 4.94e-6 / 4.41e-6 (1.8× spread), K_gap
mode 6 (tie) / 5 / 5. relu-6: wd 0.3 mode 5 (vs 4 at wd 1/3), γ 2.67e-8 vs
4.7e-8. The §2.5 line "wd level (flat across 0.3/1/3) — dead" is contradicted
at wd=0.3. Under the theory γ is wd-independent, so this spread is either
non-convergence at 10k epochs (then pooled unconverged runs contaminate the
headline ledger row 4.40e-6) or a genuine dynamics residue beyond the +1 —
either way unaccounted.

**B5 (moderate) — near-ties remove falsifiability at fixed p.** Ansatz margins
at K* vs K*±1 differ by 1.7% (ν=4), ~6% (ν=6), ~8% (ν=5), and the +1 magnitude
is underived (admitted). Hence {K*, K*+1} are both "consistent" and the theory
cannot lose on trained K at p=113. All discriminating power sits in the unrun
§6 predictions (esp. #3, fixed-support γ(K) peak).

**B6 (moderate).** Quad-3 "dense is the *genuine* ℓ2 optimum" is the same
asymmetric-search inference as C9: weak Morwani-style hand constructions vs a
lossy D optimizer (6.1e-6). Both are lower bounds; a stronger sparse
construction could reverse the ordering. The P-N7 kill was accepted honestly,
but "genuine" overstates.

### C. Prereg scoring integrity

**C1.** P-N4 relu-2 bar "≥2": actual 54.0/27.2 = **1.985 < 2** — a failed bar
scored "pass at the edge". Together with relu-6 1.34 > 1.3, "mostly PASS"
contains two failed sub-bars (one acknowledged, one not).

**C2.** The instruments section's own rule — "cells with < half grokked are
reported as a DATA FLOOR, not scored" — is implemented nowhere in
`analyze_nudial.py` and is violated where it matters: E relu(0,2) **42/96
grokked** is load-bearing for P-N9/C8's 2.3× placement claim; relu-2 m128
(29/96) is the denominator of P-N4's relu-2 arm; B is 32/96. The §2.5 dial
table also drops nudial_results.md's "(29/96 grok)" annotation.

**C3.** "Modal K width-invariant 16× for every ν≥4" (C5) is contradicted by
the pack's own table: relu-4 m=2048 mode is **6**, not 5.

**C4.** relu-3's "modal K_gap 12" is an 11/12 **tie** (18/18) at the
instrument's hard ceiling — `committee_from_coeffs` cannot return >12 (docket
says "~13"; actual max 12). The relu-3 cell is right-censored, and it feeds
the V4 fit (E[K] obs 9.18 at ν=3 is computed under censoring).

### D. Math

**D1.** The §2.4/§5.3 second-moment identity is **wrong as printed**:
Σ_{Δ≠0} f² = pK/2 − K², not pK/2 − K (numerically confirmed at K=2,3,4,
p=113: 109/160.5/210 vs claimed 111/166.5/222). The downstream
max f ≥ ½ − 1/(4K) + o(1) likely survives for fixed K (the K² correction is
o(p)), but the printed algebra is false.

**D2.** The boxed pigeonhole bound skips the floor: the proof gives
M_K < K(1−cos(2π/N)) with N = ⌊(p−1)^{1/K}⌋, which is *weaker* than the boxed
K(1−cos(2π p^{−1/K})); it also needs N ≥ 2 (K ≲ log₂ p) or it is vacuous.
Asymptotically immaterial; as a "theorem" line it is sloppy.

**D3.** "Free amplitudes gain ≤2.5% for K ≤ K* at every ν" is false at the
margin: 3.3% at (ν=3, K=2), 2.8% at (ν=4, K=2) — run with the pack's own
`unequal_mass`. Direction fine, constant wrong; and since the gain optimizer
is local ascent, true gains could be larger still.

**D4.** K* ≈ 4log p/(ν−2) is obtained by applying the lattice formula (stated
valid for K ≪ log p) at K ~ log p and beyond; the pack's own math note
measures slopes 1.4 (ν=4) and 0.8 (ν=6) vs predicted 2 and 1. The docket's C4
row states the law with more confidence than its derivation or numerics
support.

**D5.** "M_K p^{2/K} → 22.3/23.5/23.7, monotone" quotes only the p=4001
column; the p-sequences are non-monotone (K=2: 21.2, 21.1, 20.8, 22.5, 22.2,
22.3) and are 40/15-restart lower bounds.

**D6.** V4's hardcoded M table, labeled "V1 output, p=113", matches neither
V1's actual output (K≥7: e.g. 5.831 vs 5.749, 6.658 vs 6.847) nor
kstar_pscale.json — a third, unreferenced provenance. Refitting with true V1
values gives the same β=9.0/tilt=0.30 (no impact), but the label is wrong.

### E. Transformer leg (C6)

**E1 (moderate).** Two incompatible explanations of the transformer's mode-4:
C6 frames the 3/4 split as the K*=3-with-4-at-94% *crossover*; the MLP program
explains ν=6 mode-4 as K*+1. The dial cohort histogram (3:20, 4:92, 5:33,
6:2, 7:1) has 5s outnumbering 3s — better read as mode = K*+1 with a +2 tail
than as a 3/4 near-tie. Pick one story; as written the same observation is
explained twice. The C6 "3:74/4:80" numbers come from a different, unnamed
cohort not in the pack.

**E2 (minor).** ν_eff = 6.6 > 6 is absorbed by a measured-not-modelled softmax
ε ∈ [0.1, 0.6] (W10 acknowledged); with A2 (no generator script) there is no
way to exclude mis-measurement.

### F. Statistics

**F1.** "n=96" cells are 32 seeds × 3 pooled wd (within-seed correlated; B4
shows wd is not exchangeable); two of the 32 seeds collide mod 1000 → 31
distinct data masks. Spearman −1.0 is over 5 (relu) / 3 (quad) cell medians.
No uncertainty is quantified anywhere in the pack. Grok-selection bias in all
non-full-grok cells (quad-5 77/96 and 33/48; E 42/96; I 18/96; relu-2 m128
29/96). C11's fit shares its data with its "predictions" (acknowledged).

### G. Prior art (web-verified today)

- Morwani 2311.07568: **supported verbatim** — Thm 7 is an L_{2,3}-margin
  result; experiments: "For quadratic network, we use a L_{2,3} regularization
  of 1e−4. For ReLU network, we use a L_2 regularization of 1e−4." Note the
  second sentence: Morwani's *ReLU* experiments use plain ℓ2 — worth citing
  precisely, since it means their own paper mixes norms across experiments.
- Tian 2410.01779: **supported** (flags the L_{2,3}-vs-ℓ2 gap; "lower-order
  solutions are preferred if trained with L2 regularization") — but it is
  **Theorem 5** in the current arXiv version, not "Thm 6".
- McCracken 2505.18266: **supported** (O(log n) features; Cor. 4.8).
- Lyu–Li 1906.05890: **supported**, and confirmed to cover neither weight
  decay nor Adam.
- **He 2602.16849: the docket overclaims it.** C7/§4 say He et al. "proves
  network-level dense under wd for quadratic (ν=3)". Direct read: density
  ("Full Diversification", Def. 4.1) is an assumed/observed condition feeding
  gradient-flow dynamics theorems; there is **no max-margin or weight-decay
  optimality theorem**. Cite it as a dense construction/dynamics result, not
  as a proof of dense optima. (The pack's own uniqueness audit §3.2 states
  this correctly; the docket's compression lost it.)
- `notes/uniqueness_sparse_margin.md` is **stale**: its recommended headline
  wording still asserts the retracted p-independence ("nearly independent of
  K and p", "verified p∈[59,211]") and K* = 2.1ν/(ν−2).

### H. Conceptual gaps beyond W1–W10

**H1.** C1's gap is wider than acknowledged: Lyu–Li has no weight decay at
all, and "weight decay changes the norm scale and the timescale, not the
stationary directions" is asserted without proof or citation. With wd the
endpoint has finite norm — a finite-temperature (soft-margin) object. The
"finite-temp objective" candidate for the +1 was killed using D's CE-on-sphere
phase — itself part of the machinery B3 discredits.

**H2.** The instruments are blind to the cross-talk C8 theorizes about: energy
shares are computed inside the cos(kΔ) product subspace only, so (2a−c)-type
energy is invisible to K_gap, K_eps, and k_eff alike. The quad "dense" numbers
describe only the Δ-structured part, and the leak mechanism is never directly
measured in any trained net.

**H3.** E-cohort confound: (0,2) vs (1,1) changes both sharing *and* where
depth sits, and the (0,2) cell barely groks (42/96). "Shared embedding before
the nonlinearity is the switch" is one reading of a two-variable change with
survivor bias.

**H4.** Margin-mode design notes (for W3 follow-up): the softmin uses an
adaptive detached scale (mean |g|) — legitimate, but β=200 over 12,769
constraints leaves the stationary point balancing the worst few percent of
constraints, not the min; and the CE→softmin hard switch at 60% is untuned.
A verified-monotone projected ascent (the pack's own suggestion) is the right
fix; until then §6.6's kill bar is the only real test of C9.

## Ranked priorities for the authors

1. Fix provenance: push preregs/results to a public timestamped record
   (separate commits at minimum; OSF-style hash ideally). Add the ν_eff
   script and the missing generators (A2, A3) to the manifest.
2. Run §6.3 (fixed-support γ(K)) — it is the only experiment that can turn
   the +1 from post-hoc into a result, and it addresses B1–B3, B5
   simultaneously.
3. Re-state C5/C9 with one instrument (or report all three, with the
   k_eff = K* fact stated plainly), fix the width-invariance and wd-flatness
   sentences, and resolve E1 (crossover vs +1 — one story).
4. Run §6.1/6.2 (p=1009) — W6 is the biggest empirical hole in C4.
5. Correct the printed math (D1–D3), the He et al. characterization, the Tian
   theorem number, and refresh the uniqueness-audit wording.
