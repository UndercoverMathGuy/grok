# Optimizer-axis uniqueness & desirability review (2026-08-30)

Fan-out workflow wf_14004610-6f0: 1 Fable scope, 5 Opus search, 14 Sonnet
adversarial verify (of 29 flagged / 40 unique items), 1 Fable condense.
Per-agent raw results: session subagents/workflows/wf_14004610-6f0/journal.jsonl

---

# Uniqueness & Positioning Report: "The Optimizer Prices the Circuit"

**Bottom line up front:** No paper states your law. Nobody computes a max-min coverage optimum M_K(p) over frequency sets, nobody produces an exact integer K*, nobody maps a real hyperparameter (Adam's ε) to committee size, and nobody predicts a committee-size dial. That core is verified open across ~10 targeted searches. But every *premise* your claims stand on now has a published owner, the qualitative version of your headline phenomenon has a public timestamp (July 2026), and one published result appears to directly contradict one of your recorded findings. This is a **go-with-repositioning**, with two mandatory experiments and one mandatory contradiction-resolution before submission.

---

## 1. UNIQUENESS VERDICT, PER COMPONENT

### 1a. The l2 margin law γ̂(K) = M_K(p)/K^(ν/2) — **NOVEL, with three flank obligations**

No prior work optimizes over frequency *sets*, produces an integer argmax K*, or states a zero-parameter plateau law. Dedicated searches for max-min coverage over frequency sets came back empty — the strongest positive uniqueness signal in the whole review.

Nearest neighbors, and the exact daylight:
- **Boursier, Pesme, Dragomir (NeurIPS, 2505.20172)** proves post-memorization training = Riemannian flow minimizing the plain l2 norm, validated on mod-add. This *supports* you — it is your best citation for "why plain l2, not Morwani's L_{2,ν}" — but it owns the "l2 selects the solution" framing. Your contribution must be stated as the **count**, not the selection principle. Check their appendix for any frequency-count remark before finalizing.
- **Musat (2511.01938)** independently observes equal-amplitude "circles" (your democratic committee) from constrained norm-minimization flow on mod-37. Partial pre-emption of the *observation* of democratic amplitudes. Stops exactly where you start: no K, no argmax, no optimizer.
- **Li et al. (AISTATS 2025, 2402.09469)** cements L_{2,ν} as the field-default margin norm for exactly your architectures (including 1L transformers, ν=k+1). This makes your plain-l2-vs-L_{2,ν} distinction *more* load-bearing: the paper needs an explicit head-to-head — same network, two norms, sparse vs dense optimum, enumeration instrument arbitrating. It cannot be a footnote.
- **2608.04451 (Fourier-alignment counterexample)** exhibits positive-probability ReLU neurons that die spectrally flat — the newest quotable objection to any cosine-ansatz instrument. Cheap defense: one measurement showing non-aligned/dead neuron mass contributes negligibly to normalized margin. Expensive to ignore.

### 1b. +1-as-optimizer-geometry — **MECHANISM PREMISE IS PUBLISHED; the structural consequence is yours**

The ε-switch between sign/l∞ geometry and GD/l2 geometry is established prior art: Wang et al. 2021 (already tracked), **Zhang, Zou, Cao (2406.10650)** proving Adam-without-ε → l∞ max-margin and explicitly framing "with ε, asymptotically GD," and **Cattaneo et al. (ICML 2024)** on the small-ε/large-ε regime dichotomy (adversarial read: it's a backward-error/gradient-norm-regularization paper, no margin-max theory of its own — cite as context, not as the owner of your mechanism; the margin lineage traces to Wang 2021, which you already track). **Flavors of Margin (ICLR 2025)** shows the two-phase γ₁-then-γ₂ Adam behavior empirically at ε=1e-8.

What remains unclaimed and verified absent everywhere: (i) the **extinction threshold w\* = (λ/u)^(1/(ν−2))** as a per-direction amplitude threshold — closest analytic cousin is the saddle-escape law τ\* = Θ(ε^(−(r−2))) in 2605.01288 (init scale, not weight decay; no pruning threshold) — derive against it and cite, or a theory reviewer calls the GD leg a relabeling; (ii) the identification of window-closure as **committee-lock time**; (iii) any consequence for **K**. Also distinguish w\* from Verma's global λ_c = 0.0158 regime boundary (2605.20441): per-direction survival vs whether grokking happens at all.

Two hard problems inside this claim:
- **The 18.4-nats window has a numerical doppelgänger.** Prieto et al. (ICLR 2025) and 2605.06152 establish float32 CE-gradient absorption at a logit gap of ~16–17 nats — nearly indistinguishable from log(1/1e-8) = 18.42. Adversarial verification confirms neither paper claims your quantitative threshold or its structural consequence (Prieto's ε remark is an appendix throwaway crediting Thilak 2022), but the confound is real. **Required control, before the ε sweep: replicate the window in float64 with ε swept, show the window tracks ε and not dtype.** Your internal evidence (1e-10 ε → CE 8.3e-10, 8/8 seeds) already contradicts the pure-float32 story — put it in the paper explicitly, because the literature now supplies the alternative explanation ready-made.
- **Vocabulary collision:** "Egalitarian" is owned by an accepted ICLR 2026 grokking paper (EGD, 2510.04930) that also argues per-direction update speeds drive grokking. Rename "egalitarian sign-phase growth" or eat a "this is EGD" reviewer reflex.

### 1c. The ε dial (prediction 4a) — **NOVEL and contrarian; needs armor**

Verified across the entire sweep: no paper connects ε to circuit *size* or *identity*. Every existing ε result is (norm switch in linear settings) / (whether-or-how-fast you grok: 2603.15492, 2605.06152, Thilak) / (generic stability-adaptivity knob: 2211.08583). Adversarial verification **deflated 2603.15492 from high to low**: it measures steps-to-grok and grok success, never K; no margin theory, no committee anywhere in its 15 pages.

But the dial faces a hostile field prior: **Zhang-Zou report ε=1e-8 ≈ ε=0 empirically** (verified verbatim: "no significant distinction… demonstrating that our theoretical setting, which disregards ε, is reasonable"), and **2602.16340** builds "ε negligible throughout the trajectory" into its foundations. "ε doesn't matter" is becoming the default position. If 4a lands, it is genuinely contrarian — good for impact, bad for reviewer priors. Armor required: (i) explain why the structural effect appears where their linear-separable margin diagnostic cannot (no recruitment dynamics, no discrete K in their setting); (ii) **hold batch size fixed and say so** — the ICLR 2026 per-sample-Adam paper (2510.26303) proves batch size independently moves Adam along the l∞↔l2 axis, an uncontrolled confound for ν_sel otherwise; (iii) report grok-success rate at every ε and pre-register that the dial claim is about runs that grok — 2603.15492 reports weight-norm divergence at ε→1e-9/1e-15, so your 1e-12 endpoint may fail via an unrelated instability, and a curve over survivors invites the exact survivorship objection your own adversarial history (T23) flags as this program's recurring failure mode.

### 1d. Lion-dense prediction (4b) — **QUANTITATIVE PREDICTION NOVEL; qualitative direction is taken, twice**

- **2607.20512 ("Active Ingredient in Muon's Grokking," verified by full-text read):** AdamW → concentrated Fourier circuit (IPR 0.085), orthogonalizing optimizers → near-uniform (0.022–0.025), on mod-97, robust across ops and LRs. No K, no law, no margin, no ε, no Lion — confirmed absent. But the bare observation "optimizer norm geometry changes committee density in grokking" has a public timestamp ahead of your submission.
- **NeurIPS 2024 (Adam vs GD one-hidden-layer, verified) + NeurIPS 2025 "Rich and the Simple":** sign-based/adaptive geometry recruits *more* features than GD — the ordering direction of 4b, in a non-grokking Gaussian toy with ε explicitly set to 0 in all theory. No count, no committee, no mod-add.

Cannot claim the ordering as your discovery. The formal prediction (uniform-dense LP optimum by symmetry, exact K) is open, and **Lion Lyapunov (ICLR 2024)** plus the **Spectral Descent/Muon NeurIPS 2025 spotlight** supply the geometry license — derive 4b in their p-norm steepest-descent language. One extra obligation from the **Discrete-Log Clock (2606.17399)**: apparent density can be a basis artifact (mod-mult "dense" spectrum is Gini-0.58-sparse in the character basis). Pre-register a basis-robustness check (participation ratio in more than one natural basis) for any density claim, including your quad/one-hot cross-talk result.

### 1e. Unified law K_mode = argmax M_K/K^(s_opt) — **THE EXPONENT IS THE PAPER; the skeleton has two owners**

- **Flavors of Margin (ICLR 2025, verified by full read):** steepest descent in *any* norm → KKT point of that norm's margin problem, instantiated for GD/CD/signGD/Shampoo/Adam. It is the container your law must be written inside — reviewers in this pool will expect its vocabulary. Verified absent: any feature-count statement (one qualitative l1-sparsity remark), any interpolating exponent, any per-tensor decomposition, any grokking.
- **Jacobs & Montúfar (2605.19458, mirror flow) — the single most dangerous framing threat, not adversarially deflated.** A continuous exponent α interpolates sparse↔dense feature learning in deep homogeneous nets via a sum|a_j|^(α/2) objective — structurally the same shape as your K^(−s_opt) pricing. A hostile reviewer's one-liner: "Claim 3 is mirror-flow implicit bias instantiated on mod-add." Your defenses are real (no count, no discrete mode, no map from a real optimizer's hyperparameter to α, no falsifiable dial) but must be pre-empted explicitly. Strongly consider adopting their α language so s_opt ↔ α is a stated mapping, not apparent reinvention.
- **Stability Annealing (2607.06013, verified deflated to low):** the ε-interpolates-norm-geometries *phrase* has an owner, but it's an annealing *rate* on linear separable data, exact-barrier characterization, zero structural conclusion. Cite and move on.
- **2602.16340 (Adam+Muon hybrid norm):** the "closest analogue of per-tensor ν_sel" characterization was verified **overclaimed** — their hybrid norm is a cross-optimizer composite (Muon on matrices, Adam on the head), not a within-Adam per-tensor ε-crossing-time exponent. Low threat; cite for the per-parameter-group precedent and rebut its "ε negligible" assumption with your 4c measurement.

**One factual landmine outside the five components:** McCracken et al. (NeurIPS 2025, 2505.18266) publishes both K ~ log(modulus) fits (so "K\* grows like log p" cannot be sold as new — yours must be pitched as the exact argmax with an absolute constant vs their expressivity sufficiency bound; they explicitly write "we don't know why it occurs") **and** reports frequency count growing logarithmically in *width* (128 → 8196), which contradicts your K≈4-flat-across-32×-width finding. That is either a counting-convention difference (embedding frequencies vs margin-carrying committee members), a bug in one of the sweeps, or a real disagreement worth a figure. **Resolve before submission** — a reviewer who knows this paper will otherwise treat your width result as refuted.

---

## 2. TOP THREATS, RANKED

| # | Threat | Verified severity | Response |
|---|--------|--------------------|----------|
| 1 | **Jacobs & Montúfar 2605.19458** — continuous-exponent sparse↔dense max-margin skeleton | High (unrebutted) | Cite as the general-theory precedent your s_opt *specializes and makes countable*. Map s_opt ↔ their α explicitly. Your deltas: discrete K, exact argmax, real-hyperparameter dial, mod-add instantiation. |
| 2 | **McCracken 2505.18266** — K~log p published + width-scaling contradiction | High (factual) | Contrast paragraph: expressivity sufficiency + curve fit vs optimizer-indexed exact argmax. Then *resolve the width contradiction empirically* before the abstract goes out. |
| 3 | **2607.20512 (Muon active ingredient)** — qualitative phenomenon timestamped | Medium (verified) | Cite up front, reproduce Fourier-IPR alongside K on your runs, **add Muon to the optimizer sweep**. If s_opt (spectral pricing) predicts Muon's density, their preprint becomes your best external validation instead of your scoop. |
| 4 | **Zhang-Zou 2406.10650 + the "ε negligible" consensus** (2602.16340, Cattaneo, Flavors) | Medium (verified) | Own the switch's provenance (Wang 2021 lineage) in related work. Address ε=1e-8≈ε=0 head-on: their linear setting has no recruitment window, no discrete K, and one tested value. Your 4c per-tensor ε-crossing measurement is the adjudicator none of them attempts. |
| 5 | **float32 confound on 18.4 nats** (Prieto ICLR 2025; 2605.06152) | Medium (gating) | Run the float64 × ε-swept window control FIRST — it is cheap and gates the Claim 2 headline. Surface your internal ε=1e-10 evidence. Distinguish "ε ends learning" (Thilak, published) from "ε sets committee size" (yours) in one explicit sentence. |
| 6 | **NeurIPS 2024/2025 richer-features line** | Medium (verified) | Cite as the closest qualitative precedent for the 4b *ordering*; claim only the quantitative, discrete, mod-add-instantiated prediction. |
| 7 | **2603.15492 ε phase diagram** | Low (deflated) but design-relevant | Cite and distinguish (accessibility/speed vs structure). Design the sweep around its reported low-ε instability; report success rates; pre-register survivor scoping. |
| 8 | **Writing-level collisions:** "egalitarian" (EGD, ICLR 2026); FSD 2606.12966 reporting committee-lock *before* grokking vs your post-grok attention-assembly finding; "optimizer picks the solution" slogan (position paper 2507.12224); Clock/Pizza precedent | Low each, cheap each | Rename the term; address the FSD timing disagreement explicitly (same-task 2026 preprint, easy reviewer point); cite the position paper as motivation ("the concrete instantiation it calls for"); cite Clock/Pizza as config-selects-algorithm precedent. |

Also: triple-check the bibliography. ICLR 2026 desk-rejected all papers with hallucinated references, and yours will lean on several obscure arXiv-only neighbors.

---

## 3. DESIRABILITY FOR ICLR MAIN TRACK

**The census, not vibes:** grokking/mod-add accepts are flat at 3–7 per venue while ICLR's program doubled — grokking's share fell ~3× from 2024 to 2026, all posters at ICLR three years running. Assume you are competing for one of roughly five grokking slots. Meanwhile **optimizer geometry is the live thread**: Flavors of Margin, the Spectral-Descent/Muon NeurIPS 2025 *spotlight*, per-sample Adam at ICLR 2026, a whole ICLR 2026 optimizer cluster, and Muon-at-scale driving demand. The asymmetry — grokking cooling, optimizer geometry hot — is the clearest framing signal in the entire search. **Lead with the optimizer; grokking is the testbed where the price is exactly computable.**

**Tier structure changed:** ICLR 2026 had zero spotlights — poster (~27%) or oral (~1% of submissions). Optimize for clearing the accept bar, not tier-climbing. If tier matters, NeurIPS still has spotlights and is where grokking papers have reached above poster.

**What has cleared the poster ceiling in this area (three papers, three moves):** LLM bridge (NeurIPS 2024 oral), universality (ICML 2025 oral), and **imported framework + explicit refutation of a prior theory** (NeurIPS 2025 spotlight, glass relaxation). Your best-matching asset is the third: the P-N12 kill of the landscape explanation of +1 (12/12 paired) and the Morwani L_{2,ν}-artifact correction. Refutation should lead the contributions, not sit in an appendix. A 60-page rigorous mod-add theory paper did *not* make ICLR 2026 — rigor alone does not carry; a mechanism plus a working *intervention* does (Prieto, ICLR 2025 main).

**Likely objections, in expected order:** (1) "This is mirror-flow / Flavors of Margin on mod-add" — pre-empted per Threat 1; (2) "ε is known to be negligible / this is float32" — controls per Threats 4–5; (3) "O(log p) features is known, and your width claim contradicts McCracken" — Threat 2; (4) "toy task, narrow hyperparameter window" — answer structurally: 10× wd + four-prime invariance early, and the ε-dial/Lion predictions are testable in *any* Adam-trained network, which is your LLM bridge without LLM experiments; (5) survivorship in the ε sweep — pre-registration and success-rate reporting.

**The headline** is the intervention plus the formula: *Adam's ε is a knob that sets how many circuits the network builds, and here is the law that predicts where it moves* — with the zero-parameter plateau number (~97% of ~157 runs within 26% of optimum, four primes, 10× wd) as the exactness credential and the pre-registered out-of-sample prediction (mode 5 at p=1009) as the accept-converter. Given review-quality decay (mean score 4.20, ~21% fully LLM-generated reviews, ACs overriding consensus), put the falsifiable predictions in plain arithmetic in the abstract where a skim cannot miss them, and treat the first paragraph + claims table as the highest-leverage text in the paper.

---

## 4. RECOMMENDATION

**GO — WITH REPOSITIONING.** The core is real and verified unclaimed: the M_K max-min object, the integer argmax, the extinction threshold w\*, the per-tensor ν_sel measurement, the ε-dial-on-K, and the p=1009 extrapolation have no owner. But every surrounding premise is now published, the qualitative phenomenon is timestamped, and the field's default position ("ε negligible") is the opposite of your Claim 2 — which makes the paper contrarian-valuable if the controls hold and fragile if they don't.

**Gating work before submission (in order):**
1. **Float64 × ε-swept window control** (cheap; gates Claim 2's 18.4-nat headline against the 16–17-nat float32 doppelgänger).
2. **Resolve the McCracken width contradiction** (K-flat-in-width vs their log-in-width; likely counting-convention, but it must be a figure, not a hope).
3. **ε-dial sweep with fixed, reported batch size + per-ε grok-success rates + basis-robustness metric; add Muon as a third point on the s_opt ladder.**

**Three highest-leverage framing moves:**
1. **Write it as an optimizer-geometry paper in the field's own language.** Frame the unified law inside Flavors of Margin's steepest-descent-in-a-norm formalism and map s_opt explicitly onto Jacobs–Montúfar's α. The pitch sentence writes itself: *two ICLR 2025 results say the optimizer's norm selects the margin; we show that norm's exponent selects the number of circuits, and Adam's ε moves it.* Grokking appears only as the testbed where K is exactly countable.
2. **Lead the contributions with the refutation and the intervention, not the explanation.** "+1 is optimizer geometry, not landscape (12/12 paired kill)" and "ε is a committee-size dial" are the two moves this venue demonstrably rewards; "we explain the anomaly" is workshop-tier framing for the identical content.
3. **Convert the two nearest preprints into validations.** Reproduce 2607.20512's Fourier-IPR on your runs and have the law predict Muon's density; pre-register 4a/4b/4d as a numbered table with plain-arithmetic outcomes in the abstract. First-claim language about "optimizer-dependent circuit structure" gets dropped everywhere; the first *quantitative, predictive, falsifiable* law is the claim, and it is defensible.