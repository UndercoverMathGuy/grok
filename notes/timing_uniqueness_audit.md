# Uniqueness audit: grokking time-prediction / flight-time law (2026-08-21)

Scope: the 5-claim timing paper (spectral delay law, outcompetition mechanism, locked
forecaster, predictability floor, predictability-onset curve) for ICLR 2027 main track.
Method: 103-agent deep-research fanout, 21 primary sources fetched, 105 claims extracted,
25 adversarially verified (19 confirmed / 6 killed), plus two direct source reads by hand.

**Headline: no peer-reviewed pre-cutoff paper scoops any claim.** The pressure is entirely
from a cluster of 2026 arXiv preprints, none refereed as of 2026-08-21. But four independent
groups converged on the log-ratio functional form this year, so a reviewer will read the
*shape* of claim (1) as known regardless of what policy requires.

---

## 0. ICLR policy — what actually binds

Verbatim, [ICLR 2027 Reviewer Guidelines](https://iclr.cc/Conferences/2027/ReviewerGuidelines):

> "We consider papers contemporaneous if they are published within the last two months. That
> means, since our full paper deadline is September 16, if a paper was published (i.e., at a
> peer-reviewed venue) on or after July 17, 2026, authors are not required to compare their
> own work to that paper. **Note that arXiv is not considered a peer-reviewed venue. As such,
> authors are not required to compare to papers solely on arXiv**: they may be excused for not
> knowing about papers not published in peer-reviewed conference proceedings or journals,
> which includes papers exclusively available on arXiv.
>
> While authors are not required to compare to contemporaneous work or unpublished arxiv
> papers, they are strongly encouraged to cite such related work if they are aware of it.
> Reviewers can make authors aware of related contemporaneous work or arxiv papers, but
> **the lack of such comparisons cannot be a basis for rejection.**"

Quote that last sentence in rebuttal if needed. Notes:

- The two-month window is conditioned on **peer-reviewed publication**. arXiv-only papers get a
  *separate, unconditional* excusal with no date attached. Preprints are more excused, not less.
- Page is internally inconsistent: it says "full paper deadline is September 16" while
  [Dates](https://iclr.cc/Conferences/2027/Dates) says Paper Deadline **Sep 25 '26**
  (abstract Sep 18). Page also still carries verbatim ICLR *2026* FAQ text elsewhere, so the
  stated July 17 2026 is probably a stale derivation. Don't build an argument that depends on
  July 17 vs July 25.
- [AC Guidelines](https://iclr.cc/Conferences/2027/AreaChairGuidelines) say **four** months, not
  two: "We consider papers contemporaneous if they are published within the last four months."
  Same 2-vs-4 discrepancy exists in the 2026 guides. The AC writes the meta-review.
- Live venue risk is narrow: **NeurIPS 2026 acceptance is not a threat** (proceedings publish
  ~Dec 2026, after any cutoff). **TMLR is** — it publishes on acceptance, rolling, so a
  May/June 2026 TMLR acceptance becomes binding prior art. 2603.13331 (posted March, revised
  May) has had the most time to clear review. **Re-check before 09-25.**

### Concurrent ICLR submissions

If these authors also submit to ICLR 2027: papers are scored **individually** against the bar,
no comparative ranking, and an unpublished submission is contemporaneous under any reading.
Formally clean. Three leak paths anyway, descending:

1. **Shared reviewers/ACs.** Affinity matching puts semantically similar submissions in front of
   the same people; "grokking delay law, modular arithmetic, weight decay" is a narrow cone. A
   reviewer primed by the other paper expresses it as "the log-ratio form feels like a natural
   consequence of known dynamics" — no citation, nothing to rebut, no policy violated. This is
   the real exposure.
2. **OpenReview is public** (submissions + reviews visible, public discussion period). Anyone can
   post a comment pointing at the arXiv paper. Doesn't force rejection; the AC reads it.
3. **Borderline margin.** Most ICLR decisions are borderline; that is where a vague novelty
   impression decides things.

Author-prestige reasoning ("this guy's preprints have 20 lifetime cites") lowers the
*specific-paper* risk — less likely to be assigned as reviewer, less likely to be known. It does
nothing about the generic-prior risk, because the convergence of four groups is itself evidence
the form reads as natural.

---

## 1. Verdict per claim

| Claim | Nearest prior/concurrent | Status | Verdict |
|---|---|---|---|
| (1) Spectral delay law `t_gen = t_mem + max(log(r_mem/r*)/λ, floor)` | 2603.13331, 2606.30388 Thm 3, 2606.12966, 2606.13753, 2605.18845 — all state `delay = log(ratio)/(rate ∝ η·λ)` in a **norm** coordinate | All arXiv-only, concurrent | **Reframe.** Cannot headline as a new law. Novelty = coordinate + per-seed resolution + invariance breadth of r* + the `max(·,floor)` term |
| (1a) λ scales with lr·wd product | **Nothing binding** — see §3, the Liu et al. attribution is false | — | **Clean.** Matched-product ratio-1.00 has no precedent; nobody runs matched-product cells |
| (1b) r* invariant across primes/widths/eras | 2606.13753 finds its threshold **drifts as p^0.38**; 2606.12966 concedes τ "not separately identifiable"; 2604.13123 fixed-p only | Concurrent, all weaker | **Clean and differentiating.** No competitor shows primes AND widths AND eras |
| (2) Outcompetition, gradient-assisted channel, nonzero rate floor as wd→0 | Nanda et al. **ICLR 2023** states the deletion account; every 2026 competitor sides with it | Nanda = binding peer-reviewed *framing* | **Strongest claim.** Nobody measures absolute growth; two theories predict delay *diverging* as wd→0, which the floor falsifies. Must contest Nanda head-on, not ignore |
| (3) Locked forecaster, ×1.19–1.26 median, R²(log)=0.91, 102 prereg runs, OOD, censoring | 2603.13331 (34.6% MAE, N=60, at t_mem); 2604.13123 (R²=0.543); 2606.12966 (binary); 2605.08237 (detection only); Notsawo 2306.13253 (ICLR 2024 **workshop**) | Concurrent + one workshop paper | **Not scooped.** Needs a baseline comparison table; dispatch Notsawo explicitly |
| (3a) Censoring call 41/46 | 2605.08237 does grok-vs-non-grok at AUROC ~0.93 | Concurrent | **Needs its own differentiation sentence** |
| (4) Predictability floor, σ ≈ 0.035 dex | 2606.12966 reports same-seed chaos *qualitatively* | Concurrent | **Survives as quantification.** No σ, no perturbation sweep, no amplification curve anywhere |
| (5) Predictability-onset curve | Nothing comparable found | — | **Cleanest novelty.** Papers with an earliness axis show skill arriving *late* |

---

## 2. Competitor roster

| ID | What it is | Date | Venue |
|---|---|---|---|
| **2603.13331** | Norm-Separation Delay Law. `T_grok − T_mem = Θ(γ_eff⁻¹ log(‖θ_mem‖²/‖θ_post‖²))`, `γ_eff = η·λ`. **Also defines `R(f_θ) = Σ_{k∉κ}|f̂_θ(k)|²`** (out-of-committee energy), collapse ~0.12→~0.002 at grok, energy-separation R²=0.77 OLS / 0.99 RANSAC. **Also a 3-input delay predictor, 34.6% MAE, N=60, issued at t_mem** | v1 03-05, v2 05-02 | arXiv only |
| 2605.18845 | First-passage prediction of grok delay under AdamW + causal validation, MAPE 17.7% | 05-13 | arXiv only |
| 2606.13753 | "Weight norm sets the grokking timescale", causal delay law; `T_grok ∝ exp(αρ)`, α≈7.5, R²=0.996 over 4 moduli; ~19× delay swing from norm vs ~2× from lr; LayerNorm removes dependence. Threshold `‖W‖_c ∝ p^0.38` | 06-11 | arXiv only |
| 2606.30388 | Thm 3: `τ_{M→G} ≈ (s̄/(ηλ))log(ρ_M⁽⁰⁾/ρ_G⁽⁰⁾) + c_τ/(bλ) + c_τ⁽²⁾/λ²`, mean first-passage on `r_t = ‖θ_t‖₂`, validated Z_127 and S_5. Zero hits for Fourier/spectral/frequency/eigen/DFT/progress-measure across 4,289 lines incl. appendices | v1 06-29, v3 07-08 | arXiv only |
| 2606.12966 | `Δt ≈ (1/λ)log(‖W_mem(t*)‖/τ)`, fit `Δt = C_p/λ`, seed-avg R² 0.89/0.94/0.99 for p=53/97/131. FSD binary precursor, mean lead 1,722 steps, 9/9 configs | 06-11 | arXiv only |
| 2604.13123 | Spectral Entropy Collapse. `ΔT(t) = C₁(H̃(t) − H̃*)^γ + C₂`, R²=0.543. H̃*=0.609 at fixed p=97 | 04-13 | arXiv only |
| 2605.08237 | Koopman/Hankel-DMD operator spectrum of correct-answer log-prob. Detection only, self-disclaimed. AUROC ~0.93, operating point TPR 0.80/FPR 0.50, median lead 1,068 steps, test fold n=17 (5 positives) | 05-07 | arXiv only |
| 2606.18465 | Disputes the critical-norm premise: logit scale mediates ~85%, norm adds 1–2% | 06-16 | arXiv only |
| Nanda et al. 2301.05217 | Progress measures via mech interp. Restricted/excluded loss, Gini of Fourier norms | 2023 | **ICLR 2023** |
| Liu et al. 2205.10343 | Effective theory of representation learning | 2022 | **NeurIPS 2022** |
| Notsawo et al. 2306.13253 | "Predicting Grokking Long Before it Happens" — early-training spectral predictor, qualitative/binary | 2023 | **ICLR 2024 workshop** |

**2603.13331 + 2605.18845 + 2606.13753 are one coordinated program, same lead author
(Truong Xuan Khanh).** That raises the threat: a single visible research line spanning theory,
causal intervention, and a delay predictor. Differentiate against the trio as a unit on five
axes: coordinate (logit-spectrum junk share vs parameter norm), resolution (per-seed vs
seed-averaged), mechanism (outcompetition vs contraction), forecast timing (first 2% vs t_mem),
protocol (locked prereg + OOD + censoring + law-free baseline vs none).

---

## 3. CORRECTION — Liu et al. NeurIPS 2022 has no lr·wd product law

The audit reported Liu et al. as binding peer-reviewed prior art for the "λ scales with lr·wd"
sub-claim. **That is false.** Verified by direct read of arXiv 2205.10343v2 on 2026-08-21.

Their formula (§3): `R(t) = Σᵢ aᵢ v̄ᵢ e^(−λᵢt)`, `t_h = 1/λ₃` ("we call λ₃ the **grokking
rate**"), `n_h = t_h/η = 1/(λ₃η)`.

**λ₃ is the third eigenvalue of their effective-theory Hessian, not weight decay.** The
resemblance of `1/(λ₃η)` to `1/(ηλ)` is a pure symbol collision. Three confirmations:

- The η in their formula is the **embedding** learning rate, and embeddings train with **zero
  weight decay** ("For the 1D embeddings, we use the Adam optimizer ... and zero weight decay";
  "weight decay has only been applied to the decoder and not to the embedding layer"). Their
  rate expression cannot be an lr×wd product.
- λ₃ is driven by **training data fraction**, not hyperparameters. Fig. 5: "Effective theory
  explains the dependence of grokking time on **data size**."
- Their lr/wd content is **qualitative 4-phase diagrams** (comprehension/grokking/memorization/
  confusion) with decoder-lr and decoder-wd as deliberately *independent* axes. A product law
  would collapse that 2D diagram along its anti-diagonal, which the figures do not show. Their
  own ablation: "representation weight decay in general does not affect model performance much."
  The matched-product result is arguably in mild *tension* with their picture.

Provenance of the error: the audit took 2606.30388's ancestry citation at face value, and that
sentence itself hedged — "the scaling τ ∝ 1/(ηλ) observed **or implied** in prior empirical and
mechanistic studies (12)". "Implied" was hardened into "prior art".

**General lesson: any audit finding derived from one paper's characterization of another is
unverified until read at source.** Findings quoted directly from source text (Nanda's deletion
sentence, 2606.12966's chaos sentence, 2603.13331's R(f_θ)) stand.

**What Liu et al. actually is, for related work:**
- `time ~ 1/rate` structure has a peer-reviewed ancestor — but RQI coordinate, toy model, 1D
  embeddings, hard-coded decoder, knob = data fraction. Cite as ancestor of the *shape*; costs
  nothing.
- Fig. 5(b) is grok-step prediction (theory vs real nets), but self-described: "agrees
  **qualitatively** ... showing the **trend**", with acknowledged "quantitative differences".
  Trend-over-data-size, not per-run.
- Their "competition" is encoder-vs-decoder **speed**, not memorizing-vs-generalizing **energy**.
  One sentence distinguishing them, since a reviewer may pattern-match the word.

---

## 4. Peer-reviewed prior art that DOES bind

Only two items, and neither forecasts per-run timing:

**Nanda et al., ICLR 2023 (2301.05217)** — the deletion account, verbatim: "As the completed
Fourier multiplication circuit both solves the task well and has lower weight than the
memorization circuit, weight decay encourages the network to shed the memorized solution."
Claim (2) must contest this head-on.

Also closer on the *coordinate* than a loss-ablation framing implies: restricted loss zeroes all
but the constant and 20 terms for five key frequencies (k ∈ {14,35,41,42,52}); but Figs 3/4 plot
**L2 norms of Fourier components** of W_E, W_L and the 2D-DFT logits, and the Gini coefficient is
computed on those norms — an amplitude/energy-concentration measure over a key-frequency set,
i.e. a real precursor to r. Differentiate against the Gini-of-Fourier-norms measure explicitly,
not only against restricted/excluded loss. Absent from it: no rate law, no r* analogue, no
per-seed timing model.

**Quotable as motivation** (same paper, camera-ready p.879-882): "we lack a general notion of
criticality that would allow us to predict when the phase transition will happen ex ante. Future
work should develop theory and practice in order to apply progress measures to predict the timing
of emergent behavior." Do **not** quote it as if the problem were untouched — the motivation
paragraph must cite Nanda AND dispatch the 2023–2026 attempts (2306.13253, 2408.08944,
2604.13123, 2606.12966, 2605.18845, 2607.23967).

---

## 5. Refuted overlaps — do NOT carry these as threats

Killed during adversarial verification:

- **2604.13123's H̃*=0.609 is not a parallel to r\*** (0-3). Demonstrated across seeds/tasks at
  fixed p=97 only; no primes, no widths, no lr·wd cells.
- **2604.13123's skill curve does not match claim (5)** (0-3). Usable error (<20%) appears only
  ~15,000 steps before T_grok, ~half the budget; mean advance warning 12,370 steps. Skill arrives
  **late**, not before memorization completes.
- **2606.30388's mechanism is the opposite of claim (2)** (0-3) — weight-decay-driven exponential
  norm contraction. Cite adversarially as contrast, not as threat.
- **2603.13331 does not report a matched-product collapse test** (0-3). It verifies inverse
  scaling in wd (R²=0.97) and in lr (R²=0.92) **separately** over 293 runs. Matched η·λ cells
  agreeing to ratio 1.00 remains untested in the literature.

Two findings carried 2-1 votes; soften the prose accordingly:
- Say 2606.12966 "reports only monotone norm/capacity decline and presents no absolute-magnitude
  decomposition", **not** "explicitly the opposite framing".
- Do **not** say 2606.12966 "abandoned per-run prediction" — it never attempted one.

---

## 6. Recommended reframe

Current claims doc leads with (1), which is the weakest thing in the set. Same results, two
different papers:

- **Lead with (1):** review reads "confirms a known law in a new coordinate."
- **Lead with (2)+(4)+(5), demote (1) to the instrument that makes per-seed forecasting
  possible:** review reads "falsifies the standard mechanism and establishes the predictability
  limit."

(2) is the asset — it doesn't just differ from the competitors, it *falsifies* two of them
(2606.12966's `Δt = C/λ` and 2606.30388's correction terms both diverge as wd→0; the rate floor
kills both). Corroboration worth citing rather than hiding: 2606.12966 finds "the fitted rate is
~0.2× the pure-decay value η·λ because gradient updates partially oppose the decay" — that is
partial independent support for the gradient-assisted channel.

Under this reframe the concurrent-submission question mostly stops mattering: no longer competing
on the axis where the overlap lives.

---

## 7. Open items, in order

1. **Read 2603.13331 at source.** Is `R(f_θ)` quantitatively the junk share r — same
   normalisation, same committee definition, same threshold behaviour? Is it among the three
   inputs of the 34.6%-MAE predictor? **This is the only remaining item that could force a
   reframe rather than merely suggest one.** Highest value, ~an afternoon.
2. **Venue re-check before 09-25** on all seven 2026 preprints, TMLR especially. Acceptance at a
   rolling-publication venue before ~July converts concurrent → binding.
3. **Sweep three under-searched angles.** These returned *empty*, not *clean*:
   - ML-wide training-nondeterminism/variance (Summers & Dinneen; Jordan, "On the variance of
     neural network training"; scaling-law prediction noise floors) → claim (4)
   - Learning-curve extrapolation + emergent-ability prediction (Domhan, Klein, LCBench, Mohr &
     van Rijn) → claim (5)
   - Competition literature beyond Nanda/Varma (Merrill sparse-vs-dense, Kumar lazy-to-rich,
     Zhong pizza-vs-clock, Morwani, Doshi, Mohamadi, Barak, Prieto, Tan & Huang, Humayun)
     → claim (2)

   **Claim (2)'s novelty verdict currently rests on absence of evidence in papers actually read.**
   Close this before committing (2) as the spine.
4. **Build the baseline comparison table** for claim (3): the five-competitor ladder with axes
   (error, issue time, OOD holdout, censoring, law-free baseline, n).

Competitor small-n weaknesses available but **do not overstate**: 2605.08237's test fold has 5
positives and its fixed threshold collapses to TPR 0/FPR 0 on a reused-seed split; 2606.12966's
"nine addition configurations (five primes, three seeds)" does not factor and its prime set
differs between PDF and HTML; 2606.13753's "four moduli" collapse excludes p=31 from a five-prime
set. Also note the field disputes the critical-norm premise itself (2606.18465), so leaning on
norm-coordinate competitors as a stable baseline is itself risky.

---

Related: [[flight-time-law]], [[liu2022-no-product-law]], [[uniqueness-audit-2026-08]],
[[iclr-2027-timeline]]. Prior audits: `notes/lambda_uniqueness_audit.md`,
`notes/two_clocks_related_work.md`.
