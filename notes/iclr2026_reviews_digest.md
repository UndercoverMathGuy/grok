# ICLR 2026 Grokking/Toy-Model Papers — Reviewer-Intelligence Digest

Condensed from scraped OpenReview forum pages (official reviews, rebuttals, meta-reviews, decisions) for 8 accepted papers. Compiled 2026-08-13.
Caveat: several page dumps were truncated at ~50k chars at capture time; in each case the loss is the tail of one rebuttal thread — all ratings, review bodies, and meta-reviews were captured. Truncations are noted inline where they matter.

All 8 papers were accepted as **Posters**. Ratings are on the ICLR 2026 scale (2/4/6/8).

---

## On The Geometry and Topology of Representations: the Manifolds of Modular Addition
**Forum:** 2olkCiSELH | **Decision:** ICLR 2026 Poster | **Ratings:** 2 (conf 4, yR58), 4 (conf 3, 3muT), 8 (conf 3, 9zL4)

**Reviewer objections:**
- **yR58 (2):** The central claim is overclaimed and mechanistically suspect: "The claim that 'the Pizza and the Clock algorithms are actually the same algorithm' is misleading." and "This papers reads to me like an argument that quick sort and merge sort are actually the same algorithm because they share some similarities." Also: "I strongly suspect that all of these apparant similarities would vanish if we considered the activation vectors projected into the row-space of the unembedding matrix." Plus presentation problems (architecture vs. mechanism conflation, "Attention 0.0/1.0" naming, confusing E notation).
- **3muT (4):** Missing connection to real LLMs (arXiv 2406.03445); insufficient engagement with prior work already showing cos(a+b) from linear superposition (2311.07568, 2402.09469); "The TDA methods (Betti numbers, persistent homology) are established tools; the contribution mainly lies in their application"; MLP-Concat "trivially produces a torus and is rarely used in realistic architectures."
- **9zL4 (8):** Only clarity — sections 4.1 and 5 need more intuition; minor notation/color-scheme nits.

**Rebuttal moves that worked:** Authors conceded the naming/claim ambiguity outright ("we never intended to make this claim") and agreed to tone down "same algorithm"; added two new quantitative tests (CKA and RSM against theorem-predicted ground-truth manifolds, e.g. torus CKA 0.994 for one_embed, clock CKA ~0); added a PCA variance table showing top-2 PCs explain >=95% (killing the "hidden clock signal in the noise" objection); directly answered the unembedding-projection question with new plots; buttressed the simple-neuron assumption with fresh R^2 fits on Nanda's and Zhong's actual released models (first-order R^2 = 0.964–0.998 vs. second-order <= 0.035); wrote a strong Global Rebuttal to the AC framing the stakes for universality. No in-thread score changes shown in the page text.

**Meta-review reasoning:** "Reviews had divergent ratings. However, they agree the questions are important and the approach is interesting. Authors provided an extensive rebuttal which addresses many of the concerns raised in the initial reviews." and "In my view the submission contributes a valuable discussion with strengths outweighing weaknesses. Therefore I recommend accept." The AC also projected counterfactual scores: "Reviewer yR58: 2 -> 4 Reviewer 3muT: 4 -> 4 Reviewer 9zL4: 8 -> 8."

**Transferable lesson:** A 2-4-8 spread can still convert to accept if you concede overclaimed framing immediately, run the exact experiment the hostile reviewer predicts will kill you (and win), and hand the AC a stakes-for-the-field narrative in a global rebuttal.

---

## Explaining Grokking and Information Bottleneck through Neural Collapse Emergence
**Forum:** sLX5P7FTfT | **Decision:** ICLR 2026 Poster | **Ratings:** 6 (conf 2, gZ1w), 4 (conf 3, ud5o), 4 (conf 3, oBWN). (Dump truncated at ~50k chars; final oBWN reply cut off. oBWN's review — last modified after rebuttal — contains "given questions are clarifyable an increase in score is appropriate as this is overall a strong paper", so the displayed 4 may already reflect a raise.)

**Reviewer objections:**
- **gZ1w (6):** Only that validation "is limited to DNN models with two classes... It should be evaluated for the models targeting datasets with more classes, for generalization" — a factual misreading.
- **ud5o (4):** "This unification is heuristic not proven: The entire IB connection rests on a one-sided variance-based upper bound (Thm 3.4) that does not prevent degenerate or trivial solutions." Also: Theorem 4.3's "highly restrictive conditions (pyramidal architecture, smooth activations, squared-loss training) that differ from the experimental setups (ReLU + cross-entropy)", and "replacing the NC2 metric with a condition-number surrogate without justification is sloppy."
- **oBWN (4):** Within-class variance may be the wrong metric — "A variance decrease is also possible by simple scaling... So from this angle the metric appears not very suitable"; novelty overlap with reconstruction-loss phase work (Schneider & Prabhushankar 2024); no normalization-layer analysis; unexplained Assumption A.5.

**Rebuttal moves that worked:** Scope discipline — repeatedly clarifying that NC is a sufficient condition and only Theorem 4.3 needs the strong assumptions, which are lifted verbatim from Jacot et al. ICLR 2025 Oral (new "Remark A.6: Validity of the Assumptions"); showing Prop 3.3 already rules out degenerate collapse (Figure 1 re-annotated with Phase 1/Phase 2); new Appendix A.2.3 + Proposition A.3 on bound tightness; the scale-invariance of Definition 3.1 defusing oBWN's scaling attack; relabeling "NC2 score" to "NC2 (condition-number)" and noting condition-number-1 is stricter than ETF-NC2; correcting gZ1w that experiments already cover 10-class MNIST/CIFAR10 (new Appendix D.1 plots); and a preemptive reconciliation with the seemingly contradictory NeurIPS 2025 "Flatness is Necessary, Neural Collapse is Not" paper (their NCC metric mixes properties; RNC1 isolates the causal one).

**Meta-review reasoning:** "the connection between neural collapse and grokking+IB is, at the best of my knowledge, new, and it will be of interest to the ICLR community. Thus, I am happy to recommend acceptance." The AC flagged their own concern that "the part on the gradient descent dynamics appears incremental with respect to [Jacot et al., 2025]", conceded oBWN's normalization question was unresolved "but one can indeed argue that this is out of scope", and noted "One of the reviewers had already raised the score and I find it possible (if not likely) that a consensus could have been reached towards accepting the paper."

**Transferable lesson:** Marginal scores (6-4-4) survive when every theorem's assumption is traceable to an accepted top-venue precedent, each objection gets a named remark/appendix in the revision, and you preempt the contradicting concurrent paper before a reviewer weaponizes it.

---

## Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test
**Forum:** blfwRondjY | **Decision:** ICLR 2026 Poster | **Ratings:** 8 (conf 3, Tf6m), 4 (conf 4, VRkv), 6 (conf 2→3, TFkw), 4 (conf 4, YR9f)

**Reviewer objections:**
- **Tf6m (8):** Metric construction is hacky — "The conversion of top-k experts per layer into comma-separated strings and computing Levenshtein distance is ad-hoc, since edit distance is sensitive to sequence length and arbitrary thresholding. This distance can also decrease simply due to stronger load-balancing or saturated routers." Also flagged that all generalization is measured after LoRA tuning (finetuning-dynamics confound) and that the NTK bound assumes fixed one-layer routing while OLMoE trains routing jointly.
- **VRkv (4):** Single-model scope — "the paper's findings and conclusions may be unreliable or not yet conclusive"; "Correlation versus causation remains underexplored"; title overclaims relative to one model.
- **TFkw (6):** Wanted a dense-model test of the core claim; "zero-cost" is overstatement; asked for causal interventions and threshold-sensitivity checks.
- **YR9f (4):** Generality across MoE configs; caught a factual overclaim — OLMoE "was trained for 1.3 epochs," not one epoch, and DCLM-Baseline has ~80% duplicates; metrics are MoE-only so "held-out monitoring may still be necessary."

**Rebuttal moves that worked:** Threshold/LoRA-rank/memorization-epsilon ablations (Appendices E.1–E.3); routing-entropy vs edit-distance analysis (r=0.24) rebutting the load-balancing confound; cross-domain edit-distance control ruling out global confounders (E.4); explicit scope concessions ("zero-cost" -> "near-zero-cost," "one-epoch" -> "near single-pass," softened title); citing external papers (C3PO, routing-manifold alignment) as cross-model corroboration; and the decisive move — pretraining a 55M nanoMoE from scratch (~25B tokens) reproducing all dynamics with |r| ~ 0.8–1.0, posted 3 days before the deadline. TFkw raised confidence 2→3; VRkv wrote "I am now considering updating (increasing) my rating" but the discussion was interrupted before scores moved. Authors also wrote a reviewer-by-reviewer summary letter "To the new AC."

**Meta-review reasoning:** "all reviewers acknowledged the clarity and novelty of analyzing grokking at LLM scale... The only shared concern was whether the findings generalize beyond the released OLMoE checkpoints. The authors subsequently trained an additional MoE model (nanoMoE) from scratch and showed highly consistent behaviors, directly addressing that remaining issue." The AC reconstructed intended scores: "Overall inferred post-discussion distribution: approximately 8, 6, 6, 4, which is within acceptance territory."

**Transferable lesson:** When every reviewer converges on one generality objection, a single from-scratch replication at a different scale — plus explicit scope-softening of overclaimed words — can flip an 8/6/4/4 profile to accept, especially if you hand the AC a quote-backed reviewer-by-reviewer summary.

---

## Li_2: A Framework on Dynamics of Feature Emergence and Delayed Generalization
**Forum:** ceIBRhJpUr | **Decision:** ICLR 2026 Poster | **Ratings:** 8 (conf 4, aGxx), 4 (conf 5, LnxJ), 4 (conf 4, sYi6), 4 (conf 2, NQrJ)

**Reviewer objections:**
- **LnxJ (4, conf 5):** "The writing feels very rushed: many symbols are undefined or unexplained, making the paper hard to follow and overly dense"; "the analysis reads like heuristic case-by-case treatment rather than a genuinely unified three-stage dynamics analysis"; Muon Theorem 8 "entirely unclear"; couldn't find the advertised scaling laws.
- **sYi6 (4):** Priority — "the key observation (that there is a lazy and rich learning regime) was already reported in [Kumar et al.]"; the framework "fails to offer more insight on what drives these transitions"; G_F "is not directly set during initialization or optimization" so its link to lr/init/weight-decay was unproven.
- **NQrJ (4):** Found a real math bug — Lemma 1's x_i^T x_j = rho assumption "is consistent with Lemma 1 only for M < 2," contradicting Theorem 2's regime; extensive missing-literature list (Saad-Solla, Ba et al., Dandi et al., Montanari-Urbani): "there is substantial overlap in phenomenology, and I was expecting the authors to put their results in perspective with the literature, not to highlight the difference in assumptions."
- **aGxx (8):** "The theory needs a nonzero weight decay to provably show the three phases, while in practice, weight decay is unnecessary for grokking to occur"; several assumptions unjustified.

**Rebuttal moves that worked:** Fixed the inconsistency by relaxing to |x_i^T x_j - rho| <= eps (NQrJ: "I thank you... for the additional analysis"); a genuinely new result mid-rebuttal — Proposition 1 showing G_F is dominated by Y~Y~^T F at the *initial* phase too, explaining grokking without weight decay from the same framework; new experiments (complex domain, non-Abelian/product-group scaling laws, ridge-regression top layer, Adam-vs-Muon); used the framework to explain four cited no-regularization grokking papers. sYi6 flipped after the hyperparameter-to-G_F derivation: "I now have a positive opinion of the work and would like to increase my score." LnxJ never re-engaged.

**Meta-review reasoning:** "the authors' successfully resolved critical concerns regarding mathematical consistency"; "the rebuttal clarified that the backpropagated gradient G_F provides sufficient signal to trigger feature learning during the initial lazy learning phase even without explicit regularization"; despite NQrJ's remaining literature complaint, "the consensus is that the proposed Li_2 framework offers new, first-principles insight into feature emergence and scaling laws, warranting acceptance."

**Transferable lesson:** A theory paper can survive an 8/4/4/4 spread if the rebuttal fixes the concrete math flaw *and* extends the theory to cover the "but it happens without weight decay" counterexample — reviewers reward new derivations far more than defensive prose; the one unmoved reviewer was the one whose complaint (contextualization vs. prior phenomenology) got argued with instead of accommodated.

---

## Decoupling Dynamical Richness from Representation Learning: Towards Practical Measurement
**Forum:** 7Mbz5uSf2J | **Decision:** ICLR 2026 Poster | **Ratings:** 8 (conf 2, GFrJ), 8 (conf 3, cwga), 4→6 (conf 3, Czxk), 4 (conf 3, ztAm — did not respond post-rebuttal)

**Reviewer objections:**
- **ztAm (4):** "The current formulation only applies to orthogonal and isotropic target functions. While this covers many classification tasks, the restriction is significant. The authors acknowledge this but don't provide a clear path to generalization." Also: "the experiments conducted are of small scale and somewhat artificial," and "The bra-ket notation and operator formalism, while mathematically precise, may limit accessibility for a broader audience."
- **Czxk (4→6):** "DLR inspects only the final-layer features; rich dynamics might manifest earlier and be attenuated by a constrained head." Plus insufficient technical background for readers and reliance on "orthogonal/isotropic targets and supervised, one-hot settings; this narrows immediate applicability (e.g., class imbalance, multilabel, regression, self-supervised)."
- **cwga (8):** Skepticism of one novelty claim: "I'm a bit skeptical of the statement that this correlation has not been observed or studied" (citing Saxe 2014, Atanasov 2022).
- **GFrJ (8):** Only nitpicks (naming the metric, clarifying edge cases).

**Rebuttal moves that worked:** Authors converted every recurring objection into a new appendix: D.4 (why isotropic targets are standard in NC literature and anisotropy is an open problem field-wide), I (intermediate-layer experiments showing the last layer has the strongest low-rank bias — turning the "last-layer-only" complaint into supporting evidence), J (Nystrom approximation accuracy/sample-size analysis). They defended small-scale experiments by re-anchoring the comparison class: comparable to NC papers, larger than lazy/rich-dynamics theory papers. Czxk raised 4→6 ("the overall contribution is non-trivial") while explicitly refusing an 8 because "the scope of validated settings is still narrow." Both 8s reaffirmed; GFrJ wrote a detailed acknowledgement endorsing the limitation-handling.

**Meta-review reasoning:** "Most initial concerns have been addressed during the rebuttal phase. While the limitation to isotropic targets remains a point of discussion, the reviewers generally agree that the paper makes a solid contribution, and view the proposed measure as a valuable tool for characterizing feature evolution and distinguishing between lazy and rich learning regimes." The AC also noted the intermediate-layer experiments "largely resolves the concern regarding layer-wise generalizability" and that the isotropy limitation "is standard in existing literature."

**Transferable lesson:** Pre-declare your scope limits in the main text, then rebut by citing subfield norms ("our scale matches NC/lazy-rich literature") and shipping targeted appendices that turn each limitation into a measured trade-off — an unresponsive 4 gets discounted by the AC if everyone else moves.

---

## Egalitarian Gradient Descent: A Simple Approach to Accelerated Grokking
**Forum:** wCnHeql3ow | **Decision:** ICLR 2026 Poster | **Ratings:** 6 (conf 5, QrdE), 6 (conf 4, 6NV3), 8 (conf 3, vvi7), 4 (conf 3, yxV7). No score changes visible in the text.

**Reviewer objections:**
- **QrdE (6, conf 5):** "Theoretical justification only exists for toy examples"; "The proposed algorithm seems to be similar to RMSProp"; wanted the FIM/NGD connection strengthened and more tasks beyond modular arithmetic (Omnigrok settings).
- **6NV3 (6):** "The theoretical example is somehow biased and too low-dimensional, which makes it hard to explain practical results"; "There is no theoretical convergence analysis on the proposed EGD algorithm"; and the theory/experiment mismatch — "Can you explain why you choose a different training and test data distribution in Section 3 (fig 5)? This is not a standard machine learning setup."
- **yxV7 (4):** "there is no benchmark or ablation quantifying the tradeoff, or on what network scales/width SVD ceases to be efficient. This undermines claims about practical usability in modern architectures"; "the paper implies EGD could be universally beneficial and 'plug-and-play' for any neural network. The only evidence is for the tightly controlled algorithmic family, and the real computational cost is glossed over."
- **vvi7 (8):** Figure readability, no code, MNIST/IMDB benchmarks missing, EGD turn-off criterion "Does this not introduce a hyperparameter?"

**Rebuttal moves that worked:** A massive appendix blitz (C–G, A.3): randomized-SVD ablations with wall-clock tables (RSVD-EGD beats vanilla SGD in wall-clock), head-to-head Grokfast comparison on MNIST and a transformer arithmetic task, EGD bolted onto Adam/RAdam/RMSprop for non-stationarity, and solution-subspace geometry analysis. The decisive theory move: for 6NV3 they derived a new Theorem 2 extending the toy model to identical train/test distributions — delivered hours before the deadline — eliminating the covariate-shift inconsistency. vvi7 confirmed satisfaction ("I have no further questions"). QrdE and yxV7 never responded; authors openly conceded convergence analysis and nonlinear theory as future work, and notably left "overstated generality" unanswered (the AC flagged this: "the authors have not made responses to these issues").

**Meta-review reasoning:** The AC's reasoning is an issue ledger, not a narrative — six concerns listed as "well addressed" (MNIST experiments, RMSProp comparison, theory/experiment consistency, ablations, failure modes, extra datasets) against five "still remaining" (all theory-depth items). Acceptance followed despite a static 6/6/8/4 spread: "Reviewer vvi7 may not change the score since the current score is very high (8), and all the comments are well addressed."

**Transferable lesson:** For a grokking-methods paper, empirical objections are fully repairable in rebuttal (appendices + one strong baseline + wall-clock numbers) while theory-depth objections can be safely deferred to future work — but answer every objection on the record, because the AC tallies unanswered items verbatim.

---

## On the Convergence Behavior of Preconditioned Gradient Descent Toward the Rich Learning Regime
**Forum:** CXlsqTAf1E | **Decision:** ICLR 2026 Poster | **Ratings:** 6 (conf 5, NxWh), 2 (conf 4, oNGp), 6 (conf 2, Gxam), 6 (conf 2, mDac)

**Reviewer objections:**
- **NxWh (6, conf 5):** The grokking link is empirical-only — "linking this with the main claim... only involves empirical experiments as in Figure 5. I believe additional theoretical bridge between this gap will strengthen the claim." Also: no unifying criterion for a "good precondition"; Figures 1/3/7 "quite messy"; how does this differ from Grokfast?
- **oNGp (2, conf 4):** "The paper's first main claim... that Gauss-Newton preconditioning ameliorates the spectral bias in optimization, is almost immediate from prior works." Also attacked coherence: "What does it mean to 'explore the NTK subspace faster'? ... How is this different from just saying that PGD trains faster?" and "This is a pretty sweeping claim that is being made based on a single experimental setting." Claimed the second and third claims contradict each other, and that the two framing papers (Kumar '24, Zhou '24) give incompatible grokking explanations.
- **Gxam (6, conf 2):** "Computational Cost and Scalability remain issues"; no deep analysis of rich-regime dynamics; "A experiment on Transformer will be very strong."
- **mDac (6, conf 2):** "The novelty is limited... mainly a clarified perspective on known conditioning effects"; experiments small; why MSE not cross-entropy on MNIST; "how can we know when the training leaves the NTK regime?"

**Rebuttal moves that worked:** Added the classical transformer modular-addition grokking experiment (page 9 + Appendix D) — directly answering Gxam and strengthening generality; revised Figure 7 to show continuing LM underperforms switching to AdamW (causal support for the "PGD helps only in lazy regime" claim); explicitly scoped PGD to GN/LM (lines 59–61); Appendix C on MSE-vs-CE replicating Omnigrok; against oNGp, demanded references and used a linear-solver analogy (CG vs GD) to defend "uniform convergence != faster convergence." Honest concessions on open problems (NTK-exit detection, rich-regime theory). No reviewer score changes visible; no post-rebuttal reviewer replies captured.

**Meta-review reasoning:** "three out of 4 reviewers gave score 6. Reviewer oNGp voted for rejection (score 2) who raised the main concern that the results are almost immediate from prior works. After reading the response from authors, I tend to agree with the authors that similar results are not in prior works since the reviewer did not provide any evidence to support the claim." Also: "the authors did a good job in responding to them."

**Transferable lesson:** A lone hostile "this is already known" reviewer loses if they cite nothing — respond by demanding references, adding the canonical transformer/modular-addition experiment, and conceding open theory honestly; three 6s plus a well-answered 2 is an accept.

---

## In-Context Algebra
**Forum:** J2peqXPQbB | **Decision:** ICLR 2026 Poster | **Ratings:** 4 (conf 3, Cayz), 6 (conf 4, XJFb), 8 (conf 4, sQQh), 2 (conf 4, 9LQ5) as captured; the meta-review states 9LQ5 raised their rating, but the updated score isn't in the page text.

**Reviewer objections:**
- **Cayz (4):** "This task is novel, however it strikes me as contrived and very toy. I struggle to see (a) how these findings will generalize to more interesting settings, (b) what this tells us about models that we did not already know"; also an over-claim that the PCA direction "causally controls identity recognition."
- **XJFb (6):** "The causal analysis fails to identify a concrete mechanism for associative composition, arguably the most conceptually interesting of the five." Post-rebuttal kill-shot: "some of the mechanisms you identify are 'heuristics' that work for the data distribution rather than reflecting true learning of symbolic computation," and coverage-within-3% "is not fully satisfactory because... you could construct a test distribution that oversamples a certain subset of inputs for which the coverage is potentially much smaller." Maintained 6.
- **sQQh (8):** Single architecture/size only; "limited to no discussion of ways in which the results might extend beyond this abstract problem setting."
- **9LQ5 (2):** The "challenge geometric embeddings" framing "may be slightly marginal for a conference" — should be a complement, not a challenge; "Omission of Associativity... The paper would be much stronger if it included a causal analysis of this failure"; single toy model, no ablations. (Authors' full response to 9LQ5 is cut off in the dump.)

**Rebuttal moves that worked:** Trained 16 additional models sweeping layers/heads/dims/data mix (Appendix B) — killing the single-architecture objection; added associativity coverage to Figure 3 rather than hiding the weak mechanism (5 mechanisms now cover 90.4% train / 84.7% held-out vs model AUC 92.4/87.3); expanded closure-cancellation detail (Appendix D); softened the PCA causal claim verbatim; reframed as complementary to Fourier/geometric prior work; corrected Figure 2c (semigroups generalize, quasigroups/magmas don't). Per the meta-review, 9LQ5 raised their score; XJFb explicitly held at 6.

**Meta-review reasoning:** "Reviewers find the paper technically sound and carefully executed, but raise consistent concerns about the limited scope and contribution. The task is seen as narrow and toy-like... After the clarifications and added experiments in the rebuttal, concerns are generally addressed. Therefore, this paper could be accepted by ICLR." And: "Reviewer 9LQ5 tends to raise his/her rating to be more positive. Others may maintain their positive ratings."

**Transferable lesson:** "Toy task" objections are survivable when soundness is unimpeachable — buy acceptance with seed/architecture sweeps, quantified mechanism-coverage numbers, honest display of the mechanism you can't explain, and verbatim softening of any over-claim a reviewer flags.

---

# Cross-Paper Synthesis

Corpus: 8 accepted posters, 28 official reviews. Every paper had at least one 4-or-below review; three papers carried a 2. All converted to accept.

## Recurring objection taxonomy (count = reviews raising it)

1. **Toy scope / single-setting generality** (~12 reviews, 7/8 papers). "Does this hold beyond one model / one prime / one architecture / modular arithmetic?" The single most common objection, and the most repairable: it was killed in-rebuttal every time it was addressed with new training runs (nanoMoE from scratch, 16-model architecture sweep, transformer modular-addition, Omnigrok-style extra datasets).
2. **Theory gaps: toy-only theory, restrictive assumptions, theory/experiment mismatch** (~10 reviews, 5 papers). "Highly restrictive conditions... that differ from the experimental setups"; "heuristic not proven"; "no theoretical convergence analysis"; the weight-decay-needed counterexample (Li_2); the empirical-only grokking bridge (Preconditioned). Repairable when a new derivation is delivered mid-rebuttal; safely deferrable to future work otherwise — no paper was rejected over deferred theory.
3. **Novelty / "already known" / missing prior work** (~8 reviews, 6 papers). Ranged from cited near-misses (Kumar et al., Saxe/Atanasov, Schneider & Prabhushankar) to the evidence-free "almost immediate from prior works" (oNGp, score 2) — which the AC explicitly discounted because "the reviewer did not provide any evidence to support the claim." Uncited novelty attacks lose; cited ones must be answered with a precise delta.
4. **Presentation / clarity / density** (~7 reviews, 6 papers). "Rushed" writing, undefined symbols, messy figures, bra-ket formalism. Never decision-relevant on its own but amplified low scores when combined with (1) or (2) — LnxJ (4, conf 5) never re-engaged.
5. **Overclaiming / framing** (~6 reviews, 5 papers). "Same algorithm" (Manifolds), "zero-cost"/title (LLM pretraining), "plug-and-play" universality (EGD), causal PCA claim and "challenge" framing (In-Context Algebra). Uniformly fixed by immediate verbatim concession + softened wording; never fatal once conceded. Arguing back was never the winning move here.
6. **Ad-hoc metric / unjustified methodological choices** (~5 reviews, 4 papers). Levenshtein-on-expert-strings, condition-number surrogate "sloppy," variance metric scaling attack, last-layer-only measurement. Fixed by controls that rule out the confound (routing-entropy r=0.24; scale-invariance of the definition; intermediate-layer experiments).
7. **Correlation vs causation / missing causal interventions** (~5 reviews, 3 papers). "Correlation versus causation remains underexplored"; requests for interventions/ablations. Partially repairable (switch-optimizer causal figure, activation patching); the deepest version — XJFb's "heuristics for the data distribution rather than true symbolic computation" — was acknowledged, not solved, and cost nothing beyond a held 6.
8. **Computational cost / scalability** (~3 reviews, 2 papers — both methods papers). Answered with wall-clock tables and randomized-SVD ablations.

## Rebuttal moves that demonstrably moved scores or the AC

- **New training runs at a different scale/architecture** — the single highest-yield move (nanoMoE; 16-model sweep; transformer grokking experiment; MNIST/Grokfast head-to-heads). Directly credited in three meta-reviews.
- **New theory delivered mid-rebuttal** (Li_2's Prop 1 removing the weight-decay requirement; EGD's Theorem 2 fixing the distribution mismatch; NC's Prop A.3 on bound tightness). Flipped sYi6, satisfied 6NV3. Reviewers reward derivations, not defensive prose.
- **Immediate concession + verbatim wording changes** for every overclaim ("we never intended to make this claim"; "zero-cost" -> "near-zero-cost"). Cheap, always worked.
- **Running the hostile reviewer's predicted-fatal experiment and winning** (Manifolds' unembedding projection; the coverage numbers in In-Context Algebra).
- **One named appendix/remark per objection** so the AC can tally items as "addressed" (Richness D.4/I/J; EGD C–G; NC Remark A.6). ACs literally kept ledgers.
- **Anchoring assumptions to accepted precedent** ("lifted verbatim from Jacot et al. ICLR 2025 Oral"; "isotropic targets are standard in the NC literature").
- **Preempting the contradicting concurrent paper** before a reviewer finds it (NC vs "Flatness is Necessary, Neural Collapse is Not").
- **AC-directed global rebuttal / reviewer-by-reviewer summary letter** — used by the two papers with the worst initial spreads (2-4-8 and 8-4-6-4), both cited favorably by ACs.
- **Demanding citations** from an evidence-free "already known" attack — the AC sided with the authors explicitly.
- What did NOT work: arguing with a contextualization complaint instead of accommodating it (NQrJ stayed at 4); leaving objections unanswered (EGD's "overstated generality" — the AC listed it verbatim as "still remaining").

## What ACs said they valued

- **Question importance over score consensus:** "they agree the questions are important and the approach is interesting" (Manifolds); willingness to override 2s and 4s, even reconstructing counterfactual scores ("2 -> 4"; "approximately 8, 6, 6, 4, which is within acceptance territory").
- **Novelty of the connection to the community:** "the connection between neural collapse and grokking+IB is... new, and it will be of interest to the ICLR community"; "new, first-principles insight."
- **Rebuttal responsiveness as evidence:** every meta-review credited the rebuttal itself ("extensive rebuttal which addresses many of the concerns"; "the authors did a good job in responding"); conversely they tallied unanswered items and unresponsive reviewers, discounting silent or evidence-free negatives.
- **Technical soundness as the floor:** "technically sound and carefully executed" made "narrow and toy-like" survivable; the acknowledged-but-out-of-scope limitation ("one can indeed argue that this is out of scope") was tolerated when declared honestly.

## Bottom line for a grokking/toy submission

Expect the 4s to come from (a) "toy/one-setting" and (b) "theory too restrictive / already known." Budget the rebuttal window for one from-scratch replication in a second setting and, if a theory claim is attacked, one new derivation — those are the only two moves that flipped scores in this corpus. Concede every wording overclaim on day one, give each objection a named appendix, and close with a global letter to the AC framing why the question matters; ACs in this pool accepted on question-importance plus rebuttal-responsiveness, not on score averages.
