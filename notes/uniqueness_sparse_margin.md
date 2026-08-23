# Uniqueness Audit: L2-vs-L_{2,nu} Margin Gap → Fourier-Sparse Modular Addition

**Claim date:** 2026-08-22
**Scope:** headline (L2 weight decay makes the max-margin solution for modular addition sparse, with K*(nu) = 2.1·nu/(nu-2)) + secondary (removal-only/addition-blocked committee-size asymmetry)

---

## 1. Verdict

The headline is **novel as a quantitative, closed-form result**, but it is **not novel as a qualitative observation**, and the claim to be first to notice *any* L2-vs-L_{2,nu} mismatch in Morwani et al.'s Theorem 7 is **false** — Tian (2410.01779) already states that gap in almost the same words ("they use the max-margin framework with a special regularization (L2,3 norm) rather than L2 loss") and already proves, for the identical quadratic-MLP/modular-addition setting, that plain-L2 weight decay favors lower-"order" (in his construction, proportionally fewer-frequency) solutions. What survives as the actual novel kernel, unclaimed anywhere in 20+ candidate sources across nine search angles, is: **(a)** the explicit per-frequency $\ell_{2/\nu}$ quasi-ball reduction applied specifically to single-frequency Fourier circuits of modular addition (not hidden-node "order," not neuron-count, not path-norm); **(b)** the closed-form $K^*(\nu) = 2.1\nu/(\nu-2)$ formula and its cross-architecture instantiation (dense at $\nu=2$, $K^*=6$ at $\nu=3$, $K^*=4$ at $\nu=4$, $K^*=3$ at $\nu=6$) reconciling the long-noted MLP-dense-vs-transformer-sparse discrepancy under one mechanism; **(c)** the Dirichlet-sidelobe near-invariant $M_K \approx K - 2.1$, nearly independent of $K$ and $p$ over $p \in [59,211]$, and its role as the number-theoretic engine of $K^*$; and **(d)** the removal-only/addition-blocked committee asymmetry tied to an escape-threshold argument for degree-$\geq 3$ circuits. None of these four were found stated, derived, or proved anywhere in the literature searched (nine angles, ~200 queries, direct full-text reads of the ~20 closest candidates). The general algebraic primitive underlying step (a) — that $\ell_2$ weight decay on a degree-$D$ multiplicatively/Hadamard-factored or homogeneous parameterization is exactly an $\ell_{2/D}$ quasi-norm on the collapsed quantity, sparsity-favoring for $D\ge3$ — **is** old and well-established outside the modular-addition/grokking context (Hoff 2017, Woodworth et al. 2020, Neyshabur–Tomioka–Srebro 2015, Kolb et al. 2025's Deep Weight Factorization); this must be credited as the mathematical primitive being specialized, not claimed as new math. **Recommended framing:** the paper should not claim to be first to flag an L2-vs-L_{2,nu} discrepancy in Morwani (Tian got there first, informally); it should claim to be first to *resolve it quantitatively* with an exact, architecture-general, number-theoretic formula that explains the MLP-vs-transformer frequency-count discrepancy and reproduces Tian's and He et al.'s qualitative/dense observations as boundary cases.

---

## 2. Confirmed kills

**None.** Every candidate initially flagged `threat: kill` by the search process was downgraded on direct full-text verification — either to `CONFIRMED_NEAR_MISS` (real, citable overlap on a narrower point, but none of the four kill conditions fully satisfied) or to `DEFLATED` (thematic-only overlap). No prior work was found that:
(i) derives/states the l2-weight-decay max-margin solution for modular addition is Fourier-sparse *with a predicted frequency count*,
(ii) explicitly names the L_{2,nu}-vs-l2 gap in Morwani's Thm 7 **and** draws the sparsity consequence from it,
(iii) predicts the *number* of Fourier frequencies from homogeneity degree/depth/norm with a closed form, or
(iv) proves the Dirichlet-sidelobe-near-constant fact and applies it to neural network circuit selection.

The two closest calls, with what they actually establish and what survives:

- **Tian, "Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets" (arXiv 2410.01779).** Explicitly writes: *"Existing theoretical work (Morwani et al., 2023) also shows group-theoretical results on algebraic tasks related to finite groups... Compared to ours, they use the max-margin framework with a special regularization (L2,3 norm) rather than L2 loss."* And proves Theorem 6 ("Occam's Razor"): *"If z=y*z' and both z (of order q) and z' are global optimal solutions, then there exists a path of zero loss connecting z and z' in the space of Z_q. As a result, lower-order solutions are preferred if trained with L2 regularization."* Fig. 5 shows, for quadratic-activation 2-layer nets on modular addition ($d\in\{23,71,127\}$), that larger weight decay shifts the solution distribution toward lower "order" (hidden-node count), which in his fixed 6-nodes-per-frequency construction is proportional to frequency count. **What survives:** Tian's "order" is a discovered semiring/path-connectivity notion of hidden-node count in one specific quadratic-network family, not a general margin-normalization argument; he gives **no closed-form count**, no generalization across $\nu=2/3/4/6$ architectures, no Dirichlet-sidelobe computation, and no removal-vs-addition asymmetry. The $K^*(\nu)$ formula and its number-theoretic derivation are untouched.

- **He, Wang, Chen, Yang, "On the Mechanism and Dynamics of Modular Addition" (arXiv 2602.16849, ICLR-cycle 2026).** States *"weight decay induces a sparsification effect in the frequency domain"* and their Definition 4.1 ("Full Diversification") proves the trained network's population covers **all** $(p-1)/2$ frequencies — i.e., their proven network-level result is Fourier-**dense**, the opposite of the sparse claim, built on quadratic (degree-2) activation. Their "sparsification" is a *per-neuron* winner-take-all dynamic (each neuron settles on one dominant frequency while its other components are pruned), not a network-level margin-optimal committee size. **What survives:** the network-level $K^*(\nu)$ count, the l_{2/\nu} quasi-ball, and the Dirichlet-sidelobe mechanism are untouched; their result is compatible with (not competing against) the claim if read as the $\nu=2$ dense boundary case predicted by $K^*(\nu)\to\infty$ as $\nu\to2$.

---

## 3. Near-misses we must differentiate from

Five sources are close enough that silence would look like an omission to an informed reviewer. Each entry gives the exact sentence(s) to place in related work.

### 3.1 Tian 2410.01779 — the single strongest near-miss
**Put in related work:** *"Tian (2024) already observes that Morwani et al.'s max-margin theorem is specific to an L2,3 regularizer rather than the plain L2 loss used in practice, and independently proves (his Theorem 6, 'Occam's Razor') that plain-L2-trained quadratic networks on modular addition prefer lower-order — in his construction, proportionally fewer-frequency — global optima. We do not claim priority on the qualitative L2-favors-sparser-solutions observation. Our contribution is the exact predicted count $K^*(\nu)=2.1\nu/(\nu-2)$, the Dirichlet-sidelobe mechanism ($M_K\approx K-2.1$) that produces it, and its generalization across $\nu=2,3,4,6$ architectures (ReLU MLP, quadratic MLP, transformer), none of which Tian's path-connectivity argument provides or can produce."*
Also must address head-on: Tian's own weight-decay experiments on the identical quadratic-MLP setup show empirically that **all** $(d-1)/2$ frequencies remain populated (order-4/order-6 circuits at every frequency) across $wd\in\{1e^{-5}..5e^{-4}\}$ and $d\in\{23,71,127\}$ — apparently in tension with the $K^*=6$ prediction. Reconcile explicitly: Tian measures per-frequency circuit *redundancy* (how many hidden nodes represent an already-included frequency) under full-data-fitting L2 loss, not the margin-optimal *subset selection* of which frequencies are included at all — different objectives.

### 3.2 He, Wang, Chen, Yang 2602.16849
**Put in related work:** *"He et al. (2026) prove a network-level diversification result (Full Diversification) under plain weight decay and quadratic activation showing all $(p-1)/2$ frequencies survive across the neuron population — i.e., Fourier-dense at the network level. Their 'sparsification effect' is a per-neuron, within-neuron phenomenon (each neuron collapses onto one dominant frequency while its other Fourier components are pruned), distinct from the network-level committee-size optimum $K^*(\nu)$ we derive. Read as the $\nu=2$ boundary of our formula ($K^*\to\infty$ as $\nu\to2$), their dense result is consistent with, not contradictory to, our prediction."*

### 3.3 McCracken et al., "Uncovering a Universal Abstract Algorithm for Modular Addition" (arXiv 2505.18266, NeurIPS 2025)
**Put in related work:** *"McCracken et al. (2025) give a theoretical, architecture-dependent prediction of frequency count — networks with trainable embeddings or depth $>1$ require only $O(\log p)$ features — but by a probabilistic achievability argument (random coset sampling + concentration inequality) showing some sparse solution generalizes, not a margin-maximization argument showing training necessarily converges to one. Their bound is asymptotic in $p$; ours, $K^*(\nu)$, is $O(1)$ in $p$ (verified $p\in[59,211]$) with exact constants. They explicitly state they do not know why the $O(p)\to O(\log p)$ transition with depth/trainable-embeddings occurs ('we don't know why it occurs'); our L2-vs-L_{2,\nu} norm-geometry argument is a candidate mechanistic answer to exactly that open question. They also cite Morwani's dense theorem uncritically as a recovered special case without ever flagging its L_{2,\nu} norm convention — the gap we exploit."*

### 3.4 Kolb et al., "Deep Weight Factorization" (arXiv 2502.02496, ICLR 2025)
**Put in related work:** *"The general algebraic identity we specialize — that plain L2 weight decay on a degree-D multiplicatively-factored parameterization is exactly equivalent to an $\ell_{2/D}$ quasi-norm penalty on the product, sparsity-favoring for $D\ge3$ ('closer to L0', Theorem 1) — is proved in general by Kolb et al. (2025) as an artificial per-scalar Hadamard reparameterization used for explicit pruning. Their $D$ is a user-chosen hyperparameter applied per parameter entry, not the network's own intrinsic architectural homogeneity degree $\nu$; they give no group/neuron-level norm analogous to Morwani's $L_{2,\nu}$, no margin-maximization framing, no modular-arithmetic or Fourier application, and no frequency-count formula. We do not claim the L2-to-$\ell_{2/D}$ mechanism as new; we claim its specific instantiation on Fourier circuits of modular addition, the exact $K^*(\nu)$ count, and the Dirichlet-sidelobe constant as novel."*

### 3.5 Hoff, "Lasso, fractional norm and structured sparse estimation using a Hadamard product parametrization" (arXiv 1611.00040, 2017)
**Put in related work:** *"The core reduction — a ridge (L2) penalty on $K$ Hadamard-product factors is exactly an $L_{2/K}$ penalty on their product, non-convex and sparsity-inducing for $K\ge3$ (Lemma 2) — is known in the sparse-statistics literature since Hoff (2017), six years before Morwani et al. and outside any neural-network or grokking context. We do not claim this algebraic fact as original; our contribution is identifying it as precisely the gap between Morwani's $L_{2,\nu}$ theorem and the plain-l2 weight decay used in every grokking experiment, and deriving the exact per-frequency Dirichlet-sidelobe constant that pins down $K^*(\nu)$ in closed form for modular-addition circuits."*

---

## 4. Mandatory citations table

| Paper | Cite it for |
|---|---|
| Morwani et al. 2023 (arXiv 2311.07568), Thm 7 | The target theorem whose L_{2,3} norm convention we contrast with plain l2 weight decay |
| **Tian 2024 (arXiv 2410.01779)** | Prior, informal statement of the L2-vs-L2,3 gap in Morwani; prior qualitative "L2 favors lower-order/sparser solutions" result on the identical quadratic-MLP setting — **must differentiate**, see §3.1 |
| **He, Wang, Chen, Yang 2026 (arXiv 2602.16849)** | Prior network-level dense-diversification result under plain weight decay for quadratic nets; per-neuron (not network-level) sparsification language — **must differentiate**, see §3.2 |
| **McCracken et al. 2025 (arXiv 2505.18266)** | Prior architecture-dependent (depth/embedding) frequency-count prediction via a different (achievability, not margin) mechanism, asymptotic not $O(1)$-in-$p$ — **must differentiate**, see §3.3 |
| **Kolb et al. 2025 (arXiv 2502.02496)** | General L2-weight-decay-on-degree-$D$-factorization $\Leftrightarrow \ell_{2/D}$ quasi-norm identity, proved outside the NN/grokking context — **credit the primitive**, see §3.4 |
| **Hoff 2017 (arXiv 1611.00040)** | Earliest known instance of the same ridge-on-Hadamard-factors $\Leftrightarrow \ell_{2/K}$ identity, in sparse regression — **credit the primitive**, see §3.5 |
| Lyu & Li 2020 (arXiv 1906.05890) | Foundational KKT/margin-maximization theorem establishing that gradient flow on homogeneous nets converges to a max-margin point under the *plain* (flat) l2 parameter norm — the correct norm setup our per-frequency argument inherits |
| Woodworth et al. 2020 (arXiv 2002.09277) | States (citing Gunasekar 2017; Li–Ma–Zhang 2018) that l2-on-factors of a depth-$D$ diagonal linear net is equivalent to $\|\beta\|_{2/D}$ regularization on the induced predictor — generic precedent for the reparametrization step, no sparsity/modular-addition/frequency claim of its own |
| Neyshabur, Tomioka, Srebro 2015 (arXiv 1503.00036) | Independently observes that flat $\ell_2$/weight-decay hypothesis classes lose convexity relative to grouped $L_{p,q}$ path-norm families once depth $\ge 3$ — foreshadows, but does not derive or connect to margin maximization or Fourier structure |
| Boursier, Pillaud-Vivien, Flammarion 2023 (arXiv 2304.00488) | Background machinery for the SECONDARY claim: zero as an unstable fixed point with an init/regularization-scale-dependent escape threshold in homogeneous reparametrizations — opposite direction (addition, not removal-only) and degree-2, not $\ge3$ |
| Chen et al., "Bridging Lottery Ticket and Grokking" (arXiv 2310.19470) | Soft citation for SECONDARY claim: alternative "weight decay as structure explorer" (discovery) framing that a reviewer might invoke against the strict pruning-only asymmetry — address in one sentence |
| y-agent blog, "On the Mechanism and Dynamics of Modular Addition..." (2026) | Empirical support for the qualitative phenomenon (weight decay prunes non-feature frequencies via per-neuron lottery-ticket dynamics) without any norm-convention or frequency-count formula |

---

## 5. Deflated candidates (one line each)

- **Neyshabur, Tomioka, Srebro 2015 COLT (1503.00036)** — proves $N^d_{2,2}$ non-convexity at depth$\ge$3 and $\ell_1$-equivalence at depth 2 (opposite valence to headline's $\nu=2$-dense claim); zero Fourier/modular-arithmetic/frequency-count content.
- **Neyshabur, Tomioka, Srebro 2015 workshop (1412.6614)** — precursor result is the $\nu=2$ (dense-predicted) case only, about active-*unit* sparsity in an infinite dictionary, not Fourier-frequency sparsity.
- **Jacot 2023, "Implicit Bias of Large Depth Networks" (2209.15055)** — depth-driven function-*rank* (Jacobian/Bottleneck rank) result, not per-neuron/per-frequency sparsity; no modular arithmetic, no Fourier, no K* formula.
- **Lyu & Li 2020 (1906.05890)** — correct norm setup (plain l2) but general theory only; no modular addition, no Morwani-gap statement, no frequency-count prediction, predates Morwani entirely.
- **Jacobs & Montúfar, "Implicit Bias of Mirror Flow" (2605.19458)** — sparse-vs-dense dichotomy is over the *regularizer's* homogeneity degree (an algorithm choice), not the *network's*; no Morwani citation, no Fourier/modular-addition content, opposite-direction axis.
- **Mohamadi et al. 2024, "Why Do You Grok?" (2407.12332)** — dense, all-$p$-frequency explicit construction for the same $\nu=3$ quadratic architecture, using $\ell_\infty$ not l2/L_{2,\nu}; no frequency-count formula.
- **Memana 2026, "Counterexample to Fourier Alignment" (2608.04451)** — single-neuron, *unregularized* (explicitly "there is no weight decay") Clarke-dynamics counterexample; proves density not sparsity; explicitly scopes itself outside Morwani's margin mechanism.
- **Woodworth et al. 2020 (2002.09277)** — states the general $\ell_{2/D}$ equivalence only as background (attributed to Gunasekar 2017 / Li-Ma-Zhang 2018), no sparsity claim, no modular addition, no frequency count.
- **Li, Ma, Zhang 2018 (1712.09203)** — trajectory/init-scale implicit-bias story (small-init low-rank recovery), not an explicit-weight-decay margin characterization; "weight decay" appears only as a loose dynamical analogy.
- **Kolb et al. (Smoothing the Edges), 2307.03571** — extends Hoff's identity to be SGD-smooth for general nonlinear/vision pruning; no modular addition, no margin theory, no Fourier content.
- **Xu et al. 2026, "Theoretical Analysis of Sparse Optimization..." (2605.25134)** — general $\ell_p$ ($p<1$) pruning method for vision CNNs (ResNet/ImageNet); keyword-only overlap with "weight decay" + "sparse," no grokking/Fourier/margin content.
- **y-agent blog (2026)** — qualitative, empirical "weight decay prunes non-feature frequencies via lottery-ticket dynamics" observation; no norm formalism, no K*(nu), dense-ensemble ("all frequencies covered") framing.
- **Boursier, Pillaud-Vivien, Flammarion 2023 (2304.00488)** — vanishing-init $\ell_1$-implicit-bias dynamics for degree-2 diagonal linear nets; coordinates are *added* from zero (opposite direction to the claim's removal-only asymmetry), explicitly non-monotone active-set size.
- **Chen et al., "Bridging Lottery Ticket and Grokking" (2310.19470)** — "weight decay as structure explorer" is qualitative lottery-ticket/pruning-based subnetwork discovery via a Fourier-Entropy heuristic, not margin theory; soft tension with the SECONDARY claim only, no formalized counterexample.

---

## 6. Coverage: angles searched, queries, blind spots

**Nine search angles, ~200 total queries**, each combining keyword search, arXiv/Semantic-Scholar citation-graph walks of Morwani (2311.07568) and He et al. (2602.16849), and direct full-text fetches (arXiv abstract + PDF + ar5iv HTML) of ~20 closest candidates:

1. **implicit-bias-homogeneous-l2L** — general depth/degree-homogeneous l2-implicit-bias literature (Gunasekar, Woodworth, Chizat, Lyu-Li, Jacot, Neyshabur).
2. **morwani-citation-trail** — exhaustive citation-graph walk of Morwani 2311.07568, including Semantic Scholar's full citing-paper list.
3. **grokking-frequency-count-predictors** — mechanistic-interpretability literature (Nanda, Zhong, Gromov, Chughtai, Doshi, He et al., Tan & Huang, Mohamadi, Stander) for any theoretical (not just empirical) frequency-count prediction.
4. **L2nu-norm-critique-generalization** — direct searches for a follow-up/critique paper reworking Morwani's framework under plain l2.
5. **dirichlet-sidelobe-number-theory** — classical analytic-number-theory/combinatorics search (Sidon sets, Bohr sets, Barker/Costas sequences, Chowla cosine problem, large sieve) for the minimax-sidelobe-near-constant fact, with or without an ML application.
6. **architecture-dense-vs-sparse-explanation** — targeted search for any existing account of the specific MLP-dense-vs-transformer-sparse discrepancy the claim says it explains.
7. **weight-decay-vs-lp-reg-feature-learning** — broader ML/compressed-sensing literature on l2-weight-decay-induces-sparsity as a general phenomenon (this angle surfaced the two genuine near-miss primitives, Hoff 2017 and Kolb et al. 2025).
8. **circuit-formation-seed-lottery-weight-decay** — literature on the SECONDARY removal-only/addition-blocked asymmetry and escape-threshold/attracting-zero-basin dynamics.
9. **openreview-blogs-theses-nonarxiv** — informal venues (OpenReview reviews/rebuttals of Morwani's own submission, LessWrong, Alignment Forum, theses, workshop papers) where a sharp gap-spotting argument might appear pre-formal-publication.
10. **2026-recency-sweep** — dedicated sweep of Jan–Aug 2026 arXiv postings and ICML 2026 / ICLR 2027 OpenReview submission cycles for very recent scoops, including direct follow-ups to He et al. (Feb 2026).

**Known blind spots:**
- **OpenReview review/rebuttal threads on Morwani's own paper** were searched by keyword but not exhaustively enumerated thread-by-thread; a reviewer comment buried in an old ICLR 2024 rebuttal thread pointing out the L2-vs-L_{2,nu} gap could exist without being indexed by the queries used.
- **Non-English or non-arXiv preprint servers** (e.g. OpenReview-only workshop submissions without arXiv mirrors, Chinese-language preprint venues) were not searched directly.
- **Semantic Scholar's citing-paper list for Morwani (2311.07568) was read but not each of the ~39 citing papers individually deep-read** — only those surfaced as thematically relevant by keyword/abstract screening were fetched in full; a citing paper with an off-topic abstract but a buried relevant remark could have been missed.
- **The Dirichlet-sidelobe-near-constant fact (angle 5)** returned no ML-adjacent hits, but classical number theory is a large literature (flat polynomials, Fekete/Turán problems, Erdős–Chowla-type results) that was sampled, not exhaustively swept; a pure-math paper proving this exact constant without any ML framing could exist unindexed by the ML-flavored query phrasing used.
- **Very recent (last 2–3 weeks of August 2026) preprints** may postdate the search's most recent crawl; given the field's pace (He et al. Feb 2026, McCracken NeurIPS 2025, Memana Aug 2026 all landed within the past six months), a fresh check immediately before submission is warranted.

---

## 7. Recommended headline wording given the prior art

Do **not** claim: *"we are the first to notice that Morwani et al.'s max-margin theorem uses L_{2,nu} rather than plain l2."* (Tian 2410.01779 already states this.)

Do **not** claim: *"we are the first to show weight decay makes modular-addition solutions sparser."* (Tian's Theorem 6 + Fig. 5 already show this qualitatively for the $\nu=3$ case.)

**Recommended wording:**

> *"Morwani et al. (2023) prove that the max-margin solution for modular addition is Fourier-dense — but under an $L_{2,\nu}$ per-neuron regularizer, not the plain $\ell_2$ weight decay every grokking experiment actually trains with. Tian (2024) already flags this norm mismatch and shows, for one architecture, that plain-$\ell_2$ training qualitatively prefers lower-order solutions. We resolve the gap quantitatively: under $\ell_2$ weight decay, the per-frequency margin constraint becomes an $\ell_{2/\nu}$ quasi-ball, and the optimal committee size within the single-frequency-circuit family has a closed form, $K^*(\nu) = 2.1\nu/(\nu-2)$, driven by a Dirichlet-sidelobe near-invariant we prove is nearly independent of $K$ and $p$. This single formula reproduces the dense $\nu=2$ ReLU-MLP regime, Tian's and He et al.'s $\nu=2/3$ observations as boundary/special cases, and correctly predicts the long-unexplained ReLU-MLP-vs-transformer frequency-count discrepancy ($K^*=4$ at $\nu=4$) without appeal to architecture-specific dynamics."*

This framing (a) pre-empts the two live near-misses by name, (b) is honest about which primitive (l2-weight-decay-on-homogeneous-map $\Rightarrow \ell_{2/\nu}$) is old math (Hoff 2017; Kolb et al. 2025) versus which application (Fourier circuits, $K^*(\nu)$ closed form, Dirichlet-sidelobe constant) is new, and (c) turns McCracken et al.'s explicitly-stated open question ("we don't know why [the O(p)→O(log p) transition] occurs") into a citation opportunity rather than a competing claim.
