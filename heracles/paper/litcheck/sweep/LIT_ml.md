# LIT_ml: ML literature sweep (2019-2026) for "Explaining Feature Sparsity Through the Margin-Regularization Trade-off"

Sweep date: 2026-09-25. Every entry below was fetched this session (arXiv abs/HTML/PDF, PMLR, NeurIPS proceedings, or iclr.cc via search result). Items marked **[venue unconfirmed]** have title/authors confirmed on arXiv but the venue was not confirmed from a fetched page.

## 0. Headline findings (read first)

1. **BIB ERROR (must fix): `mccracken2025` is wrong.** arXiv 2505.18266 is *"Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks"* by **Gavin McCracken, Gabriela Moisescu-Pareja, Vincent Letourneau, Doina Precup, Jonathan Love** (v1 23 May 2025, only version; no journal-ref). The bib's title "Emergence of sparse and dense circuits in modular arithmetic networks" / "McCracken, Andrew and others" does not exist at that ID.
   - What they prove (Thm 4.7, HTML v1): if `m > (2 ln n - 2 ln(2-2rho)) / ln(pi/delta) - 1` random frequencies are drawn uniformly from {1..n/2}, then w.p. >= rho the logit h(k)=sum_l cos(2 pi f_l (k-i-j)/n) has h(i+j)-h(k) > delta m for all wrong k. Cor 4.8: "Learning O(log n) distinct frequencies gives a logit margin Omega(log n)". This is **unnormalised** (no norm, no weight decay) and a **sufficiency** statement.
   - They say (same HTML): "it gives that the maximum margin solution requires all frequencies to be learned" [citing Morwani] and on the O(n)->O(log n) transition "we don't know why it occurs."
   - **Overlap with our Thm 4.x (logarithmic scaling):** our lower bound Gamma_s(p) >= 1/2 ceil(8 ln p)^{1-s} is a random/log-many-frequency coverage argument of the same kind. Cite McCracken as prior for "O(log p) random frequencies suffice to separate all offsets"; our new parts are (i) the matching *upper* bound on Gamma_s, (ii) K* = Theta(log p) for the *optimum* (both directions, with share floor), (iii) the normalised margin Theta(p^{-nu/2}(log p)^{1-nu/b}) over all networks, (iv) Cor. "o(log p) features attain vanishing fraction". Their paper does not contain any of these.
2. **Closest structural near-miss (concurrent, Sept 2026): Kam, Bessafi, Cadet, arXiv 2609.17926**, "Symmetry without a manifold: intrinsic dimension on orbits" (under review, NeurReps 2026). Appendix A.3.2, Eq. 15: worst-case separation Delta_S(beta)=min_{s!=0} sum_{k in S} beta_k (1-cos(omega_k s)), maximised subject to **||beta||_2 = B** (a convex, l2 budget on amplitudes, *not* derived from homogeneity). Proposition 5: uniform allocation beta_k=B/sqrt(K_max) is optimal at full frequency content (separation B p / sqrt(2(p-1))); for proper subsets numerical optimum beats uniform by 25% (K=4), 28% (K=10). Quadratic two-layer net. A.3.1: unconstrained, loss < 1e-12 from K=4 at p=47, so "the norm budget (weight decay), not mode availability, is the binding constraint". **This is our budget-splitting programme with exponent 1/s replaced by 2 (convex), which is exactly why they get dense.** They do not derive the exponent from homogeneity, do not get sparsity, K*, log p, shapes, bottlenecks, or ablation. Must cite as concurrent; it is actually a nice foil ("a convex budget gives dense; the homogeneity-derived concave budget gives sparse").
3. **Groups section near-miss: Shutman, Louidor, Tessler, arXiv 2509.06931**, "Learning words in groups: fusion algebras, tensor ranks and grokking". Prop 5.2: rank of the single-irrep projection of the word tensor <= m_d (type I), 3 m_{d/2} (type II), 8 m_{d/2} (type III), where **m_d = tensor rank of real d x d matrix multiplication**; the net "effectively implements efficient matrix multiplication in the sense of Strassen". Hadamard (product) model; Lemma 4.6: square-activation net implements it at doubled width. **No weight decay, no margin, no norm** (pure GD). Our groups section claim "each irrep pays a price set by how cheaply the matrix-multiplication tensor can be written as a sum of products ... schoolbook, d^nu neurons" must cite this. Our novelty = the norm price and the resulting count, not the tensor-rank link.
4. **No paper found that proves any of:** the L_{2,b} (2<=b<nu) universal bound gamma*_b <= C_b^{nu/b} Gamma_{nu/b}(p); the interference-free exact optimum (nu-block, cost lemma nu^{1-r}); finitely many shapes in s; bottleneck offsets; feature-ablation bound 1-(1-w)^s; K*=Theta(log p); gamma*_b = Theta(p^{-nu/2}(log p)^{1-nu/b}). Searches listed in Section 6.
5. **Generic mechanism is known and must be credited:** L2 on a product of D factors = l_{2/D} quasi-norm (AM-GM) → sparsity. Hoff 2017; Dai-Karzand-Srebro 2021 (Schatten/l_{2/L} representation cost); Kolb et al. ICLR 2025; Ongie-Parhi 2026. Our "amplitude A costs A^{b/nu}" line (approach.tex, end of Sec 3.2) is this fact; do not present the AM-GM exponent as new.
6. **Consistency checks passed:** (a) our b=nu bound for power neurons, n! B_p Gamma_1(p) = 2 n! / ((2n+2)^{(n+1)/2} (p-1) p^{(n-1)/2}), equals **Li et al. AISTATS 2025 Theorem 4.1** gamma* exactly (k=n) — Cor. `cor:dense` can say it recovers both Morwani Thm 7 and Li Thm 4.1. (b) at n=2 it equals Morwani Thm 7 sqrt(2/27)/(p^{1/2}(p-1)).
7. **Citation precision fix: Wei et al. Theorem 4.1 is stated for binary logistic loss** with f a-positive-homogeneous and "some general norm", regulariser lambda||Theta||^r. The multi-class cross-entropy version is **Theorem C.1** (Appendix C, "Multi-class Setting"): "Assume gamma* > 0 in the multi-class setting with cross entropy loss. Then as lambda -> 0, gamma_lambda -> gamma*." Our approach.tex cites [Theorem 4.1] for cross-entropy; change to "Theorem 4.1 and its multi-class form, Theorem C.1". Morwani also cites it as "Wei et al. 2019a, Theorem 4.1".

## 1. Direct predecessors (already cited; verified)

| key | verified reference | what it proves (location) | relation |
|---|---|---|---|
| morwani2024feature | Morwani, Edelman, Oncescu, Zhao, Kakade. *Feature emergence via margin maximization: case studies in algebraic tasks.* ICLR 2024 (spotlight). arXiv 2311.07568. https://arxiv.org/abs/2311.07568 ; https://iclr.cc/virtual/2024/poster/18073 | Thm 7: quadratic nets, m >= 4(p-1), max L_{2,3} margin gamma* = sqrt(2/27)/(p^{1/2}(p-1)); every neuron single-frequency; "for every frequency zeta ... at least one neuron uses this frequency". Thm 8 sparse parity (x^k). Thm 9 groups: gamma* = 2/(3 sqrt(3|G|)) / sum_{n>=2} d_{R_n}^{2.5}, all non-trivial irreps used, each neuron in one irrep (real irreps). Lemma 2 / Lemma 6: single-neuron class-weighted certificate. Fig 4(b): ReLU + L2 "nearly 1-sparse in frequency space but not quite". | Our b=nu endpoint; we extend to b<nu. |
| li2025nsum | Li, Liang, Shi, Song, Zhou. *Fourier Circuits in Neural Networks and Transformers: A Case Study of Modular Arithmetic with Multiple Inputs.* AISTATS 2025, PMLR 258:523-531. https://proceedings.mlr.press/v258/li25b.html ; arXiv 2402.09469 | Thm 4.1: neurons (u_1^T x_1+...+u_k^T x_k)^k w, L_{2,k+1}, m >= 2^{2k-2}(p-1): gamma* = 2(k!)/((2k+2)^{(k+1)/2}(p-1)p^{(k-1)/2}); every neuron one frequency; every frequency used. | Our power-neuron family at b=nu; general-nu dense endpoint is theirs. Add pages to bib. |
| mccracken2025 | **FIX**: McCracken, Moisescu-Pareja, Letourneau, Precup, Love. *Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks.* arXiv 2505.18266 (2025). https://arxiv.org/abs/2505.18266 | Thm 4.7 / Cor 4.8 above; O(log n) features for deep/embedding nets, 1-layer MLPs found floor(n/2). | See §0.1. |
| nanda2023progress | Nanda, Chan, Lieberum, Smith, Steinhardt. *Progress measures for grokking via mechanistic interpretability.* ICLR 2023 (oral). https://arxiv.org/abs/2301.05217 ; https://iclr.cc/virtual/2023/oral/12572 | Fourier-multiplication algorithm with a few key frequencies; memorise/circuit/cleanup phases. | Empirical sparse count. |
| chughtai2023toy | Chughtai, Chan, Nanda. *A Toy Model of Universality: Reverse Engineering how Networks Learn Group Operations.* ICML 2023, PMLR 202:6243-6267. https://proceedings.mlr.press/v202/chughtai23a.html | Group-composition via representations; which irreps and order vary across seeds. | Empirical few-irrep count. |
| wei2019regularization | Wei, Lee, Liu, Ma. *Regularization Matters: Generalization and Optimization of Neural Nets v.s. their Induced Kernel.* NeurIPS 2019 (vol. 32). https://proceedings.neurips.cc/paper/2019/hash/8744cf92c88433f8cb04a02e6db69a0d-Abstract.html | Thm 4.1 (binary) / Thm C.1 (multi-class CE): normalized margin of global optimum of lambda-regularised loss -> gamma* as lambda -> 0, any a-homogeneous f, general norm. | See §0.7. |
| lyu2020gradient | Lyu, Li. *Gradient Descent Maximizes the Margin of Homogeneous Neural Networks.* ICLR 2020. https://arxiv.org/abs/1906.05890 ; https://iclr.cc/virtual_2020/poster_SJeLIgBKPS.html | GF/GD on logistic/CE for homogeneous models → KKT point of L2 max-margin problem. | Cite for "what GD converges to" (only KKT, and only L2 = b=2 when unregularised). |
| gromov2023grokking | Gromov. *Grokking modular arithmetic.* arXiv 2301.02679 (2023). https://arxiv.org/abs/2301.02679 | Analytic weights (feature maps) solving modular arithmetic, found by GD and AdamW, no regularisation. | Construction precedent (phases). |
| power2022grokking | Power, Burda, Edwards, Babuschkin, Misra. *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.* arXiv 2201.02177. https://arxiv.org/abs/2201.02177 | Grokking phenomenon. | Background. |

## 2. Must-add citations (verified), grouped

### 2a. Concurrent / near-miss on the core claim
- **Kam, Bessafi, Cadet (2026).** *Symmetry without a manifold: intrinsic dimension on orbits.* arXiv 2609.17926 (15 Sep 2026; under review, NeurReps 2026). https://arxiv.org/abs/2609.17926 — see §0.2. Relevance: HIGH (same max-min coverage functional with an l2 amplitude budget; dense optimum).
- **He, Wang, Chen, Yang (2026).** *On the Mechanism and Dynamics of Modular Addition: Fourier Features, Lottery Ticket, and Grokking.* arXiv 2602.16849. https://arxiv.org/abs/2602.16849 — Def 4.1 "full diversification": every frequency carried by exactly N neurons + phase conditions; Prop 4.2 majority voting approximates indicator; quadratic activation; weight decay only in a 3-stage grokking narrative, no count theorem. Relevance: medium (dense, per-neuron dynamics; our "feature = nu neurons with roots-of-unity phases" is their phase-symmetry idea in the product model — cite for the phase-cancellation intuition).
- **He, Wang, Zhang, Chen, Yang (2026).** *Neural Networks Provably Learn Spectral Representations for Group Composition.* arXiv 2606.02993. https://arxiv.org/abs/2606.02993 — neurons converge to single irreps; for abelian groups "uniform diversification across nontrivial representations". Relevance: medium (dense, dynamics, no weight-decay count).
- **Levine (2026).** *Math for AI safety: an invitation for mathematicians.* arXiv 2609.15289. https://arxiv.org/abs/2609.15289 — Open Problem **MAIS-O5** (Sec 4.3): one-hidden-layer, Gaussian init, gradient flow on CE **with weight decay**: "determine its distribution — which irreps are learned, with what probability, as a function of the width and the decay strength." Relevance: our count theorem answers the *how many* half of this open problem (not the probability law). Good citation for motivation. (The MAIS registry items O52/O60 are referenced in 2608.04451 as internal MAIS documents with no URL; **unconfirmed**, do not cite O52 by number.)

### 2b. Weight decay on products = concave quasi-norm (the exponent 1/s)
- **Hoff (2017).** *Lasso, fractional norm and structured sparse estimation using a Hadamard product parametrization.* Computational Statistics & Data Analysis 115:186-198. arXiv 1611.00040. https://arxiv.org/abs/1611.00040 — "a subclass of L_q penalties with q<=1 can be expressed as sums of L_2 penalties" (multiplicative reparametrisation). The scalar version of our AM-GM step.
- **Dai, Karzand, Srebro (2021).** *Representation Costs of Linear Neural Networks: Analysis and Design.* NeurIPS 34. https://proceedings.neurips.cc/paper/2021/hash/e22cb9d6bbb4c290a94e4fff4d68a831-Abstract.html — representation cost of l2 on parameters for linear nets; which l_p/group norms arise; bias to sparsity for diagonal nets.
- **Kolb, Weber, Bischl, Rügamer (2025).** *Deep Weight Factorization: Sparse Learning Through the Lens of Artificial Symmetries.* ICLR 2025. arXiv 2502.02496. https://arxiv.org/abs/2502.02496 — depth-D factorisation + L2 on factors ↔ non-convex sparse regularisation.
- **Ongie, Parhi (2026).** *Representation Costs in Data Science: Foundations and the Quasi-Banach Spaces of Deep Neural Networks.* arXiv 2606.14954. https://arxiv.org/abs/2606.14954 — depth-L ReLU + weight decay: representation cost is a power of a quasi-seminorm; nonconvex balls for L>2.
- **Outmezguine, Levi (2024).** *Decoupled Weight Decay for Any p Norm.* arXiv 2404.10824. https://arxiv.org/abs/2404.10824 **[venue unconfirmed]** — L_p (bridge) weight decay for sparsity. Low relevance.

### 2c. Implicit bias of homogeneous nets / optimisers (why the norm matters)
- **Chizat, Bach (2020).** *Implicit Bias of Gradient Descent for Wide Two-layer Neural Networks Trained with the Logistic Loss.* COLT 2020, PMLR 125:1305-1338. https://arxiv.org/abs/2002.04486 — infinite-width homogeneous two-layer: GF limit is a max-margin classifier in a non-Hilbertian (F1) space.
- **Nacson, Gunasekar, Lee, Srebro, Soudry (2019).** *Lexicographic and Depth-Sensitive Margins in Homogeneous and Non-Homogeneous Deep Models.* ICML 2019 (camera-ready per arXiv comment). https://arxiv.org/abs/1905.07325 — regularisation path → (lexicographic) max-margin.
- **Gunasekar, Lee, Soudry, Srebro (2018).** *Characterizing Implicit Bias in Terms of Optimization Geometry.* arXiv 1802.08246 **[venue unconfirmed; ICML 2018]**. https://arxiv.org/abs/1802.08246 — steepest descent / mirror descent converge to geometry-dependent max-margin.
- **Tsilivis, Gronich, Kempe, Vardi (2025).** *Flavors of Margin: Implicit Bias of Steepest Descent in Homogeneous Neural Networks.* ICLR 2025 (proceedings PDF surfaced in search: proceedings.iclr.cc/paper_files/paper/2025/file/1c9ee8c9e1a0c7de49e15720c4ce6f78-Paper-Conference.pdf). https://arxiv.org/abs/2410.22069 — algorithm-dependent geometric margin; limit points are KKT points of the corresponding margin problem; Adam/Shampoo connection.
- **Xie, Li (2024).** *Implicit Bias of AdamW: l_inf-Norm Constrained Optimization.* ICML 2024, PMLR 235:54488-54510. https://proceedings.mlr.press/v235/xie24e.html — AdamW converges to KKT of loss under l_inf constraint. **Validity attack**: trained AdamW nets are not L_{2,b}-maximisers; pre-empt in experiments.
- **Jacobs, Montúfar (2026).** *Implicit Bias of Mirror Flow in Homogeneous Neural Networks: Sparse and Dense Feature Learning.* arXiv 2605.19458. https://arxiv.org/abs/2605.19458 — Thm 4.9: hyperbolic-entropy (alpha=1) potentials bias to fewer active neurons, smoothed homogeneous (alpha>=2) to more. No modular tasks, no Morwani. The generic "an exponent decides sparse vs dense" sentence is theirs; ours is task-specific counts.
- **Lyu, Jin, Li, Du, Lee, Hu (2024).** *Dichotomy of Early and Late Phase Implicit Biases Can Provably Induce Grokking.* ICLR 2024. https://arxiv.org/abs/2311.18817 — large init + small wd: kernel predictor then sharp transition to min-norm/max-margin; homogeneous nets on arithmetic. Justifies studying the max-margin endpoint.
- **Mohamadi, Li, Wu, Sutherland (2024).** *Why Do You Grok? A Theoretical Analysis of Grokking Modular Addition.* ICML 2024. https://arxiv.org/abs/2407.12332 — kernel regime needs constant fraction of data; quadratic two-layer nets with bounded l_inf norm generalise with fewer samples.

### 2d. Grokking / modular arithmetic mechanism (background, verified)
- **Zhong, Liu, Tegmark, Andreas (2023).** *The Clock and the Pizza.* NeurIPS 2023. https://arxiv.org/abs/2306.17844
- **Doshi, He, Das, Gromov (2024).** *Grokking Modular Polynomials.* arXiv 2406.03495 **[venue unconfirmed]**. https://arxiv.org/abs/2406.03495 — analytic solutions extended to "modular addition with many terms" (our n-operand setting, constructive, no margin).
- **Ding, Guo, Michaud, Liu, Tegmark (2024).** *Survival of the Fittest Representation: A Case Study with Modular Addition.* ICML 2024 Workshop on Mechanistic Interpretability (per mlanthology). https://arxiv.org/abs/2405.17420 — "By increasing the embedding dimension, we also observe more surviving frequencies." **Validity attack on width-invariance** (embedding dim, threshold instrument).
- **Varma, Shah, Kenton, Kramár, Kumar (2023).** *Explaining grokking through circuit efficiency.* arXiv 2309.02390. https://arxiv.org/abs/2309.02390 — parameter-norm efficiency of generalising vs memorising circuits.
- **Mallinar, Beaglehole, Zhu, Radhakrishnan, Pandit, Belkin.** *Emergence in non-neural models: grokking modular arithmetic via average gradient outer product.* arXiv 2407.20199 (arXiv comment: "Accepted to ICML 2025"). https://arxiv.org/abs/2407.20199
- **Kunin, Marchetti, Chen, Karkada, Simon, DeWeese, Ganguli, Miolane (2025).** *Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks.* NeurIPS 2025. https://arxiv.org/abs/2506.06489 — quadratic nets on modular addition learn Fourier features "in decreasing order of coefficient magnitude" (order, not count; small init, no wd).
- **Tian (2025).** *Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets.* NeurIPS 2025. https://arxiv.org/abs/2410.01779 — semi-ring structure on weights of 2-layer quadratic nets for abelian groups; "training dynamics favors simpler solutions under weight decay". Order = hidden nodes, not frequencies.
- **Tian (2025).** *Provable Scaling Laws of Feature Emergence from Learning Dynamics of Grokking* (HTML title "Li_2: A Framework on Dynamics of Feature Emergence and Delayed Generalization"). arXiv 2509.21519. https://arxiv.org/abs/2509.21519 — dynamics; roles of wd/lr/sample size; no count law found in abstract.
- **Pracher, de Jong, Lieshaus, Jeffares, Rulands (2026).** *A Spectral Theory of Grokking: Weight Decay induces Feature Learning.* arXiv 2609.26679 (22 Sep 2026). https://arxiv.org/abs/2609.26679 — single-mode NTK reduction; grokking time ~ lr x wd; explicitly neglects multi-mode interactions; no count. Low overlap; cite as concurrent wd-in-grokking theory.
- **Memana (2026).** *A Counterexample to Fourier Alignment in Single-Neuron Modular Addition.* arXiv 2608.04451. https://arxiv.org/abs/2608.04451 — single ReLU neuron can die with uniform Fourier energy; "single-frequency alignment is not a general consequence of training". Relevant to our "interference is present in trained nets" message.
- **Huang, Li (2025).** *Provable Benefits of Sinusoidal Activation for Modular Addition.* arXiv 2511.23443. https://arxiv.org/abs/2511.23443 — low relevance.
- **Nguyen (2026).** *The Discrete-Log Clock.* ICML 2026 MI Workshop. arXiv 2606.17399 — low relevance (multiplication sparsity in character basis).
- **Tan, Gai, Zhang (2026).** *Beyond Neural Collapse: Task-Intrinsic Geometry Governs Neural Representations in Modular Arithmetic.* arXiv 2606.08985 — "task-structured trade-off between separation, symmetry, and complexity"; K = number of classes, not features. Low.

### 2e. Groups / irreps / tensor rank
- **Shutman, Louidor, Tessler (2025).** arXiv 2509.06931 — see §0.3. https://arxiv.org/abs/2509.06931
- **Marchetti, Hillar, Kragic, Sanborn (2024).** *Harmonics of Learning: Universal Fourier Features Emerge in Invariant Networks.* COLT 2024. https://arxiv.org/abs/2312.08550 — invariant nets' weights recover the group Fourier transform (irreps).
- **Stander, Yu, Fan, Biderman (2024).** *Grokking Group Multiplication with Cosets.* ICML 2024, PMLR 235:46441-46467. https://proceedings.mlr.press/v235/stander24a.html
- **Wu, Jaburi, Drori, Gross (2025).** *Towards a unified and verified understanding of group-operation networks.* ICLR 2025. https://arxiv.org/abs/2410.07476
- **Marchetti, Kunin, Myers, Acosta, Miolane (2026).** *Sequential Group Composition: A Window into the Mechanics of Deep Learning.* ICML 2026. https://arxiv.org/abs/2602.03655 — learn "one irreducible representation of the group at a time"; shallow nets need width exponential in sequence length k (relevant to n-operand scaling).
- **Larson (2026).** *The Rank the Task Demands: A Causal Rank Law for Matrix Memories Trained on Group Composition.* arXiv 2609.12259 — recruited rank = minimal faithful real rep dimension; S4 vs A5 preregistered. Relevant to our S5/A5 counts (one faithful irrep; Galois pair).

### 2f. Tensor / polynomial / multilinear networks
- **Razin, Maman, Cohen (2021).** *Implicit Regularization in Tensor Factorization.* ICML 2021. https://arxiv.org/abs/2102.09972 — GD ≈ greedy low-tensor-rank search. A product-neuron network is a CP factorisation; cite for "sum of product neurons = CP rank".
- **Kileel, Trager, Bruna (2019).** *On the Expressive Power of Deep Polynomial Neural Networks.* arXiv 1905.12207 **[venue unconfirmed; NeurIPS 2019]**. https://arxiv.org/abs/1905.12207
- **Chrysos, Moschoglou, Bouritsas, Panagakis, Deng, Zafeiriou (2020).** *Pi-nets: Deep Polynomial Neural Networks.* CVPR 2020. https://arxiv.org/abs/2003.03828
- **Yarotsky, Golikov, Gusev (2026).** *Gradient Flow Through Diagram Expansions.* ICML 2026 spotlight. arXiv 2602.04548 — CP-decomposition learning regimes. Low.

### 2g. Superposition / capacity allocation (conceptual neighbours of "budget splitting")
- **Elhage et al. (2022).** *Toy Models of Superposition.* arXiv 2209.10652 / Transformer Circuits Thread. https://arxiv.org/abs/2209.10652
- **Scherlis, Sachan, Jermyn, Benton, Shlegeris.** *Polysemanticity and Capacity in Neural Networks.* arXiv 2210.01892 **[venue unconfirmed]**. https://arxiv.org/abs/2210.01892 — optimal capacity allocation: important features get full capacity, least important ignored. Conceptual analogue of our "features priced out".

### 2h. Representation cost (function space) — optional
- **Savarese, Evron, Soudry, Srebro (2019).** *How do infinite width bounded norm networks look in function space?* arXiv 1902.05040 **[venue unconfirmed; COLT 2019]**.
- **Ongie, Willett, Soudry, Srebro (2019).** *A Function Space View of Bounded Norm Infinite Width ReLU Nets: The Multivariate Case.* arXiv 1910.01635 **[venue unconfirmed; ICLR 2020]**.
- **Wen, Jacot.** *Which Frequencies do CNNs Need? Emergent Bottleneck Structure in Feature Learning.* arXiv 2402.08010 **[venue unconfirmed]** — frequency-count ("CBN rank") cost of representation; no optimum count. Low.

## 3. Claim-by-claim novelty verdict

| Paper claim | Prior art found | Verdict |
|---|---|---|
| Averaged-margin / phase lemma (Lem avg, phase) | Morwani Sec 3 class-weighted margin (Lemma 2/6), phase condition Sec 4 | Adapted; paper already credits. OK. |
| AM-GM price A^{b/nu} | Hoff 2017; Dai-Karzand-Srebro 2021; Kolb 2025 | Known mechanism; cite. |
| Budget-splitting programme Gamma_s (max-min coverage with l_{1/s} budget) | Kam et al. 2609.17926 Eq.15/Prop 5 with **l2** amplitude budget (dense) | Concave-exponent version + derivation from homogeneity + theorems: not found. Cite Kam as concurrent. |
| Universal bound, any width (Thm universal) | Morwani/Li only at b=nu | Not found for b<nu. |
| b=nu recovers dense optimum (Cor dense) | Morwani Thm 7, Li Thm 4.1 | Exact agreement verified numerically-symbolically for both; say so. |
| Sign rule / cost lemma nu^{1-r}, nu-block with rotated roots-of-unity phases | Phase symmetry in He et al. 2602.16849 (Def 4.1 cond. iii: exp(i iota sum phi_m)=0); Gromov analytic phases; Doshi many-term solutions | Cancellation-by-phases idea exists; the cost lemma (Hadamard/determinant bound, equality iff exactly nu neurons) not found. |
| Finitely many shapes in s; bottleneck offsets; ablation bound 1-(1-w)^s | none | Not found. |
| K* = Theta(log p); gamma* = Theta(p^{-nu/2}(log p)^{1-nu/b}); o(log p) features vanish | McCracken Thm 4.7/Cor 4.8 (sufficiency of O(log n) random freqs, unnormalised) | Sufficiency direction is McCracken's; tight two-sided normalised statement not found. |
| Groups: irrep price = cost of matmul tensor as sum of products | Shutman et al. Prop 5.2 (tensor rank m_d, Strassen), no norm; Morwani Thm 9 (d^{2.5} at b=nu) | Tensor-rank link is theirs; norm price & count is new. Cite both. |
| S5 one irrep / A5 Galois pair observed | Chughtai 2023, Stander 2024, Wu 2025, Larson 2026 (faithful-rank law) | Empirical agreement; cite. |
| "What decides how many features" posed | Levine MAIS-O5 (2609.15289) | Cite as the open question. |

## 4. Validity attacks to pre-empt (from ML literature)
- Ding et al. 2405.17420: surviving frequency count rises with embedding dimension (contradicts width-invariance if our experiments claim it).
- Xie-Li ICML 2024 / Tsilivis ICLR 2025: AdamW/steepest descent maximise a different (l_inf / algorithm-dependent) margin; trained AdamW nets are evidence for the wrong norm. Use GD cohort.
- Lyu-Li 2020 gives only KKT points, and Wei is global-min as lambda→0 — the paper already says "Nothing here concerns whether gradient descent finds them"; keep.
- Memana 2608.04451 + He et al.: trained neurons are not always single-frequency; supports our "interference present in trained nets" framing.

## 5. BibTeX (only for confirmed entries; new or corrected)

```bibtex
@article{mccracken2025,
  title={Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks},
  author={McCracken, Gavin and Moisescu-Pareja, Gabriela and Letourneau, Vincent and Precup, Doina and Love, Jonathan},
  journal={arXiv preprint arXiv:2505.18266},
  year={2025}
}
@inproceedings{li2025nsum,
  title={Fourier Circuits in Neural Networks and Transformers: A Case Study of Modular Arithmetic with Multiple Inputs},
  author={Li, Chenyang and Liang, Yingyu and Shi, Zhenmei and Song, Zhao and Zhou, Tianyi},
  booktitle={Proceedings of the 28th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  series={PMLR}, volume={258}, pages={523--531}, year={2025}
}
@article{kam2026symmetry,
  title={Symmetry without a manifold: intrinsic dimension on orbits},
  author={Kam, Chon-Fai and Bessafi, Miloud and Cadet, Fr{\'e}d{\'e}ric},
  journal={arXiv preprint arXiv:2609.17926}, year={2026}
}
@article{shutman2025words,
  title={Learning words in groups: fusion algebras, tensor ranks and grokking},
  author={Shutman, Maor and Louidor, Oren and Tessler, Ran},
  journal={arXiv preprint arXiv:2509.06931}, year={2025}
}
@article{he2026mechanism,
  title={On the Mechanism and Dynamics of Modular Addition: Fourier Features, Lottery Ticket, and Grokking},
  author={He, Jianliang and Wang, Leda and Chen, Siyu and Yang, Zhuoran},
  journal={arXiv preprint arXiv:2602.16849}, year={2026}
}
@article{he2026groups,
  title={Neural Networks Provably Learn Spectral Representations for Group Composition},
  author={He, Jianliang and Wang, Leda and Zhang, Fengzhuo and Chen, Siyu and Yang, Zhuoran},
  journal={arXiv preprint arXiv:2606.02993}, year={2026}
}
@article{levine2026math,
  title={Math for {AI} safety: an invitation for mathematicians},
  author={Levine, Lionel},
  journal={arXiv preprint arXiv:2609.15289}, year={2026}
}
@article{hoff2017lasso,
  title={Lasso, fractional norm and structured sparse estimation using a {H}adamard product parametrization},
  author={Hoff, Peter D.},
  journal={Computational Statistics \& Data Analysis}, volume={115}, pages={186--198}, year={2017}
}
@inproceedings{dai2021representation,
  title={Representation Costs of Linear Neural Networks: Analysis and Design},
  author={Dai, Zhen and Karzand, Mina and Srebro, Nathan},
  booktitle={Advances in Neural Information Processing Systems}, volume={34}, year={2021}
}
@inproceedings{kolb2025deep,
  title={Deep Weight Factorization: Sparse Learning Through the Lens of Artificial Symmetries},
  author={Kolb, Chris and Weber, Tobias and Bischl, Bernd and R{\"u}gamer, David},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2025}
}
@article{ongie2026representation,
  title={Representation Costs in Data Science: Foundations and the Quasi-{B}anach Spaces of Deep Neural Networks},
  author={Ongie, Greg and Parhi, Rahul},
  journal={arXiv preprint arXiv:2606.14954}, year={2026}
}
@inproceedings{chizat2020implicit,
  title={Implicit Bias of Gradient Descent for Wide Two-layer Neural Networks Trained with the Logistic Loss},
  author={Chizat, L{\'e}na{\"i}c and Bach, Francis},
  booktitle={Conference on Learning Theory (COLT)}, series={PMLR}, volume={125}, pages={1305--1338}, year={2020}
}
@inproceedings{tsilivis2025flavors,
  title={Flavors of Margin: Implicit Bias of Steepest Descent in Homogeneous Neural Networks},
  author={Tsilivis, Nikolaos and Gronich, Eitan and Kempe, Julia and Vardi, Gal},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2025}
}
@inproceedings{xie2024adamw,
  title={Implicit Bias of {AdamW}: $\ell_\infty$-Norm Constrained Optimization},
  author={Xie, Shuo and Li, Zhiyuan},
  booktitle={International Conference on Machine Learning (ICML)}, series={PMLR}, volume={235}, pages={54488--54510}, year={2024}
}
@article{jacobs2026mirror,
  title={Implicit Bias of Mirror Flow in Homogeneous Neural Networks: Sparse and Dense Feature Learning},
  author={Jacobs, Tom and Mont{\'u}far, Guido},
  journal={arXiv preprint arXiv:2605.19458}, year={2026}
}
@inproceedings{lyu2024dichotomy,
  title={Dichotomy of Early and Late Phase Implicit Biases Can Provably Induce Grokking},
  author={Lyu, Kaifeng and Jin, Jikai and Li, Zhiyuan and Du, Simon S. and Lee, Jason D. and Hu, Wei},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2024}
}
@inproceedings{mohamadi2024grok,
  title={Why Do You Grok? A Theoretical Analysis of Grokking Modular Addition},
  author={Mohamadi, Mohamad Amin and Li, Zhiyuan and Wu, Lei and Sutherland, Danica J.},
  booktitle={International Conference on Machine Learning (ICML)}, year={2024}
}
@inproceedings{marchetti2024harmonics,
  title={Harmonics of Learning: Universal {F}ourier Features Emerge in Invariant Networks},
  author={Marchetti, Giovanni Luca and Hillar, Christopher and Kragic, Danica and Sanborn, Sophia},
  booktitle={Conference on Learning Theory (COLT)}, year={2024}
}
@inproceedings{stander2024cosets,
  title={Grokking Group Multiplication with Cosets},
  author={Stander, Dashiell and Yu, Qinan and Fan, Honglu and Biderman, Stella},
  booktitle={International Conference on Machine Learning (ICML)}, series={PMLR}, volume={235}, pages={46441--46467}, year={2024}
}
@inproceedings{wu2025unified,
  title={Towards a unified and verified understanding of group-operation networks},
  author={Wu, Wilson and Jaburi, Louis and Drori, Jacob and Gross, Jason},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2025}
}
@inproceedings{zhong2023clock,
  title={The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks},
  author={Zhong, Ziqian and Liu, Ziming and Tegmark, Max and Andreas, Jacob},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2023}
}
@inproceedings{kunin2025agf,
  title={Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks},
  author={Kunin, Daniel and Marchetti, Giovanni Luca and Chen, Feng and Karkada, Dhruva and Simon, James B. and DeWeese, Michael R. and Ganguli, Surya and Miolane, Nina},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2025}
}
@inproceedings{tian2025composing,
  title={Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets},
  author={Tian, Yuandong},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2025}
}
@inproceedings{razin2021tensor,
  title={Implicit Regularization in Tensor Factorization},
  author={Razin, Noam and Maman, Asaf and Cohen, Nadav},
  booktitle={International Conference on Machine Learning (ICML)}, year={2021}
}
@inproceedings{marchetti2026sequential,
  title={Sequential Group Composition: A Window into the Mechanics of Deep Learning},
  author={Marchetti, Giovanni Luca and Kunin, Daniel and Myers, Adele and Acosta, Francisco and Miolane, Nina},
  booktitle={International Conference on Machine Learning (ICML)}, year={2026}
}
@article{ding2024survival,
  title={Survival of the Fittest Representation: A Case Study with Modular Addition},
  author={Ding, Xiaoman Delores and Guo, Zifan Carl and Michaud, Eric J. and Liu, Ziming and Tegmark, Max},
  journal={arXiv preprint arXiv:2405.17420}, year={2024}
}
@article{elhage2022superposition,
  title={Toy Models of Superposition},
  author={Elhage, Nelson and Hume, Tristan and Olsson, Catherine and Schiefer, Nicholas and Henighan, Tom and Kravec, Shauna and Hatfield-Dodds, Zac and Lasenby, Robert and Drain, Dawn and Chen, Carol and Grosse, Roger and McCandlish, Sam and Kaplan, Jared and Amodei, Dario and Wattenberg, Martin and Olah, Christopher},
  journal={arXiv preprint arXiv:2209.10652}, year={2022}
}
@article{larson2026rank,
  title={The Rank the Task Demands: A Causal Rank Law for Matrix Memories Trained on Group Composition},
  author={Larson, Samuel},
  journal={arXiv preprint arXiv:2609.12259}, year={2026}
}
@article{memana2026counterexample,
  title={A Counterexample to Fourier Alignment in Single-Neuron Modular Addition},
  author={Memana, Gautam Neelakantan},
  journal={arXiv preprint arXiv:2608.04451}, year={2026}
}
```

## 6. Searches run (for the record)
WebSearch queries included: modular addition max margin weight decay number of frequencies log p; grokking few frequencies weight decay 2026; L2 regularization max-margin sparse frequencies Morwani; O(log p) frequencies suffice; group composition irreps cost matrix multiplication tensor rank; max-margin homogeneous L2 sparse features quasi-norm feature count; number of frequencies grows with modulus; polynomial activation degree fewer Fourier features; Morwani extension weight decay irreps count 2026; worst-case offset 1-cos budget; L_{2,b} margin exponent below homogeneity; feature ablation margin budget bound; Tibshirani/Hoff product parametrization; representation cost Dai-Karzand-Srebro; MAIS registry; plus venue lookups. Semantic Scholar citations of Morwani (arXiv:2311.07568) pulled via API (~40 citing papers listed; the relevant ones are all above). OpenReview pages were blocked by a browser check; ICLR venues confirmed via iclr.cc / proceedings.iclr.cc search hits or arXiv comments.

Gaps: concurrent ICLR 2027 submissions invisible; S2 citation list for Morwani appears incomplete; harmonic-analysis / additive-combinatorics prior art for Gamma_s (Chowla cosine problem, sidelobe design, Alon-Roichman) is outside this ML sweep.
