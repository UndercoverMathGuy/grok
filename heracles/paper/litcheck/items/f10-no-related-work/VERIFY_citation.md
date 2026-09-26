# VERIFY (citation lens) -- F10-no-related-work (task 74)

Verdict: checker's SUPPORTED is UPHELD. All eight recommended citations exist and say what is claimed. Two bib entries need venue upgrades, and the existing intro sentence citing McCracken misattributes the cause of the transition.

## Source check (repo, 2026-09-25)
- The TODO is now at `sections/intro.tex:12`, not line 22: the file is being edited live, and the checker's line number is stale.
- No related-work section exists.
- Citation counts over paper.tex and sections/*.tex: lyu2020gradient 0, gromov2023grokking 0, li2025nsum 0, power2022grokking 0. The item's claim is confirmed.
- The `mccracken2025` bib entry has the wrong title ("Emergence of sparse and dense circuits...") and the wrong author ("Andrew McCracken and others"). Confirmed.

## Pages fetched this session
- arXiv 2402.09469v4 (Li, Liang, Shi, Song, Zhou; comment "AIStats 2025"). Thm 4.1 uses neurons (u_1(a_1)+...+u_k(a_k))^k w with the L_{2,k+1} margin and gamma* = 2(k!)/((2k+2)^{(k+1)/2}(p-1)p^{(k-1)/2}). It states: "For each frequency zeta in {1,...,(p-1)/2}, there exists one neuron using this frequency only." PMLR entry proceedings.mlr.press/v258/li25b.html gives pages 523-531, which matches the checker's bibtex.
- arXiv 2505.18266 (McCracken, Moisescu-Pareja, Letourneau, Precup, Love). Title: "Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks".
  - Cor 4.8: "Learning O(log n) distinct frequencies gives a logit margin Omega(log n)".
  - Transition: "Adding either depth, or a trainable embedding matrix for inputs, causes the network to transition from learning floor(n/2) types of neurons to much fewer types."
  - "we don't know why it occurs."
  - The OpenReview API (note zuHs6RHQwT) gives venue "NeurIPS 2025 poster", venueid NeurIPS.cc/2025/Conference, so the paper is PUBLISHED at NeurIPS 2025.
- arXiv 2605.19458 (Tom Jacobs, Guido Montufar, 19 May 2026). Thm 4.9: with R the hyperbolic entropy (alpha=1) or a smoothed homogeneous mirror potential (alpha=p>=2), max-margin is reformulated as min sum |a_j|^{alpha/2} s.t. ||w_j||_{L_alpha}=1. alpha=1 gives sparse and p=3 gives dense. There is no modular or Fourier content.
- arXiv 2410.01779 (Tian): comment "NeurIPS 2025 (camera ready)". Abstract: "weight decay favors simpler solutions". Semi-ring structure for quadratic nets.
- arXiv 2306.17844 (Zhong, Liu, Tegmark, Andreas; NeurIPS 2023): "small changes to model hyperparameters and initializations can induce the discovery of qualitatively different algorithms".
- arXiv 2406.03495 (Doshi, He, Das, Gromov). The PDF footer reads "ICLR 2024 Workshop BGPT". The paper uses phi(x)=x^S with cosine weights and iid random phases psi_k. Cross-terms are suppressed by 1/N and "vanish due to random phase approximation". There is no margin or norm analysis. OpenReview/DBLP list it only as CoRR 2024.
- arXiv 2301.02679 (Gromov): "these feature maps are also found by vanilla gradient descent as well as AdamW". arXiv only.
- arXiv 1906.05890 (Lyu, Li; ICLR 2020): the smoothed normalized margin increases, and the normalized margin converges to a KKT-point objective value.
- arXiv 1810.05369 (Wei, Lee, Liu, Ma): "the global minimizer of a weakly-regularized cross-entropy loss is the max normalized margin solution among all neural nets."
- arXiv 2201.02177 (Power et al.): arXiv. The bib is fine.

## Issues
1. **Misattribution in the existing text** (`intro.tex:6`): "the count moves from dense to sparse as the degree of the network changes \citep{mccracken2025}". McCracken et al. attribute the transition to adding DEPTH or TRAINABLE EMBEDDINGS, and say they do not know why it happens. They say nothing about degree or homogeneity. Suggested rewording: "and the count drops from about p/2 to O(log p) once depth or trainable embeddings are added \citep{mccracken2025}".
2. **mccracken2025 venue:** it is NeurIPS 2025, not an arXiv preprint. Use @inproceedings (corrected bibtex below). Note the accent in Létourneau.
3. **Overclaim in the checker's suggested text:** "our Theorem [scaling] supplies the matching upper bound". McCracken's result is sufficiency for an UNNORMALISED logit margin. thm:scaling is about normalised L_{2,b} maximizers: they use Theta(log p) frequencies, and K=o(log p) is suboptimal. The two quantities differ. Suggested wording: "we show that Theta(log p) frequencies are necessary and sufficient at the normalised L_{2,b} maximum".
4. **Jacobs-Montufar gloss:** the exponent is that of the mirror potential (the optimiser geometry), not a weight-norm penalty or the network's homogeneity. Say "the exponent of the mirror potential separates sparse from dense max-margin solutions". Sparse and dense are illustrated with experiments (hyperbolic entropy vs p=3), not proved as a count law.
5. **doshi2024polynomials venue:** the PDF is marked "ICLR 2024 Workshop BGPT". Cite it as that workshop, or keep the arXiv form. Both are acceptable.
6. **wei2019regularization wording** (`intro.tex:4`, current live text): "maximum margin solutions are approached by networks when trained with vanishing regularization". Wei et al. prove this for the GLOBAL MINIMISER of the weakly regularised loss, not for trained networks. The checker's phrasing ("weak regularisation approaches the global maximum") is fine. The live intro sentence should say "global minimisers" (it is also syntactically broken right now: an unclosed $).
7. li2025nsum, tian2025composing, zhong2023clock, gromov2023grokking and lyu2020gradient are correct as the checker gave them. The existing li2025nsum entry lacks volume and pages; the checker's version adds PMLR 258:523-531.
