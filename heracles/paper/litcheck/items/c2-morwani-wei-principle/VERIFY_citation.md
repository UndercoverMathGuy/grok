# C2-morwani-wei-principle: citation verification (task 76)

## Sources I fetched this session
- Wei, Lee, Liu, Ma, arXiv 1810.05369v4. I read the full PDF text.
- The NeurIPS 2019 proceedings page (vol. 32).
- Morwani et al., arXiv 2311.07568: abstract page ("Spotlight at ICLR 2024") and the full PDF text.
- Rosset, Zhu, Hastie, "Margin Maximizing Loss Functions", NIPS 16 (2003), proceedings page. Optional citation.

## Wei et al. 2019: CONFIRMED
**Setup (Sec. 4).** f(.;Theta) is a-positive-homogeneous and continuous in Theta. The norm is general. The objective is L_lambda = mean binary logistic loss + lambda ||Theta||^r with r>0 fixed (eq. 4.1). Theta_lambda is in argmin L_lambda; existence is Claim C.3.

**Theorem 4.1, verbatim.** "Assume the training data is separable by a network f(.;Theta*) in F with an optimal normalized margin gamma*>0. Then, the normalized margin of the global optimum of the weakly-regularized objective (equation 4.1) converges to gamma* as the regularization goes to zero." That is, gamma_lambda -> gamma* as lambda -> 0.

**Multi-class version.** The binary statement is Thm 4.1. The multi-class cross-entropy version is Theorem C.1 (App. C.1, eqs. C.1-C.2, margin f_y - max_{j!=y} f_j): "Assume gamma*>0 in the multi-class setting with cross entropy loss. Then as lambda->0, gamma_lambda->gamma*." Thm 4.1 itself is proved by reduction to C.1 with l=2 (App. C.3).

**Fit to our paper.** L_{2,b} is a norm for b>=1. The regularizer is ||.||_{2,b}^b, so r=b>0. Product and power neurons are continuous and nu-homogeneous, and the width is fixed. All hypotheses hold.

**Caveat.** The theorem is about the normalized margin *value* of global minimisers. The phrase "maximum margin solutions are approached" follows in one line: the sphere is compact and h is continuous, so every limit point of Theta_lambda/||Theta_lambda|| is a max-margin solution. Wei does not state this in those words. It is fine for an intro.

**Bib.** Title, authors and venue match. Optionally add volume={32}.

## Morwani et al. 2024: CONFIRMED
- Sec. 2 contains: "Theorem 1 (Wei et al. (2019a), Theorem 4.1). For any norm, a fixed r>0 and any homogeneous function f with homogeneity constant nu>0, if gamma*>0, then lim_{lambda->0} gamma_lambda = gamma*."
- The next sentence is: "This provides the motivation behind studying maximum margin classifiers as a proxy for understanding the global minimizers of L_lambda as lambda->0."
- The abstract says: "our main technique demonstrates how the principle of margin maximization alone can be used to fully specify the features learned by the network."
- Bib is correct: 5 authors, ICLR 2024.

## NEW ISSUES (the checker missed these; intro.tex changed after the check, mtime 09-25 18:29)
The current intro.tex l.4 reads: "\citet{morwani2024feature} introduced the principle of \emph{margin maximization} that enabled the analysis of the cause of feature emergence without training dynamics. This is backed \citep{wei2019regularization}, showed that such maximum margin solutions are approached with vanishing regularization."

The sentence was being edited while I read it. An earlier read the same minute showed a different, also broken, version.

1. **Grammar and citation form are broken.** "This is backed \citep{...}, showed that ..." has no subject for "showed", and \citep renders "(Wei et al., 2019)". Fix: "\citet{wei2019regularization} showed that ...".
2. **"Introduced the principle of margin maximization" overclaims.** Morwani et al. did not introduce margin maximization. Their own related-work section credits Soudry et al. 2018, Wei et al. 2019, Lyu & Li 2019, Ji & Telgarsky 2020 and Chizat & Bach 2020, and Wei credits Rosset et al. 2003/2004 for the linear case. What Morwani contributed is showing that margin maximization *alone* can fully specify the learned features in algebraic tasks, which is the wording of their abstract. Suggested text: "\citet{morwani2024feature} showed that margin maximization alone determines which features emerge, without analysing training dynamics: by \citet{wei2019regularization}, the global minimizers of weakly regularized cross-entropy approach maximum margin as the regularization vanishes."
3. **Precision fix still stands.** approach.tex l.3 should cite Theorem 4.1 together with Theorem C.1, the multi-class cross-entropy version.

## Optional canonical addition
Rosset, Zhu, Hastie, "Margin Maximizing Loss Functions", NIPS 16 (2003). This is the linear-model precursor that Wei et al. build on ("inspired by the result of Rosset et al."). Cite it only if the intro wants a lineage for "vanishing regularization -> max margin". It is not required.

## Harness note
WebSearch returned "this session has used its web search budget (200 of 200 WebSearch calls)". The litcheck verify agents appear to share one session-wide search cap. Once it is spent, every later verifier gets no searches. That can make the stage slow or degrade novelty searches. Raise CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION or split the runs.

## Verdict
The citations and the checker's SUPPORTED verdict hold for the claim as itemised. The current intro wording needs the fixes in points 1-2.
