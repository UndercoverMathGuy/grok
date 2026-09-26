# C2-morwani-wei-principle — cited_claim check

Sentence (intro.tex l.4): "Morwani et al. showed that for algebraic tasks the answer follows from one principle: with vanishing regularization, networks approach the maximum margin solution [Wei et al. 2019], which can be characterized analytically."
Related (approach.tex l.3): "By [Wei et al., Theorem 4.1], the global minimizers of cross-entropy plus lambda||theta||_{2,b}^b attain normalized margin tending to gamma*_b as lambda->0".

## Verdict: SUPPORTED (one small precision fix recommended)

### Wei, Lee, Liu, Ma, NeurIPS 2019 (arXiv 1810.05369; proceedings page fetched, title/authors/venue match refs.bib)
- Sec 4 setting: f(.;Theta) a-positive-homogeneous in Theta, continuous in Theta; ANY general norm ||.||; objective = binary logistic loss + lambda||Theta||^r, r>0 fixed (eq. 4.1); Theta_lambda in argmin (existence: Claim C.3).
- Theorem 4.1 (verbatim): "Assume the training data is separable by a network f(.;Theta*) in F with an optimal normalized margin gamma* > 0. Then, the normalized margin of the global optimum of the weakly-regularized objective (equation 4.1) converges to gamma* as the regularization goes to zero. gamma_lambda -> gamma* as lambda -> 0."
- Thm 4.1 is stated for BINARY logistic loss. The multi-class cross-entropy version is Theorem C.1 (App. C, eq. C.1-C.2, margin = f_y - max_{j!=y} f_j): "Assume gamma* > 0 in the multi-class setting with cross entropy loss. Then as lambda -> 0, gamma_lambda -> gamma*." Text after Thm 4.1: "it extends to the multi-class setting with cross-entropy loss ... Section C."
- Conditions vs our paper: L_{2,b} with 2<=b<=nu is a true norm (b>=1); r=b>0; neurons polynomial => continuous and nu-homogeneous; width fixed. All hypotheses met. Result concerns GLOBAL minimisers only (paper already says "Nothing here concerns whether gradient descent finds them").

### Morwani et al. ICLR 2024 (arXiv 2311.07568, fetched)
- Sec 2 quotes "Theorem 1 (Wei et al. (2019a), Theorem 4.1). For any norm ||.||, a fixed r>0 and any homogeneous function f with homogeneity constant nu>0, if gamma*>0, then lim gamma_lambda = gamma*." then: "This provides the motivation behind studying maximum margin classifiers as a proxy for understanding the global minimizers of L_lambda as lambda->0." Abstract: "the principle of margin maximization alone can be used to fully specify the features learned by the network." So attributing the "one principle" framing to Morwani is accurate.

## Issues
1. Minor: Thm 4.1 is binary logistic; our setting is multi-class CE -> cite "Theorem 4.1 and its multi-class version Theorem C.1" (approach.tex). Morwani also cites only Thm 4.1, so this is not an error, just precision.
2. Minor wording: "networks approach" really means "global minimisers of the regularised loss approach (in normalized margin)". Intro is informal and approach.tex makes it precise; acceptable. Optional: "the global minimizers of weakly regularized cross-entropy approach the maximum margin solution".

## Fix
approach.tex: "By \citet[Theorem~4.1]{wei2019regularization}" -> "By \citet[Theorem~4.1; multi-class version Theorem~C.1]{wei2019regularization}".
Optional intro: "with vanishing regularization, the global minimizers of the regularized loss approach the maximum margin solution".
Bib entries correct; optionally add author "Liu, Qiang" already present, and pages 9709--9721 not verified (omit).
