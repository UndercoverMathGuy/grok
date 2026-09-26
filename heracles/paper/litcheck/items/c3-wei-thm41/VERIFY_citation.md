# VERIFY (citation lens): C3-wei-thm41, task 77

Claim (approach.tex:3): "By \citet[Theorem~4.1]{wei2019regularization}, the global minimizers of cross-entropy plus lambda||theta||_{2,b}^b attain normalized margin tending to gamma*_b as lambda -> 0."

## What I fetched this session
- arXiv 1810.05369v4 PDF (curl + pdftotext), dated 25 Apr 2020.
- NeurIPS 2019 camera-ready PDF (proceedings.neurips.cc/paper_files/paper/2019/file/8744cf92c88433f8cb04a02e6db69a0d-Paper.pdf).
- NeurIPS abstract page and official Bibtex (…-Bibtex.bib): Advances in Neural Information Processing Systems 32, volume={32}, year={2019}, authors Wei, Colin; Lee, Jason D; Liu, Qiang; Ma, Tengyu.

## Checked against the source
- Sec. 4, eq. (4.1): f(.;Theta) is a-positive-homogeneous in Theta (a>0) and continuous. The loss is (1/n) sum log(1+exp(-y_i f)) + lambda||Theta||^r for "some general norm ||.||" and fixed r>0. gamma_lambda = min_i y_i f(x_i; Theta_bar_lambda), with Theta_bar = Theta/||Theta|| (notation Sec. 1.2). gamma* = max_{||Theta||<=1} min_i y_i f.
- Theorem 4.1 (identical number in arXiv v4 and in the NeurIPS camera-ready): if the data are separable with gamma*>0, then gamma_lambda -> gamma* as lambda -> 0, for the global optimum Theta_lambda in argmin L_lambda. This is the BINARY logistic case.
- The text after Thm 4.1 says the result extends to multi-class CE (Section C). App. C.1, eq. (C.1): softmax CE + lambda||Theta||^r. Margin (C.2): f_y - max_{j!=y} f_j of Theta_bar_lambda. **Theorem C.1**: "Assume gamma*>0 in the multi-class setting with cross entropy loss. Then as lambda -> 0, gamma_lambda -> gamma*." Claim C.3 shows a minimizer exists. Sec. C.3 proves Thm 4.1 by reduction to C.1.

## How the hypotheses match
- Product and power neurons are nu-positive-homogeneous and polynomial, hence continuous: a = nu.
- ||.||_{2,b} with 2<=b<=nu is an l_b norm of l_2 norms, so it is a genuine norm (b>=1). Wei allows any norm, so b != nu is covered.
- Regulariser exponent r = b is allowed (any r>0).
- The paper's normalized margin h(theta)/||theta||_{2,b}^nu equals Wei's gamma_lambda = h(Theta_bar) by a-homogeneity. gamma*_b equals Wei's gamma* over the unit ball. The directions agree.
- The claim covers global minimizers only. The paper's "Nothing here concerns whether gradient descent finds them" matches this scope.
- gamma*_b > 0 holds on the full table (a positive-margin construction exists). This is Wei's separability hypothesis.

## Verdict
The substance is fully supported. The only defect is the pointer: modular addition uses multi-class CE, which is Theorem C.1, not Theorem 4.1 (binary logistic). The checker's PARTIALLY_SUPPORTED verdict and its fix, "[Theorem~4.1; Theorem~C.1 for multi-class cross-entropy]" or just "[Theorem~C.1]", are correct. Note that in the NeurIPS version Thm C.1 is in the supplementary appendix, while the arXiv version includes it inline.

## Bib
The current entry lacks volume={32} and has booktitle "Advances in Neural Information Processing Systems (NeurIPS)". Add volume={32} (and editor/publisher if wanted), following the official NeurIPS bibtex. The checker's recommended bibtex is correct.

## Other issue found (neighbouring sentence, not this item)
intro.tex:4 is grammatically broken where it cites Wei: "...enabled the analysis of the emergence of features without training dynamics as \citep{wei2019regularization}, showed that such maximum margin solutions are approached with." A clause is missing: it should say that Wei et al. showed max-margin solutions are approached by global minimizers of weakly regularized CE. This belongs to item c2 but should be fixed.

## Better citation?
None needed. Wei et al. is the canonical statement for general homogeneous nets and general norms. The linear precursor is Rosset, Zhu and Hastie (2004), which Wei et al. credit; it is optional and I did not fetch it.
