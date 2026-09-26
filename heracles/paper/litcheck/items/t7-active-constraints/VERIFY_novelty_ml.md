# VERIFY novelty_ml: T7-active-constraints (task 53)

**Item.** Appendix proof of Prop. bottleneck (appendix.tex:164-170): "At a vertex, the active constraints have rank |F_p|", so the coverage rows for the tight offsets, restricted to the support S, have rank K. This gives K linearly independent tight offsets and an invertible KxK system for y_S.

**Checker verdict.** KNOWN (textbook LP). **My verdict.** Upheld: KNOWN. This is a standard LP fact, and our only contribution is the interpretation.

## What I checked (fetched this session)
- **arXiv 2302.12364** (Liu, Bunea, Niles-Weed, "Asymptotic confidence sets for random linear programs"). I extracted the PDF text myself. Verbatim: "basic feasible solutions are precisely extreme points (vertices) of the feasible set of (1) (Bertsimas and Tsitsiklis, 1997, Theorem 2.3)". Its bib entry: "D. Bertsimas and J. Tsitsiklis. Introduction to Linear Optimization. Athena Scientific, 1st edition, 1997. ISBN 1886529191." This confirms the theorem number and the bib fields.
- **Wikipedia, "Basic solution (linear programming)".** Gives "Of all the constraints that are active at that vector, at least n of them must be linearly independent" and cites Bertsimas & Tsitsiklis (1997), Athena Scientific, p. 50, ISBN 978-1-886529-19-9. This matches the checker's Def. 2.9 on p. 50. The bibtex is correct.
- **Rosset, Zhu, Hastie (2004)**, JMLR 5:941-973, "Boosting as a Regularized Path to a Maximum Margin Classifier". I read the PDF text myself.
  - This is the ML-lens precedent for the same rank argument in margin maximisation. Appendix B, Thm 3 and its corollaries: the l1-constrained and l1-margin-maximising solutions have "at most n (the number of observations) non-zero coefficients". The proof is that H_A has rank at most n, so a support larger than n is linearly dependent (Sec. 5.1.1 and App. B).
  - Our s=1 programme is exactly an l1-margin (LP-boosting) problem: frequencies play the weak learners and offsets play the observations. So "support size <= number of independent tight offsets" is the known ML sparsity fact.
  - Our statement is the sharper vertex version: the K tight offsets are linearly independent on S. It is not new in the ML literature either. It is the standard basic-solution argument.
- **Wikipedia, "Remez algorithm".** Confirms the Chebyshev analogue the checker mentions:
  - n+1 coefficients plus the level E are pinned by n+2 reference points.
  - Here the level is normalised to 1 in P_p, so K free coordinates are pinned by K tight offsets.
  - The analogy is fine as an informal remark, but it is not needed. For s>1 the objective is concave rather than a linear minimax, so I would not lean on it.

## ML-lens adversarial search
- I found no ML paper that states the "tight offsets = simultaneous hardest wrong answers" reading for modular-addition max-margin. Morwani et al. obtain the dense case, where all wrong classes are equally tight. That reading stays ours, but it is interpretive, not technical.
- LIMITATION: WebSearch was unavailable. The session's 200-search budget was already used up when this verifier started ("200 of 200 WebSearch calls"). So the 2025-2026 arXiv sweep was done by direct fetches only, not by keyword search. For a textbook step, this does not change the verdict.

## Citation checks
- bertsimas1997: exists. Authors, title, publisher and year are correct. ISBN 978-1-886529-19-9 (= 1886529191). Thm 2.3 (vertex = extreme point = BFS) is confirmed by a secondary quote. Def. of basic solution (p. 50) is confirmed via Wikipedia. I did not fetch the book itself: athenasc.com failed the SSL handshake.
- Recommended edit (checker's): "At a vertex, the active constraints have rank |F_p| (a vertex is a basic feasible solution; \citealp[Thm.~2.3]{bertsimas1997})". This is correct and sufficient.
  - B&T's Thm 2.3 also covers the "extreme point" definition used in Prop. shapes (i), so one citation serves both steps.
- Optional: rosset2004boosting, for a one-clause ML pointer if the text notes that s=1 is the l1-margin/LP-boosting case with support bounded by the tight constraints. It is not required.
- Do not add the Chebyshev remark without a canonical approximation-theory source. None was fetched here.
