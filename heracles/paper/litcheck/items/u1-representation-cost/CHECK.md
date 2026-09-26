# U1-representation-cost — CHECK

Location: sections/approach.tex, line 45, last sentence: "The exponent comes from homogeneity: a feature is a product of nu weight factors, and at fixed product their total L_{2,b} cost is least when they are equal (AM--GM), so amplitude A costs a multiple of A^{b/nu}=A^{1/s}." At present nothing is cited there.

Verdict: CITATION_FOUND. This is a textbook fact and should be credited to others.

## Fetched and confirmed sources
1. Tibshirani, "Equivalences Between Sparse Models and Neural Networks", tech note, 15 Apr 2021. PDF: https://www.stat.berkeley.edu/~ryantibs/papers/sparsitynn.pdf (full text read).
   - Sec 2.2 "Multi-layer case", eqs. (8)->(9): min_{prod_l a_l = c} sum_l a_l^2 = k|c|^{2/k}, and the minimum is attained at |a_l| = |c|^{1/k}. The note says: "we can interpret this as saying that the AM-GM inequality ... is tight when a_1 = ... = a_k". So a k-layer linear net with weight decay equals l_p-penalised regression with p = 2/k < 1, which is nonconvex and sparse, and "even more so for greater depth".
   - Sec 3.2 gives the group version: an l_p group lasso with p = 2/k.
   - This is our exact sentence in the case b = 2. Ours generalises it to an L_{2,b} cost, which gives A^{b/nu}. That generalisation is a one-line change.
   - This is the closest match and the one to cite first.
2. Hoff, "Lasso, fractional norm and structured sparse estimation using a Hadamard product parametrization", arXiv 1611.00040 (2016, rev. 2017). The abstract says "a subclass of L_q penalties with q<=1 can be expressed as sums of L_2 penalties". The journal ref (CSDA 115:186-198) comes from the sweep and was not visible on the fetched arXiv page.
3. Dai, Karzand, Srebro, "Representation Costs of Linear Neural Networks: Analysis and Design", NeurIPS 34 (2021). Confirmed on the proceedings page. It covers the "regularization cost in predictor space induced by l2 regularization on the parameters", including which l_p and group norms arise.
4. Gunasekar, Lee, Soudry, Srebro, "Implicit Bias of Gradient Descent on Linear Convolutional Networks", NeurIPS 31 (2018). Confirmed on papers.nips.cc. Depth-L linear conv nets converge to a predictor "related to the l_{2/L} bridge penalty in the frequency domain".
   - This is a nice parallel: the exponent 2/depth produces sparsity over Fourier frequencies.
   - It is about implicit bias (gradient descent), not explicit weight decay, so cite it as "cf.".
5. Kolb, Weber, Bischl, Rügamer, "Deep Weight Factorization: Sparse Learning Through the Lens of Artificial Symmetries", ICLR 2025 (arXiv 2502.02496, venue from the arXiv page). L2 on D factors is equivalent to nonconvex sparse regularisation.
6. Ongie, Parhi, arXiv 2606.14954 (2026). Weight-decay representation cost of depth-L ReLU nets is a quasi-norm with a nonconvex ball for L > 2. Optional.

## Considered, not recommended here
- Woodworth et al. 2020 (arXiv 2002.09277), "Kernel and Rich Regimes". This is about gradient-flow implicit bias and init scale, not the weight-decay-to-quasi-norm identity, so it is off-point for this sentence.
- Jacobs-Montufar 2605.19458. Already in the sweep. It is about mirror-flow potentials and belongs in the related-work sentence on "exponent decides sparse/dense", not here.

## Limits of this check
- The WebSearch budget ran out and Semantic Scholar returned 429, so the Tibshirani venue was checked only from the PDF title page. The PDF shows no venue, so cite it as a technical note or manuscript.
