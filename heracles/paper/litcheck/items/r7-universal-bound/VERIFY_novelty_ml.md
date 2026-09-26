# VERIFY (novelty_ml): R7-universal-bound (thm:universal)

Verdict: the checker's PARTIALLY_KNOWN holds.
- Known: the b = nu endpoint (Morwani Thm 7; Li et al. AISTATS 2025) and the pricing ingredient (weight decay on a product of factors gives an l_{2/L} (here b/nu) quasi-norm cost on Fourier coefficients).
- Not found anywhere: an upper bound on the max L_{2,b} margin for modular addition when b < nu, valid for every width and every kind of neuron.

## Tooling note
WebSearch was unusable: the session's budget was spent (200/200). So the search went through the arXiv API (export.arxiv.org, which later rate-limited to empty responses), the arXiv listing search and direct WebFetch of primary pages.

## Adversarial search (arXiv abstract field)
Queries run: "modular addition"+margin; "modular arithmetic"+margin; grokking+margin; "modular addition"+"weight decay"; "modular addition"+sparse+frequencies; homogeneous+margin+Fourier; "weight decay"+"number of features"+margin; mirror flow sparse dense; grokking+"weight decay"+frequencies; and the arXiv listing search "modular addition margin" (39 hits, newest first).
None of them has a max-margin upper bound for b < nu. Papers checked by abstract or HTML:
- 2609.26679 (Pracher et al., 22 Sep 2026), spectral theory of grokking: homogeneous nets under L2 weight decay, but about NTK dynamics and timescales. No margin bound.
- 2606.08985 (Tan, Gai, Zhang 2026): Schatten / weight-decay surrogates in modular arithmetic, used for rank-2 cyclic vs simplex ETF. It does not count frequencies and has no margin bound. Related at most.
- 2511.23443 (Huang, Li 2025), sine networks: a margin-based *generalization* bound. It does not bound the maximum margin.
- 2602.16849 (He et al.), 2607.23967, 2603.13331, 2606.05863: dynamics only.
- 2410.01779v4 (Tian), Thm 5 (fetched): "there exists a path of zero loss ... As a result, lower-order solutions are preferred if trained with L2 regularization." No margin bound and no frequency count. The fetched page gives no venue; the checker's "NeurIPS 2025" is unverified here, but it is not a recommended citation.
- 2605.19458 (Jacobs, Montufar): mirror flow, sparse vs dense. No modular addition and no L_{2,b} bound.

## Checker's confirmed citations (all re-fetched)
- morwani2024feature: OK. Thm 7: max L_{2,3} margin sqrt(2/27)/(p^{1/2}(p-1)) for m >= 4(p-1). The paper "primarily focus[es] on maximum margin with respect to L_{2,nu} norm", with no L_{2,2} analysis.
- li2025fourier: the content is OK (PMLR 258:523-531; m >= 2^{2k-2}(p-1); max L_{2,k+1} margin). **But refs.bib already has this paper as `li2025nsum`.** Adding li2025fourier would duplicate it, so reuse li2025nsum and add pages 523--531 and series PMLR vol 258.
- hoff2017lasso: OK. The quote "a subclass of L_q penalties with q<=1 can be expressed as sums of L_2 penalties" is verified. Crossref confirms the journal version: Computational Statistics & Data Analysis 115:186-198 (2017), doi 10.1016/j.csda.2017.06.007. Cite it as @article in CSDA, not as an arXiv preprint.
- dai2021representation: OK, NeurIPS 34 (2021). PDF Table 1: "Diagonal network ... l_{2/d,2} quasi-norm"; "Convolutional Network ... l_{2/d,2} quasi-norm on Fourier domain". This supports the claim.
- fan2023euclidean: OK, TMLR Jan 2023. Euclidean norms of CP components are "monotonic transformations of tensor Schatten-p quasi-norm".
- kam2026symmetry: OK. arXiv 2609.17926, 15 Sep 2026. App. A.3.2 has Eq (15), the max over ||beta||_2 = B of min_s sum_k beta_k(1-cos(omega_k s)). Prop 5 ("Optimal allocation") makes the uniform allocation optimal. The quadratic net sigma(z)=z^2 is used, and the l2 budget is justified by weight decay ("Weight decay does not permit unbounded logits..."). There is no universal bound. The contrast in the recommended text is correct.

## Missed by the checker (the most canonical precedent for the pricing step)
- Gunasekar, Lee, Soudry, Srebro, "Implicit Bias of Gradient Descent on Linear Convolutional Networks", NeurIPS 31 (2018), arXiv 1806.00468 (NeurIPS page fetched). Its abstract: GD on depth-L linear conv nets "converges to a linear predictor related to the l_{2/L} bridge penalty in the frequency domain".
  This is the Fourier-domain version of "amplitude A costs A^{b/nu}" at b = 2 (nu = L factors) and predates Hoff and Fan. It should head the pricing citation.

## Suggested edit to the recommended sentence
"The pricing step is the Fourier-diagonal case of the fact that l2 penalties on multiplicative factors induce l_q quasi-norm costs \citep{gunasekar2018implicit,hoff2017lasso,dai2021representation,fan2023euclidean}". Keep \citet{li2025nsum} (the existing key) in place of li2025fourier.
