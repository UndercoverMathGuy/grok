# VERIFY (novelty_ml): T2-amgm-quasinorm, task 43

Claim (approach.tex:45 and the product-neuron budget in app:models): AM-GM over nu weight factors makes amplitude A cost A^{b/nu}, so weight decay on degree-nu products acts as an l_{b/nu} quasi-norm on amplitudes.

Checker verdict: KNOWN. **Upheld.**

## Checker's confirmed citations (all re-fetched this session)
- **Tibshirani 2021 working note** (stat.cmu.edu PDF, dated April 15 2021). Confirmed.
  - Sec 2.1: the min over ab=c is attained at a=b, "this is just saying that the arithmetic mean-geometric mean (AM-GM) inequality ... is tight".
  - k-layer case: min_{prod a_l=c} sum a_l^2 = k|c|^{2/k}, which is an l_p penalty with p=2/k and is nonconvex for k>2 (eq. 9).
- **Hoff 2017.** arXiv 1611.00040 abstract: "a subclass of L_q penalties with q<=1 can be expressed as sums of L_2 penalties". Crossref confirms CSDA vol. 115, pp. 186-198, DOI 10.1016/j.csda.2017.06.007. Bib correct.
- **Gunasekar, Lee, Soudry, Srebro, NeurIPS 2018** (arXiv 1806.00468). Thm 2a (l_{2/L} bridge penalty on Fourier coefficients, first-order stationary point) confirmed.
  - **Refinement:** the arXiv version also has explicit representation-cost lemmas.
    - Lemma 6 (diagonal): R(beta) = min ||w||_2^2 = L ||beta||_{2/L}^{2/L}.
    - Lemma 7 (conv): the same with beta-hat in Fourier.
  - These are the exact explicit-weight-decay version of our step, on Fourier amplitudes. Cite Lemma 7 (not only Thm 2a). Lemma numbers are from the arXiv PDF; the proceedings version may number them differently.
- **Dai, Karzand, Srebro, NeurIPS 2021.** The proceedings abstract confirms l2 on weights inducing predictor-space costs, "including l_p norms". Fine as a framework citation.
- **Kolb, Weber, Bischl, Rugamer, ICLR 2025** (arXiv 2502.02496). Thm 1 confirmed: L2 on D Hadamard factors vs L_{2/D} on w, "have the same global and local minima". The related-work section traces the idea to Grandvalet 1998 and Neyshabur et al. 2015.

## Prior art the checker missed or left unconfirmed
- **Neyshabur, Tomioka, Srebro, ICLR 2015 workshop** (arXiv 1412.6614), Thm 1: weight decay on a two-layer ReLU net equals l1 on the top layer with unit-norm hidden units. The proof is AM-GM plus rescaling invariance of homogeneous units.
  - This is the nonlinear, per-neuron version (b=nu=2, s=1).
  - It is closer to our product/power neurons than diagonal linear nets are, and it is the canonical ML precedent.
- **Grandvalet 1998**, ICANN 98, pp. 201-206, Springer, DOI 10.1007/978-1-4471-1599-1_27. Now confirmed via Crossref and the Kolb bib. It is the earliest source for "quadratic penalty on factors = lasso".
- **L_{2,b} generalisation** (exponent b/nu for general b): I did not find an explicit prior statement.
  - It is a one-line extension (AM-GM on the b-th powers). It should not be claimed as a contribution.
  - This check was incomplete: the session's WebSearch budget was exhausted (200/200) and the arXiv API returned HTTP 429.

## Recommended sentence (amends the checker's)
"...so amplitude A costs a multiple of A^{b/nu}. This is the standard fact that l2 weight decay on a product of k factors is an l_{2/k} quasi-norm on the product \citep{grandvalet1998least,neyshabur2015search,hoff2017lasso,tibshirani2021equivalences,kolb2025deep}; \citet[Lemma 7]{gunasekar2018implicit} obtain exactly this cost on the Fourier coefficients of deep linear convolutional networks."
