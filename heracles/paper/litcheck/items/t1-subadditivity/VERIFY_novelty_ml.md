# T1-subadditivity: verify (novelty_ml lens), task 41

Claim (approach.tex, proof of thm:universal, l.64-67): r=b/nu<=1, t->t^r subadditive, so
sum_j A_j^r <= sum_{i,j}|beta_ij|^r; "splitting a feature over several neurons never makes it cheaper".

## Verdict: checker's KNOWN upheld
The inequality is the elementary p-norm inequality (0<r<=1). In ML, the same fact appears as p-subadditivity of
weight-decay representation costs. Nothing here is new, and the paper should not present it as new. The paper's only
contribution at this step is using it so that the bound needs no hypothesis on the neurons.

## Citations checked (fetched this session)
- Ongie & Parhi, arXiv:2606.14954 (v5, 5 Sep 2026, math.FA; v1 12 Jun 2026). I extracted the PDF text and read:
  - Def 3.5: p-seminorm, Q(f+g)^p <= Q(f)^p + Q(g)^p, 0<p<=1.
  - Prop 3.6: an absolutely p-homogeneous, subadditive K gives a p-seminorm K^{1/p}.
  - Prop 3.13: parameter q-subadditive plus homogeneous gives a subadditive representation cost.
  - Sec. 8, after (8.30): "the subadditivity follows by concatenating neurons".
  - Lemma 9.2 and Thm 9.3: the depth-L cost R_L is subadditive and absolutely 2/L-homogeneous, so it is a (2/L)-seminorm.

  All confirmed. Caveat: the "concatenating neurons" sentence is the shallow ReLU case with q=p=1, i.e. ordinary
  subadditivity (r=1). The r<1 analogue is Lemma 9.2 and Thm 9.3, so cite those. It is an unrefereed arXiv preprint.
  The bib entry is fine.
- Gunasekar, Lee, Soudry, Srebro, arXiv:1806.00468. The PDF footer reads "32nd Conference on Neural Information
  Processing Systems (NeurIPS 2018)", so the venue is confirmed. The abstract gives the l_{2/L} bridge penalty. The
  precise statement is in Sec. 5: Lemma 6 (diagonal) and Lemma 7 (conv, Fourier domain), R_P(beta) = min ||w||_2^2 =
  L ||beta_hat||_{2/L}^{2/L}. That is an explicit representation cost, not only a claim about GD.

  Caveat: these lemmas support the AM-GM pricing step (l.~49, "amplitude A costs A^{b/nu}"), not the
  subadditivity or splitting step. In the diagonal/conv setting each coordinate has one path, so there is no splitting
  across neurons. Cite at l.~49, or cite "Lemmas 6-7" explicitly if it stays at l.67.

## Adversarial search (ML)
- Kolb, Weber, Bischl, Ruegamer, "Deep Weight Factorization: Sparse Learning Through the Lens of Artificial
  Symmetries", arXiv:2502.02496, ICLR 2025 (confirmed from the PDF header). Lemma 1 and Thm 1: a D-factor Hadamard
  factorisation with an L2 penalty is equivalent to an L_{2/D} quasi-norm penalty, with the same global and local
  minima. This is an explicit-regularisation precedent for the AM-GM pricing, and a better fit for weight decay than
  the GD-implicit-bias framing. Kolb cites Dai, Karzand & Srebro (NeurIPS 2021) and Jacot (2023) as the
  representation-cost source for L_{2/D} on diagonal nets. I did NOT fetch Dai et al., so it is not recommended here.
  It is probably the canonical citation for "diagonal/product nets give l_{2/L}" and is worth fetching before the
  camera-ready.
- Jacobs & Montufar, arXiv:2605.19458 (mirror flow, sparse versus dense). I grepped the full text: no
  subadditivity or splitting argument. It is not a precedent for this step.
- Nothing found that states the exact step, a margin bound for L_{2,b} with b<nu that is uniform over mixed or
  interfering neurons through l_r subadditivity. The r=1 (b=nu) version is the additive single-neuron argument that
  the paper already attributes to Morwani et al. I did not re-fetch Morwani this session.
- Not verified: Kalton-Peck-Roberts (the checker had only retailer data). The inequality is elementary; drop the book
  unless someone checks the bib entry.

## Process note (relevant to the slow verify stage)
- WebSearch returned "session has used its web search budget (200 of 200)".
- export.arxiv.org and api.semanticscholar.org returned HTTP 429, and DuckDuckGo returned a CAPTCHA.
- Parallel verifiers share one search budget and hit rate limits, so they fall back to direct PDF fetches, which is
  slow. Raise CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION or throttle concurrency.
