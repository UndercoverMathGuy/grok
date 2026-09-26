# VERIFY (novelty_ml): R8-dense-corollary, task 5

Verdict: **KNOWN (upheld)**. The checker's verdict and both confirmed citations hold up under an independent check.

## Citations re-checked (primary sources fetched this session)
- **morwani2024feature**: arXiv 2311.07568v2 PDF, read with pdftotext. Theorem 7 (m >= 4(p-1), quadratic neurons, L_{2,3}): gamma* = sqrt(2/27) * 1/(p^{1/2}(p-1)). Part 2: every frequency in {1..(p-1)/2} is used.
  App. F.3 ("Proof that all frequencies are used") applies Lemma 6 to write f = F(a,b) + lambda 1[a+b=c]. This gives f-hat(j,j,-j) = lambda p^2 at every nonzero j, so every max-margin net has the same diagonal amplitude at every frequency. The flat maximizer is therefore already in Morwani (for n=2).
  iclr.cc/virtual/2024/poster/18073 confirms the title, the five authors and "Spotlight Poster". The bib entry is fine.
- **li2025nsum**: arXiv 2402.09469v4 PDF. Theorem 4.1 (informal version of Thm G.2): if m >= 2^{2k-1}(p-1)/2, then gamma* = 2(k!)/((2k+2)^{(k+1)/2}(p-1)p^{(k-1)/2}), each neuron uses one frequency, and every frequency is used.
  The PMLR page v258/li25b confirms: Proceedings of the 28th AISTATS, PMLR vol 258, pp. 523--531, 2025. The bib should add the volume, pages and series.
- **Constant check (mine)**: n! * B_p * p/(p-1), with B_p = 2(2p nu)^{-nu/2} and nu = n+1, divided by Li's gamma*, equals 1 to machine precision for n=2..5 and p in {7, 113, 1009}. At n=2 it also equals Morwani's sqrt(2/27)/(p^{1/2}(p-1)). The paper's appendix arithmetic, 4(6p)^{-3/2} p/(p-1) = (2/(3 sqrt6)) / (p^{1/2}(p-1)), is correct.

## Adversarial ML search
- arXiv abstract search "modular addition margin", newest first, through Sept 2026. The only max-margin characterisations are Morwani 2311.07568 and Li 2402.09469. Huang-Li 2511.23443 (sine networks) gives margin-based generalization only, with no max-margin constant and no frequency count.
  The 2026 hits (2608.07436 Muon post-grokking collapse, 2606.05863 two training clocks, 2606.12966 circuit synchronisation) do not characterise the optimum.
- arXiv search "modular addition Fourier frequencies weight decay" returned only 2602.16849 (He et al., already audited) and 2606.12966. Neither characterises the optimum.
- Tian, arXiv 2410.01779, "Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets" (NeurIPS 2025). It studies quadratic activation with L2 loss and weight decay on Abelian groups and builds global optimizers algebraically. It is not a margin result, and its abstract does not state a dense or equal-amplitude endpoint. It does not overlap with R8. It could be cited in related work for the sparse s>1 side, but not for this item.

## What is new in R8
Only two things: the explicit uniqueness of the flat maximizer inside the programme Gamma_1, which is an elementary averaging and Fourier-inversion argument, and the framing of the dense endpoint as the s=1 case of a family that is sparse for s>1. The corollary must be presented as recovering Morwani Thm 7 (n=2) and Li Thm 4.1 (general n), not as a new result. The checker's recommended text change is appropriate.

## Tooling note (relevant to workflow slowness)
- The WebSearch tool refused every query: "this session has used its web search budget (200 of 200 WebSearch calls)". The budget is shared across the session, so later verifier agents get no search at all.
- export.arxiv.org/api returned HTTP 429, and api.semanticscholar.org returned 429. Parallel agents are being rate-limited.
- WebFetch of arxiv.org/search/ HTML still worked and was used instead.
