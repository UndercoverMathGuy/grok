# VERIFY (novelty_math) — R34 conjecture-known (task 38), 2026-09-25

## Verdict
The checker's verdict stands: NOVEL_AS_FAR_AS_FOUND. Both confirmed citations are real and say what the checker reports.
The recommended edit (a) is wrong in one respect, and it adds a duplicate bib key. Details below.

## Search constraints (these matter for the whole verify stage)
- WebSearch refused on the first call: "this session has used its web search budget (200 of 200 WebSearch calls)".
  The budget is shared across the session, so every later verify task will run with no WebSearch.
- The arXiv API (export.arxiv.org) returned HTTP 429. The Semantic Scholar API returned 429. DuckDuckGo returned a CAPTCHA. Bing returned irrelevant results.
- What worked: arxiv.org/search (web UI) through WebFetch, plus direct PDF downloads.

## Citations checked (fetched this session)
- Morwani et al. (arXiv 2311.07568v2; PDF read with pdftotext). Theorem 7: one-hidden-layer quadratic network (u(a)+v(b))^2 w, with m >= 4(p-1) neurons.
  Max L_{2,3} margin = sqrt(2/27) / (p^{1/2}(p-1)). Every max-margin network uses single-frequency neurons, and every frequency in 1..(p-1)/2 is used.
  CONFIRMED. The bib entry `morwani2024feature` already exists.
- Li, Liang, Shi, Song, Zhou (PMLR 258:523-531, AISTATS 2025; arXiv 2402.09469 PDF read). Theorem 4.1 (informal version of Thm G.2): power network with k inputs, m >= 2^{2k-1}(p-1)/2 neurons.
  Max L_{2,k+1} margin = 2(k!) / ((2k+2)^{(k+1)/2} (p-1) p^{(k-1)/2}). Each neuron uses one frequency, and all frequencies are used.
  CONFIRMED. The PMLR page confirms volume 258 and pages 523-531.
  NOTE: refs.bib ALREADY has this paper as `li2025nsum`. The checker's new key `li2025fourier` would duplicate it.
  Use `li2025nsum` instead, and add the pages/series fields to it if wanted.

## Problem with the checker's recommended edit (a)
- Morwani Thm 7 and Li Thm 4.1 are about the POWER (sum-then-power) network. The conjecture in app:conjecture is about PRODUCT networks.
  So the sentence "At the endpoint b=nu the global maximum over all networks is known and dense (Morwani Thm 7; Li Thm 4.1)" is inaccurate for this conjecture.
- For product networks, b=nu is settled by the paper's own Corollary cor:dense: the universal bound holds for every network, and the flat nu-block attains it (appendix l.32).
  That proof uses Morwani's averaged-margin certificate (their Section 3).
- Suggested wording: "At b=nu the conjecture holds by Corollary cor:dense, whose certificate is the averaged-margin argument of Morwani et al. (Section 3). The analogous global maxima for the power network are Morwani et al. Thm 7 (n=2) and Li et al. Thm 4.1 (general n)."
- The window nu - 4/(3(p-1)) <= b <= nu is then an O(1/p) extension below that endpoint. This is a fair framing, provided a proof is supplied.

## Adversarial novelty search (mathematics lens, plus ML nearest neighbours)
- arxiv.org/search, abstracts matching "modular addition margin" (39 hits, newest first, through Sep 2026). Only Morwani, Li et al., and 2511.23443 are on topic.
  2511.23443 is about sine-network generalization bounds. It says nothing on the global L2 margin or interference.
- Searches for "modular addition weight decay number of frequencies" and "max-margin grokking global optimum Fourier" returned 0 hits.
- Chowla cosine problem (6 hits, including 2609.05338 and 2509.05260). These minimise the minimum of a cosine sum.
  None treats a max-min over Z_p under an l_q or quasi-norm budget. None bears on interference.
- "cosine sum minimum nonnegative coefficients" returned 0 hits.
- Nearest ML neighbour, not cited in the paper: Tian, "Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets", arXiv 2410.01779 (NeurIPS 2025 camera ready; full PDF read).
  It builds global optimizers of the MSE (L2) loss for 2-layer quadratic networks on Abelian groups by composing per-frequency order-4/order-6 partial solutions.
  It shows that weight decay favours low-order solutions. Every frequency is used, and it contrasts its approach with Morwani's L_{2,3} max-margin framework.
  It does NOT address the L_{2,b} margin, and does not ask whether cross-frequency interference can beat the interference-free value. So it does not settle the conjecture.
  It is an optional related-work citation ("global optima composed per frequency").
- Nothing found proves or refutes the conjecture for 2 <= b < nu. Nothing found exhibits an interfering local maximum, or the Z12/D3/S4 counterexamples.
- Confidence is moderate, not high, because WebSearch was unavailable (see above).

## Internal issues (agree with the checker)
- "certified aiming depth" is undefined (TODO at appendix.tex l.293-294).
- The "Known" items have no proofs or pointers. Where only numerical evidence exists, mark it "verified numerically".
