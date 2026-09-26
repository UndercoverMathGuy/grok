# VERIFY (citation lens) -- F3-explains-transition (task 67)

Verdict: checker UPHELD. PARTIALLY_SUPPORTED stands.

## Sources fetched this session
- arXiv abs 2505.18266 (citation_* meta): title "Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks"; authors Gavin McCracken, Gabriela Moisescu-Pareja, Vincent Letourneau, Doina Precup, Jonathan Love; 2025-05-23.
- arXiv HTML 2505.18266 (full text grepped with math alttext).
- neurips.cc/virtual/2025/poster/115005: NeurIPS 2025 poster, same title, author list with "Vincent Létourneau" (accent confirmed).
- arXiv abs 2311.07568: Morwani, Edelman, Oncescu, Zhao, Kakade; comment "Accepted as Spotlight at ICLR 2024".
- arXiv HTML 2311.07568: Theorem 7, Figure 4, Figure 7 captions read directly.

## McCracken: what is varied, what is seen (verified)
- Fig 4 caption: "adding either depth, or a trainable embedding matrix for inputs, causes the network to transition from learning floor(n/2) types of neurons to much fewer types". Moduli 59-66, one-hot MLPs.
- Table 5 (App E.4): one-hot 1-4 hidden layers and embedding-1-layer, 1024 neurons, L2 = 1e-5 in every row, lr 7.5e-4, 90% train. Regularization is held fixed.
- Sec 5.1: "While we are the first to point out how to cause a phase transition from O(n) to O(log(n)) learned frequencies (Fig. 4), we don't know why it occurs."
- G.9.1: count "can be shifted based on hyperparameters" (width, batch) "but ... it is still logarithmic". G.9.2 / Fig 41: count decays with depth "across almost all hyperparameter combinations". No weight-decay-driven dense-to-sparse transition is reported anywhere, and the norm exponent is never changed (always plain L2).
- Activation never varied (ReLU throughout).

## Morwani (verified)
- Thm 7: max L_{2,3} margin of 1-hidden-layer quadratic net on Z_p; part 2 "For every frequency ... at least one neuron ... uses this frequency."
- Fig 7: "(a) ReLU network trained with L2 regularization and (b) Quadratic network trained with L_{2,3} regularization ... final distribution across frequencies is close to uniform."
- Fig 4: per-neuron max-power (0.9 ReLU+L2, ~1 quad+L_{2,3}); not a count transition.
- Venue ICLR 2024 confirmed (arXiv comment). Existing refs.bib entry is correct.

## refs.bib
- `mccracken2025` is wrong: title "Emergence of sparse and dense circuits in modular arithmetic networks" and "McCracken, Andrew" do not match arXiv 2505.18266. (Could not run a web search for the bogus title: the session's WebSearch budget is exhausted, 200/200. The arXiv ID resolves to a different title and first author, which is sufficient.) Checker's replacement bibtex is correct, including L{\'e}tourneau.

## Additions beyond the checker
1. The paper's own appendix (appendix.tex, "Where the approach does not apply") says networks with a shared linear layer in front of the neurons are priced by a nuclear norm and "are outside this paper". A trainable embedding and a second hidden layer are exactly such shared layers. So McCracken's sparse-side architectures are explicitly outside the theory, not merely "not covered by a theorem". The checker's rewrite "(each raises the degree of a ReLU network)" is literally true for parameter homogeneity (3 instead of 2) but the paper's nu is not the operative quantity for these nets. The abstract should say "consistent with" / "offers an interpretation of", and the text should state that deep and embedding networks lie outside the theorems (appendix.tex already says so).
2. The "regularization" knob in the paper is the norm exponent b of L_{2,b}. Neither McCracken (L2 fixed at 1e-5) nor Morwani (exponent changes only together with activation, and both endpoints are dense) shows an exponent-driven count transition. Support that half with the paper's own experiments only.
3. sections/prior.tex:9 contains the same "as the degree of the network changes \citep{mccracken2025}" sentence; it is not \input in paper.tex, but fix it if revived.

## Process note (for the user's slowness question)
- WebSearch returned "this session has used its web search budget (200 of 200 WebSearch calls)". Remaining verifiers that rely on search will be blocked or degraded.
- The shared scratchpad holds ~280 files, with the Morwani paper downloaded at least 6 times (mor.html, morw.html, morwani.html, morwani_v2.html, morwani/m.html, morwani/v2.html, plus 2 PDFs and 2 tarballs) and McCracken PDF twice: verifiers re-fetch the same sources independently.
