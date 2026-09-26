# U8-nuclear-norm — CITATION_FOUND

Location: sections/appendix.tex l.285, "Where the approach does not apply": a shared linear layer "is priced by a nuclear norm that rewards carrying many frequencies in one neuron".
refs.bib has no Srebro/Gunasekar/Ongie entries (grep).

## Verified sources (all fetched this session)
1. Srebro, Rennie, Jaakkola. Maximum-Margin Matrix Factorization. NIPS 17 (2004).
   https://proceedings.neurips.cc/paper_files/paper/2004/hash/e0688d13958a19e087e123148555e4b4-Abstract.html
   PDF Lemma 1: ||X||_Sigma = min_{X=UV'} ||U||_Fro ||V||_Fro = min_{X=UV'} (1/2)(||U||_Fro^2 + ||V||_Fro^2).
   -> the core fact: L2 weight decay on a two-factor linear product = trace (nuclear) norm of the product.
2. Ongie, Willett. The Role of Linear Layers in Nonlinear Interpolating Networks. arXiv 2202.00856 (2022). Venue not confirmed (arXiv only in what I fetched).
   Intro: "adding linear layers to a ReLU network with weight decay regularization is akin to using a two-layer ReLU network with nuclear or Schatten norm regularization on the weight matrix."
   Lemma 3.2: R_L(f) = min (1/L)||a||_2^2 + ((L-1)/L)||W||_{S_q}^q, q = 2/(L-1); one linear layer (L=3) -> q=1 = nuclear norm (Sec 3, text before Lemma 3.8: "the Phi_3 loss involves minimizing over the nuclear norm of a matrix").
   -> this is the closest match: a shared linear layer in front of nonlinear neurons, with weight decay, induces a nuclear-norm price on the effective first-layer matrix.
3. Gunasekar, Woodworth, Bhojanapalli, Neyshabur, Srebro. Implicit Regularization in Matrix Factorization. NIPS 30 (2017).
   https://proceedings.neurips.cc/paper/2017/hash/58191d2a914c6dae66371c9dcdc91b41-Abstract.html
   Implicit (GD) bias to min nuclear norm (a conjecture with partial proof). Optional, and only tangential: the paper's claim is about explicit L2 pricing, not GD.

## Caveats
- "rewards carrying many frequencies in one neuron" is the paper's own reading: a nuclear norm charges singular values, not per-neuron supports, so rank/feature sharing across neurons is cheap. No source states it for Fourier features. Cite 1+2 for the pricing fact only.
- Semantic Scholar/dblp were rate-limited and the WebSearch budget was exhausted, so Ongie-Willett's published venue is unconfirmed. Cite it as arXiv.
