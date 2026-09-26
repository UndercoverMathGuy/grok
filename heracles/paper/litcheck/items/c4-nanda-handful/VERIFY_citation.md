# VERIFY (citation lens) — C4-nanda-handful, task 78

Sentence (sections/intro.tex:6): "With ordinary weight decay they use a handful of frequencies \citep{nanda2023progress} ..."

## Fetched this session
- https://arxiv.org/abs/2301.05217 — title "Progress measures for grokking via mechanistic interpretability"; authors Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, Jacob Steinhardt. Matches refs.bib exactly.
- https://api.semanticscholar.org/graph/v1/paper/arXiv:2301.05217 — venue International Conference on Learning Representations, 2023. Confirms booktitle/year in bib. (OpenReview and dblp were blocked by bot challenges.)
- https://arxiv.org/html/2301.05217 — content checks:
  - Sec 3: one-layer ReLU transformer, P=113, d=128, 4 heads, 512 MLP units; full-batch AdamW, lr 1e-3, weight decay lambda=1.
  - Sec 4.1: "only 5 appear to be used significantly in later parts of the model (k in {14,35,41,42,52})"; "W_E is sparse in the Fourier basis -- it only has significant nonnegligible norm at 6 frequencies."
  - App C.2.1: other seeds "use a varying number of different key frequencies" (e.g. {2,9,19,31}).
  - Sec 5.3: "Our networks do not grok on the modular arithmetic task without weight decay or some other form of regularization."

## Verdict
SUPPORTED. Bib entry is correct (title, all 5 authors, ICLR 2023, arXiv id). "A handful" fits both 5 key frequencies and 6 embedding frequencies, and the other seeds also use small sets.
Caveats (no change needed): the setup is a transformer with AdamW's decoupled weight decay, not an explicit l2 penalty. "With ordinary weight decay" describes this setup; it does not say weight decay alone causes the sparsity. Nanda Sec 5.2 attributes circuit formation to weight decay, so the phrasing is safe. The checker's optional rewording (naming "five", "one-layer transformer", "p=113", "AdamW") is accurate if the authors want it.
No better or more canonical citation needed: this is the primary source.

## Workflow note
WebSearch returned "this session has used its web search budget (200 of 200 WebSearch calls)". Verify agents appear to share one session-wide search cap, which is now exhausted, so later verifiers can't search and fall back to WebFetch only. This probably contributes to the slow or degraded verify stage.
