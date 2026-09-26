# C4-nanda-handful — cited claim check

Sentence (sections/intro.tex:6): "With ordinary weight decay they use a handful of frequencies \citep{nanda2023progress}".

Source fetched: https://arxiv.org/abs/2301.05217 and https://arxiv.org/html/2301.05217
Nanda, Chan, Lieberum, Smith, Steinhardt, "Progress measures for grokking via mechanistic interpretability", ICLR 2023 (bib entry matches the title and authors).

What the paper says:
- Sec 3: one-layer ReLU transformer, d=128, 4 heads, 512 MLP units, P=113, AdamW lr 1e-3, weight decay lambda=1.
- Sec 4.1: "the five key frequencies" k in {14,35,41,42,52}; "W_E is sparse in the Fourier basis -- it only has significant nonnegligible norm at 6 frequencies"; Sec 4.2: W_L approx rank 10 (= 5 key freqs x sin/cos).
- App C.2.1: other seeds use other small sets, e.g. {2,9,19,31} (4 freqs).
- Sec 5.3: "our networks do not grok ... without weight decay or some other form of regularization"; Sec 5.2: Gini coefficient of W_E/W_L rises during cleanup (sparser in Fourier basis).

Verdict: SUPPORTED. "A handful" (4-6 of 56) under AdamW weight decay is exactly what the paper reports.
Caveats: (i) "ordinary weight decay" = AdamW decoupled decay (lambda=1), not an l2 penalty on the loss. (ii) The network is a transformer, not the one-hidden-layer MLP the paper theorises about. The sentence says "trained networks", so it is still accurate.
Optional fix: "...use a handful of frequencies (five in a one-layer transformer at p=113 trained with AdamW) \citep{nanda2023progress}". Adding the bib `url`/arXiv field is optional.
