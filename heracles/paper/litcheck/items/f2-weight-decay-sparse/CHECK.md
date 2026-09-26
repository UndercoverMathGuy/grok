# F2-weight-decay-sparse: "networks trained with ordinary weight decay consistently show a sparse selection"

Verdict: PARTIALLY_SUPPORTED. "Consistently" overclaims, and the literature contains a documented dense counterexample.

## Supporting (fetched 2026-09-25)
- Nanda et al., ICLR 2023 (arXiv 2301.05217). Sec 3: "full batch gradient descent using the AdamW optimizer with learning rate 0.001 and weight decay parameter lambda=1". Sec 4.1: "only 5 appear to be used significantly ... k in {14,35,41,42,52}"; "W_E is sparse in the Fourier basis -- it only has significant nonnegligible norm at 6 frequencies". Sec 4.4: ablating all components except the key frequencies improves performance.
- Chughtai, Chan, Nanda, ICML 2023, PMLR 202:6243-6267. App C: "weight decay with lambda=1, and the AdamW optimizer". Sec 5.1: logits are explained by "a very sparse set of directions ... the 'standard' and 'sign' representations". Sec 6: "different numbers of representations are learned" (the most common case is two). The precise circuits are "arbitrary" across seeds.
- Zhong, Liu, Tegmark, Andreas, NeurIPS 2023 (arXiv 2306.17844). App F.1: "AdamW ... weight decay factor 2". Models use a few circles; Sec 3.3: "first six principal components (three circles)" give 91.4% accuracy. Sec 3.4: pizzas come with "accompanying pizzas". This is only a few frequencies, and the paper gives no count law.
- Stander, Yu, Fan, Biderman, ICML 2024, PMLR 235:46441-46467. S5/S6 MLPs decompose the task using subgroups and cosets, so they use a small number of irreps and cosets.

## Contradicting / qualifying
- McCracken, Moisescu-Pareja, Letourneau, Precup, Love, arXiv 2505.18266 (2025). Sec 4.4, Fig 4: "Previous work shows floor(n/2) types of neurons are found, each specializing to a sinusoid with one frequency with 1 hidden layer. ... adding either depth, or a trainable embedding matrix for inputs, causes the network to transition from learning floor(n/2) types of neurons to much fewer types." And: "The presence of the trainable embeddings is why the models trained in [prior work] were observed to learn handfuls of frequencies (3-7) instead of floor(n/2)". App E.4 Table 5: the one-hot 1-hidden-layer MLP was trained with **L2 regularization 1e-5**, 1024 neurons, lr 7.5e-4, 90% train. So these are weight-decayed networks that are DENSE (every frequency is used).
- Gromov, arXiv 2301.02679 (2023). The 2-layer quadratic MLP uses the full set of frequencies ("The full set of independent frequencies is obtained by varying k from 0 to (p-1)/2"). It was trained without explicit regularisation, but AdamW and GD+wd were also tried.
- Our own records (memory: MAIS registry pivot, nsum ladder) show that nu=2 nets and the ReLU 2-operand MLP are dense. The paper's theory also predicts a dense window at low degree. So the paper's own results contradict "consistently".

## Secondary issue: how the intro characterises McCracken
The intro says the count "moves from dense to sparse as the degree of the network changes (mccracken2025)". McCracken attributes the transition to **depth or a trainable embedding**, and does not mention "degree". Both changes raise the homogeneity degree, so the reading is defensible, but it has to be said explicitly. McCracken says "we don't know why it occurs" (Sec 5.1).

## Recommended text
Abstract: "Empirically, however, networks trained with ordinary weight decay often use only a handful of features (transformers, embedding MLPs, group-composition MLPs), while shallow one-hot MLPs use all of them."
Intro: "With ordinary weight decay, transformers and embedding MLPs use a handful of frequencies (Nanda; Zhong) or a few irreps (Chughtai; Stander). One-hot one-hidden-layer MLPs instead use all floor(p/2) frequencies (Gromov; McCracken). Adding depth or a trainable embedding, both of which raise the homogeneity degree, moves the count from dense to sparse (McCracken, Fig 4)."
