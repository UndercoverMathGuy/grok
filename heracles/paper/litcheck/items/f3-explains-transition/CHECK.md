# F3-explains-transition: CHECK

Claim (paper.tex:57, the abstract): "our theory explains the transition in the number of features from dense to sparse, as seen empirically in \citet{mccracken2025}, by showing that this transition arises on changing either the homogeneity degree or regularization of the network."
Also intro.tex:6: "the count moves from dense to sparse as the degree of the network changes \citep{mccracken2025}".

Verdict: PARTIALLY_SUPPORTED. The transition is real, but McCracken did not vary degree or regularization.

## What McCracken et al. actually do (fetched arXiv HTML v1 2505.18266, NeurIPS 2025 poster page)
- Paper: "Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks". Authors: Gavin McCracken, Gabriela Moisescu-Pareja, Vincent Letourneau, Doina Precup, Jonathan Love. Published at NeurIPS 2025 (poster 115005; OpenReview zuHs6RHQwT).
- refs.bib is WRONG. It has the title "Emergence of sparse and dense circuits in modular arithmetic networks" and the author "Andrew McCracken". Neither exists.
- Sec 4.4 / Fig 4 ("Average number of frequencies found in MLPs over moduli 59-66"): "Previous work shows floor(n/2) types of neurons are found ... with 1 hidden layer [Gromov, Morwani]. We show that adding either depth, or a trainable embedding matrix for inputs, causes the network to transition from learning floor(n/2) types of neurons to much fewer types in Fig. 4."
- App E.4 / Table 5 gives the setup:
  - one-hot ReLU MLPs with 1, 2, 3 and 4 hidden layers, plus an embedding MLP with 1 hidden layer;
  - 1024 neurons, lr 7.5e-4, 90% train split;
  - L2 regularization = 1e-5 in EVERY row, so regularization is held fixed.
- Only ReLU is used. The activation / polynomial degree is never varied, and quadratic nets are cited only as prior theory (Gromov).
- App G.9.1: "More overparameterized means less frequencies". G.9.2: "Adding depth also means less frequencies are learned".
- Sec 5.1: "While we are the first to point out how to cause a phase transition from O(n) to O(log(n)) learned frequencies (Fig. 4), we don't know why it occurs."

## Assessment
- What McCracken varies is architecture (depth, trainable embedding). It does not vary "homogeneity degree or regularization". Still, both of their knobs change the homogeneity degree nu of a ReLU net:
  - a one-hot net with 1 hidden layer has nu = 2 = b, so weight decay sits at the dense endpoint. Our appendix.tex:283 says exactly this.
  - adding a trainable embedding or another layer gives nu >= 3 > b = 2, which is the sparse regime.
  - So the "degree" half can be defended as a reinterpretation. It needs to be stated as ours, and the paper must name the knob McCracken actually turned.
- The "regularization" half has NO support in McCracken, where L2 is fixed at 1e-5. Morwani et al. (ICLR 2024) also do not show it:
  - their Fig 4 compares quadratic+L_{2,3} with ReLU+L2 on per-neuron sparsity (max power ~0.9 vs 1);
  - their Fig 7 finds near-uniform coverage of all frequencies for both.
  - So the regularization-induced transition is evidenced only by our own experiments.
- "Explains" is too strong for ReLU. Our theory proves results for product/power neurons. For ReLU, the appendix gives only the b=nu upper bound and states "density of the maximum margin ReLU network is a statement about the bound's maximizer, not a theorem about the network". The deep and embedding ReLU nets that McCracken finds sparse are not covered by any theorem of ours.
- What is consistent: the O(log n) sparse count McCracken observes (their Fig 3, and Thm 4.7/Cor 4.8 on sufficiency) matches our K* = Theta(log p). They leave the "why" open, so offering an explanation is a legitimate contribution if it is phrased as consistent-with or suggests.

## Suggested rewrite (abstract)
"...Further, our theory accounts for the transition from dense to sparse feature counts: raising the homogeneity degree above the regularization exponent, or lowering the exponent below the degree, moves the optimum from all frequencies to Theta(log p) of them. This is consistent with McCracken et al. (2025), who observe that adding depth or a trainable embedding (each raises the degree of a ReLU network) moves one-hidden-layer networks from floor(p/2) to O(log p) frequencies."
- intro.tex:6: replace "as the degree of the network changes" with "when depth or a trainable embedding is added, which raises the network's degree".
- Fix the bib entry.
