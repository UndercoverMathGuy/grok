# C6-mccracken-transition — cited-claim check

Where it appears: paper.tex l.57 (abstract): "transition in the number of features from dense to sparse---as seen empirically in \citet{mccracken2025}---... arises on changing either the homogeneity degree or regularization"; sections/intro.tex l.6: "the count moves from dense to sparse as the degree of the network changes \citep{mccracken2025}".

## Source (fetched 2026-09-25)
https://arxiv.org/abs/2505.18266 and https://arxiv.org/html/2505.18266
- Title: "Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks"
- Authors: Gavin McCracken, Gabriela Moisescu-Pareja, Vincent Letourneau, Doina Precup, Jonathan Love
- arXiv v1 23 May 2025, only version, no journal ref (cs.LG)

## What it actually says
- Sec 4.4 / Fig 4 ("Average number of frequencies found in MLPs over moduli 59-66"): "Previous work shows floor(n/2) types of neurons are found, each specializing to a sinusoid with one frequency with 1 hidden layer. We show that adding either depth, or a trainable embedding matrix for inputs, causes the network to transition from learning floor(n/2) types of neurons to much fewer types."
- Setup: ReLU MLPs, Adam, weight decay (not varied).
- Thm 4.7 / Cor 4.8: O(log n) distinct frequencies give logit margin Omega(log n). The margin is not normalised, and the result only shows that O(log n) is enough.
- Sec 5.1: "We don't know why it occurs."
- The paper never mentions homogeneity degree or polynomial degree, and it never varies weight decay or the regularizer.

## Verdict: PARTIALLY_SUPPORTED (and the bib entry is WRONG)
1. **Bib entry.** The title and author ("Emergence of sparse and dense circuits...", "McCracken, Andrew and others") do not exist at that ID. Replace them with the entry below.
2. **Dense-to-sparse transition.** Supported: McCracken observe dense (floor(n/2)) → sparse (few, ~log n).
3. **The trigger is "degree".** Only supported by our own reading of their result. They change *depth* or add a *trainable embedding*. For bias-free ReLU nets, each of these raises the homogeneity degree by one: a one-hot 1-hidden-layer net is 2-homogeneous, and adding an embedding or a layer makes it 3-homogeneous. So "degree" is our reinterpretation, and McCracken do not frame it that way. (If their nets have biases, the net is not exactly homogeneous. The fetched page did not state bias usage.)
4. **"or regularization" (abstract).** NOT supported by McCracken. They do not vary the regularizer. Make sure the abstract sentence does not read as if McCracken saw a regularization-driven transition.

## Fix
intro.tex l.6: "...and the count drops from dense ($\lfloor p/2\rfloor$ frequencies) to sparse ($O(\log p)$) when a ReLU network gains depth or a trainable embedding, both of which raise its homogeneity degree \citep{mccracken2025}."
paper.tex l.57: "...explains the dense-to-sparse transition observed empirically when depth or a trainable embedding is added \citep{mccracken2025}, by showing that such a transition arises when either the homogeneity degree or the regularization changes."
Also cite them at Thm (log scaling) lower bound as prior for "O(log p) frequencies suffice" (their Thm 4.7/Cor 4.8, unnormalised).

## Corrected bib
@article{mccracken2025,
  title={Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks},
  author={McCracken, Gavin and Moisescu-Pareja, Gabriela and Letourneau, Vincent and Precup, Doina and Love, Jonathan},
  journal={arXiv preprint arXiv:2505.18266},
  year={2025}
}
