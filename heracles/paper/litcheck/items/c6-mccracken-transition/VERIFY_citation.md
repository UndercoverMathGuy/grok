# VERIFY (citation lens): C6-mccracken-transition, task 80

Date: 2026-09-25. Sources fetched this session:
- https://arxiv.org/abs/2505.18266 (abs page), https://arxiv.org/html/2505.18266v1, and https://arxiv.org/pdf/2505.18266v1 (full text, grepped locally)
- OpenReview API search (api2.openreview.net/notes/search), forum zuHs6RHQwT: venue "NeurIPS 2025 poster", venueid NeurIPS.cc/2025/Conference, with the official _bibtex
- Not reachable: the OpenReview camera-ready PDF (bot challenge), Semantic Scholar (HTTP 429), dblp (timeout)

## Bib entry
The current refs.bib entry is wrong. It gives title "Emergence of sparse and dense circuits in modular arithmetic networks" and author "McCracken, Andrew and others". Neither exists. The real paper is:

Gavin McCracken, Gabriela Moisescu-Pareja, Vincent Létourneau, Doina Precup, Jonathan Love.
"Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks."
arXiv 2505.18266, v1 23 May 2025 (the only version).
**Accepted at NeurIPS 2025 (poster), OpenReview id zuHs6RHQwT.**

The checker said the paper had no venue, and its bibtex is `@article ... arXiv preprint` with "Letourneau" missing the accent. That is out of date. The canonical entry is the NeurIPS 2025 inproceedings (official OpenReview bibtex):

    @inproceedings{mccracken2025,
      title={Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks},
      author={McCracken, Gavin and Moisescu-Pareja, Gabriela and L{\'e}tourneau, Vincent and Precup, Doina and Love, Jonathan},
      booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
      year={2025},
      url={https://openreview.net/forum?id=zuHs6RHQwT}
    }

## Does the paper support the sentence?
In the paper, intro.tex l.6 and the abstract (paper.tex l.57) both say the count moves from dense to sparse "as the degree changes".

Supported, with these caveats:
- **Sec 4.4 / Fig 4** ("Average number of frequencies found in MLPs over moduli 59-66"). Verbatim: "We show that adding either depth, or a trainable embedding matrix for inputs, causes the network to transition from learning ⌊n/2⌋ types of neurons to much fewer types in Fig. 4. The presence of the trainable embeddings is why the models trained in [4, 8, 5] were observed to learn handfuls of frequencies (3-7) instead of ⌊n/2⌋." Confirmed.
- **The trigger is depth or a trainable embedding, not "degree".** The words "homogeneity" and "homogeneous" do not appear anywhere in the paper. "Degree" appears only as in "degree 1 sine functions".
- **The nets are probably not homogeneous.** The CRT construction explicitly uses "the y-intercept (neuron bias)" (Sec 4.2). The appendix points to PyTorch MLP classes and never says the nets are bias-free. So the checker's proposed clause "both of which raise its homogeneity degree" is the authors' own interpretation, not McCracken's. Drop it, or mark it as ours: "in the bias-free models we study, both correspond to raising the degree".
- **The checker's "they never vary weight decay" is overstated.** App. G.9 / Fig 41 shows layer-wise average-frequency heat maps over (learning rate x weight decay), and G.9.1 says "the average number of frequencies found can be shifted based on hyperparameters, but this is something we don't fully understand". What they never report is a dense-to-sparse transition driven by regularization. So the abstract must still not credit them with a regularization-driven transition. The checker's recommended change 3 is right.
- **Thm 4.7 / Cor 4.8** (arXiv v1 numbering): "Learning O(log n) distinct frequencies gives a logit margin Ω(log n); after softmax, incorrect classes receive at most n^{-Ω(1)} probability mass." This is a sufficiency result for random frequencies, not an optimality result. Confirmed.
- **Sec 5.1**: "...we are the first to point out how to cause a phase transition from O(n) to O(log(n)) learned frequencies..., we don't know why it occurs." Confirmed. This is a good hook for the paper.
- The numbering above is from arXiv v1. The NeurIPS camera-ready PDF could not be fetched (OpenReview bot challenge). Re-check Sec/Thm numbers against it before citing them by number.

## Verdict
PARTIALLY_SUPPORTED stands. The citation is real and supports a dense-to-sparse transition under depth or a trainable embedding. The bib entry must be replaced with the NeurIPS 2025 inproceedings version above, not the arXiv @article. "Degree" and "homogeneity" are our reinterpretation.

## Harness note (re: user's slowness question)
- In this subagent, WebSearch refused on the first call: "this session has used its web search budget (200 of 200 WebSearch calls)". The litcheck verifiers appear to share one exhausted 200-search budget.
- Semantic Scholar returned 429 and dblp timed out after 60s. Each timed-out fetch costs a full minute.
