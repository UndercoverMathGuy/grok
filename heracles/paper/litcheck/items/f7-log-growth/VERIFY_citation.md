# VERIFY (citation lens): F7-log-growth, task 71

Claim (intro, informal theorem bullet 3): "K* grows as Theta(log p) with modulus and decreases as nu/b grows".
Checker verdict: PARTIALLY_SUPPORTED. The log-p growth was observed empirically by McCracken et al.; the nu/b half has no precedent.

## Sources fetched this session
- arXiv abs 2505.18266. Title "Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks".
  - Authors: Gavin McCracken, Gabriela Moisescu-Pareja, Vincent Letourneau, Doina Precup, Jonathan Love.
  - Only v1 exists (23 May 2025); v2 returns 404. There is no journal-ref on arXiv.
- arXiv html 2505.18266v1. Every quote the checker relied on is verbatim in v1:
  - Fig. 3 caption: log fits of the frequency count vs n, with "very high R^2".
  - "We follow prior work in applying L2 regularization".
  - Thm 4.7: m frequencies drawn uniformly at random from {1..n/2}. Cor 4.8: "Learning O(log n) distinct frequencies gives a logit margin Omega(log n)".
  - Sec 4.4: "adding either depth, or a trainable embedding matrix ... transition from learning floor(n/2) types of neurons to much fewer".
  - Sec 5.1: "we don't know why it occurs".
  - The words "degree" and "homogeneity" do not appear.
  - Sec 4.3, verbatim: "Analogously to Morwani et al. [7], it gives that the maximum margin solution requires all frequencies to be learned". This refers to 1-layer one-hot ReLU nets.
- OpenReview API (api2.openreview.net/notes/search):
  - Note zuHs6RHQwT has the same title and the same 5 authors, venue "NeurIPS 2025 poster" (venueid NeurIPS.cc/2025/Conference). **The paper is published at NeurIPS 2025.**
  - Note Y71rktd4oT, "Interpreting neural networks depends on the level of abstraction: Revisiting modular addition", has the same authors and was "Submitted to ICLR 2025". It is the earlier version of the same work. Its abstract already proposes "a model where networks aim for a constant logit margin, predicting O(log(n)) frequencies". So this is not separate prior art, but the log-n margin argument dates to the ICLR 2025 submission (Oct 2024).
  - The camera-ready PDF on OpenReview sits behind a browser challenge and could not be fetched. Figure and theorem numbers and verbatim quotes are therefore verified only against arXiv v1.
- arXiv abs 2306.17844: Zhong, Liu, Tegmark, Andreas, "The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks". Comments: "Accepted by NeurIPS 2023". The checker's bib entry is correct.

## Existing refs.bib entry (wrong)
`mccracken2025` currently has title "Emergence of sparse and dense circuits in modular arithmetic networks" and author "McCracken, Andrew and others". Both fields are wrong. It must be replaced.

## Findings
1. The checker's verdict stands (PARTIALLY_SUPPORTED), and both citations exist and say what is claimed.
2. Better canonical form: cite the NeurIPS 2025 version (@inproceedings), not the arXiv @article. Two options:
   - Keep the arXiv eprint in the note or eprint field and cite "(arXiv v1)" for the verbatim quotes and figure numbers, since the camera-ready was not checked.
   - Drop the verbatim quote "don't know why it occurs" from the paper text.
3. Nuance to add: McCracken's own theory says the maximum-margin solution in 1-layer one-hot ReLU nets "requires all frequencies" (following Morwani). This is our dense endpoint at b = nu. The Theta(log p) optimum at b < nu should be framed as the plain-l2 counterpart, not as contradicting them.
4. The intro/prior line 6 misparaphrase ("as the degree of the network changes") is confirmed. McCracken's variable is depth or a trainable embedding. Their text never uses "degree".
5. Operational note: this session's WebSearch budget is exhausted ("200 of 200 WebSearch calls"), and the Semantic Scholar API returns 429. Verifiers running after this point cannot web-search. That likely explains the slow, degraded verify stage.
