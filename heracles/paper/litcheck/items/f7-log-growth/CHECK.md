# F7-log-growth: intro informal-theorem bullet 3
Claim checked: "K* ... decreases as nu/b grows and increases as Theta(log p) with the modulus."

**Verdict: PARTIALLY_SUPPORTED.**
- The log-p growth is supported empirically by McCracken et al. 2025. Their paper also contains a sufficiency theorem.
- The two-sided Theta(log p) statement *at the optimum* is new as far as I found.
- I found no paper that addresses the claim that K* decreases in nu/b. McCracken's depth observation is qualitatively consistent with it, but it does not support it directly.

## Sources (all fetched 2026-09-25)
### 1. McCracken, Moisescu-Pareja, Letourneau, Precup, Love, arXiv 2505.18266 (v1, 23 May 2025), https://arxiv.org/html/2505.18266v1
- **Fig. 3 caption, verbatim:** "The number of frequencies found in clocks, pizzas, and MLPs as the modulus n increases. We plot the data on logarithmic and linear axes, showing logarithmic fits have very high R^2 scores."
  - This is a direct empirical precedent for "count grows like log p".
  - It covers trained nets with L2 regularisation: "We follow prior work in applying L2 regularization".
- **Sec 4.3:** Thm 4.7 and Cor 4.8, "Learning O(log n) distinct frequencies gives a logit margin Omega(log n)".
  - The margin is unnormalised.
  - The result is sufficiency only.
  - Same section: "it gives that the maximum margin solution requires all frequencies to be learned" (citing Morwani).
- **Sec 4.4 / Fig. 4:** "Adding either depth, or a trainable embedding matrix for inputs, causes the network to transition from learning floor(n/2) types of neurons to much fewer." One-hot 1-layer MLPs learn about floor(n/2) frequencies.
- **Sec 5.1:** "While we are the first to point out how to cause a phase transition from O(n) to O(log(n)) learned frequencies (Fig. 4), we don't know why it occurs."
- I found no statement relating the count to activation degree or homogeneity.

### 2. Zhong, Liu, Tegmark, Andreas, "The Clock and the Pizza", NeurIPS 2023, https://arxiv.org/abs/2306.17844
- This is the source of the clock and pizza models that McCracken's Fig. 3 counts.
- The abstract has no count-versus-p statement. Background citation only.

### 3. Morwani et al., ICLR 2024, https://arxiv.org/html/2311.07568
- The b = nu endpoint is dense (all frequencies).
- Fig. 1a / 4b: ReLU with L2 regularisation gives "neurons are nearly 1-sparse in frequency space". This is sparsity *per neuron*, not the frequency count, so it does not support the nu/b claim.

### 4. Search results
- A web search for number-of-frequencies-versus-modulus found only McCracken with an explicit log-p fit.
- Nanda 2023 and He et al. 2602.16849 do not give a count law.
- Ding et al. 2405.17420 finds more surviving frequencies as the embedding dimension grows. That is a width axis, not a p axis.
- A Semantic Scholar search returned HTTP 429 and was not retried.

## Assessment
- **Log p growth.**
  - Empirical side: this is McCracken's observation, so cite it at this bullet.
  - Our contribution is the explanation of the observation they explicitly could not explain, namely the two-sided Theta(log p) optimum.
  - Frame the bullet as explaining that observation, not as discovering it.
- **Decrease in nu/b.** No precedent was found in either direction. Two points are consistent with it:
  - McCracken's one-hot 1-layer MLP (ReLU, weight decay, so nu = b = 2 and s = 1) is dense at floor(n/2). This matches our s = 1 dense endpoint.
  - Adding depth raises nu at a fixed l2 penalty, so nu/b grows, and McCracken sees fewer frequencies.

  Both are only qualitative. McCracken attributes the transition to depth or trainable embeddings, not to degree.
- **Intro line 6 flag.** The sentence "the count moves from dense to sparse as the degree of the network changes \citep{mccracken2025}" paraphrases McCracken inaccurately. Their variable is depth or a trainable embedding, not "degree". Reword it (see recommended change).
- **Bib entry.** The `mccracken2025` entry has the wrong title and authors. See sweep §0.1.
