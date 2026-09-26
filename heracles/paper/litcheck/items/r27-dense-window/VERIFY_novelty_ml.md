# VERIFY (novelty_ml): R27 dense window, task 29

**Verdict: UPHELD.** NOVEL_AS_FAR_AS_FOUND stands, and McCracken et al. is confirmed as the right context citation.

## Internal checks
- **Numerics.** I reran dense_thresh.py for p = 7, 11, 13. It gives 1.5475, 1.3454, 1.2850, and each time the uniform vertex v = 2/p is the only full-support vertex. I did not rerun p >= 17 (the enumeration is slow), so for those three primes I rely on the checker.
- **(s-1) ln p at the six thresholds.** Values run from 1.065 at p=7 down to 0.501 at p=23. They decrease, which fits the O(1/log p) upper bound, but tightness is not shown.
- **Constant.** The appendix proof (appendix.tex, scaling proof, "Upper bound" item) gives w_j >= (m/2)^{1/(s-1)} with m >= L^{1-s}/2, where L = ceil(8 ln p). That yields w_j >= 4^{-1/(s-1)}/L. So the bound in cyclic.tex l.135, ln4/ln((p-1)/(2L)), is correct. But the theorem as stated (cyclic.tex l.116) only gives the share floor 4^{-1/(s-1)}/(9 ln p), and from that one gets ln4/ln((p-1)/(18 ln p)). The mismatch the checker flagged is real and harmless. Fix it by stating the L-floor in the theorem or by writing 18 ln p.
- **Vacuity.** The analytic bound only bites for p >= 73, as the checker says. At p <= 23 the evidence is the exact values.

## Novelty search (ML lens)
**Tooling was degraded.**
- WebSearch returned "200 of 200 WebSearch calls" used, so the budget is exhausted for the session.
- The arXiv export API returned empty bodies (throttled).
- The Semantic Scholar API returned HTTP 429.

**Searches that did run** (arxiv.org/search via WebFetch, abstract field, newest first):
- "modular addition frequencies sparse dense" returned 1 hit: 2606.17399, a multiplicative basis for modular multiplication. Not relevant.
- "modular addition margin": 2311.07568 (Morwani), 2402.09469 (Li, multi-input dense endpoint), 2511.23443 (sine activations, width-2), 2606.05863 (two clocks), 2608.07436 (Muon collapse). None gives a dense-to-sparse window in a norm exponent.
- "grokking number of frequencies": 2604.06256 (spectral edge dynamics) and 2306.13253. Not relevant.
- "modular addition Fourier features weight decay": 2609.26679 (Pracher et al., 22 Sep 2026) and 2602.16849 (He et al.). The Pracher abstract (fetched) has no dense/sparse count, exponent or log p window. He et al. is dense diversification with no threshold (see the uniqueness audit).
- "grokking sparse dense features margin", "modular addition logarithmic number of frequencies" and "max-margin homogeneous networks sparse dense norm exponent" all returned 0 results.

**Related papers fetched this session:**
- Jacobs & Montufar 2605.19458. The abstract covers sparse-to-dense feature learning set by the mirror map in homogeneous nets. It has no dimension-dependent threshold and does not treat modular addition.
- Kam, Bessafi & Cadet 2609.17926. The HTML Prop. 5 shows the uniform allocation is optimal under an l2 budget at full frequency content. There is no exponent threshold, and Sec. A.4 leaves the weight-decay allocation open. It is a contrast point, not prior art.

## Confirmed citation re-verified: McCracken et al. 2025 (arXiv 2505.18266)
- abs page:
  - Title: "Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks"
  - Authors: Gavin McCracken, Gabriela Moisescu-Pareja, Vincent Letourneau, Doina Precup, Jonathan Love
  - Date: v1, 23 May 2025. No journal reference.
- HTML: "adding either depth, or a trainable embedding matrix for inputs, causes the network to transition from learning floor(n/2) types of neurons to much fewer types". Also: "we don't know why it occurs. Theorem 4.7 only says this solution should have great margins, and it does." Cor 4.8 gives O(log n) frequencies giving margin Omega(log n), which is sufficiency only. They use L2 regularization but do not vary weight decay strength or degree, and they give no theory of when the dense regime occurs.
- The support is as the checker described.

**Bib nits:**
1. The paper cites the key `mccracken2025` (refs.bib l.25, with the wrong title "Emergence of sparse and dense circuits..." and author "Andrew"). Fix the fields under the existing key rather than introducing `mccracken2025universal`, or \cite calls elsewhere (abstract, intro) will break or duplicate.
2. arXiv spells the name "Letourneau" with no accent. `L{\'e}tourneau` is harmless but differs from the source.

## Out-of-scope flag
2609.26679 (Pracher et al., 22 Sep 2026) predicts "the grokking timescale is controlled by the product of learning rate and weight decay" and finds "inverse-product scaling" on modular addition. This bears on the shelved flight-time / matched-product line (memory: "matched-product result has no peer-reviewed precedent"). It is not relevant to this item.
