# VERIFY (novelty_math): R27-dense-window

Claim (cyclic.tex, final paragraph): the optimum can be dense only if
s-1 <= ln4/ln((p-1)/(2 ceil(8 ln p))) = O(1/log p). Exact dense thresholds are
s = 1.548, 1.345, 1.285, 1.217, 1.194, 1.160 at p = 7, 11, 13, 17, 19, 23.

## Search capacity (important)
- WebSearch was unavailable: "this session has used its web search budget (200 of 200 WebSearch calls)".
- The arXiv export API returned HTTP 429 Too Many Requests on two queries.
- So the adversarial literature search was limited to direct fetches of known pages.
- Both limits are shared across the parallel verifier agents. They are the likely cause of the slow verify stage.

## Citation check
- McCracken, Moisescu-Pareja, Letourneau, Precup, Love, "Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks", arXiv:2505.18266, v1 23 May 2025.
  - Fetched from https://arxiv.org/abs/2505.18266 and https://arxiv.org/html/2505.18266.
- Verbatim quotes confirmed:
  - "adding either depth, or a trainable embedding matrix for inputs, causes the network to transition from learning floor(n/2) types of neurons to much fewer types"
  - "we don't know why it occurs"
  - Cor 4.8: "Learning O(log n) distinct frequencies gives a logit margin Omega(log n)"
- The paper gives no threshold in a norm exponent or regularizer.
- The recommended bibtex is correct.
- The existing refs.bib entry `mccracken2025` has the wrong title ("Emergence of sparse and dense circuits...") and the wrong first author ("McCracken, Andrew"). It must be replaced.

## Internal numerics: independent check
The analytic statement is correct.
- The share floor in the proof is w_j >= (m/2)^{1/(s-1)} with m >= (1/2) L^{1-s}, where L = ceil(8 ln p).
- Density needs ((p-1)/2) * 4^{-1/(s-1)}/L <= 1, and that gives the displayed bound.
- The checker is right that the theorem as stated has 9 ln p where the text uses ceil(8 ln p). This is a cosmetic mismatch.

NEW: the O(1/log p) envelope is very loose. The real window looks like Theta(1/p).
- Script: adj_vertex_thresh.py (this folder).
- The only full-support vertex of P_p is the uniform u = 2/p, since M is invertible.
- For r = 1/s < 1, phi = sum v^r is strictly concave, so every minimiser is a vertex.
- Consider any vertex adjacent to u: drop one frequency j0 and relax one constraint delta0. If phi at that vertex is below phi(u), the optimum is not dense.
- So the smallest s at which an adjacent vertex beats u is a constructive upper bound on the dense threshold at each p (up to floating point).

| p | exact threshold (paper) | adjacent-vertex upper bound | (s-1)*p exact / adj | (s-1)*ln p adj |
|---|---|---|---|---|
| 7 | 1.548 | 1.5475 | 3.83 / 3.83 | 1.07 |
| 11 | 1.345 | 1.3647 | 3.80 / 4.01 | 0.87 |
| 13 | 1.285 | 1.3103 | 3.71 / 4.03 | 0.80 |
| 17 | 1.217 | 1.2349 | 3.68 / 3.99 | 0.67 |
| 19 | 1.194 | 1.2083 | 3.68 / 3.96 | 0.61 |
| 23 | 1.160 | 1.1688 | 3.67 / 3.88 | 0.53 |
| 31 | - | 1.1212 | - / 3.76 | 0.42 |
| 59 | - | 1.0600 | - / 3.54 | 0.25 |
| 71 | - | 1.0492 | - / 3.49 | 0.21 |
| 113 | - | 1.0302 | - / 3.41 | 0.14 |

- (s-1)*p is nearly constant: about 3.7 for the exact values and 3.4 to 4.0 for the adjacent bound.
- (s-1)*ln p falls steadily.
- So the dense window empirically scales like about 3.5/p, not 1/log p.
- At p = 113 the analytic bound allows s-1 <= 3.58. The one-pivot certificate already shows the optimum is sparse for s > 1.031.

Recommendation:
- Keep O(1/log p) as the proven statement.
- Add that the exact thresholds follow (s-1) ~ 3.7/p, and that a one-pivot certificate gives s <= 1.031 at p = 113.
- Or say that "O(1/log p) is a loose envelope".
- Do not say the window "shrinks like 1/log p".
- Whether Theta(1/p) can be proven is open. A naive argument (drop one frequency and rescale) only gives s <= ~2, so the adjacent vertex's reweighting is essential.

## Novelty (math lens)
Reformulation used for the search:
- sum_j v_j (1 - cos(2 pi j delta/p)) is the delta-th Laplacian eigenvalue of the weighted Cayley graph Cay(Z_p, {+-j}) with generator weights v_j.
- So the programme asks for the cheapest generator weights, under the concave cost sum v^{1/s}, that give spectral gap >= 1.
- The uniform (complete-graph) weights are the l1 optimum.
- The dense window asks when the complete graph stays optimal under a concave cost.
- The lower bound in the paper is already credited to Alon-Roichman.

Result:
- I could not run searches in this direction (budget exhausted, arXiv 429).
- I found no source for a p-dependent window of full-support optimality for l_q (q -> 1) costs.
- The checker's NOVEL_AS_FAR_AS_FOUND verdict stands, but at low confidence given the limited search.
- If search capacity returns, try: "fastest mixing / absolute algebraic connectivity with concave cost", "sparse weighted Cayley graph spectral gap l_q", "l_q minimisation q -> 1 full support threshold".
