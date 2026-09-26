# R29-kstar-table: independent verify (novelty_math lens), task 32

## Verdict: UPHELD (NOVEL_AS_FAR_AS_FOUND)
I found no mathematical source that computes the maximiser of the L_{1/s}-budget worst-offset coverage programme at p=13, 23, 59, 71 or 113, and none that reports its switch points in s. The table entries and switch values are the paper's own computations.

## Known mathematics the result sits on (cite or acknowledge)
1. **Minimisers are vertices, and switches happen between vertices.** This is standard L_q (0<q<1) sparse-minimisation theory.
   - Source: Jiang & Ye preprint (web.stanford.edu/~yyye/lpmin.pdf), fetched this session. Theorem 3: "The set of all basic feasible solutions of (1) is exactly the set of its all local minimizers", for min sum x_j^p subject to Ax=b, x>=0, with 0<p<1.
   - Published as Ge, Jiang, Ye, Math. Program. 129:285-299 (2011), DOI 10.1007/s10107-011-0470-2. The metadata was checked on Crossref and Semantic Scholar.
   - Appendix app:shape (i) proves the same fact for the inequality polyhedron P_p (concave objective, which attains its minimum at a vertex). The proof is self-contained and correct, but it is a known technique and should carry a one-clause citation.
   - The switch points at p=13 and p=23 are new numbers, but the mechanism behind them is not new: it is parametric concave minimisation over a fixed vertex set.
2. **Graph reading of the programme.** Sum_j y_j (1 - cos(2 pi j delta/p)) is half the Laplacian eigenvalue at character delta of the weighted circulant (Cayley) graph on Z_p that puts weight y_j on edges +-j. So:
   - Gamma_s(p) is half the largest algebraic connectivity of a weighted circulant graph with an L_{1/s} edge-weight budget.
   - M_K(p) is the same quantity over unweighted circulants with K generators.
   - The paper already cites Alon-Roichman (appendix line 196) for the log p lower bound, but it never states this reading.
   - A search for circulant algebraic-connectivity maximisation found one paper, Shahbaz-Belur-Ganesh arXiv:2110.01918. It covers unweighted graphs with fixed edge counts only, gives no connection-set tables and has no L_q budget. So the specific numbers are still unanticipated.
   - Recommendation: a one-sentence remark connecting the programme to circulant spectral gap and epsilon-biased sets would pre-empt a reviewer who knows this area. It is optional.

## Empirical-count citations (checker's "confirmed"): all re-verified
- **Nanda et al. 2301.05217, Sec 4.1.** "k in {14, 35, 41, 42, 52}" and "Of the six non-zero frequencies, five key frequencies" both confirmed from the text. arXiv title and authors are right. The ICLR 2023 venue could not be re-checked because the OpenReview page is behind a login wall.
- **Chughtai et al., PMLR 202:6243-6267.** Title, authors and pages confirmed on the PMLR page.
  - Table 5 (MLP), C113 seeds 1-4: 13, 11, 15 and 10 key frequencies. Confirmed.
  - Table 6 (transformer), C113: 3, 4, 4 and 3. Confirmed.
  - D59 transformer: 3-5 including sign.
- **Zhong et al. 2306.17844.** p=59 throughout. Model A's circles are 2pi/59·{17, 3, 44}; three circles give 91.4%. Confirmed.
- **McCracken et al. 2505.18266.** The true title is "Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks", by Gavin McCracken, Moisescu-Pareja, Letourneau, Precup and Love (v1, 23 May 2025).
  - The refs.bib entry (title "Emergence of sparse and dense circuits...", author "McCracken, Andrew") is WRONG.
  - Sec 4.3 has "handfuls of frequencies (3-7)" and Fig 4 covers moduli 59-66. Both confirmed.

## Heuristic flags: agree with the checker
- The caption says the enumeration is exhaustive only up to K<=8, 7, 5, so every cell is certified only among supports of size at most Kmax.
- (113, nu=3)=7 is above Kmax=5, and the equal-share argmax is 5 there. Mark it as a heuristic value or a band.
- Line 133, "Table gives the counts", reads as exact and should say "computed counts".

## Minor corrections to the checker
- The checker suggests citing Zhong for "3-7". Zhong reports 3 circles (6 with accompanying circles) at p=59, so cite Zhong for p=59 specifically, not for the range.
- Do not map the ReLU-MLP counts (10-15) onto any nu without caveat.

## Process note (the user asked why the workflow is slow)
- WebSearch returned "session has used its web search budget (200 of 200)" on the first call.
- The arXiv export API returned empty bodies (it looks rate-limited by parallel agents).
- As a result, verifiers fall back to WebFetch and curl, which is slow, and novelty searches are degraded.
