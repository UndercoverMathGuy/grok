# VERIFY (novelty_ml) -- R26-few-features (cyclic.tex cor:few), task 27

Claim: any product/power network with K = o(log p) non-zero frequency amplitudes attains a vanishing fraction of gamma*_b(p), interference allowed.
Checker verdict: PARTIALLY_KNOWN. The equal-share case is classical (expanders / Dirichlet). The weighted l_{1/s} version and its use for network margins are new.

## ML-lens search (assume it is already in the ML literature)

The shared WebSearch budget was exhausted (200/200), and the arXiv export API and the Semantic Scholar API returned HTTP 429. I searched with WebFetch on arxiv.org/search plus direct paper fetches.

- **McCracken, Moisescu-Pareja, Letourneau, Precup, Love, arXiv 2505.18266** (fetched abs + html). "Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks".
  - Thm 4.7 / Cor 4.8 give **sufficiency** only: "Learning O(log n) distinct frequencies gives a logit margin Omega(log n)". Random frequencies with m > 2 ln n - 2 ln(2-2rho).
  - No lower bound or necessity statement. The only related remark (Sec 4.3) is that single-frequency solutions exist but "have poor margins".
  - So it is the complementary upper-bound direction of cor:few, which is the matching necessity result for the normalized max-margin. It should be cited at cor:few / thm:scaling as the ML counterpart.
  - BIB BUG: refs.bib `mccracken2025` has the wrong title ("Emergence of sparse and dense circuits...") and the wrong author ("McCracken, Andrew"). The arXiv page says Gavin McCracken et al., with the title above. Already flagged in c6/f10 items; corrected bibtex below.
- **He, Wang, Chen, Yang, arXiv 2602.16849** (fetched html). They define "full diversification" (every frequency represented) and show noise cancellation by majority voting (Prop 4.2). No bound on the number of frequencies, and no statement that few frequencies give a vanishing margin.
- **arxiv.org/search "grokking Fourier frequencies"** (10 hits, 2023-2026). These include 2606.17399, 2606.12966, 2604.06256, 2603.23784, 2503.22059 and 2510.00468. None states a log-p lower bound on the frequency count.
  - 2605.07648 (auxiliary modulus) showed up in an S2 hit list; it is title-only and looks off-topic.
- The existing scoop check (notes, 09-19) covered Morwani (dense at L_{2,nu}), Li et al. 2402.09469, Ding et al. 2405.17420, Mohamadi et al. 2407.12332, Tian CoGO and Kunin AGF. None has a K = o(log p) necessity statement.

ML verdict: no precedent in the ML literature for the necessity direction. The closest is McCracken 2025's sufficiency bound, which should be cited alongside.

## Checker's citations, re-verified

- **alon1994random**: CONFIRMED.
  - PDF fetched; text extracted with pdftotext. Prop 3 reads verbatim: "For every fixed delta > 0 there is a constant c = c(delta) > 0 such that the following holds; If A is an Abelian group of order n, and X(A,S) is a delta-expander, then |S| >= c log2 n."
  - The abstract says "For Abelian groups this is essentially tight".
  - Crossref (10.1002/rsa.3240050203): RSA 5(2) 271-284, 1994. The bib is correct.
  - Nuance: Prop 3 is about **vertex** expansion and is qualitative. It gives normalized one-sided gap -> 0 when |S| = o(log n), through the easy direction of the eigenvalue-expansion link, which the paper notes holds for non-constant degree.
  - For s > 1, the equal-share ratio is x^{s-1} * gap. So a qualitative gap -> 0 does NOT by itself give cor:few; the rate e^{-2x} is needed. The recommended sentence should call Prop 3 the "qualitative analogue", not the corollary.
- **friedman2006spectral**: metadata CONFIRMED via Crossref (JCTB 96(1) 111-121, 2006, DOI 10.1016/j.jctb.2005.06.012). ScienceDirect returns 403.
  - The zbMATH review (Zbl 1083.05024, fetched via the zbMATH API) states the result as lambda_2 >= d - C_d d n^{-4/d} + o(d n^{-4/d}) (Thms 6, 7).
  - That is a fixed-d, n -> infinity statement with a d-indexed constant. The checker's "absolute constant" is not supported by what I could read.
  - The recommended sentence "a 2K-regular Cayley graph on Z_p has second eigenvalue 2K - O(K p^{-2/K})" uses it uniformly in K growing with p. It should instead cite the elementary Dirichlet pigeonhole bound, which is uniform in K and has the same exponent (4/d = 2/K): lambda_2 >= 2K cos(2 pi p^{-1/K}) >= 2K - 4 pi^2 K p^{-2/K}. FMT can stay as the sharp fixed-d result.
- Recommended text says "the positive-Fourier inequality is needed for all K = o(log p)". This is the checker's own reasoning. A weighted geometry-of-numbers (Minkowski ellipsoid) version of Dirichlet may also reach it; I did not close the small-weight case. Suggest "we use", not "is needed". That is for the novelty_math lens.

## Result
upheld = true. The verdict PARTIALLY_KNOWN and the one confirmed citation (alon1994random) survive. Caveats on the FMT uniformity and the "needed" wording are above. Add McCracken 2025 (corrected bib) as the sufficiency counterpart.

## Infra note (for the orchestrator / user)
- This session's WebSearch budget is exhausted (every call returns "200 of 200 WebSearch calls").
- The arXiv export API and the Semantic Scholar API both return HTTP 429 (rate-limited).
- Parallel verifiers are sharing one search quota and hammering the same free APIs. That is the likely cause of the slow verify stage: agents fall back to many slow WebFetch round-trips.
