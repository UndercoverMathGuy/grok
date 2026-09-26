# VERIFY (novelty_ml) - R22 finitely many shapes (prop:shapes, app:shape)

Verdict: the checker's PARTIALLY_KNOWN holds. Both proof steps are classical. I found no ML or learning-theory paper that states the result for Gamma_s(p), or says that the max-margin shape is piecewise constant in the exponent s.

## Search limits (infrastructure)
- WebSearch refused every call: "this session has used its web search budget (200 of 200 WebSearch calls)". All parallel verifier agents draw on one session-wide budget of 200 searches, and it is used up.
- The arXiv API (export.arxiv.org) and the Semantic Scholar search API both returned HTTP 429 (rate limited). DuckDuckGo returned a CAPTCHA, and Bing returned junk.
- So the adversarial search below relied on direct fetches of known candidates, not open-ended search. Every verifier running after this point will be equally limited. This also explains why the verify stage is slow: agents retry against blocked or rate-limited endpoints.

## Citations the checker confirmed (checked myself)
- ge2011note: CONFIRMED.
  - Crossref (10.1007/s10107-011-0470-2) gives Math. Program. 129(2):285-299, 2011; authors Ge, Jiang, Ye.
  - I extracted the text of preprint lpmin_v14.pdf myself. Problem (1) is min sum x_j^p s.t. Ax=b, x>=0, 0<p<1.
  - "Theorem 3 The set of all basic feasible solutions of (1) or (2) is exactly the set of local minimizers." The proof is a directional-concavity argument.
  - "Theorem 1 The Lp (0 < p < 1) minimization problems (1) and (2) are both strongly NP-hard."
  - Caveat: the numbering comes from the preprint. Cite by content, or check the numbering in the journal version.
- hirsch1961extreme: METADATA CONFIRMED (Crossref: CPAM 14(3):355-369, 1961, Hirsch & Hoffman).
  - I could not read the body (Wiley paywall). This is the standard citation for "a concave function bounded below on a line-free polyhedron attains its minimum at a vertex", but I did not see it in the text.
- jameson2006counting: METADATA CONFIRMED (Crossref: Math. Gazette 90(518):223-234, 2006; title names Descartes' rule and Laguerre's extensions).
  - The author's Lancaster PDF returned HTTP 500, so I did not see the theorem statement.
  - The paper's Rolle induction is self-contained anyway, so this citation is optional.

## Prior art the checker missed (ML lens)
1. Gunasekar, Lee, Soudry, Srebro, "Implicit Bias of Gradient Descent on Linear Convolutional Networks", NeurIPS 31 (2018), arXiv:1806.00468. Fetched: arXiv abstract and NeurIPS proceedings page.
   - Result: GD on depth-L linear convolutional networks converges in direction to a predictor related to the l_{2/L} bridge penalty in the frequency domain, i.e. max margin under an l_{2/L} quasi-norm (exponent < 1) on DFT coefficients.
   - This is the closest ML precedent for our framing. There, a margin problem under a sub-1 quasi-norm on Fourier amplitudes has an exponent fixed by depth, and it produces frequency sparsity. Here the programme maximises worst coverage subject to sum y_j^{1/s} = 1, with s set by (nu, b).
   - Gunasekar et al. do NOT characterise optima as polyhedron vertices, and do not show piecewise constancy in the exponent. So novelty is intact, but the paper should cite it where it introduces the programme and the exponent 1/s.
2. Fung & Mangasarian, "Equivalence of Minimal l0- and lp-Norm Solutions of Linear Equalities, Inequalities and Linear Programs for Sufficiently Small p", JOTA 151(1):1-10, 2011, DOI 10.1007/s10957-011-9871-x.
   - Metadata confirmed via Crossref and Semantic Scholar. The abstract is elided in both, and I could not find the tech-report PDF (two guessed Wisconsin DMI reports were other papers).
   - The title states the result: for p small enough, lp-minimal solutions over polyhedral sets coincide with l0-minimal (sparsest) solutions.
   - This covers the s -> infinity end of prop:shapes (r = 1/s -> 0). The last interval's shape is a minimum-support vertex of P_p. That is a special case of the switching statement, and it is prior art for it. The finite-switch statement over all of (1, infinity) is not in it, as far as the title and metadata show.

## Proof sanity (brief)
- (i) is correct. The midpoint argument works because t^r is strictly concave on [0,infinity) for 0<r<1, including at 0.
- (ii) is correct. Two wording issues:
  - At switch values of s, the minimising set is a union of the sets on either side.
  - "The support of a maximizer is piecewise constant" should read "up to tied vertices whose non-zero entries agree as multisets (e.g. the maps j -> uj mod p)". The support itself is not unique; only its size and the multiset of ratios are.

## Recommendation
- Keep the checker's sentence citing hirsch1961extreme, ge2011note and jameson2006counting (Jameson optional).
- Add Gunasekar et al. 2018 as the ML precedent for quasi-norm Fourier-domain margins.
- Optionally cite Fung-Mangasarian for the large-s limit, the minimum-support vertex.
- Claim novelty only for applying these facts to Gamma_s(p) and for the resulting jumps (fixed support and ratios, then a switch).
