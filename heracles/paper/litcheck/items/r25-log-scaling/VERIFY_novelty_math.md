# VERIFY (novelty_math) -- R25-log-scaling (task 26)

## Operational note (important)
- WebSearch returned "session has used its web search budget (200 of 200)" on my first query. No search queries ran in this verify pass.
- export.arxiv.org API returned HTTP 429 (rate-limited, likely from parallel verifier agents).
- So this pass used direct WebFetch of known primary URLs only. It is not a fresh search sweep. The Chowla, small-bias and sidelobe angles could not be searched again. Every later verifier in this workflow run will hit the same WebSearch cap.

## Citations re-fetched this session (all confirmed)
- Alon & Roichman, RSA 5(2):271-284 (1994), PDF from web.math.princeton.edu/~nalon/PDFS/exp1.pdf.
  - Thm 2 (Sec 3.1): (1+eps)log2 n random elements, |mu*_1| <= 1-delta a.s., any abelian group.
  - The remark after it covers Z_n with eps log2 n elements.
  - Prop 3 (Sec 3.1): an abelian delta-expander has |S| >= c log2 n. It is proved by a diameter/counting argument on the unweighted set S.
  - Abstract: "For Abelian groups this is essentially tight."
  - The refs.bib entry is correct. It could add doi 10.1002/rsa.3240050203.
- Friedman, Murty & Tillich, JCTB 96 (2006) 111-121, doi 10.1016/j.jctb.2005.06.012.
  - Abstract: lambda_2 >= d - O(d n^{-4/d}) with an absolute constant, and this "cannot be improved" for large odd prime n.
  - Sec 1 names a "large trace" approach (cites Cioaba [Ci05]).
  - Cites AR Prop 3 as the folklore result.
  - The recommended bibtex is correct. The issue number (1) is not printed in the PDF header. Volume, pages and DOI match.
- Cioaba, C. R. Acad. Sci. Paris Ser. I 342 (2006) 635-638, doi 10.1016/j.crma.2006.03.005.
  - Thm 1.2: a Serre-type count of eigenvalues >= k - eps for abelian Cayley graphs, proved via closed walks (r-tuples summing to the identity).
  - The recommended bibtex is correct. Optional citation.
- McCracken, Moisescu-Pareja, Letourneau, Precup, Love, arXiv 2505.18266 (v1, 23 May 2025), title "Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks".
  - Cor 4.8: "Learning O(log n) distinct frequencies gives a logit margin Omega(log n)". The margin is not norm-normalised.
  - "we don't know why it occurs."
  - Sufficiency only; necessity is not proved.
  - The CURRENT refs.bib entry has the wrong author ("Andrew") AND the wrong title ("Emergence of sparse and dense circuits..."). Replace it with the checker's bibtex.
  - Side flag: intro.tex cites mccracken2025 for "count moves from dense to sparse as the degree of the network changes". Check that this paper actually says that. I did not verify it; it is outside this item.

## Mathematics-lens findings the checker missed
1. Simultaneous Dirichlet approximation (pigeonhole). This is classical; see Schmidt, Diophantine Approximation, LNM, Springer 1980, Thm 1B p.27 as cited on Wikipedia.
   - Given any K frequencies j_1..j_K, there is delta != 0 mod p with ||j_i delta/p|| <= ~p^{-1/K} for all i. Pigeonhole the p points (j_i delta/p)_i in T^K.
   - Since 1-cos(2 pi x) <= 2 pi^2 ||x||^2, F_y(delta) <= 2 pi^2 p^{-2/K} sum_j y_j for ANY weights y.
   - Under the budget sum y_j^{1/s} <= 1 with s >= 1 we have sum y_j <= 1. So the K-sparse value is <= ~2 pi^2 p^{-2/K}, for arbitrary weights.
   - Consequence: the "arbitrary weights" extension of FMT is NOT new at fixed K. Pigeonhole gives it at once, with the sharp exponent 2/K. Cor few's exponent is 1/(2 a_s K), which is weaker.
   - Cor few is still not implied in general. Pigeonhole gives fraction 2 pi^2 e^{-2x} (ln p)^{s-1} of the optimum, with x = ln p/K. That fails to vanish when x grows slower than ((s-1)/2) ln ln p. Cor few's (ln p)^{1-s}-relative form covers the whole range K = o(log p).
   - The checker's recommended sentence "Theorem extends this to arbitrary weights" should therefore be narrowed: arbitrary weights are free for the support-size statement; the uniform concave-budget bound is what is new.
2. Random-walk literature: Hildebrand, "A survey of results on random random walks on finite groups", Probab. Surveys 2 (2005) 33-63.
   - Sec 3.2, Thm 2 (after Greenhalgh): on an abelian group of order n, ANY k generators with uniform weights need >= gamma n^{2/(k-1)} steps to mix.
   - An exercise extends this to unequal probabilities.
   - Sec 3.3 covers k growing with n (k = (log n)^a).
   - This is the mixing-time twin of the FMT/Dirichlet support bound. Optional citation.
3. Technique. eq:posfourier (average of prod_j e^{t y_j cos} >= prod_j I_0(t y_j), because all Fourier coefficients are non-negative) is a trace/closed-walk (Riesz-product-type) positivity argument. It sits in the same family as FMT's "large trace" approach and Cioaba's closed-walk count.
   - The step that is not in those sources is the concavity step under the l_{1/s} budget (eq:psisum), which turns the positivity into a uniform (ln p)^{1-s} bound.

## Verdict
PARTIALLY_KNOWN, upheld.
- Known:
  - the equal-share Theta(log p) law (AR 1994 Thm 2 + Prop 3; FMT 2006);
  - the fixed-K arbitrary-weight vanishing gap (Dirichlet / pigeonhole);
  - the O(log n)-frequency sufficiency in ML (McCracken 2025, Cor 4.8).
- Not found (with the caveat that search was blocked this pass):
  - the two-sided weighted bound under the concave l_{1/s} budget with explicit constants;
  - the maximiser support Theta(ln p) and the share floor;
  - Cor few in the intermediate regime ln p/ln ln p << K = o(ln p);
  - the network rate gamma*_b = Theta(p^{-nu/2}(ln p)^{1-nu/b}).
