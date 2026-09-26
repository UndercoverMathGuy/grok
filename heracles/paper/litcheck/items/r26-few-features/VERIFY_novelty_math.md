# VERIFY (novelty_math): R26-few-features, cor:few (cyclic.tex l.125; proof appendix.tex l.250)

## Verdict: checker NOT upheld

The mathematical core of cor:few is a short consequence of a classical tool: Dirichlet's pigeonhole principle with a separate radius for each frequency (Minkowski's linear forms theorem, equivalently the lower bound on the size of a Bohr set with several radii). This covers the weighted case under the l_{1/s} budget for every K = o(log p), and it gives a sharper rate than the paper's Bessel argument.

- The checker says Dirichlet only reaches K = O(log p / log log p). That is true only when every frequency gets the same radius.
- The checker says the positive-Fourier inequality (eq:posfourier) is *needed* for all K = o(log p). That is false.
- **Do not add the checker's recommended_change sentence as written**, because it states that false claim in the paper.

## Argument (verified, and checked numerically)

Setup:
- y >= 0 is supported on a set S with |S| = K.
- The budget is sum y_j^{1/s} <= 1.

Steps:
1. Put rho_j = (c/y_j)^{1/2}. Let S' be the set of j with rho_j < 1/2.
2. Pigeonhole on the box indices floor(N_j {j delta/p}), with N_j = ceil(1/rho_j) <= 2/rho_j. This gives some delta != 0 with ||j delta/p|| <= rho_j for every j in S', provided 2^{K'} c^{-K'/2} prod_{S'} y_j^{1/2} < p, where K' = |S'|.
3. By AM-GM on y^{1/s}, prod_{S'} y_j <= K'^{-sK'}.
4. So it suffices to take c = 4(1+eta) max_{k<=K} k^{-s} p^{-2/k}.
5. Bound F term by term:
   - for j in S': y_j (1 - cos) <= 2 pi^2 y_j rho_j^2 = 2 pi^2 c;
   - for j not in S' (so y_j <= 4c): y_j (1 - cos) <= 2 y_j <= 8c.
6. Hence m(y) <= 2 pi^2 K c.
7. For K <= 2 ln p / s the max in step 4 is at k = K. This gives

   **m(y) <= 8 pi^2 K^{1-s} p^{-2/K} = 8 pi^2 (ln p)^{1-s} x^{s-1} e^{-2x},  with x = ln p / K.**

8. Relative to Gamma_s ~ (ln p)^{1-s}, the ratio is O(x^{s-1} e^{-2x}), which tends to 0 whenever K = o(log p).
   - The paper's eq:Kbound decays only like x^s e^{-s x/(6+s)}.
   - The network step (h <= ||A||_{1/s} m(A/||A||)) is the paper's own and is unchanged.

Numerical check (scratch wd.py):
- Grid: p in {101, 1009, 10007, 100003}, s in {1.25, 1.5, 2, 3}, K <= 6, with random supports and random shares on the budget sphere.
- Result: max m / (8 pi^2 K^{1-s} e^{-2x}) = 0.25. The bound is never violated.
- At these p the paper's bound (eq:Kbound) is vacuous (values of 1 to 10, against m of about 1e-3).

The same pigeonhole, together with the tail bound y_(i) <= i^{-s}, also gives the upper bound Gamma_s = O((ln p)^{1-s}) in thm:scaling. So the Bessel/positive-Fourier route is not needed there either. It is correct, just not required.

## Citations

- **alon1994random**: CONFIRMED. I fetched the PDF at web.math.princeton.edu/~nalon/PDFS/exp1.pdf and extracted the text.
  - Prop 3 reads verbatim: "For every fixed delta > 0 there is a constant c = c(delta) > 0 such that the following holds; If A is an Abelian group of order n, and X(A,S) is a delta-expander, then |S| >= c log2 n."
  - The abstract says "For Abelian groups this is essentially tight".
- **friedman2006spectral**: metadata CONFIRMED via the Crossref API record for 10.1016/j.jctb.2005.06.012.
  - Title, authors (Friedman, Murty, Tillich), JCTB 96(1):111-121, Jan 2006 all match. Semantic Scholar agrees.
  - Content NOT fetched by me: ScienceDirect returns 403, the Cioaba CRAS page returns 403, and Cioaba's homepage refused the connection. The lambda_2 >= d - O(d n^{-4/d}) statement therefore rests on the checker's secondary source.
  - It covers only equal shares, so it is still a fine cite for that case.
- **New (missed)**: the canonical source for Minkowski's linear forms / convex body theorem, which gives weighted simultaneous Dirichlet.
  - Cassels, An Introduction to the Geometry of Numbers, Springer 1959, doi 10.1007/978-3-642-62035-5. Metadata confirmed via Crossref; theorem number not verified (paywall).
  - Also Schmidt, Diophantine Approximation, LNM 785, Springer 1980, doi 10.1007/978-3-540-38645-2. Metadata via Crossref.
  - Wikipedia "Minkowski's theorem" (fetched) notes that it proves simultaneous Dirichlet approximation and points to Cassels and Schmidt.

## Recommended text instead of the checker's

> For equal amplitudes this is classical: abelian Cayley graphs need Omega(log n) generators to expand [Alon-Roichman, Prop. 3], quantitatively [Friedman-Murty-Tillich]. The weighted case also follows from Dirichlet's pigeonhole with frequency-dependent radii rho_j proportional to y_j^{-1/2} (Minkowski's linear forms theorem, [Cassels]), which gives m(y) <= 8 pi^2 K^{1-s} p^{-2/K} for K <= 2 ln p / s.

Alternatively, replace the Bessel proof of eq:Kbound with this argument: it is shorter and sharper. The claim that is new is the transfer to network margins (interference allowed), not the Diophantine inequality.

## Process note (relevant to the workflow slowness)

- This session's WebSearch budget is EXHAUSTED ("200 of 200 WebSearch calls"). Every WebSearch call from verifiers now returns nothing.
- The arXiv export API returned HTTP 429 (Too Many Requests).
- Verifiers therefore fall back to slow guessed-URL WebFetch chains. This is likely a major cause of the slow verify stage.
- Fix: raise CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION, or run fewer verifiers per session.
