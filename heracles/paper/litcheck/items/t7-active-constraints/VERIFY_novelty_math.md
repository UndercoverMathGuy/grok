# T7-active-constraints: VERIFY (novelty_math lens), task 54

Claim (appendix.tex:166, proof of Prop. bottleneck): at a vertex of P_p the active constraints have rank |F|; the bounds v_j=0 (j not in S) are unit vectors off S, so the active coverage rows restricted to S have rank K, giving K independent tight offsets and an invertible KxK system for y_S.

## Verdict: KNOWN (upheld). Textbook LP; proof step is correct.

- Maths check: correct. Coordinates in S are positive, so their bound constraints are inactive. Active set = {e_j : j not in S} plus coverage rows for delta in D. Rank |F| means the coverage rows must span R^S modulo span{e_j : j not in S}, which gives rank K on S. Invertibility then gives uniqueness of y_S. The kappa_j(-delta)=kappa_j(delta) reduction to {1..(p-1)/2} is fine.
- Citation bertsimas1997 [Thm 2.3]: confirmed through secondary sources I fetched myself.
  - arXiv 2302.12364 (Liu, Bunea, Niles-Weed; I extracted the PDF text myself): "basic feasible solutions are precisely extreme points (vertices) of the feasible set of (1) (Bertsimas and Tsitsiklis, 1997, Theorem 2.3)". Bib: "Athena Scientific, 1st edition, 1997. ISBN 1886529191".
  - Wikipedia "Basic solution (linear programming)": definition with "at least n of them must be linearly independent", citing B&T 1997, p. 50, ISBN 978-1-886529-19-9. ISBN-10 1886529191 converts to 978-1-886529-19-9, check digit verified.
  - MIT OCW 6.251J (Fall 2009, Bertsimas's course), lecture 2 PDF: states "x is a vertex <=> x is an extreme point <=> x is a BFS". BFS is defined as "subspace spanned by {a_i : i in I} is R^n", where I is the set of tight constraints. This is the same statement, without a theorem number.
  - I could not fetch the book itself because the athenasc.com SSL handshake failed. The theorem number (2.3) rests on arXiv 2302.12364; the page number (p. 50) for the definition rests on Wikipedia. Both agree with my own knowledge of the book. The bibtex fields are correct.
- Chebyshev analogue: the checker's note is reasonable but loose. A closer classical analogue is the Shapley-Snow (1950) kernel theorem for matrix games: extreme optimal strategies are determined by a square nonsingular submatrix, and the opponent is made indifferent on it. Here the K support frequencies and the K tight offsets form a KxK nonsingular "kernel". Crossref confirms the paper exists: DOI 10.1515/9781400881727-004, "Basic solutions of discrete games", Shapley & Snow, Contributions to the Theory of Games I (AM-24), pp. 27-36. I could NOT fetch its text: degruyterbrill returned 405, and Semantic Scholar and arXiv API returned 429. So I do not report it as a verified citation; it is only an optional pointer. No citation is needed for the analogy.
- Adversarial novelty: nothing to find. The paper does not claim novelty for this step. The only non-textbook content is the interpretation of the tight offsets as simultaneous hardest wrong answers. In ML terms, these are the support vectors / tight points of the max-margin KKT system, which is also standard.

## Recommendation
Keep the checker's change: "(a vertex is a basic feasible solution; \citealp[Thm.~2.3]{bertsimas1997})". Optional: a phrase comparing the offsets to a Chebyshev reference set or a Shapley-Snow kernel, uncited.

## Process note (relevant to the workflow being slow)
In this session the WebSearch budget is used up (200/200 calls). Every WebSearch returned "Web search was not performed". export.arxiv.org and api.semanticscholar.org both returned HTTP 429 (rate limited). Every verifier running after this point cannot search, and its fallback APIs are throttled, which slows it down and weakens its checks.
