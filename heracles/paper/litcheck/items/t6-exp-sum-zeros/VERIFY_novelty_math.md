# VERIFY (novelty_math) -- T6-exp-sum-zeros, task 52

Claim (appendix.tex:156, app:shape (ii)): a real exponential sum sum_{k=1}^N c_k e^{lambda_k u}, distinct lambda_k, not all c_k = 0, has at most N-1 real zeros; proof by Rolle induction.

## Verdict: KNOWN (checker upheld)
Classical: Laguerre's extension of Descartes' rule of signs to generalized (Dirichlet) polynomials. The paper does not claim it as new; it only needs a citation.

## Sources fetched this session
- Jameson, "Counting zeros of generalised polynomials: Descartes' rule of signs and Laguerre's extensions", Math. Gazette 90(518):223-234 (2006). I checked the metadata on Crossref and read the full text via the author's copy (maths.lancs.ac.uk/jameson/zeros.pdf).
  - Eq. (2): F(x) = sum_j a_j e^{p_j x}.
  - Theorem 3.1: Z(F) <= number of sign changes of (a_j), where zeros are counted with their orders.
  - **Corollary 3.2: "A generalized polynomial, or a Dirichlet polynomial, of length n, has at most n-1 zeros (counted with their orders)."** This is exactly the paper's statement.
  - Right after Cor. 3.2, Jameson gives the paper's proof: "taking F_0(x) = e^{-p_k x}F(x), so that one term becomes constant and F_0' has length n-1".
  - Historical note: the result appears in Laguerre (1883) and in Polya-Szego, Aufgaben II.
  - Best pinpoint citation: **Jameson 2006, Cor. 3.2**.
- Polya-Szego, Problems and Theorems in Analysis II, Classics in Mathematics, Springer 1998, DOI 10.1007/978-3-642-61905-2. Crossref confirms authors, year and series. OpenLibrary (ISBN 9783540636861) confirms the subtitle "Theory of Functions. Zeros. Polynomials. Determinants. Number Theory. Geometry".
  - Crossref chapter record 10.1007/978-3-642-61905-2_2 is "The Location of Zeros" (Part Five), pp. 36-70.
  - I could not verify the sub-chapter title "Rolle's Theorem and Descartes' Rule of Signs" or the problem number (V.75). Cite as [Part Five] or [Part V, pp. 36-70], not by chapter or problem number.
- Borwein-Erdelyi, Polynomials and Polynomial Inequalities, Springer 1995. Crossref confirms authors, year, publisher and the GTM series. OpenLibrary (ISBN 9780387945095) confirms **GTM vol. 161**. So the checker's from-memory volume number is correct.
  - The "Descartes' rule on p. 102" location is second-hand. It comes from Erdelyi, arXiv 1602.02315 ("Inequalities for exponential sums"), whose PDF I fetched: "Descartes' Rule of Signs is stated and proved on page 102 of [3]", with [3] = Borwein-Erdelyi 1995.
  - Minor: the same passage says "Section 3.2 of [1]" for Descartes systems, and that paper's [1] is Bernstein 1926, not Borwein-Erdelyi. The checker's summary copied this literally. It doesn't matter for this item.
- Gosse-Vauchelet, arXiv 1709.06891 (PDF fetched): Appendix 3.1, "Elementary proof of the Polya-Szego estimate", gives the same Rolle induction "following [37, page 10]", where [37] = Karlin-Studden, Tchebycheff Systems (Wiley 1966).
  - This confirms the attribution to Polya-Szego and to Karlin-Studden.
  - I did not fetch Karlin-Studden itself, so I am not recommending it as confirmed.

## Proof check
The paper's induction is correct as written.
- Base case N=1: c_1 e^{lambda u} has no zero when c_1 != 0.
- Inductive step: g = e^{-lambda_1 u} f has the same zeros as f. g' is an (N-1)-term exponential sum with exponents lambda_k - lambda_1 (still distinct) and coefficients c_k(lambda_k - lambda_1).
  - If all of those coefficients are zero, g is a non-zero constant, so it has no zeros.
  - Otherwise, Rolle gives Z(g) <= Z(g') + 1 <= N-1.
- Applying it with u = 1/s and lambda_k = log(v_j) is correct.

## Recommendation
Keep the one-line proof. Change the bullet to: "... at most N-1 real zeros (classical: Laguerre's extension of Descartes' rule of signs; see \citet[Cor. 3.2]{jameson2006counting} and \citet[Part V]{polya1998problems})."
Drop the "Ch. 1" pinpoint for Polya-Szego unless someone checks it against the book. Borwein-Erdelyi is optional; its fields are correct, including GTM 161.

## Search note
The WebSearch budget for the whole session is used up: "200 of 200 WebSearch calls", returned on my first query. Crossref also returned HTTP 429 once. I did this check with WebFetch, curl and pdftotext only. Other verifier agents that depend on WebSearch will get no search results and will stall or retry, which probably explains the slow verify stage.
