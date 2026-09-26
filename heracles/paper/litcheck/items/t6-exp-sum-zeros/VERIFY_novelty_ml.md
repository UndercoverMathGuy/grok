# VERIFY (novelty_ml) — T6-exp-sum-zeros, task 51

Claim (appendix app:shape (ii), appendix.tex:156): a real exponential sum sum_{k=1}^N c_k e^{lambda_k u}, distinct lambda_k, not all c_k = 0, has at most N-1 real zeros (Rolle induction).

## Verdict: KNOWN (upheld). Classical, not an ML contribution.

## Primary-source checks (this session)
- Jameson, "Counting zeros of generalised polynomials: Descartes' rule of signs and Laguerre's extensions", Math. Gazette 90(518):223-234 (2006). Author PDF fetched (maths.lancs.ac.uk/jameson/zeros.pdf). Page 1 header: "Math. Gazette 90, no. 518 (2006), 223-234". Eq. (2) is F(x) = sum_j a_j e^{p_j x} ("Dirichlet polynomial").
  - Theorem 3.1: Z(F) <= S[(a_j)], the number of sign changes.
  - **Corollary 3.2**: "A generalized polynomial, or a Dirichlet polynomial, of length n, has at most n-1 zeros (counted with their orders)." This is exactly the paper's statement.
  - The remark after it gives the paper's own proof: multiply by e^{-p_k x} so one term becomes constant, then differentiate to get length n-1, then Rolle.
  - Jameson also says Laguerre's results "were reproduced in the form of exercises ... in [P-Sz], Part V, chapter 1". This backs the checker's "Part V, Ch. 1" pointer. Problem number still unverified.
- Polya-Szego II (Classics in Mathematics, Springer 1998, DOI 10.1007/978-3-642-61905-2). Crossref: title, authors, year, publisher OK. Springer page confirms the subtitle "Theory of Functions. Zeros. Polynomials. Determinants. Number Theory. Geometry", and chapter _2 "The Location of Zeros" (= Part Five), pp. 36-70. I could not see the chapter-1 title or problem numbers directly.
- Borwein-Erdelyi, Polynomials and Polynomial Inequalities, GTM (Springer NY 1995, DOI 10.1007/978-1-4612-0793-1). Crossref OK. The Springer page shows "volume 161", which resolves the checker's caveat. Erdelyi arXiv 1602.02315 (PDF fetched), lines ~1179-1180: "Descartes' Rule of Signs is stated and proved on page 102 of [3]", where [3] = this book.
- Side note: arXiv 1709.06891, which the checker used as evidence, is "Some examples of kinetic scheme whose diffusion limit is Il'in's exponential-fitting". That is a numerical-analysis paper, so it was a tangential source. It is not recommended for citation.

## ML-lens novelty
The paper does not claim the lemma as an ML contribution. It is a proof step inside Prop. shapes (ii), the "finitely many switches" argument. The lemma is textbook (Laguerre 1883; Polya-Szego V.1; Karlin-Studden: exponentials form a Chebyshev/Descartes system). No ML priority question arises.

The **use** of the lemma (the minimizing vertex of phi = sum v_j^{1/s} changes only finitely often in s) is also a standard piecewise-parametric argument. Further ML-literature search was blocked: WebSearch returned "session has used its web search budget (200 of 200)". So I ran no new ML-literature queries. The verdict does not depend on them.

## Recommendation
Keep the one-line Rolle proof. Add: "(classical: generalized Descartes rule of signs; see \citet[Part V, Ch. 1]{polya1998problems}, \citet[Cor. 3.2]{jameson2006counting})". Jameson Cor. 3.2 is the most precise pinpoint. Borwein-Erdelyi GTM 161 (p. 102) is optional, and its volume number is now confirmed.
