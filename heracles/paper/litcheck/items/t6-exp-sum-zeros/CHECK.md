# T6-exp-sum-zeros: KNOWN (classical)

Where it appears: appendix, proof of Prop. shapes (ii) (sections/appendix.tex, around line 156): "a real exponential sum sum_k c_k e^{lambda_k u} with N distinct exponents, not all c_k zero, has at most N-1 real zeros", proved by induction on N using Rolle's theorem.

## Verdict
KNOWN. This is textbook material. It follows from the generalized Descartes rule of signs (Laguerre): after substituting t = e^u the zeros number at most the sign changes of (c_k), which is at most N-1. Equivalently, {e^{lambda_k u}} is a Chebyshev (Haar/Descartes) system on R. The paper's Rolle-induction proof (multiply by e^{-lambda_1 u}, differentiate) is the standard textbook proof.

## Sources (fetched this session)
- Polya & Szego, Problems and Theorems in Analysis II, Springer (Classics in Mathematics reprint 1998; 1976 English edition). DOI 10.1007/978-3-642-61905-2, confirmed via Crossref. The Part Five title "The Location of Zeros" and its Ch. 1 "Rolle's Theorem and Descartes' Rule of Signs" come from a search-result summary. The problem number that is often quoted (V.75) is NOT verified: I could not open the book.
- Borwein & Erdelyi, Polynomials and Polynomial Inequalities, Springer GTM 161, 1995. DOI 10.1007/978-1-4612-0793-1, confirmed via Crossref. Erdelyi, arXiv 1602.02315 (fetched), says: "Section 3.2 of [1] ... gives an introduction to Descartes systems. Descartes' Rule of Signs is stated and proved on page 102 of [3]", where [3] = Borwein-Erdelyi 1995.
- Jameson, "Counting zeros of generalised polynomials: Descartes' rule of signs and Laguerre's extensions", Math. Gazette 90(518):223-234, 2006. DOI 10.1017/S0025557200179628, confirmed via Crossref. It is a readable modern exposition of exactly the generalized-exponent Descartes rule.
- Karlin & Studden, Tchebycheff Systems, Wiley 1966, p.10. arXiv 1709.06891 App. 3.1 (fetched) gives the same Rolle induction "following [37, page 10]" and calls the bound (with polynomial coefficients, at most N_n-1 roots) "the Polya-Szego estimate". I did not fetch the book's own metadata page.

## Recommended change
In app:shape (ii), after "at most N-1 real zeros", add
"(this is classical: the generalized Descartes rule of signs, see \citet[Part V, Ch. 1]{polya1976problems} or \citet{jameson2006counting})".
Keep the one-line Rolle proof so the paper stays self-contained.
