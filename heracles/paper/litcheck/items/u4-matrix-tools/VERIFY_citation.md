# VERIFY (citation lens): U4-matrix-tools, task 92

Result: the checker's verdict (CITATION_FOUND) is upheld. All three recommended sources exist and support the appendix tools. There are four refinements.

## What I fetched this session
- **DLMF 5.12** (https://dlmf.nist.gov/5.12)
  - 5.12.1: B(a,b) = Gamma(a)Gamma(b)/Gamma(a+b), for Re a, Re b > 0.
  - 5.12.3: int_0^inf t^{a-1}(1+t)^{-(a+b)} dt = B(a,b).
  - No equation in 5.12 states pi/sin(pi a).
- **DLMF 5.5** (https://dlmf.nist.gov/5.5)
  - 5.5.3: Gamma(z)Gamma(1-z) = pi/sin(pi z).
  - So the paper's step int_0^inf u^{a-1}/(1+u) du = pi/sin(pi a) at a = 1-r needs three equations together: 5.12.3 (with b = 1-a), 5.12.1 and 5.5.3. The checker was right about this.
- **DLMF citation page** (https://dlmf.nist.gov/help/cite)
  - Official form: "NIST Digital Library of Mathematical Functions. https://dlmf.nist.gov/, Release 1.2.8 of 2026-09-15. F. W. J. Olver, A. B. Olde Daalhuis, D. W. Lozier, B. I. Schneider, R. F. Boisvert, C. W. Clark, B. R. Miller, B. V. Saunders, H. S. Cohl, and M. A. McClain, eds."
  - The checker's @misc entry puts the equation numbers inside the bib note and drops the release. Use the official entry, and put the equation numbers in \citep[...].
- **Horn & Johnson, Matrix Analysis, 2nd ed.** (Crossref, https://api.crossref.org/works/10.1017/CBO9781139020411)
  - Authors, title, 2nd edition, CUP, print 2012-10-22, DOI: all correct.
  - Crossref chapter records: Ch. 0 "Review and Miscellanea" (pp. 1-42); Ch. 1 "Eigenvalues, Eigenvectors, and Similarity" (pp. 43-82); Ch. 7 "Positive Definite and Semidefinite Matrices" (pp. 425-516).
  - I **could not verify theorem numbers**; Hadamard as Thm 7.8.1 is from memory. Web search was unavailable (see below). Cite Ch. 7 (Hadamard) and Ch. 1 (principal minors, AB vs BA) without theorem numbers unless someone checks a physical copy.
- **Schilling, Song & Vondracek, Bernstein Functions: Theory and Applications**
  - Crossref (https://api.crossref.org/works/10.1515/9783110269338): De Gruyter, 2012, DOI correct.
  - Open Library (https://openlibrary.org/isbn/9783110252293.json): 2nd ed., De Gruyter Studies in Mathematics 37, 450 pp.
  - Crossref chapter records: "Preface to the second edition" and **"Chapter 8. Thorin-Bernstein functions"** (DOI 10.1515/9783110269338.109).
  - So the bib fields the checker gave "from memory" (edition 2, series, vol. 37) are right. Cite \citep[Ch.~8]{schilling2012bernstein} for the other-penalties remark.

## Issues
1. **Name clash on "Sylvester's identity"** (appendix line 55).
   - The identity used there is det(I+AB) = det(I+BA). It is usually called the Weinstein-Aronszajn identity or *Sylvester's determinant theorem* (Wikipedia, fetched).
   - "Sylvester's determinant identity" normally means a different result: the bordered-minor identity (Wikipedia, fetched).
   - From memory, Horn-Johnson's own "Sylvester's identity" in Ch. 0 is also the bordered-minor one. So the checker's pointer \citep[Ch.~0--1]{horn2012matrix} risks sending readers to the wrong statement.
   - Suggested text: "by the identity det(I_nu + A B) = det(I_N + B A) (Sylvester's determinant theorem; e.g. \citealp[Ch.~1]{horn2012matrix}, where AB and BA share their non-zero eigenvalues)".
2. **Hadamard theorem number is unverified.** Use \citep[Ch.~7]{horn2012matrix}, or check that it is Thm 7.8.1 in a copy of the 2nd edition before quoting it.
   - Also: HJ state the equality case for positive definite matrices. The paper's "equality iff diagonal" is still correct, because the diagonal entries are all 1, so equality (det = 1) forces the Gram matrix to be positive definite.
3. **DLMF bib entry**: replace it with the official NIST:DLMF entry (release 1.2.8). Write "\citep[Eqs.~5.12.1, 5.12.3, 5.5.3]{dlmf}".
4. **SSV**: fields confirmed. Add Ch. 8.
   - The remark's psi(x) = cx + int log(1+xt) mu(dt) matches the Thorin-Bernstein log-representation (with psi(0) = 0) after t -> 1/t. I checked the chapter title, but not the exact theorem statement (paywalled).

Math check: I re-derived lem:bernstein by integration by parts plus the substitution u = xt, and got x^r pi/(r sin pi r). The lem:minor step det((Delta U*U)_{L,L}) = prod d_l det(U*U)_{L,L} is also correct.

## Operational note (for the workflow owner)
WebSearch refused every query in this subagent with "this session has used its web search budget (200 of 200 WebSearch calls)". All later verifiers in this litcheck run are therefore working without web search. They fall back to WebFetch on guessed URLs, which is slower and gives weaker verification. Raising CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION, or splitting the run, would fix it.
