# U4-matrix-tools: textbook citations for the appendix tools

Where: sections/appendix.tex, lem:minor (line ~50-55), lem:hadamard (~58-62), lem:bernstein (~82-88), and the "positive average" remark (~107).
Verdict: CITATION_FOUND. These are standard tools, so citing textbooks is enough. There is no novelty question here.

## Sources I confirmed

- **Horn and Johnson, *Matrix Analysis*, 2nd ed., Cambridge University Press, 2012 (print 2012-10-22). DOI 10.1017/CBO9781139020411.**
  - Confirmed through Crossref metadata (edition-number 2; ISBN 9780521839402 and 9780521548236).
  - The CUP page for the 1985 edition (DOI 10.1017/CBO9780511810817) also loaded.
  - The book covers the Sylvester identity det(I+AB)=det(I+BA), the expansion of the characteristic polynomial by principal minors (E_k = sum of k x k principal minors), and Hadamard's inequality for positive semidefinite matrices, with equality iff the matrix is diagonal.
  - **I did not verify section numbers.** From memory: Hadamard's inequality is Thm 7.8.1; the principal-minor expansion is in Sec. 1.2; det(I+AB)=det(I+BA) is in Ch. 0/1. Cite by chapter, or check the numbers against a copy before quoting them.
- **DLMF Sec. 5.12 (https://dlmf.nist.gov/5.12), fetched.**
  - Eq. 5.12.3: int_0^inf t^{a-1}(1+t)^{-(a+b)} dt = B(a,b), for Re a, Re b > 0.
  - Eq. 5.12.1: B(a,b) = Gamma(a)Gamma(b)/Gamma(a+b).
  - Take b = 1-a. Then with **DLMF 5.5.3** (https://dlmf.nist.gov/5.5, fetched), Gamma(z)Gamma(1-z) = pi/sin(pi z), you get int_0^inf u^{a-1}/(1+u) du = pi/sin(pi a) for 0<a<1.
  - Note: 5.12 alone does **not** state pi/sin(pi a). You need to cite 5.12.3 together with 5.5.3.
- **Schilling, Song and Vondracek, *Bernstein Functions: Theory and Applications*, De Gruyter, 2012. DOI 10.1515/9783110269338.**
  - Confirmed through Crossref (authors, title, publisher, 2012-09-25). This is the 2nd edition, De Gruyter Studies in Mathematics 37. The edition and series come from memory; Crossref does not list them.
  - It is relevant to the remark after lem:bernstein. There, functions of the form cx + int log(1+xt) mu(dt) are Bernstein (indeed Thorin-Bernstein) functions, and x^r (0<r<1) is a complete Bernstein function.
  - The lemma itself is elementary, so this citation is optional. It helps most for the "not special to power penalties" remark. I did not verify the chapter number for Thorin-Bernstein functions (from memory, Ch. 8).

## Recommended edits
- lem:minor proof: "Sylvester's identity [Horn-Johnson 2012]" and "Expanding by principal minors [Horn-Johnson 2012]".
- lem:hadamard proof: "Hadamard's inequality [Horn-Johnson 2012, Thm 7.8.1]". Check the theorem number first.
- lem:bernstein proof: "by the Beta integral [DLMF 5.12.3 with b=1-a, and 5.5.3]".
- Remark ~line 107: "i.e. psi is a Bernstein function of Thorin type [Schilling-Song-Vondracek 2012]".
