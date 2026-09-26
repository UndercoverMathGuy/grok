# VERIFY (citation lens): U7-group-tensor, task 95

Sentence (groups.tex:4): "each irreducible representation pays a price set by how cheaply the
matrix-multiplication tensor can be written as a sum of products; at even degree this is the
schoolbook algorithm, with $d^\nu$ neurons per $d$-dimensional representation."

## Checker's confirmed citations, re-fetched this session

- **strassen1969**: checked on Crossref (10.1007/BF02165411). Title "Gaussian elimination is not optimal", V. Strassen, Numer. Math. 13(4):354-356, 1969. The bib is correct.
- **blaser2013**: checked on the ToC landing page and the full PDF, extracted with pdftotext.
  - Series: Theory of Computing Library, Graduate Surveys 5, 2013, doi 10.4086/toc.gs.2013.005. ToC's own BibTeX gives pages={60}, and the page footer reads "pp. 1-60", so "1--60" is fine.
  - Sec. 1, after Eq. (1.1): "altogether uses n3 multiplications". Confirmed.
  - Def. 4.4: rank = minimal number of products of linear forms. Def. 4.5: triads. Confirmed.
  - Minor slip in the checker's evidence: Example 5.10 gives R(<2,2,2>) **<= 7**, not "= 7". Equality is Hopcroft-Kerr/Winograd 1971, not Bläser's statement. The paper sentence does not rely on this.
- **shutman2025**: checked on the arXiv abs page and v1 html.
  - Title, authors and year are correct. Submitted 8 Sep 2025, v1 only. I could not confirm a venue because Semantic Scholar returned 429 and the web-search budget was exhausted.
  - Prop. 5.2 is quoted correctly: rank <= m_d (type I), 3 m_{d/2} (type II), 8 m_{d/2} (type III). The naive Gaussian method gives d^3. The model is the Hadamard model C^T(Ax ⊙ By), trained with plain GD and no regularisation.
  - Wording fix: the bound m_d holds only for type-I (real) irreps and is stated per "basic self-conjugate" representation. Say "each real irreducible representation", not "each irreducible representation". On S5 and A5 every irrep is real, so the application is fine.
- **chughtai2023toy** (already in the bib): PMLR 202:6243-6267, ICML 2023. Checked on the PMLR page. The bib entry lacks volume and pages. Optionally add `series={PMLR}, volume={202}, pages={6243--6267}`.

## Missed citation, and a better one for this exact sentence

**Derksen, "On the Nuclear Norm and the Singular Value Decomposition of Tensors", Found. Comput. Math. 16(3):779-811, 2016, doi 10.1007/s10208-015-9264-x, arXiv:1308.3860.** I checked the bib data on Crossref and read the arXiv v2 full text.

- **Thm 1.9 / Cor 1.10.** The schoolbook decomposition (4) of M_{p,q,r} is a diagonal SVD with pqr singular values equal to 1. So ||M_{p,q,r}||_* = pqr, and the **schoolbook attains the nuclear norm**.
  - Verbatim: "the sum of the lengths of the pure tensors in the decomposition (5) [Strassen] is 2√2+12 > 8. This shows that minimization of the rank, and minimization of the nuclear norm do not always coincide."
- **Thm 1.11 / Cor 1.12.** The group-algebra multiplication tensor T_G = Σ_{ghk=1} g⊗h⊗k has a diagonal SVD. The singular value sqrt(n/d_i) appears with multiplicity d_i^3, one block per irrep, and "T_i corresponds to matrix multiplication in Hom(Z_i,Z_i)". This gives ||T_G||_* = sqrt(n) Σ d_i^{5/2}.

Why this matters:
- The paper's "price" is a **norm** (L_{2,b} budget), not a rank. Under a norm, Strassen's rank saving does not pay: the schoolbook wins. The paper's own notes say the same (S5_THEORY row 4c: "Strassen's rank-7 orbit costs exactly 10 > 8 at q = 2").
- Citing only Strassen/Bläser, which are about rank, for a "price" whose optimum is the schoolbook could suggest the opposite of what is proved.
- Derksen is the direct mathematical precedent for two claims at degree 3 with the nuclear-norm price:
  - the group-multiplication tensor splits by irreps into matmul blocks with d^3 pure terms;
  - the schoolbook is norm-optimal.
- The paper's generalisation is to degree ν (iterated tensor <d,...,d>, d^ν terms) and to the L_{2,b} exponent. That generalisation is new relative to Derksen, and the paper should say so.
- Possibly useful: if the ν=3, q=ν case is the nuclear norm, then Derksen Cor 1.10 may settle the ν=3 schoolbook optimality that S5_THEORY lists only as COMPUTED at q=2 and 2.5. I have not verified this mapping.

Optional canonical book in place of or alongside Bläser: Bürgisser, Clausen, Shokrollahi, *Algebraic Complexity Theory*, Grundlehren, Springer 1997, doi 10.1007/978-3-662-03338-8. I checked the metadata on Crossref only; the chapter content was not fetched.

## Recommended sentence

"... a price set by how cheaply the matrix-multiplication tensor can be written as a sum of products under the norm \citep{strassen1969,blaser2013}; unlike for tensor rank, the schoolbook decomposition is norm-optimal, as \citet{derksen2016nuclear} showed for the nuclear norm of the matrix-multiplication and group-algebra tensors, and we prove its degree-$\nu$ analogue at even $\nu$, with $d^\nu$ neurons per $d$-dimensional representation. \citet{shutman2025} bound the width each real irreducible representation needs by the tensor rank; we add its price under the $L_{2,b}$ norm."

## Verdict
CITATION_FOUND is upheld, and all three confirmed citations are real and correctly described, apart from two small points: the "=7" slip and "irreducible" vs "type-I/real". Add Derksen 2016 as the primary citation for the norm price and schoolbook optimality.

## Operational note (re: slow verify stage)
- This session's WebSearch budget is exhausted: every call returns "used its web search budget (200 of 200 WebSearch calls)".
- The Semantic Scholar API returns HTTP 429.
- As a result, each verify agent falls back to slower direct fetches and PDF downloads. That is probably a large part of why the verify stage is slow. Raising CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION, or running fewer verify agents in parallel, would help.
