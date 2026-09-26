# VERIFY (citation lens): U6-cosine-sum-minima, task 94

Verdict: the checker's CITATION_FOUND stands. All three citations exist and their bib fields are correct. The two main citations say what the checker claims. The proposed paper text needs four small fixes (below).

## What I fetched this session
The three PDFs were downloaded and converted with pdftotext. Metadata comes from api.crossref.org.

1. **Alon & Roichman, "Random Cayley graphs and expanders", RSA 5(2):271-284, 1994.** DOI 10.1002/rsa.3240050203 (Crossref matches the bib entry). Text from web.math.princeton.edu/~nalon/PDFS/exp1.pdf.
   - Thm 2 (Sec 3.1). For any abelian A of order n, a random multiset of (1+eps)log2 n elements gives |mu*_1| <= 1-delta almost surely. The proof bounds the tail for each character and then takes a union bound over the n-1 characters. That is exactly the argument in app:scaling (Hoeffding plus a union bound over offsets), so the existing appendix sentence "cyclic case of the random Cayley graph argument of Alon-Roichman" is accurate. It should cite Thm 2 specifically.
   - Prop 3 (Sec 3.1), verbatim: "If A is an Abelian group of order n, and X(A,S) is a delta-expander, then |S| >= c log2 n." The paper introduces it as showing "that the log n term in Theorem 1 is necessary for every Abelian group". Prop 3 is about vertex expansion. It carries over to our one-sided lambda_2 through the easy direction (spectral gap implies expansion), which the paper says "works for non-constant degrees as well".
   - Numbering check. The author PDF is a 2002 retypeset, but FMT (2006, published version) cites "[AR94, Proposition 3, Section 3]", so the numbering agrees with the journal version.
   - Prop 7' exists: a closed-walk lower bound on max |mu| for l = Omega(log n/log log n). It is two-sided.
2. **Friedman, Murty & Tillich, "Spectral estimates for Abelian Cayley graphs", JCTB 96(1):111-121, 2006.** DOI 10.1016/j.jctb.2005.06.012 (Crossref and the PDF header both match the proposed bibtex). The abstract matches the checker's quote.
   - Thm 2 (Sec 2) gives lambda_2 >= d - Delta^2(n/2^t, r). Here lambda_2 is the second largest eigenvalue, the one-sided quantity, which is the right one: with y_j = 1, m(y) = K - lambda_2/2.
   - Combined with the naive sphere-packing bound (3), this holds for every d and n, not only asymptotically. Cor 3 states it as lambda_2 >= d - C_d d n^{-4/d}.
   - Tightness: Thm 7 (Sec 4) holds for fixed even d with most graphs on an odd prime n.
   - The introduction credits AR94 Prop 3 as the proof of the folklore fact that "Abelian Cayley graphs cannot be good expanders". This independently confirms Prop 3 is the canonical citation for necessity.
3. **Alon, "Perturbed identity matrices have high rank: proof and applications", CPC 18(1-2):3-15, 2009.** DOI 10.1017/S0963548307008917 (Crossref matches). Text from identity1.pdf.
   - Thm 7.1: an eps-biased set of n +-1 variables needs m >= Omega(n/(eps^2 log 1/eps)). The checker left out a condition: this holds only if eps >= 2^{-n/2}.
   - It is two-sided, stated for Z_2^n, and the uniform case is credited to [3] (Alon-Goldreich-Hastad-Peralta). It adds nothing the paper needs, so I recommend dropping it.

## Problems in the recommended change
1. **Scaling.** "With equal shares, m(y) is K minus half the second eigenvalue of Cay(Z_p,{+-j})" is only true for y_j = 1. Under the budget, equal shares are y_j = K^{-s}, so m(y) = K^{-s}(K - lambda_2/2). Write "with y_j equal to 1 on K frequencies" or "proportional to".
2. **Direction.** "Pigeonhole alone gives only K = O(log p/log log p)" should read **Omega**(log p/log log p). Dirichlet's bound gives m(y) <= 2pi^2 (sum_j y_j) p^{-2/K} <= 2pi^2 p^{-2/K}. Comparing this with Gamma_s >= (1/2)(9 ln p)^{1-s} gives a lower bound on K of order log p/log log p. As written, the sentence claims an upper bound.
3. **FMT tightness.** "Tight for prime n" should read "tight for fixed d and prime n" (Thm 7 fixes d). Our regime is K of order log p, where only the lower bound, Thm 2 with (3), is uniform. Suggested pin: \citep[Thm.~2, Cor.~3]{friedman2006spectral}.
4. **Pin the existing appendix citation.** Change `\citet{alon1994random}` in app:scaling to `\citet[Thm.~2]{alon1994random}`.

The rejection of Chowla is correct. Chowla's problem asks how negative min sum cos must be. Here the question is how close to K the value max_{delta!=0} sum cos must come, which is the opposite tail. The checker did not recommend the Chowla papers, so I did not re-verify their arXiv IDs.
