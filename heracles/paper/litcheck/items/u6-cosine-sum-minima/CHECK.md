# U6-cosine-sum-minima (uncited background, cyclic.tex thm:scaling discussion)

Statement to support: "sets over Z_p whose cosine sums stay away from their peak (small one-sided Fourier bias / expanding Cayley graphs) need Theta(log p) elements".

## Verdict: CITATION_FOUND

The quantity the paper controls is m(y) = min_{delta!=0} sum_j y_j (1 - cos(2 pi j delta/p)). With equal weights on K frequencies, this is K minus half the second eigenvalue of the 2K-regular Cayley graph Cay(Z_p, {+-j}). So the classical statement to cite is "abelian Cayley graphs need Omega(log n) generators to expand; random O(log n) suffice".

### Fetched and confirmed this session
1. Alon & Roichman, "Random Cayley graphs and expanders", RSA 5(2):271-284, 1994. The metadata is confirmed via Crossref and the text via the author PDF https://web.math.princeton.edu/~nalon/PDFS/exp1.pdf.
   - Abstract: "c(delta) log n random elements ... second largest eigenvalue ... at most (1-delta) ... For Abelian groups this is essentially tight."
   - Sec. 3, Proposition 3 (verbatim): "For every fixed delta>0 there is a constant c=c(delta)>0 such that the following holds; If A is an Abelian group of order n, and X(A,S) is a delta-expander, then |S| >= c log_2 n." The proof uses diameter/volume counting.
   - Proposition 7': a closed-walk lower bound on |mu_1| for arbitrary S in an abelian group. This is the same "count trivial closed walks" idea as our eq:posfourier.
   - Already in refs.bib as alon1994random. It is currently cited only for the random lower bound (appendix). It should ALSO be cited for the matching Omega(log) necessity (Prop. 3).
2. Friedman, Murty & Tillich, "Spectral estimates for Abelian Cayley graphs", JCTB 96(1):111-121, 2006. The metadata is confirmed via Crossref and the text via the author PDF https://www.rocq.inria.fr/secret/Jean-Pierre.Tillich/publications/Abelian.pdf.
   - Abstract (verbatim): "for fixed d, a d-regular Cayley graph on an Abelian group of order n has second eigenvalue bounded below by d - O(dn^{-4/d}) ... for a large odd prime, n, the O(dn^{-4/d}) cannot be improved".
   - Section 2 is the sphere-packing proof (Theorem 2). Section 4, Theorem 7, covers random Cayley graphs on Z/nZ with n prime.
   - This is the quantitative unweighted version of our upper bound. With equal shares, m <= O(K p^{-2/K}), which is o(Gamma) unless K = Omega(log p).
3. Alon, "Perturbed identity matrices have high rank: proof and applications", CPC 18(1-2):3-15, 2009. Metadata is confirmed on the Cambridge page (doi 10.1017/S0963548307008917). Text from the author PDF identity1.pdf.
   - Theorem 7.1: an eps-biased family of n +-1 variables needs a sample space of size m >= Omega(n/(eps^2 log(1/eps))). In group language, eps-biased sets in Z_2^n have size Omega(log|G|/(eps^2 log 1/eps)).
   - This is the TWO-sided small-bias lower bound, and it is stated for Z_2^n only. The rank argument transfers to any abelian group (Gram matrix of characters), but the paper does not state that.
   - Our condition is one-sided (it only needs cosine sums bounded away from the peak). So this is weaker/adjacent. At most, cite it as "cf." for the eps-dependence.

### Checked, not the right citation
- Chowla cosine problem (Bourgain, Ruzsa, Sanders; Bedert arXiv 2509.05260; Jin-Milojevic-Tomon-Zhang arXiv 2509.03490; Shankar arXiv 2609.05338; see sweep/LIT_math.md). It bounds how NEGATIVE min_x sum_{a in A} cos(ax) must be. Ours is the opposite tail: some delta != 0 where sum y_j cos(j delta) is near sum y_j (Dirichlet/Bohr). No implication either way. Do not cite, or cite in one contrastive clause only.
- Sidelobe/coding design (Welch-type bounds, AIKPS90 thin sets with small Fourier coefficients). The Ajtai-Iwaniec-Komlos-Pintz-Szemeredi 1990 metadata is confirmed by the sweep, not by me. These give explicit near-log-size two-sided constructions. Optional.
- Dirichlet simultaneous approximation / Bohr-set size (Tao-Vu Lemma 4.20, via the sweep). This is the one-line pigeonhole proof of the unweighted upper bound. Tao-Vu was not fetched here, so no bib entry.

## What is not covered by any of these
They are all unweighted (unit generator weights, fixed degree) statements. The paper's weighted version is not in them: arbitrary shares y under a concave l_{1/s} budget, with two-sided Theta((log p)^{1-s}), a support of Theta(log p), a share floor, and o(log p) giving a vanishing fraction. Our pigeonhole alone gives only K = O(log p/log log p) in the weighted case (see r26 note). I found no weighted analogue (web-search budget exhausted this session; relied on fetched PDFs and sibling items t9/r25/r26).

## Recommended change (cyclic.tex, after the thm:scaling paragraph "The last statement needs no answer ...", or after cor:few)
Add: "With equal shares, $m(y)$ is $K$ minus half the second eigenvalue of the Cayley graph $\mathrm{Cay}(\sZ_p,\{\pm j\})$. So the unweighted case of Theorem~\ref{thm:scaling} is the classical fact that Cayley graphs of abelian groups need $\Omega(\log n)$ generators to expand, while $O(\log n)$ random ones suffice \citep[Thm.~2, Prop.~3]{alon1994random}. Quantitatively, a $d$-regular abelian Cayley graph on $n$ vertices has $\lambda_2\ge d-O(dn^{-4/d})$, which is tight for prime $n$ \citep{friedman2006spectral}. Our contribution is the weighted version under the concave budget $\sum_jy_j^{1/s}\le1$, where the pigeonhole argument alone gives only $K=O(\log p/\log\log p)$. (Unlike Chowla's cosine problem, which asks how negative a cosine sum must become, this concerns how close to its peak it must return.)"
Optional: cite alon2009perturbed as the two-sided small-bias analogue.
