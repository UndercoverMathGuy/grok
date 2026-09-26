# R25 log-scaling (thm:scaling, cor:few, app:scaling): literature check

Verdict: PARTIALLY_KNOWN. The core is a weighted version of a classical expander fact. The weighted l_{1/s} statement, the share floor and the translation to network margin are new as far as I found.

## The dictionary a reviewer will spot
cov(j,delta) = 1 - cos(2 pi j delta/p). So the coverage sum_j y_j (1 - cos(2 pi j delta/p)) is half of the Laplacian eigenvalue at the character delta of the weighted Cayley graph Cay(Z_p, {+-j} with weight y_j).
- Gamma_s(p) = (1/2) max spectral gap of a weighted Cayley graph on Z_p, under an l_{1/s} budget on the edge weights.
- With equal shares on K frequencies: coverage = K^{1-s} * (normalised gap of a 2K-regular abelian Cayley graph).
- The appendix already says the lower bound is "the cyclic case of the random Cayley graph argument of Alon-Roichman". The upper bound and the "need Theta(log p) frequencies" statement are the other half of the same classical story. The paper does not credit that half.

## Prior art found (fetched and read)
1. Alon, Roichman. Random Cayley graphs and expanders. RSA 5(2):271-284, 1994. doi:10.1002/rsa.3240050203. PDF: https://web.math.princeton.edu/~nalon/PDFS/exp1.pdf (metadata confirmed at collaborate.princeton.edu).
   - Thm 1: c(delta) log2 n random elements give E|mu1*| < 1 - delta.
   - Thm 2: for abelian groups, (1+eps) log2 n random elements suffice. The remark after it: for Z_n, eps log2 n random elements already give a constant gap.
   - Prop 3 (Sec 3.1): "If A is an Abelian group of order n, and X(A,S) is a delta-expander, then |S| >= c log2 n." The proof is by diameter counting.
   - Abstract: "For Abelian groups this is essentially tight."
   - Also: Prop 7' uses a closed-walk count, the same kind of trace argument as our positive-Fourier inequality.
2. Friedman, Murty, Tillich. Spectral estimates for Abelian Cayley graphs. JCTB 96(1):111-121, 2006. doi:10.1016/j.jctb.2005.06.012. PDF: https://www.rocq.inria.fr/secret/Jean-Pierre.Tillich/publications/Abelian.pdf
   - Abstract: "for fixed d, a d-regular Cayley graph on an Abelian group of order n has second eigenvalue bounded below by d - O(d n^{-4/d}) ... for a large odd prime n, the O(dn^{-4/d}) cannot be improved."
   - Constants: C_d asymptotically <= 4e/pi (naive sphere packing, Sec 2) or pi^2/2 (Sec 3).
   - With d = 2K (+-j), the normalised gap is at most C p^{-2/K}, and this is tight at primes. That is the unweighted, equal-share analogue of our Cor few and of eq. (Kbound), which gives p^{-1/(2 a_s K)}. Their exponent is sharper; ours holds for arbitrary weights.
   - Intro: "Another possible approach might be based on the fact that an Abelian Cayley graph has a large trace." Our Bessel/heat-trace inequality is an instance of that approach.
3. Cioaba. Closed walks and eigenvalues of Abelian Cayley graphs. C. R. Acad. Sci. Paris Ser. I 342:635-638, 2006. doi:10.1016/j.crma.2006.03.005.
   - Lower-bounds closed walks by the "balanced" words. This is the combinatorial twin of our step "resonances only add non-negative terms". Optional citation.
4. McCracken, Moisescu-Pareja, Letourneau, Precup, Love. arXiv 2505.18266 (2025).
   - Thm 4.7 / Cor 4.8 (Sec 4.3): "Learning O(log n) distinct frequencies gives a logit margin Omega(log n)". The margin is unnormalised, and this covers only the sufficiency direction.
   - They say "we don't know why it occurs". They give no necessity bound and do not cite Alon-Roichman.
   - Our lower bound on Gamma_s overlaps with this. The refs.bib entry `mccracken2025` is wrong: the correct first author is Gavin McCracken, and the correct title is "Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks".

## Checked and only related
- Chowla cosine problem (Bourgain; Ruzsa; Sanders; Bedert arXiv 2509.05260; arXiv 2509.03490; arXiv 2609.05338). It concerns how NEGATIVE min_x sum_{a in A} cos(ax) must be. Our quantity is the other tail: how close max_{delta != 0} sum y_j cos(2 pi j delta/p) must get to its value at delta = 0, i.e. simultaneous Dirichlet approximation. Not needed as a citation.
- Small-bias sets / eps-biased sets in Z_p (AGHP; Katz). These are two-sided |bias| <= eps. Our coverage condition is a one-sided spectral gap. The lower bound Omega(log p) for constant eps would imply ours only in the unweighted case. AR Prop 3 is the cleaner citation.
- Weighted Cayley sparsification (Hsieh-Lee arXiv 2508.08078) and fastest-mixing chains: adjacent, not needed.

## What is new (not found)
- The weighted problem with a concave l_{1/s} budget on generator weights (s > 1): two-sided Theta((log p)^{1-s}) with explicit constants, holding for ARBITRARY non-equal weights. AR/FMT are for unweighted generator sets. FMT's sphere-packing proof uses integer word counts and does not directly give the weighted statement.
- The maximiser support is Theta(log p), and every maximiser has a share floor. This is an optimality statement, not an expander-existence statement.
- The translation to gamma*_b = Theta(p^{-nu/2} (log p)^{1-nu/b}) over all product networks, interference included, and Cor few for networks.

Caveat: the equal-share restriction of the theorem is essentially a corollary of AR Thm 2 (lower) plus FMT (upper). Write this openly; a reviewer from TCS/combinatorics will recognise it.

## Recommended text change
In cyclic.tex after thm:scaling, or in app:scaling, add:
"Equivalently, the coverage (1/2)*sum_j y_j(2-2cos(2 pi j delta/p)) is half the Laplacian eigenvalue of the weighted Cayley graph Cay(Z_p, {+-j}) at the character delta. So Gamma_s(p) is the largest spectral gap of such a graph under an l_{1/s} budget on the generator weights. For unweighted generator sets, Theta(log p) generators are known to be both sufficient \citep{alon1994random} (Thm 2) and necessary \citep[Prop. 3]{alon1994random}, and K generators leave a normalised gap O(p^{-2/K}) \citep{friedman2006spectral}. Theorem~\ref{thm:scaling} extends this to arbitrary weights under the concave budget, locates the optimum and bounds its shares. \citet{mccracken2025} observed that O(log n) random frequencies give an Omega(log n) (unnormalised) logit margin; our upper bound shows this order is also necessary."
Also fix the `mccracken2025` bib entry.
