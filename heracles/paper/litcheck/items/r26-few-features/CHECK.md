# R26-few-features: cor:few (cyclic.tex l.125; proof appendix.tex l.250 and eq:Kbound)

Statement: any network with at most K non-zero amplitudes has h/||theta||^nu <= C_b^{nu/b} (3/2) (x/(e^{x/(2a_s)}-1))^s (ln p)^{1-s}, where x = ln p/K. So K = o(log p) gives a vanishing fraction of gamma*_b. Interference is allowed.

## Reduction to known objects
The coverage sum is F_y(delta) = sum_j y_j (1 - cos(2 pi j delta/p)). With equal weights on a set S, min_{delta != 0} F equals |S| minus the largest non-trivial eigenvalue of Cay(Z_p, S u -S) (eigenvalues are character sums; Alon-Roichman Sec. 3 proof of Prop 2). So the unweighted form of cor:few says that an abelian Cayley graph with K = o(log p) generators has a vanishing normalised one-sided spectral gap.

## Prior art found (verified)
1. Alon & Roichman 1994, RSA 5(2):271-284. PDF fetched from web.math.princeton.edu/~nalon/PDFS/exp1.pdf.
   - Prop 3: "For every fixed delta>0 there is a constant c=c(delta)>0 ... If A is an Abelian group of order n, and X(A,S) is a delta-expander, then |S| >= c log2 n." The proof is a diameter/volume count.
   - Abstract: "For Abelian groups this is essentially tight".
   - Prop 2: the Z_2^m case with constant 1+Omega(delta log 1/delta).
   - The paper already cites them (appendix l.196) for the lower bound only.
2. Friedman, Murty & Tillich, "Spectral estimates for Abelian Cayley graphs", JCTB 96(1):111-121, 2006, DOI 10.1016/j.jctb.2005.06.012.
   - Result: a d-regular abelian Cayley graph on n vertices has lambda_2 >= d - O(d n^{-4/d}), with an absolute constant.
   - Confirmed from the search-result abstract, from the DOI resolving to pii S0095895605000948, and from Cioaba's C. R. Acad. Sci. 342 (2006) 635-638 (fetched). That note cites it as ref. [6] and states the bound verbatim.
   - The ScienceDirect page itself returned 403.
   - With equal shares y = K^{-s} on K frequencies (d = 2K), this gives m <= C K^{1-s} e^{-2x}. So m/Gamma_s <= C x^{s-1} e^{-2x} -> 0, which is exactly cor:few for equal shares, with a better exponent.
3. Folklore (Dirichlet simultaneous approximation / pigeonhole): some delta != 0 has ||j delta/p|| <= p^{-1/K} for all K frequencies. This is the one-line proof behind item 2.
4. epsilon-biased set lower bounds, e.g. Alon, "Perturbed identity matrices have high rank", CPC 18 (2009) 3-15 (search result only, not fetched):
   - They give size Omega(log N/(eps^2 log 1/eps)).
   - They are two-sided and mainly for Z_2^n. That is weaker relevance than the one-sided Cayley-gap bounds above.
5. Cioaba 2006 (fetched): for fixed k, abelian Cayley graphs have >= C|G| eigenvalues >= k - eps. Same phenomenon, not needed.

## What is new
The weighted version with a concave l_{1/s} budget.
- Weights with sum y^{1/s} <= 1 and support <= K. Dirichlet alone gives m <= 2 pi^2 e^{-2x} ||y||_1 <= 2 pi^2 e^{-2x}. Compared with Gamma_s ~ (ln p)^{1-s}, this vanishes only when K = O(log p/log log p), not for all o(log p). The weighted pigeonhole does not directly exploit the budget.
- The paper's positive-Fourier / Bessel I_0 averaging (eq:posfourier + eq:psisum) closes the gap to the full o(log p).
- It is stated relative to gamma* over all networks, which includes interference.
- I found no weighted / quasi-norm-budget analogue in the expander, Bohr-set or ML literature. One targeted search came back empty.
- The Bessel trick is standard in flavour: moment-generating averaging over characters with non-negative Fourier coefficients, as in Bohr-set and Chowla-type arguments. I found no specific source.

## Verdict: PARTIALLY_KNOWN
The equal-share / unweighted specialisation is classical (Alon-Roichman Prop 3 qualitatively; Friedman-Murty-Tillich quantitatively). The weighted l_{1/s} statement over all K = o(log p) and its application to network margins appear new.

## Recommended edit
After cor:few (or in the proof), add: "For equal shares this is the classical fact that Cayley graphs of abelian groups need Omega(log n) generators to expand \citep[Prop.~3]{alon1994random}; quantitatively, a 2K-regular Cayley graph on Z_p has second eigenvalue 2K - O(K p^{-2/K}) \citep{friedman2006spectral}. The corollary extends this to unequal amplitudes under the l_{1/s} budget, where the pigeonhole argument alone gives only K = O(log p/log log p)." Add the friedman2006spectral bib entry.
