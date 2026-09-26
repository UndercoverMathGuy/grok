# T8-random-frequencies: CHECK

Item: app:scaling lower bound (appendix.tex l.182-196). Draw L = ceil(8 ln p) i.i.d. uniform frequencies. Apply Hoeffding (cos in [-1,1], mean -1/(p-1)), so P[avg cos > 1/2] <= e^{-L/8} <= 1/p. A union bound over the p-1 offsets then gives coverage >= L/2 at every delta != 0.

Verdict: **KNOWN** (standard technique). The paper already credits it correctly ("cyclic case of the random Cayley graph argument of Alon-Roichman").

## Arithmetic check
Hoeffding with range 2 gives P[mean - E >= t] <= exp(-L t^2/2). The needed deviation is t = 1/2 + 1/(p-1) >= 1/2, so the bound is <= e^{-L/8} <= 1/p because L >= 8 ln p. The union bound gives (p-1)/p < 1. Coverage = sum(1 - cos) >= L - L/2. Correct.

## Sources (all fetched this session)
- Alon & Roichman, "Random Cayley graphs and expanders", RSA 5(2):271-284, 1994 (Crossref 10.1002/rsa.3240050203; PDF at web.math.princeton.edu/~nalon/PDFS/exp1.pdf).
  - Thm 1: for any group of order n, c(delta) log2 n random elements give E|mu1*| < 1 - delta.
  - Sec 3.1, Thm 2 (Abelian): a multiset of (1+eps) log2 n random elements (with repetitions) gives |mu1*| <= 1 - delta a.s. The proof bounds each nontrivial character separately (P(chi(sigma+sigma^-1) >= 1.8) <= 1/2) and then takes a union bound over characters. The eigenvalues of X(Z_p, ±J) are exactly 2 sum_l cos(2 pi J_l delta/p). Our statement is this with a one-sided bound 1/2 and explicit constant 8. The multiset-with-repetitions model matches ours exactly.
- Naor & Naor, "Small-Bias Probability Spaces: Efficient Constructions and Applications", SICOMP 22(4):838-856, 1993 (10.1137/0222053).
  - This is about explicit constructions over Z_2^n. Its main relevance is the eps-biased-set language. Secondary citation only.
- Alon, Goldreich, Hastad, Peralta, "Simple constructions of almost k-wise independent random variables", RSA 3(3):289-304, 1992 (10.1002/rsa.3240030308).
  - Explicit small-bias sets over Z_2^n. Secondary.
- Azar, Motwani, Naor, "Approximating probability distributions using small sample spaces", Combinatorica 18(2):151-171, 1998 (10.1007/pl00009813).
  - Small-bias spaces over Z_d, i.e. the cyclic (non-binary) setting. Statement not read; metadata only.
- McCracken et al., arXiv 2505.18266, Thm 4.7 (verbatim): "Suppose m > 2 log_e n - 2 log_e(2-2rho) log_e(pi/delta) - 1 [as rendered]. Then, with probability at least rho, for all k != i+j mod n, h(i+j)-h(k) > delta m."
  - Here h(k) = sum_l cos(2 pi f_l (k-i-j)/n) with uniform random f_l. Cor 4.8: "Learning O(log n) distinct frequencies gives a logit margin Omega(log n)."
  - This is the same random-cosine lower bound in the modular-addition ML setting (unnormalised margin, no optimality). It must be cited at this lemma.

## Recommendation
Keep Alon-Roichman and Hoeffding. Add McCracken et al. 2025 as the direct ML precedent, which needs a fixed bib entry (see sweep item 1). Optionally add one sentence in eps-biased language citing Naor-Naor 1993 / AGHP 1992. Novelty sits entirely in the matching upper bound, not in this lemma.
