# VERIFY (novelty_math): T8-random-frequencies, task 56

Verdict: KNOWN. The checker's verdict is upheld. The lemma is the cyclic case of Alon-Roichman (1994), Sec. 3.1. One recommended sentence overclaims (see 3).

## What I checked myself
- Alon & Roichman, "Random Cayley graphs and expanders", RSA 5(2):271-284, 1994. I fetched the PDF at web.math.princeton.edu/~nalon/PDFS/exp1.pdf and extracted it with pdftotext. Crossref DOI 10.1002/rsa.3240050203 matches.
  - Thm 2 (Sec 3.1): "for any Abelian group A of order n the Cayley graph X(A,S) of A with respect to a (multi-) subset S of (1+eps) log2 n (not necessarily distinct) random elements satisfies |mu*_1| <= 1-delta almost surely".
  - The proof bounds each nontrivial character separately (Pr(chi(sigma+sigma^-1) >= 1.8) <= 1/2, binomial tail). It then takes a union bound: "There are n-1 nontrivial characters". Our proof has the same structure, with Hoeffding in place of the binomial tail and a one-sided threshold of 1/2.
  - The Remark after Thm 2 treats the cyclic group explicitly: "the Cayley graph X(Z_n,S) of the cyclic group Z_n with respect to a subset S of eps log2 n random elements satisfies |mu*_1| <= 1-delta almost surely". So "cyclic case" is literally in the source. Cite Sec. 3.1 (Thm 2 and the Remark).
  - **Prop 3:** "If A is an Abelian group of order n, and X(A,S) is a delta-expander, then |S| >= c log2 n." This is the classical unweighted counterpart of our matching upper bound and of "every maximizer uses >= c_s ln p frequencies" (cyclic.tex l.116).
- Chen, Moore & Russell, "Small-Bias Sets for Nonabelian Groups", APPROX-RANDOM 2013, LNCS pp. 436-451. DOI 10.1007/978-3-642-40328-6_31, confirmed via Crossref; arXiv 1304.5010 abs fetched. The abstract says: "The Alon-Roichman theorem shows that random sets of size O(log|G|/eps^2) suffice" (to be eps-biased). This gives the canonical attribution of the eps-biased reading for general groups, including Z_p. It fits better than Naor-Naor or AGHP, which work over Z_2^n.
- McCracken et al., arXiv 2505.18266. I fetched the abs and HTML.
  - The title is "Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks". The authors are Gavin McCracken, Gabriela Moisescu-Pareja, Vincent Letourneau, Doina Precup and Jonathan Love. **The current bib entry is wrong.**
  - Thm 4.7: frequencies uniform on {1..n/2} and h(k) = sum_l cos(2 pi f_l (k-i-j)/n). If m > [2 ln n - 2 ln(2-2rho)]/[ln(pi/delta) - 1], then with probability >= rho the margin h(i+j)-h(k) > delta m holds for all k != i+j. Confirmed. This is the ML precedent, and it is sharper in its constant than our 8 ln p.
- Crossref metadata confirmed for:
  - Naor-Naor, SICOMP 22(4):838-856, 1993;
  - AGHP, RSA 3(3):289-304, 1992;
  - Hoeffding, JASA 58(301):13-30, 1963, DOI 10.1080/01621459.1963.10500830.
- I re-derived the arithmetic:
  - E cos = -1/(p-1).
  - Hoeffding on [-1,1] gives exp(-L t^2/2), and t >= 1/2 gives <= e^{-L/8} <= 1/p.
  - The union bound gives (p-1)/p < 1.
  - The budget step sum n_j^r <= sum n_j holds for r <= 1.
  - All correct.

## Issues
1. **The Naor-Naor "1/2-biased multiset" wording is inaccurate.** Small bias is two-sided. With L = ceil(8 ln p), the two-sided union bound would be about 2(p-1)e^{-L t^2/2} with t slightly below 1/2 on the lower tail, and that exceeds 1. So the draw is not certified to be 1/2-biased. Say "one-sided analogue of an eps-biased set". If the vocabulary is wanted, cite Chen-Moore-Russell or Alon-Roichman, not Naor-Naor or AGHP. Otherwise drop the sentence.
2. The mccracken2025 bib entry must be fixed before it is cited. The checker flagged this correctly.
3. **"What is new here is the matching upper bound" overclaims.** In the unweighted setting, the fact that Theta(log n) generators are necessary for a spectral gap in abelian groups is Alon-Roichman Prop 3; the pigeonhole/Dirichlet diameter argument is classical. What is new is the weighted L^r-quasinorm (budget) version, with explicit constants and share bounds, proved by the positive-Fourier/Bessel argument. Suggested wording: "the unweighted fact that Omega(log p) frequencies are necessary is \citet[Prop 3]{alon1994random}; what is new is the budget-weighted upper bound and its constants."
4. The constant 8 is loose: McCracken needs about 2.4 ln n at delta = 1/2. This is harmless, but do not present 8 ln p as tight.

## Process note
WebSearch is exhausted for this session: "200 of 200 WebSearch calls" on the first call. All the work here used direct WebFetch, curl to Crossref, and local pdftotext. This is a main reason the verify stage is slow. Each verifier burns turns on failed searches and 429s.
