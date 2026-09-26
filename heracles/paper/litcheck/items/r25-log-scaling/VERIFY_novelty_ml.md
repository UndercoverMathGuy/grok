# VERIFY (novelty_ml): R25-log-scaling

Claim (cyclic.tex thm:scaling, cor:few; appendix app:scaling): 1/2 ceil(8 ln p)^{1-s} <= Gamma_s(p) <= 3/2 (2a_s)^s (ln p)^{1-s}; maximizers use Theta(ln p) frequencies with a share floor; gamma*_b = Theta(p^{-nu/2}(ln p)^{1-nu/b}) over all product networks; Cor few: K-frequency networks get at most ~(ln p)^{1-s} x^s e^{-sx/(2a_s)}, x = ln p/K.

Checker verdict: PARTIALLY_KNOWN. **Upheld, with one correction (Cor few).**

## Operational note (why this stage is slow)
- WebSearch returned "session has used its web search budget (200 of 200 WebSearch calls)" on the first call. The parallel verifiers share this budget, and it is now used up.
- export.arxiv.org returned HTTP 429 six times in a row (15 s backoff each). api.semanticscholar.org also returned 429.
- As a result, each verifier falls back to WebFetch on arxiv.org/search listing pages, with retries. That is much slower per query.
- What worked: arxiv.org/search via WebFetch, arxiv.org/abs pages, api.crossref.org via curl, and PDFs via WebFetch followed by pdftotext on the saved binary.

## Citations the checker confirmed: all re-fetched this session
- **alon1994random** (web.math.princeton.edu/~nalon/PDFS/exp1.pdf, pdftotext). The abstract says "For Abelian groups this is essentially tight". Thm 2: (1+eps) log2 n random elements suffice for any abelian A. The Remark after Thm 2: eps log2 n suffice for Z_n. Prop 3: "If A is an Abelian group of order n, and X(A,S) is a delta-expander, then |S| >= c log2 n." CONFIRMED. The bib fields match. A DOI could be added.
- **friedman2006spectral** (Tillich PDF, pdftotext). JCTB 96 (2006) 111-121. The abstract says "for fixed d ... second eigenvalue bounded below by d - O(dn^{-4/d}), where the implied constant is absolute ... for a large odd prime n ... cannot be improved". It also mentions the "large trace" approach [Ci05]. CONFIRMED. Caveat: this is a fixed-d statement, so it supports "fixed K" wording and not K growing with p.
- **cioaba2006closed** (numdam PDF, pdftotext; Crossref). CRAS 342 (2006), issue 9, 635-638. It proves that abelian Cayley graphs contain many closed walks of even length (Thm 1.2, a Serre-type count). CONFIRMED. Add number={9}.
- **mccracken2025** (arxiv.org/abs/2505.18266 and the /html version). The correct title is "Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks". The authors are Gavin McCracken, Gabriela Moisescu-Pareja, Vincent Letourneau, Doina Precup and Jonathan Love. Thm 4.7 and Cor 4.8 state: "Learning O(log n) distinct frequencies gives a logit margin Omega(log n)". The margin is unnormalised, only sufficiency is shown, and the paper says "we don't know why it occurs". CONFIRMED. The existing refs.bib entry is wrong: its title ("Emergence of sparse and dense circuits...") and first author ("Andrew") do not match. Also check that intro.tex's claim ("count moves from dense to sparse as the degree changes") is actually in this paper; that belongs to item c6.

## Adversarial ML search
Queries run on arxiv.org/search via WebFetch:
- "modular addition" frequencies (14 hits)
- "modular addition" margin (6 hits)
- MAIS modular addition
- grokking frequencies logarithmic
- modular arithmetic logarithmic number of frequencies (0 hits)

Abstracts read:
- 2511.23443 (Huang-Li, sine activation): an Õ(p) sample-complexity and margin bound; no frequency count.
- 2606.05863 (Tan-Gai-Zhang, two clocks): a Schatten penalty; no log p.
- 2608.04451 (MAIS-O60 counterexample): single-neuron alignment.
- 2602.16849 (He et al.): already audited in memory; its diversification is dense.

None states a necessity bound on the number of frequencies, a weighted or quasi-norm-budget version, or a Theta over all networks. The only ML precedent for log n remains McCracken (sufficiency only).

Related framing precedent, not about log p: Gunasekar-Lee-Soudry-Srebro, arXiv 1806.00468. Their abstract says GD on depth-L linear convolutional networks converges to an "l_{2/L} bridge penalty in the frequency domain". This is the ML precedent for "a concave quasi-norm budget on Fourier amplitudes gives sparse frequencies", which is the l_{1/s} budget in eq:programme. It is worth one sentence where the programme is introduced (item R4), not in thm:scaling.

## Correction to the checker: Cor few is largely classical (Dirichlet)
The checker says the weighted form of Cor few was "not found anywhere". But Dirichlet's simultaneous approximation theorem gives a stronger version for arbitrary weights in one line. Source: Schmidt, Diophantine Approximation, LNM 785, p.27 Thm 1B, as cited on Wikipedia; the statement was fetched this session.

The argument:
1. Take alpha_i = j_i/p for the K used frequencies and N = p-1.
2. Some 1 <= q <= p-1 has ||q j_i/p|| <= (p-1)^{-1/K} for every i.
3. Hence 1 - cos(2 pi q j_i/p) <= 2 pi^2 (p-1)^{-2/K}.
4. Because r = 1/s < 1 and sum_j y_j^r <= 1, we also have sum_j y_j <= 1.
5. Therefore m(y) <= 2 pi^2 (p-1)^{-2/K} for any weights. The exponent 2/K is FMT's tight exponent; eq:Kbound only achieves s/((6+s)K).

What Dirichlet does NOT give:
- (a) The o(log p) conclusion over the whole range. It gives it only for K <~ 2 ln p/((s-1) ln ln p).
- (b) The sharp upper bound (ln p)^{1-s}. A heavy/light split plus Dirichlet loses a (ln ln p)^{s-1} factor.

So the Bessel heat-trace argument (eq:posfourier) is what buys the sharp weighted rate, and that part stays unfound in the literature.

Recommendation: in app:scaling, say that for fixed K the few-frequency decay follows from Dirichlet's theorem with the better exponent 2/K, and that the heat-trace bound is needed for the sharp rate and for K close to log p. Do not present Cor few's fixed-K decay as new.

## Verdict
PARTIALLY_KNOWN stands.
- Known: the equal-share/unweighted Theta(log p) (AR Thm 2 + Prop 3 + FMT); the fixed-K few-frequency decay for any weights (Dirichlet); the ML sufficiency of log n frequencies (McCracken).
- Not found: the sharp weighted two-sided bound under the l_{1/s} budget with explicit constants; the maximiser support Theta(ln p) and the share floor; the network Theta(p^{-nu/2}(ln p)^{1-nu/b}).
