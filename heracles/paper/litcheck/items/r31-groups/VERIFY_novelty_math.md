# VERIFY (novelty_math) -- R31-groups, task 34

Verdict: I uphold the checker's PARTIALLY_KNOWN. The main claim (the ceiling fixes the count at one faithful irrep + sign on S5 and the Galois pair on A5, with structured interference beating the ceiling on D3/S4/Z12) was not found anywhere. The pricing machinery is assembled from known pieces, and I found two further precedents for the "norm prefers schoolbook, rank prefers Strassen" framing that the checker missed.

## Tooling degradation (this explains why the verify stage is slow)
- WebSearch refused every call: "this session has used its web search budget (200 of 200 WebSearch calls)". The budget is shared by the session, so every later verifier gets zero searches.
- The arXiv export API (export.arxiv.org/api/query) and the Semantic Scholar API both returned HTTP 429 Too Many Requests.
- DuckDuckGo returned a CAPTCHA page. Bing returned irrelevant results (nuclear power).
- Workarounds that did work: the arxiv.org/search HTML page, arxiv.org/abs pages, and PDFs saved by WebFetch and then run through local pdftotext. Large PDFs over 10 MB (Marchetti 2602.03655, Cheng 2506.22134) fail in WebFetch.

## Confirmed citations (fetched and checked this session)
- Derksen 1308.3860. Thm 1.9: the schoolbook is a diagonal SVD with pqr singular values equal to 1. Cor 1.10: ||M_{p,q,r}||_* = pqr. The remark that Strassen's decomposition has length 2sqrt2+12 > 8, so "minimization of the rank, and minimization of the nuclear norm do not always coincide", is verbatim. Thm 1.11 gives the singular values of T_G. The nuclear-norm formula sqrt(n) sum d_i^{5/2} is **Cor 1.12**, not Thm 1.11, so cite Cor 1.12 for the d^{2.5}.
- Christandl-Zuiddam 1606.04085, eqs (7)-(8): n^{f(G)} = R(A) <= R(T_n(G)), and "For bipartite graphs, each inequality in (8) is an equality." Verified.
- Fan-Ding-Yang-Zhang-Udell 2012.03436, "Published in TMLR (01/2023)". Def 3 (tensor Schatten-p quasi-norm) and Thm 1 (Schatten-q/d = inf (1/d) sum_i sum_j ||x_i^(j)||^q) are verified. Its footnote also credits Lacroix et al. 2018 (p=2/3, d=3) as an earlier special case.
- Cheng et al. 2506.22134: the abstract says the variational Schatten-p quasi-norm is "a common upper bound for the Schatten-p quasi-norms of arbitrary unfolding matrices". I could not open the PDF (too large), so the theorem number is unverified.
- Shutman-Louidor-Tessler 2509.06931. Prop 5.2 (the rank of the irrep projection is <= m_d, 3m_{d/2}, ...) is verified. The abstract says "in the sense of Strassen". One correction: the experiments use "AdamW optimizer without weight decay", not "plain GD".
- Marchetti et al. 2602.03655: the HTML confirms that "(k+1) n_rho^{k+1} sigma-pi-sigma neurons" suffice (Thm 4.3 construction). The Lemma B.8 label itself fell in a truncated part of the page. The arXiv page lists no venue; I did not verify ICML 2026.
- Morwani 2311.07568, Thm 9: the formula is verified, and the theorem holds for S_k up to k=5. **Missed by the checker:** the hypothesis m >= 2 sum d_R^3 means Morwani's own L_{2,3} construction already uses ~d^3 = d^nu neurons per irrep at nu=3. The d^nu neuron count therefore has precedent at nu=3 in Morwani as well as in Marchetti.
- Chughtai 2302.03025 and Stander 2312.06581 exist with the stated authors. The abs pages show no journal-ref, so I did not verify the PMLR volume and pages. Wu et al. 2410.07476 is ICLR 2025, verified.

## Prior art the checker missed (math lens)
1. **Ye & Lim, "Fast structured matrix computations: tensor rank and Cohn-Umans method", arXiv 1601.00292, Sec 3.2.** It defines sum |lambda_i| over a unit-factor decomposition as the numerical-stability measure of a bilinear algorithm, and the nuclear norm as the "optimal numerical stability". It then shows that the rank-optimal Gauss algorithm for complex multiplication (cost 2(1+sqrt2)) loses to the usual algorithm (||mu_C||_* = 4, from Friedland-Lim Prop 6.1). This is the canonical statement that "rank-optimal != norm-optimal" for bilinear algorithms. The paper's price N_r is the r<1 version of this stability measure. Ye-Lim's footnote 2 says the l_1 sum cannot be replaced by l_p for p>1. Cite it next to Derksen's remark.
2. **Dumas, Pernet & Sedoglavic, "Strassen's algorithm is not optimally accurate", arXiv 2402.05630.** They minimise a norm-based growth factor "along the unique orbit of Strassen's 2x2-matrix multiplication tensor decomposition". The paper's numerics ("Strassen's orbit under GL_2^3, polished, costs exactly 10 > 8") do the same thing (optimise a norm cost over the Strassen orbit) with a different objective. Cite it as a methodological precedent.
3. The closely related 2506.19405 (automated fast and accurate recursive matmul, growth factors) is in the same line. I saw it only in the search listing; it is not needed.

## Novelty of the math pieces
- K1 at r=1: Derksen Cor 1.10. Known.
- K2 (even nu, schoolbook optimal for all r in (0,1]): the max-cut flattening of the even-cycle trace tensor is a permutation matrix (CZ), plus the p-subadditivity of Schatten quasi-norms (McCarthy/Rotfel'd; Cheng Thm 1 for unfoldings). This is a three-line assembly. An arXiv abstract search for a Schatten-p (p<1) computation for matmul, cycle or iterated-matmul tensors returned nothing. It should be presented as a lemma, as the checker recommends.
- Odd nu (Strassen 10 > schoolbook 8 at q=2): this is the r<1 analogue of the Derksen and Ye-Lim observations. Found nowhere as a number. It is numerics only.
- Group-level count claims (S5 one irrep + sign, A5 Galois pair, interference beating the ceiling on D3/S4/Z12): no prior found. Coverage is limited because web search was unavailable.
