# U1-representation-cost: VERIFY (citation lens), task 89

Sentence checked: approach.tex, end of the programme paragraph (about line 45): "The exponent comes from homogeneity ... (AM--GM), so amplitude A costs a multiple of A^{b/nu}=A^{1/s}." It has no citation now, and refs.bib has none of the candidate entries.

Result: I UPHOLD the checker's CITATION_FOUND. Two bibtex fixes and one wording fix are needed.

## Fetched this session
1. Tibshirani, "Equivalences Between Sparse Models and Neural Networks", 15 Apr 2021. I read the full text of the PDF on stat.berkeley.edu.
   - Sec 2.2, eqs (8)-(9): min over prod a_l = c of sum a_l^2 equals k|c|^{2/k}. The note says this is "the AM-GM inequality ... tight when a_1 = ... = a_k", so the penalty is an l_p penalty with p = 2/k, which is nonconvex for k > 2 and gives "sparse solutions, even more so for greater depth". Sec 3.2 gives the l_{2/k} group-lasso version.
   - No venue is shown, so cite it as a technical note. CONFIRMED.
2. Hoff, arXiv 1611.00040. The abstract confirms that L_q penalties with q <= 1 equal sums of L_2 penalties under a Hadamard (product) parametrisation.
   - Crossref for DOI 10.1016/j.csda.2017.06.007 confirms the published version: Computational Statistics & Data Analysis 115:186-198 (Nov 2017).
   - FIX: cite the journal version, not "arXiv preprint". CONFIRMED.
3. Dai, Karzand, Srebro, NeurIPS 34 (2021). I read the proceedings page and the full PDF.
   - Table 1: a depth-d diagonal network induces the l_{2/d} quasi-norm ("lp quasi-norms iff 2/p in N": diagonal networks).
   - A depth-d convolutional network induces the matrix l_{2/d,2} quasi-norm on the Fourier domain.
   - This is the strongest ML citation: it has the depth-to-exponent rule and the Fourier-domain version. CONFIRMED.
4. Gunasekar, Lee, Soudry, Srebro, NeurIPS 31 (2018). The abstract confirms depth-L linear conv nets converge to a predictor "related to the l_{2/L} bridge penalty in the frequency domain".
   - This is implicit bias, not weight decay, so a "cf." citation is right. CONFIRMED.
5. Kolb, Weber, Bischl, Rügamer, arXiv 2502.02496. The comment line reads "accepted at ICLR 2025". It proves D-factor L2 is equivalent to nonconvex sparse regularisation.
   - OpenReview was behind a browser check, so the venue is confirmed only from the arXiv comment. CONFIRMED.

## Extra (canonical, older)
- Grandvalet, "Least Absolute Shrinkage is Equivalent to Quadratic Penalization", ICANN 98 (Perspectives in Neural Computing), pp. 201-206, 1998. DOI 10.1007/978-1-4471-1599-1_27. The Crossref record was fetched.
- This is the classical origin of the two-factor (lasso = ridge on a product) fact. I read only the metadata, not the body.

## Checked, not for this sentence
- Jacobs-Montúfar, arXiv 2605.19458. The abs page confirms it is about mirror-flow max-margin in homogeneous nets, going from sparse to dense depending on the mirror map. It is not about the weight-decay quasi-norm. It belongs in related work, as the checker said.
- Woodworth et al. 2020: I did not re-fetch it. The checker's rejection is plausible, since that paper is about implicit bias and init scale.

## Issues with the recommended change
- (a) The recommended text says "nonconvex l_q quasi-norm with q<1 ... here with q=b/nu". But the paper's range is s = nu/b in [1, nu/2], so q = b/nu is in [2/nu, 1]. At s = 1 (b = nu) we get q = 1, which is the convex l_1 case, not a quasi-norm. Write "q = b/nu <= 1" (strictly below 1 when b < nu).
- (b) Tibshirani, Dai and Hoff all use b = 2: a sum of squares over all parameters. The paper's cost is a per-neuron L_{2,b} cost, (sum_l a_l^2)^{b/2}. The same AM-GM step gives nu^{b/2} A^{b/nu}. The text should say "generalising the b=2 identity" rather than imply that the sources state the L_{2,b} form.
- (c) Hoff's bibtex should use the CSDA journal version.

## Operational note (for the workflow owner)
- This session's WebSearch budget was already exhausted: "200 of 200 WebSearch calls" on the first search.
- arXiv export API returned 429, and OpenReview served a browser-verification wall.
- The parallel verifiers appear to share one search budget and are being rate-limited. That would explain a slow verify stage where verifiers stall or fall back to fetch-only.
