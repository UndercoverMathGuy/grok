# U7-group-tensor: citations for "the irrep's price is set by decomposing the matrix-multiplication tensor; at even degree this is the schoolbook algorithm"

Location: sections/groups.tex, line 4. The sentence currently has no citations. refs.bib has 11 entries and none of them is Strassen, Blaser, Stander or Shutman.

## Verdict: CITATION_FOUND

1. Strassen 1969, "Gaussian elimination is not optimal", Numer. Math. 13(4):354-356, doi:10.1007/BF02165411. I confirmed the bibliographic data through the Crossref API (api.crossref.org/works/10.1007/BF02165411). Springer's own page redirects to a login. This is the canonical source for "the matmul tensor has a cheaper decomposition than schoolbook": 7 products for 2x2.
2. Blaser 2013, "Fast Matrix Multiplication", Theory of Computing Library Graduate Surveys 5, pp. 1-60, doi:10.4086/toc.gs.2013.005. I fetched the landing page and the PDF. Relevant statements:
   - Sec. 1, Eq. (1.1): the standard algorithm "altogether uses n^3 multiplications". This is the schoolbook count.
   - Def. 4.4/4.5: rank R(F) is the minimal number of products of linear forms, which equals the minimal number of triads summing to the tensor.
   - Sec. 5: <k,m,n> is the matmul tensor. Def. 5.1 defines omega. Example 5.10 gives R(<2,2,2>)=7 (Strassen) and 19 <= R(<3,3,3>) <= 23.
   Use it as the textbook reference for "tensor rank = number of products".
3. Shutman, Louidor, Tessler 2025, arXiv 2509.06931, "Learning words in groups: fusion algebras, tensor ranks and grokking". Prop. 5.2: "Denote by m_d the tensor rank of matrix multiplication for real dxd matrices. Then ... rank(delta_{G,phi^{⊗3}}) <= m_d (type I), 3m_{d/2} (type II), 8m_{d/2} (type III)." They note that the naive (Gaussian) implementation gives the bound d_phi^3. This is the closest ML precedent: neuron count per irrep is set by the matmul tensor rank. Their setting is product/Hadamard nets trained with plain GD, with no norm and no margin. Our contribution is the NORM price, i.e. what each product costs under L_{2,b}, and the count that follows from it. Cite them in this sentence as prior work on the rank side.
4. Stander et al. 2024 (ICLR/ICML: PMLR 235:46441-46467, "Grokking Group Multiplication with Cosets"). I confirmed the metadata, but the abstract has nothing on the matmul tensor or irrep costs; it is about coset circuits. It is NOT a citation for this sentence. It belongs in the S5 empirical discussion, as a contrast: coset circuits vs. the one-irrep result.
5. Chughtai et al. 2023 is already in the bib. Its GCR algorithm (rho(a)rho(b) as matrix multiplication inside irreps) is the natural mechanistic citation for "networks multiply irrep matrices". Consider adding \citet{chughtai2023toy} here too.

## Caveat
The paper's claim is "d^nu neurons per d-dim irrep at even degree". That is the degree-nu analogue of schoolbook's d^3 (a nu-fold product/trace). Blaser and Strassen only cover the bilinear, 3-tensor case. The d^nu count is our own extension, so word it as "the analogue of the schoolbook algorithm".
