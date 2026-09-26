# R31-groups: irrep price = cost of the matrix-multiplication tensor; S5 one irrep, A5 Galois pair; interference beats ceiling on D3/S4/Z12

Verdict: PARTIALLY_KNOWN. The pricing ingredients are known. The count result built on them was not found in prior work.

Our source for the claim: heracles/s5theory/S5_THEORY.md §4-5, §9. Prop S: the invariant block is the chain-trace tensor T_rho = <d,...,d>. Thm G2 gives the ceiling with price kappa. Lemma F is a flattening bound. K1 gives 1 <= kappa <= d^{nu(1-r)}. K2: at even nu the schoolbook is optimal, with d^nu neurons. K3 is the odd-nu lower bound. §9: D3/S4/Z12 counterexamples. The paper's groups.tex is a 1-paragraph skeleton with no proofs.

## Prior art found (all fetched this session)

1. **Derksen, "On the Nuclear Norm and the SVD of Tensors", FoCM 16:779-811 (2016), arXiv 1308.3860.**
   - Thm 1.9: the schoolbook decomposition of M_{p,q,r} = sum e_ij (x) e_jk (x) e_ki is a diagonal SVD, and its singular values are 1 (pqr times).
   - Cor 1.10: ||M_{p,q,r}||_* = pqr.
   - He remarks that Strassen's decomposition has length sum 2 sqrt2 + 12 > 8, so "minimization of the rank, and minimization of the nuclear norm do not always coincide".
   - Thm 1.11: the group-algebra tensor T_G has singular values sqrt(n/d_i), each with multiplicity d_i^3. So ||T_G||_* = sqrt(|G|) sum d_i^{2.5}.
   - Consequences for us:
     - Our K1 at r=1 (kappa=1, "nuclear norm = d^nu") is Derksen Cor 1.10 at nu=3.
     - Our "Strassen loses to schoolbook under the norm" (10 > 8 at q=2) is the quasi-norm analogue of his remark.
     - Morwani Thm 9's sum d^{2.5} is exactly Derksen's nuclear norm of T_G. That is a nice consistency check to state in the paper.
2. **Christandl & Zuiddam, "Tensor surgery and tensor rank", Computational Complexity 28:27-56 (2019), arXiv 1606.04085.**
   - Eq. (7)-(8): the max-cut flattening gives n^{f(G)} <= R(T_n(G)). "For bipartite graphs, each inequality in (8) is an equality."
   - Even cycles are bipartite, so the tensor rank of tr(A_1...A_k) for even k is d^k.
   - Our "rank corollary" (tr(ABCD) has rank d^4) is therefore KNOWN. The flattening proof of K2 is the same cut argument.
3. **Fan, Ding, Yang, Zhang, Udell, "Euclidean-Norm-Induced Schatten-p Quasi-Norm Regularization...", TMLR (01/2023), arXiv 2012.03436.**
   - Def 3: the tensor Schatten-p quasi-norm is inf (sum |s_i|^p)^{1/p} over unit-factor CP decompositions.
   - Thm 1: ||X||_{S_{q/d}}^{q/d} = inf (1/d) sum_i sum_j ||x_i^(j)||^q.
   - Our N_r(T) is exactly ||T||_{S_r}^r with r = q/nu. "L_q weight decay on product-neuron factors = tensor Schatten-(q/nu) quasi-norm" is their Thm 1 (for p=1, d=3 it is Cheng et al. 2016 and Lacroix et al. 2018).
4. **Cheng, Zhang, Zhang, Xu, Ji, Zhou, "Low-Rank Tensor Recovery via Variational Schatten-p Quasi-Norm and Jacobian Regularization", arXiv 2506.22134 (2025).**
   - Thm 1: the variational S_p quasi-norm upper-bounds ||T_(D)||_{S_p}^p for every unfolding T_(D).
   - This is our "N_r(T) >= sum_k sigma_k(flat_S T)^r" (Lemma F applied to a flattening).
   - Hu, LAA 478:188-199 (2015), proves the p=1 version for nuclear norms (crossref-confirmed).
   - The underlying matrix inequality is McCarthy/Rotfel'd p-subadditivity, which S5_THEORY already credits.
5. **Shutman, Louidor, Tessler, arXiv 2509.06931 (2025).**
   - Prop 5.2: rank(delta_{G,phi}^{(x)3}) <= m_d (type I), 3 m_{d/2} (type II), 8 m_{d/2} (type III), where m_d is the tensor rank of d x d matrix multiplication.
   - The paper says the network "effectively implements efficient matrix multiplication in the sense of Strassen".
   - Their setting is Hadamard/product model, GD, no weight decay, and a width (rank) price. **Contrast:** under L_{2,b} the price is a Schatten quasi-norm, and there the schoolbook beats Strassen (Derksen at r=1; our K2 at even nu; numerics at nu=3).
6. **Marchetti, Kunin, Myers, Acosta, Miolane, "Sequential Group Composition", ICML 2026, arXiv 2602.03655.**
   - Lemma B.8: with N >= (k+1) n_rho^{k+1} sigma-pi-sigma (product) neurons, one irrep solves the cost minimisation. The construction is s_w = E_{a0 a1}, S = E_{a_h a_{h+1}}, i.e. the schoolbook with d^nu terms (nu = k+1), replicated k+1 times for phase cancellation.
   - This is an existence/width construction only, with no optimality claim.
   - Remark B.7 also cites Carlini et al. for Waring rank (relevant to power vs product neurons).
7. **Morwani et al., ICLR 2024, arXiv 2311.07568.**
   - Thm 9: quadratic nets, L_{2,3}, real irreps, and a character-sum condition. gamma* = 2/(3 sqrt(3|G|)) / sum_{n>=2} d_{R_n}^{2.5}. Every neuron is single-irrep and every non-trivial irrep appears (this holds for S_k up to k=5).
   - App I.2: for general groups only a subset of irreps is present.
   - This is our b=nu (r=1, kappa=1) dense endpoint. On S5 it predicts ALL irreps, whereas our b<nu result predicts one.
8. **Chughtai, Chan, Nanda, ICML 2023 (PMLR 202), arXiv 2302.03025.**
   - S5 MLP over 50 seeds: "the 1d sign and 4d standard representations are most commonly learned... 6d is never learned". The number of key reps varies.
   - They explicitly say: "the network preferred the standard representation over the standard sign representation, when in fact standard sign offers better performance for fixed weight norm", and "models are trading off weight against performance ... which is least complex, is unclear".
   - This is exactly the question our norm price answers. **Tension:** Chughtai observed variable counts (often sign + standard, sometimes 5d). Our claim "one representation on S5, as observed" must say "one faithful irrep (+ sign patch)" and name our architecture/regime.
9. **Stander, Yu, Fan, Biderman, ICML 2024, arXiv 2312.06581.**
   - One-hidden-layer S5/S6 nets "decompose the group arithmetic using the permutation group's subgroups" (coset circuits).
   - Our §9.3(a) "coset relaxation" is the mechanism by which interference beats the symmetric ceiling on D3/S4/Z12 (the sign character separates the cosets of an index-2 subgroup). It is the margin-theoretic counterpart of Stander's coset circuits and should cite them.
10. **Wu, Jaburi, Drori, Gross, ICLR 2025, arXiv 2410.07476.** Unifies the Chughtai (irrep) and Stander (coset) explanations. Cite alongside item 9.
11. **Levine, arXiv 2609.15289, MAIS-O5.** Poses "which irreps are learned ... as a function of the width and the decay strength". Motivation.

## Searches run
- nuclear norm of the matrix multiplication tensor Derksen
- Morwani group composition d^{2.5} theorem
- Chughtai S5 standard sign representation learned
- tensor rank iterated matrix multiplication trace flattening lower bound
- tensor Schatten-p quasi-norm matrix multiplication
- Hu nuclear norm flattenings
- Schatten-p quasi-norm flattening lower bound CP
- weight decay group composition irrep cost matmul decomposition schoolbook Strassen
- nuclear norm iterated matmul cyclic trace
- max margin group composition which irreps weight decay 2025 2026
- Greps of Kam et al. 2609.17926, He et al. 2606.02993, Marchetti 2602.03655 and Levine 2609.15289 for tensor rank / Strassen / nuclear / schoolbook

## Not found
- Any computation of the tensor Schatten-r quasi-norm (0<r<1) of <d,...,d> / cycle tensors, i.e. our K2 value d^nu at even nu. It is, however, a two-line combination of Christandl-Zuiddam's bipartite cut plus the Schatten-p flattening bound (Cheng Thm 1 / McCarthy). Present K2 as a lemma, not a headline.
- Any result that the weight-decay (L_{2,b}, b<nu) max-margin network on a group uses one irrep on S5 / the Galois pair on A5, or that structured same-irrep interference exceeds the symmetric ceiling on D3/S4/Z12. NOVEL_AS_FAR_AS_FOUND for these.

## Recommended edits to groups.tex
- After "sum of products": "(the tensor-rank version of this link, without a norm, is Shutman et al. 2025, Prop 5.2; under weight decay the relevant quantity is instead the tensor Schatten-(b/nu) quasi-norm, Fan et al. 2023, Thm 1)."
- After "schoolbook algorithm": "(at r=1 this is Derksen's nuclear norm ||<d,d,d>||_* = d^3, Cor 1.10, which also gives the sum d^{2.5} in Morwani et al. Thm 9 via Thm 1.11; at even nu the bound is the bipartite-cut flattening of Christandl & Zuiddam 2019 combined with the Schatten-quasi-norm flattening inequality). The schoolbook construction itself appears in Marchetti et al. 2026 (Lemma B.8)."
- After "one representation on S5": "(plus the sign patch) ... consistent with Chughtai et al. 2023 and Stander et al. 2024, and answering the norm trade-off Chughtai et al. left open."
- After the D3/S4/Z12 sentence: "the excess comes from coset-structured interference (cf. Stander et al. 2024; Wu et al. 2025)".
- Put the contrast with Morwani Thm 9 in one line: at b=nu every irrep is used; at b<nu one is.
