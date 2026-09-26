# R6 single-neuron budget: literature check

Verdict: **PARTIALLY_KNOWN**. At b = nu the result is known. The extension to b < nu is a direct generalisation using the same steps. The ReLU bound was not found anywhere.

## Our statement (approach.tex l.54-58; appendix.tex app:models l.270-283)
C_b := max_{||omega||<=1} sum_j |beta_j|^{b/nu}.
- Product neuron: C_b^{nu/b} = B_p = 2(2p nu)^{-nu/2} for all 2 <= b <= nu. Proof: AM-GM on each frequency, then superadditivity of t^{b/2}, then Parseval. Equality holds for balanced single-frequency cosines.
- Power neuron: C_b^{nu/b} <= n! B_p, because only the multilinear monomial survives, and it has coefficient n!.
- ReLU neuron (b = nu = 2): C_2 <= p^{-n}/2. Proof: Cauchy-Schwarz, Parseval on the diagonal, and Var(sigma(z)) <= Var(z).

## Prior art (both PDFs downloaded and read this session)
1. **Morwani et al., ICLR 2024 (arXiv 2311.07568), Appendix, cyclic proof of Thm 7, eqs. (6)-(9)**. The quadratic neuron (u_a+v_b)^2 w has expected class-weighted margin 2/((p-1)p^2) sum_{j!=0} u^(j) v^(j) w^(-j). The proof uses Plancherel ("||u^||^2+||v^||^2+||w^||^2 <= p"), then "the inequality of quadratic and geometric means" (eq. 9): |u^||v^||w^| <= (z(j)/3)^{3/2}. Next, Cauchy-Schwarz with ||z||_2 <= ||z||_1 gives sqrt(2/27)/(p^{1/2}(p-1)). Equality holds only for a balanced single frequency, sqrt(2/(3p)) cos(...). This is our argument exactly, at n=2 and b=nu=3. Their Lemma 18 is the group analogue: two Cauchy-Schwarz steps plus AM-GM on Frobenius norms, which gives the d^{-1/2} per-irrep factor.
2. **Li, Liang, Shi, Song, Zhou, AISTATS 2025 (PMLR 258:523-531; arXiv 2402.09469)**.
   - Lemma D.7: for the power neuron (sum u_i(a_i))^k w, the single-neuron class-weighted margin reduces to k!/((p-1)p^k) sum_{j!=0} w^(-j) prod u^_i(j). This is our "only the multilinear term survives, with coefficient n!" step.
   - Lemma D.8 (formal Lemma 4.2): uses "inequality of quadratic and geometric means" and then max_{||z||_1<=p/2} sum z(j)^{(k+1)/2} <= (p/2)^{(k+1)/2}. This is our superadditivity step at b = nu.
   - The maximum is 2(k!)/((2k+2)^{(k+1)/2}(p-1)p^{(k-1)/2}), attained only by balanced single-frequency cosines. It equals n! B_p * p/(p-1) = n! B_p Gamma_1(p) with k = n.
   - So at b = nu, for every n, the power-neuron budget is known and attained. The product neuron is the same computation without the k! factor.

## What is new
- b < nu: the budget sum_j |beta_j|^{b/nu}, where the exponent b/nu < 1 turns into a quasi-norm on amplitudes, and the observation that its value B_p does not depend on b. The steps are the same (AM-GM, then t^{b/2} superadditive, then Parseval). Morwani/Li only need t^{nu/2}. This is a routine generalisation, and reviewers will read it as such.
- Equality at b = 2 lets a neuron carry many frequencies. This is a small new observation.
- ReLU C_2 <= p^{-n}/2 (Lipschitz-variance argument): no precedent found. Morwani (Sec. 6 / Fig 4b) and Li treat ReLU only empirically. Memana 2608.04451 studies the dynamics of a single ReLU neuron, not a budget.

## Recommended changes
- approach.tex l.54 already says "as in Morwani, is a single-neuron optimization". Extend it: "for b = nu this computation is Morwani et al. (2024, App., eqs. 6-9) at n=2 and Li et al. (2025, Lemmas D.7-D.8) for general n; for b < nu the same AM-GM/Parseval argument applies with t^{b/2} superadditive."
- appendix app:models power-neuron paragraph: cite Li Lemma D.7 for the n!-multilinear step.
- The power-neuron bound "C_b^{nu/b} <= n! B_p" can be stated as "=". beta_j is exactly n! times the product-neuron expression (the paper says so), so the balanced single-frequency cosine neuron attains n! B_p at every b. Li Lemma D.8 confirms this at b = nu for all k. At present the paper only notes attainment at n=2, b=3. Before changing it, the author should confirm that the "only multilinear term" claim covers every mode entering beta_j.
- **li2025nsum is in refs.bib but is not cited in any section.** The only mention is a TODO comment at intro.tex:22. It must be cited: it is the direct precedent for the power-neuron budget and for Cor dense at general n.

## Searches
WebSearch: "ReLU network modular addition maximum margin single neuron bound Fourier Lipschitz variance weight decay"; "max margin modular addition L2,b norm below homogeneity single neuron Fourier AM-GM power neuron multilinear". Fetched: arxiv.org/html/2311.07568, arxiv.org/pdf/2311.07568 (read in full, text extracted), arxiv.org/abs/2311.07568, arxiv.org/html/2402.09469, arxiv.org/pdf/2402.09469 (read in full, text extracted), proceedings.mlr.press/v258/li25b.html.
