# VERIFY (citation lens) — C1-morwani-dense-baseline (task 75)

## Sources fetched this session (primary)
- arXiv 2311.07568 abs page (metadata) + full LaTeX e-print (Feb 2024 = v2) + arXiv HTML v2 (theorem numbers).
- arXiv 2402.09469 abs page + LaTeX e-print (Li, Liang, Shi, Song, Zhou; comment "AIStats 2025").
- arXiv 1810.05369 abs page + LaTeX e-print v4 (Wei, Lee, Liu, Ma).
- OpenReview forum page blocked by a browser check, so I could not compare the ICLR camera-ready numbering. WebSearch was also unavailable: the session had used all 200 of its WebSearch calls.

## morwani2024feature: exists, bib correct, CONFIRMED
- The title, all five authors and the order match. The abs page comment reads "Accepted as Spotlight at ICLR 2024".
- The neuron is quadratic, phi = (u(a)+v(b))^2 w, so nu = 3.
- **Thm 7** (arXiv v2 numbering; source label thm:cyclic): "with m >= 4(p-1) neurons. The maximum L_{2,3}-margin ... gamma* = sqrt(2/27) * 1/(p^{1/2}(p-1))". The theorem goes on:
  - (1) each neuron's u, v, w equal lambda*cos(theta + 2 pi zeta x/p) at a single frequency zeta, with theta_u + theta_v = theta_w;
  - (2) "For every frequency zeta in {1..(p-1)/2}, at least one neuron in the network uses this frequency."
  
  The paper's wording "one frequency per neuron, every frequency used" is accurate. It is missing the width condition m >= 4(p-1).
- **Thm 8** (sparse parity): x^k neurons, L_{2,k+1}, m >= 2^{k-1}, gamma* = k! sqrt(2(k+1)^{-(k+1)}), and spt(u_i) = S. This is not an "all features" statement.
- **Thm 9** (groups): quadratic neurons at L_{2,3}, for G with real irreps. It requires m >= 2 sum_{n>=2} d^3 and sum_{n>=2} d^{1.5} chi_{R_n}(C) < 0 for every non-trivial conjugacy class C. Then gamma* = 2/(3 sqrt(3|G|) sum d^{2.5}), each neuron is spanned by one non-trivial irrep, and every non-trivial irrep is used.
- **Thm 1** is Wei et al. Thm 4.1, verbatim: "For any norm, fixed r>0, any homogeneous f ... lim_{lambda->0} gamma_lambda = gamma*".
- **Lemma "Combining neurons"**: needs b = nu. This supports the intro paragraph "Where the dense theorem stops".
- **Discussion**, verbatim: "Our proof holds for the case of x^2 activations (x^k activations, in the k-sparse parity case) and L_{2,nu} norm, where nu is the homogeneity constant". This supports the intro sentence "The dense theorem is stated for the L_{2,nu} norm".

## Value check: the b=nu bound recovers Morwani's margin
- B_p = 2(2p nu)^{-nu/2}. At nu = 3 this gives 2B_p = 4(6p)^{-3/2}.
- 2B_p p/(p-1) = (4/6^{1.5}) / (p^{1/2}(p-1)).
- 4/6^{1.5} = 0.2721655 = sqrt(2/27). **Exact match** with Morwani Thm 7.
- Bonus: the power-neuron value n! B_p p/(p-1) equals 2 n!/((2n+2)^{(n+1)/2} (p-1) p^{(n-1)/2}). This is exactly the gamma* of **Li et al. Thm 4.1** (k = n operands). So the bound is attained at b = nu for every n, not only for n = 2, and the paper can say so.

## Checker's verdict: PARTIALLY_SUPPORTED. Upheld.
The overreach the checker found is real. paper.tex para 1 says "in networks where L_{2,nu} matches nu, all features would be used for modular addition or group composition". Morwani prove this only at nu = 3, with the conditions listed above for groups. The general-nu version for modular addition is Li et al., and there nu = k+1 is tied to the number of operands k.

## Corrections to the checker's recommended change
1. The bib key is `li2025nsum`, not `li2025fourier`.
2. Li et al. Thm 4.1 is about k-input modular addition with power neurons (sum u_t(a_t))^k w at L_{2,k+1}, requiring m >= 2^{2k-1}(p-1)/2. Write it as "k-operand modular addition with power neurons (nu = k+1)", not "general nu".
3. Theorem numbers 7/8/9 come from arXiv v2. I could not check them against the ICLR camera-ready. Either cite the arXiv version for theorem numbers or check the camera-ready PDF.

## Other issues in the live intro.tex (edited during this session)
- "Morwani introduced the principle of margin maximization" overclaims. Max-margin implicit bias predates them (Wei et al. 2019, whom they cite as Thm 1; Lyu & Li 2020). Their contribution is using max margin to **characterise features** analytically. Suggested wording: "used margin maximization to characterise which features emerge".
- "[Wei] showed that such maximum margin solutions are approached by networks when trained with vanishing regularization" overclaims. Wei Thm 4.1 is about the **global optimum** of the weakly regularized loss. It says nothing about the training trajectory, and approach.tex already says so correctly.
- Wei Thm 4.1 as stated is for **binary logistic** loss. The multi-class cross-entropy analogue is Thm C.1 (arXiv v4, Appendix C "Missing Material for Section 4"). approach.tex uses it for multi-class cross-entropy, so cite "Theorem 4.1 (multi-class: Theorem C.1)".
