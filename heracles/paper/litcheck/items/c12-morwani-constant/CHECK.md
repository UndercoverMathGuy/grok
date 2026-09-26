# C12-morwani-constant: check

**Verdict: SUPPORTED** (with one minor precision fix)

Claim: approach.tex cor:dense, appendix app:dense and app:models (power neuron) say that
2B_p p/(p-1) = sqrt(2/27)/(p^{1/2}(p-1)) is exactly Morwani et al.'s maximum L_{2,3} margin, and that their network attains the bound, so C_3 = 2B_p.

## Source
Fetched https://arxiv.org/html/2311.07568 and https://arxiv.org/abs/2311.07568.
- Title, authors and venue confirmed: "Feature emergence via margin maximization: case studies in algebraic tasks", Morwani, Edelman, Oncescu, Zhao, Kakade. The arXiv comment says "Spotlight at ICLR 2024".
- **Theorem 7** (cyclic groups): the setting is a prime p > 2, one-hidden-layer networks of width m >= 4(p-1), and quadratic activations. The neuron is phi({u,v,w},a,b) = (u_a + v_b)^2 w. The theorem states: "the maximum L_{2,3}-margin is gamma* = sqrt(2/27) * 1/(p^{1/2}(p-1))". At the maximum, each neuron is a single-frequency cosine, and every frequency zeta in {1,...,(p-1)/2} is used by at least one neuron.

## Checks
1. **Constant.** 2B_p p/(p-1) = 4(6p)^{-3/2} p/(p-1) = (2/(3 sqrt 6)) / (p^{1/2}(p-1)) = sqrt(2/27)/(p^{1/2}(p-1)). I checked this numerically at p = 3, 5 and 113, and the two sides agree to machine precision.
2. **Network.** Our power neuron at n=2, (u_1(a_1)+u_2(a_2))^2 w with nu = 3, is exactly their neuron. The norm b = nu = 3 is their L_{2,3}.
3. **Margin definition.** Our g, h and gamma*_b in approach.tex match their definitions (the logit gap, minimised over the dataset, maximised over the unit L_{2,3} ball).
4. **"Attains the bound, so C_3 = 2B_p".** This is logically fine. Their Theorem 7 gives the value of the global maximum, which is attained by their construction. Our bound is an upper bound valid at every width, and the two values are equal, so the bound is tight and C_3^{nu/b} = 2B_p at b = 3. One caveat: their theorem assumes m >= 4(p-1). The equality C_3 = 2B_p is therefore a statement about the supremum over widths, which is how C_b is defined. That is fine, but it is worth saying explicitly.
5. **Intro claim** ("every maximum L_{2,3}-margin network uses Fourier features, one frequency per neuron, and every frequency is used"). This matches Theorem 7.

## Fix
No correction is needed. Optionally, give the theorem number and the width condition, e.g. in cor:dense: "exactly their maximum $L_{2,3}$-margin \citep[Theorem~7]{morwani2024feature} (attained at width $m\ge4(p-1)$)". The bib entry is correct (ICLR 2024, arXiv 2311.07568).
