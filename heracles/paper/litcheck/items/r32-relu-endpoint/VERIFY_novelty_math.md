# VERIFY (novelty_math): R32-relu-endpoint, task 36

**Result:** the checker's verdict (PARTIALLY_KNOWN) and its three confirmed citations hold up. However, **the ReLU budget constant in appendix.tex:283 is wrong for n >= 2.** The checker recommended claiming this constant as the new part, so the recommendation must change.

## 1. The constant C_2 <= p^{-n}/2 is false for n >= 2

The conventions come from prelim.tex:7. The DFT is unnormalised (hat v(j) = sum_k v(k) rho^{-jk}), F = {1..(p-1)/2}, and ||omega||^2 = sum_t ||u_t||^2 + ||w||^2.

**Claim tested:** sum_{j in F} |hat h(j..j) hat w(j)| <= (p/4) ||omega||^2.

**Numerical test** (script: `verify_novelty_math_relu_c2.py` in this folder). For each u, the script uses the w that is optimal under Cauchy-Schwarz and computes the largest value of the ratio (LHS / ||omega||^2) over 200 trials:

| p  | n | max found | paper p/4 | corrected p^n/4 |
|----|---|-----------|-----------|-----------------|
| 5  | 1 | 1.01      | 1.25      | 1.25            |
| 11 | 2 | **5.86**  | 2.75      | 30.25           |
| 23 | 2 | **25.85** | 5.75      | 132.25          |
| 23 | 3 | **31.28** | 5.75      | 3041.75         |

The paper's bound is violated for every n >= 2. The argument's own steps give the following instead:

- hat h(j,...,j) = hat G(j), where G(s) = sum over {x : a_1+...+a_n = s} of h(x) = p^{n-1} E[h | S = s].
- Parseval on Z_p gives sum_{j != 0} |hat h(j..j)|^2 = p^{2n} Var(E[h|S]) <= p^{2n} Var(h).
- Var(h) <= Var(S) = sum_t Var(u_t(a_t)) <= p^{-1} sum_t ||u_t||^2.
- Restricting to F halves both sums. Cauchy-Schwarz and AM-GM then give sum_F |hat h hat w| <= (p^n/2) ||u|| ||w|| <= (p^n/4) ||omega||^2.
- Hence **C_2 <= 2 p^{-(n+1)} * p^n/4 = 1/(2p)**, which does not depend on n.

The paper's p^{-n}/2 matches this only at n = 1. The error loses the factor p^{n-1} that comes from h living on Z_p^n.

**Impact:** qualitative only. Corollary dense (the flat maximizer of Gamma_1) does not depend on C_2, so the "b = nu endpoint" sentence survives. The numerical ReLU bound C_2 Gamma_1(p) must be corrected to p/(2p(p-1)) = 1/(2(p-1)).

**Fix for appendix.tex:283:** replace "\le\frac p4\|\omega\|_2^2" with "\le\frac{p^n}4\|\omega\|_2^2", and "C_2\le2p^{-(n+1)}\cdot\frac p4=\frac{p^{-n}}2" with "C_2\le2p^{-(n+1)}\cdot\frac{p^n}4=\frac1{2p}". It would help to spell out the diagonal-Parseval step (conditional expectation on the sum), because that is where the factor was lost.

## 2. Novelty from the mathematics side

- **Var(sigma(X)) <= Var(X) for 1-Lipschitz sigma.** This is folklore: Var(Y) = (1/2) E(Y - Y')^2 with Y' an independent copy. No citation needed.
- **The diagonal-coefficient identity.** hat h(j..j) is the DFT of the push-forward of h to the sum S. This is standard Fourier analysis on Z_p^n.
- **The flat maximizer at r = b/nu = 1.** This is the equalizer solution of a linear max-min programme. It is standard, and the paper proves it itself (Cor dense).
- **"ReLU + L2 = the b = nu case".** This follows directly from the definitions in Morwani et al.: ReLU is 2-homogeneous in (u, w), so L2 = L_{2,nu}. It is not a new idea.

No published statement of the ReLU per-neuron Fourier budget was found. The session's WebSearch budget was exhausted (200/200), so this search relied on fetched primary sources only.

## 3. Citations the checker confirmed (fetched this session)

- **morwani2024feature (arXiv 2311.07568v2):** confirmed.
  - Fig. 4(b): "nearly 1-sparse in frequency space but not quite".
  - App. C, Fig. 7(a): ReLU with L2 regularisation, "the final distribution across frequencies is close to uniform".
  - Conclusion: "Our proof holds for the case of x2 activations ... and L2,nu norm ... Empirical findings suggest that the results may be transferable to other architectures and norms."
- **gromov2023grokking (arXiv 2301.02679):** confirmed.
  - Sec. 3: "The weights (6)-(7) also work for other activation functions, including ReLU, however the 100% accuracy is achieved at higher width" (App. B).
  - App. B: "We do not have an analytic proof of this fact".
- **he2026mechanism (arXiv 2602.16849, He/Wang/Chen/Yang):** confirmed.
  - Obs. 3: "every frequency k is represented among the neurons".
  - Def. 4.1: full diversification.
  - Experiments use ReLU; the theory uses x^2.
  - This paper is **not in refs.bib** and needs to be added.

## 4. Additional ML-side checks (no ReLU theorem found)

- **li2025nsum (arXiv 2402.09469), fetched:** the theory uses the polynomial activation (sum u^T x)^k only. Thm 4.1 proves that all frequencies are used at L_{2,k+1}. There is no ReLU claim.
- **Mohamadi et al. (ICML 2024, arXiv 2407.12332), fetched:** quadratic activation with an l_inf norm. No ReLU margin result.
