# R6 single-neuron budget: independent verification (novelty_math lens)

Verifier's corrected verdict: the product and power budgets are **KNOWN**, and the ReLU bound as stated is **FALSE for n >= 2**.
- **Product neuron.** The budget is known at every b. At b=2 it is the Hadamard-product / l_{2/L} representation-cost lemma (Hoff 2017; Gunasekar et al. 2018, Lemma 7, in Fourier form). At b=nu it is Morwani (n=2) and Li (all n). The range 2<b<nu follows in one line from monotonicity of l_q quasi-norms.
- **Power neuron.** The budget is Li Lemma D.7 plus the product budget. It holds with equality.
- **ReLU neuron.** The stated bound C_2 <= p^{-n}/2 is wrong for n >= 2 by a factor p^{n-1}. The same argument, done correctly, gives C_2 <= 1/(2p) for every n.

The checker's verdict (PARTIALLY_KNOWN) is **not upheld** as written, for two reasons:
- (a) It calls the b=2 multi-frequency equality "a small new observation". That observation is Gunasekar Lemma 7.
- (b) It recommends keeping the ReLU bound as ours. That bound is incorrect.

Both of the checker's confirmed citations survive my independent check.

## 1. Checker's citations: re-fetched and re-read

### morwani2024feature: CONFIRMED
- **Metadata.** https://arxiv.org/abs/2311.07568 gives title "Feature emergence via margin maximization: case studies in algebraic tasks" by Morwani, Edelman, Oncescu, Zhao, Kakade. The Comments field says "Accepted as Spotlight at ICLR 2024". The refs.bib entry is correct.
- **Source read.** I downloaded the PDF (v2) and read Appendix F.1, "Proofs for cyclic groups (Theorem 7)", pp. 25-26.
  - Eq. (6): the expected class-weighted margin is `2/((p-1)p^2) sum_{j!=0} u^(j) v^(j) w^(-j)`.
  - "By Plancherel's theorem, the norm constraint is equivalent to ||u^||^2+||v^||^2+||w^||^2 <= p".
  - Eq. (9): "By the inequality of quadratic and geometric means", `|u^||v^||w^| <= ((|u^|^2+|v^|^2+|w^|^2)/3)^{3/2}`.
  - Cauchy-Schwarz with ||z||_2 <= ||z||_1 <= p/2 then gives `sqrt(2/27)/(p^{1/2}(p-1))`.
  - "The only way to achieve ||z||_2 = p/2 is to place all the mass on a single frequency".
- **Groups.** Lemma 18 (p. 36) is the group analogue: two Cauchy-Schwarz steps, then AM-GM on Frobenius norms.
- **Match to our result.** This is exactly our b=nu=3, n=2 case, with the same equality case.

### li2025nsum: CONFIRMED
- **Metadata.** https://proceedings.mlr.press/v258/li25b.html gives title "Fourier Circuits in Neural Networks and Transformers: A Case Study of Modular Arithmetic with Multiple Inputs" by Li, Liang, Shi, Song, Zhou. It appeared at AISTATS 2025 (the 28th), PMLR vol. 258, pp. 523-531.
- **Source read.** I downloaded the arXiv PDF (2402.09469v4).
- **Lemma D.7** (p. 32). For `eta = E[(u_1(a_1)+...+u_k(a_k))^k w(...)]`, the paper says "We note that all terms are zero rather than w(.) prod u_i(a_i)", and gets the objective `k!/((p-1)p^k) sum_{j!=0} w^(-j) prod u^_i(j)`, which is eq. (15). This is our "only the multilinear monomial survives, coefficient n!".
- **Lemma D.8** (formal Lemma 4.2, pp. 33-34). Eq. (20) is QM-GM; after it, the paper takes `max_{||z||_1<=p/2} sum z(j)^{(k+1)/2} <= (p/2)^{(k+1)/2}`. The maximum is `2(k!)/((2k+2)^{(k+1)/2}(p-1)p^{(k-1)/2})`, attained only at a balanced single frequency. With n=k and nu=k+1, this is n! B_p * p/(p-1).
- **Bib entry.** The checker's bibtex fields are right. The current refs.bib entry lacks series, volume and pages; add them.
- **Citation status.** It is still cited nowhere in the sections. The only mention is the TODO comment at intro.tex:22.

## 2. Prior art the checker missed (math lens)

### Gunasekar, Lee, Soudry, Srebro, NeurIPS 2018: the b=2 product budget, including the "many frequencies at no extra cost" equality
- **Metadata.** https://arxiv.org/abs/1806.00468 gives title "Implicit Bias of Gradient Descent on Linear Convolutional Networks". The PDF footer reads "32nd Conference on Neural Information Processing Systems (NeurIPS 2018), Montréal, Canada."
- **Lemma 3.** For full-width circular convolutions, "beta^ = diag(w^_1) ... diag(w^_{L-1}) w^_L": the predictor's DFT is the Hadamard product of the layers' DFTs.
- **Lemma 7** (Sec. 5). "For a depth-L convolutional network ... R_Pconv(beta) = min_{w:Pconv(w)=beta} ||w||_2^2 = L ||beta^||_{2/L}^{2/L}".
- **Lemma 6 proof** (App. D, eq. (85)). The proof is per-coordinate AM-GM, "(a) ... follows as arithmetic mean is greater than the geometric mean". The upper bound is achieved by balancing.
- **Why this is the same statement.** Our product neuron's class-averaged coefficient is `beta_j = 2p^{-nu} prod_t u^_t(j) conj(w^(j))`. This is the same Hadamard product of nu DFTs. Parseval turns the ||.||_2^2 cost into Fourier mass. Our C_2 statement, `sum_{j in F}|beta_j|^{2/nu} <= C_2 ||omega||^2` with equality for balanced neurons carrying any number of frequencies, is Gunasekar Lemma 7 with L = nu, up to normalisation and the pairing of j with -j.
- **Consequence.** The appendix remark "at b=2 the superadditivity step is an identity, so a neuron may carry several frequencies at no extra cost" is the separability of the l_{2/L}^{2/L} representation cost. It is not new.
- **Venue.** Confirmed from the PDF footer. No page numbers were fetched, so the bib entry below omits pages.

### Hoff, CSDA 2017: the origin of the AM-GM / Hadamard-product -> l_{2/K} fact
- **Metadata.** Crossref for DOI 10.1016/j.csda.2017.06.007 gives "Lasso, fractional norm and structured sparse estimation using a Hadamard product parametrization", Peter D. Hoff, Computational Statistics & Data Analysis 115 (2017) 186-198.
- **Lemma 2** (arXiv 1611.00040, Sec. 3.1). With q = 2/K, the penalty lambda||beta||_q^q equals the minimum over factorisations u_1 o ... o u_K = beta of `lambda(u_1^T u_1+...+u_K^T u_K)/K`. The proof: "the minimum value of sum_k u_k^T u_k subject to prod_k u_{k,j} = beta_j is attained when |u_{k,j}| is constant across k".
- **Relation to our proof.** This is our per-frequency AM-GM step, in coordinates rather than Fourier modes.

### The 2 < b < nu range is a one-line corollary
- Let x_j = |beta_j|^{2/nu}. For b >= 2, sum_j x_j^{b/2} <= (sum_j x_j)^{b/2}. This is our superadditivity step, and equivalently the monotonicity l_1 ⊂ l_{b/2}.
- It gives C_b <= C_2^{b/2}, with equality on 1-sparse beta.
- The b=2 optimum (Gunasekar/Hoff) is attained by a single-frequency balanced cosine. So C_b^{nu/b} = C_2^{nu/2} = B_p for all b.
- "B_p does not depend on b" is therefore not an independent result. It is the statement that the 1-sparse optimum is simultaneously optimal for every l_q quasi-norm with q <= 1.

### Other items checked (no overlap)
- **Mohamadi, Li, Wu, Sutherland, ICML 2024** (arXiv 2407.12332, PDF grepped). Their margin analysis is l_inf-normalised, for quadratic networks only. They give no per-neuron l_2 budget and no ReLU margin bound. The only occurrence of ReLU is in a reference title.
- **He, Wang, Chen, Yang 2026** (arXiv 2602.16849, PDF grepped). Their "single-neuron contribution" is a dynamics / majority-voting notion, not a norm budget.
- **HF papers search.** Queries: "max margin modular addition ReLU network Fourier features", "representation cost weight decay product parametrization l2/L quasi-norm Fourier", "modular addition multiple inputs power activation margin k! Fourier". None surfaced a ReLU single-neuron margin budget.
- **Search limits.** The WebSearch budget for this session was exhausted before these searches. Semantic Scholar and the arXiv API were rate-limited. Coverage of 2025-26 ReLU-margin work is therefore thinner than I would like.

## 3. The ReLU bound is wrong for n >= 2 (verified numerically and analytically)

**Claim** (appendix.tex, ReLU paragraph, l. 283): `sum_{j in F}|h^(j..j) w^(j)| <= (p/4)||omega||^2`, hence C_2 <= p^{-n}/2.

**Error.** Parseval on Z_p^n with the unnormalised transform (prelim.tex) gives:
- `sum_{k != 0}|h^(k)|^2 = p^n sum_x (h - mean h)^2 = p^{2n} Var(h)`.
- Var(h) <= Var(sum_t u_t(a_t)) <= sum_t ||u_t||^2 / p. This is the 1-Lipschitz variance inequality, Var f(X) = E(f(X)-f(X'))^2/2 <= Var X.
- The diagonal pairs (j..j) and (-j..-j) are distinct non-zero multi-indices, so `sum_{j in F}|h^(j..j)|^2 <= p^{2n-1} U / 2`, where U = sum_t ||u_t||^2.
- Also sum_{j in F}|w^(j)|^2 <= p W/2, where W = ||w||^2.
- Cauchy-Schwarz then gives `sum_{j in F}|h^(j..j) w^(j)| <= (p^n/2) sqrt(UW) <= (p^n/4)||omega||^2`.

So the constant is p^n/4, not p/4. The correct conclusion is **C_2 <= 2p^{-(n+1)} * p^n/4 = 1/(2p)**. This equals B_p at nu=2 and does not depend on n. The two agree only at n=1.

**Numerical counterexample** (`verify_relu_c2.py` in this directory). The script computes sum_{j in F}|beta_j| / ||omega||^2 for sigma = ReLU, using beta_j = 2p^{-(n+1)} h^(j..j) conj(w^(j)).

| p | n | cosine neuron | best of 200 random | claimed p^{-n}/2 | corrected 1/(2p) |
|---|---|---|---|---|---|
| 7 | 2 | 1.262e-2 | 9.88e-3 | 1.020e-2 (VIOLATED) | 7.14e-2 |
| 11 | 2 | 8.129e-3 | 3.71e-3 | 4.132e-3 (VIOLATED) | 4.55e-2 |
| 13 | 2 | 6.893e-3 | 2.70e-3 | 2.959e-3 (VIOLATED) | 3.85e-2 |
| 11 | 3 | 8.4e-6 | 4.189e-4 | 3.757e-4 (VIOLATED) | 4.55e-2 |
| 13 | 3 | 4.2e-6 | 2.956e-4 | 2.276e-4 (VIOLATED) | 3.85e-2 |

- At n=2 the cosine neuron scales as p * C ≈ 0.088-0.090, i.e. Theta(1/p), not Theta(p^{-2}).
- At n=1 every value is below p^{-1}/2 = 1/(2p), consistent with the claim there.

**Downstream impact.** The only use of the bound is the appendix sentence that Theorem universal and the flat maximiser of Cor. dense apply to ReLU. That conclusion survives, because it only needs a finite C_2. Any stated ReLU margin value must use 1/(2p): the correct bound is gamma*_2 <= 1/(2(p-1)). No other section uses p^{-n}/2 (grepped).

**Novelty.** I found no precedent for a ReLU single-neuron margin budget on modular addition. Morwani and Li treat ReLU only empirically, Mohamadi works with l_inf, and He et al. study dynamics. The ingredients are textbook: Cauchy-Schwarz, Parseval, and the Lipschitz variance contraction.

## 4. Power neuron: the "<=" should be "="

- `verify_power_neuron.py` (p=11, n=2,3, random neurons) finds max |beta_pow - n! beta_prod| ≈ 3e-16. The identity beta_pow = n! beta_prod holds for every omega, not only at the optimum.
- The cosine witness attains C_b^{nu/b} = n! B_p to 7 digits at b in {2, 2.5, nu} for both n.
- Since the norm is the same, C_b(power) = (n!)^{b/nu} C_b(product) exactly. The power-neuron budget is therefore C_b^{nu/b} = n! B_p. Li Lemma D.8 is the b=nu case.
- The paper's "C_3 = 2B_p at n=2, b=3" is a special case.

## 5. Recommended changes (the author applies them; I did not edit any paper file)

1. **approach.tex l.54-58.** After "is a single-neuron optimization", write roughly: "Per frequency this is the AM-GM step behind the $\ell_{2/L}$ representation cost of Hadamard-product parametrizations \citep{hoff2017lasso}, in the Fourier form of \citet[Lemma~7]{gunasekar2018implicit}. At $b=\nu$ the resulting budget is \citet[App.~F.1]{morwani2024feature} for $n=2$ and \citet[Lemmas~D.7--D.8]{li2025nsum} for general $n$. Intermediate $b$ follows by monotonicity of $\ell_q$ quasi-norms, with the same single-frequency optimum."
   - Present the product/power budget as a lemma with credit, not as a contribution.
2. **appendix.tex l.281 (power neuron).** Replace "$C_b^{\nu/b}\le n!\,B_p$" with "$=$", because $\beta_j$ is exactly $n!$ times the product-neuron coefficient and the cosine neuron attains it. Add "(cf. \citet[Lemma~D.7]{li2025nsum})". Also change approach.tex l.58 accordingly.
3. **appendix.tex l.283 (ReLU).**
   - Change "$\le\frac p4\|\omega\|_2^2$" to "$\le\frac{p^n}4\|\omega\|_2^2$".
   - Change "$C_2\le\frac{p^{-n}}2$" to "$C_2\le\frac1{2p}$". This is $B_p$ at $\nu=2$ and is independent of $n$.
   - Update the item statement to match. The qualitative conclusion (dense flat bound) is unchanged.
4. **Citing li2025nsum.** It must be cited somewhere in the body. It is currently cited nowhere.
5. **appendix.tex l.275 (b=2 remark).** Soften "a neuron may carry several frequencies at no extra cost" by noting that it is the separability of the $\ell_{2/\nu}$ cost (Gunasekar Lemma 7).

## Sources fetched this session
- https://arxiv.org/abs/2311.07568 and https://arxiv.org/pdf/2311.07568 (full text; App. F.1 pp. 25-26, Lemma 18 p. 36)
- https://proceedings.mlr.press/v258/li25b.html, https://arxiv.org/pdf/2402.09469 (Lemmas D.7-D.8, pp. 32-34)
- https://arxiv.org/abs/1806.00468 and https://arxiv.org/pdf/1806.00468 (Lemmas 3, 6, 7; App. D eq. (85))
- https://arxiv.org/abs/1611.00040, https://arxiv.org/pdf/1611.00040 (Lemma 2, Sec. 3.1), and Crossref / doi.org bibtex for 10.1016/j.csda.2017.06.007
- https://arxiv.org/abs/2407.12332 and https://arxiv.org/pdf/2407.12332 (grepped for ReLU and margin)
- https://arxiv.org/pdf/2602.16849 (grepped)
- HF papers search (hf://papers), 4 queries
