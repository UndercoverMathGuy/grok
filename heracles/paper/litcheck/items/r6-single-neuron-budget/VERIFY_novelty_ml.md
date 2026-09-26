# R6 single-neuron budget: independent verification (lens: novelty_ml)

**Verdict:** PARTIALLY_KNOWN still holds, but the checker did not get the split right.
- The budget is known at **both** ends of the b-range:
  - b = nu: Morwani (n = 2) and Li et al. (general n).
  - b = 2: Gunasekar et al. 2018, Lemma 7, stated for linear convolutional networks. The checker missed this.
- The middle range 2 < b < nu follows in one line from the b = 2 case.
- The ReLU bound, as written in the paper, is **false for n >= 2**. The paper's own argument gives C_2 <= 1/(2p), not p^{-n}/2. A numerical counterexample is below.

So I do **not** uphold the checker's result as it stands. Its two citations are correct, but it missed one close precedent, and its recommendation to keep the ReLU constant would leave an error in the paper.

## 1. The checker's citations, re-fetched and re-read

### Morwani et al.
- arXiv 2311.07568. The abs page says "Accepted as Spotlight at ICLR 2024" (v2, 19 Feb 2024). I downloaded the PDF and extracted the text.
- The computation is in Appendix F ("Proofs for cyclic groups (Theorem 7)"), subsection F.1, eqs. (4)-(9):
  - Eq. (6): the single-neuron class-weighted margin equals `2/((p-1)p^2) * sum_{j!=0} u^(j) v^(j) w^(-j)`.
  - Plancherel: `||u^||^2+||v^||^2+||w^||^2 <= p`.
  - Eq. (9): "By the inequality of quadratic and geometric means", `|u^||v^||w^| <= (z(j)/3)^{3/2}`.
  - Cauchy-Schwarz and `||z||_2 <= ||z||_1 <= p/2` then give `sqrt(2/27) * 1/(p^{1/2}(p-1))`.
  - "the only way to achieve ||z||_2 = p/2 is to place all the mass on a single frequency."
- **Confirmed.** This is the b = nu = 3, n = 2 case, for the quadratic (power) neuron.
- Lemma 18 (groups) uses Cauchy-Schwarz and AM-GM, with a `1/sqrt(d_R)` factor. Confirmed.
- **Scope statement.** Morwani restrict themselves to L_{2,nu}:
  - Sec. 3: "we will primarily focus on maximum margin with respect to L2,nu norm".
  - Sec. 7: "Our proof holds for the case of x^2 activations ... and L2,nu norm".
  - Lemma 5 is stated for any L_{a,b}. Only its "combining neurons" part needs b = nu.
- Recommended pinpoint: "App. F.1, eqs. (6)-(9)". This is arXiv v2 numbering.

### Li, Liang, Shi, Song, Zhou
- PMLR page proceedings.mlr.press/v258/li25b.html confirms the details: Proceedings of The 28th International Conference on Artificial Intelligence and Statistics, PMLR 258, pp. 523-531, 2025.
- The model is Eq. (2): `phi = (u_1^T x_1 + ... + u_k^T x_k)^k w`, with norm L_{2,k+1} only.
- **Lemma D.7.** The objective is `k!/((p-1)p^k) sum_{j!=0} w^(-j) prod_i u^_i(j)`. The proof says "all terms are zero rather than w(.) prod u_i(a_i)".
- **Lemma D.8** (App. D.5; formal version of Lemma 4.2). Eq. (20) is QM-GM. Then `max_{||z||_1<=p/2} sum z^{(k+1)/2} <= (p/2)^{(k+1)/2}` via `||z||_2 <= ||z||_1`. This gives the value `2(k!)/((2k+2)^{(k+1)/2}(p-1)p^{(k-1)/2})`, and the maximiser is a balanced single-frequency cosine.
- Algebra check: this equals `n! * B_p * p/(p-1)` with k = n. Confirmed.
- Their argument goes through `||z||_2 <= ||z||_1`, i.e. the superadditivity of t^{(k+1)/2}. That is only the b = nu case.
- **Confirmed.** li2025nsum is in refs.bib but is not cited in any section (grep: the only mention is the TODO at intro.tex:22). It must be cited.

## 2. Prior art the checker missed

### Gunasekar, Lee, Soudry, Srebro, "Implicit Bias of Gradient Descent on Linear Convolutional Networks", NeurIPS 31 (2018), arXiv 1806.00468
Sources: arXiv abs, the NeurIPS proceedings page and the bibtex file were all fetched. I also read the PDF.

**Lemma 7 (Sec. 5):** for a depth-L full-width linear convolutional network, `beta^ = diag(w^_1)...diag(w^_{L-1}) w^_L` and

`R_Pconv(beta) = min_{w: Pconv(w)=beta} ||w||_2^2 = L ||beta^||_{2/L}^{2/L}`.

Lemma 6 gives the same statement for diagonal networks. The proof (App. D, eq. 87) runs as follows:
- `sum_d |beta^[d]|^{2/L} = sum_d prod_l |w^_l[d]|^{2/L} <= (1/L) sum_d sum_l |w^_l[d]|^2`. This is AM-GM at each Fourier coordinate.
- `||w||^2 = ||w^||^2`, by orthonormality of the DFT.
- Equality holds when the moduli are balanced. **Any number of Fourier coordinates can be active at no extra cost.**

This is our C_2. A product neuron's beta_j is `2p^{-nu} prod_t u^_t(j) conj(w^(j))`, a product of nu Fourier transforms at the same frequency. That is exactly the "depth-nu single-channel convolution" structure. So at b = 2:
- the per-frequency AM-GM pricing,
- the exponent 2/nu,
- the observation in app:models that at b = 2 "a neuron may carry several frequencies at no extra cost"

are Gunasekar Lemma 7, up to the real-signal Parseval factor p/2 and the normalisation 2p^{-nu}.

The sentence in approach.tex l.42 ("at fixed product their total L_{2,b} cost is least when they are equal (AM-GM), so amplitude A costs a multiple of A^{b/nu}") is the same idea as Gunasekar's l_{2/L} bridge penalty, applied in the frequency domain.

**Middle range 2 < b < nu.** Since `||x||_{b/2} <= ||x||_1` for b >= 2, C_b = C_2^{b/2}. So `C_b^{nu/b} = C_2^{nu/2} = B_p` for every b in [2, nu]. The claim that the value does not depend on b follows in one line from the b = 2 case.

**What is still ours in R6:**
- carrying this over to the modular / group-composition margin, where the neuron has interference modes and beta_j is the diagonal mode;
- using the per-neuron l_{b/nu} budget as the input to Gamma_s;
- the power-neuron n! (only jointly with Li).

**Suggested sentence for approach.tex l.54-58:** "At b = 2 this is the l_{2/nu} representation cost of a product of Fourier transforms (Gunasekar et al., 2018, Lemma 7); at b = nu it is the single-neuron computation of Morwani et al. (2024, App. F.1) and Li et al. (2025, Lemmas D.7-D.8)."

### Kunin et al., "Alternating Gradient Flows", NeurIPS 2025 (per the arXiv comment), arXiv 2506.06489
I fetched the abs page and read the PDF.

**Theorem 5.1:** for the quadratic neuron on modular addition, the unit vectors maximising the utility `U = (2/p^3) sum_k |x^[k]|^2 u^[k] v^[k] w^[k] x^[k]` are single-frequency cosines with amplitude `sqrt(2/(3p))`, and the maximal value is `2/(27 p^3) |x^[xi]|^3`.

This is the same single-neuron sphere problem (n = 2, weighted), set in a dynamics paper. It is a third instance of the b = nu computation. Citing it is optional.

### Adjacent, not precedent for R6
Jagadeesan, Razenshteyn, Gunasekar, COLT 2022, arXiv 2102.12238 (abs fetched). For multi-channel linear convolutional networks, the induced regulariser is:
- the nuclear norm of the Fourier coefficients when K = 1;
- the l_{2,1} group norm of the Fourier coefficients when K = D.

This is relevant to app:models "Where the approach does not apply" (the nuclear-norm pricing of a shared layer) and to item u8, not to R6.

## 3. Side finding: the ReLU bound is wrong as stated (appendix.tex l.283)
The paper states `sum_{j in F_p} |h^(j..j) w^(j)| <= (p/4) ||omega||^2` and hence `C_2 <= p^{-n}/2`.

**Correct derivation.**
- Parseval on Z_p^n gives `sum_{k!=0} |h^(k)|^2 = p^{2n} Var(h)`, so `sum_{j in F_p} |h^(j..j)|^2 <= p^{2n} Var(h) / 2`.
- `Var(h) <= sum_t Var(u_t) <= ||u||^2 / p`. So `sum |h^(j..j)|^2 <= p^{2n-1} ||u||^2 / 2`.
- Similarly `sum_{F_p} |w^(j)|^2 <= p ||w||^2 / 2`.
- Cauchy-Schwarz then gives `sum |h^ w^| <= (p^n/2) ||u|| ||w|| <= (p^n/4) ||omega||^2`.
- Therefore **`C_2 <= 2p^{-(n+1)} * p^n/4 = 1/(2p)`**, not p^{-n}/2. The two agree only at n = 1.

**Numerical counterexample** (scripts `verify_scripts/relu.py` and `verify_scripts/relu2.py`):
- Setting: p = 7, n = 2, with beta_j computed two independent ways (the paper's FFT formula, and a direct time-domain eta(delta) plus its DFT). The two agree to 1e-16.
- Random hill-climb value: `sum_j |beta_j| / ||omega||^2 = 0.0154`.
- The paper's bound is p^{-2}/2 = 0.0102, so it is violated by a factor of 1.51. The corrected bound is 1/(2p) = 0.0714, which is satisfied.

**Consequence.** Only the constant changes. The b = nu endpoint statement for ReLU and the flat maximiser of Cor. dense do not depend on the value of C_2. The fix is in app:models: replace `p/4` with `p^n/4`, and `p^{-n}/2` with `1/(2p)`. The item statement should change the same way.

**Novelty of the (corrected) ReLU bound.** The ML searches found nothing:
- Morwani, Li, Mohamadi et al., Huang & Li (2511.23443) and He et al. (2602.16849) have no single-neuron ReLU upper bound.
- Huang & Li treat ReLU only through expressivity and constructive margins.

## 4. Power neuron: "<=" can be "=" (numerically verified)
Script `verify_scripts/chk.py`:
- For random weights at (p,n) = (7,2) and (5,3), beta_j(power) / beta_j(product) equals 2 = 2! and 6 = 3! exactly, at every j.
- A unit-norm balanced single-frequency cosine neuron gives `(sum_j|beta_j|^{b/nu})^{nu/b} = n! * B_p` to 1e-11:
  - b in {2, 2.5, 3} at n = 2;
  - b in {2, 3, 4} at n = 3.

So `C_b^{nu/b} = n! B_p` for the power neuron at every b in [2, nu]. Li Lemma D.8 is the b = nu case. The checker's suggestion 3 is correct.

## 5. Search log
WebSearch was unavailable: the session budget was exhausted. The Semantic Scholar API and the dblp API returned 429 or timed out. Instead I used the arXiv web search (arxiv.org/search, abstract field, via curl) and direct fetches of arXiv, PMLR and NeurIPS pages.

**arXiv queries:**
- "modular addition" margin
- "modular arithmetic" margin
- "group composition" margin
- "modular addition" "weight decay"
- grokking margin
- "Fourier features" "max margin"
- modular addition multilinear
- "linear convolutional" "bridge penalty"
- several others that returned empty

**Read in full or scanned** (none has an l_2 single-neuron Fourier budget for b < nu, or any ReLU budget):
- Mohamadi et al. ICML 2024 (2407.12332): l_inf margins, generalisation.
- Tian, NeurIPS 2025 (2410.01779): CoGS algebra, no norm budget.
- Doshi et al. 2024 (2406.03495): x^S constructions only.
- Huang & Li 2025 (2511.23443).
- He et al. 2026 (2602.16849).
- Pracher et al. 2026 (2609.26679): NTK / weight-decay dynamics.
- Tan et al. 2026 (2606.05863): Schatten penalty for deep linear networks.
- Lyu et al. 2023 (2311.18817): listed in results only, abstract-level.

**Not confirmed:** Dai, Karzand, Srebro, "Representation costs of linear neural networks" (NeurIPS 2021). I could not fetch a page for it, so there is no bib entry.
