# T2-amgm-quasinorm: verify (novelty_math lens, task 44)

Claim: AM-GM over the nu weight factors makes amplitude A cost A^{b/nu}. Weight decay on degree-nu products therefore acts as an l_{b/nu} quasi-norm on amplitudes.
- The live text is now approach.tex:52-54. The checker's location, approach.tex:45, is now a `%Original:` comment at line 67.
- The appendix version is appendix.tex app:models, "Product neuron".

Checker verdict: KNOWN. **Upheld.** This run re-fetched every source itself; downloads are in scratchpad/t44v.

## The five confirmed citations
- **Tibshirani 2021** (stat.cmu.edu/~ryantibs/papers/sparsitynn.pdf, dated April 15 2021, working notes).
  - The abstract says a k-layer net with weight decay is equivalent to l_p-penalised regression with p = 2/k < 1.
  - Sec 2.1 names AM-GM.
  - Eq. 8-9 give min_{prod a_l=c} sum a_l^2 = k|c|^{2/k}.
  - Verdict: OK.
- **Hoff 2017** (arXiv 1611.00040). The abstract says "a subclass of L_q penalties with q<=1 can be expressed as sums of L_2 penalties". Crossref for DOI 10.1016/j.csda.2017.06.007 gives CSDA vol. 115, pp. 186-198, dated 2017-11. The bib entry is correct.
- **Gunasekar, Lee, Soudry, Srebro, NeurIPS 2018**. The proceedings page gives vol. 31. The arXiv v2 PDF confirms Thm 2a.
  - The better anchor is **Lemma 6** (diagonal network) and **Lemma 7** (linear convnet). Lemma 7 states R(beta) = min_{P(w)=beta} ||w||_2^2 = L ||beta-hat||_{2/L}^{2/L}.
  - This is an explicit l2 representation cost on Fourier coefficients. It is not only an implicit-bias statement.
  - The proof, App. D eq. 83, says "arithmetic mean is greater than the geometric mean".
  - Cite as `\citet[Lemmas~6--7]{gunasekar2018implicit}`. These are arXiv numbers; check them against the proceedings PDF.
- **Dai, Karzand, Srebro, NeurIPS 2021**. The proceedings page gives vol. 34, pp. 26884-26896. The abstract covers "regularization cost in predictor space induced by l2 regularization on the parameters ... l_p norms". OK.
- **Kolb, Weber, Bischl, Rugamer, ICLR 2025** (arXiv 2502.02496; PDF header "Published as a conference paper at ICLR 2025").
  - Thm 1: the L_{2/D}-penalised problem and the L2-on-D-Hadamard-factors problem "have the same global and local minima".
  - Related work traces the idea to Grandvalet 1998, Hoff 2017, Neyshabur et al. 2015 and Tibshirani 2021. OK.

## What the checker missed

### 1. In-domain precedent: Morwani et al. 2024, already in refs.bib as `morwani2024feature`
Source: arXiv 2311.07568v2, App. F.1, proof of Theorem 7 (cyclic groups), eqs. (7)-(9).

For the single-neuron problem of the quadratic network, their proof runs:
- Plancherel turns the l2 budget into a Fourier budget.
- Then comes a per-frequency inequality: "By the inequality of quadratic and geometric means, |u^(j)||v^(j)||w^(j)| <= ((|u^|^2+|v^|^2+|w^|^2)/3)^{3/2}".
- Then Cauchy-Schwarz bounds the sum over frequencies.

Our app:models "Product neuron" budget runs:
- per-frequency AM-GM, (m_j/nu)^{nu/2};
- then superadditivity of t^{b/2};
- then Parseval.

This is the same argument, generalised from (nu,b) = (3,3) to general (nu,b). Their group-case appendix makes the same AM-GM step on the irrep Frobenius norms.

Morwani is already cited in the paper, but not for this step. It should be credited at app:models, and at approach.tex:52-54 alongside the representation-cost citations.

### 2. Optimisation-literature folklore: Poon & Peyre
Poon & Peyre, "Smooth over-parameterized solvers for non-smooth structured optimization", Mathematical Programming 201:897-952 (2023), DOI 10.1007/s10107-022-01923-3 (Crossref); arXiv 2205.01385.
- Sec 4.1: "It is well known that quadratic variational forms exist for many nonsmooth regularisers, including nuclear norm, l_q".
- Lemma 4 gives a mixed-exponent Hadamard form, (1/q)||x||_q = min_{x=u.v} (1/2)||u||^2 + (1/2beta) sum |v|^{2beta}. This is a Young-inequality generalisation, which is more general than our equal-exponent step.
- Prop. 10 gives the three-factor form that reaches q < 2/3.

This confirms the step is textbook-level variational-form folklore in optimisation, not only in the ML implicit-bias literature. It is an optional extra citation.

### 3. Neyshabur, Tomioka, Srebro, ICLR 2015 workshop (arXiv 1412.6614)
- Thm 1: weight decay on a two-layer ReLU net equals l1 on the top layer with unit-norm hidden units.
- The proof is AM-GM plus positive-homogeneous rescaling.
- This is the per-neuron version, at b = nu = 2. The novelty_ml verifier also flagged it.

### Matrix analogue (only for the appendix's nuclear-norm remark)
Srebro, Rennie, Jaakkola, "Maximum-Margin Matrix Factorization", NeurIPS 17 (proceedings page fetched): low-norm factorisation in a max-margin problem. It is not needed for T2.

## The L_{2,b} "generalisation" is not new
Per neuron, the paper's cost is ||omega||_2^b, i.e. (sum_t ||u_t||^2)^{b/2}. So the L_{2,b} price is exactly the b=2 identity min sum_t x_t^2 = nu |c|^{2/nu}, raised to the power b/2.
- That gives nu^{b/2} |c|^{b/nu}.
- It is a one-line corollary and must not be presented as a contribution.

The only genuinely new ingredients are:
- the use of this price as the budget of the worst-offset coverage programme (eq:programme);
- the resulting K* and log p law.

## Problems with the current live text (approach.tex:52-54)
- "the minimum penalty (sum of the weights)". The penalty is the sum of squared factor norms, raised to b/2 under L_{2,b}. It is not a sum of weights.
- "when they are exactly equal". AM-GM equality holds when the factor norms (per-frequency magnitudes) are equal, not when the weights are equal.
- "The least penalty to achieve amplitude A is A^{b/v}".
  - Typo: `v` should be `\nu`.
  - The constant is dropped. It should say "a constant multiple of A^{b/\nu}", as the commented-out original did.
- The checker's recommended citation sentence is anchored to approach.tex:45, which is now a comment. It must go into the live paragraph at lines 52-54.

## Verdict
KNOWN, upheld.
- Keep the checker's paragraph, placed at the live lines.
- Change the Gunasekar pointer to Lemmas 6-7.
- Add Morwani App. F.1 as the in-domain source of the per-frequency AM-GM plus Parseval single-neuron argument that app:models generalises.
- Optionally add Poon-Peyre 2023 as the optimisation-folklore citation.

## Search limits
WebSearch was unavailable: the session budget of 200/200 was exhausted. All sources were fetched directly with curl, arXiv, NeurIPS proceedings or the Crossref API. The StackExchange API queries on MathOverflow and math.SE returned nothing relevant.
