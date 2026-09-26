# R5-coverage: independent verification, lens novelty_ml

Item: approach.tex `prop:coverage`: `h(theta) <= Gamma_s(p) ||A||_{1/s}`, with `||A||_{1/s} = (sum_j A_j^{1/s})^s`.
Checker verdict: NOVEL_AS_FAR_AS_FOUND. Checker cited kam2026symmetry and morwani2024feature.

## Verdict
**Partly upheld.** No ML paper I found proves this bound for arbitrary one-hidden-layer networks at general s = nu/b. Two points qualify that:

1. **The checker missed a precedent for the normalisation template.** Gunasekar, Lee, Soudry and Srebro (NeurIPS 2018) already use the same homogeneity reduction for linear convolutional nets. An l2 weight cost becomes an l_{2/L} quasi-norm on the DFT coefficients, and the max-margin problem is posed over that quasi-norm ball. Prop:coverage is the analogue of their "margin <= (max margin on the unit quasi-norm ball) x quasi-norm" rescaling, with b = 2 and nu = L. The R4 checker found this paper; the R5 checker did not cross-reference it.
2. **The checker's description of Kam et al. is imprecise, and the suggested sentence overstates them** (details below).

So the R5 verdict should match R4's: the template is known and this instance is new. prop:coverage is a one-line normalisation, and no novelty claim should rest on it.

## Citations checked (all fetched this session)

### morwani2024feature: CONFIRMED
- arXiv abs page https://arxiv.org/abs/2311.07568:
  - Title: "Feature emergence via margin maximization: case studies in algebraic tasks".
  - Authors: Morwani, Edelman, Oncescu, Zhao, Kakade.
  - Versions: v1 13 Nov 2023, v2 19 Feb 2024. The comments field says "Accepted as Spotlight at ICLR 2024".
- ICLR page https://iclr.cc/virtual/2024/poster/18073 lists it as a "Spotlight Poster".
- The bib in refs.bib is correct.
- Content, from the arXiv v2 PDF text:
  - Section 4 is "Cyclic groups". The phase condition "theta_u* + theta_v* = theta_w*" is Theorem 7(1).
  - `eta_{u,v,w}(delta) := E_{a,b}[(u(a)+v(b))^2 w(a+b-delta)]` is defined in **Appendix F.1** (proof of Thm 7), not in the main text. It is averaged uniformly over delta != 0 (their eq. 4). That uniform average is the s=1 averaging, not a worst-offset one.
  - The per-frequency amplitude |u^(j)||v^(j)||w^(j)| appears with the phase choice in eq. 7. AM-GM ("inequality of quadratic and geometric means") is eq. 9.
  - Theorem 7 gives gamma* = sqrt(2/27)/(p^{1/2}(p-1)).
- There is no network-level worst-offset coverage bound and no general-s budget. The checker's description ("Fourier-amplitude/phase-condition analysis at b = nu") is accurate.
- Minor: approach.tex says "the quantity eta_{u,v,w}(delta) of Morwani". A pointer to "Appendix F.1" would help a reader find it.

### kam2026symmetry: CONFIRMED to exist; its role is overstated
- arXiv abs page https://arxiv.org/abs/2609.17926:
  - Title: "Symmetry without a manifold: intrinsic dimension on orbits".
  - Authors: Chon-Fai Kam, Miloud Bessafi, Frédéric Cadet.
  - v1 15 Sep 2026. The comments say "Under review at NeurReps 2026 (NeurIPS workshop, Proceedings Track)".
  - The bib as an arXiv preprint is correct.
- App. A.3.2, "Constrained amplitude", read from the v1 PDF:
  - Eq. (15) is `Delta_S(beta) = min_{s!=0} sum_{k in S} beta_k (1 - cos(omega_k s))`, under `||beta||_2 = B`. Then `Delta*_S(B) = max_{||beta||=B} Delta_S(beta)`.
  - **This is the separation of the idealised Fourier-logit solution** ("the logit separation between the correct class and the competitor at offset s is sum_k beta_k(1-cos(omega_k s))"). It is **not** an upper bound on the margin of an arbitrary network.
  - B is exogenous ("Weight decay does not permit unbounded logits"). It is not derived from homogeneity.
- Prop. 5 / Eq. (16): uniform `beta_k = B/sqrt(K_max)` is optimal at full content, with `Delta* = Bp/sqrt(2(p-1))`.
- **The proof of Prop. 5 (Eq. 18) contains `Delta_S(beta) <= p/(p-1) * sum_k beta_k`.** This is exactly the s = 1 case of prop:coverage, with Gamma_1(p) = p/(p-1) as in our cor:dense. Their proof uses the same averaging identity sum_{s!=0}(1 - cos) = p. Only after that does Cauchy-Schwarz turn it into l2. So l2 enters only through that step.
- For proper subsets, uniform allocation is **not** optimal. Nelder-Mead beats it by 25% at K=4 and by 28% at K=10. So "their optimum is dense" holds only at full frequency content.
- The authors disclaim the budget story:
  - "the comparison gives no support to the budget account and we do not claim it does";
  - "we record its absence rather than presenting the calculation of this appendix as an explanation of Sec. 6".
- Consequence: the checker's phrase "the 'margin <= return x budget' rescaling is implicit there too, but with exponent 2" is loose. What Kam actually has is the s = 1 (l1) rescaling for idealised logits, followed by Cauchy-Schwarz. Change the suggested wording from "where the optimum is dense" to "where at full frequency content the uniform allocation is optimal (their Prop. 5)".

## Missed prior art (verified)

### gunasekar2018implicit: Gunasekar, Lee, Soudry, Srebro, "Implicit Bias of Gradient Descent on Linear Convolutional Networks", NeurIPS 2018 (vol. 31)
- Venue and bib confirmed from the proceedings page https://proceedings.neurips.cc/paper_files/paper/2018/hash/0e98aeeb54acf612b9eb4e48a269814c-Abstract.html and its official BibTeX. The arXiv v2 PDF (1806.00468) footer says "32nd Conference on Neural Information Processing Systems (NeurIPS 2018)".
- Theorem 2a (read from the PDF): the GD limit direction is "a scaling of a first order stationary point of min ||beta^||_{2/L} s.t. forall n, y_n <beta, x_n> >= 1" (eq. 10), where beta^ is the vector of Fourier coefficients. The paper notes that p = 2/L < 1 makes this a quasi-norm and the problem non-convex.
- Lemma 7: `R_Pconv(beta) = min_{w: Pconv(w)=beta} ||w||_2^2 = L ||beta^||_{2/L}^{2/L}`. This is the AM-GM price of a product of L factors, the scalar analogue of our "amplitude A costs A^{b/nu}".
- Relation to prop:coverage:
  - The two are the same template (homogeneity, then a Fourier quasi-norm budget, then max-margin over the quasi-norm ball), specialised to b = 2 and nu = L.
  - For linear predictors their reduction is exact. Ours needs the phase lemma (averaging) plus subadditivity across neurons (thm:universal), because the model is nonlinear, and it applies to the task's coverage vectors on Z_p.
  - Neither our general b nor the Z_p coverage functional is in their paper.

## Other ML papers checked; none contains the bound
Each was read in full text (arXiv PDF, grepped for margin, 1-cos, budget, quasi-norm):
- **Li et al. 2402.09469** (AISTATS 2025): follows Morwani, with the single-neuron class-weighted certificate at b = nu (App. D).
- **McCracken et al. 2505.18266** (Thm 4.7): the unnormalised coverage h(i+j)-h(k) for unit amplitudes, used as a sufficiency statement. No norm and no upper bound.
- **He, Wang, Chen, Yang 2602.16849**: Remark 5.1 links early-phase CE to Morwani's average margin. No amplitude-budget bound.
- **Huang, Li 2511.23443**: a single-frequency construction with margin 1-cos(2 pi/p). Not a bound on all networks.
- **Mohamadi et al. 2407.12332**: l_inf-normalised margin, generalisation bounds. No Fourier budget.
- **Tian 2410.01779**: no margin/Fourier budget bound.
- **Tan, Gai, Zhang 2606.08985**: a 1 - cos angular cost on embedding angles (Prop 5.5, Thm 5.6). Not a budget over frequencies.
- **Tan, Gai, Zhang 2606.05863**: a Schatten quasi-norm from layerwise weight decay. This is the generic mechanism, not this bound.
- **Tikeng Notsawo et al. 2506.05718** (ICML 2025): the abstract covers l1/nuclear regularisation and grokking. Not relevant.
- **Rangamani 2503.22059**: empirical only.
- **Gu et al. 2504.03162**: unrelated.

## Searches run (this verification)
- The WebSearch budget was already exhausted (200/200), so no WebSearch calls ran here.
- **arXiv API**, abstract/title search:
  - `"modular addition" AND margin`
  - `"modular addition" AND "weight decay" AND frequencies`
  - `grokking AND "margin maximization"`
  - `grokking AND Fourier AND sparse`
  - `"cyclic group" AND margin`
  - `"group composition" AND margin`
  - `"Fourier features" AND "weight decay" AND margin`
  - `"modular arithmetic"/"modular addition" AND "implicit bias"`
  - `"quasi-norm" AND margin AND homogeneous`
  - `"modular addition" AND sparse`
  - `grokking AND "weight decay" AND sparse`
  - `"feature sparsity" AND margin`
  - `modular AND frequencies AND logarithmic`
- **Semantic Scholar**:
  - The citations of arXiv:2311.07568 were pulled: 40 citing papers, all screened by title.
  - Keyword search and snippet (full-text) search were rate-limited (HTTP 429) and returned nothing.
- **OpenAlex**: 4 keyword searches, filtered to 2024 onwards. Nothing relevant.
- **Gaps:** the full-text snippet search failed; concurrent ICLR 2027 submissions cannot be seen.

## Recommended change
- The proposition itself needs no change.
- Next to eq:programme / prop:coverage, extend the R4 sentence rather than adding a new one:

> "The same homogeneity reduction, in which an $\ell_2$ weight cost becomes an $\ell_{2/L}$ quasi-norm on Fourier coefficients over which the margin is maximised, appears for depth-$L$ linear convolutional networks \citep[Thm.~2a, Lemma~7]{gunasekar2018implicit}. Proposition~\ref{prop:coverage} is its analogue for one-hidden-layer networks on $\mathbb{Z}_p$, with the phase lemma in place of linearity. Concurrently, \citet[App.~A.3.2]{kam2026symmetry} evaluate the same worst-offset separation for idealised Fourier logits under an $\ell_2$ amplitude budget and find the uniform allocation optimal at full frequency content."

- Do not describe Kam's optimum as "dense" without that qualifier.
- Optionally, cite Morwani "App. F.1" for eta_{u,v,w}.
