# R7-universal-bound (approach.tex thm:universal) — literature check, 2026-09-25

Statement: for any one-hidden-layer nu-homogeneous net, any width, 2<=b<=nu:
gamma*_b <= C_b^{nu/b} Gamma_{nu/b}(p).
Proof = averaged-margin/phase lemma (Morwani-style) + coverage programme + pricing
(sum_j A_j^{b/nu} <= sum_{ij}|beta_ij|^{b/nu} <= C_b ||theta||_{2,b}^b by subadditivity of t^r, r<=1).

## Verdict: PARTIALLY_KNOWN (b=nu endpoint known; b<nu general bound not found)

### b = nu endpoint (KNOWN)
- Morwani et al. ICLR 2024 (arXiv 2311.07568, HTML fetched). All theorems at L_{2,nu}; Thm 7:
  max L_{2,3} margin = sqrt(2/27)/(p^{1/2}(p-1)) for the quadratic net (width >= 4(p-1)).
  No theorem for plain L2 (b<nu); Fig 4b shows L2 ReLU nets "nearly 1-sparse ... but not quite" (empirical only).
  Their class-weighted-margin argument (Sec 3, Lemma 2/6) is the template of our Lemma avg.
- Li, Liang, Shi, Song, Zhou, AISTATS 2025 PMLR 258:523-531 (page fetched): maximum L_{2,k+1} margin
  for k-input sum with power activation, m >= 2^{2k-2}(p-1). Only L_{2,k+1} (b=nu).
- Our cor:dense reproduces Morwani Thm 7 exactly; the paper already says so. Fine.

### b < nu (esp. b=2 weight decay): NOT FOUND
Checked and ruled out:
- Mohamadi et al. ICML 2024 (2407.12332, HTML fetched): l_inf norm; Prop 3.8 existence of zero-loss
  solutions with ||theta||_inf <= floor(h/8p)^{-1/3}; Thm 4.5 generalisation for Omega(p) l_inf margin.
  No upper bound on max margin over all widths, no frequency count.
- Tian, NeurIPS 2025 (2410.01779, HTML fetched): semi-ring/CoGS; Thm 5 "lower-order solutions are preferred
  if trained with L2 regularization" (path-connectivity argument); no margin bound.
- Jacobs & Montufar 2026 (2605.19458, abs fetched): mirror-flow implicit bias, sparse vs dense neurons;
  no modular addition, no L_{2,b} margin bound.
- Kam, Bessafi, Cadet 2026 (2609.17926, HTML fetched): App A.3.2 Eq 15 max_{||beta||_2=B} min_s sum beta_k(1-cos),
  Prop 5 uniform allocation optimal; budget from weight decay but posited as l2 on amplitudes for the quadratic
  net only; they state no claim for all networks/widths. Convex budget -> dense. Concurrent; closest analogue
  of the programme, but not of the universal bound (they never derive the exponent b/nu nor bound over all nets).
- Memana 2026 (2608.04451, fetched): single ReLU neuron, no weight decay. Irrelevant.
- Nguyen 2026 (2606.17399, fetched): mech-interp of modular multiplication, no margin bounds. Irrelevant.
- McCracken et al. 2025 (2505.18266, via sweep): unnormalised sufficiency result; not an upper bound.

### Pricing step (A^{b/nu} via AM-GM + subadditivity): KNOWN INGREDIENT
- Hoff 2017 (arXiv 1611.00040, fetched): "a subclass of L_q penalties with q<=1 can be expressed as sums of L_2 penalties" (Hadamard reparametrisation).
- Dai, Karzand, Srebro NeurIPS 2021 (proceedings page fetched): representation cost of l2 on parameters, l_p costs.
- Fan, Ding, Yang, Zhang, Udell, TMLR 2023 (arXiv 2012.03436, fetched): Euclidean norms of CP components
  are "monotonic transformations of tensor Schatten-p quasi-norm". Our product neuron = rank-one CP term,
  so pricing a feature at A^{b/nu} is the diagonal (Fourier) case of this identity. New candidate citation.
- Kolb et al. ICLR 2025, Ongie-Parhi 2026: per sweep (not refetched here).

### What is new
The combination: an upper bound on the max L_{2,b} margin, b<nu, valid for every width and every
neuron (mixed frequencies, interference allowed), reducing to a concave-budget coverage programme.
No paper found with a margin bound for modular addition under any norm other than L_{2,nu} (and l_inf existence results).

### Recommended text change (approach.tex after thm:universal proof)
Add: "At b=nu this is the bound of \citet{morwani2024feature} (Thm 7) and \citet{li2025fourier}; for b<nu no
bound was previously available. The pricing step is the diagonal case of the fact that l2 penalties on factors
induce l_q (q=2/nu) quasi-norm costs \citep{hoff2017lasso,dai2021representation,fan2023euclidean}; \citet{kam2026symmetry}
study the same coverage max-min with a convex l2 budget on amplitudes and find a dense optimum."

Searches run: modular addition max margin L2 weight decay sparse Fourier any width; "max-margin" modular addition L_{2,nu};
Tian algebraic objects weight decay; margin maximization number of frequencies log p; weight decay rank-one tensor
Schatten quasi-norm 2/N; modular addition weight decay fewer frequencies theorem 2026.
