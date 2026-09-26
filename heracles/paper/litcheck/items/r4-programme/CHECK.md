# R4-programme: novelty check of eq:programme (approach.tex)

Item: Gamma_s(p) = max_{y>=0, sum y_j^{1/s} <= 1} min_{delta!=0} sum_j y_j kappa_j(delta), kappa_j(delta)=1-cos(2 pi j delta/p); K* = |supp y*|.

## Verdict: PARTIALLY_KNOWN (as a *programme shape*); the specific programme + its role (margin bound for homogeneous nets, K* count) is not found.

## Three reformulations a reviewer will spot, each with prior art

1. **Max-margin linear classifier under an l_q bridge quasi-norm (q=1/s<1) on Fourier coefficients.**
   Gamma_s is the l_{1/s}-quasi-norm max-margin of a nonnegative linear separator for the p-1 "points" kappa(delta).
   - Gunasekar, Lee, Soudry, Srebro, NeurIPS 2018 (arXiv 1806.00468), **Theorem 2a** (read from PDF): for depth-L linear
     convolutional nets, the GD limit direction is "a scaling of a first order stationary point of
     min_beta ||beta_hat||_{2/L} s.t. forall n, y_n<beta,x_n> >= 1" (eq. 10), beta_hat = DFT coefficients; Thm 2 (L=2):
     ||beta_hat||_1. They note p=2/L<1 is non-convex, cite Ge et al. 2011. Exponent 2/L comes from depth homogeneity,
     exactly like our 1/s = b/nu from AM-GM.
   - Jagadeesan, Razenshteyn, Gunasekar, COLT 2022 (arXiv 2102.12238): l2 weight norm in multichannel linear convnets
     induces nuclear / l_{2,1} group-sparse norms on Fourier coefficients.
   => The "homogeneity -> Fourier-domain l_{2/depth} quasi-norm margin -> frequency sparsity" template is KNOWN (for linear
   convnets, generic data). New in ours: the task-specific constraint set (cosine coverage of all offsets of Z_p),
   the extension to multiplicative/power neurons with L_{2,b}, the fact that it upper-bounds *every* network (Thm universal),
   and the counts (K*, log p, shapes). Must cite Gunasekar et al. next to eq:programme.

2. **Weighted Laplacian spectral gap of a circulant (Cayley) graph.**
   sum_j y_j (1-cos(2 pi j delta/p)) is the delta-th Laplacian eigenvalue of the weighted Cayley graph Cay(Z_p,{+-j}) with
   edge weights y_j, so Gamma_s = max algebraic connectivity under an l_{1/s} weight budget.
   - s=1 (l1 budget, convex): Boyd, "Convex Optimization of Graph Laplacian Eigenvalues", Proc. ICM 2006, 3:1311-1319
     (edge-weight choice to maximise lambda_2 is convex). Our Cor. dense (uniform = complete graph) is the circulant case.
   - s -> infinity (l0 / cardinality budget): Ghosh & Boyd, "Growing Well-connected Graphs", 45th IEEE CDC 2006 (read):
     "choose k edges from E_cand that lead to the greatest increase in algebraic connectivity ... maximize lambda_2(L(E_base u E))
     subject to |E|=k", with SDP relaxation.
   - Random O(log p) generators give constant gap: Alon-Roichman 1994 (already cited in appendix).
   => Our programme interpolates between Boyd's convex l1 problem (s=1) and Ghosh-Boyd's cardinality problem (s->inf)
   via a concave l_{1/s} budget, restricted to circulants. The interpolation + its closed-form/Theta(log p) answers were not found.

3. **Coverage max-min with an amplitude budget in modular addition (ML).**
   - Kam, Bessafi, Cadet, arXiv 2609.17926 (15 Sep 2026), App. A.3.2 Eq. 15 (fetched HTML): Delta_S(beta)=min_{s!=0} sum_k beta_k(1-cos(omega_k s))
     with ||beta||_2 = B; Prop. 5: uniform allocation optimal at full content. Same objective, convex l2 budget -> dense.
     Concurrent; cite as foil.
   - Morwani et al. ICLR 2024: the s=1 endpoint (dense, gamma* matches our Cor. dense).

## Adjacent, no overlap
- Chowla cosine problem (Bedert 2509.05260 etc.): minimum of unweighted cosine sums, different extremal problem (see sweep LIT_math §4).
- l_p (p<1) minimisation theory: Ge-Jiang-Ye 2011 Thm 3 (basic feasible solutions = local minimisers) supports vertex/shape claims; generic.
- Thinned-array peak-sidelobe design (random sparse arrays) is the s->inf unweighted analogue; searches found no l_q-budget version. Not fetched; no bib.

## Searches run
- "maximizing algebraic connectivity edge weights convex optimization Ghosh Boyd"
- "maximize minimum nonzero Laplacian eigenvalue circulant graph weights sparse concave cost"
- "max-min cosine sum nonnegative coefficients l_p quasi-norm constraint p<1 sparse optimal support"
- "linear programming boosting margin maximization l_q quasi-norm q<1 sparse ..." (LPBoost: l1 margins sparse; no q<1)
- "frequency selection nonnegative weights maximize worst-case 1-cos offset cyclic group sparse ... log p" (nothing)
- "maximum margin l_p quasi-norm p<1 support size bound ..." (generic)
- "weighted Cayley graph cyclic group maximize spectral gap budget on weights sparse generating set" (nothing)

## Recommended change
After eq:programme add: "Equivalently, Gamma_s is the l_{1/s}-quasi-norm maximum margin of a nonnegative separator of the
coverage vectors kappa(delta); the same 'depth/homogeneity -> l_{2/L} bridge penalty on Fourier coefficients' reduction appears
for linear convolutional networks \citep{gunasekar2018implicit}. It is also the largest Laplacian spectral gap of a weighted
circulant graph on Z_p under an l_{1/s} weight budget, interpolating between the convex l_1 problem (s=1, \citealp{boyd2006convex})
and the cardinality-constrained problem (s->inf, \citealp{ghosh2006growing}). \citet{kam2026symmetry} study the same coverage
objective under a convex l_2 budget and find the uniform (dense) allocation optimal; the concave budget is what makes ours sparse."
