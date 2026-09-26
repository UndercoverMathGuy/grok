# F6-where-dense-stops (framing claim, intro "Where the dense theorem stops")

Claim: Morwani find the best individual neurons and then combine them. Combining is free exactly at L_{2,nu}. For b<nu, concentrated output amplitude is cheaper, so each feature has a price.

## Morwani et al. (ICLR 2024, arXiv 2311.07568), fetched as arXiv HTML
- Sec 3.3 blueprint: (1) take a uniform q on symmetric data, (2) find the optimal single neurons with Lemma 5 "Single neuron optimization", (3) combine them with Lemma 5 "Combining neurons".
- Lemma 3 (binary) is verbatim: "If b=ν (the homogeneity constant of the network) and ω1*,...,ωm* ∈ Ωq*, then for any neuron scaling factors Σλi^ν=1, λi≥0 we have that θ={λiωi*} belongs to Θq*." Lemma 5 is the multi-class version.
- Sec 3.2: "Thus, we will primarily focus on maximum margin with respect to L_{2,ν} norm in this work."
- App. E.1 remark, verbatim: "Note that the analysis can be used to compute optimal norms for b≠ν however, for any such b we would not get the same flexibility to build a θ* satisfying Equation [1]."
- Sec 7: "Our proof holds for the case of x² activations ... and L_{2,ν} norm".
- Fig 4b: ReLU trained with L2 is "nearly 1-sparse" per neuron. That is neuron-level sparsity, not a statement about how many frequencies the network uses.

So the description of the proof structure is accurate, and "combining is free exactly at b=nu" is Morwani's own condition in Lemma 3/5. Morwani do remark on b≠nu, but only to say that the flexibility to combine is lost (App. E.1). They say nothing about the direction of the effect, sparsity, or a price per feature. The mechanism ("concentration is cheaper for b<nu") is not theirs.

## Mechanism precedent (AM-GM: L2 on a product of D factors = l_{2/D} quasi-norm, which is concave, so concentrated amplitude is cheaper)
- Hoff 2017 (arXiv 1611.00040, CSDA): L_q penalties with q≤1 written as sums of L2 penalties through a Hadamard-product reparametrisation.
- Dai, Karzand, Srebro, NeurIPS 2021: representation costs of l2-regularised linear nets (l_p/quasi-norm costs).
- Kolb et al., ICLR 2025 (2502.02496): L2 on D factors is equivalent to l_{2/D} non-convex sparse regularisation.
All three give the generic "fewer, larger is cheaper" principle. None applies it to margin, Fourier features, or modular addition.

## Search for anyone remarking on b<nu giving sparse features in algebraic tasks
Covered by the sweep (LIT_ml.md). Kam et al. 2609.17926 use a convex l2 amplitude budget and get dense. Nobody states the b<nu sparsity pricing for these tasks.

Verdict: SUPPORTED. The intro may cite Morwani App. E.1 for "combining is free exactly at b=nu", and should cite Hoff/Kolb/Dai et al. for the concave-cost mechanism.
