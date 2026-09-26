# F4-margin-peaks: abstract, final two sentences

Claim (paper.tex l.58): "we empirically show that the normalized margin attained by these networks peaks at the predicted optimum ... matching the result of Morwani et al. for their specific case. The optimums are also consistent with the empirical results seen in Chughtai et al."

Verdict: PARTIALLY_SUPPORTED.

## 1. Internal status (the most serious problem)
- sections/experiments.tex has only TODO comments. Nothing in the paper shows a "peak at the predicted optimum". The abstract asserts a result that the paper does not contain.
- sections/groups.tex is a one-paragraph skeleton. It says "one representation on S_5 and the Galois-conjugate pair on A_5, as observed" and gives no citation.

## 2. Morwani et al. (arXiv 2311.07568, ICLR 2024). Fetched arxiv.org/html/2311.07568
- Fig 1(c): "For the quadratic activation, the network asymptotically reaches the maximum L_{2,3} margin predicted by our analysis" (mod-71, trained with L_{2,3} regularisation). Fig 6(a) shows the same for S_5: "It approaches the theoretical maximum margin that we predict." Fig 5 shows it for sparse parity with L_{2,5}.
- Our Cor. dense (approach.tex l.69-71) reproduces their value, sqrt(2/27)/(p^{1/2}(p-1)). The "matches Morwani for their case" part is therefore true at the level of the formula: this is a theorem-level match, not a new empirical one.
- Caveat on wording: their experiments reach the maximum over training. They do not show a peak across a family of network sizes or feature counts. If "peaks" means "max over K at K*", Morwani provides no such curve.

## 3. Chughtai, Chan, Nanda (ICML 2023, PMLR 202:6243-6267). Fetched the PMLR page and arxiv.org/html/2302.03025v2
Setup: one-hidden-layer ReLU MLP (and a transformer), AdamW, lr 1e-3, weight decay lambda=1, 40% training fraction. This is not our product/power network and not the max-margin regime.
- S_5, Fig 7 caption: "the 1d sign and 4d standard representations are most commonly learned, standard_sign (4d), 5d_a and 5d_b are learned approximately equally and less often, and 6d is never learned. (Right) ... Most commonly we have two key representations (typically sign and standard), but sometimes we learn more." (50 MLP seeds)
- Table 5, S_5 MLP (4 seeds): {sign, std-sign, std, 5d-a}, {sign, std}, {sign, std, 5d-b}, {sign, std}. Table 6, S_5 transformer: {sign, std-sign} x2, {sign, std} x2.
  => "one faithful irrep + sign" is the modal outcome, not universal: 2/4 MLP seeds carry extra irreps. The claim is consistent only if the sign is counted as a patch and not as a feature.
- A_5, Table 5 MLP: {std(4d), 3d-a, 3d-b}, {std, 3d-a, 3d-b}, {3d-a, 5d-a, std}, {5d-a, 3d-a, std, 3d-b}. Table 6 transformer: {3d-a, 3d-b} x2, {5d-a, 3d-b, 3d-a}, {std}.
  => The Galois pair on its own is observed in 2/4 transformer seeds and 0/4 MLP seeds. Every MLP seed also carries the 4d standard irrep (and sometimes 5d). "the Galois-conjugate pair on A_5, as observed" is NOT what Chughtai's MLPs show. If "as observed" means our own runs (memory: heracles S5 program, A5 = Galois conjugate), say so and cite our experiments, not Chughtai.
- Sec 6: "the network preferred the standard representation over the standard_sign representation, when in fact standard_sign offers better performance for fixed weight norm." A reviewer can use this against any claim that the norm-optimal ceiling picks the irrep. Our ceiling should claim the count only, not which irrep (consistent with the internal finding that "the ladder predicts count, not the seed's choice").
- Cyclic C_113, Table 5 MLP: 10-15 key frequencies per seed. Table 6 transformer: 3-4. The MLP counts are well above a log-p K* of ~4-6. Do not say our cyclic optimum is "consistent with Chughtai" without qualifying architecture and degree (ReLU = nu 2 in our accounting, the dense endpoint per appendix l.283).

## Recommended change
Replace the last two abstract sentences with something like: "Trained networks approach the predicted margin ceiling, recovering the maximum L_{2,3}-margin of Morwani et al. at the dense endpoint, and the predicted irrep count (one faithful irrep plus the sign character on S_5) agrees with the modal outcome reported by Chughtai et al." Keep "peaks at the predicted optimum" only if experiments.tex actually contains a margin-vs-K plot. In groups.tex, cite chughtai2023toy (Fig 7, Tables 5-6) for S_5 as "modal", and either drop "as observed" for the A_5 Galois pair or attribute it to our own runs plus the Chughtai transformer runs (2/4). Note that the Chughtai MLPs add the 4d standard irrep. Add volume/pages to the chughtai bib entry.
