# R37 appendix remarks (app:remarks, app:conjecture): lit check

Verdict: RELATED_ONLY. The remarks are internal. There is one standard precedent to cite, for "b > nu gives a margin that is unbounded in width".

## Sub-claims
1. **Equal-share bound** Gamma_s >= max_K K^{-s} M_K, with M_K <= pK/(p-1). This follows directly from the programme, and the averaging step is elementary. M_K = K - max_{delta!=0} sum_{j in S} cos(2 pi j delta/p) is K(1 - Fourier bias of S). That links it to small-bias sets (the Alon-Roichman line is already cited in app. B). Status: internal. No prior art is needed.
2. **Only interval patterns are needed in (JF).** This is internal to the proof of Lemma cost. No literature applies.
3. **The factor counts operand factors, not degree** ((n+1)^{-(nu-b)/b}). Internal.
4. **Power-net interference-free population** (6 neurons, 0.408 vs 3^{-1/2}; no population with tied embeddings). These are internal computations. Li et al. (AISTATS 2025, Thm 4.1) give the power-network L_{2,nu} optimum at b = nu (checked under another item). Nothing I found addresses b < nu attainment for power nets.
5. **b > nu: margin unbounded in width.** Proof: split a neuron into H copies, each scaled by H^{-1/nu}. The output is unchanged, and sum ||.||^b scales as H^{1-b/nu} -> 0. This is the classical width-splitting argument for group norms. Neyshabur, Tomioka, Srebro, COLT 2015 (PMLR 40:1376-1401). Sec. 3.1, after Thm 1: the width dependence "disappears ... as long as q <= p* (i.e. 1/p+1/q >= 1)". Sec. 3.2 / Thm 3 proof: "replacing the top unit with H copies of it ... we have reduced the norm by factor H^{1/q-1/p*}", and "we can shatter an arbitrarily large number of points with arbitrarily low gamma by using enough hidden units". Their setting is ReLU depth-d mu_{p,q} group norms, not our nu-homogeneous neurons with L_{2,b}. The mechanism is identical, so the claim is a trivial specialisation. **PARTIALLY_KNOWN in spirit; cite.**
6. **b < 2: superadditivity fails by |F|^{1-b/2}.** This is an elementary fact about t -> t^{b/2}. Internal.

## Morwani et al. (ICLR 2024) on b != nu
From arxiv.org/html/2311.07568: "we will primarily focus on maximum margin with respect to L_{2,nu} norm" (Sec 3.2). App. E.1 remark: "the analysis ... can be used to compute optimal norms for b != nu as well -- however, for any such b we would not get the same flexibility to build a theta* satisfying Equation [1]". They say nothing about b > nu being unbounded or about b < 2. The paper's b range [2, nu] extends their b = nu. No overlap.

## Savarese et al. COLT 2019
PMLR 99:2667-2690. The standard reference for rescaling-balanced weight decay on infinite-width 2-layer ReLU nets, i.e. the representation cost. It is relevant to the ReLU paragraph (b = nu = 2 weight decay equals the bound), not to these remarks. Cite it only if the ReLU paragraph needs one.

## Searches
- Morwani HTML, full-text read for the b vs nu discussion.
- Neyshabur-Tomioka-Srebro PDF (arXiv 1503.00036v2), read Sec. 3.1-3.2.
- PMLR pages for the Neyshabur 2015 and Savarese 2019 venue confirmation.
