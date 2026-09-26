# VERIFY (citation lens): F4-margin-peaks, task 68

Claim (paper.tex abstract, l.58): "we empirically show that the normalized margin attained by these networks peaks at the predicted optimum for the network, matching the result of Morwani et al. for their specific case. The optimums are also consistent with the empirical results seen in Chughtai et al."

Checker verdict: PARTIALLY_SUPPORTED. **Upheld.**

## Internal
- sections/experiments.tex has only TODO comments, and sections/groups.tex is a skeleton ("Skeleton only"). Nothing in the paper yet shows a margin *peak* over feature count. The empirical half of the abstract sentence is currently unsupported.

## morwani2024feature (fetched arxiv.org/abs/2311.07568 and arxiv.org/html/2311.07568)
- Exists. Authors match the bib: Morwani, Edelman, Oncescu, Zhao, Kakade. The Comments field says "Accepted as Spotlight at ICLR 2024". The bib entry is correct.
- Theorem 7: the maximum L_{2,3}-margin on D_p with m >= 4(p-1) is gamma* = sqrt(2/27) / (p^{1/2}(p-1)). This matches Cor. dense (approach.tex l.70) and appendix.tex l.32.
- Fig 1(c): "For the quadratic activation, the network asymptotically reaches the maximum L_{2,3} margin predicted by our analysis." Fig 6(a) (S5, L_{2,3}): "Evolution of the normalized L_{2,3} margin ... approaches the theoretical maximum margin". Fig 5 shows the same for sparse parity with L_{2,5}.
- Morwani plots margin against training time, **not against the number of features**. So "matching Morwani" can only mean that the dense-endpoint value (b = nu) matches Theorem 7. It cannot mean a peak.
- Theorem 9 (groups, quadratic): gamma* = 2/(3 sqrt(3|G|)) / sum_{n>=2} d_{R_n}^{2.5}, and every non-trivial irrep is present. If the paper also claims "matching Morwani" for groups, the group ceiling at b = nu must reproduce this formula. groups.tex does not state it yet.

## chughtai2023toy (fetched proceedings.mlr.press/v202/chughtai23a.html and arxiv.org/html/2302.03025v2, with full-text tables extracted)
- PMLR 202, pp. 6243-6267, ICML 2023. Authors Chughtai, Chan, Nanda. The recommended bib is correct. The existing refs.bib entry lacks volume and pages, so use the recommended one.
- Setup: one-hidden-layer ReLU MLP (d=256, h=128) and a transformer, AdamW, wd = 1, 40% of the table. This is not the paper's product/power class.
- Fig 7 caption (50 S5 MLP seeds), verbatim: "the 1d sign and 4d standard representations are most commonly learned, standard_sign (4d), 5d_a and 5d_b are learned approximately equally and less often" ... "Most commonly we have two key representations (typically sign and standard), but sometimes we learn more."
- Table 5, S5 MLP: {sign, std-sign, std, 5d-a}, {sign, std}, {sign, std, 5d-b}, {sign, std}. That is 2/4 seeds exactly "sign + one faithful irrep".
- Table 6, S5 transformer: {sign, std-sign} x2, {sign, std} x2. That is 4/4 seeds "sign + one 4d faithful irrep", the strongest support. The checker listed these rows but did not point this out.
- Table 5, A5 MLP: {std, 3d-a, 3d-b}, {std, 3d-a, 3d-b}, {3d-a, 5d-a, std}, {5d-a, 3d-a, std, 3d-b}. The Galois pair alone appears in 0/4 seeds. Seed 3 has only 3d-a. The checker's "3d-a/3d-b" is accurate.
- Table 6, A5 transformer: {3d-a, 3d-b} x2, {5d-a, 3d-b, 3d-a} (test loss 1.95e-1, not converged), {standard}. The pair alone appears in 2/4 seeds.
- C113: MLP key-frequency counts are 13, 11, 15, 10 (so 10-15). Transformer counts are 3, 4, 4, 3.
- Sec 6, verbatim: "the network preferred the standard representation over the standard_sign representation, when in fact standard_sign offers better performance for fixed weight norm." So a norm ceiling can at most predict the count, not which irrep is chosen.

## Extra (fetched)
- Stander, Yu, Fan, Biderman, "Grokking Group Multiplication with Cosets", ICML 2024, PMLR 235:46441-46467 (proceedings.mlr.press/v235/stander24a.html; arXiv 2312.06581v2).
  - They reverse-engineer one-hidden-layer FC nets on S5/S6 (128 S5 seeds).
  - They report that the weights "concentrate on specific irreps in the group Fourier basis", but attribute this to coset circuits, not realized matrix representations.
  - "Every model has a few sign circuit neurons that correspond to A_n" (App. H.1, Fig 6).
  - They say most of Chughtai's evidence is also consistent with coset circuits.
  - This matters because the paper counts features by irreps. Cite it next to Chughtai as a second S5 source and as the known challenge to the irrep reading. The irrep *count* is observed either way.

## Recommendation
Keep the checker's rewrite. Specifically:
1. Drop "peaks" unless experiments.tex gains a plot of margin against K or b.
2. Rephrase "matching Morwani" as "recovers the maximum L_{2,3} margin of Morwani et al. (Theorem 7) at b = nu".
3. Limit the "consistent with Chughtai" statement to S5: most common in the 50-seed MLP study, 4/4 transformer seeds. For A5, state honestly that 2/4 transformer seeds and 0/4 MLP seeds show the pair alone.
4. Do not claim cyclic consistency: the C113 MLPs use 10-15 frequencies.
5. Optionally add stander2024grokking.
