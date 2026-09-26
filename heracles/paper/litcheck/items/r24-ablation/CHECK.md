# R24-ablation: Prop. "removing a feature" (cyclic.tex prop:ablation)

Statement: for y feasible with budget sum y_j^{1/s}=1, worst coverage m >= (1-eps)Gamma_s(p), share w_j=y_j^{1/s}:
  m - 2 y_j <= m_{-j} <= (1-w_j)^s m/(1-eps); a maximizer loses >= 1-(1-w_j)^s in [w_j, s w_j] of its margin.
The proof is elementary: the lower bound holds because |coverage| <= 2; the upper bound holds because the
reduced vector, rescaled to unit budget, is feasible (homogeneity of degree s).

## Verdict: NOVEL_AS_FAR_AS_FOUND (as a theorem). Empirical analogues exist and should be cited.

## Searches run
- "theoretical bound margin loss ablating a single Fourier frequency modular addition network"
- "Nanda progress measures grokking ablating key frequencies loss restricted excluded loss"
- "Chughtai toy model of universality ablation irreducible representations key representations loss"
- "ablation single frequency grokked network loss increase proportional to its norm share weight decay theory"
- "provable bound effect of removing one feature on max-margin classifier sparse Fourier features lower bound loss fraction of margin"
- Direct reads of Morwani et al. (arXiv html 2311.07568) and McCracken et al. (arXiv html 2505.18266v1, App. G.8)

## Findings
1. Nanda, Chan, Lieberum, Smith, Steinhardt, "Progress measures for grokking via mechanistic interpretability",
   ICLR 2023 (PDF header confirmed; arXiv 2301.05217). Sec. 4.4 "Correctness checks: ablations":
   "ablating any key frequency causes a significant increase in loss ... In contrast, ablating other frequencies
   does not hurt the model at all." Fig. 6 shows the loss after ablating each k in {1..56}. The intro says that
   "ablating key frequencies used by the model reduces performance to chance, while ablating the other 95% of
   frequencies slightly improves performance". Sec. 5 defines restricted and excluded loss. This is EMPIRICAL
   only, with no quantitative law linking the loss to a feature's norm share. Our proposition is the first
   theoretical counterpart we found. Already cited in the paper as nanda2023progress (intro, approach).
2. Chughtai, Chan, Nanda, "A Toy Model of Universality", ICML 2023, PMLR 202:6243-6267 (confirmed on PMLR page).
   They confirm the GCR algorithm "using ablations" of the key-representation directions. This is empirical,
   with no bound. Already in the bib as chughtai2023toy.
3. McCracken et al. arXiv 2505.18266. App. G.8 "Noise and ablation studies", Fig. 38, removes neurons from
   frequency clusters and reports the loss and accuracy. This is empirical. Their Thm 4.7/Cor 4.8 concern how the
   margin scales with the NUMBER of frequencies, not the loss from removing one.
4. Morwani et al. (ICLR 2024; arXiv 2311.07568): no theorem about removing a frequency from the max-margin
   network. The WebFetch summary reports that the paper does not analyse margin degradation from removing
   components.
5. Adjacent fields: nothing found that states a "share -> fractional loss" bound for max-min allocations under
   a concave l_{1/s} budget. The bound is a one-line consequence of homogeneity. Reviewers may call it
   elementary, but it is not published in this setting.

## Recommendation
Keep the proposition. In the sentence after it (cyclic.tex, around l.87), add that it is the theoretical
counterpart of the per-frequency logit ablations of Nanda et al. (Sec. 4.4, Fig. 6) and the
representation ablations of Chughtai et al. For example: "This is the theoretical counterpart of the
key-frequency ablations of \citet[Sec.~4.4]{nanda2023progress} and the representation ablations of
\citet{chughtai2023toy}: every feature is load-bearing in proportion to its share of the regularization budget."
Optionally cite McCracken et al. (App. G.8) for cluster-level ablations. Present the proposition as
elementary (a consequence of homogeneity) rather than deep, so reviewers do not see it as overclaiming.
