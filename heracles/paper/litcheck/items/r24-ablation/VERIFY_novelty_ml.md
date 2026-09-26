# VERIFY (novelty_ml): R24-ablation, cyclic.tex prop:ablation

Claim: for y feasible with sum_j y_j^{1/s}=1 and worst coverage m >= (1-eps)Gamma_s(p), with share w_j=y_j^{1/s}:
  m - 2y_j <= m_{-j} <= (1-w_j)^s m/(1-eps).
So removing feature j from a maximizer costs at least the fraction 1-(1-w_j)^s of the margin, which lies in [w_j, s w_j].
Proof: |coverage| <= 2 gives the lower bound. For the upper bound, rescale the reduced vector to unit budget (degree-s
homogeneity), then apply optimality of Gamma_s, then Bernoulli.

Checker verdict: NOVEL_AS_FAR_AS_FOUND. My verdict: UPHELD. The inequality is new, but there is one missed precedent
for the interpretive sentence (see below).

## Constraint on this pass
The session-wide WebSearch budget (200/200) was exhausted before this verifier ran, so no WebSearch queries
could be issued. Everything below comes from direct fetches of primary sources (arXiv PDFs, PMLR, NeurIPS
proceedings, Crossref, Springer PDF). The adversarial search was therefore targeted, not open-ended.

## Confirmed citations (re-verified myself)
1. nanda2023progress (arXiv 2301.05217v3; PDF header "Published as a conference paper at ICLR 2023"; authors
   Nanda, Chan, Lieberum, Smith, Steinhardt).
   - Sec. 4.4 "Correctness checks: ablations", Logit frequencies: "ablating any key frequency causes a significant
     increase in loss ... In contrast, ablating other frequencies does not hurt the model at all."
   - Fig. 6 shows the loss when ablating each k in {1..56}.
   - The intro contribution (4) sentence matches.
   - Their ablation zeroes Fourier components of the logit table. This is the same operation as the paper's remark
     that deleting a nu-block removes exactly y_j cos(...) from the table, so it is a good analogue.
   - Empirical only: no bound. CONFIRMED.
2. chughtai2023toy: PMLR v202, pp. 6243-6267, authors Chughtai, Chan, Nanda. The abstract says they "confirm our
   understanding using ablations". Empirical. CONFIRMED. The refs.bib entry lacks volume and pages; the checker's
   fuller bibtex is fine.
3. mccracken2025: arXiv 2505.18266v1, "Uncovering a Universal Abstract Algorithm for Modular Addition in Neural
   Networks", by Gavin McCracken, Gabriela Moisescu-Pareja, Vincent Letourneau, Doina Precup, Jonathan Love.
   - App. G.8 "Noise and ablation studies" and Fig. 38 remove neurons from frequency clusters and report loss and
     accuracy. Empirical. CONFIRMED.
   - NOTE: the current refs.bib entry is still WRONG. It has the title "Emergence of sparse and dense circuits..."
     and the author "McCracken, Andrew and others". Replace it with the checker's corrected entry.

## Adversarial ML search (assume it is known; try to find it)
- Morwani et al. 2311.07568v2 (full PDF): no occurrence of "remov"/"ablat". No feature-removal theorem.
- He, Wang, Chen, Yang 2602.16849v1 (Feb 2026, the closest 2026 theory paper). App. A.2 "Ablation studies" restricts
  the number of frequencies |K| in {1,2,4,8} and the phase ranges of a fully-diversified parametrisation, and
  reports CE (empirical, Table 2). There is no bound on the loss from removing one feature.
- Tian 2410.01779v4 (CoGO, algebraic global optimisers): no removal or ablation result. The only "remove" is about
  a loss term.
- MISSED BY CHECKER (classical ML precedent for the interpretation, not for the inequality): Guyon, Weston,
  Barnhill, Vapnik, "Gene Selection for Cancer Classification using Support Vector Machines", Machine Learning
  46(1-3):389-422, 2002 (Crossref-verified; full PDF read).
  - Sec. 2.5 "Feature ranking by sensitivity analysis" uses "the change in cost function DJ(i) caused by removing
    a given feature". Following OBD (LeCun et al.), DJ(i)=(1/2)(d^2J/dw_i^2)(w_i)^2. For linear SVMs, J=(1/2)||w||^2,
    so "This justifies the use of (w_i)^2 as a feature ranking criterion."
  - That is the heuristic "a feature's importance to a max-margin classifier is its share of the l2
    regularization budget". It is a second-order Taylor estimate at fixed dual variables, not a bound.
  - Prop. ablation makes this rigorous, with two-sided bounds, for the budget-splitting programme. The same
    rescaling argument, applied to a unit-norm linear SVM, gives margin loss >= 1-sqrt(1-w_i^2) >= w_i^2/2. Note
    that for that s<1 case the [w,sw] bracket reverses.
  - The paper's sentence "Every feature is load-bearing in proportion to its share of the regularization budget"
    restates the SVM-RFE justification, so a reviewer from feature selection or pruning will recognise it.
    Recommend crediting it.
  - LeCun, Denker, Solla, "Optimal Brain Damage", NIPS 2 (1989), proceedings page fetched. This is the source of
    the saliency approximation, and is an optional second cite.
- Not searchable this pass (budget): the boosting and ensemble-pruning analogue. At s=1 (l1 budget, a game value),
  the upper bound is m_{-j} <= (1-w_j)m. This is a triviality of LP duality and the value of a game, and is likely
  folklore. It does not threaten the s>1 statement.

## Recommendation
- Keep the verdict. The proposition is novel as an inequality in this setting and is elementary, so present it as
  a consequence of homogeneity.
- Add the checker's sentence (Nanda Sec. 4.4 and Fig. 6; Chughtai). Also add a clause crediting the heuristic, e.g.
  "...: every feature is load-bearing in proportion to its share of the regularization budget, a rigorous form of
  the weight-magnitude ranking of SVM-RFE \citep[Sec.~2.5]{guyon2002gene}, which rests on the second-order
  saliency of \citet{lecun1989obd}."
- Fix the mccracken2025 bib entry (title and authors).
