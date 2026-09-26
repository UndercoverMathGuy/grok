# R24-ablation: novelty_math verification (task 24)

Claim (cyclic.tex, prop:ablation): if y is feasible for the programme with sum_j y_j^{1/s} = 1 and worst coverage
m >= (1-eps) Gamma_s(p), and w_j = y_j^{1/s}, then m - 2 y_j <= m_{-j} <= (1-w_j)^s m/(1-eps). So removing j from a
maximizer costs at least the fraction 1-(1-w_j)^s of the margin, and this fraction lies in [w_j, s w_j].

## Verdict: UPHELD. NOVEL_AS_FAR_AS_FOUND as a stated result, but the technique is textbook.

### Is the proof correct? (re-derived)
- Lower bound: 0 <= cov_j(delta) = 1 - cos(.) <= 2, so each offset loses at most 2 y_j. Correct.
- Upper bound: F(y) = min_delta sum_j y_j cov_j(delta) is positively homogeneous of degree 1, and
  Q(y) = (sum_j y_j^{1/s})^s is positively homogeneous of degree 1. So F(y) <= Gamma_s(p) Q(y) for every y >= 0.
  This is the standard gauge/value-function inequality. It needs no convexity of Q, so the quasi-norm case s > 1 is
  fine. Q(y_{-j}) = (1-w_j)^s, which gives the bound. Correct, including the w_j = 1 case.
- Interval: (1-w)^s <= 1-w for s >= 1, and Bernoulli gives (1-w)^s >= 1 - s w. Correct.
- Extra remark worth knowing: at a maximizer, KKT plus Euler's identity for the degree-1 homogeneous F give
  y_j g_j = w_j m for the KKT supergradient g. So the first-order ablation loss is exactly w_j m, which is the lower
  end of the interval (the "Euler allocation" of a homogeneous objective). Concavity alone gives only
  m_{-j} <= (1-w_j) m. The homogeneity bound (1-w_j)^s m is strictly sharper for s > 1. This is a remark, not a
  required citation.

### Adversarial search, mathematics lens (assume it is known and try to find it)
- It is a one-line consequence of positive homogeneity plus Bernoulli's inequality, so it is folklore-level as a
  technique. I found no source that states a share-to-fractional-loss bound for a max-min coverage programme under
  an l_{1/s} (concave) budget.
- Loose analogues considered: (i) frame erasures (Parseval frame: deleting f_j lowers the lower frame bound by at
  most ||f_j||^2); (ii) matrix games (deleting a strategy used by every optimal strategy strictly lowers the value);
  (iii) Euler/Aumann-Shapley allocation for homogeneous functions. None of these is the proposition, and none
  needs citing. I fetched no primary page for them, so I cite none.
- arXiv API queries (the WebSearch budget was exhausted this session, so I queried the export.arxiv.org API
  directly): "Parseval frame" AND erasure; ti:"optimal frames for erasures"; removing AND neuron AND margin AND
  homogeneous; "modular addition" AND ablation AND margin; "modular addition" AND margin; "max-margin" AND Fourier
  AND "weight decay"; "matrix game" AND "deleting a strategy". Nothing relevant came back.
- StackExchange API (MathOverflow and Math.SE): "maximin allocation remove one coordinate homogeneous", "value of
  matrix game after deleting a strategy", "lower frame bound after erasure of one vector". Nothing relevant came back.
- Full-text scans (arXiv HTML) for ablation or removal statements in the closest theory papers found none in:
  Morwani et al. 2311.07568v2; Li-Liang-Shi-Song 2402.09469v4; Huang-Li 2511.23443; Tian 2410.01779; Gromov
  2301.02679. McCracken et al. 2505.18266v1 has only empirical ablations (G.8).

### Checker's confirmed citations, re-verified this session
1. nanda2023progress: VERIFIED from the PDF text. The header reads "Published as a conference paper at ICLR 2023",
   with authors Nanda, Chan, Lieberum, Smith, Steinhardt. Sec. 4.4 is "Correctness checks: ablations". Its
   logit-frequencies paragraph says "ablating any key frequency causes a significant increase in loss ...
   ablating other frequencies does not hurt the model at all." Fig. 6 caption: "The loss of the transformer
   ... when ablating each frequency k in {1,...,56}". The intro says "(4) ablating key frequencies used by the
   model reduces performance to chance, while ablating the other 95% of frequencies slightly improves
   performance." Sec. 5.1 defines restricted and excluded loss. The ablation is on the 2D-DFT components of the
   logit table, which is the same operation as removing y_j cos(2 pi j(a+b-c)/p) in the block network. That
   makes the comparison apt.
2. chughtai2023toy: VERIFIED on the PMLR page. Title "A Toy Model of Universality: Reverse Engineering how
   Networks Learn Group Operations", authors Chughtai, Chan, Nanda, PMLR 202:6243-6267. The abstract says they
   "confirm our understanding using ablations". The refs.bib entry is fine but could add volume and pages.
3. mccracken2025: VERIFIED on the arXiv abs page and HTML v1 (only v1 exists, no journal-ref). Title
   "Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks", authors Gavin McCracken,
   Gabriela Moisescu-Pareja, Vincent Letourneau, Doina Precup, Jonathan Love. App. G.8 "Noise and ablation
   studies" and Fig. 38 ("Ablation study results ... randomly removing neurons ... from a cluster") are present.
   Thm 4.7 and Cor 4.8 concern the margin versus the number of frequencies (O(log n)), not removing one frequency.

## Issues
- BIB ERROR (must fix): refs.bib `mccracken2025` currently has title "Emergence of sparse and dense circuits in
  modular arithmetic networks" and author "McCracken, Andrew and others". Both are wrong for arXiv:2505.18266.
  Use the checker's corrected entry. This key is also cited in intro.tex.
- Wording: Nanda's ablations are on a trained transformer and measure cross-entropy loss. The proposition
  concerns (near-)maximizers of the programme and worst-case coverage. Say "an idealized (max-margin)
  counterpart" or "a quantitative analogue" rather than "the theoretical counterpart". The proposition is also
  silent on Nanda's observation that ablating non-key frequencies slightly helps. That is trivial here, since
  m_{-j} = m for j outside the support.
- Presentation: the checker's advice holds. Frame this as an elementary consequence of homogeneity, not a
  headline result.
