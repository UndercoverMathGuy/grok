# R23-bottleneck: independent verification (lens: novelty_ml, task 21)

Verdict: PARTIALLY_KNOWN, which upholds the checker. Both citations the checker marked as confirmed survive. I found three ML-side analogues that the checker missed. None of them states our result.

## What I checked (all fetched this session)
- cyclic.tex prop:bottleneck and the proof in appendix.tex app:shape. The proof is correct: at a vertex of P_p, the only constraints that bind coordinates in S are coverage rows, so those rows restricted to S have rank K. Also cov(j, delta) = 1 - cos(2 pi j delta / p) >= 0, so the s = 1 game value is positive.
- Morwani et al., arXiv 2311.07568v2 (PDF). The Lemma 6 bullet reads verbatim: "For any (x,y) in spt(q*), f(theta_hat,x,y) - max_{y'} f(theta_hat,x,y') = gamma*, i.e., all points in the support of q* are on the margin for any maximum margin solution." CONFIRMED.
  - Refinement: the closer dense analogue of "hardest wrong answers" is condition C.1 plus App. F.2. There, for the L_{2,3} construction, "for each input (a,b), f takes the same value on all incorrect labels c'". So all p-1 wrong answers tie in the dense case, against our >= 2K in the sparse case. Li et al. 2402.09469 has the same all-on-margin statement for the k-input case.
- Shapley and Snow. Crossref for DOI 10.1515/9781400881727-004 gives "3. BASIC SOLUTIONS OF DISCRETE GAMES", Shapley, L. S. and Snow, R. N., Contributions to the Theory of Games (AM-24) Vol I, pp. 27-36, issued 1951-12-31.
  - The statement is confirmed through Attia and Oliu-Barton, arXiv 1810.08798v2. Thm 2.3.1 characterises basic solutions through a square sub-game with G'y' = val 1. Prop 2.3.4(iii) gives Ker(G' - vU') = <y'>, which is uniqueness. Their bibliography entry reads "Contributions to the Theory of Games 1 (1950), 27-35".
  - The bib entry's 1950 / 27-35 therefore matches common usage but not the DOI record (1951 / 27-36). This is minor. CONFIRMED.

## Adversarial ML search (arXiv API + direct PDF reads)
- I listed about 60 of the most recent arXiv abstracts with "modular addition", back to 2024-02. I read or grepped the candidates closest to the margin theory: 2609.26679 (spectral theory of grokking, Sept 2026), 2608.04451, 2606.05863, 2511.23443 (sine MLPs, margin bounds), 2506.06489 (AGF), 2606.02993 and 2602.16849 (He et al.), 2410.01779 (Tian), 2407.12332, 2505.18266 and 2609.12259.
- None states that a sparse max-margin modular-addition network has at least K tight wrong-answer offsets, or that its amplitude ratios are fixed by an equalising square system. None mentions vertices, basic solutions or Shapley-Snow.
- ML/optimisation analogues the checker missed (all fetched):
  1. Breiman, "Prediction Games and Arcing Algorithms", Neural Computation 11(7):1493-1517, 1999; tech report PDF read. Def 2.2 frames minimum-margin maximisation over convex combinations of base predictors as a zero-sum matrix game whose value is given by the minimax theorem. Our s = 1 programme is exactly this game, with frequencies as base predictors and offsets as instances. This is the ML-native route to Shapley-Snow.
  2. Ge, Jiang and Ye, "A note on the complexity of Lp minimization", Math. Programming 129(2):285-299, 2011; Stanford preprint read. Thm 3: "The set of all basic feasible solutions of (1) or (2) is exactly the set of local minimizers", for min sum x_j^p with 0 < p < 1 over {Ax = b, x >= 0}. This is the sparse-recovery form of step (i) of prop:shapes with r = 1/s < 1. Caveat: it assumes a bounded feasible set, while P_p is unbounded, so it is an analogue and not a drop-in citation. The paper's own proof covers this case.
  3. Rosset, Zhu and Hastie, JMLR 5:941-973, 2004; PDF read. The l1-margin / boosting solution has "at most n (the number of observations) non-zero coefficients" (Sec. 5.1.1, App. B, Thm 3). The proof writes "|A| equations in |A| variables" for the nonzero coefficients. This is the classic ML form of "#features <= #binding constraints, and the nonzeros solve a square system".
- LPBoost (Demiriz, Bennett and Shawe-Taylor, Machine Learning 46:225-254, 2002): metadata only (Crossref). The full text was paywalled, so it is not recommended.

## Conclusion
- The linear-independence and uniqueness step is the LP basic-solution fact. At s = 1 it is Shapley-Snow applied to the game in Breiman 1999. For s > 1 it is the concave-budget / l_p-with-p<1 vertex property, cf. Ge-Jiang-Ye Thm 3.
- What is new in ML is the instantiation to modular addition and its reading: at least 2K simultaneously hardest wrong answers on every input, with amplitude ratios pinned by them. This sparse-regime counterpart to Morwani's "all wrong answers tie" was not found anywhere.

## Recommended wording tweak
Keep the checker's sentence, but:
- say "extreme (basic) optimal strategies" at s = 1, because at s = 1 a maximizer need not be a vertex;
- optionally add "(the minimum-margin game of \citet{breiman1999prediction})";
- add "cf. \citet{ge2011note} for the analogous vertex property of l_p minimisation, p<1".
