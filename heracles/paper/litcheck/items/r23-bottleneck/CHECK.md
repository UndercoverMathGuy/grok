# R23-bottleneck (cyclic.tex prop:bottleneck): literature check

Verdict: PARTIALLY_KNOWN. The mechanism is textbook. Only the instantiation for this programme and the network reading are ours.

## What the proposition is, mathematically
The proof (appendix.tex, app:shape) has three steps.
1. A maximizer equals Gamma_s(p) times a vertex of P_p. This comes from Prop shapes: a concave, coordinatewise non-decreasing phi is minimised over a line-free polyhedron.
2. The standard vertex characterisation: the active constraints have full rank, so K coverage rows are linearly independent on the support S.
3. y_S is the unique solution of the invertible KxK system. Then there is symmetry delta <-> -delta and the network gap identity.

Steps 1-2 are standard facts:
- A concave function over a polyhedron with no lines attains its minimum at an extreme point. This is Rockafellar, Convex Analysis, Sec. 32. I did not verify the exact theorem number online, so cite it as "Sec. 32" or state it as standard. The paper already proves it inline.
- A vertex of {Ax >= b} has n linearly independent tight constraints. This is basic LP geometry and a basic feasible solution.

## Closest prior art: Shapley-Snow basic solutions (1950/51)
At s = 1 (l1 budget), Gamma_1(p) is the value of the matrix game with payoff cov(j, delta). An extreme optimal strategy then corresponds to a square nonsingular sub-game, the Shapley-Snow kernel. On that sub-game the strategy is the unique solution of the equalising system.

Quote, taken from the restatement in Attia and Oliu-Barton, arXiv 1810.08798, Thm 2.3.1: "A couple (x,y) in X* x Y* is a basic solution of G if and only if there exists a square sub-game G' satisfying ... (4) x'^T G' = val(G) 1 and G' y' = val(G) 1". Prop 2.3.4(iii): "Ker(G' - vU') = <y'>", which is uniqueness.

The bottleneck proposition is the s > 1 analogue. The concave budget still forces a vertex of the same polyhedron P_p, so the same square equalising system appears. Metadata was confirmed through Crossref for DOI 10.1515/9781400881727-004: Shapley, L. S. and Snow, R. N., "Basic Solutions of Discrete Games", Contributions to the Theory of Games (AM-24) Vol. I, Princeton UP, pp. 27-36 (1951; often cited as 1950).

## Chebyshev equioscillation
The flavour is the same: a minimax optimum is pinned by at least (dimension) equally extremal points. Chebyshev alternation gives n+2 alternation points under the Haar condition. For general non-Haar systems, the Kolmogorov and Rivlin-Shapiro criteria give at most n+1 extremal points in the characterising set. The survey is Goldsztejn, arXiv 2310.01851.

Our statement is not a special case of these results. The coverage system is not Haar, the objective is a max-min under a concave budget rather than a best approximation, and our count is a lower bound (at least K tight offsets) that comes from vertex rank. So cite this only as an analogy, if at all.

## ML-side analogues
Morwani et al. ICLR 2024, Lemma 6: "all points in the support of q* are on the margin for any maximum margin solution". That is the dense-case analogue of "tight wrong answers". Their certificate puts mass on all wrong labels. Ours gives exactly K tight offset classes, up to +/-, pinned by vertex rank.

I found no ML paper stating that the number of tight wrong-label offsets is at least the number of features, or that amplitude ratios are determined by an equalising system. Searches covered modular addition max-margin with tight wrong labels and vertices, plus Shapley-Snow and Chebyshev characterisations.

## Recommendation
After the proposition (or in its proof), add one sentence: "This is the vertex (basic-solution) characterisation of linear programming. At s=1 it is the Shapley-Snow description of extreme optimal strategies of the matrix game with payoff cov(j,delta) \citep{shapley1950basic}. The concave budget s>1 keeps the optimum at a vertex (Prop. shapes), so the same square equalising system persists." Do not present the linear-independence and uniqueness part as a new technique. The novelty is the interpretation: at least K equally hard wrong answers, and amplitude ratios fixed by them. The empirical observation of exactly K offsets at p<=23 is also ours.
