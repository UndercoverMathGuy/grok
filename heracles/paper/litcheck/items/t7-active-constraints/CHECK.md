# T7-active-constraints: CHECK

Item: appendix.tex line 166, proof of Prop. bottleneck (cyclic.tex:73): "At a vertex, the active constraints have rank |F_p| ... coverage rows for delta in D, restricted to S, have rank K."

## Verdict: KNOWN (a textbook LP fact; the proof step is correct)

- This is the definition and characterisation of a basic feasible solution. A point x of a polyhedron in R^n is a vertex/extreme point iff it is a basic feasible solution, meaning that among the active constraints there are n linearly independent ones. That is Bertsimas & Tsitsiklis, Introduction to Linear Optimization (Athena Scientific, 1997), Def. 2.9 (p. 50) and Thm 2.3.
  - Confirmed: Wikipedia "Basic solution (linear programming)" cites B&T 1997, p. 50, ISBN 978-1-886529-19-9. arXiv 2302.12364 (fetched) says "basic feasible solutions are precisely extreme points (vertices) of the feasible set (Bertsimas and Tsitsiklis, 1997, Theorem 2.3)". Its bib gives Athena Scientific, 1st ed., 1997.
- Our step: P_p = {v >= 0 : sum_j v_j cov_j(delta) >= 1 for all delta}. At a vertex, n = |F_p| of the active constraints are independent. The active bounds v_j = 0 (j not in S) are unit vectors outside S. So the active coverage rows, restricted to S, must have rank K. This is just the standard "basic variables <-> invertible basis matrix B" argument (B&T Sec 2.3; the basis matrix of K columns is invertible, x_B = B^{-1} b).
- Minimax analogue: the programme is a max-min (discrete Chebyshev-type) problem. It is standard that a basic optimal solution of the LP form of discrete linear Chebyshev approximation is pinned by n+1 active "reference" points; with the Haar condition this becomes the alternation theorem. Here we get K tight offsets rather than K+1 because the normalisation constraint is moved into the scaling. The analogy is worth one phrase in the prose ("the analogue of a Chebyshev reference set"), but it does not need a citation. I did not fetch a canonical approximation-theory source (e.g. Cheney 1966), so it is unconfirmed and no bib entry is given.
- Nothing novel in the technique. What is ours is the interpretation: the tight offsets are simultaneous hardest wrong answers in the constructed network.

## Recommendation
Add one citation at "At a vertex, the active constraints have rank |F_p|": "(a vertex is a basic feasible solution; \citealp[Thm 2.3]{bertsimas1997})".

```bibtex
@book{bertsimas1997,
  author    = {Bertsimas, Dimitris and Tsitsiklis, John N.},
  title     = {Introduction to Linear Optimization},
  publisher = {Athena Scientific},
  address   = {Belmont, MA},
  year      = {1997},
  isbn      = {978-1-886529-19-9}
}
```
Sources: https://en.wikipedia.org/wiki/Basic_solution_(linear_programming) ; https://arxiv.org/pdf/2302.12364 (Sec. 2, p. 3-4)
