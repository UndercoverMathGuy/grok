# R23-bottleneck: independent verification, novelty_math lens (task 22)

Verdict: the checker's PARTIALLY_KNOWN is upheld. The technique is textbook LP geometry together with a known theorem from concave (l_r, r<1) minimisation. The instantiation for P_p and the network reading ("2K simultaneously hardest wrong answers pin the amplitude ratios") were not found anywhere. I add one citation the checker missed, correct one gloss, and note one consistency point.

Note: this session's WebSearch budget was already exhausted when I started (200/200). All searching below used WebFetch against the arXiv API, the Crossref API and primary PDFs.

## What the proposition is, mathematically (proof re-read)
Proof location: appendix.tex, app:shape.
- A maximizer is Gamma_s(p) times a minimiser of phi(v) = sum_j v_j^r (r = 1/s < 1) over the polyhedron P_p = {v >= 0 : Cv >= 1}. It is a vertex by the strict-concavity midpoint argument.
- At a vertex the active constraints have rank |F|. The constraints v_j = 0 for j outside S only span coordinates outside S, so the tight coverage rows restricted to S have rank K. Choosing K of them gives an invertible square system whose unique solution is y_S.
- I checked the argument and it is correct. The step delta -> p - delta is valid because the chosen independent rows are distinct vectors.

## Known-fact search (maths lens)
1. **Concave l_r minimisation at vertices / basic solutions: KNOWN (checker missed this).**
   - Source: Ge, Jiang and Ye, "A note on the complexity of Lp minimization", Math. Program. 129(2):285-299, 2011, DOI 10.1007/s10107-011-0470-2. I fetched and read the PDF. Metadata confirmed via Crossref.
   - Theorem 3 says: "The set of all basic feasible solutions of (1) or (2) is exactly the set of local minimizers." Here (1) is min sum_j x_j^p over {Ax=b, x>=0} with 0<p<1.
   - The "local minimiser implies extreme point" half uses the same midpoint and strict-concavity argument as our Prop shapes (i).
   - Caveat: they assume a bounded feasible set and full-rank A. P_p is unbounded, but the argument is unchanged; our proof handles compactness through sublevel sets.
   - This is the closest direct statement of steps (a) and (b), and it is closer than Rockafellar Sec. 32.
2. **Vertex implies K independent tight rows implies unique square solution.** This is the definition of a basic solution in LP. It is textbook and needs no special citation.
3. **Shapley-Snow (s = 1): CONFIRMED.**
   - I read Thm 2.3.1 and Prop 2.3.4 in the Attia and Oliu-Barton PDF (arXiv 1810.08798).
   - Thm 2.3.1 says basic solutions correspond to square kernels with x'^T G' = val 1 and G' y' = val 1.
   - Prop 2.3.4(iii) gives Ker(G' - vU') = <y'>.
   - Nonsingularity holds when val > 0 via det = val * S(co). Gamma_1 > 0, so it holds here.
   - Mapping: our y is the frequency player's strategy. The kernel columns are tight offsets, and y solves the equalising system on them. This matches.
   - Crossref gives the book as "issued 1951-12-31" (a placeholder date) with pages 27-36. Attia and Oliu-Barton cite "(1950), 27-35". The checker's bibtex (1950, 27--35, editors Kuhn and Tucker, DOI) is acceptable. The editors were confirmed from the Crossref book record.
4. **Chebyshev equioscillation: analogy only (agreed).**
   - Goldsztejn arXiv 2310.01851 exists. It is a survey of optimality conditions for multivariate Chebyshev approximation: Kolmogorov, Rivlin-Shapiro, Kirchberger.
   - Our coverage system is not Haar, and our count is a lower bound that comes from vertex rank.
   - A closer classical analogue would be the LP (Stiefel exchange / "reference set") formulation of discrete linear Chebyshev approximation. I did not fetch a source for it, so I do not recommend citing it.
5. **Programme-specific searches found nothing matching.**
   - Queries: arXiv API on "modular addition" + margin (2020-2026); "modular addition" + frequencies + weight decay/sparse (2024-2026); "cosine sum" + minimum + sparse.
   - The hits are Morwani 2311.07568, Li et al. 2402.09469 (dense L_{2,k+1}), He et al. 2602.16849, Discrete-Log Clock 2606.17399, and others.
   - None characterise the optimal frequency set by vertices, active constraints, or equalising equations. None say that the number of tight wrong-label offsets is at least the number of features.

## Corrections to the checker
- **Morwani et al. (ICLR 2024), Lemma 6.** The quote is verbatim and correct. However, Lemma 6 concerns *data points* (x,y) in spt(q*) being on the margin, not wrong labels. The statement that every wrong class is equally hard is made explicitly in the proof of Theorem 7 in their Appendix F: "we know by Lemma 6 that the function f implemented by the network has equal margin across different inputs and across different classes for the same input". Cite it as "Lemma 6 and the proof of Theorem 7".
- **s = 1 is the dense endpoint, and it lines up with Morwani.**
  - F = {1..(p-1)/2}. At s = 1 the average coverage over offsets is p/(p-1), so the unique maximizer is uniform y with Gamma_1(p) = p/(p-1).
  - Every one of the (p-1)/2 offsets is tight. I verified numerically that the (p-1)/2 x (p-1)/2 coverage matrix has full rank for p = 5, 7, 11, 13, 23, 113, so K = (p-1)/2.
  - The s -> 1 instance of the bottleneck statement is therefore exactly "all wrong answers equally hard" (Morwani). s > 1 cuts this down to at least K of them.
  - The checker's phrase "At s=1 it is exactly Shapley-Snow" is right, but the s = 1 kernel is the whole game. Say this, because it makes the Morwani contrast precise rather than a loose analogy.

## Recommended wording (revised)
After prop:bottleneck or in app:shape, write:

"This is the basic-solution characterisation of a vertex. Minimisers of a separable concave l_r cost (r<1) over a polyhedron are basic feasible solutions \citep[Thm 3]{ge2011note}. At s=1 the programme is the matrix game with payoff cov(j,delta), and the proposition is the Shapley-Snow kernel of its (unique, uniform) optimal strategy \citep{shapley1950basic}: every offset is tight, the dense picture of \citet[Lemma 6, App. F]{morwani2024feature}. For s>1 the kernel shrinks to K offsets."

Claim as new only the reading: at least K equally hard wrong answers fix the amplitude ratios, and exactly K is observed at p <= 23.
