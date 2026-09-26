# VERIFY R4-programme, lens novelty_math (task 2)

## Verdict
The checker's PARTIALLY_KNOWN verdict holds. One of its bib entries needs a correction, and I add one citation (optimisation lens).

## Mathematical reformulation (mine, checked by hand)
By homogeneity, Gamma_s(p)^{-1/s} = min { sum_j y_j^{1/s} : sum_j y_j kappa_j(delta) >= 1 for all delta != 0, y >= 0 },
with kappa_j(delta) = 1 - cos(2 pi j delta / p) >= 0.
This is l_q quasi-norm minimisation (q = 1/s < 1) over a pointed covering polyhedron. It is the textbook "l_p for sparsity" problem of
Ge, Jiang & Ye, Math. Program. 129:285-299 (2011).
- I read the preprint PDF (web.stanford.edu/~yyye/lpmin_v14.pdf) and Crossref confirms the metadata.
- Problem (1): min sum x_j^p subject to Ax=b, x>=0, with 0<p<1.
- Theorem 3: "The set of all basic feasible solutions of (1) or (2) is exactly the set of local minimizers."
- They also prove that global minimisation is strongly NP-hard.

Ours is their form after adding slacks. The objective runs over the y block only, which is still a concave function on a polyhedron, so vertex optimality carries over (see the r22 item).

So the mechanism "concave budget makes the optimum sparse" is a standard optimisation fact. It must not be presented as new. What is new is the specific circulant cosine-coverage instance, the fact that it upper-bounds every L_{2,b} network (Thm universal), and its solution (K* = Theta(log p), shapes).

Graph view (checker's point 2, upheld): sum_j y_j kappa_j(delta) is the delta-th Laplacian eigenvalue of the weighted Cayley graph Cay(Z_p, {+-j}).
- At s=1, the problem is Boyd's weight-budgeted max-algebraic-connectivity problem on K_p. Group averaging gives a circulant optimum.
- As s -> infinity, it is the Ghosh-Boyd cardinality problem.

Chowla cosine problem: a different extremum (the most negative value of an unweighted cosine sum), as the checker and items t9/u6 say. No overlap.

## Checker citations, re-fetched this session
- gunasekar2018implicit: arXiv 1806.00468 abstract fetched. It says GD on depth-L linear convnets converges to "a linear predictor related to the l_{2/L} bridge penalty in the frequency domain". Supports the cited use. The NeurIPS 2018 vol. 31 entry is fine. I did not re-read Thm 2a.
- kam2026symmetry: arXiv 2609.17926 exists (Kam, Bessafi, Cadet, submitted 15 Sep 2026). The HTML App. A.3.2 has Delta_S(beta) = min_{s!=0} sum_k beta_k (1 - cos(omega_k s)) with ||beta||_2 = B. Prop. 5 says the uniform allocation is optimal. Confirmed.
  - In our notation their budget is s = 1/2, which is outside our range [1, nu/2]. That explains why their optimum is dense.
  - The bib title case differs from arXiv ("Symmetry without a manifold: intrinsic dimension on orbits"). Harmless.
- boyd2006convex: Boyd's page confirms Proc. ICM 2006, vol. 3, pp. 1311-1319, and lists "maximizing algebraic connectivity" among its applications. Confirmed.
  - Wording: Boyd does not treat circulants explicitly. Say "a special case of", not "its circulant case".
- ghosh2006growing: exists. Crossref: Ghosh & Boyd, "Growing Well-connected Graphs", Proc. 45th IEEE CDC, 2006, DOI 10.1109/CDC.2006.377282.
  - **Pages are 6605-6611, not 6605-6610.** Fix the bib and add the DOI.
- jagadeesan2022inductive: arXiv 2102.12238 confirmed as COLT 2022. The abstract gives nuclear norm and group-sparse norms on Fourier coefficients. Fine as an optional citation.

## Extra citation
- ge2011note (DOI 10.1007/s10107-011-0470-2): cite at eq:programme or in the vertex/shape proposition. Item r22 also recommends it; place it once.

## Could not do (harness limits)
- The WebSearch budget for this session was exhausted (200/200), and the arXiv API, the Semantic Scholar API and dblp all returned 429 or an access-denied page.
- So I could not run fresh keyword searches for sparse algebraic-connectivity designs with an l_p (p<1) budget, E-optimal design with concave costs, or MathOverflow.
- Residual risk: a control or graph-design paper may use an l_p (p<1) edge-weight budget for lambda_2. It would be generic (arbitrary graphs), not the circulant instance or its log p solution.

## Harness note (for the user's slowness question)
The verify phase runs at effort 'xhigh' over 98 task files. At the time of writing, only 16 verify_*.json files existed. The shared WebSearch cap is used up, so verifiers fall back to WebFetch and scholarly APIs. Those APIs are rate-limiting the run (429 from arXiv export, Semantic Scholar and dblp). Agents lose time on retries and blocked fetches.
