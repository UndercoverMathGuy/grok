# R4-programme: independent verification (lens: novelty_ml)

Verdict: the checker's PARTIALLY_KNOWN stands. The programme's shape has close ML precedents. The specific programme was not found in the ML literature: a cosine-coverage max-min over Z_p under an l_{1/s} budget with s = nu/b, used as a bound on the margin of every network, with K* as the argmax support. A reviewer will also point to prior art on the exponent/AM-GM step, which the checker missed.

## Checker's confirmed citations (each re-fetched this session)
- gunasekar2018implicit: CONFIRMED. The arXiv 1806.00468v2 PDF text was read.
  - Thm 2 covers depth two. Thm 2a covers any depth L: the limit direction is "a scaling of a first order stationary point of ... min ||beta_hat||_{2/L} s.t. forall n, y_n<beta,x_n> >= 1".
  - The paper states: "When L > 2, and thus p = 2/L < 1, problem (10) is non-convex and intractable Ge et al. [2011]."
  - The bib fields are fine.
- kam2026symmetry: CONFIRMED. arXiv 2609.17926, submitted 15 Sep 2026, "Symmetry without a manifold: intrinsic dimension on orbits", by Kam, Bessafi and Cadet.
  - HTML App. A.3.2, Eq. 15: Delta_S(beta) = min_{s!=0} sum_{k in S} beta_k(1-cos(omega_k s)), with ||beta||_2 = B.
  - Prop. 5: with S = {1..K_max}, the uniform allocation is optimal and attains B p / sqrt(2(p-1)). For proper subsets, uniform is not optimal.
  - There is no l_q budget with q<1. Treat it as concurrent work, as the checker says.
- boyd2006convex: CONFIRMED. The ICM 2006 PDF was read. Section 3 states: "maximize lambda_2 subject to w >= 0, 1^T w = 1", described as "allocating a fixed total edge weight ... to maximize the smallest positive Laplacian eigenvalue" (absolute algebraic connectivity).
  - Section 7 states that for an edge-transitive graph with 1^T w = 1, "an optimal solution is given by uniform edge weights". This is the symmetry route to the existence half of Cor. dense: Z_p^* acts transitively on the frequency classes. Uniqueness still needs the paper's own argument.
  - Vol. 3, pp. 1311-1319 is consistent with Boyd's page.
- ghosh2006growing: EXISTS and SAYS WHAT IS CLAIMED, but the PAGE RANGE IS WRONG.
  - The PDF has 7 pages. The printed page numbers run 6605 to 6611, and the last page is printed "6611". The bib should read pages={6605--6611}, not 6605--6610.
  - Eq. (2) is indeed "maximize lambda_2(L(E_base u E)) subject to |E| = k, E subset E_cand". Eq. (3) is its Boolean form, and the paper uses an SDP relaxation.
  - Header: 45th IEEE CDC, San Diego, Dec 13-15 2006.
- jagadeesan2022inductive: CONFIRMED. arXiv 2102.12238 abstract: "the induced regularizer for K=1 and K=D ... nuclear norm and the l_{2,1} group-sparse norm, respectively" (COLT 2022).

## Missed prior art (ML lens)
The sentence in approach.tex, "The exponent comes from homogeneity: ... total L_{2,b} cost is least when they are equal (AM-GM), so amplitude A costs A^{b/nu}", has direct precedents. At b=2 it is a known result.
- Tibshirani, "Equivalences Between Sparse Models and Neural Networks", note dated 15 Apr 2021 (PDF read).
  - A k-layer network with weight decay is "equivalent to an l_p-penalized regression problem, where p = 2/k < 1".
  - It uses the same AM-GM step: min_{prod a_l = c} sum a_l^2 = k|c|^{2/k}.
- Kolb, Weber, Bischl, Rugamer, ICLR 2025 (arXiv 2502.02496), Thm 1. Depth-D factorization with L2 on the factors has the same global and local minima as the L_{2/D} quasi-norm penalty. This is the peer-reviewed version.
- Jacobs and Montufar, arXiv 2605.19458 (May 2026), Thm 4.9.
  - The max-margin problem of homogeneous nets under mirror flow reduces to min sum_j |a_j|^{alpha/2}.
  - alpha=1 gives sparse solutions (fewer active neurons) and alpha>=2 gives dense ones.
  - This is the generic statement "the exponent of a per-unit cost decides sparse vs dense". The paper's recommended sentence "the concave budget that homogeneity forces is what makes ours sparse" should cite it and not present that mechanism as new.

## Searches (WebSearch was unavailable; see harness note), via arxiv.org/search and fetches
- abstract search "modular addition" margin (6 hits): nothing new. 2511.23443 (sine activation) has margin-based generalisation only.
- abstract search "modular addition" frequencies (14 hits): 2608.04451 (single-neuron Fourier alignment counterexample) has no budget programme. Other hits were covered by the 09-19 scoop check.
- "grokking margin frequencies", "grokking weight decay sparse Fourier features modular", "max-margin quasi-norm": 0 hits.
- "weight decay" "quasi-norm": 1 hit, 2510.21945 (Schatten-p for linear nets, rank not frequency).
- Tian 2410.01779 (CoGO) abstract: no frequency-count or budget programme.

Nothing in the ML literature states the coverage max-min under an l_{1/s} budget with q = b/nu < 1, or its K* = Theta(log p) solution.

## Harness note (relevant to the slow verify stage)
- WebSearch returned "this session has used its web search budget (200 of 200 WebSearch calls)", so every later verifier has no WebSearch.
- export.arxiv.org/api returned HTTP 429 (rate-limited by concurrent agents).
- Verifiers must fall back to WebFetch of arxiv.org/search, which is slower and gives shallower results. Raising CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION or throttling concurrency would help.
