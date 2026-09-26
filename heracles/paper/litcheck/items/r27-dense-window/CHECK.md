# R27 dense-window (cyclic.tex, final paragraph)

Claim: (a) a maximizer of Gamma_s(p) can use all (p-1)/2 frequencies only if
s-1 <= ln4 / ln((p-1)/(2 ceil(8 ln p))) = O(1/log p);
(b) exact dense thresholds s = 1.548, 1.345, 1.285, 1.217, 1.194, 1.160 at p = 7, 11, 13, 17, 19, 23.

## Verdict: NOVEL_AS_FAR_AS_FOUND (as a statement). The closest context is McCracken et al. 2025, which is an observation with no theory.

## Internal checks
1. Numerics REPRODUCED independently with dense_thresh.py (this directory).
   - Method: enumerate every vertex of P_p = {v >= 0 : sum_j v_j (1 - cos(2 pi j delta/p)) >= 1 for all delta}.
     For each support S and each set T of tight constraints with |T| = |S|, solve the square system, then keep the solution if it is positive and feasible.
   - Then bisect for the largest s at which a full-support vertex minimizes sum v^{1/s}.
   - Results: p=7 1.5475, p=11 1.3454, p=13 1.2850, p=17 1.2166, p=19 1.1938, p=23 1.1597. All six match the paper.
   - The only full-support vertex is always the uniform one, v = 2/p.
   - Vertex counts: 7, 31, 87, 391, 961, 4951.
   - The p=13 threshold equals the 6->4 switch at s=1.285 given in the shapes paragraph, so the two are consistent.
   - Caveat for the notes: an earlier version of my script compared the exact uniform vertex against vertices rounded to 1e-10. That spuriously reported "not dense" at p=13 and 17. The fix was to take the dense candidate from the enumerated set as well. Any rerun should use a consistent tolerance.
2. The analytic bound (a) is correct but vacuous until p >= 73. Its value is:
   | p | bound on s-1 |
   |---|---|
   | 73 | 49.2 |
   | 113 | 3.58 |
   | 1009 | 0.63 |
   | 10007 | 0.33 |
   | 10^5 | 0.22 |
   | 10^6 | 0.165 |

   So at every p where it is exactly computed (p <= 23) the analytic bound says nothing, and it only reaches the p=23 exact value (0.160) near p ~ 10^6. At weight decay with nu=3 (s-1 = 0.5), the bound forces sparsity only for p > ~2000. The claim in the paper is asymptotic and correctly worded as O(1/log p). Still, a reader could read "the range shrinks as p grows" as proven monotonicity, and it is not proven. It is an upper envelope plus six exact points.
3. Derivation. The inequality uses the proof-internal share floor w_j >= (m/2)^{1/(s-1)} >= 4^{-1/(s-1)}/ceil(8 ln p), with m >= (1/2) ceil(8 ln p)^{1-s}. The stated theorem floor is 4^{-1/(s-1)}/(9 ln p), which would give ln4/ln((p-1)/(18 ln p)), a slightly weaker bound. The text says "by the share floor of Theorem", but it uses the sharper, unstated constant. This is harmless: either state the sharper floor, or write 18 ln p.
4. (s-1) ln p at the exact thresholds is 1.07, 0.83, 0.73, 0.61, 0.57, 0.50. That is decreasing, so at small p the threshold falls faster than c/log p. This is consistent with the O(1/log p) upper bound, but it does not show the bound is tight. Tightness (a matching lower bound on the dense window) is not claimed and not proven.

## Literature
- McCracken, Moisescu-Pareja, Letourneau, Precup, Love, arXiv 2505.18266 (2025), "Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks". Fetched https://arxiv.org/html/2505.18266.
  - Empirical dense/sparse transition: one-hot one-layer MLPs learn floor(n/2) neuron types. "adding either depth, or a trainable embedding matrix for inputs, causes the network to transition from learning floor(n/2) types of neurons to much fewer types". Also: "we don't know why it occurs".
  - Cor 4.8: "Learning O(log n) distinct frequencies gives a logit margin Omega(log n)". This is a sufficiency result only.
  - Their transition is architectural (embedding/depth). Ours is in the norm exponent s. Our dense endpoint (s=1) and the O(1/log p) window give one candidate mechanism: a network whose effective price exponent is exactly 1 is dense, and anything strictly above 1 is sparse at large p. Cite it as the empirical phenomenon we give an account of, without claiming to explain their architecture axis. Their embedding/depth nets are not in our neuron-additive class; the paper already says shared linear layers are out of scope.
- Morwani et al. ICLR 2024 Thm 7 and Li et al. AISTATS 2025 Thm 4.1 prove density at b = nu, the s=1 endpoint (see item r8). They say nothing about s > 1.
- Kam, Bessafi, Cadet arXiv 2609.17926: a convex l2 budget gives a uniform (dense) optimum. That is a contrast point, not a threshold (see sweep).
- Jacobs & Montufar arXiv 2605.19458 Thm 4.9: the homogeneity of the potential decides sparse vs dense in mirror flow. The spirit is related ("exponent decides"), but the setting and quantities differ, and they give no p-dependent window.
- lp-quasi-norm (0<q<1) minimisation over polyhedra: minimisers are vertices (Ge, Jiang, Ye 2011; see item r22). I found no result giving a q -> 1 window for full-support optimality as a function of dimension for cosine coverage constraints.
- Searches run: "modular addition number of frequencies dense to sparse transition weight decay exponent log p max margin"; "lp quasi-norm minimization q<1 ... threshold q close to 1 dense". I also checked arXiv 2606.17399 (modular multiplication spectra) and 2608.04451 (MAIS-O60 counterexample). Neither is relevant.

## Recommended change
- Cite McCracken et al. here, e.g. "... shrinks as p grows, consistent with the observation of McCracken et al. (2025) that dense solutions (all floor(p/2) frequencies) are fragile to changes of architecture."
- Add "(the bound is informative only for p >= 73; below that the exact values are the evidence)".
- Say "is at most" rather than implying the window provably shrinks monotonically.
- Align the constant with the theorem statement: use 18 ln p, or state the floor 4^{-1/(s-1)}/ceil(8 ln p) in the theorem.
- Fix the mccracken2025 bib entry (wrong title/authors; see sweep item 1).
