# R37 appendix remarks: verification, novelty_math lens (task 40)

Verdict: the checker's RELATED_ONLY holds. Its two confirmed citations check out. It missed one classical mathematical precedent: Slepian's group codes, which cover the Equal-shares paragraph.

## Citations the checker confirmed (re-fetched this session)
- **neyshabur2015norm.** PMLR v40 page fetched: title, authors, "Proceedings of The 28th Conference on Learning Theory", vol. 40, pp. 1376-1401, 2015. All correct. I extracted the full text of arXiv 1503.00036v2 and checked the quotes verbatim:
  - Sec. 3.1: "the dependence on the width H disappears ... as long as q <= p* (i.e. 1/p + 1/q >= 1)".
  - Sec. 3.2, Thm 3 proof: "recursively replacing the top unit with H copies of it and adding an averaging unit on top ... we have reduced the norm by factor H^{1/q-1/p*}".
  - Sec. 3.2: "we can shatter an arbitrarily large number of points with arbitrarily low gamma by using enough hidden units".
  - Caveat: their norm is the per-layer group norm mu_{p,q}, a product over layers, and the Thm 3 construction adds an averaging layer (d >= 3). The paper's norm is the per-neuron L_{2,b}. The mechanism is the same: duplicate and rescale units, so a q-sum above the homogeneity-matched exponent shrinks. At nu = 2, p = 2 the thresholds coincide (q <= p* = 2 matches b <= nu = 2). So write "cf. the unit-duplication argument of \citet{neyshabur2015norm}, Sec. 3.2", not "the argument of".
- **savarese2019infinite.** PMLR v99 page fetched: title, authors (Savarese, Evron, Soudry, Srebro), COLT 2019, vol. 99, pp. 2667-2690. All correct. The abstract is about infinite-width, bounded-Euclidean-norm ReLU networks and the minimal norm needed to represent f. It is relevant only to the ReLU paragraph, as the checker says.

## Sub-claims, maths lens
1. **Equal shares, M_K <= pK/(p-1).** MISSED PRECEDENT. With kappa_j(delta) = 1 - cos(2 pi j delta/p), put energy y_j in rotation plane j. The code X_delta = D(delta) X_0 is a cyclic group code (Slepian 1968), and ||X_delta - X_0||^2 = 2 sum_j y_j kappa_j(delta). So:
   - M_K(p) is half the squared minimum distance of the best equal-energy cyclic group code of p points with frequency set S.
   - The averaging bound is Slepian's Theorem 2: for a nontrivial real-irreducible orthogonal representation, sum_R d^2(X_R, X) = 2g for every unit initial vector X.
   - At s = 1 the programme is exactly Slepian's "initial vector problem" for Z_p. Its optimum is the regular simplex (Slepian also recalls that the regular simplex is optimal). This is the paper's Gamma_1(p) = p/(p-1), with the unique flat maximizer. That matters more for item r8-dense-corollary.
   - For s > 1 the l_{1/s} budget is not an energy constraint, so the sparse programme itself is not in Slepian.
   - Verified from the full BSTJ text (archive.org bstj47-4-575, pp. 575-602): the Sec. IV heading "THE INITIAL VECTOR PROBLEM", Theorem 2 on p. 591, and the conclusion (p. 596): "the outstanding problem is that of finding a tractable method of choosing the initial vector to maximize the nearest neighbor distance".
   - Also relevant: Biglieri and Elia, "Cyclic-group codes for the Gaussian channel", IEEE TIT 22(5):624-629, 1976, DOI 10.1109/tit.1976.1055598. It optimizes the frequency choice of cyclic group codes. Only its metadata could be fetched, so I do NOT recommend it as confirmed.
2. **Only interval patterns in (JF).** Internal to the proof. Nothing to find.
3. **The factor counts operand factors, not degree.** Internal.
4. **Power-network population (6 neurons, 0.408; no population with tied embeddings).** Internal numerics. Note that 0.408 = 6^{-1/2} to 3 decimals (0.40825), and 0.408/0.577 = 1/sqrt 2. If this is exact, say so, or state more digits. The n! coefficient on the multilinear monomial is the multinomial/polarization identity, which is standard.
5. **b > nu gives a margin unbounded in width.** H copies scaled by H^{-1/nu} cost H^{1-b/nu} -> 0. Equivalently, the l_r norm (r = b/nu > 1) of a spread vector is H^{1/r-1} -> 0. Elementary. The Neyshabur et al. analogy is apt (see the caveat above).
6. **b < 2, factor |F|^{1-b/2}.** This is the standard l_r versus l_1 norm-equivalence constant (power-mean inequality), attained by the flat vector. It is elementary and needs no citation. The arithmetic checks.

## Arithmetic checked
- sum_{delta != 0} kappa_j(delta) = p, so the average over the p-1 offsets is pK/(p-1).
- K^{-s} M_K <= p K^{1-s}/(p-1).
- 6^{-1/2} = 0.4082 and 3^{-1/2} = 0.5774.

## Recommended changes
- Accept the checker's Neyshabur sentence, softened to "cf.".
- Optional, one clause in the Equal-shares paragraph: "(M_K is half the squared minimum distance of the cyclic group code of \citet{slepian1968group} with frequency set S; the averaging bound is his Theorem 2, and at s=1 (b=nu) the programme is his initial-vector problem, solved by the regular simplex)".

## Operational note (for the workflow owner)
- WebSearch returned "this session has used its web search budget (200 of 200 WebSearch calls)" on my first query. The verifiers share one session-wide search cap.
- arXiv export, Semantic Scholar and Crossref (intermittently) returned HTTP 429.
- Parallel verifiers are exhausting and rate-limiting the shared web tools. This is a likely cause of verify-stage slowness and of degraded verification.
