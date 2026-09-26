# VERIFY (novelty_ml): R37-appendix-remarks, task 39

Verdict: RELATED_ONLY, upheld. Both citations the checker confirmed are confirmed again.

## Search conditions (this matters for how much weight to put on the verdict)
- WebSearch was unavailable. The session-wide budget was already used up ("200 of 200 WebSearch calls").
- The arXiv API (export.arxiv.org) returned HTTP 429.
- The Semantic Scholar API returned HTTP 429.
- DuckDuckGo HTML returned a CAPTCHA.
- So the adversarial search was limited to fetching the likely prior-art papers directly.

## Primary sources fetched in this session
1. Neyshabur, Tomioka, Srebro, "Norm-Based Capacity Control in Neural Networks", COLT 2015, PMLR 40:1376-1401.
   - The PMLR page matches the bib exactly.
   - The arXiv 1503.00036v2 PDF was read as extracted text. Verbatim:
     - Sec. 3.1: "the dependence on the width H disappears, and the bound depends only on the magnitude, as long as q <= p* (i.e. 1/p + 1/q >= 1)".
     - Thm 3 proof: "recursively replacing the top unit with H copies of it and adding an averaging unit on top of that ... this operation does not change the output of the network ... we have reduced the norm by factor H^{1/q-1/p*}".
     - Also: "we can shatter an arbitrarily large number of points with arbitrarily low gamma by using enough hidden units".
   - The mechanism is the same as ours. The construction is not: theirs uses ReLU, depth >= 3, an averaging layer and mu_{p,q}. Ours rescales H copies of one homogeneous neuron by H^{-1/nu} in a single layer.
   - Suggested wording: "the same splitting mechanism as Neyshabur et al. (2015, Sec. 3.2)". Avoid "their argument".
2. Savarese, Evron, Soudry, Srebro, "How do infinite width bounded norm networks look in function space?", COLT 2019, PMLR 99:2667-2690.
   - The PMLR page matches the bib.
   - Its content is the representation cost of 2-layer ReLU networks. It does not bear on the b>nu or b<2 remarks.
   - The citation is optional, as the checker said.
3. Morwani et al. (arXiv 2311.07568, HTML), App. E.1, verbatim: "the analysis in above proof can be used to compute optimal norms for b != nu as well - however, for any such b we would not get the same flexibility to build a theta* satisfying Equation [1]".
   - This is the closest prior remark on b != nu.
   - It says nothing about b>nu being unbounded in width or about b<2.
   - Thm 7 gives the L_{2,3} max margin of the quadratic (power, n=2) network with all frequencies.
4. Li, Liang, Shi, Song, Zhou (arXiv 2402.09469, HTML; AISTATS 2025).
   - Uses only L_{2,k+1}, with b = homogeneity.
   - Makes no remark on other exponents, width-unboundedness or splitting.
   - Does not consider sparse or equal-share restrictions on K < (p-1)/2 frequencies. Its Thm 4.1 uses every frequency.

## Remark-by-remark (ML lens)
- Equal-share bound Gamma_s >= max_K K^{-s} M_K. No ML precedent found. Morwani and Li use only the dense endpoint (all frequencies). This is internal.
- Only the interval patterns are needed in (JF). Internal.
- The factor counts operand factors, not degree. Internal. No precedent in Li et al.
- Power-net interference-free population facts (6 neurons, 0.408; none with tied embeddings). Internal computations. No precedent in Morwani or Li.
  - Internal consistency flag, not a literature issue. App. B says Morwani's power network attains the budget at n=2, b=3 (Cor. dense). App. conjecture says "no interference-free power network is known to attain the class bound". State that the b=3 attainment is by a network carrying interference, or restrict the sentence to b<nu. The source already has a TODO next to it.
- b>nu gives a margin unbounded in width. This is a trivial specialisation of Neyshabur et al. 2015. Cite it as above.
- b<2 breaks superadditivity by |F|^{1-b/2}. Elementary. No citation needed.

## Recommendation
Adopt the checker's change, with the softer wording "same splitting mechanism as". Savarese is optional. No novelty threat was found.
