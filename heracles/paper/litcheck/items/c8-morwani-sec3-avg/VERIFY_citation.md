# VERIFY (citation lens): C8-morwani-sec3-avg, task 82

Source I checked: the arXiv LaTeX source of 2311.07568 v2 (last revised 19 Feb 2024), downloaded from https://arxiv.org/e-print/2311.07568, and the abs page (v1 13 Nov 2023, v2 19 Feb 2024). The OpenReview PDF (id i9wDX850jR) was behind a browser check and could not be fetched.

## Claim 1: "minimum margin is at most any averaged class-weighted margin [Section 3]"
- In 3-theoretical-approach-arxiv.tex, Section 3 "Theoretical Approach" defines g (line 31 on) and h = min_D g.
- The live subsections are 3.1 Binary Classification, 3.2 Multi-Class Classification and 3.3 Blueprint. The "Specialization to Neural Networks" block, lines 183-230, sits inside \iffalse and is not compiled.
- Sec 3.2, line 119: the class-weighted margin g' = f[y] - sum_{y' != y} tau(x,y)[y'] f[y'], with tau any distribution over the incorrect labels.
- Line 122, verbatim: "Note that g'(theta,x,y) >= g(theta,x,y) as g' replaces the max by a weighted sum."
- So h <= E_q g <= E_q g' for every q and every admissible tau. The direction is correct.
- Our choice (uniform q, point mass on y - delta with delta != 0) is an admissible tau.
- Verdict: SUPPORTED. "Section 3" is correct. "Section 3.2" is more precise.

## Claim 2: "network-level analogue of eta_{u,v,w}(delta)"
- A-cyclic.tex, line 7: \eta_{u,v,w}(\delta) := \E_{a,b}[(u(a)+v(b))^2 w(a+b-\delta)].
- It is defined inside the proof in the appendix "Proofs for cyclic groups (Theorem thm:cyclic)".
- Appendix order in main-arxiv.tex: A Further Related Work, B Experimental details, C Additional Experiments, D Alternative construction, E Proofs for the Theoretical Approach, F Proofs for cyclic groups. So eta is in Appendix F of arXiv v2, and it is not in the main text.
- Our eta_theta(delta) = E_x f(theta,x)[y(x) - delta] is the sum over neurons of the single-neuron logit at the label minus delta. It uses the same sign convention (a+b-delta = y-delta).
- Verdict: the notation is exact and "network-level analogue" is accurate.

## Bib
- Title, the five authors (Morwani, Edelman, Oncescu, Zhao, Kakade), ICLR 2024 and arXiv:2311.07568 are all correct.

## Caveat
- The appendix letter "F" is checked for arXiv v2 only. I could not fetch the ICLR camera-ready.
- v2 postdates acceptance and has an iclr/ style directory, so it is very likely the same, but this is not verified.
- If "Appendix F" is added, the bib should point to arXiv v2, or the reference should say "the proof of their cyclic-group theorem", which does not depend on the version.

## Result
The checker's verdict (SUPPORTED) is upheld. Its optional change (Section 3.2; Appendix F for eta) is accurate for arXiv v2.
