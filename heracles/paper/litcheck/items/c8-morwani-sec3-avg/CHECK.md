# C8-morwani-sec3-avg: cited claim check

Location: approach.tex, Sec. "Averaged margins and Fourier amplitudes" (lines 7 and 11).
Citation: morwani2024feature (Morwani, Edelman, Oncescu, Zhao, Kakade, ICLR 2024, arXiv:2311.07568).

Verdict: SUPPORTED. Both pointers are correct. One optional precision fix is below.

## Sources read
- arXiv HTML https://arxiv.org/html/2311.07568 (v2, 19 Feb 2024): the section structure and the quote below.
- arXiv source tarball, files 3-theoretical-approach-arxiv.tex and A-cyclic.tex (in the scratchpad).

## Claim 1: "minimum margin is at most any averaged class-weighted margin [Section 3]"
- Morwani Sec. 3 is "Theoretical Approach", and Sec. 3.2 is "Multi-Class Classification".
- Sec. 3.2 defines the class-weighted margin g'(θ,x,y) = f[y] - Σ_{y'≠y} τ(x,y)[y'] f[y'].
- It then states: "Note that g′(θ,x,y) ≥ g(θ,x,y) as g′ replaces the max by a weighted sum."
- Sec. 3 also writes γ* = max_θ min_{q∈P(D)} E_q[g].
- So min_D g ≤ E_q[g] ≤ E_q[g'] for any q and any τ. That is exactly our sentence, and the direction is right.
- Our choice of uniform q with τ as a point mass on y-δ is an admissible τ, since τ only has to be a distribution over the incorrect labels.
- Morwani themselves use uniform q* and uniform τ (Sec. 3.3). Our point-mass τ is our own choice, which is fine.

## Claim 2: "network-level analogue of the quantity η_{u,v,w}(δ) of Morwani"
- The quantity is defined in Morwani App. F ("Proofs for cyclic groups", proof of Thm 7, the subsection proving Eq. theta*-g') as η_{u,v,w}(δ) := E_{a,b}[(u(a)+v(b))^2 w(a+b-δ)].
- It is a single quadratic neuron's mean logit at the candidate δ below the label. Our η_θ(δ) = E_x[f(θ,x)[y(x)-δ]] is the sum over neurons of the same quantity, extended to n operands. So "network-level analogue" is accurate.
- The notation is correct. Note that η does not appear in the main text; it is only in App. F.

## Recommended change (optional precision)
- Line 7: change "\citep[Section~3]{morwani2024feature}" to "\citep[Section~3.2]{morwani2024feature}".
- Line 11: change "the quantity $\eta_{u,v,w}(\delta)$ of \citet{morwani2024feature}" to "the quantity $\eta_{u,v,w}(\delta)$ of \citet[Appendix~F]{morwani2024feature}", so readers can find it.
- The bib entry is correct.
