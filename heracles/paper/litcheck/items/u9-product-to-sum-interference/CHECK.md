# U9: product-to-sum interference (cyclic.tex, "Interference at the feature's own frequency")

Verdict: CITATION_FOUND

The paragraph currently cites nothing. It says that product neurons produce sign-flipped copies such as cos(2pi j(a-b)/p), which do not depend on the label, and that several neurons have to cancel them.

## Sources (all fetched 2026-09-25)

1. **Gromov, "Grokking modular arithmetic", arXiv 2301.02679 (2023)** (bib key gromov2023grokking, already in refs.bib). Sec. 3.1:
   - Eq. (10): the squared first-layer activation expands to a constant, plus cos(2pi k 2n/p + 2phi1), plus cos(2pi k 2m/p + 2phi2), plus cos(2pi k(n+m)/p + phi1+phi2), plus **cos(2pi k(n-m)/p + phi1-phi2)**.
   - Eq. (11): the second-layer preactivation contains the terms cos(2pi k(n-m-q)/p + ...) and cos(2pi k(n-m+q)/p + ...). These are exactly our beta_{j,T}, T = {2}, copies.
   - Eq. (12): phi1 + phi2 = phi3 makes the (n+m-q) term add up constructively.
   - Eqs. (14)-(15): "we need to ensure that all other waves (i.e. all terms, but the third term in (11)) interfere destructively". This holds when the phases are random, since sum_k cos(2pi k s/p + phi_k) << N. The same passage says that greater width means "better wave interference".
   - This is the closest precedent: the same term, and the same idea that neurons cancel it across the width. His cancellation is only approximate (random phases, MSE loss). Ours is an exact condition (Def. if) and a rigid per-neuron sign rule (Lemma sign).
2. **Morwani et al., ICLR 2024, arXiv 2311.07568** (morwani2024feature). App. F.3: a single neuron's DFT is nonzero only when j = 0, k = 0, or "j,k,l = +-zeta". That set includes the mixed-sign (zeta, -zeta, ...) modes. The max-margin f must have f_hat supported only on j = k = -l (plus l = 0). App. F.2 builds 8 neurons per frequency with paired phase choices, so the sum computes exactly cos(a+b-c) and the other terms cancel. This implicitly requires the network to be interference-free. They never name the (a-b) term.
3. **He, Wang, Chen, Yang, arXiv 2602.16849 (Feb 2026)**. Sec. 4, Eq. (4.1): a quadratic neuron's output is the signal cos(omega(x+y-j)) times cos(omega(x-y)/2)^2 plus "residual noise". Def. 4.1 (full diversification) sets phase sums to zero, and Prop. 4.2 shows that the diversified neurons cancel that noise. This is the same mechanism in a 2026 paper. It is not in refs.bib; the ml sweep already lists it.
4. **Nanda et al., ICLR 2023** (nanda2023progress). I checked the PDF text. It describes constructive and destructive interference of cos waves in the logits, but I found **no** discussion of the (a-b) product-to-sum term. Do not cite it for this sentence.

## Recommendation
After "...interference at the feature's own frequency.", add:
"These copies are visible in the explicit solutions of \citet{gromov2023grokking} (Eqs. 10-11), where they are cancelled only approximately, by random phases. They are also implicit in the exact max-margin construction of \citet{morwani2024feature} (App. F.2-F.3) and in the phase-diversification condition of \citet{he2026mechanism} (Def. 4.1)."
Add a bib entry for He et al. 2602.16849.
