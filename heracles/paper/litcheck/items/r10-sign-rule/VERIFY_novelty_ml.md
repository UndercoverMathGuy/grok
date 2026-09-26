# VERIFY (novelty_ml): R10-sign-rule (task 9)

Claim (cyclic.tex lem:sign, proof in app:models): for product and power neurons, beta_{ij,T} = beta_{ij} prod_{t in T} xi_{itj}, where each xi_{itj} is a unit complex number.

## Verdict
The checker's PARTIALLY_KNOWN stands. Its evidence and its attribution are incomplete: it found only He et al. 2026, but the same mechanism is in three earlier papers.

## Sources fetched this session (2026-09-25)
1. **He, Wang, Chen, Yang, arXiv 2602.16849** (abs and HTML).
   - Authors in full: Jianliang He, Leda Wang, Siyu Chen, Zhuoran Yang. Submitted 18 Feb 2026, arXiv only.
   - Eq. (4.1): a quadratic neuron's contribution is cos^2(w(x-y)/2) * { cos(w(x+y-j)) + 2cos(wj+2phi_m) + cos(w(x+y+j)+4phi_m) }. The checker quoted this correctly.
   - Def. 4.1(iii) is typeset literally as exp(i*iota*sum_{m in N_k} phi_m) = 0, which is a typo, because exp is never 0. The intended condition, sum_m exp(i iota phi_m) = 0, is the one used in the display with underbrace "=0 due to condition (iii)". Cite that display, or "Sec. 4", not the literal Def. 4.1(iii).
   - Right after Eq. 4.1, He et al. write: "Similar results have also been reported in Gromov (2023); Zhong et al. (2023); Nanda et al. (2023); Doshi et al. (2023)." So He et al. are not the original source of the per-neuron noise terms.
2. **Gromov 2023, arXiv 2301.02679** (PDF read).
   - Two-input quadratic net, weights cos(2 pi k n/p + phi_k^{(i)}), with the phase constraint (12) phi^(1) + phi^(2) = phi^(3).
   - The paper says (12) "leaves some phases in every single term in (11) apart from the third one." Those terms "interfere destructively" across neurons, see (14)-(15), under random phases for large N.
   - This is the original statement that one neuron carries phase-shifted copies which must cancel across neurons. The cancellation is approximate, not exact.
   - Already in refs.bib as gromov2023grokking, with correct fields, but cited nowhere in sections/*.tex.
3. **Doshi, He, Das, Gromov 2024, arXiv 2406.03495, "Grokking Modular Polynomials"** (abs and HTML).
   - Addition of S terms with power activation x^S, which is our power neuron.
   - In the expansion (App. A.2, boxed Eq. 18) the only phase-free term is the signal. The extra terms, including mixed-sign ones, "increase exponentially in S" and "are suppressed by a factor of 1/N" by the random-phase approximation.
   - This is the n-operand power-neuron precedent that the checker said does not exist. It does not give the exact per-operand unit-factor identity.
4. **Morwani et al. ICLR 2024, arXiv 2311.07568** (HTML).
   - Uses hat u(-j) = conj(hat u(j)) explicitly, in the step to Eq. 6.
   - App. F.2 builds 8 neurons per frequency from the product-to-sum expansion of cos(a+b-c), with phases (0,0,0), (0,pi,pi), ... . This gives exact cross-neuron cancellation of the sign-flipped terms in the quadratic net.
   - Morwani et al. is already cited in the paper, but not for this point.

## What is new vs. known
- Known:
  - Hermitian symmetry (textbook; Morwani uses it).
  - Per-neuron sign-flipped copies with phase factors that must cancel across neurons: Gromov 2023 (2 operands, approximately); Doshi et al. 2024 (S operands, power activation, approximately); Morwani 2024 F.2 and He et al. 2026 (2 operands, exactly).
- Not found: the exact per-operand multiplicative form, one unit factor per operand multiplied over the flipped set T, stated as a lemma for product and power neurons and turned into the finite system eq:JF.
- The lemma is a trivial generalisation, so keep it as bookkeeping and do not present it as a contribution.

## Recommended change (amends the checker's)
Replace the He-only sentence after lem:sign with:
"The lemma is Hermitian symmetry of the weights' Fourier transform. Per-neuron sign-flipped copies that must cancel across neurons are the mechanism of \citet{gromov2023grokking} (approximate cancellation by random phases), extended to many-operand power networks by \citet{doshi2024grokking}, and made exact for two-operand quadratic networks by \citet[App.~F.2]{morwani2024feature} and \citet[Sec.~4]{he2026mechanism}; Lemma~\ref{lem:cost} prices that cancellation."
Add doshi2024grokking and he2026mechanism to refs.bib. gromov2023grokking is already there.

## Infra note (relevant to the workflow slowness)
WebSearch refused with "this session has used its web search budget (200 of 200 WebSearch calls)". The remaining verify tasks in this session have no WebSearch and must fall back to WebFetch or curl of known URLs.
