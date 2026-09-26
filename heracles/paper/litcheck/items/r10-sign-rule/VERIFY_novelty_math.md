# R10-sign-rule: verify (lens: novelty_math), 2026-09-25

Claim (cyclic.tex lem:sign, proof in appendix.tex app:models): for product and power neurons, beta_{ij,T} = beta_{ij} prod_{t in T} xi_{itj}, where each xi is a unit complex number.

## Math check
- Proof re-derived.
- Product neuron: the Fourier transform of a tensor-product function is the product of the 1D transforms (separability). Real weights give hat u(-j) = conj(hat u(j)) (Hermitian symmetry), so xi_{tj} = conj(hat u_t(j))/hat u_t(j). When hat u_t(j) = 0, both sides are 0, so xi := 1 is fine.
- Power neuron: this is a pigeonhole argument on the multinomial expansion. A monomial with sum e_t = n that has an e_t >= 2 must have some e_t' = 0, so only the multilinear term n! prod u_t reaches all-nonzero modes. Correct.
- Everything used is textbook: Hermitian symmetry, as in Wikipedia "Hermitian function" (fetched): "f is real-valued iff the Fourier transform of f is Hermitian". No citation is needed.
- The exact per-operand statement (one unit factor per flipped operand, multiplied over T, for n operands) was not found stated as a lemma anywhere.

## Prior art the checker missed (all fetched and read this session)
1. **Gromov 2023, arXiv 2301.02679, Sec. 3.1, Eqs. (10)-(15).** Already in refs.bib as gromov2023grokking. The checker dismissed it after reading only the abstract.
   - Eq. (11) lists every term of one quadratic neuron, including the sign-flipped copies cos(2pi k(n - m -/+ q)/p + phi1 - phi2 -/+ phi3) next to the signal cos(2pi k(n + m - q)/p + phi1 + phi2 - phi3). Relative to the signal, the flipped operand carries the unit factor e^{-2i phi2}: exactly our xi, at n = 2.
   - Eqs. (14)-(15) then cancel all of these copies across neurons ("interfere destructively") with random phases. The cancellation is approximate, O(sqrt N) against N.
   - This predates He et al. 2026 by three years for the n = 2 quadratic case.
2. **Doshi, He, Das, Gromov 2024, "Grokking Modular Polynomials", arXiv 2406.03495, App. A.2, Eq. (18).**
   - Covers S-operand modular addition with the x^S power activation, which is the paper's power neuron.
   - Only the multilinear term, with coefficient S!/2^S, survives. The other terms, including the sign-flipped copies, "drop out due to random phase approximation" and are "suppressed by a factor of 1/N". The number of these terms grows exponentially in S.
   - Eq. (15) (the modular multiplication case) shows flipped copies carrying phases such as -2 psi.
   - This is the n-operand power-neuron analogue, again with approximate cancellation.
3. **Morwani et al. ICLR 2024, App. F.2 (arXiv 2311.07568 PDF p. 27).**
   - An exact interference-free quadratic network with 8 neurons per frequency. It expands cos(a+b-c) into four product terms and implements each with a pair of neurons with phases in {0, +-pi/2, pi}, and states that "every term other than [the target] cancels out in the sum".
   - This is an exact cross-neuron cancellation of the flipped copies at n = 2.
4. **Li et al. AISTATS 2025, arXiv 2402.09469, Lemma 4.3 / Lemma E.3, Eq. (25).**
   - The exact k-operand version: 2^{2k-1} x^k-power neurons per frequency realise cos(a_1+...+a_k-c) exactly.
   - This is a feasible point of eq:JF for general n, found by trigonometric expansion. It is not stated as a sign rule.
   - The checker said Li has "no sign-flip or conjugate lemma". That is true, but the checker missed this exact n-operand cancelling construction.
5. **He, Wang, Chen, Yang 2026, arXiv 2602.16849.** Confirmed: authors Jianliang He, Leda Wang, Siyu Chen, Zhuoran Yang; submitted 18 Feb 2026.
   - Eq. 4.1 contains the flipped term cos(omega(x+y+j) + 4phi_m).
   - Def. 4.1(iii) states a phase-symmetry condition with iota in {2,4}. It gives exact cancellation for the tied-embedding quadratic model.

## Verdict
PARTIALLY_KNOWN is upheld.
- The lemma is elementary and, in its exact n-operand per-operand form, is not stated elsewhere.
- Its content is visible in earlier work:
  - n = 2: Gromov 2023 Eq. (11).
  - General S with power activation: Doshi et al. 2024 Eq. (18), explicitly for n = 2 and implicitly for general S.
  - Cross-neuron cancellation: Gromov 2023 and Doshi 2024 approximately; Morwani App. F.2, Li Lemma 4.3 and He Def. 4.1(iii) exactly.
- The checker's recommended sentence credits He et al. for "sign-flipped copies and their cancellation across neurons". That gives priority to the latest of five sources. Gromov 2023 should carry the primary credit.

## Recommended sentence (replaces the checker's)
"The lemma is Hermitian symmetry of the weights' Fourier transform. The sign-flipped copies are explicit in the expansion of \citet[Eq.~11]{gromov2023grokking}. There, and for $S$ operands with power activation in \citet[App.~A.2]{doshi2024grokking}, they are suppressed by random phases across neurons. Exact cancelling constructions appear in \citet[App.~F.2]{morwani2024feature}, \citet[Lemma~4.3]{li2025nsum} and \citet[Def.~4.1(iii)]{he2026mechanism}. Lemma~\ref{lem:cost} prices this cancellation."
- Add doshi2024grokking and he2026mechanism to refs.bib.
- Note: the Morwani/Li constructions are the natural foil for prop:block (nu neurons versus 8 or 2^{2k-1}). Flag this for the R21 nu-block item.

## Operational note
The session-wide WebSearch budget was exhausted (200/200) when this verifier started, so every search fails immediately. Verifiers also share one scratchpad directory with generic filenames (morwani.txt was clobbered to 0 bytes mid-task). Both slow the verify stage.
