# R10-sign-rule (cyclic.tex lem:sign; appendix app:models)

Statement: for product and power neurons, beta_{ij,T} = beta_{ij} prod_{t in T} xi_{itj}, where each xi is a unit complex number.
Proof in the paper: real weights have hat u(-j) = conj(hat u(j)), so hat u(-j) = xi * hat u(j) with |xi| = 1. The coefficient at the mode flipping T is the product with those factors swapped in. For power neurons, only the multilinear monomial reaches all-nonzero modes.

## Verdict: PARTIALLY_KNOWN (elementary; a quadratic-net special case appears in prior work)

## Evidence (fetched 2026-09-25)
1. The core step is Hermitian symmetry of the DFT of a real vector, hat u(-j) = conj(hat u(j)). This is textbook, and no citation is needed.
2. He, Wang, Chen, Yang, arXiv 2602.16849 (2026), Sec. 4.
   - Eq. (4.1) decomposes one quadratic neuron with phase phi_m into three parts:
     - the signal cos(omega(x+y-j));
     - cos(omega j + 2 phi_m);
     - cos(omega(x+y+j) + 4 phi_m), which is a sign-flipped copy carrying a phase factor, exactly the xi-product of our lemma.
   - Def. 4.1(iii) requires, for iota in {2,4}, that exp(i iota phi_m) summed over the neurons of frequency k is 0. That is phase cancellation of these copies across neurons, the quadratic analogue of our eq:JF.
   - So "one neuron is never clean, and phases must cancel across neurons" is already known for the 2-input quadratic net.
3. Morwani et al., ICLR 2024 (arXiv 2311.07568).
   - They use hat u(j) hat v(j) hat w(-j) with conjugate symmetry.
   - Thm 7 has the phase condition theta_u + theta_v = theta_w.
   - They do not state a sign-flip factor for each operand.
4. Li et al., AISTATS 2025 (arXiv 2402.09469).
   - Their neurons are one-sparse cosines (Lemma 4.2).
   - They state no sign-flip or conjugate relation (checked in the HTML).
5. Gromov 2023 (arXiv 2301.02679). The abstract has nothing on this.
6. Search: "modular addition Fourier conjugate phases cancel spurious terms ... roots of unity". The only hit was He et al.

## What is new vs known
- Known: Hermitian symmetry, and, for quadratic 2-input nets, per-neuron sign-flipped noise terms with phase factors that cancel across neurons (He et al. Eq. 4.1, Def. 4.1(iii)).
- Not found: the general n-operand form, i.e. one unit factor per operand that multiplies over the flipped set T, for both product and power neurons, used as the exact interference-free system eq:JF.
- The general form is a trivial generalisation. Present it as a bookkeeping lemma.

## Recommended change
After lem:sign (cyclic.tex, the paragraph starting "So a single neuron is never interference-free..."), add:
"The lemma is Hermitian symmetry of the weights' Fourier transform. For two-input quadratic networks, these sign-flipped copies and their cancellation by phase symmetry across neurons appear in \citet[Eq.~4.1, Def.~4.1(iii)]{he2026mechanism}; Lemma~\ref{lem:cost} prices that cancellation."
Add the bib entry below. Check first that no He et al. key already exists; refs.bib has none as of this check.
