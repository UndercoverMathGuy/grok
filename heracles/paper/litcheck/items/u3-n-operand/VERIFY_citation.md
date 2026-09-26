# VERIFY (citation lens): U3-n-operand. Task 91

Verdict: UPHELD. CITATION_FOUND stands. Both recommended citations exist, their bib fields are correct, and they say what the checker claims.

## Fetched this session (2026-09-25)
1. https://proceedings.mlr.press/v258/li25b.html
   - The PMLR page gives: Li, Liang, Shi, Song, Zhou, "Fourier Circuits in Neural Networks and Transformers: A Case Study of Modular Arithmetic with Multiple Inputs", Proc. 28th AISTATS, PMLR vol. 258, pp. 523--531, 2025. This matches the checker's bibtex.
2. https://arxiv.org/html/2402.09469 (v4, 9 Mar 2025)
   - Sec 3.1 defines the neuron phi(theta_i,a) = (u_{i,1}(a_1)+...+u_{i,k}(a_k))^k w_i and the constraint ||theta||_{2,k+1} <= 1, with L_{2,k+1} = (sum_i ||theta_i||_2^{k+1})^{1/(k+1)}. This is our power neuron with n=k and b=nu=k+1.
   - Thm 4.1 (Sec 4): if m >= 2^{2k-1}(p-1)/2, then gamma* = 2(k!)/((2k+2)^{(k+1)/2}(p-1)p^{(k-1)/2}), and for each frequency in 1..(p-1)/2 some neuron uses only that frequency.
3. https://arxiv.org/abs/2406.03495 and https://arxiv.org/html/2406.03495
   - Doshi, He, Das, Gromov, "Grokking Modular Polynomials", submitted 5 Jun 2024, with no journal reference listed. Sec 2 "Modular Addition with Many Terms", Eq. 1: f = W phi(U^(1)e_{n1}+...+U^(S)e_{nS}) with phi(x)=x^S. The solution is analytic/constructive. The paper has no max-margin or norm theorem.
   - The OpenReview API lists it only as "CoRR 2024" (DBLP import, id Vf0k9wmZiO), so it is still a preprint. `@article{..., journal={arXiv preprint arXiv:2406.03495}}` is correct.

## Additional check: Li Thm 4.1 equals our bound exactly
Take n=k and nu=k+1:
n! B_p * p/(p-1) = 2 k! (2p(k+1))^{-(k+1)/2} * p/(p-1) = 2 k! / ((2k+2)^{(k+1)/2} p^{(k-1)/2} (p-1)).
This is exactly Li's gamma*. So at b=nu and large enough width, Theorem universal together with Cor. dense (Gamma_1 = p/(p-1)) is attained for EVERY n, not only n=2. Consequences for the text:
- appendix.tex l.279 says "At n=2, b=3 the bound is attained by Morwani ... so C_3 = 2B_p there". This can be strengthened to "at b=nu, C_nu = n! B_p for every n, attained by the network of \citet[Thm 4.1]{li2025nsum} (width m >= 2^{2n-1}(p-1)/2)".
- Cor. dense (approach.tex l.69-70) can add the general-n power value n! B_p p/(p-1), citing Li Thm 4.1, next to the Morwani n=2 value.
- The recommended sentence "At b=nu the maximum margin of the power network, with every frequency used, is computed by \citet[Thm 4.1]{li2025nsum}" is accurate. Add the width condition to it.

## Bib-field notes
- The existing refs.bib entry li2025nsum (l.53) has no volume, pages or series. Replace it with the PMLR fields (vol 258, pp. 523--531, PMLR) as recommended.
- doshi2024grokking must be added. The fields are as recommended.

## Better or more canonical citation?
None found. Li et al. is the primary source for the n-operand power-neuron max-margin family. Doshi et al. is the right secondary source for the architecture. I could not run a venue search because the session's WebSearch budget was exhausted (200/200). I used the OpenReview API instead.
