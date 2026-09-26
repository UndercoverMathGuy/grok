# U3-n-operand: CITATION_FOUND

Location: sections/prelim.tex line 5 ("Task and neurons"). The task and the power neuron are introduced there, but the text cites only morwani2024feature (for n=2). `li2025nsum` is in refs.bib (line ~54) and is never \cite'd.

## Verified sources (fetched 2026-09-25)
1. Li, Liang, Shi, Song, Zhou. "Fourier Circuits in Neural Networks and Transformers: A Case Study of Modular Arithmetic with Multiple Inputs". AISTATS 2025, PMLR 258:523-531. https://proceedings.mlr.press/v258/li25b.html ; arXiv 2402.09469 (v1 Feb 2024).
   - Sec 3.1: k-input dataset (a_1+...+a_k) mod p; neuron phi(theta_i,a) = (u_{i,1}(a_1)+...+u_{i,k}(a_k))^k w_i; constraint ||theta||_{2,k+1} <= 1. These are exactly our D_p and our power neuron with n=k, b=nu=k+1.
   - Thm 4.1 (Sec 4): if m >= 2^{2k-1}(p-1)/2, gamma* = 2(k!)/((2k+2)^{(k+1)/2}(p-1)p^{(k-1)/2}), and every frequency 1..(p-1)/2 has a neuron using only that frequency (dense, general nu).
2. Doshi, He, Das, Gromov. "Grokking Modular Polynomials". arXiv 2406.03495 (2024, preprint). https://arxiv.org/abs/2406.03495
   - Sec 2 "Modular Addition with Many Terms", Eq. 1: f = W phi(U(e_{n1}+...+e_{nS})), phi(x)=x^S. Same power-neuron family (unnormalised, constructive/analytic solution, no margin theorem). Secondary citation.

## Recommendation
prelim.tex line 5: after defining the power neuron, write "... both with nu=n+1; this task and the power neuron are those of \citet{li2025nsum} (see also \citealp{doshi2024grokking}), and at n=2 the latter is the quadratic neuron of \citet{morwani2024feature}." Also cite Li et al. Thm 4.1 wherever the b=nu dense endpoint / n! B_p power bound is discussed (approach.tex l.58, appendix.tex l.266/279), and credit them as owning the general-nu dense L_{2,nu+1} result. Our new content is b<nu (sparse regime) and the product neuron.
