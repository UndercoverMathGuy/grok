# C1 morwani-dense-baseline — CHECK

Source fetched: arXiv 2311.07568 abstract page (title, authors, "Accepted as Spotlight at ICLR 2024")
+ full LaTeX source (e-print) + arXiv HTML v2 for theorem numbering.
Morwani, Edelman, Oncescu, Zhao, Kakade. "Feature emergence via margin maximization: case studies in algebraic tasks". ICLR 2024.
refs.bib entry `morwani2024feature` is correct.

## What Morwani actually proves (numbers from arXiv HTML v2)
- Thm 1 = restatement of Wei et al. 2019 Thm 4.1: for any norm, homogeneous f, gamma_lambda -> gamma* as lambda -> 0 (global minimisers of regularised loss).
- Neuron (Sec 2): phi({u,v,w},a,b) = (u_a + v_b)^2 w, quadratic activation, nu = 3.
- **Thm 7 (cyclic)**: m >= 4(p-1). "The maximum L_{2,3}-margin ... is gamma* = sqrt(2/27) * 1/(p^{1/2}(p-1)). Any network achieving this margin satisfies: (1) for each neuron ... there exists a scaling constant lambda and a frequency zeta such that u,v,w are cosines at zeta with theta_u+theta_v=theta_w; (2) For every frequency zeta in {1..(p-1)/2}, at least one neuron in the network uses this frequency."
- **Thm 8 (sparse parity)**: activation x^k, L_{2,k+1}, m >= 2^{k-1}: gamma* = k! sqrt(2(k+1)^{-(k+1)}); every active neuron has support exactly S. (Not an "all features" result; not relevant to C1.)
- **Thm 9 (groups)**: quadratic activation, L_{2,3}, G with *real* irreps, m >= 2 sum d^3, AND the character condition sum_{n>=2} d_{R_n}^{1.5} chi_{R_n}(C) < 0 for every non-trivial conjugacy class C. gamma* = 2/(3 sqrt(3|G|)) / sum d^{2.5}. Each neuron spanned by one non-trivial irrep; every non-trivial irrep used by at least one neuron.
- Informal theorem (intro): "For m >= 4(p-1), any global minimizer of the loss [vanishing regularization] satisfies: every neuron uses one frequency; at least one neuron of each frequency."
- Sec 3 "Combining neurons: If b = nu ... theta = {lambda_i omega_i*} with sum lambda_i^nu = 1 belongs to Theta*" — supports intro para 3 ("combining is free exactly when exponent = nu").
- Discussion (Sec 6): "Our proof holds for the case of x^2 activations (x^k activations, in the k-sparse parity case) and L_{2,nu} norm."

## Check of each attributed piece
1. "at L_{2,nu} with nu = homogeneity all features are used for mod-add and group composition" (paper.tex abstract/para 1):
   PARTIAL. Morwani prove it only for nu = 3 (quadratic), not for general nu. General nu for mod-add
   (power neurons, k operands, L_{2,k+1}, every frequency used) is Li et al. AISTATS 2025 Thm 4.1.
   Group case additionally needs real irreps + the character-sum condition + width >= 2 sum d^3.
2. "For quadratic networks every max L_{2,3}-margin net uses Fourier features, one frequency per neuron, every frequency used" (intro para 1): SUPPORTED by Thm 7 (needs m >= 4(p-1); also phase condition theta_u+theta_v=theta_w).
3. "with vanishing regularization, networks approach the max margin solution (Wei)": Morwani Thm 1 — statement is about global minimisers, not trained networks. Minor.
4. "our bound at b=nu recovers their max margin exactly": value matches Thm 7: 2 B_p p/(p-1) = 4(6p)^{-3/2} p/(p-1) = sqrt(2/27)/(p^{1/2}(p-1)) (checked numerically: 4/6^1.5 = 0.27217 = sqrt(2/27)). SUPPORTED for the cyclic quadratic case; cite "Theorem 7".

Verdict: PARTIALLY_SUPPORTED (over-generalisation to arbitrary nu and unconditional groups in paper.tex).
