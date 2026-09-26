# T13-relu-budget: ReLU single-neuron budget (appendix app:models, ReLU paragraph, appendix.tex l.283)

Verdict: PARTIALLY_KNOWN. The Fourier + Cauchy-Schwarz single-neuron template comes from Morwani et al. The 1-Lipschitz variance step that extends it to ReLU does not appear in any source I found.

## What we do
beta_j = 2 p^{-(n+1)} h^(j,...,j) conj(w^(j)). The bound uses three steps:
- Cauchy-Schwarz over j.
- Parseval, restricted to the diagonal coefficients.
- Var(sigma(X)) <= Var(X), which holds because sigma is 1-Lipschitz.

The result is C_2 <= p^{-n}/2 at b = nu = 2.

## Sources checked
- Morwani, Edelman, Oncescu, Zhao, Kakade, ICLR 2024 (arXiv 2311.07568v2; full PDF read).
  - App. F.1 (proof of Thm 7, quadratic neuron) has the same template: Plancherel turns the norm constraint into a Fourier constraint, then Cauchy-Schwarz (and AM-GM) bounds a single neuron's Fourier sum.
  - They also note that "E_c[w(c)] = 0, because shifting the mean of w does not affect the margin". That is the fact that disposes of the j = 0 term.
  - ReLU appears only empirically: Fig 1a, Fig 4, and the L2-regularised ReLU experiments. There is no ReLU margin bound.
- Gromov 2023 (arXiv 2301.02679; full PDF grepped). The analysis uses quadratic activation. Line ~598 says: "The weights (6)-(7) also work for other activation functions, including ReLU", and that claim is empirical. There is no ReLU bound.
- Li et al., AISTATS 2025 (2402.09469). Power activations only.
- Memana 2026 (arXiv 2608.04451, "A Counterexample to Fourier Alignment in Single-Neuron Modular Addition"). This is ReLU single-neuron dynamics, a counterexample for MAIS-O60. It has no margin or budget bound.
- McCracken et al. 2025 (2505.18266). Constructs random-frequency ReLU solutions and bounds an unnormalised margin. It has no single-neuron norm budget.
- The variance lemma Var f(X) <= L^2 Var X for L-Lipschitz f is textbook. The one-line proof takes an independent copy X': Var f(X) = (1/2) E (f(X) - f(X'))^2. No citation is needed; a footnote with the proof is enough.

## Searches run
- "ReLU network modular addition maximum margin upper bound Fourier single neuron proof"
- "1-Lipschitz function variance contraction Var(f(X)) <= Var(X) lemma"

## Suggested edit
Add one clause: "(the Plancherel/Cauchy-Schwarz single-neuron argument of Morwani et al., 2024, App. F.1, extended to ReLU via the 1-Lipschitz variance bound)". Also check the j = 0 term. The text sums over j in F_p, but Var(h) does not control h^(0) = p^n E[h]. The bound should either sum over j != 0, or cite the fact that the mean of w does not affect the margin (Morwani App. F.1), so w^(0) = 0 without loss.
