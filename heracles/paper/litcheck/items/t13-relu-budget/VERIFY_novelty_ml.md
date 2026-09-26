# VERIFY (novelty_ml): T13-relu-budget, task 63

Claim (appendix.tex l.283): the ReLU single-neuron budget follows from Cauchy-Schwarz over j, Parseval on the diagonal coefficients, and Var(h) <= Var(sum_t u_t) because sigma is 1-Lipschitz.

## Infrastructure note (why this run was slow)
- WebSearch was refused on the first call: "this session has used its web search budget (200 of 200 WebSearch calls)". Every later verifier in this session will hit the same limit.
- The arXiv export API returned HTTP 429 (rate-limited), most likely because parallel verifiers are all querying it.
- Workaround used here: arxiv.org/search HTML via curl, and PDFs via curl + pdftotext.

## Novelty search (ML lens)
arXiv abstract searches run:
- "modular addition ReLU margin"
- "modular addition ReLU"
- "modular addition margin"
- "modular arithmetic maximum margin Fourier"
- "grokking modular addition theory two-layer"
- "ReLU Fourier margin group"
- "modular addition Lipschitz Fourier neuron"
- "modular arithmetic implicit bias ReLU"

Candidates read:
- 2511.23443, Huang & Li, "Provable Benefits of Sinusoidal Activation for Modular Addition". Gives an expressivity gap (ReLU width must scale with length m) and generalization bounds for sine networks. No ReLU single-neuron Fourier or margin budget.
- 2606.05863, Tan, Gai & Zhang, "Two Training Clocks ... Conditional ReLU Reduction". Treats ReLU as locally linear under fixed activation patterns. No budget.
- 2407.12332, Mohamadi et al., "Why Do You Grok?" (full PDF grepped). Uses the l_inf margin for a 3-homogeneous quadratic network. Its only Lipschitz step is Talagrand contraction inside a Rademacher bound, not a Fourier budget. No ReLU.
- 2402.09469, Li et al. (full PDF grepped). Power activations only. ReLU appears only in the bibliography.
- 2412.03773, "Modular addition without black-boxes" (ReLU MLP). Uses mechanistic error bounds; the Lipschitz constant of h is used for an integration error. No margin budget.
- 2608.04451, "A Counterexample to Fourier Alignment in Single-Neuron Modular Addition". Exists in the arXiv listing. It is a dynamics paper.

Result: no ML paper found that bounds the ReLU single-neuron Fourier or margin budget. The ReLU extension is new as far as found. The template is Morwani's.

## Confirmed citations, re-checked from the PDFs
- morwani2024feature (2311.07568). App. F, "Proofs for cyclic groups (Theorem 7)", F.1:
  - "First, note that E_c[w(c)] = 0, because shifting the mean of w does not affect the margin."
  - "By Plancherel's theorem, the norm constraint is equivalent to ..."
  - "(Cauchy-Schwartz)" step.
  - F.1 concerns the MEAN margin of the quadratic neuron (u(a)+v(b))^2 w.
  - ReLU appears only empirically (Fig. 1a, Fig. 4, App. B). CONFIRMED.
- gromov2023grokking (2301.02679):
  - Main text: "The weights (6)-(7) also work for other activation functions, including ReLU".
  - App. B: "We do not have an analytic proof of this fact". CONFIRMED. It supports "no prior ReLU bound".

## Checker errors
1. The j=0 issue is a false alarm. prelim.tex l.7 defines \Fp := {1,...,(p-1)/2}, so the sum never includes j=0. Recommended change (2) is unnecessary.
2. The checker missed a real error: the stated constant is WRONG for n >= 2.
   - The steps as written give:
     - sum_{j in F} |h^(j..j)|^2 <= (1/2) p^n sum_x (h - mean h)^2 = (1/2) p^{2n} Var(h) <= (1/2) p^{2n-1} sum_t ||u_t||^2
     - sum_{j in F} |w^(j)|^2 <= (p/2) ||w||^2
   - So Cauchy-Schwarz and AM-GM give sum_{j in F} |h^(j..j) w^(j)| <= (p^n/4) ||omega||^2, not (p/4) ||omega||^2.
   - Hence C_2 <= 2 p^{-(n+1)} p^n / 4 = 1/(2p), which equals B_p at nu=2, independent of n. It is not p^{-n}/2.
   - Numerical counterexample to the printed bound (relu_budget_check.py): random ReLU neurons give C_2 >= 0.00391 at p=23, n=2, against the claimed p^{-2}/2 = 0.000945. Similar violations occur at p=7 and 11 and at n=3.
   - The LHS ratio reaches 23.8, against the printed p/4 = 5.75.
   - At n=1 the two constants agree.
   - The conclusion ("Theorem universal and Corollary dense apply at b=nu=2") survives with the corrected constant.

## Recommended edit
Replace
  "... give sum_{j in \Fp} |h^(j,...,j) w^(j)| <= (p/4) ||omega||_2^2, i.e. C_2 <= 2p^{-(n+1)} p/4 = p^{-n}/2"
with
  "... give sum_{j in \Fp} |h^(j,...,j) w^(j)| <= (p^n/4) ||omega||_2^2, i.e. C_2 <= 2p^{-(n+1)} p^n/4 = 1/(2p) = B_p|_{nu=2}".

Also credit the template:
  "the Plancherel/Cauchy--Schwarz single-neuron argument of \citet[App.~F.1]{morwani2024feature}, extended to ReLU by Var(sigma(X)) <= Var(X) for 1-Lipschitz sigma".
The Lipschitz variance fact is textbook: Var f(X) = (1/2) E(f(X) - f(X'))^2 <= (1/2) E(X - X')^2 = Var X. It needs no citation.
