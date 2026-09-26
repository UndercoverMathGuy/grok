# T13-relu-budget: verify, novelty_math lens (task 64)

## Result
The PARTIALLY_KNOWN verdict holds, and so do both confirmed citations. Separately, I found a correctness error that the checker missed, and the checker's j=0 concern turns out not to apply.

## Citations checked (PDFs fetched and text extracted this session)
- **Morwani et al. 2024 (arXiv 2311.07568v2), App. F.1 "Proofs for cyclic groups (Theorem 7)".** Confirmed verbatim:
  - "First, note that E_c[w(c)] = 0, because shifting the mean of w does not affect the margin."
  - "By Plancherel's theorem, the norm constraint is equivalent to ||u^||^2+||v^||^2+||w^||^2 <= p".
  - The quadratic-geometric mean (AM-GM) step, eq. (9).
  - The "(Cauchy-Schwartz)" step, ending at sqrt(2/27)/(p^{1/2}(p-1)).

  The theory covers the quadratic neuron only. ReLU appears only in experiments (Fig. 1a/4/7, L2-regularised ReLU runs).
- **Gromov 2023 (arXiv 2301.02679).** Confirmed verbatim: "The weights (6)-(7) also work for other activation functions, including ReLU, however the 100% accuracy is achieved at higher width compared to quadratic activation function (more details in Appendix B)". This is an empirical statement, and the paper has no ReLU bound.

## Novelty (math lens)
Each ingredient is standard:
- Parseval on Z_p^n.
- Cauchy-Schwarz.
- Var f(X) <= L^2 Var X for L-Lipschitz f. This is textbook, with a one-line proof via an independent copy. No citation is needed.

The Plancherel + Cauchy-Schwarz single-neuron template is Morwani's. I found no prior ReLU single-neuron Fourier budget for modular-addition margins. The paper should present this as a routine extension of Morwani App. F.1, not as a new technique.

My independent web search was cut short. The session's WebSearch budget was exhausted (200/200) on my first query, so the novelty search here relies on the fetched PDFs and on the checker's search log.

## Checker's j=0 concern does not apply
prelim.tex defines \Fp := {1,...,(p-1)/2}, so the sum over j in \Fp already excludes j=0. The fix the checker recommended (sum over j != 0, or assume w^(0)=0) is not needed.

## Correctness error the checker missed (appendix.tex l.283)
The paper claims sum_{j in Fp} |h^(j,...,j) w^(j)| <= (p/4)||omega||^2, and therefore C_2 <= p^{-n}/2.

This holds only for n=1. With the paper's unnormalised DFT (prelim.tex), Parseval on Z_p^n gives:
- sum_{j in Fp} |h^(j..j)|^2 <= (1/2) p^{2n} Var(h) <= (1/2) p^{2n-1} sum_t ||u_t||^2
- sum_{j in Fp} |w^(j)|^2 <= (p/2)||w||^2

Cauchy-Schwarz then gives (p^n/2) sqrt(U)||w|| <= (p^n/4)||omega||^2. So the correct constant is:
- the sum is bounded by (p^n/4)||omega||^2;
- C_2 <= 2p^{-(n+1)} * p^n/4 = 1/(2p), which does not depend on n. At n=1 this equals p^{-n}/2.

Numerical check (relu_budget_check.py in this folder; random and cosine ReLU neurons, w rescaled to equal norm):

| p  | n | best ratio found | p/4   | p^n/4 |
|----|---|------------------|-------|-------|
| 11 | 2 | 5.81             | 2.75  | 30.2  |
| 23 | 2 | 25.3             | 5.75  | 132   |
| 47 | 2 | 105.0            | 11.75 | 552   |

At n=2 the stated bound is violated by a factor of up to about 9, and the violation grows like p. Every ratio found stays below p^n/4.

The qualitative conclusion survives: the b=nu=2 endpoint and the flat maximizer of Cor. dense apply with any constant. The displayed constant and C_2 must be corrected for n >= 2.

## Suggested text
"...give sum_{j in Fp}|h^(j,..,j) w^(j)| <= (p^n/4)||omega||_2^2, i.e. C_2 <= 2p^{-(n+1)} * p^n/4 = 1/(2p) at b=nu=2 (the Plancherel/Cauchy-Schwarz single-neuron argument of \citet[App.~F.1]{morwani2024feature}, extended to ReLU by Var sigma(X) <= Var X for 1-Lipschitz sigma)."

## Note for the orchestrator (the user asked why the workflow is slow)
1. **WebSearch budget.** This session hit the per-session WebSearch cap (200/200) before this task's first query. Every later verifier gets "Web search was not performed", so its novelty searches silently degrade. The cap is set by CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION.
2. **arXiv PDFs through WebFetch.** WebFetch on arXiv PDFs returns unreadable binary, and the small summariser model says it cannot find the text. The PDF is saved locally, so each verifier then needs extra rounds to run pdftotext on it. Fetching arxiv.org/abs or the HTML version (arxiv.org/html/<id>), or pre-extracting the texts once and sharing them, would save several turns per task.
