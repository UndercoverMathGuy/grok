# VERIFY (citation lens) — C11-morwani-single-neuron

Claim (approach.tex l.54): "It remains to price amplitudes in norm, which, as in \citet{morwani2024feature}, is a single-neuron optimization."

## What I fetched this session
- arXiv abs 2311.07568: title "Feature emergence via margin maximization: case studies in algebraic tasks";
  authors Morwani, Edelman, Oncescu, Zhao, Kakade (order matches bib); v2 19 Feb 2024; comment "Accepted as Spotlight at ICLR 2024".
  Bib entry (inproceedings, ICLR, 2024, note arXiv) is correct.
- arXiv PDF v2, full text extracted locally.
  - Sec 3.1: "Since the expected margin of the network decomposes into the sum of expected margin of individual neurons,
    finding a maximum expected margin network simplifies to finding maximum expected margin neurons."
  - Lemma 3 (binary), bullet "Single neuron optimization: Any theta in Theta*_q has directional support only on Omega*_q";
    bullet "Combining neurons" requires b = nu.
  - Lemma 5 (class-weighted, the multi-class version actually used for modular addition): same two bullets.
  - App E.1 proof: E_q g = sum_i ||w_i||_a^nu E_q psi(w_i/||w_i||_a) <= gamma sum_i ||w_i||_a^nu <= gamma max_{||w||_b<=1} ||w||_nu^nu.
  - App E.1 Remark: "the analysis in above proof can be used to compute optimal norms for b != nu as well - however, for any such b
    we would not get the same flexibility to build a theta* satisfying Equation 1."
- arXiv abs 1810.05369 (Wei, Lee, Liu, Ma) checked as a possible more general source; the abstract states the weak-regularization ->
  max-normalized-margin result, not a single-neuron reduction. Not a better citation for this sentence (already cited elsewhere for Thm 4.1).

## Assessment
Checker's verdict PARTIALLY_SUPPORTED stands. The shared idea is real and correctly attributed: per-neuron additivity + homogeneity
reduce a network-level norm-constrained maximisation to a unit-norm single-neuron maximisation, followed by a norm-allocation step.
Differences the checker named are accurate: Morwani's per-neuron objective is a *linear* expected (class-weighted) margin under a
fixed dual distribution q, used inside the certificate-pair argument at b = nu; the paper's C_b = max_{||w||<=1} sum_j |beta_j(w)|^{b/nu}
is a concave l_{b/nu} budget used in a subadditivity upper bound for 2 <= b <= nu, with no dual distribution.

One point the checker did not stress, which helps the paper: Morwani's own Remark (App E.1) says the single-neuron/norm-allocation
analysis extends to b != nu; only the "combining" step needs b = nu. So "as in Morwani" is a fair attribution of the reduction step.

Precision fix: for multi-class modular addition the relevant lemma is Lemma 5 (Lemma 3 is binary). Citing "Lemma 5 (cf. Lemma 3)" or
"Lemmas 3 and 5" is fine; the checker's recommended rewording is accurate. A shorter acceptable version:
"It remains to price amplitudes in norm. As in the single-neuron step of \citet[Lemma~5]{morwani2024feature}, homogeneity reduces this to
one neuron, though here the per-neuron quantity is an $\ell_{b/\nu}$ budget of Fourier coefficients rather than an expected margin."

No missed prior art that is a better or more canonical source for this specific step; no bogus citations.
