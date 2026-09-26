# VERIFY (citation lens) — U9 product-to-sum interference (task 97)

Sentence: cyclic.tex, paragraph "Interference at the feature's own frequency" (lines 6-7, 19). The
paragraph currently cites nothing. It says product neurons also carry sign-flipped copies
cos(2 pi j(sum_{t not in T} a_t - sum_{t in T} a_t - c)/p), and several neurons must cancel them.

Sources fetched this session (arXiv PDFs, read with pdftotext): 2301.02679, 2311.07568 (and abs page),
2602.16849, 2301.05217, 2402.09469.

## Checker's confirmed citations

1. gromov2023grokking (arXiv 2301.02679v1, single author, 2023). CONFIRMED.
   - Eq. (10): the squared first-layer activation contains cos(2 pi k(n-m)/p + phi1 - phi2).
   - Eq. (11): the second-layer preactivation contains cos(2 pi k(n-m-q)/p ...) and cos(2 pi k(n-m+q)/p ...).
     These are exactly the beta_{j,T} copies for n = 2.
   - Eqs. (12)-(15): phi1 + phi2 = phi3 makes the third line constructive. "All other waves ... interfere
     destructively" only because random phases give sum_k cos(2 pi k s/p + phi_k) << N. So the
     cancellation is approximate.
   - Sec. 4 says trained networks have approximately random phases that approximately satisfy (12). This
     supports "in trained networks". Bib entry is correct.

2. morwani2024feature (arXiv 2311.07568; the abs page says "Accepted as Spotlight at ICLR 2024"). CONFIRMED.
   - The authors and title in the bib are exact.
   - App. F.2: 8 neurons per frequency, in paired phases, sum to exactly sum_zeta cos_zeta(a+b-c).
   - App. F.3: one neuron's DFT is nonzero only at j=0, k=0, or j,k,l = +-zeta. The max-margin f is
     F(a,b) + lambda*1[a+b=c], so f_hat is supported on l=0 or j=k=-l.
   - This is stronger than "implicit". In their L_{2,3} setting, EVERY max-margin network is
     interference-free, because mixed-sign modes with l != 0 vanish. The paper could say so.

3. he2026mechanism (arXiv 2602.16849v1, He, Wang, Chen, Yang, Yale). CONFIRMED, but the checker's
   gloss overstates it.
   - Eq. (4.1): the per-neuron output is cos(w(x-y)/2)^2 {primary signal + phase-carrying terms}.
   - Def. 4.1(iii) with iota=2 (sum_m exp(2 i phi_m) = 0) cancels the phase-carrying terms EXACTLY. Those
     terms contain the a-b copies cos(w(x-y-+j) -+ 2phi).
   - Prop. 4.2 / (B.1)-(B.2), however, KEEPS the residual noise at 2x-j and 2y-j, the "Achilles' heel"
     (they credit Zhong et al. 2023). Those are other-mode terms, which the paper allows.
   - So "the diversified neurons collectively cancel that noise" is wrong as stated. Diversification
     cancels the own-frequency sign-flipped copies, not all of the residual noise. The recommended
     wording ("the phase-diversification condition ... (Def. 4.1)") is fine.

Nanda 2023: the checker is right. There is no a-b product-to-sum term, only constructive and
destructive interference of the logit cosines. Do not cite it here.

## Missed / better citation

- li2025nsum (arXiv 2402.09469v4, Li, Liang, Shi, Song, Zhou; already in refs.bib). App. E.2
  generalises Morwani F.2 to k operands. It uses 2^k * 2^{k-1} neurons per frequency, via the identity
  2^k k! prod a_i = sum_c (-1)^{...} (sum c_j a_j)^k. The network computes exactly
  sum_zeta cos_zeta(sum a_i - c), so every sign-flipped copy cancels, for general n.
- This is the right citation for the general-T / n-operand statement. Cite it with Morwani.

## Suggested sentence

"These copies appear explicitly in the analytic solution of \citet{gromov2023grokking} (Eqs. 10-11),
which cancels them only approximately, through random phases. The exact max-margin constructions of
\citet{morwani2024feature} (App. F.2-F.3) for n=2 and \citet{li2025nsum} (App. E.2) for general n
cancel them exactly across neurons. So does the phase-diversification condition of
\citet{he2026mechanism} (Def. 4.1), which leaves only other-mode noise."
Add the he2026mechanism bib entry (fields verified).

Process note: WebSearch returned "session has used its web search budget (200 of 200)". Verifiers in
this workflow share one session-level search budget. Once it is spent, later verifiers cannot search.
