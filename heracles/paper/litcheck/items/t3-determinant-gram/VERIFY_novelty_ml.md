# VERIFY (novelty_ml): T3-determinant-gram, the proof of Lemma cost (app:cost)

Verdict: the checker's verdict of KNOWN (as technique) stands. I did not uphold the check as submitted, for two reasons: a confirmed citation has a wrong DOI, and one ML precedent was missed.

## What was checked
- The claim itself: sections/cyclic.tex (Lemma cost) and sections/appendix.tex (app:cost).
- The alternative classical proof that the checker proposed. I checked it by hand and it is correct.
  - H/beta is upper triangular with unit diagonal, so all its eigenvalues are 1.
  - Weyl's majorant theorem with phi(x) = x^r gives nu <= sum_i sigma_i(H)^r.
  - Rotfel'd/Thompson eq. (1), applied inductively to the N rank-one summands, bounds that by sum_l |d_l|^r. Each summand has the single singular value |d_l|.
  - The equality case (exactly nu neurons with orthogonal chain vectors) is not immediate on this route. The paper's determinant route gives it cleanly.

## Adversarial ML search (assume the lemma is already in the ML literature)
I searched for an ML paper that gives any of three things: a lower bound on the number of neurons per frequency needed to cancel sign-flipped copies, a cost of cancellation of the form nu^{1-r}, or a determinant/Gram argument.

- **Morwani et al. 2024** (arXiv 2311.07568; fetched the HTML).
  - One neuron per frequency with phase constraint theta_u + theta_v = theta_w; width m >= 4(p-1).
  - The norm is L_{2,3} at b = nu, which is the triangle-inequality regime.
  - No lower bound and no cancellation-cost lemma.
- **Li, Liang, Shi, Song, Zhou (AISTATS 2025)** (arXiv 2402.09469; fetched the HTML).
  - 2^{2k-1} neurons per frequency via a sum-to-product identity; width m >= 2^{2k-2}(p-1).
  - The width condition is sufficient, not a lower bound.
  - Norm L_{2,k+1} (b = nu); no cost inequality; no determinants or Gram matrices.
- **Doshi, He, Das, Gromov 2024, "Grokking modular polynomials"** (arXiv 2406.03495; fetched the HTML).
  - Cross terms cancel only on average, through random i.i.d. phases, with width growing exponentially.
  - No exact minimum and no cost comparison.
- **Tian, NeurIPS 2025, "Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets"** (arXiv 2410.01779 v4; fetched the abs and HTML). This is the closest ML precedent, and the checker missed it.
  - The order-3 solution z_syn = rho(u_syn), with u_syn = [omega_3, omega_3, omega_3] (cube roots of unity), gives "each frequency has exactly 3 hidden nodes" when Eq. 3 is optimized without its last (R_*) term.
  - That configuration is the nu = 3 instance of the equality case and the nu-block: nu neurons whose phases are nu-th roots of unity cancel the sign-flipped copy.
  - Theorem 5 says only qualitatively that "lower-order solutions are preferred if trained with L2 regularization", via a zero-loss path argument.
  - What Tian lacks:
    - any lower bound on the order;
    - any cost inequality;
    - any claim that order 3 is minimal (per the fetched text);
    - any determinant, Gram, Hadamard or singular-value tool.
  - So Tian does not scoop the lemma. It should be cited as prior construction and observation of the nu = 3 roots-of-unity block in quadratic networks.
- **Representation-cost / Schatten-quasi-norm literature** (Dai, Karzand, Srebro, NeurIPS 2021, fetched; Shang et al., arXiv 1606.00668, fetched).
  - These give the ML framing that l2 weight decay on factorized parameters induces l_{2/L} / Schatten quasi-norm costs.
  - None of them contains the JF/triangular-H argument or the nu^{1-r} cancellation cost.
- **arXiv API sweeps** (modular + margin + Fourier, newest first; modular + addition + margin + neurons) turned up no 2025-2026 paper with a per-frequency neuron lower bound or a cancellation cost.

**ML-lens conclusion.** The technique is standard mathematics. The inequality is classical: Weyl majorant plus Rotfel'd-Thompson. Its application is new to ML: forced interference cancellation in product networks costs a factor nu^{1-r} and needs at least nu neurons with orthogonal chain vectors, which pins the neuron count for b < nu. I found no ML prior art for that. The equality-case configuration at nu = 3 was constructed by Tian (2025), without an optimality or lower-bound proof.

## Verification of the checker's confirmed citations
- **hornjohnson2012**: WRONG DOI.
  - The checker's bibtex says edition 2, 2012, doi 10.1017/CBO9780511810817.
  - Crossref and the Cambridge page show that DOI is the FIRST edition (print 27 Dec 1985, online 2012).
  - The 2nd edition is doi 10.1017/CBO9781139020411 (Crossref: edition 2, published 2012-10-22; Cambridge higher-education page confirmed).
  - Fix the DOI (or use edition 1 / 1985 with the old DOI). Section numbers are still unverified.
- **thompson1976**: CONFIRMED.
  - Fetched the msp.org page and the PDF. Pacific J. Math. 66(1):285-290, 1976, doi 10.2140/pjm.1976.66.285.
  - Eq. (1) is verbatim for n-square (arbitrary complex) A, B, C = A+B, with f increasing concave and f(0) = 0, stated and proved by Rotfel'd.
  - Crossref gives Sept 1976 and the PDF header says November 1976; either way the year is 1976.
- **weyl1949**: metadata CONFIRMED (Crossref and PMC): PNAS 35(7):408-411, 1949, doi 10.1073/pnas.35.7.408.
  - I could not read the scanned text (PMC and PNAS block the PDF).
  - The product inequality |lambda_1...lambda_k| <= sigma_1...sigma_k is attributed to Weyl (Wikipedia, fetched). The additive majorant form for phi(e^s) convex increasing is the standard consequence.
  - Content is plausible and standard, but not verified from the primary text in this session.
- **bhatia1997**: metadata CONFIRMED via Crossref: Matrix Analysis, GTM 169, Springer, 1997, doi 10.1007/978-1-4612-0653-8.
  - Thm II.3.6 and the Ch. V integral representation are unverified.
  - Prefer the DOI URL over the Amazon URL.

## Recommended changes
1. Add Tian (2025) near Proposition block / the equality case of Lemma cost. Suggested sentence: "for n = 2 the three-neuron cube-root-of-unity configuration appears as the order-3 solution z_syn of [tian2025cogs], without an optimality or lower-bound argument".
2. Correct the Horn-Johnson DOI.
3. Keep the checker's other recommendations: the Weyl + Rotfel'd-Thompson remark, and not presenting the determinant technique as new.
