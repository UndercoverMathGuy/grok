# VERIFY (novelty_ml): T8-random-frequencies, task 55

Claim (appendix app:scaling, lower bound): draw a random multiset of L = ceil(8 ln p) frequencies. Hoeffding plus a union bound over the p-1 offsets gives every offset coverage >= L/2, so Gamma_s(p) >= (1/2) L^{1-s}.

## Verdict: KNOWN, and the checker is upheld

The technique is textbook, and the paper already says so. The ML literature has a direct precedent for the same random-cosine statement in modular addition (McCracken et al. 2025). No novelty should be claimed for this lemma. Only the budget normalisation into Gamma_s is new.

## Math re-check
- Setup: Fp = {1..(p-1)/2}, and the sum over Fp of cos(2 pi j delta/p) is -1/2, so E cos = -1/(p-1).
- Hoeffding (Thm 2, range 2) gives exp(-L t^2/2) with t = 1/2 + 1/(p-1) >= 1/2. That is at most e^{-L/8}, which is at most 1/p.
- The union bound gives (p-1)/p < 1.
- The arithmetic is correct. One slack: the offsets delta and -delta give the same cosine, so only (p-1)/2 events are needed.

## Citations verified this session
1. Alon & Roichman, "Random Cayley graphs and expanders", RSA 5(2):271-284, 1994, doi 10.1002/rsa.3240050203.
   - Crossref metadata matches the bib. I read the text of the PDF from web.math.princeton.edu/~nalon/PDFS/exp1.pdf.
   - Sec 3 Thm 2 (Abelian groups): a multi-subset S of (1+eps) log2 n elements, "not necessarily distinct", chosen with repetitions.
   - The proof bounds each nontrivial character, then says "There are n-1 nontrivial characters and hence the probability that at [least one]..." (a union bound).
   - The random model and the per-character plus union-bound structure match our lemma. Their per-character tail is a binomial/Chernoff count, not Hoeffding, so "cyclic case of the argument" is fair. CONFIRMED. The DOI can be added.
2. McCracken, Moisescu-Pareja, Letourneau, Precup, Love, "Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks", arXiv 2505.18266 (v1 23 May 2025).
   - Title and authors confirmed on the abs page. The EXISTING bib entry is WRONG: it has the title "Emergence of sparse and dense circuits...", the author "McCracken, Andrew", and "others".
   - Thm 4.7 (HTML v1): h(k) = sum_{l<=m} cos(2 pi f_l (k-i-j)/n), where the f_l are drawn uniformly from {1..n/2}. If m > [2 ln n - 2 ln(2-2 rho)] / [ln(pi/delta) - 1], then with probability >= rho, h(i+j) - h(k) > delta m for all k != i+j.
   - Cor 4.8: O(log n) distinct frequencies give an Omega(log n) logit margin.
   - The checker's transcription of the inequality is garbled (it dropped the fraction bar). The substance is correct.
   - At delta = 1/2, McCracken needs m of about 2.4 ln n, versus our 8 ln p. Our constant is loose by comparison, which a reviewer could notice.
   - CONFIRMED as the direct ML precedent. It belongs next to the lemma.
3. Hoeffding 1963, JASA 58(301):13-30. Crossref metadata matches the existing bib.
4. Naor & Naor, SICOMP 22(4):838-856, 1993 (10.1137/0222053), and Alon-Goldreich-Hastad-Peralta, RSA 3(3):289-304, 1992 (10.1002/rsa.3240030308). Crossref metadata matches the checker's bibtex.
   - Caveat: both work over Z_2^n, and eps-bias is a TWO-sided bound |E chi| <= eps.
   - Our condition is one-sided (mean cos <= 1/2), so the proposed sentence "(one-sided) 1/2-biased multiset of Z_p \citep{naor1993small}" stretches the definition.
   - Keep it optional, or phrase it as "a one-sided analogue of small-bias sets".

## Adversarial ML search (limited: see process note)
- The arXiv API query (abs "modular addition" AND margin) turned up:
  - Morwani 2311.07568
  - Li et al. 2402.09469
  - Huang-Li 2511.23443 (sine MLPs, width-2 exact)
  - two 2026 grokking papers
  None of them has a random-frequency margin lemma.
- The Semantic Scholar list of papers citing 2505.18266 (14 entries, including 2602.03655, 2512.25060, 2606.17399 and 2605.07648): no abstract claims a random-frequency or log-p construction.
- HF papers search (3 queries): nothing beyond McCracken, Morwani and He et al. 2602.16849.
- No further ML precedent found beyond McCracken.

## Process note (relevant to the verify stage being slow)
- WebSearch returned "session has used its web search budget (200 of 200)" on my FIRST call. The verify agents share one per-session budget, which is already exhausted, so every later verifier has to improvise through fallbacks.
- export.arxiv.org and api.semanticscholar.org returned HTTP 429, which points to parallel agents hitting rate limits.
- These cause slow and degraded verifiers. Raising CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION and lowering the verify concurrency (or serialising arXiv API calls) should help.
