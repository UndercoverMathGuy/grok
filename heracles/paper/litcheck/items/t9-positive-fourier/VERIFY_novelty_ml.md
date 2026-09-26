# VERIFY (novelty_ml) — T9-positive-fourier (task 57)

Item: appendix app:scaling, eq:posfourier:
(1/p) sum_delta exp(-t F_y(delta)) >= prod_j e^{-t y_j} I_0(t y_j), because e^{z cos} has non-negative Bessel Fourier coefficients; resonances only add non-negative terms.

Checker verdict: PARTIALLY_KNOWN. **Upheld.**

## Pipeline note (why the verify stage is slow)
- WebSearch returned "this session has used its web search budget (200 of 200 WebSearch calls)" on my first query. The verifier subagents share one session budget, so every later verifier has no web search.
- export.arxiv.org/api/query returned HTTP 429 on all three parallel queries, so arXiv is rate-limiting this machine.
- WebFetch cannot parse the numdam/Tillich PDFs. It only saves the binary, so each one needs a local pdftotext pass.
- Workarounds I used: Hugging Face papers search (hf_fs) for discovery, curl + pdftotext for PDFs, and arxiv.org/abs pages, which worked where the API did not.
- Fix: raise CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION, or run the verifiers in fewer or smaller batches and throttle arXiv API calls.

## Confirmed citations, re-verified from the PDFs (fetched and text-extracted this session)
1. **Cioabă, "Closed walks and eigenvalues of Abelian Cayley graphs"**, C. R. Acad. Sci. Paris Ser. I 342 (2006) 635–638, doi:10.1016/j.crma.2006.03.005 (numdam PDF). The bib fields are correct.
   - Sec. 2 puts closed r-walks in bijection with r-tuples in S^r whose product is 1.
   - W_{2r} counts the 2r-tuples in which each x_i appears as often as x_i^{-1} and each involution appears an even number of times. Eq. (2) is Phi_{2r}(X) >= n W_{2r}(X), and Lemma 2.1 gives W_{2r} in closed form.
   - Thm 1.2 (Serre-type): at least C|G| eigenvalues are >= k - eps.
   - The checker's description is accurate. For odd prime p there are no involutions, so W_{2r} is exactly the closed-walk count on Z^K. Our inequality is the weighted, continuous-time exponential generating function of Cioabă's (2).
   - Cioabă proves only the unweighted combinatorial count. The weighted/Bessel form is our own.
2. **Friedman, Murty, Tillich, "Spectral estimates for Abelian Cayley graphs"**, JCTB 96 (2006) 111–121, doi:10.1016/j.jctb.2005.06.012 (Tillich's PDF). The bib fields are correct.
   - The abstract states lambda_2 >= d - O(d n^{-4/d}). For fixed d and a large odd prime n, the bound cannot be improved: most d-regular [abelian Cayley] graphs on prime n vertices have lambda_2 <= d - Omega(d n^{-4/d}).
   - Sec. 2 is the sphere-packing argument (Thm 2, Cor 3). Sec. 3 is the covering argument, which pushes a Dirichlet eigenfunction on the universal cover forward.
   - The folklore sentence is at p. 112 and cites "[AR94, Proposition 3, Section 3]".
   - Minor correction: the "large trace" approach is credited to [Ci05], which is **Cioabă's 2005 Queen's PhD thesis**, not the CRAS note.

## ML-lens adversarial search
- **McCracken, Moisescu-Pareja, Letourneau, Precup, Love, "Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks"**, arXiv 2505.18266 (abstract page fetched; PDF downloaded and read).
  - This is the closest ML result. Thm 4.7 and Cor 4.8 say that with m > (2 ln n - 2 ln(2-2rho))/(ln(pi/delta) - 1) random frequencies, h(i+j) - h(k) > delta*m for all wrong k with probability >= rho. So O(log n) frequencies give an Omega(log n) margin.
  - The proof in App. C is a heuristic: a Taylor approximation plus a volume/union bound over the points d*v near Z^m. It is the **sufficiency** direction, the analogue of our Alon–Roichman/Hoeffding half.
  - It states that the max-margin solution uses all frequencies (the Morwani view), and says in its limitations that it does not know why O(log n) arises.
  - It has no upper bound on the margin and no necessity (K >= c log p) argument. It never uses Bessel functions, exponential moments, or positivity of Fourier coefficients.
  - Conclusion: our eq:posfourier and its log p consequence are not in McCracken. McCracken should be cited next to Thm scaling as the empirical and sufficiency precedent for log n.
- **Bib error found:** refs.bib `mccracken2025` gives title "Emergence of sparse and dense circuits in modular arithmetic networks" and author "McCracken, Andrew and others" for arXiv:2505.18266. The arXiv page shows "Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks" by Gavin McCracken et al. (submitted 23 May 2025). Fix the bib.
- Also checked, all via HF paper search:
  - Morwani et al. 2311.07568: max margin uses all frequencies.
  - Gromov 2301.02679: the dense analytic solution.
  - He et al. 2602.16849: dense diversification, with no bound on the number of frequencies.
  - None of them has the positive-Fourier / Bessel inequality or a log p necessity bound.
- The ingredient "e^{z cos theta} = sum_k I_k(z) e^{ik theta} with I_k >= 0" is textbook: the Jacobi–Anger expansion, which also gives the von Mises characteristic function. Averaging a product of positive-definite factors over Z_p and keeping the constant term is the standard periodisation / heat-kernel-on-a-quotient comparison: p_t^{Z_p}(0) = sum over the lattice of p_t^{Z^K}(lambda) >= p_t^{Z^K}(0). The inequality itself is therefore not novel. What is new is the weighted version under the l_{1/s} budget, the psi <= a_s ln(1+z^r) step, and the log p conclusion.

## Limits of this check
- Web search was exhausted and arXiv API queries were rate-limited, so ML discovery went only through HF paper search. This is a lighter search than intended.
- I did not re-fetch Alon–Roichman. The checker cites Prop. 2 for abelian tightness, while FMT cite AR94 Prop. 3 for the folklore; the numbering should be checked before quoting.

## Verdict
PARTIALLY_KNOWN, upheld. The mechanism is known: the closed-walk / free-cover comparison (Cioabă 2006, eq. 2) and the covering argument (FMT 2006). The weighted, budget-normalised version and the Theta(log p) conclusion were not found in ML or in the fetched math sources.
- Keep the checker's recommended sentence.
- Add McCracken et al. beside Thm scaling as the sufficiency-only, heuristic ML precedent.
- Fix the mccracken2025 bib entry.
