# VERIFY (citation lens): C14 alon1994random

Sentence: sections/appendix.tex l.196 (app:scaling, lower bound): "This is the cyclic case of the random Cayley graph
argument of \citet{alon1994random}."

## Sources fetched this session
- Crossref: https://api.crossref.org/works/10.1002/rsa.3240050203. Title, authors, RSA 5(2):271-284, 1994 and the abstract all match.
- Full text, author copy: https://m.tau.ac.il/~nogaa/PDFS/exp1.pdf, linked from Alon's publications list
  (https://m.tau.ac.il/~nogaa/PDFS/publications.html, which also gives "Random Structures and Algorithms 5 (1994), 271-284").
  I downloaded it and read Sections 1 and 3. The earlier checker could not get the full text.
- arXiv 2505.18266 (abs page and HTML v1), because the checker recommends citing it at the same sentence.

## Bib entry: CORRECT
The title, authors, journal, volume, number, pages and year all match Crossref. Optionally add doi={10.1002/rsa.3240050203}.

## Content: SUPPORTED, and more precisely pinned than the checker managed
- Theorem 1 (general groups): E|mu*_1[X(G,S)]| < 1-delta for |S| = c(delta) log2 n random elements. It is proved with Wigner's
  trace method (closed random walks, Lemma 1), NOT with Chernoff over characters.
- Section 3, Theorem 2 (abelian groups): with (1+eps) log2 n random elements (multiset, drawn with repetition),
  |mu*_1| <= 1-delta almost surely. The proof works one character at a time. Eigenvalues are character sums
  sum_{s in S u S^-1} chi(s). Independence gives a tail bound for each fixed nontrivial character, then a union bound
  runs over the n-1 nontrivial characters. This is the structure of our Hoeffding plus union-bound argument over
  delta != 0 on Z_p, and like ours it draws with repetition.
- The Remark after Theorem 2 treats the cyclic group Z_n explicitly: eps log2 n random elements already give |mu*_1| <= 1-delta(eps).
- Proposition 3: for any abelian group, a delta-expander Cayley graph needs |S| >= c(delta) log2 n. This is the source
  of "essentially tight for abelian groups".
- So "cyclic case of the random Cayley graph argument" is accurate. The best pinpoint is Section 3 / Theorem 2 (the abelian
  argument), not Theorem 1, whose proof technique is different.

## Corrections to the checker
1. The checker says Alon-Roichman bound the second-LARGEST eigenvalue one-sidedly. In fact the body defines mu_1 as the
   second largest eigenvalue IN ABSOLUTE VALUE, and Theorems 1 and 2 both bound |mu*_1|. Only the abstract's wording is
   one-sided. Their bound is two-sided and therefore stronger than our one-sided need (cos-average <= 1/2), so the
   attribution still holds.
2. The checker says "for abelian groups the AR proof is the same Chernoff bound". This is right in structure: a tail bound
   per character plus a union bound. They use a binomial count with the 1.8 threshold rather than Hoeffding, and it is
   Theorem 2, not Theorem 1.
3. The "analogue of our matching upper bound" clause in the recommended change should cite Proposition 3. It is an analogy,
   not an implication, because Prop 3 is a diameter-counting bound for unweighted expanders, while our upper bound concerns
   weighted allocations. It is fine as a remark if it is worded as an analogy.

## Side finding (not this item, but the recommended change points to it)
The refs.bib entry mccracken2025 is WRONG:
- The bib gives title "Emergence of sparse and dense circuits in modular arithmetic networks" and author "McCracken, Andrew and others".
- arXiv 2505.18266 is actually "Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks" by Gavin McCracken,
  Gabriela Moisescu-Pareja, Vincent Letourneau, Doina Precup and Jonathan Love (v1 23 May 2025).
- Its Theorem 4.7 (m > ... log n random frequencies give h(i+j)-h(k) > delta m with probability >= rho) and Corollary 4.8
  ("Learning O(log n) distinct frequencies gives a logit margin Omega(log n)") exist as the checker describes. That makes it
  a fair ML precedent at this sentence, but the bib entry must be fixed first. It is also cited in intro.tex:6 and prior.tex:9.

## Suggested sentence
"This is the cyclic case of the random Cayley graph argument of \citet[Sec.~3, Thm.~2]{alon1994random}: the averages
$\frac1L\sum_\ell\cos(2\pi J_\ell\delta/p)$ are the nontrivial normalized eigenvalues of $\mathrm{Cay}(\sZ_p,\{\pm J_\ell\})$,
and their abelian-group proof is likewise a per-character tail bound plus a union bound over the $p-1$ nontrivial characters;
\citet{mccracken2025} (Thm.~4.7) use $O(\log p)$ random frequencies for the same purpose in trained networks."

## Process note (for the orchestrator)
WebSearch returned "this session has used its web search budget (200 of 200 WebSearch calls)". Verifier subagents in this
workflow appear to share one exhausted search budget. Each subagent then falls back to probing URLs by hand, which is
likely part of why the verify stage is slow. Raising CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION, or running fewer verifiers
per session, would help.
