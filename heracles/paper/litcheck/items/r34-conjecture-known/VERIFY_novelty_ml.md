# VERIFY (novelty_ml): R34-conjecture-known, task 37

Verdict: **upheld**. The checker's call, NOVEL_AS_FAR_AS_FOUND, stands. Both confirmed citations exist and say what the checker claims. One scoping correction to the recommended edit (a) is given below.

## Tooling note (for the orchestrator)
- WebSearch was unavailable. The first call returned "this session has used its web search budget (200 of 200 WebSearch calls)". Every verifier in this session shares that cap.
- The arXiv export API returned HTTP 429 after a single query, because parallel verifiers are hitting it.
- Search was therefore done with WebFetch on arxiv.org/search (abstract field), Semantic Scholar citation lists (curl), and direct PDF reads.
- The scratchpad is shared. My first PMLR PDF guess (a 404 page) was written to scratchpad/li.pdf and overwrote whatever another agent had there. All later files went to scratchpad/t037/.

## What I searched (2026-09-25)
- arxiv.org/search, abstract field:
  - "modular addition" margin: 6 hits. They are Li 2402.09469, Morwani 2311.07568, Huang-Li 2511.23443 (sine nets), 2606.05863, 2608.07436 and one quantum paper. None concerns b<nu global optima, interference, or local maxima.
  - "modular addition" "weight decay" frequencies: 2606.12966, 2606.08985 and 2602.16849. Abstracts fetched, none relevant.
  - "margin maximization" Fourier features: only Morwani and Li.
  - The following queries returned 0 hits: grokking margin sparse frequencies; max-margin "local maximum" neural network global; modular addition sparse frequencies regularization optimal; margin "irreducible representations" neural network group.
- Semantic Scholar citation lists: all 40 indexed citers of Morwani 2311.07568 and all 22 of Li 2402.09469. I screened the abstracts and fetched the plausible ones:
  - Tian 2410.01779 (NeurIPS 2025, CoGS): global solutions for **L2 loss**, quadratic nets. It is not about margin and does not compare interference-free vs interfering solutions.
  - Tian 2509.21519 (Li_2 scaling laws): local maxima of an energy E in the dynamics. It says nothing about global margin optimality.
  - 2602.04548 (Yarotsky et al.): CP-decomposition gradient flow. Irrelevant.
  - 2602.03655 (Marchetti et al.): sequential group composition, irreps learned one at a time. No global optimality result and no counterexamples.
  - 2606.08985 (Tan-Gai-Zhang): cyclic vs simplex geometry. Irrelevant.
  - 2511.23443 (Huang-Li): sine nets, margin generalization bound. It does not characterize the max-margin solution.

None of these proves or refutes that the interference-free value is the global max for 2<=b<nu. None has an interfering local max, and none has a cyclic/dihedral/S4 counterexample in which interference beats the interference-free ceiling.

## Confirmed citations, checked myself (PDFs read)
- **morwani2024feature** (arXiv 2311.07568v2, PDF read). Thm 7 covers one-hidden-layer quadratic nets (u(a)+v(b))^2 w with m >= 4(p-1) neurons. It states:
  - The maximum L_{2,3}-margin is sqrt(2/27) / (p^{1/2}(p-1)).
  - Every maximizer has single-frequency neurons.
  - Every frequency is used.
  - The certificate is Lemma 6 (with Lemma 2), from Wei et al. 2019 Thm 4.1.

  The bib fields match refs.bib (ICLR 2024).
- **li2025fourier** (PMLR 258:523-531; the proceedings page was fetched and title, authors, volume, pages and year all match; arXiv 2402.09469v4 PDF read). Thm 4.1 covers power neurons (sum_i u_i(a_i))^k w with m >= 2^{2k-1}(p-1)/2. It gives:
  - The max L_{2,k+1} margin: 2(k!) / ((2k+2)^{(k+1)/2} (p-1) p^{(k-1)/2}).
  - Single-frequency neurons, with all frequencies used.

  **Duplicate key:** refs.bib already holds this paper as `li2025nsum`. Reuse that key and do not add `li2025fourier`. The existing entry could gain `series=PMLR, volume=258, pages=523--531`.

## Issues with the checker's recommended change
1. **Scope of edit (a).** Morwani Thm 7 and Li Thm 4.1 are theorems about **power** (quadratic) neurons. The conjecture and the window are about **product** networks. For product networks the b=nu endpoint is the paper's own result: Thm universal + Cor. cor:dense + Prop. block at flat y (appendix.tex l.32). Suggested wording:

   > At b = nu the global maximum is known and dense: for power networks by Morwani et al. (2024, Thm 7; n=2) and Li et al. (2025, Thm 4.1), and for product networks by Corollary cor:dense.

   The checker's description of the window as "extending the certificate O(1/p) below b=nu" is its own reading. No proof is in the paper to confirm it.
2. The internal gaps the checker lists remain true: "certified aiming depth" is undefined, and there is a TODO at appendix.tex l.293-294. None of the "Known" items has a proof or pointer. These are not literature problems.

## Missed prior art
None found that settles or pre-empts the conjecture. Tian (NeurIPS 2025, 2410.01779) is the nearest "global solutions for modular-addition nets" work, but it is about L2 loss rather than margin. The r20 and f10 items already recommend it for related work. It is not needed in app:conjecture.
