# R34 conjecture-known (appendix app:conjecture) — literature check

Verdict: NOVEL_AS_FAR_AS_FOUND (the conjecture is open and so far only internal). One piece is partly known: the b = nu end of the "window".

## What the paper says (sections/appendix.tex l.287-297)
- Conjecture: for product nets, the interference-free value is the global maximum.
- Known results: p=3 for all n; the window nu-4/(3(p-1)) <= b <= nu; b >= 5/2 at p=5; the single-frequency class at n=2, b=2 when the "certified aiming depth" is >= 0.3476; a strict local max; an interfering local max at p=5 (0.837); the group analogue is false on Z12/D3/S4 (1.06-1.62x).
- The source has a TODO comment on l.294: "certified aiming depth" is not defined anywhere in the paper. The appendix gives no proofs of the "Known" items.

## Searches run (2026-09-25)
1. global maximum margin modular addition network sparse Fourier features conjecture interference proof
2. "modular addition" max-margin "local maximum" not global multilinear network Fourier sparse optimum 2026 arXiv
3. max-min trigonometric polynomial nonnegative coefficients l_q budget maximize minimum sum beta_k(1-cos k s) optimal support
4. "MAIS" open problem maximum margin modular addition global optimum weight decay sparse frequencies Levine 2026

## Findings
- Morwani et al. ICLR 2024 (arXiv 2311.07568), Thm 7: the global max of the L_{2,3} margin for the quadratic net (b = nu = 3). Every frequency is used, and the result is certified by a dual (Lemma 2/6).
- Li et al. AISTATS 2025 (arXiv 2402.09469), Thm 4.1: the same for power neurons at L_{2,k+1} (b = nu).
- Both results sit at the endpoint b = nu of the paper's window. At that endpoint the global optimum over ALL networks, interference included, is already known for the power/quadratic model. The paper's own appendix (l.279) says the bound is attained by the Morwani net at n=2, b=3.
- So the window claim for the product model near b = nu is an extension of the Morwani/Li certificate approach. The endpoint should be credited to them.
- arXiv 2608.04451 (Memana, Aug 2026) is a counterexample to MAIS-O60. It is about single-neuron ReLU dynamics and has nothing to do with the global margin optimum. Not relevant.
- No paper found proves or disproves that the interference-free / sparse value is the global max for b < nu. None exhibits an interfering local max, and none reports a group-level counterexample.
- The harmonic-analysis search (max-min cosine polynomials under an l_q budget) turned up only Littlewood/Fejer-type extremal results. None of them settles this programme.

## Internal issues (not literature)
- "certified aiming depth" is undefined. Define it or drop the clause.
- The "Known" claims have no proofs in the appendix. Add proof sketches or pointers (P3_THEOREM.md, junky local max at p=5, and so on), or soften them to "verified numerically / proved in supplementary".
- "true in the window ... for every p" needs a proof or a citation to an internal lemma.

## Recommended change
- In app:conjecture, add: "At the endpoint b = nu the global maximum over all networks is known (Morwani et al., 2024, Thm 7; Li et al., 2025, Thm 4.1), and it is dense. Our window result extends that certificate a distance O(1/p) below b = nu."
- Define "certified aiming depth", or remove that sentence.
