# VERIFY (citation lens): C5-chughtai-irreps (task 79)

## Sources I fetched this session
- PMLR page: https://proceedings.mlr.press/v202/chughtai23a.html. Title "A Toy Model of Universality: Reverse Engineering how Networks Learn Group Operations"; Chughtai, Chan, Nanda; ICML 2023, PMLR 202, pp. 6243-6267.
- Full text of arXiv 2302.03025, from the PDF via pdftotext.
- Stander et al., PMLR 235:46441-46467 (ICML 2024), https://proceedings.mlr.press/v235/stander24a.html (arXiv 2312.06581).
- Wu, Jaburi, Drori, Gross, ICLR 2025, https://arxiv.org/abs/2410.07476.
- Tian, NeurIPS 2025, https://arxiv.org/abs/2410.01779. Abelian only, quadratic activation, L2 loss, so it does not apply to the group claim.

## What Chughtai says (checked in the text myself)
- **Setup (App. C).** One-hidden-layer ReLU MLP (d=256 embeddings, h=128) and a 1-layer ReLU transformer. Trained on 40% of the table with AdamW, wd lambda=1, lr 1e-3, full batch.
- **Groups tested:** C113, C118, D59, D61, S5, S6, A5. S4 is not tested.
- **Table 5, MLP, S5:** {sign, standard-sign, standard, 5d-a}; {sign, standard}; {sign, standard, 5d-b}; {sign, standard}.
- **Table 5, MLP, A5:** {standard, 3d-a, 3d-b} x2; {3d-a, 5d-a, standard}; {5d-a, 3d-a, standard, 3d-b}. The 4d standard irrep is in all 4 seeds.
- **Table 5, MLP, cyclic and dihedral:** C113 uses 10-15 key frequencies, D59/D61 use 9-15.
- **Table 6, transformer, S5:** {sign, standard-sign} x2 and {sign, standard} x2. So all 4 transformer seeds are sign plus exactly ONE 4d faithful irrep. The checker listed these rows but did not draw this conclusion; it is stronger support for "one representation on S5".
- **Table 6, transformer, A5:** {3d-a, 3d-b} x2; {5d-a, 3d-b, 3d-a}; {standard}. The 3-irrep seed has test loss 1.95e-01, so it did not fully generalize.
- **Fig 7 (S5 MLP, 50 seeds):** "Most commonly we have two key representations (typically sign and standard), but sometimes we learn more."
- **Sec 6:** "Number of representations learned varies" and "Transformers consistently learn fewer representations than MLPs". Networks preferred standard over standard-sign "when in fact standard sign offers better performance for fixed weight norm".

## Verdict on the checker
I uphold PARTIALLY_SUPPORTED. Every factual statement in the checker's evidence matches the text. The recommended bibtex (PMLR 202, 6243-6267) matches the PMLR page. The current refs.bib entry lacks volume and pages; it is correct otherwise.

Refinements:
1. **S5.** Support is stronger than the checker says: 4/4 transformer seeds, 2/4 MLP seeds, and the Fig 7 mode are all sign plus one 4d irrep.
2. **A5.** The Galois 3d pair alone appears in 2/4 transformer seeds. It appears in 3/4 if we count the non-generalized seed 3. It appears in 0/4 MLP seeds as the whole committee, because the 4d standard irrep is always added. "As observed" for A5 must be scoped to transformers, or attributed to the paper's own runs.
3. **"A few irreducible representations" (intro).** This is fine for S5/S6/A5 and for their transformers generally. For their cyclic and dihedral MLPs (9-15) it is too broad. Scope it to non-abelian groups.
4. **groups.tex.** S4 appears only in the structured-interference sentence, which is not attributed to Chughtai. Still, no text should imply Chughtai tested S4.
5. **Architecture.** Chughtai's models are ReLU/AdamW, not max-margin product networks, so "consistent with" is the right strength. Do not say "predicts".

## Better or extra citations (optional)
- Stander et al. (ICML 2024) reverse-engineer S5/S6 MLPs as coset circuits and contrast this with Chughtai's irreps account.
- Wu et al. (ICLR 2025) unify the two accounts.
If the paper says networks "use a few irreps", citing Stander and Wu next to Chughtai heads off a reviewer who favors the coset reading. Chughtai remains the canonical source for irrep counts per seed.

## Harness note (re: slowness)
WebSearch returned "this session has used its web search budget (200 of 200 WebSearch calls)". The budget is shared across the workflow session, so verifier agents now cannot search and must fall back to direct WebFetch. That likely explains part of the slow or low-yield verify stage. Raising CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION, or capping searches per task, would help.
