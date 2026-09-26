# C5-chughtai-irreps — citation check

Verdict: PARTIALLY_SUPPORTED. The citation is correct. The wording claims more than the paper supports.

## Source (fetched 2026-09-25)
- PMLR page https://proceedings.mlr.press/v202/chughtai23a.html: Chughtai, Chan, Nanda, "A Toy Model of Universality: Reverse Engineering how Networks Learn Group Operations", ICML 2023, PMLR 202:6243-6267. The bib entry's metadata is correct. Add the pages and volume.
- Full text: arXiv 2302.03025 PDF. I extracted the text and read Tables 5 and 6, Figs 7 and 8, Sec 6 and App. C.

## What Chughtai actually report
- Setup (App. C): the networks are one-hidden-layer ReLU MLPs with separate left/right embeddings and an unembedding, plus a 1-layer transformer. Training uses AdamW with weight decay lambda=1. These are not our product/power neurons, and their models are not max-margin.
- Groups tested: C113, C118, D59, D61, S5, S6, A5. **S4 is not tested.**
- Key irreps (Table 5 for the MLP, Table 6 for the transformer, 4 seeds each):
  - S5 MLP: {sign, standard} x2, {sign, standard, 5d-b}, {sign, standard-sign, standard, 5d-a}.
  - S5 transformer: {sign, standard-sign} x2, {sign, standard} x2.
  - A5 MLP: {standard, 3d-a, 3d-b} x2, {3d-a, 5d-a, standard}, {5d-a, 3d-a, standard, 3d-b}.
  - A5 transformer: {3d-a, 3d-b} x2, {5d-a, 3d-b, 3d-a}, {standard}.
  - S6 MLP: 3-5 irreps. S6 transformer: {5d-b} or {sign, 5d-b}.
  - C113 MLP: 10-15 key frequencies (not "a few"). C113 transformer: 3-4.
- Fig 7 (S5 MLP, 50 seeds): "Most commonly we have two key representations (typically sign and standard), but sometimes we learn more." In the same figure, the 6d irrep is never learned.
- Sec 6: "Number of representations learned varies" across seeds. "Transformers consistently learn fewer representations than MLPs." The networks preferred standard over standard-sign, "when in fact standard sign offers better performance for fixed weight norm".
- App.: "We also use a factored architecture ... low-rank implicit bias, which may encourage a sparse number of representations to be learned."

## Checking the sentences
1. Intro, "a few irreducible representations \citep{chughtai2023toy}": supported for S5, S6 and A5 (1-5 irreps, modally 2 on S5). It overstates for their cyclic and dihedral MLPs, which use 9-15 irreps. Scope it to the non-abelian groups.
2. Abstract, "optimums consistent with Chughtai": partly consistent.
   - S5: our "one faithful irrep + sign patch" matches the modal {sign, standard}. Their MLPs sometimes add 5d / standard-sign, though.
   - A5: our "Galois-conjugate pair {3d-a, 3d-b}" matches only 2 of the 4 transformer seeds. Every one of their MLP seeds also contains the 4d standard irrep (3-4 irreps in total).
   - S4: they have no data for it.
   - Their architecture and optimiser differ from our model class. They also stress that the count varies from seed to seed.
3. groups.tex, "one representation on S5 and the Galois-conjugate pair on A5, as observed": if "as observed" means Chughtai, it holds for S5 (modulo sign) and only for the A5 transformers. If it means our own runs, say so.

## Fix
- Intro: "...or a few irreducible representations on non-abelian groups such as S5 and A5 \citep{chughtai2023toy}".
- Abstract: "The predicted optima are also consistent with the dominant representations reported by \citet{chughtai2023toy} (sign and standard on S5)." Alternatively, drop the A5 claim, or qualify it to "the 3d Galois pair in their transformer runs".
- Add "pages 6243--6267, volume 202, PMLR" to the bib entry.
