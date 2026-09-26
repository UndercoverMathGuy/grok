# VERIFY (citation lens): F8-groups-observed (task 72)

Claim (groups.tex:4): "... fixes the number of features: one representation on $S_5$ and the Galois-conjugate pair on $A_5$, as observed." The sentence has no citation.

## What I fetched myself this session
- Chughtai, Chan, Nanda, arXiv:2302.03025v2. I read the full PDF with pdftotext.
- PMLR page https://proceedings.mlr.press/v202/chughtai23a.html: PMLR 202, pages 6243-6267, ICML 2023.
- Stander, Yu, Fan, Biderman, PMLR 235 PDF, and the page https://proceedings.mlr.press/v235/stander24a.html: pages 46441-46467, ICML 2024. The fields match the checker's bibtex.

## Chughtai et al. (checker's reading confirmed)
- **Table 6 (Transformer), S5:** {sign, standard-sign} in seeds 1-2 and {sign, standard} in seeds 3-4. Every seed uses one faithful 4d irrep plus sign.
- **Table 6, A5:**
  - Seeds 1-2 use {3d-a, 3d-b}.
  - Seed 3 uses {5d-a, 3d-b, 3d-a}, and its test loss is 1.95e-01, so it did not fully generalise.
  - Seed 4 uses {standard}.
  - So the exact conjugate pair appears in 2/4 seeds, and the pair is contained in 3/4.
- **Table 5 (MLP):**
  - S5 uses 2-4 key representations: {sign, standard-sign, standard, 5d-a}, {sign, standard}, {sign, standard, 5d-b}, {sign, standard}.
  - A5 always includes the 4d standard irrep. The pair (both 3d-a and 3d-b) appears in seeds 1, 2 and 4. Seed 3 has 3d-a only.
- **Fig. 7 (50 MLP seeds on S5):** "Most commonly we have two key representations (typically sign and standard), but sometimes we learn more."
- **Sec. 6:** "Transformers consistently learn fewer representations than MLPs."
- **App. D.1.4:** the 6d irrep of S5 splits into two 3d irreps of A5.
- The paper never uses the words "Galois conjugate". That the two 3d characters are exchanged by sqrt5 -> -sqrt5 is a textbook fact, not something Chughtai states.

## Stander et al.
- The paper studies S5 and S6 only, with no A5 runs. It uses ReLU MLPs in Chughtai's setup and aggregates 128 S5 runs.
- **Sec. 7:** "neurons concentrated on a single irrep". This is a per-neuron statement, not a count of irreps per network.
- **Sec. 7 also says:** "D10 4 will have coset circuits concentrated on both (3, 2) or (2, 2, 1), depending on the run." That is, the 5d irreps are used in some runs.
- It gives no support for "one representation per network". At most it is a caveat: Stander reinterprets the irrep concentration as coset circuits.

## Assessment
PARTIALLY_SUPPORTED is upheld.
- Prior work supports the sentence only for transformers, and only if "one representation" means one faithful 4d irrep plus the sign character.
- For A5 the support is 2/4 exact and 3/4 containing the pair, with one of those seeds not fully generalised.
- ReLU MLPs contradict the sentence: they use more irreps on S5, and on A5 they always add the 4d standard irrep.
- None of these are max-margin product/power networks, so the match is only suggestive.

## Problem with the checker's recommended change
- The rewrite points "as observed in our runs" at Section~\ref{sec:experiments}.
- The current experiments.tex, 11 lines of TODO subsections, has no group experiments: it covers only the interference fraction, the s-exponent and the degree/norm dials.
- The paper therefore has no in-paper referent for "as observed". The heracles S5/A5 runs (S5 32/32 one irrep + sign; A5 conjugate pair) are not in the draft.
- **Either** add a group paragraph or table (experiments or appendix) **or** make the sentence rest on the Chughtai Table 6 citation alone.

## Bib fixes
- **chughtai2023toy:** add series = PMLR, volume = 202, pages = 6243--6267. The title capitalisation should match PMLR: "A Toy Model of Universality: Reverse Engineering how Networks Learn Group Operations".
- **stander2024grokking:** not in refs.bib. The checker's entry is correct: PMLR 235, 46441--46467, 2024.

## Suggested sentence
"... one faithful representation (plus the sign character) on $S_5$ and the Galois-conjugate pair of 3-dimensional representations on $A_5$, as in the transformer runs of \citet[Table 6]{chughtai2023toy}. ReLU MLPs there use more representations \citep[Fig. 7, Table 5]{chughtai2023toy}, and \citet{stander2024grokking} read the irrep concentration on $S_5$ as coset circuits."

Add "and in our runs (Appendix X)" only once those runs are in the paper.

## Process note (relevant to the workflow being slow)
When I first called WebSearch in this verifier it returned "this session has used its web search budget (200 of 200 WebSearch calls)". So WebSearch is exhausted for the whole session. Any verifier with the novelty lenses (novelty_ml, novelty_math) that depends on search will fail or keep retrying from here on. This verifier fell back to curl + pdftotext and direct WebFetch.
