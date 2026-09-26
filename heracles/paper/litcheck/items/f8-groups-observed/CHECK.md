# F8-groups-observed: "one representation on S5 and the Galois-conjugate pair on A5, as observed"

Where: groups.tex line 4. The sentence has no citation. Verdict: PARTIALLY_SUPPORTED.

## Chughtai, Chan, Nanda, ICML 2023 (PMLR 202), arXiv 2302.03025
I read the full PDF text.
- Sec 5: on the mainline S5 model, the logits are explained by the characters of "standard" and "sign". The paper calls these "the key representations". So the mainline model uses one faithful irrep plus sign.
- Table 6 (Transformer, 4 seeds per group). Key representations, listed in the order learned:
  - S5: {sign, standard-sign} x2, {sign, standard} x2. That is exactly one faithful 4d irrep plus sign in 4 of 4 seeds.
  - A5: {3d-a, 3d-b}, {3d-a, 3d-b}, {5d-a, 3d-b, 3d-a}, {standard}. The Galois-conjugate 3d pair appears in 3 of 4 seeds, and alone in 2 of 4. One seed uses only the 4d standard irrep.
- Table 5 (MLP, 4 seeds). S5 uses 2 to 5 key representations. A5 is {standard, 3d-a, 3d-b} x2, {3d-a, 5d-a, standard}, {5d-a, 3d-a, standard, 3d-b}.
- Fig 7 (MLP, 50 seeds on S5): "Most commonly we have two key representations (typically sign and standard), but sometimes we learn more."
- Sec 6: "Number of representations learned varies ... Transformers consistently learn fewer representations than MLPs."
- App D.1.4: the 6d irrep of S5 splits into two 3d irreps of A5. These two are the Galois-conjugate pair.

## Stander, Yu, Fan, Biderman, ICML 2024 (PMLR 235:46441-46467)
- This paper covers S5 and S6 only. It has no A5 runs.
- It reports that neuron activations "concentrated on a single irrep" (Sec 7). It explains this as coset circuits rather than GCR.
- It does not count irreps per network, so it does not directly support "one representation".
- It is useful as a caveat: irrep concentration may be an artefact of coset structure.

## Assessment
- The prior literature supports the claim for transformers: 4/4 seeds on S5, and 2 to 3 of 4 seeds on A5. The S5 count is "one faithful irrep + sign", not literally one representation.
- The claim is contradicted, or only loosely supported, for MLPs. In Chughtai's ReLU MLPs the S5 count is 2 to 5 key representations, and A5 adds the 4d standard irrep.
- The paper's own runs (S5: 32/32 one faithful irrep + sign patch; A5: Galois-conjugate pair) are presumably the "observed". If so, the text should say that and point to the experiment section.
