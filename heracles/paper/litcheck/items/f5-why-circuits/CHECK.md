# F5-why-circuits — intro para 1 sentence 1

Statement: "Mechanistic interpretability finds the circuits trained networks implement, but typically leaves open why they arise."

## Verdict: SUPPORTED (by an existing bib entry). Cite it, because the wording is nearly the same.

### Primary support: Morwani et al., ICLR 2024 (arXiv 2311.07568; bib key morwani2024feature is already in refs.bib)
- Introduction, paragraph 1, verbatim from the PDF (v2): "The research area of mechanistic interpretability (Olah et al., 2020; Elhage et al., 2021; Olsson et al., 2022; Elhage et al., 2022) aims to dissect individual trained neural networks ... identifying and interpreting sub-circuits ... **Mechanistic interpretability analyses typically leave open the question of why the observed representations arise as a result of training.**"
- Abstract: "... this paper explores a complementary question -- why do networks arrive at particular computational strategies?"
- Our sentence is a close paraphrase of theirs, so we need to attribute it. Leaving it uncited reads as borrowed wording.

### Secondary support: Sharkey et al. 2025, "Open Problems in Mechanistic Interpretability" (arXiv 2501.16496)
- Sec 2.1.2(d): "exploring theories about why neural networks generalize appears to be a promising avenue"; "... a more dynamic, developmental view of how mechanistic structure emerges in the training process".
- Sec 3.3.2 poses forecasting "capabilities that arise during training" as an open question.
- This source supports "open" but is less direct than Morwani.

### Citations for "finds the circuits"
- Olah et al. 2020, "Zoom In: An Introduction to Circuits", Distill, doi 10.23915/distill.00024.001. Metadata fetched from distill.pub.
- Elhage et al. 2021, "A Mathematical Framework for Transformer Circuits", transformer-circuits.pub. Title and authors fetched.
- nanda2023progress and chughtai2023toy are already in the bib and serve as the modular-addition and group examples.

### Checked and not useful
Bereska & Gavves (TMLR 2024, arXiv 2404.14082). The fetched abstract does not frame why circuits form as an open problem.

## Recommended edit
"Mechanistic interpretability finds the circuits trained networks implement \citep{olah2020zoom,elhage2021mathematical,nanda2023progress}, but typically leaves open \emph{why} they arise \citep{morwani2024feature}."
Morwani is cited again in the next sentence, so another option is to keep the citation there and write "As \citet{morwani2024feature} note, ...".
