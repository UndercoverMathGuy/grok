# VERIFY (citation lens): F5-why-circuits

Claim (intro.tex:4, and word for word again at prior.tex:7): "Mechanistic interpretability finds the circuits trained networks implement, but typically leaves open *why* they arise." Neither copy has a citation.

## Citations I fetched myself
- **morwani2024feature** (arXiv 2311.07568v2, PDF text extracted). Intro para 1: "Mechanistic interpretability analyses typically leave open the question of why the observed representations arise as a result of training." The abstract says it "explores a complementary question – why do networks arrive at particular computational strategies?" CONFIRMED. Our sentence is a near-paraphrase of this line, so it needs this citation.
  Venue: the OpenReview API has note i9wDX850jR, "ICLR 2024 spotlight". The refs.bib fields (title, authors, ICLR 2024) are correct.
- **olah2020zoom** (Crossref, DOI 10.23915/distill.00024.001). Title "Zoom In: An Introduction to Circuits". Authors: Olah, Cammarata, Schubert, Goh, Petrov, Carter. Distill vol 5, issue 3, 2020-03-10. The bibtex is correct; volume=5 and number=3 could be added. (The distill.pub page was over the fetch size limit, so I used the DOI metadata.)
- **elhage2021mathematical** (transformer-circuits.pub). Title, venue (Transformer Circuits Thread, 2021-12-22) and author order (Elhage, Nanda, Olsson, Henighan, Joseph, Mann, Askell, Bai, Chen, Conerly, ...) all match. The paper's stated aim is to "reverse engineer" transformers. CONFIRMED.
- **sharkey2025open** (arXiv 2501.16496). Title and the first 10 authors in order match (29 authors in total). It was submitted on 2025-01-27 and is arXiv-only. Sec 2.1.2(d) contains "take a more dynamic, developmental view of how mechanistic structure emerges in the training process". This is real but WEAK support. It asks for theory and for developmental methods. It does not say that MI leaves open why circuits arise. It is optional at most.

## Verdict
SUPPORTED, and the checker's verdict stands. Add \citep{morwani2024feature} to the "leaves open why" clause, in both intro.tex and prior.tex. The "finds circuits" clause can take olah2020zoom, elhage2021mathematical and nanda2023progress (already in refs.bib). Drop Sharkey, or keep it only as a secondary "see also".
Separately, the same sentence appears in both intro.tex:4 and prior.tex:7. That is copy-paste duplication and should be fixed either way.
