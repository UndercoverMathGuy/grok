# VERIFY (citation lens): F1-long-desired, task 65

Claim (paper.tex:56, abstract sentence 1): "Explaining the number of features in a network has been a long-desired result in mechanistic interpretability". refs.bib has no Olah, Elhage or Sharkey entry, so the sentence has no citation.

Checker verdict: PARTIALLY_SUPPORTED. **Upheld**, with two corrections: one bib fix and one misattributed quote.

## elhage2022superposition (Toy Models of Superposition): CONFIRMED
- I downloaded the page with curl (8.2 MB) and stripped the HTML myself.
- Published Sept 14, 2022, Transformer Circuits Thread.
- The checker's bibtex is exactly the page's official "BibTeX Citation" block: same key, authors and note.
- The Open Questions section is authors' main text. It is introduced as "questions which strike us as most important". It says, verbatim: "How many features should we expect to be stored in superposition? ... it would be nice to have a clearer understanding – and perhaps tighter bounds!" The quote is 2.4k characters after the heading and before the Comments & Replications body.
- Also main text (Feature Dimensionality): "one question is just how many features the model learns to represent".
- **Misattribution.** The checker also cites "we ultimately want to count the features used by real neural networks, where the number of features is unknown." That sentence is NOT by Elhage et al. It sits in the Comments & Replications section, in the entry "Extracting Features with Sparse Autoencoders" by Lee Sharkey, Dan Braun and Beren Millidge (Conjecture). Do not attribute it to elhage2022superposition.

## sharkey2025open (Open Problems in Mechanistic Interpretability): CONFIRMED, bib needs update
- arXiv 2501.16496 exists: submitted 27 Jan 2025, only v1, 29 authors. The checker's author order matches arXiv.
- **Venue.** The OpenReview API (note 91H76m9Z94) shows it was "Accepted by TMLR" (venueid TMLR, Survey Certification, 2025). Cite the TMLR version rather than "arXiv preprint". The TMLR bibtex is in verify_citation.json. Its author list differs slightly: Saunders comes before Michaud, and the names read "Jeffrey Wu", "Joseph Isaac Bloom", "Jessica Mary Rumbelow" and "Thomas McGrath".
- The quote is verified in the arXiv v1 HTML, Sec. 2.1.2 ("Reverse engineering step 1: Neural network decomposition"). It says that conceptual work may "provide bounds on the number of features we should expect models to be capable of learning (Hänni et al., 2024; Adler & Shavit, 2024; Bushnaq & Mendel, 2024)". I could not check the TMLR camera-ready PDF, because openreview.net/pdf returned an HTML wall.
- Caveat: this is about capacity bounds (superposition). It is not about how many features a regularised network selects. It supports "open question" only in a loose sense.

## olah2020zoom (Zoom In): CONFIRMED (optional)
- Distill, published March 10, 2020, DOI 10.23915/distill.00024.001.
- The bib fields match the official Distill BibTeX; only the field order differs.
- Claim 1, verbatim: "Features are the fundamental unit of neural networks. They correspond to directions." This supports only that features are central. It does not raise the counting question.

## Assessment
- The substance is supported: counting features is posed as an important open question (Elhage 2022, Open Questions; Sharkey 2025, Sec. 2.1.2).
- "Long-desired result" overstates it. The question dates to about 2022, and mech interp's "number of features" means superposition capacity or SAE dictionary size, not Fourier frequencies or irreps.
- The checker's rewrite is sound: "How many features a network learns is a central open question in mechanistic interpretability \citep{elhage2022superposition,sharkey2025open}." "Central" is backed by Elhage's "most important" framing. Add the bridging clause in the intro.

## Operational note
WebSearch reported that the session's search budget is exhausted ("200 of 200 WebSearch calls"). Every verifier running after that point has no search and must fall back to direct fetches and APIs. That is likely a contributor to the slow verify stage.
