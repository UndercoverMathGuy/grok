# F1-long-desired: abstract sentence 1

Claim: "Explaining the number of features in a network has been a long-desired result in mechanistic interpretability". No citation. refs.bib has no Olah, Elhage or Sharkey entry (checked with grep).

Verdict: PARTIALLY_SUPPORTED. The literature does pose the question, and there are citations for it. "Long-desired result" overstates things: the question has been stated explicitly as open only since 2022. Also, in mech interp "features" means superposition/SAE features, not Fourier frequencies. Soften the wording and add citations.

## Evidence (all fetched 2026-09-25)
- Elhage et al. 2022, Toy Models of Superposition, Transformer Circuits Thread, https://transformer-circuits.pub/2022/toy_model/index.html
  - The "Open Questions" section lists: "How many features should we expect to be stored in superposition? ... It seems like results from compressed sensing should be able to give us useful upper-bounds, but it would be nice to have a clearer understanding – and perhaps tighter bounds!"
  - In the main text: "one question is just how many features the model learns to represent". They measure it with ||W||_F^2.
  - Replication / SAE section: "we ultimately want to count the features used by real neural networks, where the number of features is unknown."
- Sharkey et al. 2025, Open Problems in Mechanistic Interpretability, arXiv 2501.16496 (Jan 2025). Sec. 2.1 (reverse engineering / decomposition): further conceptual work "may ... provide bounds on the number of features we should expect models to be capable of learning (Hänni et al., 2024; Adler & Shavit, 2024; Bushnaq & Mendel, 2024)". This supports "open problem" directly.
- Olah et al. 2020, Zoom In, Distill, doi 10.23915/distill.00024.001. Claim 1: "Features are the fundamental unit of neural networks." This supports only "features are central". It says nothing about counting them.

## Caveat
These sources concern capacity: how many features fit in superposition. The paper's "features" are Fourier frequencies/irreps chosen by the margin-regularisation trade-off. Bridge the two explicitly rather than implying that mech interp wanted exactly this.

## Suggested rewrite
"How many features a network learns is a central open question in mechanistic interpretability \citep{elhage2022superposition,sharkey2025open}."
Optionally cite olah2020zoom for the notion of feature.
