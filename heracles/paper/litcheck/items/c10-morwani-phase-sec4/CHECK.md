# C10-morwani-phase-sec4: verdict SUPPORTED (a small wording fix is optional)

Sentence (approach.tex:36): "...largest when each beta_j is real and positive, i.e. each frequency contributes a cosine peaked at the label (the analogue of the phase condition of \citet[Section~4]{morwani2024feature})."

Source fetched: https://arxiv.org/html/2311.07568 and https://arxiv.org/abs/2311.07568
- Title: "Feature emergence via margin maximization: case studies in algebraic tasks". Authors: Morwani, Edelman, Oncescu, Zhao, Kakade. The arXiv page says "Accepted as Spotlight at ICLR 2024". This matches refs.bib.
- Section list: 1 Intro, 2 Prelims, 3 Theoretical Approach (3.1 binary, 3.2 multi-class with class-weighted margin, 3.3 blueprint), 4 Cyclic groups (modular addition), 5 Sparse parity, 6 Finite groups, 7 Discussion.
- The phase condition is in Section 4, Theorem 7. Each neuron is u(a)=lambda cos(theta_u* + 2 pi zeta a/p), with v and w written the same way, "for some phase offsets theta_u*, theta_v*, theta_w* in R satisfying theta_u* + theta_v* = theta_w*".
- Why the condition matters: it aligns the phases so that the neuron's output cosine peaks at c = a+b. This comes from the fetch tool's summary and was not checked word for word.

Assessment:
- The section number is correct, and so is the direction of the claim: phase alignment maximises the margin.
- Their condition is stated per neuron, for the phases of the weights (theta_u + theta_v = theta_w). Ours is stated per network frequency: the Fourier coefficient beta_j is real and positive. Calling ours an "analogue" is accurate.
- Checked in passing: approach.tex:7 cites [Section 3] for the class-weighted-margin bound. That is correct, since Section 3.2 introduces the class-weighted margin.

Optional fix: name the theorem, i.e. "(the analogue of the phase condition $\theta_u^*+\theta_v^*=\theta_w^*$ of \citet[Theorem~7]{morwani2024feature})". This form is more precise.
