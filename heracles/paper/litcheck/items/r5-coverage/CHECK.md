# R5-coverage: approach.tex prop:coverage, h(theta) <= Gamma_s(p) ||A||_{1/s}

Verdict: NOVEL_AS_FAR_AS_FOUND. The step itself is trivial; the object it bounds is new.

## What it is
This is a one-line normalisation. Rescale A to y = A/||A||_{1/s}, which is feasible for the programme Gamma_s. Then apply the phase lemma (R4) and use the fact that h is linear in A. The proposition adds nothing mathematically beyond R4 and the definition of Gamma_s. Its novelty is the novelty of the l_{1/s}-budgeted programme, and that is checked under R4 and the programme item.

## Prior art
- Kam, Bessafi, Cadet, arXiv 2609.17926 (fetched and confirmed: 15 Sep 2026, under review at NeurReps 2026). Per the sweep notes (LIT_ml.md §0.2), App. A.3.2 Eq. 15 maximises the same worst-offset coverage min_s sum_k beta_k(1-cos(omega_k s)), but under the convex budget ||beta||_2 = B. Their analogue of prop:coverage, "separation <= best return x budget", is implicit in the same rescaling. With exponent 2 in place of 1/s the budget is convex and the optimum is dense (their Prop 5). This is the closest analogue. Cite it as concurrent work.
- Morwani et al. (ICLR 2024; fetched and confirmed as a Spotlight). Their max-margin analysis goes through Fourier amplitudes with the phase condition (Sec. 4) and uses a certificate at b = nu. They do not state a normalised programme with a general l_{1/s} budget. Our Cor dense recovers their Thm 7 at s = 1.
- Generic "linear objective over a norm ball = dual-norm scaling" homogeneity arguments are textbook. No citation is needed.

## Searches
Relied on the sweep's searches (LIT_ml.md Sec. 6) for the programme and the coverage functional. The item says no external check is needed beyond R4. Confirmed the two relevant pages on arXiv.

## Recommendation
No change to the proposition. When the programme eq:programme is introduced, next to prop:coverage, add one sentence: "A related max-min coverage functional with a convex l_2 amplitude budget appears concurrently in Kam et al. (2026, App. A.3.2), where the optimum is dense; the homogeneity-derived l_{1/s} budget (1/s <= 1) is what makes ours sparse."
