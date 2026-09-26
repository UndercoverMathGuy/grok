# R1-avg-margin (approach.tex, lem:avg)

Our statement: h(theta) <= eta_theta(0) - eta_theta(delta) for every delta != 0, where
eta_theta(delta) = E_x f(theta,x)[y(x)-delta].

## Verdict: KNOWN. It is a trivial specialisation of Morwani et al. (ICLR 2024), Sec 3.2.

Source: arXiv:2311.07568v2 HTML (https://arxiv.org/html/2311.07568v2). Fetched this session and grepped.
Venue: "Accepted as Spotlight at ICLR 2024" (from the arXiv abstract page).

- Sec 3.2 "Multi-Class Classification" defines the class-weighted margin for any weighting tau(x,y) in Delta(wrong labels):
  g'(theta,x,y) = f(theta,x)[y] - sum_{y' != y} tau(x,y)[y'] f(theta,x)[y'],
  and says: "Note that g'(theta,x,y) >= g(theta,x,y) as g' replaces the max by a weighted sum."
  Averaging over any q (Sec 3 uses q* uniform on the data, Sec 3.3 step 1) then gives min margin <= E_q g'.
- App. F.1 (proof of Thm 7) defines, for a single quadratic neuron,
  eta_{u,v,w}(delta) := E_{a,b}[(u(a)+v(b))^2 w(a+b-delta)],
  and maximises eta(0) - E_{delta != 0} eta(delta) (their Eq. 4) with uniform tau(a,b)[c'] = 1/(p-1).

Our lemma is Morwani's inequality with q uniform on D_p and tau(x) the point mass on y(x)-delta, applied to the whole network rather than to one neuron. By linearity, eta_theta = sum_i eta_{omega_i}. Their eta uses a uniform tau, which is the average of ours over delta. Nothing here is new. The paper already credits them in the right way: "at most any averaged class-weighted margin [Morwani, Section 3]" and "network-level analogue of eta_{u,v,w}".

## Checks and suggestions
1. The citation is correct. Sec 3 is right; Sec 3.2 is more precise. eta_{u,v,w} is defined in App. F.1, not in the main text. To be precise, write "\citet[Section~3.2]{morwani2024feature}" and "the quantity eta_{u,v,w}(delta) of \citet[Appendix~F.1]{morwani2024feature}".
2. A difference worth one clause: Morwani's tau is uniform over wrong labels, which gives only E_delta. Ours puts a point mass at each offset, so we get one inequality per delta. This is what lets Lemma phase / Prop coverage take a min over delta, which is where coverage and sparsity come from. The step is trivial, but it is the only real departure. Its novelty is expository, not a result.
3. The bib entry morwani2024feature is correct: title, 5 authors, ICLR 2024 all confirmed.

## Searches
- Fetched arxiv.org/html/2311.07568v2. Extracted the Sec 3.2 text and every occurrence of eta.
- Fetched arxiv.org/abs/2311.07568 for the venue.
- I did not search further. "Worst-case margin <= any convex combination of wrong-class logits" is elementary, and no other citation is needed.
