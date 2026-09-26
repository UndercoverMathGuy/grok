# VERIFY (citation lens) -- C12-morwani-constant (task 86)

Verdict: UPHELD. The checker's SUPPORTED verdict and its one confirmed citation both hold.

## Sources fetched this session
- https://arxiv.org/abs/2311.07568. Title "Feature emergence via margin maximization: case studies in algebraic tasks". Authors: Morwani, Edelman, Oncescu, Zhao, Kakade. The arXiv comment reads "Accepted as Spotlight at ICLR 2024". v1 is dated 2023-11-13 and v2 2024-02-19. The bib entry `morwani2024feature` is correct in title, authors, venue, year and arXiv note.
- https://arxiv.org/html/2311.07568v2, downloaded with curl and grepped directly rather than relying on a summarizer.
  - Theorem 7 is in Section 4 (cyclic groups). Its conditions are a prime p>2, width m >= 4(p-1), and the neuron (u_a+v_b)^2 w. Its statement is verbatim: "The maximum L_{2,3}-margin of such a network on the dataset D_p is gamma* = sqrt(2/27) * 1/(p^{1/2}(p-1))". The same theorem says that each neuron in any maximizer uses a single frequency with theta_u+theta_v=theta_w, and that every zeta in {1..(p-1)/2} is used by at least one neuron.
  - Appendix F.1 defines eta_{u,v,w}(delta) := E_{a,b}[(u(a)+v(b))^2 w(a+b-delta)]. This is the quantity approach.tex:11 refers to.
  - The class-weighted margin is in Section 3.2 and the phase condition is in Section 4 / Theorem 7. Both pinpoints in approach.tex are accurate.
- OpenReview forum page: blocked by a bot-check, so the venue is taken from the arXiv comment.

## Independent check of the constant
- The paper defines B_p = 2(2p nu)^{-nu/2}. At nu=3 this gives 2B_p = 4(6p)^{-3/2}, and 4(6p)^{-3/2} * p/(p-1) = (2/(3 sqrt6))/(p^{1/2}(p-1)) = sqrt(2/27)/(p^{1/2}(p-1)).
- Rederivation from Morwani App. F eq. (8): take one frequency with |u^|=|v^|=|w^|=sqrt(p/6). This gives 4/((p-1)p^2) * (p/6)^{3/2}, which is the same number.
- Numerically at p = 3, 5, 7, 113, 1009 all three expressions agree to 2e-16 relative error.
- The C_3 = 2B_p logic is sound:
  - Upper bound: the power neuron satisfies C_3 <= 2B_p (n! B_p with n=2), and Thm universal gives gamma*_3 <= C_3 * p/(p-1).
  - Lower bound: Morwani attain gamma* = 2B_p p/(p-1) at width 4(p-1), so C_3 >= 2B_p.
  - C_3 is a single-neuron, width-free quantity, so attainment at any one width is enough.
- The definitions of g, h, the L_{2,b} norm and gamma* match Morwani Section 3 exactly. Morwani's norm is on the concatenated (u,v,w), as ours is.

## Recommended edit (optional; the paper is not wrong without it)
- Add a pinpoint and the width condition: "exactly their maximum $L_{2,3}$-margin \citep[Theorem~7]{morwani2024feature} (attained at width $m\ge4(p-1)$)".
- Optionally, at approach.tex:11, write \citep[Appendix~F.1]{morwani2024feature} for eta_{u,v,w}(delta), since it is defined there and not in the main text.

## No better citation
This is the primary and only source for the constant. Gromov (2023) gives the analytic quadratic construction, but not the max-margin value.

## Operational note
WebSearch returned "session has used its web search budget (200 of 200)". Later verifier tasks in this workflow run will have no WebSearch and must rely on WebFetch or curl.
