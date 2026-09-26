# VERIFY (novelty_ml): R3-phase-lemma

Claim (approach.tex, lem:phase): h(theta) <= sum_j Re(beta_j) kappa_j(delta) <= sum_j A_j kappa_j(delta), where kappa_j(delta) = 1 - cos(2 pi j delta / p).

Checker verdict: PARTIALLY_KNOWN. **Upheld.**

## Search limits (these affect how complete this check is)
- WebSearch: the session-wide budget is used up ("200 of 200 WebSearch calls"), so every call failed.
- The arXiv API returned HTTP 429 (rate limited) after the first two queries.
- The Semantic Scholar snippet API also returned 429.
- Searching therefore used the arXiv API (2 queries got through), the OpenAlex search API (3 queries), and direct fetches of arXiv HTML.

## What I fetched and verified myself
- **Morwani et al. 2311.07568v2 (HTML, downloaded and grepped).**
  - Theorem 7 is in Section 4 (Cyclic groups) and contains the phase condition theta_u* + theta_v* = theta_w*. Confirmed.
  - Lemma 5 and the g' >= g class-weighted margin are in Sec 3.2. Confirmed.
  - eta_{u,v,w}(delta) := E_{a,b}[(u(a)+v(b))^2 w(a+b-delta)] is defined in **Appendix F.1** ("Proof that Equation 3 is satisfied"), not in Sec 4.
  - F.1 also gives the per-offset Fourier form tilde-eta(delta) = p^{-3} sum_j u^(j) v^(j) w^(-j) rho^{j delta}. It then averages uniformly over delta != 0, giving Eq. (6): 2/((p-1)p^2) sum_{j!=0} u^ v^ w^.
  - So for a single quadratic neuron, the pre-averaging per-offset expression is already in Morwani F.1. The lemma's new step is to apply it network-wide, at one offset, without averaging, and bound the result by amplitudes. That is elementary.
  - The approach.tex pointer "the quantity eta_{u,v,w}(delta) of Morwani" should cite App. F.1.
- **Nanda et al. 2301.05217 (HTML).** Sec 4.2 approximates the logits by sum_k alpha_k cos(w_k(a+b-c)), fitted by OLS, and this explains 95% of the variance. Confirmed. The paper is descriptive and gives no bound.
- **McCracken et al. 2505.18266 (HTML + Crossref).**
  - Thm 4.7 (Sec 4.3) takes h(k) = sum_{l=1}^m cos(2 pi f_l (k-i-j)/n) with random f_l. With probability >= rho, h(i+j) - h(k) > delta m.
  - This is the equal-amplitude coverage used as a probabilistic lower bound for an idealized model, not an upper bound for arbitrary networks. Confirmed.
  - Title and authors are correct. Crossref DOI 10.52202/085713-1476 gives NeurIPS 38, pp. 49512-49573. The bib entry should be @inproceedings (NeurIPS 2025), not an arXiv @article, and the current refs.bib entry is wrong.
- **Kam, Bessafi, Cadet 2609.17926 (abs + HTML).**
  - Submitted 15 Sep 2026. Comment: "Under review at NeurReps 2026 (NeurIPS workshop, Proceedings Track)".
  - App. A.3 Eq. 15: Delta_S(beta) = min_{s!=0} sum_k beta_k (1 - cos(omega_k s)).
  - Loss bound: L <= log(1 + (p-1) e^{-Delta}).
  - Prop. 5: uniform allocation is optimal under ||beta||_2 = B, with value Bp/(sqrt2 (p-1)).
  - All of this is for an idealized aligned-cosine logit model with no phases. It is not a bound for arbitrary networks. Confirmed.
  - Calling it a "heuristic loss proxy" undersells it: within their model the loss bound is rigorous.

## Adversarial novelty search (ML lens): candidates checked, none state the lemma
- 2606.02993, He, Wang, Zhang, Chen, Yang, "Neural Networks Provably Learn Spectral Representations for Group Composition". Results are about dynamics and convergence to single irreps. It gives no margin bound in amplitudes and no L2 max-margin.
- 2606.08985, Tan, Gai, Zhang, "Beyond Neural Collapse: Task-Intrinsic Geometry Governs Neural Representations in Modular Arithmetic".
  - Prop. 5.5 splits the post-collapse cross-entropy into an angular term beta*(1 - cos(alpha - phi)). This is a phase-mismatch cost inside the loss, not a margin upper bound.
  - Thm 6.4 gives a Theta(K) Schatten/weight-decay advantage of the cyclic code over the ETF, where K is the number of classes.
  - Tangential. It could go in related work but is not needed for this lemma.
- 2609.07755, Wang, Kevrekidis, Belkin: GD with weight decay and grokking, general setting. Not relevant.
- The arXiv API (modular addition AND margin) also returned 2511.23443, 2402.09469, 2606.05863, and 2608.07436. Titles and abstracts do not suggest an amplitude-coverage margin bound. 2511.23443 and 2402.09469 were already checked by the novelty_math verifier.

## Conclusion
- Not found in the ML literature: an upper bound on the min-margin of an **arbitrary** network (with interference and arbitrary phases) by the amplitude-weighted coverage at a **single offset**.
- Every ingredient is known:
  - min <= class-weighted average (Morwani Sec 3.2);
  - the per-offset Fourier expansion of eta for one neuron (Morwani App. F.1);
  - the 1-cos separation for aligned cosine logits (Nanda Sec 4.2, McCracken Thm 4.7, Kam Eq. 15).
- Present the lemma as an adaptation of Morwani App. F.1 before averaging over offsets. PARTIALLY_KNOWN stands.

## Recommended wording
"(the analogue of the phase condition in \citet[Theorem~7]{morwani2024feature}. For a single neuron, the middle expression before averaging over $\delta$ is their $\tilde\eta$ expansion \citep[Appendix~F.1]{morwani2024feature}, and averaging it uniformly over $\delta\neq0$ recovers their uniformly class-weighted margin. For idealized phase-aligned cosine logits the right-hand side is the familiar logit separation \citep[Section~4.2]{nanda2023progress}; \citep[Theorem~4.7]{mccracken2025}. Concurrently, \citet[Appendix~A.3]{kam2026symmetry} bound the cross-entropy of such logits by the worst-offset separation.)"
