# R1-avg-margin: independent verification (lens: novelty_ml)

Item: approach.tex `lem:avg`. For every theta and delta != 0,
h(theta) <= eta_theta(0) - eta_theta(delta), where eta_theta(delta) := E_x f(theta,x)[y(x)-delta].

## Verdict: KNOWN. The checker is upheld, with two corrections to its evidence and wording.

The lemma is Morwani et al.'s class-weighted-margin inequality, specialised as follows:
- q is uniform on D_p.
- tau(x,y) is the point mass on y - delta.

Morwani's own framework already allows weights that depend on the offset, so a point mass at delta is not a departure from it (see B below).

## What I fetched this session

All pages below were fetched with curl and converted to text. I grepped them myself.

1. **Morwani, Edelman, Oncescu, Zhao, Kakade**, "Feature emergence via margin maximization: case studies in algebraic tasks".
   - Abstract page: https://arxiv.org/abs/2311.07568. Metadata gives the title and the 5 authors. The comment reads "Spotlight at ICLR 2024".
   - HTML: https://arxiv.org/html/2311.07568v2
   - OpenReview API: note i9wDX850jR has venue "ICLR 2024 spotlight" and the same title and authors ("Sham M. Kakade").
   - Bib entry `morwani2024feature`: correct.

2. **Li, Liang, Shi, Song, Zhou**, "Fourier Circuits in Neural Networks and Transformers: A Case Study of Modular Arithmetic with Multiple Inputs".
   - https://arxiv.org/abs/2402.09469. The arXiv comment says "AIStats 2025".
   - HTML at https://arxiv.org/html/2402.09469.
   - Already in refs.bib as `li2025nsum`, and that entry is correct.

3. Adversarial candidates. All were fetched as arXiv HTML and grepped for margin, weighting and offset constructs:

   | arXiv | Paper | What it does with margins | Per-offset averaged margin? |
   |---|---|---|---|
   | 2407.12332 | Mohamadi et al., "Why Do You Grok?" | l_inf margin plus a generalization bound | No |
   | 2602.16849 | He et al., "On the Mechanism and Dynamics of Modular Addition" | Remark 5.1: uniform "average margin" l_AM only | No |
   | 2511.23443 | "Provable Benefits of Sinusoidal Activation for Modular Addition" | Margin generalization bounds | No |
   | 2410.07476 | "Towards a Unified and Verified Understanding of Group-Operation Networks" | Per-input margin lower bounds for verification | No; these are not upper bounds |
   | 2608.04451 | "A Counterexample to Fourier Alignment in Single-Neuron Modular Addition" | None | No |
   | 2602.03655 | "Sequential Group Composition" | None | No |
   | 2410.01779 | Tian, "Composing Global Solutions ..." | MSE loss, no margin | No |
   | 2509.21519 | "Provable Scaling Laws of Feature Emergence ..." | No margin analysis | No |
   | 2604.00316 | "Breaking Data Symmetry is Needed ..." | No margin analysis | No |
   | 2506.06489 | Kunin et al., "Alternating Gradient Flows" | App. F links utility to Morwani Thm 7 | No margin bound |

   I also scanned the Semantic Scholar list of papers citing 2311.07568 (40 returned) for later ML work. No other ML paper states the per-offset network-level inequality as a result. That is unsurprising, since it is a one-line instance of g' >= g.

## Evidence, quoted from arXiv v2 HTML

### A. Sec. 3.2 "Multi-Class Classification" (the general inequality)

Morwani defines "some tau: D -> Delta(Y) that assigns a weighting of incorrect labels to every datapoint", and then:

> g'(theta,x,y) = f(theta,x)[y] - sum_{y' != y} tau(x,y)[y'] f(theta,x)[y'].
> "Note that g'(theta,x,y) >= g(theta,x,y) as g' replaces the max by a weighted sum. Moreover, by linearity of expectation ..."

The expectation is then written as a sum over neurons, so the statement is already at network level.

tau(x,y) may be any distribution on the wrong labels. The point mass e_{y-delta} is allowed, which gives lem:avg directly.

### B. App. I, Eq. (10), with Lemmas 16, 18 and 19 (offset-dependent weights; not mentioned by the checker)

Morwani writes:

> "Let h_{u,v,w}(c) := E_{a,b}[(u(a)+v(b))^2 w(a∘b∘c)]. Recall we seek solutions for the following weighted margin maximization problem
> h_{u,v,w}(e) - sum_{c != e} tau_c h_{u,v,w}(c), where sum_{c != e} tau_c = 1. (10)"

This is exactly an averaged margin whose weights depend only on the offset c between the candidate and the label.

- **Lemma 16** evaluates Eq. (10), with tau constant on conjugacy classes, as
  sum_m [1 - sum_n tau_{C_n}|C_n| chi_{R_m}(C_n)/d_{R_m}] tr(alpha beta gamma^T)/d^2.
- **Lemma 19** chooses a non-uniform tau_{C_n} = sum_R z_R chi_R(C_n). Morwani explains why: "If we naively chose tau_{C_n} to weigh the conjugacy classes uniformly, then the maximizers ... would be only ... the sign representation".

For an abelian group every conjugacy class is a single element, so a point mass at delta is an admissible tau.

Caveat: App. I assumes groups with real representations, so Z_p is not literally covered by that appendix. The device itself (weights over offsets, not only uniform ones) is Morwani's.

### C. App. F.1 (the cyclic case)

> "eta_{u,v,w}(delta) := E_{a,b}[(u(a)+v(b))^2 w(a+b-delta)]"

Eq. (4) maximises eta(0) - E_{delta != 0} eta(delta) with "tau(a,b)[c'] := 1/(p-1)" (Sec. 5 blueprint text).

Our eta_theta is sum_i eta_{u_i,v_i,w_i} for the quadratic network. The sign conventions match: f[c] = sum_i (u_i(a)+v_i(b))^2 w_i(c).

### D. Li et al. (AISTATS 2025), Definition 3.6

Li et al. restate the class-weighted margin for k-input modular addition, "so g(theta,x,y) <= g'(theta,x,y)", with uniform tau(a_1..a_k)[c'] = 1/(p-1) (App. D).

Our lemma is stated for n operands, so this is the n-operand precedent for the same device.

## Corrections to the checker

1. The checker says: "The only difference: we use a point-mass tau at each offset instead of a uniform tau."
   - This understates Morwani. Offset-weighted (non-uniform) tau is already in App. I Eq. (10) and Lemmas 16, 18 and 19.
   - The per-delta choice is therefore not even a methodological departure. The only thing new is choosing the point mass and then taking the min over delta (lem:phase / prop:coverage), and that belongs to later items.

2. The checker's optional clause, "Morwani et al. use a uniform weighting over wrong labels", is inaccurate as a blanket statement.
   - It is true for modular addition (App. F.1). It is false for general groups (App. I, Lemma 19).
   - If the clause is added, scope it: "for modular addition, Morwani et al. use the uniform weighting (App. F.1) ...".

3. The pointer fixes (Section 3.2 and Appendix F.1) are correct. I confirmed both in v2. Section and appendix numbers refer to arXiv v2, which matches the ICLR camera-ready era (Feb 2024).

## Cross-item flags (outside R1 scope, for the orchestrator)

- **R3 phase lemma / coverage.**
  - Morwani App. I Lemma 16/18 has the coefficient [1 - sum_n tau_{C_n}|C_n| chi_R(C_n)/d_R].
  - With a point-mass tau at delta and a 2-dim real rep of Z_p (chi = 2cos(2 pi j delta/p), d = 2), this equals 1 - cos(2 pi j delta/p), which is our coverage cov_j(delta).
  - This is the representation-theoretic precedent for "coverage". It should be credited there, with the caveat that App. I assumes real representations.
- **R8 dense corollary / dual.** Lemma 19's tau equalises the per-representation value across irreps. That is the dual-optimal offset weighting at b = nu, the group analogue of Gamma_1.
- **R2 spectrum.** Morwani App. F.3 already uses the diagonal Fourier coefficients f-hat(j,k,l) at j = k = -l as the per-frequency signature. Our beta_j ∝ f-hat_theta(j,...,j;-j) is the n-operand version.

## Recommended text changes (lem:avg itself needs no change)

- approach.tex l.7: change `\citep[Section~3]{morwani2024feature}` to `\citep[Section~3.2]{morwani2024feature}`.
- approach.tex l.11: change `of \citet{morwani2024feature}` to `of \citet[Appendix~F.1]{morwani2024feature}`.
- Optional: add "; see also \citet[Definition~3.6]{li2025nsum} for $k$ inputs" after the l.7 citation.
- If a clause about uniform weighting is added, use this wording: "For modular addition \citet{morwani2024feature} weight all wrong labels uniformly, which is the average of Lemma~\ref{lem:avg} over $\delta$; offset-dependent weights appear in their group-theoretic analysis (their Appendix~I, Eq.~(10))."
- Do not present lem:avg as a contribution.

## Search limitations (disclosed)

- The WebSearch budget for the session was exhausted (200/200) before this task started.
- The arXiv export API and the Semantic Scholar search API returned HTTP 429 most of the time, and DuckDuckGo served a CAPTCHA.
- The arXiv web search (arxiv.org/search) worked for about 12 queries before it too was rate-limited. Hits: 2402.09469, 2311.07568, 2511.23443, 2606.05863, 2608.07436, 2605.18845, 2609.18078, 2602.08302 and 2311.18817. None of them has a per-offset margin bound.
- Coverage of the 2025-2026 arXiv is therefore partial: the citing-paper list plus keyword hits.
- Given that the inequality is a one-line specialisation of a published statement, the verdict does not depend on the gaps.
