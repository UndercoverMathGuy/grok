# R1-avg-margin: independent verification (lens: novelty_math)

Item: approach.tex lem:avg. For every theta and delta != 0, h(theta) <= eta_theta(0) - eta_theta(delta), where eta_theta(delta) := E_x f(theta,x)[y(x)-delta].

Checker's verdict: KNOWN (a trivial specialisation of Morwani et al. Sec 3.2).
My verdict: UPHELD, KNOWN. The checker missed one credit: Li et al. (AISTATS 2025) already define the n-input eta. The bib key is in refs.bib, but no current section cites it.

## What I fetched this session
- https://arxiv.org/abs/2311.07568: title, 5 authors, v1 13 Nov 2023, v2 19 Feb 2024, Comments "Accepted as Spotlight at ICLR 2024".
- https://arxiv.org/html/2311.07568v2: full text, extracted with math alttext and grepped.
- OpenReview API (api2.openreview.net/notes/search): note i9wDX850jR, venue "ICLR 2024 spotlight", authors Depen Morwani, Benjamin L. Edelman, Costin-Andrei Oncescu, Rosie Zhao, Sham M. Kakade. The forum page itself is behind a browser check, and the camera-ready PDF was not retrievable.
- https://arxiv.org/abs/2402.09469 and https://arxiv.org/html/2402.09469v4 (Li, Liang, Shi, Song, Zhou): Comments "AIStats 2025".
- https://arxiv.org/html/2407.12332 (Mohamadi et al., "Why Do You Grok?"), for adversarial coverage.
- https://arxiv.org/html/2511.23443v1 ("Provable Benefits of Sinusoidal Activation for Modular Addition"), for adversarial coverage.
- Boyd & Vandenberghe, Convex Optimization, CUP 2004, PDF at https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf, Sec 5.4.1.
- Y. Zhao, Graph Theory and Additive Combinatorics, CUP 2023: book page https://yufeizhao.com/gtacbook/ and chapter PDF https://yufeizhao.com/gtacbook/6.pdf.
- Search tools used: arXiv API, OpenAlex, OpenReview API. WebSearch was exhausted (session cap), and Semantic Scholar returned 429.

## Checker claims, checked one by one against Morwani arXiv v2
1. Sec 3.2 "Multi-Class Classification" defines the class-weighted margin g'(theta,x,y) = f[y] - sum_{y' != y} tau(x,y)[y'] f[y'] for any tau in Delta(wrong labels). It then says, verbatim: "Note that g'(theta,x,y) >= g(theta,x,y) as g' replaces the max by a weighted sum." CONFIRMED.
2. The Sec 3 preamble (before 3.1) writes gamma* = max_theta min_{q in P(D)} E_q[g], and Lemma 2 uses the max-min inequality (citing Boyd & Vandenberghe). The paper's pointer "Section 3" is therefore accurate for "min margin <= any averaged class-weighted margin", since it combines the Sec 3 preamble with Sec 3.2. "Section 3.2" would be more precise for g' >= g. Either is fine.
3. Sec 3.3 "Blueprint", step 1: "we consider q* to be uniformly distributed on the input data points." CONFIRMED.
4. The uniform tau(a,b)[c'] := 1/(p-1) is stated in the Sec 4 proof outline of Theorem 7 and used again in App. F.1. CONFIRMED. The checker placed it in F.1, which is also true.
5. App. F.1 "Proof that Equation 3 is satisfied" defines eta_{u,v,w}(delta) := E_{a,b}[(u(a)+v(b))^2 w(a+b-delta)] and maximises eta(0) - E_{delta != 0} eta(delta), which is their Eq. (4). CONFIRMED.
6. Linearity of expectation over neurons is stated in Sec 3.2, just before Lemma 5, so eta_theta = sum_i eta_{omega_i} holds by linearity. CONFIRMED.
7. The bib entry morwani2024feature is correct in title, authors and ICLR 2024. It has "Sham" where the paper has "Sham M."; that is harmless.

## Why lem:avg is a specialisation (my own derivation)
Take Morwani's tau(x,y) to be the point mass on y-delta. This is admissible because delta != 0, so y-delta is a wrong label. Then g' = f[y] - f[y-delta] >= g, and taking the mean under uniform q gives eta_theta(0) - eta_theta(delta) >= E g >= min g = h(theta).

Morwani's uniform tau is the average of these point masses over delta. The only new step in the paper is to keep each delta separately instead of averaging. Their framework already allows any tau, so even this per-delta form is an instance of their Sec 3.2. It is not a result. Its value is expository: it sets up the min over delta in lem:phase and prop coverage, and those are separate items (R3/R5).

## Adversarial search (mathematics lens)
- Pure mathematics. The inequality is "minimum <= any convex combination", i.e. the max-min / weak-duality inequality. Boyd & Vandenberghe Sec 5.4.1, eq. (5.46): sup_z inf_w f(w,z) <= inf_w sup_z f(w,z) "for any f ... This general inequality is called the max-min inequality." Morwani already invokes this for Lemma 2. It is textbook and needs no extra credit beyond Morwani.
- Group-averaging / interference. The claim that averaging over inputs at a fixed offset kills every part of the logit table that does not depend on the offset is the network-level form of Morwani's step in App. F.1: "E_{a,b}[u(a)^2 w(a+b-delta)] = ... = 0". There they note that the non-bilinear terms of a quadratic neuron average out. Suggestion: optionally say "as in Morwani App. F.1" next to the "interference averages away" sentence.
- Additive combinatorics. For a product neuron, eta_i(delta) is a shifted "solutions to a linear equation" average, E_{a_1..a_n} u_1(a_1)...u_n(a_n) w(a_1+...+a_n-delta). Its Fourier expansion, used right after the lemma in eq:spectrum, is the textbook identity in Zhao, Graph Theory and Additive Combinatorics (CUP 2023). Chapter 6, Prop. 6.1.9 ("Fourier and 3-AP"): Lambda(f,g,h) = sum_r f^(r) g^(-2r) h^(r). Exercise 6.1.11 is the general k-variable version: E_{a_1x_1+...+a_kx_k=0} f_1(x_1)...f_k(x_k) = sum_r f^_1(a_1 r)...f^_k(a_k r). This concerns R2-spectrum rather than lem:avg. The numbering is from the free manuscript PDF, which the author says has "different pagination" from the published version, so cite it at chapter level.
- ML margin theory, as a check that no one already states the per-delta network bound.
  - Li et al. (AISTATS 2025, arXiv:2402.09469v4) restate the class-weighted margin (Def. 3.6: "g' uses a weighted sum rather than max, so g <= g'"). They define the k-input single-neuron eta in Def. D.5 (App. D.4): eta_{u_1..u_k,w}(delta) := E[(u_1(a_1)+...+u_k(a_k))^k w(a_1+...+a_k-delta)], with uniform tau (Lemma D.7). There is no per-delta bound and no network-level bound.
  - Mohamadi et al. ("Why Do You Grok?", arXiv:2407.12332) use an l_inf-normalised margin and a constructed network. There is no averaged-margin upper bound.
  - arXiv:2511.23443 (sinusoidal activation) has no class-weighted or averaged margin.
- Also noticed, relevant to R2: Morwani App. F.3 ("Proof that all frequencies are used") already uses the multidimensional DFT and its diagonal coefficients f^(j,j,-j). This is the same object as our beta_j := 2 p^{-(n+1)} f^_theta(j,...,j;-j).

## Issues the checker missed
1. Li et al. credit is missing. The paper works with n operands, but the n-input eta (Li et al. Def. D.5) and the k-input class-weighted margin (Li et al. Def. 3.6) are not credited. li2025nsum is in refs.bib and in a stale paper.bbl, but no current section cites it (intro.tex l.22 has only a TODO). Suggested edit at approach.tex l.11: "the network-level analogue of the quantity $\eta_{u,v,w}(\delta)$ of \citet[Appendix~F.1]{morwani2024feature} (for $n$ operands, \citealp[Definition~D.5]{li2025nsum})".
2. Minor. The uniform tau appears first in Morwani's Sec 4 proof outline of Thm 7, not in Sec 3.3; the checker's attribution to App. F.1 is also correct. No action needed.
3. Caveat. The Section/Appendix numbers were read from arXiv v2 (19 Feb 2024). The ICLR camera-ready PDF on OpenReview could not be fetched (browser check), so its numbering was not compared directly.

## Recommended paper changes (lem:avg itself unchanged; do not present it as a contribution)
- approach.tex l.7: "\citep[Section~3]" is acceptable; "\citep[Section~3.2]" is more precise.
- approach.tex l.11: add "[Appendix~F.1]" for Morwani and credit Li et al. Def. D.5 for n operands (see issue 1).
- Optional: after the lemma, add "Morwani et al. weight all wrong labels uniformly, which gives the average of this bound over delta; keeping each delta is what later allows the minimum over offsets."
- Optional, for R2-spectrum: cite Zhao (2023, Ch. 6) for the Fourier identity of linear-equation averages, and Morwani App. F.3 for the diagonal DFT coefficient.
