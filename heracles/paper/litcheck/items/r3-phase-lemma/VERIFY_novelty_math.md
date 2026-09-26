# R3-phase-lemma: independent verification (lens: novelty_math)

Item: approach.tex lem:phase. For every theta and every delta != 0,
h(theta) <= sum_j Re(beta_j) kappa_j(delta) <= sum_j A_j kappa_j(delta), where kappa_j(delta) = 1 - cos(2 pi j delta / p).

Checker's verdict: PARTIALLY_KNOWN.
My verdict: PARTIALLY_KNOWN (upheld). The checker's framing of Kam et al. needs sharpening, and the McCracken bib entry should cite the NeurIPS 2025 version.

All pages below were fetched on 2026-09-25. WebSearch was exhausted (200/200) and arXiv, Semantic Scholar and DBLP returned 429 errors, so discovery ran through OpenAlex filtered search plus direct arXiv HTML fetches.

## 1. What the lemma is, mathematically
It has three steps.
- (i) The minimum is at most any weighted average. This is Morwani's class-weighted margin with tau set to the point mass on y - delta. Equivalently, it is LP/minimax weak duality.
- (ii) The identity eta(0) - (eta(delta) + eta(-delta))/2 = sum_j Re(beta_j) (1 - cos(2 pi j delta/p)) for a real function on Z_p. This is the cosine form of the even part. It is the same algebra as the spectral form of a variogram, gamma(h) = C(0) - C(h) = int (1 - cos) dF, and as 1 - Re phi(t) = E[1 - cos tX] for characteristic functions.
- (iii) Re(beta) <= |beta|.

Every step is elementary. For (ii) I tried to find a fetched page that states the spectral variogram formula explicitly:
- Gneiting, Sasvari and Schlather, Adv. Appl. Prob. 33(3):617-630 (2001), doi 10.1239/aap/1005091356. I fetched the Cambridge landing page; the abstract does not show the formula.
- Wikipedia "Variogram" gives only 2 gamma = C(s1,s1) + C(s2,s2) - 2 C(s1,s2), citing Cressie 1993.

So I recommend no bib entry for (ii). The identity needs no citation.

## 2. Verification of the checker's citations

### morwani2024feature (arXiv 2311.07568v2 HTML): CONFIRMED, with location corrections
- Sec 3.2 defines the class-weighted margin g'(theta,x,y) = f[y] - sum tau(x,y)[y'] f[y'] and says: "Note that g'(theta,x,y) >= g(theta,x,y) as g' replaces the max by a weighted sum."
  - tau is an arbitrary distribution over wrong labels, so a point mass is allowed.
  - Lemma 5 (Sec 3.2) is the single-neuron / combining-neurons lemma for optimisers of E[g']. It is not the inequality itself. The checker's "Sec 3.2 / Lemma 5 gives the class-weighted margin" is loosely right.
- Theorem 7 is in Section 4 ("4 Cyclic groups (modular addition)"). Its statement includes the phase condition: "phase offsets theta_u*, theta_v*, theta_w* in R satisfying theta_u* + theta_v* = theta_w*".
- The Sec 4 proof outline says: "We will use the uniform class weighting: tau(a,b)[c'] := 1/(p-1) ... E_{a,b}[psi'] = 2/((p-1)p^2) sum_{j != 0} u^(j) v^(j) w^(-j)".
- NOTE: eta_{u,v,w}(delta) := E_{a,b}[(u(a)+v(b))^2 w(a+b-delta)] is defined in **Appendix F.1** (the proof of Thm 7), not in Section 4.
  - The derivation that forces the phases, "choose them to satisfy theta_u(j) + theta_v(j) = theta_w(j), so that cos(theta_u(j) + theta_v(j) - theta_w(j)) = 1 is maximized", is also in App. F.1 (Eqs. 7-8).
  - approach.tex calls eta "the quantity eta_{u,v,w}(delta) of Morwani". It should point to App. F.1.
- Refinement claim, checked by algebra. For the power neuron at n = 2, beta_j = 2 * 2 p^{-3} u^ v^ conj(w^).
  - The average over delta != 0 of the lemma's middle term is (p/(p-1)) sum_{j in F_p} Re(beta_j) = 2/((p-1)p^2) sum_{j != 0} u^ v^ w^(-j).
  - This is exactly Morwani App. F.1 Eq. (6). So the per-offset bound refines their uniform-weight bound, as the checker says.

### mccracken2025 (arXiv 2505.18266): CONFIRMED; the current refs.bib entry is WRONG
- The abs page gives the title "Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks" and the authors Gavin McCracken, Gabriela Moisescu-Pareja, Vincent Letourneau, Doina Precup and Jonathan Love. v1 is dated 23 May 2025.
- Venue: the OpenAlex record carries the DOI 10.52202/085713-1476. Its Crossref CSL resolves to "Advances in Neural Information Processing Systems 38", pp. 49512-49573, 2025, with the author spelled Letourneau (accented in the metadata).
  - This makes it a NeurIPS 2025 paper, so the bib should be @inproceedings, not an arXiv @article.
- refs.bib currently has the title "Emergence of sparse and dense circuits in modular arithmetic networks" and the author "McCracken, Andrew and others". Both are wrong.
- Sec 4.3, verbatim: "We model the neural network's output at logit k by h(k) = sum_{l=1}^m cos(2 pi f_l (k-i-j)/n), where each frequency f_l is drawn uniformly at random from {1,...,n/2}."
  - Theorem 4.7: if m > (2 log n - 2 log(2-2 rho)) / (log(pi/delta) - 1), then with probability >= rho, h(i+j) - h(k) > delta m for all k != i+j.
  - This is an equal-amplitude, phase-aligned model and a probabilistic lower bound on separation. It is not an upper bound for arbitrary networks. The checker describes it correctly.

### kam2026symmetry (arXiv 2609.17926): CONFIRMED, but the checker UNDERSTATES the overlap
- The abs page gives the title "Symmetry without a manifold: intrinsic dimension on orbits", the authors Chon-Fai Kam, Miloud Bessafi and Frederic Cadet, v1 dated 15 Sep 2026, and the comment "Under review at NeurReps 2026 (NeurIPS workshop, Proceedings Track)".
- App. A.3.2, verbatim: "The logit separation between the correct class and the competitor at offset s is sum_{k in S} beta_k (1 - cos(omega_k s)), so the quantity that controls the loss is the worst case separation Delta_S(beta) = min_{s != 0} sum_{k in S} beta_k (1 - cos(omega_k s)), (15)". It adds L <= log(1 + (p-1) e^{-Delta_S}).
- The checker missed the next part. Kam then pose "the max-min problem Delta*_S(B) = max_{||beta|| = B} Delta_S(beta)", with an **l2 budget on the logit amplitudes**.
  - They prove **Proposition 5**: at full support S = {1..K_max}, uniform allocation is optimal, with value B p / sqrt(2(p-1)).
  - The proof uses sum_{s != 0} (1 - cos(omega_k s)) = p, followed by Cauchy-Schwarz. This is the same averaging step as approach.tex's "sum_{delta != 0} kappa_j(delta) = p" and as cor:dense.
  - They also report that for restricted S the numerical optimum is non-uniform (+25% at K = 4 and +28% at K = 10, p = 47).
- So Kam state (a) the RHS quantity of lem:phase, (b) the max-min budget-allocation programme with an l2 budget, and (c) a dense-optimum proposition.
  - Their logits come from the idealised exact construction, Eq. (8) / Prop. 2 (8 quadratic units per frequency, phase-aligned, no interference).
  - Their budget is on logit coefficients, not on weights. They explicitly decline to claim the budget account ("the comparison gives no support to the budget account and we do not claim it does").
  - They give no bound for arbitrary networks and no exponent 1/s from AM-GM.
- For THIS item, Kam do not state lem:phase. For the programme (eq:programme) and cor:dense items, Kam are a concurrent overlap that must be cited. (Posted 10 days before the deadline.)
- "Heuristic loss proxy" is slightly unfair. For their idealised logits the separation formula is exact, and the loss inequality is rigorous for that model.

### nanda2023progress (arXiv 2301.05217 HTML): CONFIRMED
- Sec 4.2, verbatim: "We approximate the output logits as the sum sum_k alpha_k cos(w_k(a+b-c)) for k in {14,35,41,42,52} and fit the coefficients alpha_k via ordinary least squares. This approximation explains 95% of the variance in the original logits."

## 3. Further search (adversarial): no statement of lem:phase for arbitrary networks found
- **Li, Liang, Shi, Song, Zhou, arXiv 2402.09469 (arXiv comment "AIStats 2025"; already in refs as li2025nsum).**
  - Def. D.1 defines eta_{u1,u2,u3,w}(delta) := E[(u1(a1) + u2(a2) + u3(a3))^3 w(a1+a2+a3-delta)].
  - Lemma D.3 uses uniform class weighting tau = 1/(p-1).
  - This is a per-neuron, uniform-over-delta version. Cite it next to Morwani where eta_theta is introduced.
- **He et al., arXiv 2602.16849, Remark 5.1.** Defines the average margin l_AM = sum_{x,y} { f[(x+y) mod p] - (1/p) sum_j f[j] }. It is uniform, not per-offset.
- **Mohamadi et al., arXiv 2407.12332, Sec 4.2.** Min-margin and l_inf-normalised margin. No Fourier per-offset bound.
- **Provable Benefits of Sinusoidal Activation for Modular Addition, arXiv 2511.23443, App. D.1.** A constructed sine network has margin exactly 1 - cos(2 pi / p). This is a construction, not an upper bound.
- **Gromov 2301.02679 and Chughtai 2302.03025.** Neither has a margin bound (grep for margin and 1-cos: 0 hits).
- **2608.04451 (a counterexample to Fourier alignment at m = 1) and 2609.26679 (a spectral theory of grokking).** Neither has an amplitude-coverage bound.
- **OpenAlex title/abstract search.** Queries: "modular addition" + margin (20 hits) and "modular addition" + Fourier, 2023-2026 (59 hits). Both scanned; nothing else relevant.

## 4. Recommended text (supersedes the checker's)
- At the definition of eta_theta, write: "the network-level analogue of the quantity $\eta_{u,v,w}(\delta)$ of \citet[Appendix~F.1]{morwani2024feature} (see also \citealp[Definition~D.1]{li2025nsum})".
- After lem:phase, write: "(the analogue of the phase condition in \citet[Theorem~7]{morwani2024feature}; averaging the bound uniformly over $\delta\neq0$ gives $\frac{p}{p-1}\sum_j\mathrm{Re}\,\beta_j$, which for their quadratic network is their uniformly class-weighted margin \citep[Appendix~F.1]{morwani2024feature}). For idealized phase-aligned cosine logits the right-hand side is the familiar logit separation \citep[Section~4.2]{nanda2023progress}, \citep[Theorem~4.7]{mccracken2025}. Concurrently, \citet[Appendix~A.3]{kam2026symmetry} bound the cross-entropy of such logits by the worst-offset separation and solve its $\ell_2$-budget allocation at full support."
- Replace the mccracken2025 bib with the NeurIPS 2025 entry. Add kam2026symmetry.
- Cross-item flag: sec:programme and cor:dense must cite Kam Prop. 5 and Eq. (15)-(16) as a concurrent l2-budget version of Gamma and its dense endpoint.
