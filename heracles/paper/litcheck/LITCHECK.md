# Literature check: synthesis

Paper: "Explaining Feature Sparsity Through the Margin–Regularization Trade-off" (ICLR 2027; due 2026-09-26 17:30 IST).

Sources read for this synthesis:
- The paper as it currently stands: `paper.tex`, `sections/*.tex` (approach.tex mtime 09-25 22:43; intro.tex 19:40; prelim.tex 20:02; cyclic.tex/appendix.tex 21:12) and `refs.bib`.
- All 70 items in `record_for_synthesis.json`, together with the full `check.json` and `verify_*.json` in `items/*/`.
- The three sweeps and `extract/INVENTORY.md`.

Line numbers below refer to the current files.

**Many items are stale.** The intro has been rewritten since the checks ran. It no longer contains the informal theorem, the "Where the dense theorem stops" paragraph, the "why circuits arise" sentence or the related-work TODO. The ReLU paragraph (appendix l.259–260) and the "Known" list of the conjecture (l.274–284) are now commented out. Where an item targets text that no longer exists, the table says so.

**Search limits.** The session's WebSearch budget (200/200) was used up early in the verify stage. The arXiv export API and Semantic Scholar returned HTTP 429. As a result, 59 of the 70 items are flagged `novelty_search_limited` (SL). Their novelty verdicts rest on the sweeps, citation-list scans and targeted fetches of primary sources, not on open search. Every "not found" and every novelty statement below for an SL item carries that caveat.

**Checked by this synthesis (fetched 2026-09-26):**
- Crossref `10.52202/085713-1476`. The McCracken et al. paper is NeurIPS 38, pp. 49512–49573, with authors G. McCracken, G. Moisescu-Pareja, V. Létourneau, D. Precup and J. Love.
- arXiv `2311.07568` (Morwani et al.). The abstract says "margin maximization alone can be used to fully specify the features learned by the network". The comment reads "Accepted as Spotlight at ICLR 2024".
- arXiv `2410.01779` (Tian). The comment reads "NeurIPS 2025 camera ready".
- arXiv `2609.15289` (Levine, "Math for AI safety: an invitation for mathematicians", 14 Sep 2026).
- arXiv `2503.06908` (Wang & Ye).
- Crossref `10.1002/j.1538-7305.1968.tb02486.x` (Slepian, BSTJ 47(4):575–602).
- Crossref `10.1090/mcom/3239` (Friedland & Lim, Math. Comp. 87(311):1255–1281).
- By hand, a re-derivation of the weighted-pigeonhole argument that the R26 maths verifier reported. It holds; see §(a) C9. With radii ρ_j=(c/y_j)^{1/2}, a box count of ∏N_j<p boxes, and AM–GM on the budget (∏y_j ≤ K^{−sK}), every feasible y with at most K ≤ 2ln p/s active frequencies has m(y) ≤ 8π²K^{1−s}p^{−2/K}. Adding the tail bound y_(i) ≤ i^{−s} gives Γ_s(p) = O_s((ln p)^{1−s}).

---

## (a) MUST-FIX list (most severe first)

The tiers are ordered by how visible and how damaging each error is to a reviewer:
1. the fabricated bibliography entry and the misattributions in the abstract and intro;
2. results the paper presents that are known or partly known;
3. wrong citation pointers;
4. missing citations.

All four categories requested for this list are covered.

### Tier 1: wrong bibliography entry and misattributions in the abstract and intro

**A1. `refs.bib` `mccracken2025` has the wrong title, authors and venue** (items C6, F3, F7, R25, R27 and more; flagged by more than 15 checkers and verifiers).
- **Current entry:** `title={Emergence of sparse and dense circuits in modular arithmetic networks}`, `author={McCracken, Andrew and others}`, `journal={arXiv preprint arXiv:2505.18266}`. No paper with that title exists.
- **What arXiv:2505.18266 actually is** (checked through arXiv and Crossref): *Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks*, by Gavin McCracken, Gabriela Moisescu-Pareja, Vincent Létourneau, Doina Precup and Jonathan Love. Advances in Neural Information Processing Systems 38 (NeurIPS 2025), pp. 49512–49573, DOI 10.52202/085713-1476. OpenReview id zuHs6RHQwT.
- **Fix:** replace the entry with the one in `refs_additions.bib` (block "REPLACES existing key mccracken2025"). Keep the key so that existing `\citet` calls still resolve.
- **Caveat:** the figure and theorem numbers used below come from arXiv v1. The camera-ready version was not reachable (OpenReview bot check).

**A2. The paper credits Morwani et al. with the "principle of margin maximization"** (C1, C2, C7; verifiers flagged it three times).
- **Where it appears:**
  - paper.tex l.55: "\citet{morwani2024feature} laid the baseline with their principle of margin-maximization,"
  - intro.tex l.3: "\citet{morwani2024feature} uses the pioneering principle of margin-maximization to enable the bounding of the margin of the network."
  - intro.tex l.7: "This paper serves to generalize the \emph{margin-maximization} principle introduced by \citet{morwani2024feature}"
- **The prior statements:**
  - Wei et al. 2019, Thm 4.1 (binary) and Thm C.1 (multi-class cross-entropy): for any norm, the normalized margin of the global minimiser of loss + λ‖Θ‖^r tends to γ* as λ→0. Morwani et al. restate this as their Thm 1.
  - Lyu & Li 2020: limit directions of gradient flow are KKT points of the L2 margin problem.
  - Morwani et al.'s own claim (abstract, fetched): "margin maximization alone can be used to fully specify the features learned by the network."
- **Fix:**
  - paper.tex: "\citet{morwani2024feature} showed that margin maximization alone determines the features learned by quadratic networks on algebraic tasks, …"
  - intro.tex l.3: "\citet{morwani2024feature} use margin maximization \citep{wei2019regularization,lyu2020gradient} to characterize the features of maximum-margin networks exactly. …"
  - intro.tex l.7: "…generalizes the maximum-margin analysis of \citet{morwani2024feature} …"

**A3. The abstract overgeneralises Morwani's theorem** (C1).
- **paper.tex l.55:** "showing provably that in networks where the regularization ($L_{2,\nu}$) matches its homogeneity degree ($\nu$), all features would be used for modular addition or group composition tasks."
- **What Morwani et al. actually prove:**
  - Thm 7: quadratic neurons (u_a+v_b)²w (ν=3), L_{2,3}, width m≥4(p−1). "The maximum L_{2,3}-margin … is γ* = √(2/27)·1/(p^{1/2}(p−1))". Every neuron is single-frequency, and every frequency is used.
  - Thm 9 (groups): quadratic neurons, L_{2,3}, real irreducible representations, m≥2Σd³, and the character condition Σ_{n≥2} d^{1.5}χ_{R_n}(C)<0. Then γ* = 2/(3√(3|G|)Σd^{2.5}), and every non-trivial irrep is used.
  - Thm 8 (parity) is not an all-features result.
  - General ν is covered only for k-operand power neurons at L_{2,k+1}, by Li et al. 2025, Thm 4.1.
- **Fix:** "…proving that quadratic networks at the matching $L_{2,3}$ norm use every Fourier frequency on modular addition \citep[Thm.~7]{morwani2024feature} and, under a character condition, every non-trivial irreducible representation on group composition (their Thm.~9); \citet[Thm.~4.1]{li2025nsum} extend the modular-addition result to $k$-operand power networks at $L_{2,k+1}$."

**A4. "consistently" contradicts published observations** (F2).
- **paper.tex l.55:** "Empirically, however, networks trained with ordinary weight decay consistently show a sparse selection of features."
- **The contradicting evidence:**
  - McCracken et al. (Sec 4.4, Fig 4; Table 5, L2 = 1e-5): one-hot one-hidden-layer ReLU MLPs learn ⌊n/2⌋ frequencies.
  - Morwani et al. (App C, Fig 7a): for ReLU networks trained with L2, "the final distribution across frequencies is close to uniform".
  - Chughtai et al. (Table 5): C113 MLPs use 10–15 key frequencies.
  - Gromov 2023 is dense too, but it uses no regulariser, so do not cite it for the weight-decay case.
- **Fix:** "Empirically, however, trained networks often use only a handful of features \citep{nanda2023progress,chughtai2023toy,zhong2023clock}, while shallow one-hot networks can use all of them \citep{morwani2024feature,mccracken2025}."

**A5. The McCracken transition is misattributed, and the paper's own scope excludes those networks** (F3, C6).
- **Where it appears:**
  - paper.tex l.56: "Further, our theory explains the transition in the number of features from dense to sparse---as seen empirically in \citet{mccracken2025}---by showing that this transition arises on changing either the homogeneity degree or regularization of the network."
  - intro.tex l.5: "\citet{mccracken2025} similarly sees that under different architectures, the number of features transitions from dense to sparse."
- **What McCracken et al. actually report:**
  - The trigger is "adding either depth, or a trainable embedding matrix for inputs" (Sec 4.4 / Fig 4).
  - L2 is 1e-5 in every Table 5 run. App. G.9 (Fig 41) shows that the average count "can be shifted based on hyperparameters". The C6 verifier reads Fig 41 as a learning-rate × weight-decay sweep; the F3 verifier lists width and batch size. Either way, they report no regularization-driven dense-to-sparse transition, and the norm exponent is always plain L2.
  - Sec 5.1: "we don't know why it occurs".
  - They never vary degree or homogeneity, and their construction uses biases (Sec 4.2).
  - A trainable embedding is a shared linear layer in front of the neurons, and the paper's own appendix l.262 ("Where the approach does not apply") places such networks outside the paper. Added depth also breaks the "sum of independent neurons" form the theorems need.
- **Fix, abstract:** "Our theory predicts a dense-to-sparse transition as the homogeneity degree rises above the norm exponent, or the exponent falls below the degree. A dense-to-sparse transition has been observed when depth or a trainable embedding is added to ReLU networks \citep{mccracken2025}; those architectures lie outside our model class, so this is a candidate explanation, not a test."
- **Fix, intro l.5:** "…transitions from dense to sparse when depth or a trainable embedding is added \citep{mccracken2025}."
- **Do not** credit McCracken with a regularisation-driven transition.

**A6. An empirical claim with no experiments behind it** (F4).
- **paper.tex l.57:** "Specifically, we empirically show that the normalized margin attained by these networks peaks at the predicted optimum for the network, matching the result of \citet{morwani2024feature} for their specific case."
- `experiments.tex` contains only TODO headings.
- Morwani et al. Figs 1(c) and 6(a) show the margin approaching the predicted maximum over training time. They do not show a peak over feature count. Only the b=ν value matches, and it matches in closed form (Cor dense).
- **Fix:** delete the sentence until experiments exist. Or replace it with: "At $b=\nu$ the predicted value is exactly the maximum $L_{2,3}$ margin of \citet[Thm.~7]{morwani2024feature}."

**A7. "Consistent with Chughtai" and "as observed" are too broad and have no referent** (C5, F8, F4).
- **Where it appears:**
  - paper.tex l.57: "The optimums are also consistent with the empirical results seen in \citet{chughtai2023toy}."
  - groups.tex l.4: "…one representation on $S_5$ and the Galois-conjugate pair on $A_5$, as observed."
- **What Chughtai et al. report:**
  - Table 6 (transformers): on S5, sign plus one faithful 4-dimensional irrep in 4/4 seeds. On A5, the 3-dimensional pair alone in 2/4 seeds. A third seed has the pair plus 5d-a but did not generalise (test loss 0.195); the fourth has {standard} only.
  - Table 5 and Fig 7 (ReLU MLPs): S5 uses 2–4 key irreps, most commonly sign plus standard. Every A5 seed includes the 4-dimensional standard irrep.
  - Chughtai et al. never write "Galois conjugate", and they never test S4.
  - Stander et al. 2024 read the same S5 concentration as coset circuits.
  - The paper contains no group experiments, so "as observed" points to nothing.
- **Fix, abstract:** "…consistent with the dominant representations (sign plus one faithful representation) that \citet{chughtai2023toy} report on $S_5$."
- **Fix, groups.tex:** "…one faithful representation (plus the sign character) on $S_5$ and the pair of 3-dimensional representations on $A_5$. These are the committees in all four $S_5$ transformer seeds and two of the four $A_5$ transformer seeds of \citet[Table~6]{chughtai2023toy}; their ReLU MLPs use more representations \citep[Fig.~7]{chughtai2023toy}, and \citet{stander2024grokking} interpret the $S_5$ concentration as coset circuits."
- **Do not** claim that the cyclic K* agrees with Chughtai. Their C113 MLPs use 10–15 frequencies, against our K*=7 at ν=3.

**A8. "Long-desired result" is unsupported** (F1).
- **paper.tex l.55:** "Explaining the number of features in a network has been a long-desired result in mechanistic interpretability."
- **Fix:** "How many features a network learns is an open question in mechanistic interpretability \citep{elhage2022superposition,sharkey2025open}."
- **Support:**
  - Elhage et al. 2022, Open Questions: "How many features should we expect to be stored in superposition? … perhaps tighter bounds!"
  - Sharkey et al. (TMLR 2025), Sec 2.1.2: work that "may … provide bounds on the number of features we should expect models to be capable of learning".
  - Do not attribute to Elhage et al. the sentence "we ultimately want to count the features…". It comes from a replication comment by Sharkey, Braun and Millidge.

### Tier 2: our results that are KNOWN or PARTIALLY_KNOWN (the exact prior statements)

**C1. Cor `cor:dense` (approach.tex l.89–91; appendix l.24–33) is KNOWN.**
- **Morwani et al., Thm 7** (as in A3). App F.3 (via Lemma 6) writes any max-margin f as F(a,b)+λ·1[a+b=c], so f̂(j,j,−j)=λp² for every j≠0. That is equal diagonal amplitude on every frequency, which is our flat maximiser.
- **Li et al. 2025, Thm 4.1:** power neurons (Σ_t u_t(a_t))^k w at L_{2,k+1} with m≥2^{2k−2}(p−1) give γ* = 2(k!)/((2k+2)^{(k+1)/2}(p−1)p^{(k−1)/2}), and every frequency is used. Their Lemma F.2 is the k-operand version of Morwani's F.3. Three verifiers checked numerically that γ* = n!·B_p·p/(p−1), our b=ν power-network value.
- **The maths is also classical:**
  - Γ₁(p)=p/(p−1) with the flat, unique maximiser is the regular-simplex bound for cyclic spherical codes (Rankin 1955; Slepian 1968, Thm 2).
  - Fiedler 1973 (§3.5–3.6) gives the same inequality in graph-spectral form.
  - Concurrently, Kam et al. 2026 prove Δ_S(β) ≤ (p/(p−1))Σβ_k in their Eq. (18).
- **Fix:** replace "exactly their maximum $L_{2,3}$-margin." with:
  > "exactly their maximum $L_{2,3}$-margin \citep[Thm.~7]{morwani2024feature}. For the $n$-operand power network, $n!B_p\,p/(p-1)$ is the maximum $L_{2,n+1}$-margin of \citet[Thm.~4.1]{li2025nsum}, and equal amplitude on every frequency is already implicit in \citet[App.~F.3]{morwani2024feature}. The value $\Gamma_1(p)=p/(p-1)$ and its flat maximizer are the regular-simplex bound for cyclic codes \citep{rankin1955closest,slepian1968group}. What is new here is that this dense endpoint is the $s=1$ member of a programme that is sparse for every $s>1$."
- **Do not** describe the "uniqueness" or the "equal amplitudes" as new (T12; the maths verifier rates these KNOWN).

**C2. The single-neuron budget C_b and B_p (approach.tex l.76–80; appendix l.247–257) is KNOWN.** The maths verifier says KNOWN; the ML verifier says PARTIALLY_KNOWN. Treat it as known.
- **b=2:** Gunasekar et al. 2018, Lemma 7 (Lemma 6 for the diagonal case): min‖w‖₂² = L‖β̂‖_{2/L}^{2/L}. The proof is AM–GM at each Fourier coordinate, with equality for balanced factors on any support. That includes "a neuron may carry several frequencies at no extra cost". Hoff 2017, Lemma 2, gives the same identity in coordinates.
- **b=ν:** Morwani et al., App F.1, eqs (6)–(9) for n=2 (Plancherel, then the inequality of quadratic and geometric means, then Cauchy–Schwarz). Li et al., Lemmas D.7–D.8, for general k; D.7 is "only the multilinear monomial survives, with coefficient k!".
- **2<b<ν:** C_b = C₂^{b/2} follows in one line from monotonicity of ℓ_q.
- **Fix:** after "(Appendix~\ref{app:models})" add: "At $b=\nu$ this is the single-neuron computation of \citet[App.~F.1]{morwani2024feature} ($n=2$) and \citet[Lemmas~D.7--D.8]{li2025nsum}; at $b=2$ it is the Fourier-domain representation cost of \citet[Lemma~7]{gunasekar2018implicit}."

**C3. The AM–GM exponent A^{b/ν} (approach.tex l.50–56) is KNOWN** (T2, U1).
- **The prior statements:**
  - Tibshirani 2021, eqs (8)–(9): min_{∏a_l=c} Σa_l² = k|c|^{2/k}, which is AM–GM, tight at equal factors.
  - Dai, Karzand & Srebro 2021, Table 1: depth-d diagonal networks induce ℓ_{2/d}; convolutional networks induce ℓ_{2/d,2} on the Fourier domain.
  - Kolb et al. 2025, Thm 1: L2 on D factors has the same minima as an L_{2/D} quasi-norm.
  - Hoff 2017.
  - Gunasekar et al. 2018, Lemmas 6–7.
  - Grandvalet 1998 (the two-factor origin).
- **Fix:** after l.54 add: "This is the familiar fact that $\ell_2$ weight decay on a product of $k$ factors acts as an $\ell_{2/k}$ quasi-norm on the product \citep{hoff2017lasso,tibshirani2021equivalences,dai2021representation,kolb2025deep}, found in the Fourier domain for linear convolutional networks by \citet{gunasekar2018implicit}; here the exponent is $q=b/\nu\le1$, and the new ingredient is its interaction with the coverage programme \eqref{eq:programme}."
- **Also fix the wording errors the T2 verifier found:**
  - l.52 "(sum of the weights)" should be "the sum of squared factor norms, raised to $b/2$".
  - l.52 "exactly equal" should be "of equal norm".
  - l.54 "$A^{b/v}$" should be "a constant multiple of $A^{b/\nu}$".

**C4. `eq:spectrum` and app:spectrum (approach.tex l.27–31; appendix l.7–22) are KNOWN** (R2).
- Morwani et al., App F.1, has the single-neuron identity: in the unnumbered display before Eq. (5), η̃(δ)=p^{−3}Σ_j û(j)v̂(j)ŵ(−j)ρ^{jδ}, "only terms where j+l=k+l=0 survive".
- Morwani et al., App F.3, already reads the network-level diagonal f̂(j,j,−j).
- Li et al. do the k-operand version inside the proof of Lemma D.7.
- The general fact is Poisson summation on finite abelian groups (O'Donnell 2014, Sec 3.3).
- **Fix:** after the equation add: "This is the network-level form of the single-neuron identity of \citet[App.~F.1]{morwani2024feature} and \citet[proof of Lemma~D.7]{li2025nsum}; \citet[App.~F.3]{morwani2024feature} already use the diagonal modes of a max-margin network."

**C5. `lem:avg` (approach.tex l.16–21) is KNOWN** (R1).
- Morwani et al., Sec 3.2: "Note that g′(θ,x,y) ≥ g(θ,x,y) as g′ replaces the max by a weighted sum". Li et al., Def 3.6, is the k-input version.
- **Fix:** title the lemma "(averaged margin; a case of \citealp[Sec.~3.2]{morwani2024feature})". Change the l.11 pointer "Section~3" to "Section~3.2" (optional).

**C6. Lemma `lem:cost` (cyclic.tex l.26–34; appendix l.37–77) is PARTIALLY_KNOWN.** The maths verifier rates the inequality KNOWN_TECHNIQUE; the ML verifier upheld PARTIALLY_KNOWN.
- **The inequality is classical once H is built.** Thompson 1976, eq. (1) (due to Rotfel'd): for f increasing and concave with f(0)=0, Σf(s_i(A+B)) ≤ Σf(s_i(A))+Σf(s_i(B)). Apply it to H=Σ_l d_l u_l u_l*, whose rank-one summands have singular value |d_l|. Then use |det H|=|β|^ν (chain lemma) and AM–GM on the singular values. The result is Σ|β_l|^r ≥ ν^{1−r}|β|^r. Weyl's majorant theorem (1949) gives the same bound.
- **Both steps of the proof have precedents.** Seiler & Simon 1975 is Step 1, the coefficient domination. Weyl 1949 is Step 2, the integral of the log generating function.
- **The count ≥ν** follows from rank H = ν. The ML verifier also gives a two-line substitution argument.
- **ν=3 in ML (Tian 2025, NeurIPS):**
  - App. Lemma 5: order-1 and order-2 solutions that satisfy R_c have r_kkk=0, so 3 nodes are needed.
  - App. Lemma 6 and Eqs 91–96: the minimum-norm order-3 solution has equal magnitudes and phases 0, 2π/3, 4π/3.
  - This is under MSE, with no ℓ_r bound.
- **Mohlenkamp & Monzón 2005** leave minimality open: "We believe that our Theorems 2 and 5 are an example of such minimal representations". For tied (symmetric) decompositions, however, minimality of ν terms is already a theorem: Comon & Ottaviani 2012, Prop 3.1, "a real binary form of degree d with d real distinct roots has real rank d" (R21 maths verifier). The ν=3 non-symmetric real rank is in Friedland & Lim, Sec 6.
- **Fix:** after the proof idea add: "Given Lemma~\ref{lem:chain}, the inequality also follows from the concave trace inequality of Rotfel'd \citep{thompson1976convex} with AM--GM on singular values; our route additionally characterizes equality. For $\nu=3$ the equality configuration is the three-node cube-root-of-unity solution of \citet{tian2025composing}."
- **Do not** write that the lemma "proves the lower bound that Mohlenkamp–Monzón conjectured" (overclaim; R12).

**C7. The ν-block construction behind Thm `thm:cyclic` (cyclic.tex l.54–58) is PARTIALLY_KNOWN** (R20, R21, T4).
- **The identity:** Mohlenkamp & Monzón 2005, Thm 2 (sine of a sum of n variables written as n separable terms; Fig 1 uses α=0, π/3, 2π/3 with equidistributed amplitude). The block is its equally spaced cosine case.
- **ML constructions:**
  - Marchetti et al. 2026 (ICML), Lemma B.8 Case 1 and eq. (71): k+1 copies of the construction with phases e^{iπj/(k+1)} (the same half-angle phases as our α_l=πl/ν), (k+1)·n_ρ^{k+1} σπσ neurons in all, cancel every cross term; for cyclic groups this is our ν-block. It is only a sufficient construction. Their Thm 4.3 says the loss lower bound is achievable once N≥(k+1)2^k n_ρ^{k+1} for power neurons (a sufficient width, with the Waring factor 2^k; not a lower bound on N).
  - Tian 2025: z_syn at ν=3.
  - He et al. 2026, Sec 4: at n=2 the three-phase configuration {0, π/3, 2π/3} satisfies their full-diversification condition and zeroes every β_{j,T} (numerical check by the R9 ML verifier), so it is a ν=3 block.
  - Friedland & Lim 2018, Lemma 6.1: the ν=3, r=1 (nuclear-norm) optimality.
- **Fix:** after l.58 add: "The block identity is the equally spaced case of the separated expansion of \citet[Thm.~2]{mohlenkamp2005trig}; the same cancellation appears as a sufficient construction in \citet[Lemma~B.8]{marchetti2026sequential} and, for $\nu=3$, as the three-node solution of \citet{tian2025composing} and a full-diversification configuration of \citet[Sec.~4]{he2026mechanism}. New here are the weights $\lambda_j^\nu\propto y^*_j$, attainment of $\Gamma_s$, and optimality (Lemma~\ref{lem:cost})."
- **Also:** "The phases are a rotated copy of the $\nu$\textsuperscript{th} roots of unity" is imprecise. The e^{iα_l}=e^{iπl/ν} are 2ν-th roots of unity; the doubled phases e^{2iα_l} are the ν-th roots of unity (T4 maths verifier).

**C8. `def:if` and `lem:sign` (cyclic.tex l.6–19) are PARTIALLY_KNOWN; the n=2 case is known.**
- **The n=2 cancellation system is stated in three places:**
  - Kunin et al. 2025 (NeurIPS), Thm F.7 and Eq. (85): equality iff the phase sums of the sign-flipped copies vanish. This is (JF) at n=2.
  - Tian 2025, Lemma 1: the zero set R_c contains the sign-flipped copies. It is a stronger condition than def:if.
  - He et al. 2026, Sec 4: full diversification. Def 4.1(iii) is typeset with a typo, so cite "Sec. 4".
- **Exact constructions:** Morwani et al., App F.2 (8 neurons per frequency; "f takes the same value on all incorrect labels c′"), and Li et al., App E.2 (2^{2k−1} neurons per frequency).
- **Approximate cancellation:** Gromov 2023, Eqs (11)–(15), "destructive interference condition", random phases.
- **The sign rule** is Hermitian symmetry. Gromov's Eq. (11) already shows the per-operand phase factors at n=2. Doshi et al. 2024, App A.2, covers the S-operand power case.
- **Fix:** see insertion plan rows 17–19. **Do not** cite Zhong et al.'s Pizza as an example of excluded interference. Its ideal logit has β_{j,T}=0 for every non-empty T (numerical check by the R9 ML verifier).

**C9. `thm:scaling`, `cor:few` and `eq:posfourier` (cyclic.tex l.107–134; appendix l.153–231) are PARTIALLY_KNOWN, and the maths of `cor:few` is classical.**
- **The equal-share Θ(log p) law is classical.**
  - Alon & Roichman 1994, Sec 3, Thm 2 (with the Remark on Z_n): O(log n) random generators suffice.
  - Alon & Roichman, Prop 3: "If A is an Abelian group of order n, and X(A,S) is a δ-expander, then |S| ≥ c log₂ n."
  - Friedman, Murty & Tillich 2006: "a d-regular Cayley graph on an Abelian group of order n has second eigenvalue bounded below by d − O(dn^{−4/d})". The abstract calls the implied constant absolute, but the result is stated for fixed d (Cor 3 writes C_d, and the tightness Thm 7 fixes d). The verifiers disagree on whether Thm 2 with bound (3) can be used uniformly in d. Do not use FMT for K growing with p; for that, use the Dirichlet pigeonhole bound, which is uniform in K.
- **Fixed K, arbitrary weights.**
  - Simultaneous Dirichlet approximation (Schmidt LNM 785; the theorem number "Thm 1B" comes from a secondary source, so cite the book without it) gives m(y) ≤ 2π²(p−1)^{−2/K}. Its exponent is sharper than eq. (Kbound).
  - Wang & Ye 2025, Thm 4.6 and Prop 4.5, give m(y) ≤ [2p sin²(π/2m)/(p−(m−1)^K)]‖y‖₁ for any non-negative weights.
- **Weighted, for all K=o(log p).** Pigeonhole with frequency-dependent radii (Minkowski's linear-forms theorem, Cassels 1959) gives m(y) ≤ 8π²K^{1−s}p^{−2/K} for K ≤ 2 ln p/s. With the tail bound y_(i) ≤ i^{−s} it also gives Γ_s(p)=O_s((ln p)^{1−s}).
  - This was reported by the R26 maths verifier with a numerical check up to p=100003, and re-derived by this synthesis.
  - The R25 ML verifier thought a log log p factor was lost. That holds only for equal radii.
  - The paper's Bessel route has one advantage: its constant stays bounded as s→1, while the pigeonhole tail term grows like 1/(s−1).
- **ML precedent:** McCracken et al., Thm 4.7 and Cor 4.8: "Learning O(log n) distinct frequencies gives a logit margin Ω(log n)". This is unnormalised and covers sufficiency only. Their constant is about 2.4 ln n at δ=1/2, against our 8 ln p.
- **The positive-Fourier inequality:**
  - Cioabă 2006, Lemma 2.1 and eq. (2), is the closed-walk (lift to Z^K) comparison.
  - DLMF 10.35.2 is the Jacobi–Anger expansion.
- **Fix:** add the paragraph in insertion plan row 27. Narrow cor:few as in §(c) N10.
- **Do not** insert the R26 checker's sentence "pigeonhole alone only reaches K=O(log p/log log p) … the positive-Fourier inequality is needed". It is false.

**C10. The programme `eq:programme` and `prop:coverage` (approach.tex l.46–74) are PARTIALLY_KNOWN.** The checker rated `prop:coverage` NOVEL_AS_FAR_AS_FOUND; both verifiers downgraded it.
- **The form is standard:**
  - Γ_s^{−1/s}=min{Σv_j^{1/s}: Cv≥1, v≥0} is ℓ_q (q<1) minimisation over a polyhedron (Ge, Jiang & Ye 2011, which also shows strong NP-hardness).
  - Γ_s is the largest spectral gap of a weighted circulant graph under an ℓ_{1/s} budget. The s=1 end is Boyd 2006 and Sun, Boyd, Xiao & Diaconis 2006 (the rescaling to "minimize cost s.t. λ₂≥1"). The s→∞ end is Ghosh & Boyd 2006.
  - The template is Gunasekar et al., Thm 2a with Lemma 7.
  - Concurrent work: Kam et al. 2026 (arXiv 2609.17926, 15 Sep 2026), App A.3.2, Eq. 15 and Prop 5: the same worst-offset separation under an ℓ₂ budget on amplitudes, with the uniform allocation optimal at full support. The authors "do not claim" the budget account.
- **Fix:** insertion plan rows 11–12.
- **Also fix** the pages of `ghosh2006growing`: 6605–6611, not 6605–6610.

**C11. `prop:shapes` and `prop:bottleneck` (cyclic.tex l.67–77) are PARTIALLY_KNOWN; their ingredients are textbook.**
- **(i) Concave minima lie at vertices:** Hirsch & Hoffman 1961 (Thm 3.1 and Cor 3.2 in the IMM-NYU report version) and Rockafellar 1970, §32.
  - Do not cite Ge, Jiang & Ye for "the vertices are exactly the local minimisers". That direction fails for the inequality-form polyhedron P_p; the maths verifier gives a counterexample with vertex (0.25, 0.75). Cite them only for "local minimisers are vertices" and for NP-hardness.
- **(ii) Laguerre's rule:** Jameson 2006, Cor 3.2: "A generalized polynomial, or a Dirichlet polynomial, of length n, has at most n−1 zeros".
- **The large-s end:** Fung & Mangasarian 2011, Prop 1.
- **Bottleneck:** the LP basic-solution fact. At s=1 it is Shapley & Snow 1950 (extreme optimal strategies correspond to a square equalising system). Rosset, Zhu & Hastie 2004 is the ML analogue.

**C12. The groups section (groups.tex l.4) is PARTIALLY_KNOWN** (R31, U7).
- **Derksen 2016:** Thm 1.9 and Cor 1.10 give ‖⟨p,q,r⟩‖_* = pqr, attained by the schoolbook decomposition; Strassen's costs more. Cor 1.12, not Thm 1.11, gives ‖T_G‖_* = √|G| Σd^{5/2}, which is where Morwani's Thm 9 gets d^{2.5}.
- **Shutman et al. 2025, Prop 5.2:** the rank per real irrep is at most m_d, the tensor rank of d×d matrix multiplication. There is no norm, and training uses GD or AdamW without weight decay.
- **Christandl & Zuiddam 2019, eqs (7)–(8):** the max-cut flattening, with equality for bipartite graphs.
- **Fan et al. 2023, Thm 1:** the tensor Schatten-p quasi-norm.
- **Marchetti et al. 2026, Thm 4.3:** the d^ν-type construction; power neurons need the Waring 2^k factor.
- **Morwani et al., Thm 9:** its width condition m ≥ 2Σd³ means their own L_{2,3} construction already uses about d³ = d^ν neurons per irrep at ν=3 (R31 maths verifier). So the d^ν neuron count has a ν=3 precedent in Morwani as well as in Marchetti.
- **Ye & Lim (arXiv 1601.00292):** rank-optimal is not norm-optimal.
- **Fix:** see insertion plan rows 34–35. The groups claims (S5/A5 counts; D3, S4 and Z12 exceeding the ceiling) have **no proofs in the appendix**; reviewers will ask for them.

### Tier 3: wrong or imprecise citation pointers

- **D1 (C3), approach.tex l.6.** "By \citet[Theorem~4.1]{wei2019regularization}". Thm 4.1 is binary logistic loss; the multi-class cross-entropy version is Thm C.1. Fix: `\citet[Thm.~4.1; Thm.~C.1 for multi-class cross-entropy]{wei2019regularization}`. It holds for any norm and positively homogeneous f, so L_{2,b} with b≥1 is covered.
- **D2 (C9), approach.tex l.32.** "a frequency with $A_j>0$ is a \emph{feature}, as in \citet{nanda2023progress,morwani2024feature}". Neither paper uses this definition. Nanda's "key frequencies" are thresholded, and Morwani's "use" is per-neuron spectral support. Fix: "…is a \emph{feature}; this is the exact counterpart of the `key frequencies' of \citet{nanda2023progress} and of the frequencies neurons `use' in \citet{morwani2024feature}."
- **D3, approach.tex l.11.** `\citep[Section~3]` → `\citep[Section~3.2]` (optional; numbering from arXiv v2).
- **D4 (U4), appendix l.50.** "Sylvester's identity" usually names the bordered-minor identity. Write "Sylvester's determinant theorem $\det(I+AB)=\det(I+BA)$ \citep[Ch.~1]{horn2012matrix}".
- **D5.** If `hornjohnson2012` from the T3 checker is used anywhere: its DOI 10.1017/CBO9780511810817 is the 1985 first edition. Use `horn2012matrix` (DOI 10.1017/CBO9781139020411). Cite by chapter only; theorem numbers are unverified.
- **D6. Sentences the verifiers rejected; do not copy them into the paper:**
  - "for ℓ_p … the vertices are exactly the local minimisers [Ge et al.]" (false for P_p).
  - "2K − O(Kp^{−2/K}) … absolute constant" [FMT]. FMT's constant is absolute only for fixed d, so the sentence cannot be applied with K growing with p.
  - "(one-sided) 1/2-biased multiset \citep{naor1993small}". Small bias is two-sided and defined over Z₂ⁿ; drop it.
  - "Lemma cost proves the lower bound they conjectured".
  - "8 or 2^{2ν−3} vs ν" without naming the model class. Morwani's and Li's counts are for power networks; the fair product-network contrast is 2^{ν−1} from the product-to-sum expansion.
  - "the argument of \citet{neyshabur2015norm}". Write "cf. the unit-duplication argument"; their norm and construction differ.
  - The R31 checker's paraphrase of Chughtai ("leaves open which representation the weight-norm trade-off prefers"). The exact text is: "Representations with more degrees of freedom may also offer better performance for fixed total weight norm, so which the model may prefer, and thus which is least complex, is unclear."
- **D7.** `\citet{alon1994random}` at appendix l.168 should be pinned to "[Sec.~3, Thm.~2]"; the character-wise tail plus union bound is Thm 2, not Thm 1. Pin Hoeffding to "Thm.~2" (optional).

### Tier 4: missing citations reviewers will flag

- **E1 (F10). There is no related-work section.** `li2025nsum`, `gromov2023grokking`, `lyu2020gradient` and `power2022grokking` are in refs.bib but cited nowhere. Add the paragraph in §(e), or cite each entry where it belongs, or delete it.
- **E2 (U3), prelim.tex l.8.** The n-operand task and the power neuron are exactly Li et al.'s setting (their Sec 3.1). Cite `li2025nsum` there, and `doshi2024grokking` for the many-term construction.
- **E3 (U1, T2), approach.tex l.50–56.** The representation-cost citations (C3).
- **E4 (U9), cyclic.tex l.7.** The sign-flipped copies: `gromov2023grokking` Eqs (10)–(11); `morwani2024feature` App F.2–F.3; `li2025nsum` App E.2; `he2026mechanism` Sec 4.
- **E5.** Tian 2025 (NeurIPS; `tian2025composing`) must be cited at def:if, lem:cost and prop:block. Reviewers who know CoGS will raise it.
- **E6.** Kam et al. 2026 is concurrent (posted 15 Sep 2026, 11 days before the deadline) and states the same max-min under an ℓ₂ budget; cite it at the programme. Jacobs & Montúfar 2026: an exponent in the mirror potential separates sparse from dense max-margin solutions in general. It is an optimiser-geometry exponent, not a homogeneity exponent.
- **E7. Scope sentence, approach.tex l.7.** Optionally, "Gradient flow on cross-entropy converges in direction to KKT points of the $L_2$ margin problem, which need not be global maxima \citep{lyu2020gradient,vardi2022margin}; with small weight decay see \citet{lyu2024dichotomy}." Lyu & Li cover only b=2 and give limit points only.
- **E8. Classical tools** (U4, U5, T5–T8): DLMF 10.32.1, 10.35.2 and 10.25.2 (Bessel); DLMF 5.12.1, 5.12.3 and 5.5.3 (the Beta integral in the x^r representation); Horn & Johnson Ch. 1 and Ch. 7; Rockafellar §32 or Hirsch & Hoffman; Jameson Cor 3.2; the Alon–Roichman and Hoeffding pins; Schilling, Song & Vondraček Ch. 8 at the "other penalties" remark (only Thorin–Bernstein functions qualify, not all Bernstein functions).
- **E9 (U8), appendix l.262.** "priced by a nuclear norm" → `\citep{srebro2004mmmf}` (Lemma 1), plus optionally `parkinson2025linear`. The verifier notes the nuclear norm is invariant under orthogonal mixing, so "rewards carrying many frequencies in one neuron" should be "does not penalize carrying many frequencies in one neuron".

### Side findings (not literature, but found by verifiers)

1. **appendix l.239, "Weaker cancellation suffices".** The inequality needs only the interval patterns. The equality set under only those patterns, however, contains non-interference-free configurations. At ν=4 the F₄^{(1)}(a) Hadamard family is orthogonal, but the non-interval pattern T={1,3} gives a nonzero value at a=0.7 (R20 maths verifier). Say that the structure statement of thm:cyclic still needs the full (JF).
2. **cyclic.tex l.134 (dense window).**
   - The analytic envelope uses the share floor 4^{−1/(s−1)}/⌈8 ln p⌉, which appendix l.213 now derives. The text, however, attributes it to "the share floor of Theorem thm:scaling", which states 4^{−1/(s−1)}/(9 ln p) and would give ln4/ln((p−1)/(18 ln p)). Either point to the appendix or state the sharper floor in the theorem.
   - The envelope is vacuous for p<73. A verifier computed that the exact window scales like about 3.7/p, not 1/log p. Say "at most O(1/log p)".
3. **The ReLU constant** (appendix l.260, commented out): if it is restored, it must read C₂ ≤ 1/(2p), not p^{−n}/2 (T13, R32, R6; numerical counterexamples). Cite Wei et al., App B.4 and Prop B.11, for ReLU+L2 being the b=ν=2 case.
4. **Table tab:kstar:** move the LaTeX-comment flag "(113, ν=3)=7 is heuristic" into the caption.
5. **intro.tex l.7:** "arbitrary $L_{q}; 2\le q \le \nu$ norms" should be $L_{2,b}$, $2\le b\le\nu$. Also fix the typo "intereference".

---

## (b) Per-item table

Final verdict = the checker's verdict as corrected by the verifiers; disagreements are noted. SL = `novelty_search_limited`. "Stale" = the text the item targets is gone from the current sources.

| Item | Location (current) | Final verdict | Citations (bib keys) | SL |
|---|---|---|---|---|
| R1 avg margin | approach.tex l.9–21 lem:avg | KNOWN (both verifiers agree): a case of Morwani Sec 3.2 | morwani2024feature [Sec 3.2; App F.1]; li2025nsum [Def 3.6] | yes |
| R2 spectrum | approach.tex l.27–31; appendix l.7–22 | KNOWN | morwani2024feature [App F.1, F.3]; li2025nsum [proof of Lemma D.7]; odonnell2014analysis (opt.) | yes |
| R3 phase lemma | approach.tex l.34–43 lem:phase | PARTIALLY_KNOWN (both upheld) | morwani2024feature [Sec 3.2, Thm 7, App F.1]; nanda2023progress [Sec 4.2]; mccracken2025 [Thm 4.7]; kam2026symmetry [App A.3]; li2025nsum | yes |
| R4 programme | approach.tex l.46–66 | PARTIALLY_KNOWN (the maths verifier's upheld=false concerns only a bib page error) | gunasekar2018implicit; ge2011note; boyd2006convex; ghosh2006growing; kam2026symmetry; jacobs2026mirror | yes |
| R5 coverage | approach.tex l.69–74 | PARTIALLY_KNOWN; the checker said NOVEL_AS_FAR_AS_FOUND and both verifiers downgraded (a rescaling; s=1 case is Kam Eq 18 and Fiedler) | kam2026symmetry; fiedler1973algebraic; sun2006fastest; gunasekar2018implicit | yes |
| R6 single-neuron budget | approach.tex l.76–80; appendix l.247–257 | **Disagreement:** maths says KNOWN, ML says PARTIALLY_KNOWN; treat as KNOWN. ReLU constant wrong (now commented out) | gunasekar2018implicit [Lemma 7]; hoff2017lasso; morwani2024feature [App F.1 eqs 6–9]; li2025nsum [Lemmas D.7–D.8] | no |
| R7 universal bound | approach.tex l.82–87 | PARTIALLY_KNOWN (both upheld): endpoints and pricing known; b<ν bound not found | morwani2024feature [Thm 7]; li2025nsum [Thm 4.1]; gunasekar2018implicit; dai2021representation; hoff2017lasso | yes |
| R8 dense corollary | approach.tex l.89–91; appendix l.24–33 | KNOWN | morwani2024feature [Thm 7, App F.3]; li2025nsum [Thm 4.1, Lemma F.2]; rankin1955closest; slepian1968group; fiedler1973algebraic; kam2026symmetry | yes |
| R9 interference-free def | cyclic.tex l.9–13 | PARTIALLY_KNOWN; the ML verifier (upheld=false) says the definition is KNOWN at n=2; drop the Zhong example | kunin2025alternating [Thm F.7]; tian2025composing [Lemma 1]; he2026mechanism [Sec 4]; morwani2024feature [App F.2]; li2025nsum [App E]; gromov2023grokking [Eqs 11–15] | yes |
| R10 sign rule | cyclic.tex l.15–19; appendix l.247 | PARTIALLY_KNOWN (elementary; credit Gromov first, not He) | gromov2023grokking [Eq 11]; doshi2024grokking [App A.2]; he2026mechanism [Sec 4]; morwani2024feature [App F.2]; li2025nsum [Lemma 4.3] | yes |
| R12 cost of a feature | cyclic.tex l.26–34; appendix l.37–77 | PARTIALLY_KNOWN. **Disagreement:** maths says the inequality is KNOWN_TECHNIQUE (Rotfel'd–Thompson) and the reduction plus equality case are new; ML upheld | thompson1976convex; tian2025composing [App Lemmas 5–6]; mohlenkamp2005trig; comon2012typical; friedland2018nuclear | yes |
| R13 cost sublemmas | appendix l.39–85 | PARTIALLY_KNOWN, with more known than the checker said | thompson1976convex; rajan2015stochastic; schilling2012bernstein [Ch 8]; seiler1975inequality; weyl1949 | yes |
| R20 cyclic theorem | cyclic.tex l.36–52; appendix l.87–101 | PARTIALLY_KNOWN (construction known). The ML verifier's upheld=false is because the checker's change was wrong: thm:cyclic does not recover the Morwani/Li power-net values | marchetti2026sequential [Lemma B.8, Thm 4.3]; tian2025composing; mohlenkamp2005trig; gromov2023grokking | yes |
| R21 ν-block | cyclic.tex l.54–58; appendix l.103–124 | PARTIALLY_KNOWN; the ML verifier (upheld=false) found the ν=3 block in Tian | mohlenkamp2005trig [Thm 2]; tian2025composing; marchetti2026sequential; friedland2018nuclear; morwani2024feature [App F.2]; li2025nsum [Lemma E.3] | yes |
| R22 finitely many shapes | cyclic.tex l.67–71; appendix l.132–145 | PARTIALLY_KNOWN; the maths verifier's upheld=false is over ge2011note being misapplied | hirsch1961extreme; rockafellar1970convex [§32]; jameson2006counting [Cor 3.2]; fung2011equivalence; ge2011note (local-min⇒vertex only) | yes |
| R23 bottleneck | cyclic.tex l.73–77; appendix l.147–149 | PARTIALLY_KNOWN | shapley1950basic; rosset2004boosting (opt.); ge2011note | yes |
| R24 ablation | cyclic.tex l.79–87 | NOVEL_AS_FAR_AS_FOUND as an inequality; its reading has a heuristic precedent (SVM-RFE) | nanda2023progress [Sec 4.4]; chughtai2023toy; guyon2002gene [Sec 2.5] | yes |
| R25 log scaling | cyclic.tex l.107–122; appendix l.153–220 | PARTIALLY_KNOWN; the order also follows from weighted pigeonhole (see C9) | alon1994random [Sec 3, Thm 2, Prop 3]; friedman2006spectral; schmidt1980diophantine; cassels1959introduction; wang2025positive; mccracken2025 [Thm 4.7, Cor 4.8]; cioaba2006closed | yes |
| R26 few features | cyclic.tex l.124–130; appendix l.222–231 | **Disagreement:** maths says KNOWN_TECHNIQUE (weighted pigeonhole; re-derived here), ML says PARTIALLY_KNOWN. Final: maths core classical; the network transfer is ours | alon1994random; friedman2006spectral; cassels1959introduction; schmidt1980diophantine; mccracken2025 | yes |
| R27 dense window | cyclic.tex l.134 | NOVEL_AS_FAR_AS_FOUND (low confidence; envelope loose) | mccracken2025 (context) | yes |
| R29 K* table | cyclic.tex l.71, 89–105 | NOVEL_AS_FAR_AS_FOUND (our computations; heuristic above the exhaustive range) | nanda2023progress; chughtai2023toy; zhong2023clock; mccracken2025 (context only) | yes |
| R31 groups | groups.tex l.4 | PARTIALLY_KNOWN; the ML verifier's upheld=false is over pointer errors (Derksen Cor 1.12; Chughtai quote; Marchetti count) | derksen2016nuclear; shutman2025words; christandl2019surgery; fan2023euclidean; marchetti2026sequential; morwani2024feature [Thm 9]; chughtai2023toy; stander2024grokking; wu2025unified; strassen1969; blaser2013; ye2018structured | yes |
| R32 ReLU endpoint | appendix l.259–260 (**commented out**) | PARTIALLY_KNOWN; the constant is wrong (should be 1/(2p)) | wei2019regularization [App B.4, Prop B.11]; morwani2024feature [App C Fig 7a, Fig 4b] | yes |
| R34 conjecture | appendix l.264–284 (the "Known" list is **commented out**) | NOVEL_AS_FAR_AS_FOUND. The b=ν endpoint: power nets per Morwani/Li; product nets by cor:dense | morwani2024feature [Thm 7]; li2025nsum [Thm 4.1]; tian2025composing (opt.) | yes |
| R37 appendix remarks | appendix l.233–243, 286 | RELATED_ONLY | neyshabur2015norm ("cf."); slepian1968group | yes |
| T1 subadditivity | approach.tex l.86 | KNOWN (elementary p-norm inequality) | none needed | yes |
| T2 AM–GM price | approach.tex l.50–56 | KNOWN | tibshirani2021equivalences; hoff2017lasso; dai2021representation; kolb2025deep; gunasekar2018implicit [Lemmas 6–7]; morwani2024feature [App F.1]; grandvalet1998least (opt.) | yes |
| T3 determinant/Gram | appendix l.47–77 | KNOWN, in technique and inequality. Both verifiers' upheld=false is over the Horn–Johnson DOI | horn2012matrix; thompson1976convex; weyl1949; seiler1975inequality; schilling2012bernstein; tian2025composing | no |
| T4 roots-of-unity phases | cyclic.tex l.58; appendix l.79–81 | PARTIALLY_KNOWN; the ML verifier's upheld=false: ν=3 is in Tian | mohlenkamp2005trig; tian2025composing; friedland2018nuclear; comon2012typical; gromov2023grokking | no |
| T5 concave vertex | appendix l.134 | KNOWN (the proof no longer calls the theorem; citation optional) | rockafellar1970convex [§32]; hirsch1961extreme; ge2011note | yes |
| T6 exp-sum zeros | appendix l.136–142 | KNOWN | jameson2006counting [Cor 3.2]; polya1998problems [Part V] | yes |
| T7 active constraints | appendix l.147 | KNOWN | rosset2004boosting (opt.); Bertsimas–Tsitsiklis (see §f) | yes |
| T8 random frequencies | appendix l.157–168 | KNOWN | alon1994random [Sec 3, Thm 2 and Remark]; hoeffding1963probability; mccracken2025 [Thm 4.7] | yes |
| T9 positive Fourier | appendix l.170–181 | PARTIALLY_KNOWN; the maths verifier's upheld=false: a weighted precedent exists (Wang–Ye) | cioaba2006closed; friedman2006spectral; wang2025positive; dlmf | yes |
| T10 Bessel bound | appendix l.183–188 | KNOWN (textbook; the proof is correct) | dlmf | yes |
| T12 dense uniqueness | appendix l.26–33 | **Disagreement:** maths says KNOWN (Rankin; Morwani F.3), ML says PARTIALLY_KNOWN. Final: KNOWN; at most, uniqueness in the reduced programme is "made explicit" | rankin1955closest; morwani2024feature [App F.3]; li2025nsum [Lemma F.2] | yes |
| T13 ReLU budget | appendix l.260 (**commented out**) | PARTIALLY_KNOWN; the constant is wrong | morwani2024feature [App F.1] | yes |
| F1 long-desired | paper.tex l.55 | PARTIALLY_SUPPORTED | elhage2022superposition; sharkey2025open | yes |
| F2 weight decay sparse | paper.tex l.55; intro.tex l.5 | PARTIALLY_SUPPORTED ("consistently" is contradicted) | nanda2023progress; chughtai2023toy; zhong2023clock; stander2024grokking; mccracken2025; morwani2024feature [App C Fig 7a] | yes |
| F3 explains transition | paper.tex l.56; intro.tex l.5 | PARTIALLY_SUPPORTED (misattributed trigger; out of the paper's own scope) | mccracken2025 | yes |
| F4 margin peaks | paper.tex l.57 | PARTIALLY_SUPPORTED; unsupported in the current paper | morwani2024feature [Thm 7]; chughtai2023toy | no |
| F5 why circuits | **stale** (sentence removed) | SUPPORTED if reinstated; attribute it to Morwani | morwani2024feature; olah2020zoom; elhage2021mathematical | yes |
| F6 where dense stops | **stale** (paragraph removed) | SUPPORTED | morwani2024feature [Lemma 5, App E.1]; hoff2017lasso; kolb2025deep; gunasekar2018implicit | yes |
| F7 log growth | intro.tex l.13 | PARTIALLY_SUPPORTED (credit McCracken's empirical log fit) | mccracken2025 [Fig 3, Cor 4.8, Sec 5.1] | yes |
| F8 groups observed | groups.tex l.4 | PARTIALLY_SUPPORTED | chughtai2023toy [Tables 5–6, Fig 7]; stander2024grokking | yes |
| F9 scope (GD) | approach.tex l.6–7 | SUPPORTED; citations optional | lyu2020gradient; lyu2024dichotomy; vardi2022margin | yes |
| F10 no related work | whole paper | SUPPORTED (the gap is confirmed) | see §(e) | no |
| C1 Morwani dense baseline | paper.tex l.55; intro.tex l.3 | PARTIALLY_SUPPORTED (overgeneralised) | morwani2024feature [Thm 7, Thm 9]; li2025nsum [Thm 4.1] | yes |
| C2 Morwani/Wei principle | intro sentence removed; approach.tex l.6 | SUPPORTED as a citation, but "introduced/pioneering principle" misattributes (A2) | wei2019regularization; morwani2024feature | yes |
| C3 Wei Thm 4.1 | approach.tex l.6 | PARTIALLY_SUPPORTED (pointer should include Thm C.1) | wei2019regularization | no |
| C4 Nanda handful | intro.tex l.5 | SUPPORTED | nanda2023progress | yes |
| C5 Chughtai irreps | paper.tex l.57; intro.tex l.5 | PARTIALLY_SUPPORTED (S5 yes; A5 only in transformers; C113 MLPs 10–15) | chughtai2023toy; stander2024grokking; wu2025unified | yes |
| C6 McCracken transition | paper.tex l.56; intro.tex l.5 | PARTIALLY_SUPPORTED; bib entry wrong (the citation verifier's upheld=false is over the NeurIPS entry) | mccracken2025 (corrected) | yes |
| C7 Morwani definitions | approach.tex l.3 | SUPPORTED | morwani2024feature [Sec 2–3] | yes |
| C8 Morwani Sec 3 | approach.tex l.11 (the η sentence is stale) | SUPPORTED (pointer → Sec 3.2) | morwani2024feature | no |
| C9 feature definition | approach.tex l.32 | PARTIALLY_SUPPORTED ("as in" is too strong) | nanda2023progress; morwani2024feature | no |
| C10 phase condition | **stale** | SUPPORTED | morwani2024feature [Thm 7] | yes |
| C11 single-neuron | **stale** | PARTIALLY_SUPPORTED | morwani2024feature [Lemma 5] | no |
| C12 Morwani constant | approach.tex l.90; appendix l.32, 257 | SUPPORTED (add the m≥4(p−1) width condition) | morwani2024feature [Thm 7] | yes |
| C13 Hoeffding | appendix l.161 | SUPPORTED | hoeffding1963probability [Thm 2] | yes |
| C14 Alon–Roichman | appendix l.168 | SUPPORTED (pin Sec 3, Thm 2) | alon1994random | yes |
| U1 representation cost | approach.tex l.50–56 | CITATION_FOUND | dai2021representation; tibshirani2021equivalences; hoff2017lasso; kolb2025deep; gunasekar2018implicit; grandvalet1998least | yes |
| U2 grokking testbed | intro (no current sentence) | CITATION_FOUND (optional framing) | power2022grokking; liu2022towards; gromov2023grokking; zhong2023clock | yes |
| U3 n-operand | prelim.tex l.8 | CITATION_FOUND (**must add**) | li2025nsum; doshi2024grokking | yes |
| U4 matrix tools | appendix l.50–72 | CITATION_FOUND | horn2012matrix; dlmf; schilling2012bernstein | yes |
| U5 Bessel/Jacobi–Anger | appendix l.172–181 | CITATION_FOUND | dlmf | no |
| U6 cosine-sum minima | cyclic.tex l.107–134 | CITATION_FOUND (the verifier fixed three errors in the proposed text) | alon1994random; friedman2006spectral | no |
| U7 group tensor | groups.tex l.4 | CITATION_FOUND (add Derksen) | derksen2016nuclear; shutman2025words; strassen1969; blaser2013 | yes |
| U8 nuclear norm | appendix l.262 | CITATION_FOUND | srebro2004mmmf; parkinson2025linear | yes |
| U9 product-to-sum | cyclic.tex l.6–7 | CITATION_FOUND | gromov2023grokking; morwani2024feature; li2025nsum; he2026mechanism | yes |

---

## (c) Novelty statements the paper can defend

Every statement for an SL item carries the qualifier "as far as we found; our literature search was limited". The recommended wording is the quoted sentence.

**Not contributions; present as known or cited:** lem:avg; eq:spectrum; cor:dense; C_b and B_p; the AM–GM exponent; ℓ_r subadditivity; lem:sign (bookkeeping); the classical steps inside prop:shapes, prop:bottleneck and app:scaling.

- **N1. Theorem universal (SL).** "For every width and every $2\le b\le\nu$, the maximum $L_{2,b}$-margin of one-hidden-layer product and power networks on modular addition is at most $C_b^{\nu/b}\Gamma_{\nu/b}(p)$. At $b=\nu$ this recovers \citet[Thm.~7]{morwani2024feature} and \citet[Thm.~4.1]{li2025nsum}; for $b<\nu$, including plain weight decay ($b=2$), we are not aware of an earlier upper bound. The ingredients (the averaged-margin inequality of Morwani et al., Fourier-domain AM--GM pricing \citep{gunasekar2018implicit}, $\ell_r$ subadditivity) are standard; the contribution is combining them through $\Gamma_s$."
- **N2. The programme Γ_s (SL).** "We reduce the margin to $\Gamma_s(p)$, the largest worst-offset coverage under an $\ell_{1/s}$ budget. Such programmes are standard ($\ell_q$ minimisation over a polyhedron; spectral-gap design for weighted circulant graphs); what is specific here is that homogeneity fixes the exponent, $s=\nu/b$, and that $\Gamma_s$ bounds every network. Concurrently, \citet{kam2026symmetry} study the same objective under an $\ell_2$ budget, where the full-support optimum is uniform."
- **N3. Lemma cost (SL).**
  - Claim: "Interference-freeness makes the chain matrix $H$ triangular with diagonal $\beta$ (Lemma~\ref{lem:chain}). From this the $\ell_r$ cost bound follows; it can also be derived from Rotfel'd's trace inequality. We characterize equality for every $\nu$ and $r<1$: exactly $\nu$ contributions of modulus $|\beta|/\nu$ with orthogonal chain vectors. The inequality needs only the interval patterns."
  - Credit Tian (ν=3, MSE, minimum norm among 3-node solutions) and Friedland–Lim (ν=3, r=1).
  - Do not claim "first proof that ν neurons are necessary".
- **N4. Theorem cyclic plus Prop block (SL).** "For $2\le b<\nu$ we determine the exact maximum $L_{2,b}$-margin over interference-free product networks, $\nu^{-(\nu-b)/b}B_p\Gamma_{\nu/b}(p)$, and show that every optimum uses exactly $|\mathrm{spt}\,y^*|$ frequencies, each carried by exactly $\nu$ equal neurons. The attaining $\nu$-neuron construction is known \citep{mohlenkamp2005trig,marchetti2026sequential,tian2025composing}; the value, its weighting $\lambda_j^\nu\propto y^*_j$, the count and the necessity statements are new as far as we found." For ν=3 the block is also a full-diversification configuration of He et al. (Sec 4); add \citet{he2026mechanism} to the list of known constructions.
- **N5. Prop shapes (SL).** "Applying classical facts (concave minima lie at vertices; Laguerre's rule of signs) shows that optimal supports and amplitude ratios are piecewise constant in $s$, with finitely many switches." The p=13 and p=23 switch values are our computations.
- **N6. Prop bottleneck (SL).** "At an optimum, at least $K$ offsets are simultaneously tight and fix the amplitude ratios. This is the linear-programming basic-solution property (at $s=1$, Shapley--Snow). In the block network these become $2K$ simultaneously hardest wrong answers." Only the last reading is new.
- **N7. Prop ablation (SL).** "Deleting feature $j$ from a maximizer costs at least a fraction $1-(1-w_j)^s\in[w_j,sw_j]$ of the margin, where $w_j$ is its share of the regularization budget. We found no prior bound of this kind. The heuristic that importance follows the norm share goes back to SVM-RFE \citep{guyon2002gene}."
- **N8. Thm scaling (SL).**
  - Claim: "For fixed $s>1$, $\Gamma_s(p)=\Theta((\log p)^{1-s})$, and every maximizer uses $\Theta(\log p)$ frequencies, each with budget share at least $4^{-1/(s-1)}/(9\ln p)$. Consequently the maximum $L_{2,b}$-margin of product networks with $2\le b<\nu$ is $\Theta(p^{-\nu/2}(\log p)^{1-\nu/b})$, interference allowed."
  - Credit: "The equal-weight analogue (abelian Cayley graphs need and admit $\Theta(\log p)$ generators; \citealp{alon1994random,friedman2006spectral}), fixed-$K$ weighted bounds (simultaneous Dirichlet approximation; \citealp{wang2025positive}) and the sufficiency of $O(\log p)$ random frequencies for an unnormalized margin \citep{mccracken2025} are known."
  - What is new: the two-sided bound under the concave budget with explicit constants, the support upper bound with share floor, and the network rate.
  - Do not present the positive-Fourier technique as the contribution. The order of Γ_s also follows from weighted pigeonhole.
- **N9. Cor few, narrowed (SL).** "For networks: any product or power network, of any width and with interference allowed, whose amplitudes are supported on $K=o(\log p)$ frequencies attains a vanishing fraction of the optimal $L_{2,b}$-margin. At the level of the programme this is classical: the qualitative equal-share analogue is that abelian Cayley graphs with $o(\log n)$ generators do not expand \citep[Prop.~3]{alon1994random}; fixed $K$ with any weights gives simultaneous Dirichlet approximation, with the sharper decay $p^{-2/K}$; and $K=o(\log p)$ under the $\ell_{1/s}$ budget follows from pigeonhole with frequency-dependent radii \citep{cassels1959introduction}. The contribution is the transfer to network margins through Lemma~\ref{lem:phase} and Theorem~\ref{thm:universal}."
  - Alon–Roichman Prop 3 concerns vertex expansion and gives no rate. For s>1 the corollary needs the rate, so call Prop 3 the qualitative analogue, not the corollary (R26 ML verifier).
- **N10. Dense window (SL, low confidence).** "The programme optimum can use every frequency only if $s-1=O(1/\log p)$. This proven envelope is loose: it is vacuous for $p<73$. We compute the exact thresholds for $p\le23$." Optionally add that the computed window behaves like about 3.7/p. No prior statement was found.
- **N11. The K* table (SL).** These are the paper's own computations. Say "computed values; exhaustive for $K\le8,7,5$, heuristic above". Present comparisons with trained transformers and MLPs as qualitative only.
- **N12. Groups (SL; no proofs yet).** "Under $L_{2,b}$, the price of an irreducible representation is the Schatten-$(b/\nu)$ quasi-norm of the matrix-multiplication tensor. At $b=\nu=3$ this is the nuclear norm of the matrix-multiplication tensor, for which the schoolbook decomposition is optimal \citep[Cor.~1.10]{derksen2016nuclear}; norm-free tensor-rank width bounds are due to \citet{shutman2025words}. New are the degree-$\nu$/$L_{2,b}$ price and the resulting count within the interference-free class."
  - Only claim S5/A5 agreement as stated in A7, and only after the appendix contains proofs.
- **N13. lem:phase (SL).** "For every network and every offset, $h(\theta)\le\sum_jA_j\kappa_j(\delta)$. The right-hand side is the familiar separation of aligned cosine logits \citep{nanda2023progress,mccracken2025}, and averaging over $\delta$ recovers Morwani et al.'s uniformly class-weighted margin; we did not find the per-offset bound stated for arbitrary networks."
- **N14. def:if (SL).** Claim only "the logit-table class for general $n$, isolated from other-mode interference and used as the domain of an exact margin theorem". At n=2 the cancellation system is in Kunin et al., Tian, and He et al.

---

## (d) Citation insertion plan

Keys marked † are new (in `refs_additions.bib`). Keys marked ‡ replace an existing refs.bib entry.

| # | File, line | Sentence (quoted) | Action | Keys |
|---|---|---|---|---|
| 1 | paper.tex 55 | "Explaining the number of features in a network has been a long-desired result…" | Rewrite (A8) | elhage2022superposition†, sharkey2025open†; opt. levine2026math† (check the wording of its Sec. 4.3 first) |
| 2 | paper.tex 55 | "…laid the baseline with their principle of margin-maximization, showing provably … all features would be used…" | Rewrite (A2, A3) | morwani2024feature [Thm 7, Thm 9], li2025nsum‡ [Thm 4.1] |
| 3 | paper.tex 55 | "…consistently show a sparse selection of features." | Rewrite (A4) | nanda2023progress, chughtai2023toy‡, zhong2023clock†, morwani2024feature [App C, Fig 7a], mccracken2025‡ |
| 4 | paper.tex 56 | "Hence we show that an optimal number of features…" / "trade-off …" | Add a clause crediting the known mechanism | dai2021representation†, gunasekar2018implicit† |
| 5 | paper.tex 56 | "…explains the transition … as seen empirically in \citet{mccracken2025}…" | Rewrite (A5) | mccracken2025‡ |
| 6 | paper.tex 57 | "…we empirically show that the normalized margin … peaks … matching the result of…" | Delete or rewrite (A6) | morwani2024feature [Thm 7] |
| 7 | paper.tex 57 | "The optimums are also consistent with … \citet{chughtai2023toy}." | Scope to S5 (A7) | chughtai2023toy‡ |
| 8 | intro.tex 3, 7 | "…uses the pioneering principle of margin-maximization…" / "…principle introduced by…" | Rewrite (A2) | wei2019regularization‡, lyu2020gradient, morwani2024feature |
| 9 | intro.tex 5 | "…as observed by \citet{nanda2023progress} and \citet{chughtai2023toy} … a sparse subset…"; "\citet{mccracken2025} similarly sees…" | Add the depth/embedding trigger; add Stander; optional grokking framing | stander2024grokking†, mccracken2025‡, zhong2023clock†; opt. power2022grokking, liu2022towards† |
| 10 | intro.tex 13 | "A general scaling law for the optimal number of frequencies…" | Add: "(logarithmic growth was observed empirically by \citet{mccracken2025}, who show $O(\log p)$ random frequencies suffice for an unnormalized margin)" | mccracken2025‡ |
| 11 | approach.tex 6 | "By \citet[Theorem~4.1]{wei2019regularization}, the global minimizers…" | Pointer: add Thm C.1 | wei2019regularization‡ |
| 12 | approach.tex 7 | "…without considering empirical achievability." | Optional: add the E7 sentence | lyu2020gradient, vardi2022margin†, lyu2024dichotomy†; opt. ji2020directional† (binary gradient flow only) |
| 13 | approach.tex 11 | "…averaged class-weighted margin \citep[Section~3]{morwani2024feature}." | Section 3.2; add Li | morwani2024feature, li2025nsum‡ [Def 3.6] |
| 14 | approach.tex 23 | "…it is termed \emph{interference}." | Add "(the `destructive interference' of \citealp{gromov2023grokking})" | gromov2023grokking |
| 15 | approach.tex 27–31 | eq:spectrum paragraph | Add the C4 sentence | morwani2024feature [App F.1, F.3], li2025nsum‡ |
| 16 | approach.tex 32 | "…is a \emph{feature}, as in \citet{nanda2023progress,morwani2024feature}." | Reword (D2) | same |
| 17 | approach.tex 34–43 | after lem:phase: "The margin is thus bounded from above…" | Add: "averaging over δ recovers the uniformly class-weighted margin of \citet[App.~F.1]{morwani2024feature}; for aligned cosine logits the right side is the familiar separation \citep[Sec.~4.2]{nanda2023progress}, \citep[Thm.~4.7]{mccracken2025}." | morwani2024feature, nanda2023progress, mccracken2025‡ |
| 18 | approach.tex 50–56 | "For a product neuron, the amplitude A is proportional to the product of the weights. Hence, the minimum penalty…" | Add the C3 sentence and fix the wording | hoff2017lasso†, tibshirani2021equivalences†, dai2021representation†, kolb2025deep†, gunasekar2018implicit†; opt. grandvalet1998least† |
| 19 | approach.tex 63–66 | after eq:programme | Add: "Equivalently, $\Gamma_s$ is the largest spectral gap of a weighted circulant graph on $\sZ_p$ under an $\ell_{1/s}$ budget on edge weights; at $s=1$ this is convex edge-weight design \citep{boyd2006convex,sun2006fastest}, and minimizing $\sum_jv_j^{1/s}$ over a polyhedron is a standard concave ($\ell_q$, $q<1$) problem \citep{ge2011note}. Concurrently, \citet{kam2026symmetry} study the same worst-offset separation under an $\ell_2$ budget, where the full-support optimum is uniform." | boyd2006convex†, sun2006fastest†, ge2011note†, kam2026symmetry†; opt. ghosh2006growing†, jacobs2026mirror† |
| 20 | approach.tex 80 | "For the product neuron $C_b^{\nu/b}=B_p$ … power neuron $C_b^{\nu/b}=n!\,B_p$" | Add the C2 sentence | morwani2024feature, li2025nsum‡, gunasekar2018implicit†, hoff2017lasso† |
| 21 | approach.tex 90 | cor:dense "…exactly their maximum $L_{2,3}$-margin." | Add the C1 text | morwani2024feature [Thm 7, App F.3], li2025nsum‡ [Thm 4.1], rankin1955closest†, slepian1968group† |
| 22 | appendix 30 | app:dense "The flat choice $y_j=2/(p-1)$ attains this, as we now show together with uniqueness." | Add: "(the averaging step is the regular-simplex bound \citealp{rankin1955closest}; cf. \citealp[§3.5--3.6]{fiedler1973algebraic}; the inversion step parallels \citealp[App.~F.3]{morwani2024feature} and \citealp[Lemma~F.2]{li2025nsum})" | rankin1955closest†, fiedler1973algebraic†, morwani2024feature, li2025nsum‡ |
| 23 | prelim.tex 8 | "The \emph{product neuron} is … and the \emph{power neuron} is … with $\nu=n+1$." | Add: "The $n$-operand task and the power neuron are those of \citet{li2025nsum} (see also \citealp{doshi2024grokking}); at $n=2$ the power neuron is the quadratic neuron of \citet{morwani2024feature}." | li2025nsum‡, doshi2024grokking†, morwani2024feature |
| 24 | cyclic.tex 7 | "…the product-to-sum formulas also produce \emph{sign-flipped} copies…" | Add: "These copies appear in the analytic solutions of \citet[Eqs.~10--11]{gromov2023grokking}, which cancel them approximately through random phases; exact cancelling constructions are \citet[App.~F.2]{morwani2024feature} and \citet[App.~E]{li2025nsum}." | gromov2023grokking, morwani2024feature, li2025nsum‡; opt. doshi2024grokking† |
| 25 | cyclic.tex 13 (after def:if) | "The condition is on the network's logit table…" | Add: "For $n=2$ the same cancellation system appears as an optimality condition in \citet[Thm.~F.7]{kunin2025alternating}, inside the zero-loss conditions of \citet[Lemma~1]{tian2025composing}, and as full diversification in \citet[Sec.~4]{he2026mechanism}; we treat general $n$ and price it (Lemma~\ref{lem:cost})." | kunin2025alternating†, tian2025composing†, he2026mechanism† |
| 26 | cyclic.tex 19 (after lem:sign) | "So a single neuron is never interference-free at a frequency it uses…" | Add: "The lemma is Hermitian symmetry of the weights' Fourier transform (cf. \citealp[Eq.~11]{gromov2023grokking})." | gromov2023grokking |
| 27 | cyclic.tex 34 (after the proof idea of lem:cost) | "…gives $\sum_lx_l^r\ge\nu$." | Add the C6 sentence | thompson1976convex†, tian2025composing†; opt. mohlenkamp2005trig†, friedland2018nuclear†, tadej2006concise† (the equality case is a dephased complex Hadamard matrix) |
| 28 | cyclic.tex 58 | "The phases are a rotated copy of the $\nu$\textsuperscript{th} roots of unity…" | Fix the roots-of-unity wording; add the C7 sentence | mohlenkamp2005trig†, marchetti2026sequential†, tian2025composing†, he2026mechanism†; opt. friedland2018nuclear†, comon2012typical† |
| 29 | appendix 134, 136–142 | app:shape (i), (ii) | Optional: "(classical; \citealp{hirsch1961extreme}, \citealp[§32]{rockafellar1970convex})"; "(Laguerre's rule; \citealp[Cor.~3.2]{jameson2006counting})" | hirsch1961extreme†, rockafellar1970convex†, jameson2006counting† |
| 30 | cyclic.tex 71 / appendix 136 | "Thus the available optimal shapes change only at finitely many exponents…" | Optional: "(for large $s$ the final shape is a minimum-support vertex; cf. \citealp{fung2011equivalence}); global minimization of $\sum v_j^{1/s}$ over a polyhedron is NP-hard in general \citep{ge2011note}, hence the exhaustive/heuristic search in Table~\ref{tab:kstar}." | fung2011equivalence†, ge2011note† |
| 31 | cyclic.tex 77 / appendix 147 | prop:bottleneck; "The active constraints at a vertex have rank…" | Add: "(at $s=1$ this is the Shapley--Snow description of extreme optimal strategies; \citealp{shapley1950basic})" | shapley1950basic†; opt. rosset2004boosting† |
| 32 | cyclic.tex 87 | "Every feature is load-bearing in proportion to its share of the regularization budget." | Add: "This is a quantitative, idealized analogue of the key-frequency ablations of \citet[Sec.~4.4]{nanda2023progress}; the heuristic goes back to SVM-RFE \citep{guyon2002gene}." | nanda2023progress, guyon2002gene† |
| 33 | cyclic.tex 122 (after thm:scaling) | "The last statement needs no answer to the open question…" | Insert: "With equal shares, $\min_{\delta\ne0}\sum_{j\in S}\cov{j}{\delta}$ is $|S|$ minus half the second eigenvalue of $\mathrm{Cay}(\sZ_p,\pm S)$, so the unweighted analogue of Theorem~\ref{thm:scaling} is the classical fact that abelian Cayley graphs need, and admit, $\Theta(\log p)$ generators \citep[Sec.~3]{alon1994random}; for fixed $K$ see \citet{friedman2006spectral}, and for arbitrary weights \citet{wang2025positive}. Empirically, trained networks' frequency counts grow logarithmically, and $O(\log p)$ random frequencies suffice for an unnormalized margin \citep{mccracken2025}." | alon1994random‡, friedman2006spectral†, wang2025positive†, mccracken2025‡ |
| 34 | cyclic.tex 130 (after cor:few) | "For fixed $\nu,b$, product networks with $K=o(\log p)$…" | Add the N9 credit sentence | alon1994random‡, schmidt1980diophantine†, cassels1959introduction† |
| 35 | appendix 168 | "This is the cyclic case of the random Cayley graph argument of \citet{alon1994random}." | Pin "[Sec.~3, Thm.~2]"; add "cf. \citet[Thm.~4.7]{mccracken2025}" | alon1994random‡, mccracken2025‡ |
| 36 | appendix 176–181 | "…show that all Fourier coefficients are non-negative, with constant coefficient $I_0(z)$." / "…resonances only add non-negative terms." | Add \citep[Eqs.~10.32.1, 10.35.2, 10.25.2]{dlmf}; add "the continuous-time closed-walk comparison of \citet{cioaba2006closed}" | dlmf†, cioaba2006closed† |
| 37 | appendix 50–57, 66–72 | "Sylvester's identity and expansion by principal minors…"; "Hadamard's inequality…"; $J_r$ substitution | D4 rename; \citep[Chs.~1, 7]{horn2012matrix}; \citep[Eqs.~5.12.1, 5.12.3, 5.5.3]{dlmf} if the Beta-integral constant is stated | horn2012matrix†, dlmf† |
| 38 | appendix 84 | Remark "other penalties" | Add: "(the admissible $\psi$ are Thorin--Bernstein functions, \citealp[Ch.~8]{schilling2012bernstein}; for the inequality alone see also \citealp{thompson1976convex})" | schilling2012bernstein†, thompson1976convex†; opt. rajan2015stochastic† |
| 39 | appendix 161 | "By Hoeffding's inequality…" | Optional pin "Thm.~2" | hoeffding1963probability‡ |
| 40 | appendix 262 | "…the shared layer is priced by a nuclear norm that rewards carrying many frequencies in one neuron" | Cite and soften to "does not penalize" | srebro2004mmmf†; opt. parkinson2025linear† |
| 41 | appendix 286 | "For $b>\nu$ the normalized margin is unbounded as the width grows." | Optional "(cf. the unit-duplication argument of \citealp[Sec.~3.2]{neyshabur2015norm})" | neyshabur2015norm† |
| 42 | appendix 235 | Equal-shares paragraph "Since the worst offset is no better than the average, $M_K\le pK/(p-1)$" | Optional "(\citealp[Thm.~2]{slepian1968group})" | slepian1968group† |
| 43 | groups.tex 4 | "…a price set by how cheaply the matrix-multiplication tensor can be written as a sum of products; at even degree this is the schoolbook algorithm…" | Add: "Without a norm, the width per real irreducible representation is bounded by the tensor rank of matrix multiplication \citep{shutman2025words,strassen1969,blaser2013}; under the norm the relevant quantity is a Schatten quasi-norm \citep{fan2023euclidean}, and at $b=\nu=3$ the schoolbook decomposition is nuclear-norm optimal \citep[Cor.~1.10]{derksen2016nuclear}, which also yields the $\sum d^{5/2}$ of \citet[Thm.~9]{morwani2024feature} (\citealp[Cor.~1.12]{derksen2016nuclear}). At even $\nu$ we use the bipartite flattening of \citet{christandl2019surgery}." | shutman2025words†, strassen1969†, blaser2013†, fan2023euclidean†, derksen2016nuclear†, christandl2019surgery†, morwani2024feature; opt. ye2018structured†, marchetti2026sequential† |
| 44 | groups.tex 4 | "…one representation on $S_5$ and the Galois-conjugate pair on $A_5$, as observed." | Rewrite (A7) | chughtai2023toy‡, stander2024grokking†; opt. wu2025unified†, he2026spectral† (neurons align to single irreps under gradient flow, Thm 4.3(ii); no count) |
| 45 | groups.tex 4 | "…networks with structured interference can exceed the ceiling on some groups…" | Optional "(cf. the coset circuits of \citealp{stander2024grokking})" | stander2024grokking† |
| 46 | new related-work paragraph | §(e) | Insert | see §(e) |

---

## (e) Draft related-work paragraph (about 260 words; only verified keys)

```latex
\paragraph{Related work.}
\citet{morwani2024feature} characterize the maximum $L_{2,3}$ margin of quadratic networks on modular addition and group composition, and \citet{li2025nsum} extend the modular-addition result to $k$-operand power networks at $L_{2,k+1}$; in both, every frequency (or irreducible representation) is used, and our bound recovers both values at $b=\nu$.
Maximum-margin solutions are the limit of weakly regularized global minimizers \citep{wei2019regularization}, and limit directions of gradient flow on homogeneous networks are KKT points of the $L_2$ margin problem \citep{lyu2020gradient,lyu2024dichotomy}, which need not be global maxima \citep{vardi2022margin}.
That $\ell_2$ weight decay on a product of factors acts as a concave $\ell_{2/L}$-type penalty, including on Fourier coefficients of linear convolutional networks, is well known \citep{hoff2017lasso,gunasekar2018implicit,dai2021representation,kolb2025deep}; our contribution is its interaction with the worst-offset coverage of modular addition.
Trained networks often use a handful of frequencies or representations \citep{nanda2023progress,chughtai2023toy,zhong2023clock,stander2024grokking}, while one-hot one-hidden-layer networks can use all of them \citep{morwani2024feature,mccracken2025}. \citet{mccracken2025} observe that the count grows logarithmically with the modulus and show that $O(\log n)$ random frequencies suffice for an unnormalized margin.
Analytic Fourier solutions cancel unwanted product-to-sum terms approximately through random phases \citep{gromov2023grokking,doshi2024grokking}; exact cancellation by three nodes per frequency appears under squared loss in \citet{tian2025composing}, and roots-of-unity product-neuron constructions in \citet{marchetti2026sequential}.
Concurrently, \citet{kam2026symmetry} analyze the same worst-offset separation under an $\ell_2$ amplitude budget, where the uniform allocation is optimal at full support.
On the mathematical side, the unweighted analogue of our logarithmic law is the classical fact that abelian Cayley graphs need, and admit, logarithmically many generators to expand \citep{alon1994random,friedman2006spectral}.
For groups, \citet{shutman2025words} relate width to the tensor rank of matrix multiplication, and \citet{derksen2016nuclear} shows that the schoolbook decomposition minimizes its nuclear norm.
```

Caveats before use:
- Stander et al. read S5 concentration as coset circuits, so cite them for sparsity, not for irreps.
- Lyu et al. 2024 is binary with L2 only.
- If space allows, add Jacobs–Montúfar ("an exponent of the mirror potential decides sparse vs dense max-margin solutions"), describing it as an optimizer-geometry exponent.

---

## (f) Unconfirmed candidates (not in `refs_additions.bib`)

1. **MAIS-O52 registry page** (github.com/lionellevine/MAIS, open-problems/MAIS-O52.md). Seen only by the LIT_cite sweep, and the two sweeps disagree about it: LIT_ml says O52 is referenced only as an internal document and should not be cited by number. The page credits AI drafting. Levine, arXiv 2609.15289, is confirmed and is in the bib. The claim that its Sec 4.3 (MAIS-O5) poses "which irreps are learned … as a function of width and decay strength" rests on the sweep only; read Sec 4.3 before citing it for that.
2. **Tsilivis et al., *Flavors of Margin* (ICLR 2025, arXiv 2410.22069).** The sweeps disagree on the author list (three authors vs four). Not checked.
3. **Xie & Li, ICML 2024 (AdamW converges to ℓ∞-constrained KKT points)** and **Ding et al., arXiv 2405.17420 (more surviving frequencies with larger embedding dimension).** Sweep only. They are validity attacks on any AdamW or width-invariance experiments; check them before experiments.tex is written.
4. **Chizat & Bach COLT 2020; Mohamadi et al. ICML 2024; Varma et al. 2023; Rangamani ICASSP 2025; Wen & Jacot 2024; Hermon & Olesker-Taylor AAP 2023; Basu–Kothari–Meka–Tudose 2607.08261; Kam–Bessafi–Cadet 2609.15047; Pracher et al. 2609.26679; Memana 2608.04451; Larson 2609.12259.** Sweep only (not checked by a checker or verifier).
5. **Tao & Vu 2006 (Lemma 4.20); Pisier 1981 (Maurey's empirical method); Koiran–Portier–Tavenas 2012; Curtiss 1918; Bauer 1958; Ajtai et al. 1990; Bilu–Lev–Ruzsa 1998; the Chowla-problem papers.** Sweep only, or seen through secondary sources. The Chowla papers should not be cited as support; they concern the opposite extremum.
6. **Bertsimas & Tsitsiklis 1997, Thm 2.3.** The content is confirmed only through secondary sources; the book was not fetched, and there is no url or doi. Add it by hand if wanted, or cite the LP fact without a reference.
7. **Causa & Re 2011.** Volume and pages are not consistently verified. Use Comon & Ottaviani instead.
8. **Items that are confirmed but optional and were left out to keep the bibliography lean:** beylkin2002numerical, beylkin2005algorithms, reznick2013length, bourin2007subadditivity, zhang2013bernstein, poon2023smooth, neyshabur2015search, mangasarian1996machine, mangasarian1999minimum, breiman1999prediction, hildebrand2005survey, chen2013smallbias, naor1993small, alon1992simple, alon2009perturbed, jagadeesan2022inductive, gunasekar2017implicit, recht2010guaranteed, savarese2019infinite, ongie2022linear, ongie2026representation, kalton1984fspace, cheng2025variational, dumas2024strassen, burgisser1997algebraic, horst1996global, borwein1995polynomials, bhatia1997, tan2026beyond, zhao2023graph, boyd2004convex, lecun1989obd.
   - Their checked BibTeX is in `items/*/check.json` and `verify_*.json`.
   - Their locators are unverified where noted (Bhatia's theorem numbers, Horst & Tuy's theorem numbers).
9. **Theorem and section numbers read only on arXiv.** Numbers for Morwani (v2), Li (v4), Gunasekar (v2 lemma numbering), McCracken (v1), Marchetti (v2) and Tian (v4) were read on arXiv versions; the camera-ready versions were not checked. Where the insertion plan pins a number, re-check it against the published PDF if you can, or cite by content.

---

## Audit log (2026-09-26)

Scope: every entry of `refs_additions.bib` re-fetched; every KNOWN / PARTIALLY_KNOWN / NOVEL statement in §(a)–(e) checked against `items/*/CHECK.md` and `VERIFY_*.md`; every quoted paper sentence checked against the current `paper.tex` and `sections/*.tex`. No paper file was edited. WebSearch was not used.

### Bibliography (75 entries: 7 replacements + 68 new)
- **Method.**
  - Crossref API for all 37 DOIs, plus the new ye2018structured DOI.
  - arXiv abs pages (citation meta and comments) for all 29 arXiv-linked entries.
  - PMLR pages for li2025nsum, chughtai2023toy, stander2024grokking and neyshabur2015norm.
  - NeurIPS proceedings pages for wei2019regularization, gunasekar2018implicit, dai2021representation and srebro2004mmmf.
  - Primary pages for the rest:
    - theoryofcomputing.org (blaser2013);
    - dml.cz (fiedler1973algebraic);
    - transformer-circuits.pub author blocks (both Elhage entries);
    - Boyd's page (boyd2006convex);
    - dlmf.nist.gov/help/cite;
    - the Tibshirani and JMLR PDFs;
    - the TMLR accepted-papers list (sharkey2025open, fan2023euclidean);
    - Open Library ISBN records (schmidt1980diophantine: Crossref lists no author; schilling2012bernstein: series and edition).
  - OpenReview itself returned a bot challenge, so the McCracken venue rests on Crossref (NeurIPS 38, pp. 49512–49573) and the Sharkey venue on the TMLR list.
- **Result.** Title, authors, year and venue match for **75/75**. No entry deleted.
- **Changes.**
  1. `ye2018structured`: was an arXiv-only 2016 entry. It is published as Found. Comput. Math. 18(1):45–95 (2018), DOI 10.1007/s10208-016-9332-x (Crossref). The entry is replaced by the journal version, with the arXiv id kept in `note`.
  2. `olah2020zoom`: added volume 5, number 3 (Crossref).
  3. `fan2023euclidean`: added issn and the OpenReview url Grhi800jVz (TMLR list, January 2023); the arXiv id moved to `note`.
  4. `sharkey2025open`: provenance comment updated (author order and Survey certification confirmed on the TMLR list).
  5. `doshi2024grokking`: comment corrected ("running header", not "footer").
  6. File header: added the audit note.
- **Not in refs_additions but checked:** nanda2023progress, lyu2020gradient, gromov2023grokking and power2022grokking in `refs.bib` match their arXiv pages.
- **Noted but unchanged:**
  - `shapley1950basic` stays at 1950; Crossref's 1951-12-31 is a placeholder.
  - `friedland2018nuclear` is online 2017, volume 87 = 2018.
  - `cioaba2006closed` could add number 9.

### LITCHECK.md corrections
1. **A5.** "L2 is fixed at 1e-5 throughout (Table 5)" overstated what McCracken fix. The C6 verifier found App. G.9 / Fig 41 hyperparameter sweeps ("can be shifted based on hyperparameters"). Replaced with: L2 = 1e-5 in every Table 5 run; G.9 shifts the count; no regularization-driven transition is reported. (Sources: items/c6 VERIFY_citation, items/f3 VERIFY_citation.)
2. **A7 groups fix.** "matching the transformer runs of Chughtai Table 6" overclaimed for A5: the 3d pair alone appears in only 2/4 A5 transformer seeds. The fix now says 4/4 S5 and 2/4 A5 seeds (items/c5, f8, f4).
3. **C6.** Added that minimality of ν terms is already a theorem for tied (symmetric) decompositions (Comon–Ottaviani Prop 3.1) and that the ν=3 non-symmetric real rank is in Friedland–Lim Sec 6 (items/r21 VERIFY_novelty_math).
4. **C7.** Three changes:
   - Marchetti's phases are e^{iπj/(k+1)}, which are half-angle phases, not "(k+1)-th-root phases" (items/r20 VERIFY_novelty_ml).
   - Their Thm 4.3 width (k+1)2^k n_ρ^{k+1} is a sufficient width, not a lower bound on N (items/r31 VERIFY_novelty_ml).
   - Added He et al. Sec 4 (three phases {0, π/3, 2π/3}) as a ν=3 block (items/r9 VERIFY_novelty_ml, numerical check). The C7 fix sentence, N4 and insertion row 28 now cite he2026mechanism.
5. **C9.**
   - FMT: "do not call its constant absolute" contradicted FMT's own abstract. It now says the constant is absolute for fixed d, the verifiers disagree on uniformity in d, and FMT should not be used for K growing with p (items/r25, u6, r26).
   - "Schmidt LNM 785, Thm 1B": marked the theorem number as secondary-source only, which matches the bib comment.
6. **C12.** Added the ν=3 precedent for d^ν neurons per irrep: Morwani Thm 9 requires m ≥ 2Σd³ (items/r31 VERIFY_novelty_math).
7. **D6.** The FMT bullet is reworded to match item 5.
8. **E6.** "posted 10 days before the deadline" is corrected to 15 Sep 2026, 11 days before 26 Sep.
9. **Side finding 2.** Updated: appendix l.213 now derives the ⌈8 ln p⌉ floor. The mismatch is only that cyclic.tex l.134 attributes it to the theorem, which states 9 ln p.
10. **N9.** Alon–Roichman Prop 3 is about vertex expansion and gives no rate, so it is now called the "qualitative equal-share analogue", not the corollary (items/r26 VERIFY_novelty_ml, u6).
11. **N12 and insertion row 43.** Derksen's nuclear-norm schoolbook optimality is for the degree-3 tensor, so the statements are restricted to b=ν=3 (items/u7, r31).
12. **Insertion row 25.** Tian Lemma 1 gives zero-loss sufficient conditions, not an optimality condition. Reworded (items/r9 VERIFY_novelty_ml).

### Checks that found no error
- **Quoted sentences.** Every quoted paper sentence in §(a) and §(d), and every line range in §(b), matches the current files, whose mtimes are unchanged since the synthesis: approach 22:43; intro 19:40; prelim 20:02; cyclic, appendix and paper.tex 21:12.
- **Verdicts.** All 70 rows of §(b) match `record_for_synthesis.json` in checker verdict and SL flag, and the verifier corrections are reported faithfully. The spot-checked source facts match the item notes:
  - Wei Thm 4.1 / C.1;
  - Morwani Thms 7–9, App F.1–F.3, Fig 7a;
  - Li Thm 4.1 and Lemmas D.7–D.8;
  - Chughtai Tables 5–6;
  - Kam Eq 18 and Prop 5;
  - Gunasekar Lemma 7;
  - Thompson eq (1);
  - Seiler–Simon;
  - Weyl (page images read by the T3 verifier);
  - Rankin and Slepian Thm 2;
  - Jameson Cor 3.2;
  - Hirsch–Hoffman;
  - Derksen Cor 1.10 / 1.12;
  - Shutman Prop 5.2;
  - Wang–Ye Thm 4.6;
  - McCracken Thm 4.7 / Cor 4.8.
- **Remaining caveats.**
  - Theorem and figure numbers for Morwani, Li, McCracken, Marchetti and Tian come from arXiv versions; OpenReview blocked the camera-ready PDFs.
  - The weighted-pigeonhole bound in C9 rests on the R26 verifier and the synthesis re-derivation, not on a published statement.

---

## Applied to the paper (2026-09-26 ~08:10 IST)

Pre-edit snapshot: `iclr2027/archive_pre_litfix_20260926/` (all .tex files, refs.bib, paper.pdf). The paper compiles with tectonic; there are no undefined references or citations. It is 20 pages.

- **bib:** refs.bib is rebuilt from the 4 unchanged entries plus all 75 in refs_additions.bib, with no duplicate keys; dlmf gained year 2026.
- **Applied:**
  - A2 and A5 in the intro, with Li cited.
  - side-5 fixes in the intro: L_{2,b} and the typo.
  - rows 8–10; the related-work paragraph (§e, without the groups sentence).
  - E2 in prelim, plus prelim notation fixes: ω_i in (R^p)^{n+1}, indices, dataset sum.
  - D1 (the user asked whether it was needed; applied because Thm 4.1 is the binary result), E7, D3, C5, row 14, C4, D2, row 17, C3 (the budget paragraph was rewritten and the wording errors are gone), row 19, C2, N1, C1.
  - interference definition rewritten (approach).
  - rows 24–28, including the C7 roots-of-unity wording fix.
  - rows 29–34 (N9 wording for cor:few), side 2 (floor attribution, p≥73, "loose"), side 4.
  - appendix:
    - new App A.2 (proof of lem:phase) and App A.3 (lem:price, the AM–GM price under L_{2,b});
    - rows 22, 35–42; D4; D7; E9; side 1;
    - Li/Morwani attainment at b=ν in app:conjecture.
- **Not applied:**
  - abstract items A3, A4, A6, A7, A8 and rows 1–7 (the user writes the abstract);
  - groups items C12, A7 (groups part) and rows 43–45 (the groups section is dropped: `\input{sections/groups}` is commented out in paper.tex);
  - optional extras: comon2012typical, friedland2018nuclear, tadej2006concise at lem:cost; power2022/liu2022 grokking framing; the DLMF Beta-integral pins.
- **Pending the user:** cuts in cyclic.tex (shapes/bottleneck/cor:few/dense window).
