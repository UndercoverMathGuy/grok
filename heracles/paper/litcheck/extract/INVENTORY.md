# Claim inventory: "Explaining Feature Sparsity Through the Margin-Regularization Trade-off"

Source: heracles/paper/iclr2027 (paper.tex + sections/*.tex + refs.bib), read in full 2026-09-25.
The inventory only; no literature verdicts yet.

## Cite accounting
- 25 `\cite*` commands in total: abstract (paper.tex) 4, intro 7, prelim 2, approach 8, appendix 4; cyclic/groups/experiments/discussion 0. One is the multi-key `\citet{nanda2023progress,morwani2024feature}`. morwani2024feature appears in 16 of them.
- Bib entries that are **never cited**: lyu2020gradient, gromov2023grokking, li2025nsum, power2022grokking. They appear only in the TODO comment at the end of intro.tex.
- Bib metadata to verify: mccracken2025 has "McCracken, Andrew and others" (placeholder author list); morwani2024feature venue/year (ICLR 2024); li2025nsum venue (AISTATS 2025?); wei2019 title wording ("v.s."); chughtai ICML 2023.
- Sections groups/experiments/discussion are skeletons (TODO). There is no related-work section.

## A. Our results (theorems, lemmas, propositions, corollaries)

| id | location | statement | what to check |
|---|---|---|---|
| R1 lem:avg | approach.tex, Lemma "averaged margin" | h(θ) ≤ η_θ(0) − η_θ(δ) for δ≠0 | Trivial; network-level version of Morwani's class-weighted averaged margin (Morwani Sec. 3, η_{u,v,w}). Credit Morwani. |
| R2 eq:spectrum | approach.tex eq (spectrum), app:spectrum | η_θ depends only on diagonal Fourier modes (j,…,j;−j); cosine expansion with β_j | Standard Fourier computation; compare with Morwani, Gromov, Nanda, Li et al. (n-operand). |
| R3 lem:phase | approach.tex "phase lemma" | h ≤ Σ Re(β_j)κ_j(δ) ≤ Σ A_j κ_j(δ) | Analogue of Morwani's phase condition (Sec. 4); check that nobody states the amplitude/coverage bound. |
| R4 programme Γ_s(p) | approach.tex eq (programme) | Budget-splitting problem max min_δ Σ y_j κ_j(δ) s.t. Σ y_j^{1/s} ≤ 1 | Novelty of the programme; relates to minimax sidelobe design, Chowla cosine problem, ℓ_q-constrained LPs. |
| R5 prop:coverage | approach.tex | h(θ) ≤ Γ_s(p) ‖A‖_{1/s} | Direct from R3 + scaling. |
| R6 C_b / B_p | approach.tex + app:models | Single-neuron budget: C_b^{ν/b} = B_p = 2(2pν)^{−ν/2} for product neuron, ≤ n! B_p for power neuron, C_2 ≤ p^{−n}/2 for ReLU | Compare Morwani's single-neuron optimisation (their Lemma for quadratic neuron), Li et al. (n-operand single-neuron). |
| R7 thm:universal | approach.tex Theorem "universal bound" | γ*_b ≤ C_b^{ν/b} Γ_{ν/b}(p), any width, any network, 2≤b≤ν | Core result; check Morwani (b=ν only), Li et al. 2025, any L_{2,b} analysis of max-margin with b≠ν. |
| R8 cor:dense | approach.tex Corollary | Γ_1(p)=p/(p−1), unique flat maximizer; recovers Morwani's max L_{2,3} margin √(2/27)/(p^{1/2}(p−1)) exactly | Compare the exact constant with Morwani Theorem (quadratic, L_{2,3}); and Li et al. general-ν dense endpoint. |
| R9 def:if | cyclic.tex Definition "interference-free network" | β_{j,T}=0 for all nonempty T | Definition; check related notions ("clean"/ "no cross-terms"). |
| R10 lem:sign | cyclic.tex + app:models | Sign rule: β_{ij,T}=β_{ij}Π_{t∈T} ξ_{itj}, unit ξ | Simple; check. |
| R11 eq:JF | cyclic.tex | Interference-free system at j is finite, p-free | — |
| R12 lem:cost | cyclic.tex + app:cost | Under (JF), Σ|β_l|^r ≥ ν^{1−r}|β|^r, equality iff exactly ν equal-modulus terms with orthogonal chain vectors | Key lemma; check against literature on sparse representations of tensors/ multilinear cancellation, Waring/rank lower bounds, "ν-th roots of unity" decompositions. |
| R13 lem:chain | app:cost | H = Σ β_l m_l m_l^* upper-triangular with diagonal β | Our construction. |
| R14 lem:minor | app:cost | det(I+tH') = Σ_L t^{|L|} Π d_l det Gram | Standard (Cauchy–Binet / Sylvester); cite a textbook. |
| R15 lem:hadamard | app:cost | 0 ≤ det Gram ≤ 1 for unit vectors, equality iff orthonormal | Hadamard's inequality; cite. |
| R16 prop:esym | app:cost | e_k(x) ≥ C(ν,k) | Our; check. |
| R17 prop:gen | app:cost | Π(1+x_l t) ≥ (1+t)^ν | Our. |
| R18 lem:bernstein | app:cost | x^r = (r sin πr/π) ∫ log(1+xt) t^{−1−r} dt | Classical (Bernstein function / Stieltjes / Beta integral); cite (e.g. Schilling–Song–Vondraček, Bernstein Functions). |
| R19 remark other penalties | app:cost | Lemma extends to ψ = cx + ∫ log(1+xt) μ(dt) (Bernstein-type) | Check known "Bernstein functions" majorization results. |
| R20 thm:cyclic | cyclic.tex Theorem | γ*_{b,IF} = ν^{−(ν−b)/b} B_p Γ_{ν/b}(p); at b<ν optimum uses |spt y*| frequencies, each on exactly ν neurons, equal modulus, orthogonal chain vectors | Main theorem; check Morwani, Li et al., Gromov constructions, any paper pricing sparsity by (ν−b). |
| R21 prop:block | cyclic.tex + app:block | ν-block: phases α_l = πl/ν cancel sign-flipped copies; table ν λ^ν 2^{−n} cos(...) | Construction; compare Morwani's construction, Gromov's analytic solution, Li et al. n-operand construction, trig identity for products of shifted cosines. |
| R22 prop:shapes | cyclic.tex + app:shape | Maximizers are vertices of polyhedron P_p minimizing Σ v^{1/s}; finitely many switches in s | Uses concave minimisation over polyhedra (vertex optimality) + exponential-sum zero count (Pólya–Szegő / Descartes rule for exponential sums). |
| R23 prop:bottleneck | cyclic.tex + app:shape | ≥K tight offsets with independent coverage rows; y unique solution; 2K hardest wrong answers | Our; relation to LP basic solutions / equioscillation (Chebyshev alternation). |
| R24 prop:ablation | cyclic.tex + app:shape | m − 2y_j ≤ m_{−j} ≤ (1−w_j)^s m/(1−ε); loss fraction in [w_j, s w_j] | Our; check against feature-ablation literature (Nanda ablations). |
| R25 thm:scaling | cyclic.tex + app:scaling | ½⌈8 ln p⌉^{1−s} ≤ Γ_s(p) ≤ 3/2 (2a_s)^s (ln p)^{1−s}; K* between c_s ln p and 9·4^{1/(s−1)} ln p; γ*_b = Θ(p^{−ν/2}(ln p)^{1−ν/b}) over all networks | Central; check Alon–Roichman, minimal-cosine-sum (Chowla cosine problem, Bourgain, Ruzsa, Sanders), "ε-biased sets" / small-bias spaces (log p size), coding sidelobes; also McCracken log-n observation. |
| R26 cor:few | cyclic.tex + app:scaling | Networks with K = o(log p) features attain vanishing fraction of γ*_b | Our; related to lower bounds on size of sets with small Fourier bias / minimal size of expander Cayley generating sets in Z_p. |
| R27 dense-window claim | cyclic.tex last para | Optimum dense only if s−1 = O(1/log p); exact dense thresholds s=1.548…1.160 for p=7..23 | Our (numerical). |
| R28 numeric shape facts | cyclic.tex | p=13 switches at s=1.285/1.407/1.678 (6,4,3,2 freq); p=23 eight switches; #tight offsets = K at p≤23; ablation ratios 1.15–1.72 etc. | Internal numerics; flag for reproducibility, not literature. |
| R29 tab:kstar | cyclic.tex Table | K*(ν/b,p) at b=2: p=59/71/113 → (5,4,3)/(5,4,3)/(7,4,4) | Compare with Nanda (5 key freqs at p=113 transformer), Chughtai, McCracken counts. |
| R30 dense-vs-IF ratios | cyclic.tex "How many features" | Dense network reaches 0.52–0.63 / 0.15–0.25 / 0.04–0.08 of IF optimum at ν=3/4/5 | Internal. |
| R31 groups claim | groups.tex | Irrep analogue: ceiling priced by matrix-multiplication tensor decomposition cost (schoolbook at even degree, d^ν neurons); attains one irrep on S5, Galois pair on A5; structured interference beats ceiling on D3, S4, Z12 | Check Chughtai (irreps), Morwani group result (all irreps at L_{2,ν}), Stander et al. (S5 grokking), tensor rank of matrix multiplication (Strassen, Bläser surveys). No proofs given in appendix yet. |
| R32 ReLU endpoint | app:models "ReLU neuron" | Weight decay on ReLU (ν=2) is the b=ν endpoint; bound's maximizer flat | Check Morwani / MAIS-O52 / Gromov; McCracken ReLU dense observations. |
| R33 shared-linear-layer exclusion | app:models | Shared first layer priced by nuclear norm; outside scope | Check low-rank / nuclear-norm implicit bias literature (e.g. Gunasekar et al. 2017; Ongie–Willett linear layers). |
| R34 conjecture + known results | app:conjecture | Conjecture: IF value is global max for product nets. Known: true at p=3; window ν−4/(3(p−1)) ≤ b ≤ ν; b≥5/2 at p=5; single-frequency class at n=2,b=2 above "aiming depth" 0.3476; strict local max; interfering local max at p=5 (0.837); false on Z12/D3/S4 (1.06–1.62×) | Internal; note "certified aiming depth" undefined; proofs missing. |
| R35 power-net attainment facts | app:conjecture | Smallest IF power population n=2 uses 6 neurons, 0.408 of universal bound; tied embeddings admit none | Internal. |
| R36 b outside [2,ν] | app:conjecture | b>ν: normalized margin unbounded in width; b<2: superadditivity fails by |F|^{1−b/2} | Check known statements about L_{2,b} norms beyond homogeneity (Morwani notes, Savarese et al.). |
| R37 equal-share approximation | app:remarks | Γ_s ≥ max_K K^{−s} M_K; M_K ≤ pK/(p−1); argmax agrees except (59,3),(113,3) | Internal. |
| R38 weaker cancellation | app:remarks | Only interval patterns T needed (C(ν,2) of them) | Internal. |
| R39 other networks factor | app:remarks | Factor ν^{−(ν−b)/b} becomes (n+1)^{−(ν−b)/b}: counts factors not degree | Internal. |

## B. Our techniques that may be known

| id | location | technique | what to check |
|---|---|---|---|
| T1 | approach.tex, thm:universal proof | Subadditivity of t^r to show splitting a feature over neurons never makes it cheaper | Standard; appears in ℓ_p-quasi-norm sparsity arguments; possibly implicit in Morwani/Wei. |
| T2 | approach.tex para after programme; app:models | AM–GM on ν factors ⇒ amplitude A costs A^{b/ν} (effective ℓ_{2/ν}/ℓ_{b/ν} quasi-norm penalty) | Known "weight decay on depth-L / degree-ν factorisations induces ℓ_{2/L} quasi-norm" (Tibshirani/ Dai et al. "representation cost", Gunasekar implicit bias linear convnets, Ergen–Pilanci, Woodworth et al. diagonal nets, Jacobs–Montúfar "exponent decides sparse/dense"). HIGH novelty risk for framing. |
| T3 | app:cost | Determinant/Gram/Hadamard argument: det(I+tH)=(1+t)^ν, minor expansion, elementary symmetric bounds, Bernstein integral to pass from generating inequality to power sums | Check majorization literature: Π(1+x_l t) ≥ (1+t)^ν ⇒ Σ x^r ≥ ν is a known "log-majorization / Bernstein function" transfer (e.g. Bhatia Matrix Analysis; weak majorization of elementary symmetric functions). |
| T4 | prop:block, remark | ν-th roots of unity phase blocks cancel all sign-flipped product-to-sum terms | Check Gromov 2023 analytic solution, Morwani construction, Li et al. 2025 (n-operand); discrete Fourier cancellation / "phase rotation" trick; polyphase codes. |
| T5 | prop:shapes (i) | Concave minimisation over a polyhedron attained at a vertex; strict concavity ⇒ only vertices | Classical (Bauer maximum principle; Rockafellar Convex Analysis Cor. 32.3.x; Horst–Tuy global optimization). Cite. |
| T6 | prop:shapes (ii) | Real exponential sum Σ c_k e^{λ_k u} has ≤ N−1 zeros (Rolle induction) | Classical (Pólya–Szegő Problems, Part V; generalized Descartes rule). Cite. |
| T7 | prop:bottleneck | Vertex ⇒ active constraints full rank ⇒ ≥K tight offsets | Basic LP; minimax/ Chebyshev alternation analogue; Haar condition. |
| T8 | app:scaling lower bound | Random frequency set of size ⌈8 ln p⌉ + Hoeffding + union bound (random Cayley graph / Alon–Roichman) | Cited to Alon–Roichman & Hoeffding; also small-bias sets (Naor–Naor; Alon–Goldreich–Håstad–Peralta), ε-biased sets in Z_p. |
| T9 | app:scaling upper bound | "Positive-Fourier" inequality: E_δ e^{−tF_y(δ)} ≥ Π e^{−ty_j} I_0(ty_j) because e^{z cos} has nonnegative Fourier (Bessel) coefficients | Check Chowla cosine problem literature (Bourgain 1986, Ruzsa 2004, Sanders), minimum of cosine sums; Bessel generating function; Uchiyama; Montgomery "Ten lectures". HIGH overlap risk: lower bounds on log p-type minima of cosine polynomials with positive coefficients. |
| T10 | app:scaling | Scalar bound ψ(z)=z−ln I_0(z) ≤ a_s ln(1+z^r), Laplace-type lower bound on I_0 | Standard Bessel asymptotics (Abramowitz–Stegun / DLMF 10.30). |
| T11 | app:shape ablation | Rescaling after removal + Bernoulli inequality | Elementary. |
| T12 | app:dense | Uniqueness of flat maximizer via Fourier inversion of constant coverage | Elementary; compare Morwani's dense proof. |
| T13 | app:models ReLU | Cauchy–Schwarz + Parseval + Lipschitz variance bound for ReLU budget | Elementary; check Morwani's treatment of ReLU / Gromov. |
| T14 | app:cost remark | Extending to Bernstein-function penalties | See T3. |

## C. Framing claims

| id | location | claim | what to check |
|---|---|---|---|
| F1 | abstract s1 | "Explaining the number of features in a network has been a long-desired result in mechanistic interpretability" | Needs citation or softening (Elhage superposition; Olah circuits; Sharkey et al. open problems 2025). |
| F2 | abstract s2 / intro para 1 | Morwani "laid the baseline ... showing provably that in networks where L_{2,ν} matches homogeneity, all features would be used for modular addition or group composition" | Verify Morwani states "all frequencies"/ "all irreps" at L_{2,ν} for both tasks and the exact hypotheses (quadratic net, ν=3; group: sum of irreps). |
| F3 | abstract s3 / intro para 2 | "networks trained with ordinary weight decay consistently show a sparse selection of features" | Nanda (5 frequencies), Chughtai, Zhong et al. "Clock and Pizza", Stander et al., Gromov, Liu et al. Omnigrok; "consistently" may be overclaim (McCracken dense). |
| F4 | abstract / intro | "our theory explains the transition from dense to sparse ... as seen empirically in McCracken 2025 ... arises on changing either the homogeneity degree or regularization" | Verify McCracken shows a dense→sparse transition with degree/ what exactly they vary. |
| F5 | abstract final | "normalized margin peaks at the predicted optimum ... matching the result of Morwani for their specific case"; "optimums consistent with Chughtai" | Empirical claim, section empty; verify Chughtai claims about number of irreps. |
| F6 | intro para 1 | "Mechanistic interpretability finds circuits ... but typically leaves open why they arise" | Needs cite (Olah 2020; Elhage 2021) or soften. |
| F7 | intro para 1 | Morwani: "with vanishing regularization networks approach the maximum margin solution [Wei 2019], which can be characterized analytically"; "every maximum L_{2,3}-margin network uses Fourier features, one frequency per neuron, and every frequency is used" | Verify Morwani Theorem (quadratic modular addition) statement exactly. |
| F8 | intro para 2 | "the count moves from dense to sparse as the degree of the network changes [McCracken]" | Verify. |
| F9 | intro para 2 | "The dense theorem is stated for the L_{2,ν} norm, not for weight decay" | Verify Morwani's discussion; Morwani may discuss L_2 empirically. |
| F10 | intro "Where the dense theorem stops" | Morwani find best individual neurons then combine; combining free exactly at L_{2,ν}; under L_{2,b}, b<ν, amplitude cheaper when concentrated | Verify Morwani's proof structure; check if they or others remark on b<ν. |
| F11 | intro informal thm bullet 1 | "At b=ν this recovers the maximum margin of Morwani exactly" | Check constant. |
| F12 | informal thm bullet 3 | K* "increases as Θ(log p)" | McCracken's log-n observation (per memory) — check exact overlap. |
| F13 | groups.tex | "one representation on S5 and the Galois-conjugate pair on A5, as observed" | Needs citation (Chughtai; Stander et al. 2024 "Grokking group multiplication with cosets"). Currently uncited. |
| F14 | approach.tex | "so we study maximum margin solutions as a proxy for them. Nothing here concerns whether gradient descent finds them" | Scope; cite Lyu–Li 2020, Ji–Telgarsky 2020, Chizat–Bach 2020 for GD→KKT of margin. |
| F15 | approach.tex "interference cannot increase the margin: it averages away" | framing | Internal. |
| F16 | cyclic.tex "How many features" | "trained networks do carry interference, which Section 5 measures"; appendix scope (ii) "trained networks ... carry interference at the sign-flipped modes" | Empirical, section empty. |
| F17 | cyclic.tex "The theorem also shows how narrow the dense regime is" | Dense regime shrinks as p grows | Internal; relates to McCracken. |
| F18 | app:conjecture | "Any proof must be global and specific to prime modulus" | Internal. |
| F19 | paper.tex comment | theorem environments "as in Morwani et al., 2024" | Comment only; ignore. |
| F20 | intro TODO | Related work to be added: Gromov 2023, Li et al. 2025, McCracken 2025, Lyu & Li 2020 | Missing related-work section; must position against these plus Jacobs–Montúfar, Ding width law, Tian 2024 (Composing global optimizers / algebraic), Doshi et al., Zhong et al. |

## D. Cited claims (grouped by source + point)

| id | source | sentences | what to check |
|---|---|---|---|
| C1 | morwani2024feature | abstract: "laid the baseline with their principle of margin-maximization, showing provably ... all features would be used for modular addition or group composition" | See F2. |
| C2 | morwani2024feature | abstract: "normalized margin ... peaks at the predicted optimum ..., matching the result of Morwani for their specific case" | See F5. |
| C3 | morwani2024feature + wei2019regularization | intro: "Morwani showed that for algebraic tasks the answer follows from one principle: with vanishing regularization, networks approach the maximum margin solution (Wei 2019), which can be characterized analytically. For quadratic networks ... every max L_{2,3}-margin network uses Fourier features, one frequency per neuron, every frequency used" | Morwani main theorem statements; "one frequency per neuron". |
| C4 | nanda2023progress | intro: "With ordinary weight decay they use a handful of frequencies" | Nanda: 5 key frequencies (transformer, p=113). Note Nanda's model is a transformer with AdamW; fine. |
| C5 | chughtai2023toy | intro: "or a few irreducible representations"; abstract: "optimums consistent with the empirical results in Chughtai" | Chughtai: networks learn a few irreps; "key representations" list. |
| C6 | mccracken2025 | abstract + intro: dense-to-sparse transition as degree changes | Verify; also bib author list. |
| C7 | morwani2024feature | intro: "Morwani find the best individual neurons and then combine them" (uncited in sentence but attributed) + informal thm "recovers the maximum margin of Morwani exactly" | Proof structure; exact constant. |
| C8 | morwani2024feature | prelim: "b=ν is the norm of Morwani"; "at n=2 the latter is the quadratic neuron of Morwani" | Morwani's quadratic neuron form (u(a)+v(b))^2 w and L_{2,3}. |
| C9 | morwani2024feature | approach: "As in Morwani, margin definitions g, h, γ*" | Definitions match. |
| C10 | wei2019regularization [Theorem 4.1] | approach: "global minimizers of CE + λ‖θ‖_{2,b}^b attain normalized margin tending to γ*_b as λ→0" | Verify Wei Thm 4.1 hypotheses: homogeneous nets, any norm? (memory: global minimiser, λ→0, any norm). Check it covers L_{2,b} and multiclass CE. |
| C11 | morwani2024feature [Section 3] | approach: "minimum margin is at most any averaged class-weighted margin" | Section number correct? |
| C12 | morwani2024feature | approach: "network-level analogue of η_{u,v,w}(δ) of Morwani" | Notation exists in Morwani. |
| C13 | nanda2023progress, morwani2024feature | approach: "a frequency with A_j>0 is a feature, as in" | Both use "key frequencies" as features. |
| C14 | morwani2024feature [Section 4] | approach: "analogue of the phase condition" | Section number correct. |
| C15 | morwani2024feature | approach: "pricing amplitudes in norm is, as in Morwani, a single-neuron optimization" | Their single-neuron lemma. |
| C16 | morwani2024feature | approach cor:dense + app:dense: "bound is 2B_p p/(p−1) = √(2/27)/(p^{1/2}(p−1)), exactly their max L_{2,3}-margin" | Verify constant in Morwani Theorem (they state γ* = √(2/27) · 1/(p^{1/2}(p−1))?). |
| C17 | morwani2024feature | app:models: "at n=2,b=3 the bound is attained by the network of Morwani, so C_3=2B_p" | Their construction attains. |
| C18 | hoeffding1963probability | app:scaling: Hoeffding for variables in [−1,1] | Standard; check bib details. |
| C19 | alon1994random | app:scaling: "This is the cyclic case of the random Cayley graph argument of Alon–Roichman" | Verify Alon–Roichman Thm: O(log|G|) random elements give expander; bib details (RSA 5(2):271–284, 1994). |

## E. Uncited background (should carry a citation)

| id | location | statement | suggested |
|---|---|---|---|
| U1 | approach.tex / app:models | AM–GM on factors ⇒ amplitude cost A^{b/ν}; L_{2,b} weight decay on products induces a quasi-norm (ℓ_{2/ν}) sparsity penalty | Representation-cost literature: Dai–Karzand–Srebro 2021; Gunasekar et al. 2018 (linear convnets); Woodworth et al. 2020; Jacobs & Montúfar; Tibshirani 2021 "Equivalences between sparse models and neural networks". |
| U2 | abstract/intro | Mech interp motivation; circuits | Olah et al. 2020; Elhage et al. 2021/2022. |
| U3 | intro | Grokking / modular addition as the test bed | Power et al. 2022 (in bib, uncited); Gromov 2023 (in bib, uncited); Liu et al. 2022; Zhong et al. 2023 "Clock and Pizza". |
| U4 | approach.tex "Nothing here concerns whether GD finds them" | Implicit bias of GD toward max margin for homogeneous nets | Lyu & Li 2020 (in bib, uncited); Ji & Telgarsky 2020; Chizat & Bach 2020; Soudry et al. 2018. |
| U5 | prelim | n-operand modular addition, power neuron (u_1+…+u_n)^n w | Li et al. 2025 (in bib, uncited) own the x^n / n-operand family and general-ν dense endpoint. HIGH priority. |
| U6 | app:cost | Sylvester determinant identity; principal-minor expansion; Hadamard's inequality | Horn & Johnson, Matrix Analysis. |
| U7 | app:cost Bernstein integral | x^r integral representation; Beta integral π/sin(πa) | Schilling–Song–Vondraček "Bernstein Functions"; DLMF 5.12. |
| U8 | app:shape | Concave function over polyhedron minimised at vertex; Rolle zero count for exponential sums | Rockafellar 1970 Thm 32.2/32.3; Pólya–Szegő Vol II Part V Problem 75. |
| U9 | app:scaling | Bessel I_k nonnegative Fourier coefficients of e^{z cos θ} (Jacobi–Anger) | DLMF 10.35. |
| U10 | app:scaling | Minimum of cosine sums over Z_p / small-bias sets of size O(log p) | Chowla cosine problem (Bourgain; Ruzsa; Sanders); ε-biased sets (Naor–Naor 1993; AGHP 1992); Kővári/ lower bounds on size of ε-biased sets (Alon 2009 "Perturbed identity matrices"). |
| U11 | cyclic.tex prop:bottleneck | Equioscillation / active constraints of minimax problems | Chebyshev alternation / LP duality (Cheney approximation theory). |
| U12 | groups.tex | Irreps as features in group composition; matrix-multiplication tensor rank / schoolbook | Chughtai; Stander et al. 2024; Strassen 1969; Bläser 2013 survey. |
| U13 | app:models "shared linear layer priced by nuclear norm" | Nuclear-norm representation cost of linear layers | Srebro et al. 2005; Gunasekar et al. 2017; Ongie & Willett 2022. |
| U14 | prop:ablation / abstract | Feature-ablation experiments in mech interp | Nanda 2023 ablations; Chughtai. |
| U15 | cyclic.tex product-to-sum sign-flipped copies | Trig product-to-sum identity (trivial) and prior observation that products of cosines create cos(a−b) interference | Nanda 2023; Gromov 2023; Morwani note on this. |
| U16 | Table / "count grows slowly with p" | Empirical counts of frequencies vs p in trained nets | McCracken 2025; Doshi et al. 2023/24; Furuta et al. |

## Adversarial priority list (highest novelty risk first)
1. T2/U1: AM–GM ⇒ quasi-norm sparsity from weight decay on degree-ν products (very well known in representation-cost literature). Our novelty must be the coverage programme and the exact ν-neuron cancellation price, not the quasi-norm.
2. R25/T9/U10: logarithmic scaling of Γ_s(p). Look hard in Chowla cosine problem, small-bias sets, sidelobe design, McCracken log-n.
3. R8/F12/U5: Li et al. 2025 (n-operand, general ν dense endpoint) and McCracken 2025 (dense→sparse transition). Uncited in text despite being in bib.
4. R12/T3: ν-neuron cost lemma via determinant; check tensor/ Waring rank of cos(sum) products, "phase retrieval / cancellation" literature, and majorization via Bernstein functions.
5. R21/T4: ν-th root phase block; check Gromov's analytic solution and Li et al. constructions.
6. R7: universal bound for b<ν; search for anyone analyzing L_2 (weight decay) max-margin on modular addition (e.g. "Morwani sequel", Tian 2024 "Composing global optimizers in reasoning tasks via algebraic objects", Ding width law, AdamW ℓ_∞ margin papers).
7. R22/T5,T6: classical; cite textbooks.
8. R31: group results, compare Chughtai, Morwani Sec. 6, Stander, tensor rank.
