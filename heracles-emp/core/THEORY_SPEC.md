# THEORY_SPEC — every formula the code uses, with its paper reference

Paper sources (read for definitions only): `heracles/paper/iclr2027/sections/{prelim,approach,cyclic,appendix}.tex`.
References below are `file:label`. The paper writes the penalty exponent as **b**; this package writes it
as **q** (same object). Everything here was re-derived and checked numerically by `tests/` (see core/TESTS.txt).

## 1. Task, models, norms

| Object | Formula | Paper | Code |
|---|---|---|---|
| Task | all p^n inputs x = (a_1..a_n) ∈ Z_p^n, label y(x) = (a_1+…+a_n) mod p, p odd prime | prelim.tex:sec:prelim ("Task and neurons") | `models.labels`, `models.inputs` |
| Product network | f(x)[c] = Σ_i Π_t U[t,i,a_t] · W[i,c], ν = n+1 | prelim.tex ("product neuron") | `models.product_logits` |
| Power network | f(x)[c] = Σ_i (Σ_t U[t,i,a_t])^n · W[i,c], ν = n+1 | prelim.tex ("power neuron") | `models.power_logits` |
| Neuron norm | ‖ω_i‖² = Σ_t ‖U[t,i,:]‖² + ‖W[i,:]‖² | prelim.tex (L_{2,b} paragraph) | `models.neuron_sqnorms` |
| L_{2,q} norm | ‖θ‖_{2,q} = (Σ_i ‖ω_i‖^q)^{1/q}, 2 ≤ q ≤ ν, s = ν/q ∈ [1, ν/2] | prelim.tex; approach.tex:sec:programme | `models.l2q_norm`, `measure.l2q_norm` |
| Margin | h(θ) = min_x min_{c≠y(x)} f(x)[y(x)] − f(x)[c] | approach.tex ("Maximum margin") | `measure.margin`, `models.margins_from_logits` |
| Normalised margin | γ = h(θ)/‖θ‖_{2,q}^ν (scale invariant: θ→cθ multiplies h by c^ν) | approach.tex ("Maximum margin") | `measure.normalized_margin`; test `test_gamma_scale_invariance` |
| Training objective | mean CE over D_p + λ Σ_i ‖ω_i‖^q (= λ‖θ‖_{2,q}^q); Wei et al. Thm 4.1: global minimisers' γ → γ*_q as λ → 0 | approach.tex ("Maximum margin") | `train.py`; CE in the algebraically identical margin form `models.ce_margin_form` |

## 2. Fourier conventions and the clean spectrum

- v̂(j) = Σ_k v(k) ρ^{−jk}, ρ = e^{2πi/p} — prelim.tex ("Fourier transform"). This is numpy's `fft` sign
  convention. Folded frequencies F = {1..h}, h = (p−1)/2.
- Multidimensional transform f̂(k_1..k_n; l) = Σ_{x,c} f(x)[c] ρ^{−(k·a + l c)} — prelim.tex; `np.fft.fftn` of the
  table T[a_1..a_n, c].
- **Clean amplitude**: β_j = 2 p^{−ν} f̂(j,…,j; −j), A_j = |β_j| — approach.tex:eq:spectrum, derived in
  appendix.tex:app:spectrum. Then η(δ) := E_x f(x)[y(x)−δ] = η̄ + Σ_j Re(β_j ρ^{jδ}), so β_j is the complex
  coefficient of cos(2πj(Σa−c)/p) (a table C·cos(2πj(Σa−c)/p + φ) has β_j = C e^{iφ}).
  *Normalisation check* (`test_fft_normalisation_and_modes`, `test_nu_block_attains_class_optimum`): the ν-block
  table ν λ^ν 2^{−n} cos(2πj(Σa−c)/p) with λ_j^ν = 2^n y_j/ν measures back β_j = y_j to 1e-12.
- **Junk modes**: β_{j,T} = 2 p^{−ν} f̂(k_1..k_n; −j), k_t = −j for t ∈ T, k_t = j otherwise, T ⊆ [n] non-empty
  (2^n − 1 modes per j) — cyclic.tex (paragraph before def:if). `measure.spectrum()['junk']`, keyed by the sign
  vector ε (ε_t = −1 iff t ∈ T).
- **Energies** (Parseval, Σ|f̂|² = p^ν Σ f²; each mode counted with its conjugate): total = Σ f²;
  clean = Σ_j 2|f̂(j..j;−j)|²/p^ν; junk = Σ_j Σ_{T≠∅} 2|f̂(εj;−j)|²/p^ν; diag = clean + junk (all modes
  (±j,…,±j;±j)); zero_mode = |f̂(0)|²/p^ν; off_diag = total − diag − zero_mode (cross-frequency modes and modes with a
  zero slot). Package definition (not in the paper), `measure.spectrum`.
- **Per-neuron coefficient** (product): β_ij = 2p^{−ν} Π_t û_{it}(j) · conj(ŵ_i(j)) — approach.tex (after eq:spectrum),
  appendix.tex:app:models. Power neuron: the same times n! — appendix.tex:app:models ("Power neuron").
  `measure.neuron_fourier`.
- **Coverage**: cov_j(δ) = 1 − cos(2πjδ/p); Σ_{δ≠0} cov_j(δ) = p — approach.tex (before lem:phase).
- **Averaged margin / phase lemma**: h(θ) ≤ η(0) − η(δ) (approach.tex:lem:avg) and
  h(θ) ≤ Σ_j Re(β_j) cov_j(δ) ≤ Σ_j A_j cov_j(δ) for every δ ≠ 0 (approach.tex:lem:phase; identity checked in
  `test_fft_normalisation_and_modes`). `measure.clean_coverage_margin` reports both right-hand sides (min over δ).

## 3. The share programme

- Γ_s(p) = max { min_{δ≠0} Σ_{j∈F} y_j cov_j(δ) : y ≥ 0, Σ_j y_j^{1/s} ≤ 1 } — approach.tex:eq:programme.
  It depends on (s, p) only, **not on n** (confirmed: eq:programme involves neither n nor ν except through s;
  thm:universal and thm:cyclic both use Γ_{ν/b}(p)). K*(s,p) = |spt y*|. Γ_s(S): the same with spt y ⊆ S.
- Offsets: cov is even in δ, so δ and p−δ give the same constraint; the code uses δ = 1..h.
- **Exact solution** (`programme.py`): with A[δ,j] = cov_j(δ) (h×h) and r = 1/s,
  C_s = min{Σ v^r : v ≥ 0, Av ≥ 1}, Γ_s = C_s^{−s}, y* = v*/C_s^s. For s ≥ 1 the cost is concave and non-decreasing
  on the recession cone {d ≥ 0}, so the minimum is at a vertex of P = {v ≥ 0, Av ≥ 1}; P does not depend on s
  (CONSEQUENCES §1). For s > 1 the cost is strictly concave along every segment, so **every** maximiser is a
  vertex; at s = 1 the maximiser is unique (approach.tex:cor:dense). `solve()` therefore lists all maximisers.
- **Dilation**: cov_{fold(uj)}(δ) = cov_j(uδ), so j → fold(uj) (u a unit mod p) maps P to itself and preserves the
  cost; the units mod ±1 act simply transitively on F (CONSEQUENCES §2). Used to enumerate only canonical
  supports, and in alignment of measured spectra.
- **Dense endpoint**: Γ_1(p) = p/(p−1), unique maximiser y_j = 2/(p−1) — approach.tex:cor:dense, appendix.tex:app:dense.
- **Certified support cap** (CONSEQUENCES §4, astra deletion argument; re-derived): for any maximiser (margin m,
  budget shares w_j = y_j^{1/s}), removing j and rescaling gives 2 w_j^s ≥ m(1 − (1−w_j)^s). Hence
  w_j ≥ (m/2)^{1/(s−1)} and K ≤ (2/m)^{1/(s−1)} ('paper' cap), and more sharply w_j ≥ w0 := first positive root of
  2w^s − m(1−(1−w)^s), K ≤ 1/w0 ('sharp' cap). Monotone in m, so any m_lb ≤ Γ_s(p) gives valid caps. `programme.kcap`.
- **Ablation bound** (CONSEQUENCES §3): for an optimal allocation, deleting j leaves margin ≤ Γ(1−w_j)^s, i.e.
  relative loss ≥ 1 − (1−w_j)^s. Checked on constructed networks in `test_ablation_on_constructed_network`.

## 4. Ceilings (re-derived)

- B_p = 2(2pν)^{−ν/2}. Product neuron: C_b^{ν/b} = B_p for every 2 ≤ b ≤ ν — approach.tex (after the definition of
  C_b), appendix.tex:app:models. Re-derivation: a balanced neuron with all ν vectors λcos(2πja/p+φ) has
  |û| = λp/2, so |β_j| = 2p^{−ν}(λp/2)^ν = 2^{1−ν}λ^ν and ‖ω‖² = νλ²p/2, giving |β_j|/‖ω‖^ν = 2(2νp)^{−ν/2} = B_p.
  Numerically: random neurons never exceed it and the balanced neuron attains it (`test_single_neuron_budget_bound`,
  `test_nu_block_attains_class_optimum`).
- **Universal ceiling, product networks**: γ*_q ≤ B_p Γ_s(p) — approach.tex:thm:universal.
- **Universal ceiling, power networks**: γ*_q ≤ n! B_p Γ_s(p) — approach.tex:thm:universal with
  C_b^{ν/b} ≤ n! B_p (appendix.tex:app:models, "Power neuron").
- **Interference-free class optimum, product networks**: γ_IF = ν^{−(ν−q)/q} B_p Γ_s(p), attained by ν-blocks with
  width ≥ ν K* — cyclic.tex:thm:cyclic, cyclic.tex:prop:block, appendix.tex:app:block. Factor ν^{(ν−q)/q} = √3, 4, 5^{3/2}
  at ν = 3, 4, 5 under q = 2; 1 at q = ν. Checked to < 1e-10 relative for p ∈ {5,7,11,13}, n ∈ {2,3,4},
  q ∈ {2, (2+ν)/2, ν}, every optimal vertex (`test_nu_block_attains_class_optimum`).
- **ν-block** (cyclic.tex:prop:block): for j ∈ spt y, λ_j = (2^n y_j/ν)^{1/ν}, α_l = πl/ν (l = 0..n):
  u_t^{(l)}(a) = λ_j cos(2πja/p + α_l), w^{(l)}(c) = λ_j cos(2πjc/p + nα_l). Table = Σ_j y_j cos(2πj(Σa−c)/p).
  `construct.nu_block`.
- **Morwani dense power network**, n = 2, q = 3 (approach.tex:cor:dense: bound 2B_p p/(p−1) = √(2/27)/(p^{1/2}(p−1))).
  Construction used here (re-derived, `construct.morwani_dense`): per j, 9 neurons with phases φ1, φ2 ∈ {0, 2π/3, 4π/3},
  φ3 = φ1+φ2, amplitude λ = (2y_j/9)^{1/3}; every non-clean trigonometric term has phase in
  {φ1+φ2, 3φ1+φ2, φ1−φ2, φ1+3φ2, 2φ1+2φ2, 2φ1, 2φ2}, which sums to zero over the 3×3 grid, so the table is exactly
  Σ_j y_j cos(2πj(a+b−c)/p). With y flat and q = 3 it attains n! B_p Γ_1(p) to < 1e-10
  (`test_morwani_dense_attains_power_ceiling`, also against the closed form).

## 5. Measured quantities (package definitions, `measure.py`)

- Budget shares w_j = A_j^{1/s}/Σ_k A_k^{1/s}; budget-normalised amplitudes ŷ = A/‖A‖_{1/s}, ‖A‖_{1/s} = (ΣA^{1/s})^s
  (approach.tex:prop:coverage: h ≤ Γ_s ‖A‖_{1/s}; at the optimum ŷ = y*).
- Per-offset worst margins h_δ = min_x f(x)[y] − f(x)[y−δ], δ = 1..p−1, and folded min(h_δ, h_{p−δ}).
- Ablation of frequency j: every weight vector v (all operand slots and the readout) is replaced by v − P_j v, P_j the
  orthogonal projection onto span{cos(2πja/p), sin(2πja/p)} (i.e. v̂(±j) := 0). Raw margins after ablation are reported.
- Counts: thresholds A_j ≥ t·max A for t = 1, 5, 10, 20 %; participation ratios (ΣA)²/ΣA² and (ΣA^{1/s})²/ΣA^{2/s};
  budget-mass counts (fewest largest shares reaching 90/95/99 %).
- Numerical note: off-support amplitudes of exact constructions are ~1e−17; raising them to 1/s puts ~1e−8 into
  budget-normalised quantities. Constructed controls therefore match budget-normalised metrics to ~1e−6 while β
  matches to 1e−12.

## 6. Predictions tested by each experiment (from PLAN.md)

| Exp | Prediction (exact statement) | Status in the paper |
|---|---|---|
| E1 | Product networks, q < ν: trained amplitudes satisfy ŷ ≈ y* for some maximiser y* modulo dilation (count K = K*(s,p), budget shares w = y*^{1/s}); γ ≤ B_p Γ_s(p) always, and the ν-block value γ_IF is the interference-free optimum | ceiling and class optimum: theorems (thm:universal, thm:cyclic). That trained (interfering) nets follow y*: a measurement, not a theorem (appendix.tex:app:remarks "Scope" (ii), app:conjecture) |
| E1b | Γ_s depends on (ν, q) only through s = ν/q (eq:programme): cells with equal s have the same prediction | the programme statement is exact; equality of trained behaviour is a measurement |
| E3 | q = ν: y* flat, y_j = 2/(p−1) (all h frequencies, equal amplitudes) — cor:dense. Power n = 2, q = 3: ceiling attained (Morwani). Power n ≥ 3: statement about the bound's maximiser only (NORMS A2) | theorem for the bound's maximiser; attainment only at n = 2 power and product ν-blocks |
| E4 | Clean allocation: deleting j costs ≥ 1 − (1−w_j)^s of margin (CONSEQUENCES §3) | theorem for the clean allocation / constructed nets; measurement on trained nets |
| E5 | At a vertex optimum on K frequencies ≥ K offsets tie at the minimum coverage and their equations fix amplitude ratios (CONSEQUENCES §7) | theorem for the programme; measurement on trained nets |
| E6 | Relabelling tokens x → ux (init and data) relabels learned frequency k → fold(u^{-1}k) exactly in exact arithmetic; with dilation-symmetric init laws P(j active) = E[K]/h (CONSEQUENCES §9) | exact statement about GD in exact arithmetic; float32 drift is measured |
| E7 | Optimal vertex (support and ratios) is piecewise constant in s with switch points from `programme.switch_points` (CONSEQUENCES §1) | exact for the programme; measurement on trained nets |
| E8 | K*(p) grows slowly (CONSEQUENCES §0 claims Θ(log p)); exact K* only where the vertex cache exists (p ≤ 41 at the time of writing), certified-cap exact where the cap enumeration is affordable, otherwise best-found (not certified) | asymptotic claim is not tested by small p; stated honestly |

## 7. Open issues (definitions that needed a choice)

1. **Notation b vs q.** The paper's penalty exponent b is this package's q. Nothing else differs.
2. **"All maximisers have the same support size" is not proved** (approach.tex footnote to eq:programme). The code
   returns every maximiser (all optimal vertices, all dilates) and `kstar()` returns a list when they disagree. So
   far every cached (p, s) we evaluated had a single optimal orbit class, but this is checked per call, not assumed.
3. **Power-network ν-block remark** (appendix.tex:app:models, "Power neuron", with a TODO(user) comment): the paper says
   the ν-block does not attain the power bound because squared monomials put interference at modes with a zero slot.
   We do not use ν-blocks for power networks; the power positive control is the Morwani construction at n = 2, q = 3
   only, which attains n! B_p Γ_1(p) exactly.
4. **Energy split** ("energy outside the diagonal") is a package definition (§2 above); the paper defines only the
   clean and junk modes.
5. **Ablation** is a package definition (§5). For constructed ν-blocks it deletes the block of j exactly; in trained
   networks it also removes junk and cross-frequency content involving ±j.
6. **Programme beyond the exact cache.** The paper's tab:kstar (p = 59, 71, 113) uses partly heuristic support search.
   This package labels every prediction: 'exact (vertex cache)', 'exact (certified K cap)' or
   'best-found (not certified)' (`programme.solve_any`).
7. No inconsistency was found between the paper's formulas and the constructions: every constant (B_p, the ν-block
   table and norm, γ_IF, the power ceiling at n = 2, q = 3, Γ_1 = p/(p−1), the p = 5 example) was confirmed to ≤ 1e−10.
