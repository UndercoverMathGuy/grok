# Paper result → Lean statement map, with fidelity notes

Paper: *Explaining Feature Sparsity Through the Margin–Regularization Trade-off*
(`heracles/paper/iclr2027/sections/{prelim,approach,cyclic,appendix}.tex`, read 2026-09-25).
All Lean names are in namespace `Heracles` (the `lem:cost` chain is in `Heracles.Cost`).
Files are under `heracles/lean/Heracles/`. "`p` an odd prime" is `[Fact p.Prime] (hp : 3 ≤ p)`.
Every theorem named here is FORMALIZED unless stated otherwise (see `STATUS.md`).

## 0. Definitions and how they match the paper (`Defs.lean`, `Network.lean`)

| Paper | Lean | Notes |
|---|---|---|
| `F_p = {1,…,(p−1)/2}` | `Freq p` (subtype of `ℕ`) | exact |
| offsets `δ ≠ 0` in `ℤ_p` | `Off p = {δ : ZMod p // δ ≠ 0}` | exact |
| `c_j(δ) = 1 − cos(2πjδ/p)` | `cov p δ j = 1 − cos(2π·j·δ.val/p)`, `covP p δ j` | `δ.val ∈ [0,p)` represents `δ`; cos is `2π`-periodic |
| `F_y(δ)`, `m(y) = min_{δ≠0} F_y(δ)` | `cover c y δ`, `worst c y` | exact |
| feasible `y ≥ 0`, `∑_j y_j^{1/s} ≤ 1` | `Feasible s y`, `budget s y = ∑ y_j ^ (1/s)` (real power) | exact |
| `Γ_s(p)` (a max) | `GammaP p s := sSup {worst y : Feasible s y}` | `exists_maximizer`: the sup is attained; `isMaximizer_iff`: maximizers = feasible points with `worst = Γ` |
| maximizer | `IsMaximizer c s y`: feasible and `worst y' ≤ worst y` for every feasible `y'` | exact |
| `spt(y)` | `supp y` | exact |
| `\curlyp_p` | `Poly c`, `PolyP p = {v ≥ 0 : 1 ≤ F_v(δ) ∀δ}` | exact |
| vertex of `\curlyp_p` | `IsVertex c v` := extreme point of `Poly c` | standard (for polyhedra vertex = extreme point = basic feasible solution); the rank property used by the paper is proved (`vertex_eq_zero_of_active`) |
| `I_0`, `ψ`, `a_s` | `I0 z = (1/π)∫_0^π e^{z cos θ}dθ`, `psi z = z − log (I0 z)`, `aS s = 3 + s/2` | exact |
| inputs, label, logit table | `Inp p n = Fin n → ZMod p`, `label x = ∑_t x_t`, `Table p n` | exact |
| margin `h(θ)` on `D_p` | `margin f = min_{x, c ≠ y(x)} (f x (y x) − f x c)` | equals `min_x (f[y] − max_{y'≠y} f[y'])` |
| `ρ^m`, `f̂(k;l)`, `β_j`, `A_j`, `η` | `rho`, `fhat`, `beta`, `amp`, `eta` | conventions of `app:spectrum`: `f̂(k;l) = ∑ f·ρ^{−(k·a+lc)}`, `β_j = 2p^{−(n+1)} f̂(j,…,j;−j)` |
| `β_{j,T}`, interference-free (`def:if`) | `betaT`, `InterferenceFree` | exact |
| product neuron, `‖ω‖_2²`, `‖θ‖_{2,b}^b`, `B_p` | `ProdParam`, `prodNeuron`, `normSq`, `normPow`, `Bp` | exact |
| network of width `m` | `θ : Fin m → Ω`, table `netTable φ θ = ∑_i φ(θ_i)` | exact |
| `γ*_b = max_{‖θ‖_{2,b} ≤ 1} h(θ)` (product, width `m`) | `gammaStarW p n m b := sSup {margin θ : normPow b θ ≤ 1}` | defined as a sup (bounded by `gammaStar_le`) |

---

## 1. `prop:ablation` (removing a feature)

**`prop_ablation`**: `p` prime, `s > 1`, `0 ≤ ε < 1`, `y` feasible with `budget s y = 1` and
`(1−ε)Γ_s(p) ≤ worst y`, `j ∈ supp y`. Then
`worst y − 2y_j ≤ worst (update y j 0)`, `worst (update y j 0) ≤ (1−w_j)^s/(1−ε) · worst y`
(`w_j = y_j^{1/s}`), and `w_j ≤ 1 − (1−w_j)^s ≤ s·w_j`.
Also: **`share_bounds`** (`w ≤ 1−(1−w)^s ≤ sw`, `s ≥ 1`, `0 ≤ w ≤ 1`); **`ablation_fraction`**
(for a maximizer with positive margin: `1 − (1−w_j)^s ≤ (m − m_{−j})/m`); **`ablation`** (the two
bounds for any coverage matrix with entries `≤ 2`).
Differences: none of substance. `j ∈ spt(y)` and `ε ≥ 0` are carried but not needed.

## 2. `prop:bottleneck` (bottleneck offsets)

**`prop_bottleneck`**: `p ≥ 3` prime, `s > 1`, `y` a maximizer, `K = |supp y|`. There is an
injective `δs : Fin K → Off p` with `1 ≤ (δs i).val ≤ (p−1)/2`, `F_y(δs i) = Γ_s(p)`, the vectors
`(c_j(δs i))_{j∈supp y}` linearly independent, and every `z` vanishing off `supp y` with
`F_z(δs i) = Γ_s(p)` for all `i` equals `y`.
**`prop_bottleneck_network`**: for the table `∑_j y_j cos(2πj(y(x)−c)/p)` (`blockTable`; this is the
table of the `prop:block` network, `blockNet_table`), on every input the gaps to the candidates
`y(x) ∓ δ_i` equal `Γ_s(p)` and are `≤` the gap to every wrong candidate.
Differences: the network sentence is stated for tables of this form (the paper's network has it
by `prop:block`).

## 3. `thm:scaling` (logarithmic scaling), `s > 1`, `a_s = 3 + s/2`, `p ≥ 3` prime

Programme level:
* **`GammaP_lower`** `½⌈8 ln p⌉^{1−s} ≤ Γ_s(p)`; **`GammaP_lower9`** `½(9 ln p)^{1−s} ≤ Γ_s(p)`;
  **`exists_alloc_lower`**: a feasible `y` with `|supp y| ≤ ⌈8 ln p⌉`, `worst y ≥ ½⌈8 ln p⌉^{1−s}`.
* **`GammaP_upper`** `Γ_s(p) ≤ (3/2)(2a_s)^s (ln p)^{1−s}`.
* **`card_supp_ge`** every maximizer: `c_s ln p ≤ |spt y|`, `c_s = (8a_s²(3·9^{s−1})^{1/s})^{−1}` (`cS`).
* **`card_supp_le`** every maximizer: `|spt y| ≤ 9·4^{1/(s−1)} ln p`.
* **`share_ge`** every maximizer: `y_j^{1/s} ≥ 4^{−1/(s−1)}/(9 ln p)` on `spt y`.

Network level (**`thm_scaling_net`**, Final.lean): product networks, `2 ≤ b < ν = n+1`, `s = ν/b`,
every odd prime `p`, every width `m ≥ ν⌈8 ln p⌉`:
`c·p^{−ν/2}(ln p)^{1−s} ≤ γ*_b ≤ C·p^{−ν/2}(ln p)^{1−s}` with explicit
`c = ν^{1−s}·2(2ν)^{−ν/2}·½·9^{1−s}`, `C = 2(2ν)^{−ν/2}·(3/2)(2a_s)^s`.
Supporting: **`gammaStar_le`** (`γ* ≤ B_pΓ_s`, any width), **`gammaStar_ge`**
(`γ* ≥ ν^{1−s}B_p·½⌈8 ln p⌉^{1−s}` at width `≥ ν⌈8 ln p⌉`), **`le_gammaStar`** (normalization).
Difference: `Θ(·)` is made explicit (constants depending only on `ν, b`); `γ*` is taken at a
fixed width `m ≥ ν⌈8 ln p⌉` (the paper's "over all networks of width at least ν⌈8 ln p⌉").

Lemmas of `app:scaling`:

| Paper step | Lean | Statement |
|---|---|---|
| scalar bound | `psi_nonneg`, `psi_le` | `0 ≤ ψ(z)`, `ψ(z) ≤ a_s ln(1+z^{1/s})` (`z ≥ 0`, `s > 1`) |
| its ingredients | `one_le_I0`, `I0_le_exp`, `I0_ge_of_one_le` | `1 ≤ I_0(z)` (all `z`), `I_0(z) ≤ e^z` (`z ≥ 0`), `I_0(z) ≥ e^{z−1/2}/(π√z)` (`z ≥ 1`) |
| (posfourier) | `posfourier` | `y ≥ 0`, `t ≥ 0`: `exp(−∑_jψ(ty_j)) ≤ (1/p)∑_{δ∈ℤ_p} exp(−tF_y(δ))` |
| (psisum) | `psisum_K`, `psisum` | `≤ K` non-zero entries: `∑ψ(ty_j) ≤ a_s∑ln(1+t^r y_j^r) ≤ a_s K ln(1+t^r/K)`; and `∑ψ(ty_j) ≤ a_s t^r` |
| (twoterms) | `twoterms` | `(1/p)∑_δ e^{−tF_y(δ)} ≤ 1/p + e^{−tm(y)}` |
| choice `t = (ln p/(2a_s))^s` | `worst_le_of_psi`, `GammaP_upper` | if `∑ψ(ty_j) ≤ ½ln p` then `m(y) ≤ (3/2)ln p/t` |
| (Kbound) | `Kbound` | `≤ K` non-zero entries (`K ≥ 1`): `m(y) ≤ (3/2)(ln p)^{1−s}(x/(e^{x/(2a_s)}−1))^s`, `x = ln p/K` |
| `e^z − 1 ≥ z²/2` | `exp_sub_one_ge_sq` | `z ≥ 0` |
| deletion / share floor | `share_ge_gamma`, `card_supp_le_ceil` | `w_j ≥ (Γ/2)^{1/(s−1)}`; `|spt| ≤ 4^{1/(s−1)}⌈8 ln p⌉` |
| random frequencies | `exists_good_draw`, `sum_cos_freq` | `J ∈ F_p^L` (`L ≥ 8 ln p`) with `∑_l c_{J_l}(δ) ≥ L/2` ∀`δ≠0`; `∑_{j∈F_p}cos(2πjδ/p) = −1/2` |
| dense-regime remark | `dense_only_if` | all `(p−1)/2` frequencies used ⇒ `s−1 ≤ ln4/ln((p−1)/(2⌈8 ln p⌉))` (if that ratio is `> 1`) |

Differences: (a) the random-frequency argument is a faithful **counting** version (tuples in
`F_p^L`; probability = count/`|F_p|^L`): Chernoff–Markov at `λ = 1/2`, Hoeffding's lemma in the
convexity form `e^{λx} ≤ cosh λ + x sinh λ`, `cosh λ ≤ e^{λ²/2}`, mean `≤ 0`; no measure theory.
(b) (posfourier) is proved through the finite DFT on `ℤ_p` (every factor
`δ ↦ e^{z cos(2πjδ/p)}` has non-negative DFT coefficients; the average of the product dominates the
product of the constant coefficients; each constant coefficient is `≥ I_0(z)` by comparing moments
with the Wallis integrals). Same statement; this avoids the paper's unjustified rearrangement of
an infinite product of Fourier series. (c) `one_le_I0`, `psi_le_self` hold for all real `z`.

## 4. `cor:few` (few features cannot be optimal)

* **`cor_few`** (generic neurons): any neuron model `φ : Ω → Table p n` with size `N ≥ 0` and budget
  constant `C ≥ 0` (`∑_j|β_j(φω)|^{b/ν} ≤ C N(ω)^b`), `0 < b < ν`, any width, any network whose
  amplitudes are non-zero on `≤ K` frequencies (`K ≥ 1`):
  `h(θ) ≤ C^{ν/b}(∑_i N(θ_i)^b)^{ν/b} · (3/2) g_s(ln p/K)(ln p)^{1−s}`, `g_s(x) = (x/(e^{x/(2a_s)}−1))^s`.
* **`cor_few_prod`**: product networks (`C^{ν/b} = B_p`), `2 ≤ b < ν`.
* **`cor_few_ratio_prod`**: product networks, `‖θ‖ > 0`:
  `h(θ)/‖θ‖_{2,b}^ν ≤ 3(9ν)^{s−1} g_s(ln p/K) · γ*_b(p)` (γ* at any width `≥ ν⌈8 ln p⌉`);
  **`tendsto_gK`**: `g_s(x) → 0` as `x → ∞` (so `K = o(log p)` gives a vanishing fraction).
Differences: the bound is stated multiplied out (`h ≤ …‖θ‖^ν`); `K ≥ 1` explicit. `C_b` may be taken
as the paper's supremum via `budget_le_Cb`.

## 5. `prop:shapes` (finitely many shapes)

* **`prop_shapes_vertex`**: every maximizer is `Γ_s(p)•v`, `v` a vertex of `\curlyp_p` minimizing
  `∑_j v_j^{1/s}` over `\curlyp_p`; **`isMaximizer_iff_minVertex`**: and conversely.
* (i) **`mem_extremePoints_of_isMin`**: a minimizer of a strictly concave function over any set is
  an extreme point. **`minimizer_isVertex`**, **`exists_minimizer`**.
* (ii) **`expsum_card_zeros_le`**: `∑_{k<N} c_k e^{λ_k u}`, distinct `λ_k`, `c ≠ 0`, has at most `N−1`
  real zeros (every finite zero set has card `≤ N−1`); **`expsum_zeros_finite`**.
* (iii) **`finite_vertices`**; **`prop_shapes_switches`**: a finite `Z ⊂ ℝ` such that the set of
  minimizing vertices is the same at any `s, s' > 1` with no point of `Z` in `[s,s']`;
  **`ties_of_multiset_eq`**; **`maximizers_between_switches`** (between switches the maximizers at
  `s'` are exactly `Γ_{s'}/Γ_s` times those at `s`: support and ratios are constant);
  **`prop_shapes_shares`**: `y_j^{1/s} = v_j^{1/s}/∑_k v_k^{1/s}`.
Differences: (i) is stronger (no monotonicity/pointedness needed); existence of a minimizer comes
from compactness of the programme instead of the Minkowski–Weyl decomposition. "Up to ties" is not
needed (tied vertices enter/leave together). The parenthetical "(such as images under
`j ↦ uj mod p`)": **`isVertex_permVec`** (the image `v ∘ π_u^{-1}` of a vertex under the permutation
`π_u(j) = min(uj mod p, p − uj mod p)` of `F_p`, `u ≠ 0`, is a vertex), **`mem_Poly_permVec`**,
**`multiset_permVec`** (same multiset of non-zero entries), **`budget_permVec`** (tied for every
`s`). Not formalized: the phrase "move continuously" (only the explicit share formula).

## 6. Section `sec:approach`

* **`lem_avg`** (`lem:avg`) `h(f) ≤ η(0) − η(δ)`, `δ ≠ 0`, any table.
* **`spectrum`**, **`spectrum_avg`** (eq. (spectrum)): `η(δ) = η̄ + ∑_{j∈F_p}Re(β_jρ^{jδ})`,
  `η(0) − (η(δ)+η(−δ))/2 = ∑_j Re(β_j)c_j(δ)`; `η̄ = p^{−(n+1)}Re f̂(0;0)` (`f̂(0;0)` is real).
* **`lem_phase`** (`lem:phase`) `h(f) ≤ ∑_j Re(β_j)c_j(δ) ≤ ∑_j A_j c_j(δ)`.
* **`prop_coverage`** (`prop:coverage`) `h(f) ≤ Γ_s(p)(∑_jA_j^{1/s})^s`, `s ≥ 1` (also `A = 0`).
* **`thm_universal`** (`thm:universal`, generic): budget constant `C`, `0 < b ≤ ν`:
  `h(θ) ≤ C^{ν/b}Γ_{ν/b}(p)(∑_iN(θ_i)^b)^{ν/b}`, any width. **`thm_universal_unit`**: max form over
  `‖θ‖ ≤ 1`. **`thm_universal_Cb`**: with the paper's `C_b = sup_{N(ω)≤1}∑_j|β_j|^{b/ν}` (`Cb`), under
  explicit homogeneity hypotheses (a rescaling `sc λ` with `φ(sc λ ω) = λ^νφ(ω)`, `N(sc λ ω) = λN(ω)`),
  boundedness of the defining set and `Nonempty Ω`. **`thm_universal_prod`**: product networks,
  `h(θ) ≤ B_pΓ_{ν/b}(p)‖θ‖_{2,b}^ν`, `2 ≤ b ≤ ν`.
  Differences: generic model ("any one-hidden-layer network with homogeneity ν"); `2 ≤ b` weakened
  to `0 < b` in the generic form; stated per network.
* `cor:dense`: **`GammaP_one`** `Γ_1(p) = p/(p−1)`; **`dense_maximizer_iff`** unique maximizer
  `y_j = 2/(p−1)`; **`dense_network_equal_amp`**: at `b = ν`, a network attaining `thm:universal`
  with positive margin has `A_j = (∑_kA_k)·2/(p−1)` for every `j`; **`dense_constant`**:
  `2B_p·p/(p−1) = √(2/27)/(p^{1/2}(p−1))` at `n = 2`; **`sum_cov_offsets`** `∑_{δ≠0}c_j(δ) = p`.
  Not formalized: that this equals Morwani et al.'s maximum `L_{2,3}` margin (external result).

## 7. Section `sec:cyclic`: `lem:sign`, `lem:cost`, `thm:cyclic`, `prop:block`

* **`lem_sign`** (product neurons): for every neuron and `j`, unit `ξ_t` with
  `β_{j,T} = β_j∏_{t∈T}ξ_t` for all `T`. **`lem_sign_pow`** (power neurons
  `powNeuron ω = (∑_t u_t(a_t))^n w(c)`), **`lem_sign_pow_same`** (the same `ξ` works).
  Power neurons (Power.lean): **`fhat_powNeuron`** (at modes with all `k_t ≠ 0`,
  `f̂ = n!·f̂^{prod}`), **`beta_powNeuron`**, **`betaT_powNeuron`**, **`pow_budget`**
  (`C_b^{ν/b} ≤ n!B_p`), **`thm_universal_pow`**. Difference: `(j : ZMod p) ≠ 0` is an explicit
  hypothesis in the transform lemmas (true for `j ∈ F_p`).
  Also **`beta_prodNeuron`**, **`betaT_prodNeuron`** (the factorized coefficients).
* **`Cost.lem_cost`**: `|ξ_{lt}| = 1`, (JF) (`Cost.JF`: `∑_lβ_l∏_{t∈T}ξ_{lt} = [T=∅]B` for every
  `T ⊆ Fin n`), `r ∈ (0,1]`: `(n+1)^{1−r}|B|^r ≤ ∑_l|β_l|^r`.
  **`Cost.lem_cost_eq`** (`r < 1`, `B ≠ 0`, equality): exactly `n+1` non-zero `β_l`, each of modulus
  `|B|/(n+1)`, pairwise orthogonal chain vectors (`∑_i conj(m_l(i)) m_{l'}(i) = 0`).
  **`Cost.lem_cost_eq_converse`**. Sub-lemmas: **`chain_lemma`** (`lem:chain`),
  **`det_one_add_smul_expand`** + **`det_one_add_smul_Hmat_expand`** (`lem:minor`: principal-minor
  expansion + Sylvester, stated for the chain-vector matrix `H`), **`lem_hadamard`** (Gram
  determinant of unit vectors in `[0,1]`, `= 1` iff orthonormal), **`prop_esym`**
  (`e_k(x) ≥ C(ν,k)`), **`card_support_ge`**, **`prop_gen`** (`(1+t)^ν ≤ ∏(1+x_lt)`),
  **`lem_bernstein`** / **`Kr_eq`** (exact constant `r sin(πr)/π`; Bernstein.lean).
  Differences: `prop:gen` is proved directly from the minor expansion (not via `prop:esym`);
  Hadamard via AM–GM on eigenvalues; orthogonality in the equality case via all 2-element minors
  (same conclusion); `B ≠ 0` explicit in the equality case.
* **`prop_block`**: the `ν` neurons `λcos(2πja/p + α_l)`, `λcos(2πjc/p + nα_l)`, `α_l = πl/ν`, have
  total table `νλ^ν2^{−n}cos(2πj(y(x)−c)/p)` and `‖ω^{(l)}‖² = νλ²p/2`. **`blockNet`** (one block
  per `j ∈ F_p`, `λ_j^ν = 2^n y_j/ν`): **`blockNet_table`**, **`blockNet_interferenceFree`**,
  **`blockNet_margin`** (`= worst y`), **`blockNet_normalized_margin`**
  (`= ν^{−(ν−b)/b}B_p·worst y` when `∑y_j^{b/ν} = 1`), **`block_attains`**.
  **`prod_budget`** (`∑_j|β_j(ω)|^{b/ν} ≤ B_p^{b/ν}‖ω‖^b`, `2 ≤ b`), **`prod_budget_attained`**,
  **`CbProd_eq`** (`C_b^{ν/b} = B_p`). Not formalized: the equality characterization of the budget.
* `thm:cyclic` (product networks, `2 ≤ b ≤ ν`, `s = ν/b`, `valueIF = ν^{−(ν−b)/b}B_pΓ_s(p)`):
  **`thm_cyclic_upper`**: every interference-free network has `h(θ) ≤ valueIF·‖θ‖_{2,b}^ν`.
  **`thm_cyclic_attain`**, **`thm_cyclic_value`**: for a maximizer `y*` and every width
  `m ≥ ν|spt(y*)|`, `valueIF` is the maximum (`IsGreatest`) of `h` over interference-free networks of
  width `m` with `‖θ‖_{2,b} ≤ 1`. **`thm_cyclic_structure`** (`b < ν`): an attaining interference-free
  network has (1) amplitudes `= c·y⋆` for a maximizer `y⋆`, `c > 0` (so it uses exactly `spt(y⋆)`),
  (2) at every used frequency exactly `ν` neurons with `β_ij ≠ 0`, each `|β_ij| = A_j/ν`, pairwise
  orthogonal chain vectors (for sign-rule unit numbers that exist), (3) `β_ij = 0` at unused
  frequencies. Also `freq_cost`, `JF_of_interferenceFree`, `ampNorm_le`.
  Differences: none in substance; the orthogonality in (2) is stated for a choice of the unit
  numbers `ξ` of `lem:sign` (these are not unique when some `û_t(j) = 0`).

## 8. Not attempted / out of scope

The conjecture appendix, all numerical tables and exact switch points (`p = 13, 23`), and remarks.

## 9. Error found: ReLU budget (`app:models`, "ReLU neuron")

Paper: `∑_{j∈F_p}|ĥ(j,…,j)ŵ(j)| ≤ (p/4)‖ω‖²`, i.e. `C_2 ≤ p^{−n}/2`.
Lean (ReluCounter.lean), with `reluNeuron u w x c = max(∑_t u_t(x_t), 0)·w(c)` and the witness
`p = 5`, `n = 2`, `u_1 = u_2 = (−2,2,−2,2,0)`, `w = (−2,0,3,−3,1)` (`‖ω‖² = 55`):
* **`relu_diag_claim_false`**: `(5/4)·55 < ∑_{j∈F_5} |ĥ(j,j)|·|ŵ(j)|`;
* **`relu_budget_claim_false`**: `(5^{−2}/2)·55 < ∑_{j∈F_5} |β_j(reluNeuron u w)|`.
So both displayed claims are false for `n = 2`. The correct bound from the paper's own argument is
`(p^n/4)‖ω‖²`, i.e. `C_2 ≤ 1/(2p)` (not formalized); it matches the paper only at `n = 1`.
