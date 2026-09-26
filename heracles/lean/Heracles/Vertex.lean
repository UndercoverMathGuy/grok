import Heracles.Ablation

/-!
# Polyhedron, vertices, `prop:shapes` (i) and `prop:bottleneck`

Paper, Section `sec:cyclic` ("The shape of the optimum"), proofs in appendix `app:shape`.

* `Poly c = {v ≥ 0 : ∑_j v_j c_j(δ) ≥ 1 for all δ}` is the polyhedron `\curlyp_p`.
* A *vertex* is an extreme point of `Poly c` (`IsVertex`).  For a polyhedron the notions
  vertex / extreme point / basic feasible solution coincide; we take "extreme point" as the
  definition and prove the rank property that the paper uses
  (`vertex_eq_zero_of_active`: at a vertex, the only direction that keeps every active
  constraint active is `0`), and finiteness of the vertex set (`finite_vertices`).
-/

open Real Finset Filter Topology

noncomputable section

set_option linter.unusedSectionVars false

namespace Heracles

/-- **`prop:shapes` (i), general form.** A minimizer of a strictly concave function over a
set `C` is an extreme point of `C`.  (No monotonicity or pointedness is needed for this
direction; existence of a minimizer is obtained separately, see `exists_minimizer`.) -/
theorem mem_extremePoints_of_isMin {E : Type*} [AddCommGroup E] [Module ℝ E] {C : Set E}
    {f : E → ℝ} (hf : StrictConcaveOn ℝ C f) {x : E} (hx : x ∈ C)
    (hmin : ∀ y ∈ C, f x ≤ f y) : x ∈ C.extremePoints ℝ := by
  rw [mem_extremePoints]
  refine ⟨hx, fun x₁ hx₁ x₂ hx₂ hmem => ?_⟩
  obtain ⟨a, b, ha, hb, hab, hxe⟩ := hmem
  by_cases h12 : x₁ = x₂
  · subst h12
    rw [← add_smul, hab, one_smul] at hxe
    exact ⟨hxe, hxe⟩
  · exfalso
    have hlt := hf.2 hx₁ hx₂ h12 ha hb hab
    rw [hxe] at hlt
    have h1 := mul_le_mul_of_nonneg_left (hmin x₁ hx₁) ha.le
    have h2 := mul_le_mul_of_nonneg_left (hmin x₂ hx₂) hb.le
    simp only [smul_eq_mul] at hlt
    have h3 : a * f x + b * f x = f x := by rw [← add_mul, hab, one_mul]
    linarith

section Abstract

variable {ι Δ : Type*} [Fintype ι] [Fintype Δ] [Nonempty Δ] [DecidableEq ι]

/-- The polyhedron `P = {v ≥ 0 : ∑_j v_j c_j(δ) ≥ 1 ∀ δ}`. -/
def Poly (c : Δ → ι → ℝ) : Set (ι → ℝ) := {v | (∀ j, 0 ≤ v j) ∧ ∀ δ, 1 ≤ cover c v δ}

/-- A vertex of `Poly c`, i.e. an extreme point. -/
def IsVertex (c : Δ → ι → ℝ) (v : ι → ℝ) : Prop := v ∈ (Poly c).extremePoints ℝ

lemma cover_add (c : Δ → ι → ℝ) (v w : ι → ℝ) (δ : Δ) :
    cover c (v + w) δ = cover c v δ + cover c w δ := by
  unfold cover
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun j _ => ?_
  simp only [Pi.add_apply]; ring

lemma cover_sub (c : Δ → ι → ℝ) (v w : ι → ℝ) (δ : Δ) :
    cover c (v - w) δ = cover c v δ - cover c w δ := by
  unfold cover
  rw [← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun j _ => ?_
  simp only [Pi.sub_apply]; ring

lemma convex_Poly (c : Δ → ι → ℝ) : Convex ℝ (Poly c) := by
  intro v hv w hw a b ha hb hab
  refine ⟨fun j => ?_, fun δ => ?_⟩
  · simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    exact add_nonneg (mul_nonneg ha (hv.1 j)) (mul_nonneg hb (hw.1 j))
  · rw [cover_add, cover_smul, cover_smul]
    have := hv.2 δ; have := hw.2 δ
    nlinarith

lemma convex_orthant : Convex ℝ {v : ι → ℝ | ∀ j, 0 ≤ v j} := by
  intro v hv w hw a b ha hb _ j
  simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
  exact add_nonneg (mul_nonneg ha (hv j)) (mul_nonneg hb (hw j))

/-- `φ(v) = ∑_j v_j^{1/s}` is strictly concave on the non-negative orthant for `s > 1`. -/
lemma strictConcaveOn_budget {s : ℝ} (hs : 1 < s) :
    StrictConcaveOn ℝ {v : ι → ℝ | ∀ j, 0 ≤ v j} (budget s) := by
  have hr0 : 0 < 1 / s := by positivity
  have hr1 : 1 / s < 1 := by rw [div_lt_one (by linarith)]; exact hs
  have hsc := Real.strictConcaveOn_rpow hr0 hr1
  refine ⟨convex_orthant, fun v hv w hw hvw a b ha hb hab => ?_⟩
  obtain ⟨j0, hj0⟩ : ∃ j, v j ≠ w j := by
    by_contra h; push Not at h; exact hvw (funext h)
  unfold budget
  simp only [smul_eq_mul, Finset.mul_sum]
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_lt_sum
  · intro j _
    have := hsc.concaveOn.2 (Set.mem_Ici.2 (hv j)) (Set.mem_Ici.2 (hw j)) ha.le hb.le hab
    simpa [smul_eq_mul] using this
  · refine ⟨j0, Finset.mem_univ _, ?_⟩
    have := hsc.2 (Set.mem_Ici.2 (hv j0)) (Set.mem_Ici.2 (hw j0)) hj0 ha hb hab
    simpa [smul_eq_mul] using this

lemma Poly_subset_orthant (c : Δ → ι → ℝ) : Poly c ⊆ {v : ι → ℝ | ∀ j, 0 ≤ v j} :=
  fun _ hv => hv.1

/-- **`prop:shapes` (i)** for the programme: every minimizer of `φ = ∑ v_j^{1/s}` over the
polyhedron is a vertex. -/
theorem minimizer_isVertex {c : Δ → ι → ℝ} {s : ℝ} (hs : 1 < s) {v : ι → ℝ}
    (hv : v ∈ Poly c) (hmin : ∀ w ∈ Poly c, budget s v ≤ budget s w) : IsVertex c v :=
  mem_extremePoints_of_isMin ((strictConcaveOn_budget hs).subset (Poly_subset_orthant c)
    (convex_Poly c)) hv hmin

/-- **Rank property of vertices.** If `v` is a vertex and a direction `d` vanishes on the zero
coordinates of `v` and keeps every active coverage constraint of `v` active
(`∑ d_j c_j(δ) = 0` whenever `∑ v_j c_j(δ) = 1`), then `d = 0`. -/
theorem vertex_eq_zero_of_active {c : Δ → ι → ℝ} {v : ι → ℝ} (hv : IsVertex c v)
    {d : ι → ℝ} (hd0 : ∀ j, v j = 0 → d j = 0)
    (hd1 : ∀ δ, cover c v δ = 1 → cover c d δ = 0) : d = 0 := by
  rw [IsVertex, mem_extremePoints] at hv
  obtain ⟨hvP, hext⟩ := hv
  have key : ∀ᶠ ε in 𝓝 (0 : ℝ), v + ε • d ∈ Poly c := by
    have h1 : ∀ᶠ ε in 𝓝 (0 : ℝ), ∀ j, 0 ≤ (v + ε • d) j := by
      rw [Filter.eventually_all]
      intro j
      by_cases hj : v j = 0
      · exact Filter.Eventually.of_forall fun ε => by simp [hj, hd0 j hj]
      · have hpos : 0 < v j := lt_of_le_of_ne (hvP.1 j) (Ne.symm hj)
        have ht : Tendsto (fun ε : ℝ => v j + ε * d j) (𝓝 0) (𝓝 (v j)) := by
          have hc : Continuous (fun ε : ℝ => v j + ε * d j) := by fun_prop
          simpa using hc.tendsto 0
        filter_upwards [ht.eventually_const_lt hpos] with ε hε
        simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
        exact hε.le
    have h2 : ∀ᶠ ε in 𝓝 (0 : ℝ), ∀ δ, 1 ≤ cover c (v + ε • d) δ := by
      rw [Filter.eventually_all]
      intro δ
      by_cases hδ : cover c v δ = 1
      · exact Filter.Eventually.of_forall fun ε => by
          rw [cover_add, cover_smul, hd1 δ hδ, hδ]; simp
      · have hgt : 1 < cover c v δ := lt_of_le_of_ne (hvP.2 δ) (Ne.symm hδ)
        have ht : Tendsto (fun ε : ℝ => cover c v δ + ε * cover c d δ) (𝓝 0)
            (𝓝 (cover c v δ)) := by
          have hc : Continuous (fun ε : ℝ => cover c v δ + ε * cover c d δ) := by fun_prop
          simpa using hc.tendsto 0
        filter_upwards [ht.eventually_const_lt hgt] with ε hε
        rw [cover_add, cover_smul]
        exact hε.le
    filter_upwards [h1, h2] with ε h1 h2
    exact ⟨h1, h2⟩
  have key' : ∀ᶠ ε in 𝓝 (0 : ℝ), v + (-ε) • d ∈ Poly c := by
    have ht : Tendsto (fun ε : ℝ => -ε) (𝓝 0) (𝓝 0) := by
      simpa using (continuous_neg.tendsto (0 : ℝ))
    exact ht.eventually key
  have hboth : ∀ᶠ ε in 𝓝[≠] (0 : ℝ), (v + ε • d ∈ Poly c ∧ v + (-ε) • d ∈ Poly c) ∧ ε ≠ 0 :=
    ((key.and key').filter_mono nhdsWithin_le_nhds).and self_mem_nhdsWithin
  obtain ⟨ε, ⟨hp, hm⟩, hε⟩ := hboth.exists
  have hseg : v ∈ openSegment ℝ (v + ε • d) (v + (-ε) • d) := by
    refine ⟨1 / 2, 1 / 2, by norm_num, by norm_num, by norm_num, ?_⟩
    funext j
    simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    ring
  have := (hext _ hp _ hm hseg).1
  have hεd : ε • d = 0 := by
    have h' : v + ε • d - v = 0 := by rw [this]; simp
    simpa using h'
  rcases smul_eq_zero.1 hεd with h | h
  · exact absurd h hε
  · exact h

/-- **The vertex set is finite.** A vertex is determined by its zero set and its active set. -/
theorem finite_vertices (c : Δ → ι → ℝ) : ((Poly c).extremePoints ℝ).Finite := by
  classical
  let g : (ι → ℝ) → Finset ι × Finset Δ := fun v =>
    (Finset.univ.filter fun j => v j = 0, Finset.univ.filter fun δ => cover c v δ = 1)
  refine Set.Finite.of_finite_image (f := g) (Set.toFinite _) ?_
  intro v hv w hw hvw
  have hv' : IsVertex c v := hv
  have hw' : IsVertex c w := hw
  simp only [g, Prod.mk.injEq] at hvw
  obtain ⟨hz, ha⟩ := hvw
  have hd := vertex_eq_zero_of_active hv' (d := v - w) (fun j hj => by
      have : j ∈ Finset.univ.filter fun j => w j = 0 := by
        rw [← hz]; simp [hj]
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at this
      simp [hj, this])
    (fun δ hδ => by
      have : δ ∈ Finset.univ.filter fun δ => cover c w δ = 1 := by
        rw [← ha]; simp [hδ]
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at this
      rw [cover_sub, hδ, this]; ring)
  exact sub_eq_zero.1 hd

/-- The worst coverage of the zero allocation is `0`. -/
lemma worst_zero (c : Δ → ι → ℝ) : worst c (0 : ι → ℝ) = 0 := by
  have := worst_smul c (le_refl (0 : ℝ)) (0 : ι → ℝ)
  simpa using this

lemma budget_pos_of_ne {s : ℝ} (_hs : s ≠ 0) {y : ι → ℝ} (hy : ∀ j, 0 ≤ y j) (hne : y ≠ 0) :
    0 < budget s y := by
  obtain ⟨j, hj⟩ : ∃ j, y j ≠ 0 := by
    by_contra h; push Not at h; exact hne (funext h)
  have hjpos : 0 < y j ^ (1 / s) := Real.rpow_pos_of_pos (lt_of_le_of_ne (hy j) (Ne.symm hj)) _
  unfold budget
  exact lt_of_lt_of_le hjpos
    (Finset.single_le_sum (fun k _ => Real.rpow_nonneg (hy k) _) (Finset.mem_univ j))

/-- A maximizer spends its whole budget (when `Γ_s > 0`). -/
theorem IsMaximizer.budget_eq_one {c : Δ → ι → ℝ} (hc2 : ∀ δ j, c δ j ≤ 2) {s : ℝ}
    (hs : 1 < s) (hΓ : 0 < Gam c s) {y : ι → ℝ} (hy : IsMaximizer c s y) :
    budget s y = 1 := by
  have hs0 : s ≠ 0 := by linarith
  have hw : worst c y = Gam c s := hy.worst_eq hc2 hs.le
  have hne : y ≠ 0 := by
    intro h; rw [h, worst_zero] at hw; linarith
  have hb := budget_pos_of_ne hs0 hy.1.1 hne
  by_contra hne1
  have hb1 : budget s y < 1 := lt_of_le_of_ne hy.1.2 hne1
  set b := budget s y
  set a := b ^ (-s) with ha
  have ha0 : 0 < a := Real.rpow_pos_of_pos hb _
  have ha1 : 1 < a := Real.one_lt_rpow_of_pos_of_lt_one_of_neg hb hb1 (by linarith)
  have hfeas : Feasible s (a • y) := by
    refine ⟨fun k => by simp only [Pi.smul_apply, smul_eq_mul]; exact mul_nonneg ha0.le (hy.1.1 k),
      ?_⟩
    rw [budget_smul ha0.le hy.1.1, ha, ← Real.rpow_mul hb.le]
    have : -s * (1 / s) = -1 := by field_simp
    rw [this, Real.rpow_neg_one, inv_mul_cancel₀ hb.ne']
  have h := hy.2 _ hfeas
  rw [worst_smul c ha0.le, hw] at h
  nlinarith

/-- **Reduction to the polyhedron.** If `y` is a maximizer then `v = Γ⁻¹ y` lies in the
polyhedron and minimizes `φ = ∑_j v_j^{1/s}` over it. -/
theorem IsMaximizer.minimizes {c : Δ → ι → ℝ} (hc2 : ∀ δ j, c δ j ≤ 2) {s : ℝ} (hs : 1 < s)
    (hΓ : 0 < Gam c s) {y : ι → ℝ} (hy : IsMaximizer c s y) :
    (Gam c s)⁻¹ • y ∈ Poly c ∧ ∀ w ∈ Poly c, budget s ((Gam c s)⁻¹ • y) ≤ budget s w := by
  have hs0 : s ≠ 0 := by linarith
  have hw : worst c y = Gam c s := hy.worst_eq hc2 hs.le
  have hΓi : 0 < (Gam c s)⁻¹ := inv_pos.2 hΓ
  refine ⟨⟨fun j => ?_, fun δ => ?_⟩, fun w hwP => ?_⟩
  · simp only [Pi.smul_apply, smul_eq_mul]; exact mul_nonneg hΓi.le (hy.1.1 j)
  · rw [cover_smul]
    have := worst_le_cover c y δ
    rw [hw] at this
    calc (1 : ℝ) = (Gam c s)⁻¹ * Gam c s := (inv_mul_cancel₀ hΓ.ne').symm
      _ ≤ (Gam c s)⁻¹ * cover c y δ := mul_le_mul_of_nonneg_left this hΓi.le
  · rw [budget_smul hΓi.le hy.1.1, hy.budget_eq_one hc2 hs hΓ, mul_one]
    -- `φ(w) > 0` and `φ(w)^{-s} w` is feasible with worst coverage `≥ φ(w)^{-s}`.
    have hwne : w ≠ 0 := by
      intro h
      obtain ⟨δ⟩ := (inferInstance : Nonempty Δ)
      have := hwP.2 δ
      rw [h] at this
      unfold cover at this
      simp at this
      linarith
    set φ := budget s w
    have hφ : 0 < φ := budget_pos_of_ne hs0 hwP.1 hwne
    set a := φ ^ (-s) with ha
    have ha0 : 0 < a := Real.rpow_pos_of_pos hφ _
    have hfeas : Feasible s (a • w) := by
      refine ⟨fun k => by simp only [Pi.smul_apply, smul_eq_mul]; exact mul_nonneg ha0.le (hwP.1 k),
        ?_⟩
      rw [budget_smul ha0.le hwP.1, ha, ← Real.rpow_mul hφ.le]
      have : -s * (1 / s) = -1 := by field_simp
      rw [this, Real.rpow_neg_one, inv_mul_cancel₀ hφ.ne']
    have hle := le_Gam hc2 hs.le hfeas
    rw [worst_smul c ha0.le] at hle
    have hw1 : 1 ≤ worst c w := le_worst_iff.2 hwP.2
    have haΓ : a ≤ Gam c s := by nlinarith
    -- raise `φ^{-s} ≤ Γ` to the power `1/s`
    have h1 : a ^ (1 / s) ≤ (Gam c s) ^ (1 / s) :=
      Real.rpow_le_rpow ha0.le haΓ (by positivity)
    have h2 : a ^ (1 / s) = φ⁻¹ := by
      rw [ha, ← Real.rpow_mul hφ.le]
      have : -s * (1 / s) = -1 := by field_simp
      rw [this, Real.rpow_neg_one]
    rw [h2] at h1
    rw [Real.inv_rpow hΓ.le]
    have hG : 0 < (Gam c s) ^ (1 / s) := Real.rpow_pos_of_pos hΓ _
    rw [inv_le_comm₀ hG hφ]
    exact h1

/-- **`prop:shapes`, first sentence (abstract).** Every maximizer is `y = Γ v` for a vertex
`v` of the polyhedron that minimizes `∑_j v_j^{1/s}` over it. -/
theorem maximizer_eq_vertex {c : Δ → ι → ℝ} (hc2 : ∀ δ j, c δ j ≤ 2) {s : ℝ} (hs : 1 < s)
    (hΓ : 0 < Gam c s) {y : ι → ℝ} (hy : IsMaximizer c s y) :
    ∃ v, y = Gam c s • v ∧ IsVertex c v ∧ ∀ w ∈ Poly c, budget s v ≤ budget s w := by
  obtain ⟨hP, hmin⟩ := hy.minimizes hc2 hs hΓ
  refine ⟨(Gam c s)⁻¹ • y, ?_, minimizer_isVertex hs hP hmin, hmin⟩
  rw [smul_smul, mul_inv_cancel₀ hΓ.ne', one_smul]

/-- Existence of a minimizer of `φ` over the polyhedron (from existence of a maximizer). -/
theorem exists_minimizer {c : Δ → ι → ℝ} (hc2 : ∀ δ j, c δ j ≤ 2) {s : ℝ} (hs : 1 < s)
    (hΓ : 0 < Gam c s) : ∃ v ∈ Poly c, ∀ w ∈ Poly c, budget s v ≤ budget s w := by
  obtain ⟨y, hy⟩ := exists_maximizer c hs.le
  exact ⟨_, (hy.minimizes hc2 hs hΓ).1, (hy.minimizes hc2 hs hΓ).2⟩

/-- Conversely, every minimizer `v` of `φ` over the polyhedron gives the maximizer `Γ v`. -/
theorem isMaximizer_of_minimizer {c : Δ → ι → ℝ} (hc2 : ∀ δ j, c δ j ≤ 2) {s : ℝ}
    (hs : 1 < s) (hΓ : 0 < Gam c s) {v : ι → ℝ} (hv : v ∈ Poly c)
    (hmin : ∀ w ∈ Poly c, budget s v ≤ budget s w) : IsMaximizer c s (Gam c s • v) := by
  obtain ⟨y, hy⟩ := exists_maximizer c hs.le
  obtain ⟨hP0, hmin0⟩ := hy.minimizes hc2 hs hΓ
  have heq : budget s v = budget s ((Gam c s)⁻¹ • y) := le_antisymm (hmin _ hP0) (hmin0 _ hv)
  have hΓi : 0 < (Gam c s)⁻¹ := inv_pos.2 hΓ
  have hb0 : budget s ((Gam c s)⁻¹ • y) = ((Gam c s)⁻¹) ^ (1 / s) := by
    rw [budget_smul hΓi.le hy.1.1, hy.budget_eq_one hc2 hs hΓ, mul_one]
  have hfeas : Feasible s (Gam c s • v) := by
    refine ⟨fun k => by simp only [Pi.smul_apply, smul_eq_mul]; exact mul_nonneg hΓ.le (hv.1 k),
      ?_⟩
    rw [budget_smul hΓ.le hv.1, heq, hb0, ← Real.mul_rpow hΓ.le hΓi.le,
      mul_inv_cancel₀ hΓ.ne', Real.one_rpow]
  refine ⟨hfeas, fun y' hy' => ?_⟩
  rw [worst_smul c hΓ.le]
  have h1 : 1 ≤ worst c v := le_worst_iff.2 hv.2
  have := le_Gam hc2 hs.le hy'
  nlinarith

/-- The budget shares of a maximizer: `y_j^{1/s} = v_j^{1/s} / ∑_k v_k^{1/s}` where
`y = Γ v` (`prop:shapes`, consequence (iii)). -/
theorem share_formula {c : Δ → ι → ℝ} (hc2 : ∀ δ j, c δ j ≤ 2) {s : ℝ} (hs : 1 < s)
    (hΓ : 0 < Gam c s) {y v : ι → ℝ} (hy : IsMaximizer c s y) (hyv : y = Gam c s • v)
    (hv : ∀ j, 0 ≤ v j) (j : ι) :
    y j ^ (1 / s) = v j ^ (1 / s) / budget s v := by
  have hb := hy.budget_eq_one hc2 hs hΓ
  rw [hyv, budget_smul hΓ.le hv] at hb
  have hG : 0 < (Gam c s) ^ (1 / s) := Real.rpow_pos_of_pos hΓ _
  have hbv : budget s v = ((Gam c s) ^ (1 / s))⁻¹ := by
    field_simp; linarith
  rw [hbv, hyv]
  simp only [Pi.smul_apply, smul_eq_mul]
  rw [Real.mul_rpow hΓ.le (hv j)]
  field_simp

/-- **`prop:bottleneck`, abstract form.**  Let `y` be a maximizer with support `S`,
`K = |S|`.  Let `A` be a set of active offsets (`F_y(δ) = Γ`) containing, for every active
offset, one with the same coverage row.  Then there are `K` distinct offsets in `A` whose
coverage rows restricted to `S` are linearly independent, and `y` is the unique vector
supported on `S` solving the corresponding `K` equations `F_z(δ_i) = Γ`. -/
theorem bottleneck_abstract {c : Δ → ι → ℝ} (hc2 : ∀ δ j, c δ j ≤ 2) {s : ℝ} (hs : 1 < s)
    (hΓ : 0 < Gam c s) {y : ι → ℝ} (hy : IsMaximizer c s y) (A : Finset Δ)
    (hA : ∀ δ ∈ A, cover c y δ = Gam c s)
    (hArows : ∀ δ, cover c y δ = Gam c s → ∃ δ' ∈ A, ∀ j, c δ' j = c δ j) :
    ∃ δs : Fin (supp y).card → Δ, Function.Injective δs ∧ (∀ i, δs i ∈ A) ∧
      (∀ i, cover c y (δs i) = Gam c s) ∧
      LinearIndependent ℝ (fun i => fun j : supp y => c (δs i) j) ∧
      ∀ z : ι → ℝ, (∀ j, j ∉ supp y → z j = 0) → (∀ i, cover c z (δs i) = Gam c s) → z = y := by
  classical
  set S := supp y with hS
  -- the vertex `v = Γ⁻¹ y`
  obtain ⟨v, hyv, hvert, -⟩ := maximizer_eq_vertex hc2 hs hΓ hy
  have hvy : ∀ j, v j = 0 ↔ y j = 0 := by
    intro j; rw [hyv]; simp [hΓ.ne']
  have hact : ∀ δ, cover c v δ = 1 ↔ cover c y δ = Gam c s := by
    intro δ; rw [hyv, cover_smul]
    constructor
    · intro h; rw [h, mul_one]
    · intro h; field_simp at h ⊢; linarith
  -- rows restricted to the support
  let row : Δ → (S → ℝ) := fun δ j => c δ j
  have hcoverS : ∀ (d : ι → ℝ), (∀ j, j ∉ S → d j = 0) → ∀ δ,
      cover c d δ = ∑ k : S, d k * row δ k := by
    intro d hd δ
    unfold cover
    rw [← Finset.sum_subset (Finset.subset_univ S) (fun j _ hj => by rw [hd j hj, zero_mul])]
    exact (Finset.sum_coe_sort S (fun j => d j * c δ j)).symm
  -- Step 1: the rows indexed by `A` span everything.
  have hspan : Submodule.span ℝ (Set.range fun δ : A => row δ) = ⊤ := by
    by_contra hne
    obtain ⟨f, hf0, hker⟩ := Submodule.exists_le_ker_of_lt_top _ (lt_top_iff_ne_top.2 hne)
    let d : ι → ℝ := fun j => if h : j ∈ S then f (Pi.single ⟨j, h⟩ 1) else 0
    have hdS : ∀ j, j ∉ S → d j = 0 := fun j hj => by simp [d, hj]
    have hfrow : ∀ δ, f (row δ) = ∑ k : S, d k * row δ k := by
      intro δ
      rw [LinearMap.pi_apply_eq_sum_univ f (row δ)]
      refine Finset.sum_congr rfl fun k _ => ?_
      simp only [d, k.2, dite_true, smul_eq_mul]
      rw [mul_comm]
      congr 2
      funext k'
      simp [Pi.single_apply, eq_comm]
    have hd := vertex_eq_zero_of_active hvert (d := d)
      (fun j hj => hdS j (by simp [hS, supp, (hvy j).1 hj]))
      (fun δ hδ => by
        obtain ⟨δ', hδ'A, hrow⟩ := hArows δ ((hact δ).1 hδ)
        have hrow' : row δ' = row δ := funext fun k => hrow k
        rw [hcoverS d hdS, ← hfrow, ← hrow']
        exact hker (Submodule.subset_span ⟨⟨δ', hδ'A⟩, rfl⟩))
    apply hf0
    refine LinearMap.ext fun u => ?_
    rw [LinearMap.pi_apply_eq_sum_univ f u]
    simp only [LinearMap.zero_apply]
    refine Finset.sum_eq_zero fun k _ => ?_
    have : d k = 0 := by rw [hd]; rfl
    simp only [d, k.2, dite_true] at this
    have h2 : (fun j => if k = j then (1 : ℝ) else 0) = Pi.single k 1 := by
      funext k'; simp [Pi.single_apply, eq_comm]
    rw [h2]
    simp [this]
  -- Step 2: extract a linearly independent spanning subfamily.
  obtain ⟨b, hbsub, hbspan, hbli⟩ := exists_linearIndependent ℝ (Set.range fun δ : A => row δ)
  have hbtop : Submodule.span ℝ b = ⊤ := hbspan.trans hspan
  have hbfin : b.Finite := (Set.finite_range _).subset hbsub
  let _ : Fintype b := hbfin.fintype
  let B : Module.Basis b ℝ (S → ℝ) := Module.Basis.mk hbli (by rw [Subtype.range_coe, hbtop])
  have hcard : Fintype.card b = S.card := by
    rw [← Module.finrank_eq_card_basis B, Module.finrank_fintype_fun_eq_card, Fintype.card_coe]
  let e : b ≃ Fin S.card := Fintype.equivFinOfCardEq hcard
  have hpre : ∀ i : Fin S.card, ∃ δ : A, row δ = (e.symm i).1 := fun i => hbsub (e.symm i).2
  choose δA hδA using hpre
  refine ⟨fun i => (δA i : Δ), ?_, fun i => (δA i).2, fun i => hA _ (δA i).2, ?_, ?_⟩
  · intro i i' h
    have : (e.symm i).1 = (e.symm i').1 := by rw [← hδA, ← hδA]; simp only at h; rw [Subtype.ext h]
    exact e.symm.injective (Subtype.ext this)
  · have : (fun i => fun j : S => c (δA i : Δ) j) = (Subtype.val ∘ e.symm) := by
      funext i; exact hδA i
    rw [this]
    exact hbli.comp e.symm e.symm.injective
  · -- Step 3: uniqueness.
    intro z hz hzeq
    set d := z - y with hd
    have hdS : ∀ j, j ∉ S → d j = 0 := by
      intro j hj
      have hyj : y j = 0 := by simpa [hS, supp] using hj
      simp [hd, hz j hj, hyj]
    have hli : LinearIndependent ℝ (fun i => fun j : S => c (δA i : Δ) j) := by
      have : (fun i => fun j : S => c (δA i : Δ) j) = (Subtype.val ∘ e.symm) := by
        funext i; exact hδA i
      rw [this]; exact hbli.comp e.symm e.symm.injective
    have htop := hli.span_eq_top_of_card_eq_finrank'
      (by rw [Module.finrank_fintype_fun_eq_card, Fintype.card_fin, Fintype.card_coe])
    let g : (S → ℝ) →ₗ[ℝ] ℝ :=
      { toFun := fun u => ∑ k : S, u k * d k
        map_add' := fun u u' => by
          simp only [Pi.add_apply, add_mul, Finset.sum_add_distrib]
        map_smul' := fun a u => by
          simp only [Pi.smul_apply, smul_eq_mul, RingHom.id_apply, Finset.mul_sum, mul_assoc] }
    have hgrow : ∀ i, g (fun j : S => c (δA i : Δ) j) = 0 := by
      intro i
      show ∑ k : S, c (δA i : Δ) k * d k = 0
      have h1 := hcoverS d hdS (δA i)
      have h2 : cover c d (δA i) = 0 := by
        rw [hd, cover_sub, hzeq i, hA _ (δA i).2, sub_self]
      rw [← h2, h1]
      refine Finset.sum_congr rfl fun k _ => by ring
    have hker : Submodule.span ℝ (Set.range fun i => fun j : S => c (δA i : Δ) j) ≤
        LinearMap.ker g := by
      rw [Submodule.span_le]
      rintro _ ⟨i, rfl⟩
      exact hgrow i
    rw [htop] at hker
    have hd0 : d = 0 := by
      funext j
      by_cases hj : j ∈ S
      · have := hker (Submodule.mem_top (x := (Pi.single (⟨j, hj⟩ : S) (1 : ℝ) : S → ℝ)))
        rw [LinearMap.mem_ker] at this
        change ∑ k : S, (Pi.single (⟨j, hj⟩ : S) (1 : ℝ) : S → ℝ) k * d k = 0 at this
        rw [Finset.sum_eq_single (⟨j, hj⟩ : S)] at this
        · simpa using this
        · intro k _ hk; simp [Pi.single_apply, hk]
        · simp
      · exact hdS j hj
    exact sub_eq_zero.1 hd0

end Abstract

end Heracles
