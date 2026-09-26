import Heracles.Defs

/-!
# Proposition `prop:ablation` ("removing a feature")

Paper, Section `sec:cyclic`, proof in appendix `app:shape`.

Let `s > 1`, `y` feasible with `∑_j y_j^{1/s} = 1` and worst coverage `m ≥ (1-ε) Γ_s(p)`,
`ε ∈ [0,1)`.  For `j ∈ spt(y)` let `w_j = y_j^{1/s}` and `m_{-j}` the worst coverage after
setting `y_j = 0`.  Then `m - 2 y_j ≤ m_{-j} ≤ (1-w_j)^s m / (1-ε)`; and
`w ≤ 1 - (1-w)^s ≤ s w` for `w ∈ [0,1]`.

We prove it for the abstract programme (any coverage matrix with entries `≤ 2`) and then
specialise to the paper's `covP p`.
-/

open Real Finset

noncomputable section

set_option linter.unusedSectionVars false

namespace Heracles

section Abstract

variable {ι Δ : Type*} [Fintype ι] [Fintype Δ] [Nonempty Δ] [DecidableEq ι]

lemma cover_smul (c : Δ → ι → ℝ) (a : ℝ) (y : ι → ℝ) (δ : Δ) :
    cover c (a • y) δ = a * cover c y δ := by
  unfold cover
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  simp [Pi.smul_apply, smul_eq_mul]
  ring

lemma worst_smul (c : Δ → ι → ℝ) {a : ℝ} (ha : 0 ≤ a) (y : ι → ℝ) :
    worst c (a • y) = a * worst c y := by
  apply le_antisymm
  · obtain ⟨δ, hδ⟩ := exists_worst_eq c y
    rw [hδ, ← cover_smul]
    exact worst_le_cover _ _ _
  · rw [le_worst_iff]
    intro δ
    rw [cover_smul]
    exact mul_le_mul_of_nonneg_left (worst_le_cover c y δ) ha

lemma budget_smul {s : ℝ} {a : ℝ} (ha : 0 ≤ a) {y : ι → ℝ} (hy : ∀ j, 0 ≤ y j) :
    budget s (a • y) = a ^ (1 / s) * budget s y := by
  unfold budget
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  simp only [Pi.smul_apply, smul_eq_mul]
  exact Real.mul_rpow ha (hy j)

lemma cover_update_zero (c : Δ → ι → ℝ) (y : ι → ℝ) (j : ι) (δ : Δ) :
    cover c (Function.update y j 0) δ = cover c y δ - y j * c δ j := by
  unfold cover
  rw [← Finset.add_sum_erase _ _ (Finset.mem_univ j),
    ← Finset.add_sum_erase _ (fun k => y k * c δ k) (Finset.mem_univ j)]
  simp only [Function.update_self, zero_mul, zero_add]
  have : ∑ x ∈ Finset.univ.erase j, Function.update y j 0 x * c δ x =
      ∑ x ∈ Finset.univ.erase j, y x * c δ x := by
    refine Finset.sum_congr rfl fun k hk => ?_
    rw [Function.update_of_ne (Finset.ne_of_mem_erase hk)]
  rw [this]
  ring

lemma budget_update_zero {s : ℝ} (hs : s ≠ 0) (y : ι → ℝ) (j : ι) :
    budget s (Function.update y j 0) = budget s y - y j ^ (1 / s) := by
  unfold budget
  rw [← Finset.add_sum_erase _ _ (Finset.mem_univ j),
    ← Finset.add_sum_erase _ (fun k => y k ^ (1 / s)) (Finset.mem_univ j)]
  simp only [Function.update_self]
  rw [Real.zero_rpow (one_div_ne_zero hs), zero_add]
  have : ∑ x ∈ Finset.univ.erase j, Function.update y j 0 x ^ (1 / s) =
      ∑ x ∈ Finset.univ.erase j, y x ^ (1 / s) := by
    refine Finset.sum_congr rfl fun k hk => ?_
    rw [Function.update_of_ne (Finset.ne_of_mem_erase hk)]
  rw [this]
  ring

/-- Lower bound of `prop:ablation`: `m - 2 y_j ≤ m_{-j}` (uses only `c ≤ 2`, `y_j ≥ 0`). -/
theorem ablation_lower {c : Δ → ι → ℝ} (hc2 : ∀ δ j, c δ j ≤ 2) {y : ι → ℝ} (j : ι)
    (hyj : 0 ≤ y j) :
    worst c y - 2 * y j ≤ worst c (Function.update y j 0) := by
  rw [le_worst_iff]
  intro δ
  rw [cover_update_zero]
  have h1 := worst_le_cover c y δ
  have h2 : y j * c δ j ≤ 2 * y j := by nlinarith [hc2 δ j]
  linarith

/-- Upper bound of `prop:ablation`: `m_{-j} ≤ (1-w_j)^s m / (1-ε)`. -/
theorem ablation_upper {c : Δ → ι → ℝ} (hc2 : ∀ δ j, c δ j ≤ 2) {s ε : ℝ} (hs : 1 < s)
    (hε1 : ε < 1) {y : ι → ℝ} (hy : Feasible s y) (hbud : budget s y = 1)
    (hm : (1 - ε) * Gam c s ≤ worst c y) (j : ι) :
    worst c (Function.update y j 0) ≤ (1 - y j ^ (1 / s)) ^ s / (1 - ε) * worst c y := by
  set w := y j ^ (1 / s) with hw
  have hs0 : s ≠ 0 := by linarith
  have hw0 : 0 ≤ w := Real.rpow_nonneg (hy.1 j) _
  have hw1 : w ≤ 1 := hy.rpow_le_one j
  have hε : 0 < 1 - ε := by linarith
  set y0 := Function.update y j 0 with hy0
  have hy0nn : ∀ k, 0 ≤ y0 k := by
    intro k
    by_cases hk : k = j
    · subst hk; simp [y0]
    · rw [hy0, Function.update_of_ne hk]; exact hy.1 k
  have hbud0 : budget s y0 = 1 - w := by rw [hy0, budget_update_zero hs0, hbud]
  rcases eq_or_lt_of_le hw1 with hw1' | hw1'
  · -- `w_j = 1`: all other coordinates vanish, so `m_{-j} = 0`.
    have hzero : y0 = 0 := by
      have hb : budget s y0 = 0 := by rw [hbud0, hw1']; ring
      unfold budget at hb
      funext k
      have hk := (Finset.sum_eq_zero_iff_of_nonneg
        (fun k _ => Real.rpow_nonneg (hy0nn k) (1 / s))).1 hb k (Finset.mem_univ k)
      have := (Real.rpow_eq_zero (hy0nn k) (one_div_ne_zero hs0)).1 hk
      simpa using this
    have hw0' : worst c (0 : ι → ℝ) = 0 := by
      have := worst_smul c (le_refl (0 : ℝ)) (0 : ι → ℝ)
      simpa using this
    rw [hzero, hw0', hw1']
    simp [Real.zero_rpow hs0]
  · -- `w_j < 1`: rescale the remaining allocation to a full budget.
    have h1w : 0 < 1 - w := by linarith
    set a := (1 - w) ^ (-s) with ha
    have ha0 : 0 < a := Real.rpow_pos_of_pos h1w _
    have hfeas : Feasible s (a • y0) := by
      refine ⟨fun k => by simp only [Pi.smul_apply, smul_eq_mul]; exact mul_nonneg ha0.le (hy0nn k),
        ?_⟩
      rw [budget_smul ha0.le hy0nn, hbud0, ha, ← Real.rpow_mul h1w.le]
      have : -s * (1 / s) = -1 := by field_simp
      rw [this, Real.rpow_neg_one]
      rw [inv_mul_cancel₀ h1w.ne']
    have hle := le_Gam hc2 hs.le hfeas
    rw [worst_smul c ha0.le] at hle
    -- `a * m_{-j} ≤ Γ ≤ m / (1-ε)`
    have hG : Gam c s ≤ worst c y / (1 - ε) := by
      rw [le_div_iff₀ hε]; linarith
    have key : a * worst c y0 ≤ worst c y / (1 - ε) := le_trans hle hG
    have hpow : (1 - w) ^ s * a = 1 := by
      rw [ha, ← Real.rpow_add h1w]; simp
    have hpos : 0 ≤ (1 - w) ^ s := Real.rpow_nonneg h1w.le _
    calc worst c y0 = (1 - w) ^ s * (a * worst c y0) := by rw [← mul_assoc, hpow, one_mul]
      _ ≤ (1 - w) ^ s * (worst c y / (1 - ε)) := mul_le_mul_of_nonneg_left key hpos
      _ = (1 - w) ^ s / (1 - ε) * worst c y := by ring

/-- `prop:ablation`, abstract form (both bounds). -/
theorem ablation {c : Δ → ι → ℝ} (hc2 : ∀ δ j, c δ j ≤ 2) {s ε : ℝ} (hs : 1 < s)
    (hε1 : ε < 1) {y : ι → ℝ} (hy : Feasible s y) (hbud : budget s y = 1)
    (hm : (1 - ε) * Gam c s ≤ worst c y) (j : ι) :
    worst c y - 2 * y j ≤ worst c (Function.update y j 0) ∧
      worst c (Function.update y j 0) ≤ (1 - y j ^ (1 / s)) ^ s / (1 - ε) * worst c y :=
  ⟨ablation_lower hc2 j (hy.1 j), ablation_upper hc2 hs hε1 hy hbud hm j⟩

end Abstract

/-- The elementary inequality `w ≤ 1 - (1-w)^s ≤ s w` for `s ≥ 1`, `w ∈ [0,1]`
(second part of `prop:ablation`). -/
theorem share_bounds {s w : ℝ} (hs : 1 ≤ s) (hw0 : 0 ≤ w) (hw1 : w ≤ 1) :
    w ≤ 1 - (1 - w) ^ s ∧ 1 - (1 - w) ^ s ≤ s * w := by
  constructor
  · have h := Real.rpow_le_rpow_of_exponent_ge' (by linarith : (0 : ℝ) ≤ 1 - w)
      (by linarith : 1 - w ≤ 1) zero_le_one hs
    rw [Real.rpow_one] at h
    linarith
  · have h := one_add_mul_self_le_rpow_one_add (by linarith : (-1 : ℝ) ≤ -w) hs
    have : 1 + -w = 1 - w := by ring
    rw [this] at h
    linarith

/-- The fraction of the margin lost when feature `j` is removed from a maximizer
(`ε = 0`): `1 - (1-w_j)^s ≤ (m - m_{-j}) / m`. -/
theorem ablation_fraction {ι Δ : Type*} [Fintype ι] [Fintype Δ] [Nonempty Δ] [DecidableEq ι]
    {c : Δ → ι → ℝ} (hc2 : ∀ δ j, c δ j ≤ 2) {s : ℝ} (hs : 1 < s)
    {y : ι → ℝ} (hy : IsMaximizer c s y) (hbud : budget s y = 1) (hpos : 0 < worst c y) (j : ι) :
    1 - (1 - y j ^ (1 / s)) ^ s ≤ (worst c y - worst c (Function.update y j 0)) / worst c y := by
  have h := ablation_upper hc2 hs (ε := 0) (by norm_num) hy.1 hbud
    (by rw [hy.worst_eq hc2 hs.le]; simp) j
  rw [le_div_iff₀ hpos]
  simp only [sub_zero, div_one] at h
  nlinarith

/-! ## The concrete statement for `Γ_s(p)` -/

/-- **Proposition `prop:ablation`** (removing a feature), for the paper's programme. -/
theorem prop_ablation (p : ℕ) [Fact p.Prime] {s ε : ℝ} (hs : 1 < s) (_hε0 : 0 ≤ ε) (hε1 : ε < 1)
    {y : Freq p → ℝ} (hy : Feasible s y) (hbud : budget s y = 1)
    (hm : (1 - ε) * GammaP p s ≤ worst (covP p) y) (j : Freq p) (_hj : j ∈ supp y) :
    worst (covP p) y - 2 * y j ≤ worst (covP p) (Function.update y j 0) ∧
      worst (covP p) (Function.update y j 0) ≤
        (1 - y j ^ (1 / s)) ^ s / (1 - ε) * worst (covP p) y ∧
      y j ^ (1 / s) ≤ 1 - (1 - y j ^ (1 / s)) ^ s ∧
      1 - (1 - y j ^ (1 / s)) ^ s ≤ s * y j ^ (1 / s) := by
  obtain ⟨h1, h2⟩ := ablation (covP_le_two p) hs hε1 hy hbud hm j
  obtain ⟨h3, h4⟩ := share_bounds hs.le (Real.rpow_nonneg (hy.1 j) _) (hy.rpow_le_one j)
  exact ⟨h1, h2, h3, h4⟩

end Heracles
