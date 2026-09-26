import Heracles.Shapes

/-!
# `prop:shapes`, parenthetical: images of a vertex under `j ↦ uj mod p` are tied vertices

For a unit `u ∈ ℤ_p^×`, multiplication by `u` permutes the frequencies `F_p` (after folding
`k ↦ min(k, p−k)`), and `c_{uj}(δ) = c_j(uδ)`.  Hence the polyhedron is invariant, vertices map to
vertices, and the multiset of non-zero entries (so `∑_j v_j^{1/s}` for every `s`) is unchanged.
-/

open Real Finset

noncomputable section

set_option linter.unusedSectionVars false

namespace Heracles

variable {p : ℕ} [Fact p.Prime]

lemma cos_congr_mod {a b : ℕ} (h : (a : ZMod p) = (b : ZMod p)) :
    Real.cos (2 * π * a / p) = Real.cos (2 * π * b / p) := by
  have hp0 : (p : ℝ) ≠ 0 := by exact_mod_cast (Fact.out : p.Prime).ne_zero
  have hmod : a % p = b % p := (ZMod.natCast_eq_natCast_iff' a b p).1 h
  have ea : (a : ℝ) = (a % p : ℕ) + p * (a / p : ℕ) := by
    exact_mod_cast (Nat.mod_add_div a p).symm
  have eb : (b : ℝ) = (b % p : ℕ) + p * (b / p : ℕ) := by
    exact_mod_cast (Nat.mod_add_div b p).symm
  have key : ∀ (r q : ℕ), Real.cos (2 * π * ((r : ℝ) + p * q) / p) = Real.cos (2 * π * r / p) := by
    intro r q
    have : 2 * π * ((r : ℝ) + p * q) / p = 2 * π * r / p + (q : ℕ) * (2 * π) := by
      field_simp
    rw [this]
    exact Real.cos_add_nat_mul_two_pi _ _
  calc Real.cos (2 * π * a / p) = Real.cos (2 * π * ((a % p : ℕ) : ℝ) / p) := by rw [ea, key]
    _ = Real.cos (2 * π * ((b % p : ℕ) : ℝ) / p) := by rw [hmod]
    _ = Real.cos (2 * π * b / p) := by rw [eb, key]

/-- Folding `k ↦ min(k, p − k)`. -/
def fold (k : ZMod p) : ℕ := min k.val (p - k.val)

lemma fold_mem (hp : 3 ≤ p) {k : ZMod p} (hk : k ≠ 0) : fold k ∈ Finset.Icc 1 ((p - 1) / 2) := by
  have h0 : k.val ≠ 0 := fun h => hk ((ZMod.val_eq_zero k).1 h)
  have hlt := ZMod.val_lt k
  have hodd : p % 2 = 1 := Nat.odd_iff.1 ((Fact.out : p.Prime).odd_of_ne_two (by omega))
  rw [Finset.mem_Icc]
  unfold fold
  constructor
  · rw [le_min_iff]; omega
  · rcases le_total k.val (p - k.val) with h | h
    · rw [min_eq_left h]; omega
    · rw [min_eq_right h]; omega

lemma cos_fold (k : ZMod p) (d : ℕ) :
    Real.cos (2 * π * (fold k : ℝ) * d / p) = Real.cos (2 * π * (k.val : ℝ) * d / p) := by
  unfold fold
  rcases le_total k.val (p - k.val) with h | h
  · rw [min_eq_left h]
  · rw [min_eq_right h]
    have hle : k.val ≤ p := (ZMod.val_lt k).le
    have := cos_reflect (p := p) d k.val hle
    rw [show 2 * π * ((p - k.val : ℕ) : ℝ) * d / p = 2 * π * d * ((p - k.val : ℕ) : ℝ) / p by ring,
      this]
    ring_nf

/-- Multiplication by a unit `u`, as a map on `F_p`. -/
def mulFreq (hp : 3 ≤ p) (u : ZMod p) (hu : u ≠ 0) (j : Freq p) : Freq p :=
  ⟨fold (u * (j.1 : ZMod p)), fold_mem hp (by
    have hj : (j.1 : ZMod p) ≠ 0 := by
      intro h
      rw [ZMod.natCast_eq_zero_iff] at h
      have := Finset.mem_Icc.1 j.2
      have : j.1 < p := by omega
      exact absurd (Nat.le_of_dvd (by omega) h) (by omega)
    exact mul_ne_zero hu hj)⟩

/-- Multiplication by a unit, as a map on the non-zero offsets. -/
def mulOff (u : ZMod p) (hu : u ≠ 0) (δ : Off p) : Off p := ⟨u * δ.1, mul_ne_zero hu δ.2⟩

lemma mulOff_bijective (u : ZMod p) (hu : u ≠ 0) : Function.Bijective (mulOff u hu) := by
  refine (Finite.injective_iff_bijective).1 fun δ δ' h => ?_
  have : u * δ.1 = u * δ'.1 := congrArg Subtype.val h
  exact Subtype.ext (mul_left_cancel₀ hu this)

/-- `c_{uj}(δ) = c_j(uδ)`. -/
lemma covP_mulFreq (hp : 3 ≤ p) (u : ZMod p) (hu : u ≠ 0) (δ : Off p) (j : Freq p) :
    covP p δ (mulFreq hp u hu j) = covP p (mulOff u hu δ) j := by
  unfold covP cov mulFreq mulOff
  simp only
  rw [cos_fold]
  congr 1
  rw [show 2 * π * ((j.1 : ℕ) : ℝ) * ((u * δ.1).val : ℝ) / p =
      2 * π * (((j.1 * (u * δ.1).val : ℕ)) : ℝ) / p by push_cast; ring,
    show 2 * π * ((u * (j.1 : ZMod p)).val : ℝ) * (δ.1.val : ℝ) / p =
      2 * π * ((((u * (j.1 : ZMod p)).val * δ.1.val : ℕ)) : ℝ) / p by push_cast; ring]
  apply cos_congr_mod
  push_cast
  simp only [ZMod.natCast_val, ZMod.cast_id', id]
  ring

lemma mulFreq_injective (hp : 3 ≤ p) (u : ZMod p) (hu : u ≠ 0) :
    Function.Injective (mulFreq hp u hu) := by
  intro j j' h
  -- compare the coverage rows: `c_j(uδ) = c_{j'}(uδ)` for every δ; take `δ = u⁻¹`
  have hrow : ∀ δ : Off p, covP p (mulOff u hu δ) j = covP p (mulOff u hu δ) j' := by
    intro δ
    rw [← covP_mulFreq hp, ← covP_mulFreq hp, h]
  set δ0 : Off p := ⟨u⁻¹, inv_ne_zero hu⟩
  have h1 := hrow δ0
  have hδ : (mulOff u hu δ0).1 = 1 := by simp [mulOff, δ0, mul_inv_cancel₀ hu]
  unfold covP cov at h1
  rw [hδ, ZMod.val_one] at h1
  -- `cos(2πj/p) = cos(2πj'/p)` with `j, j' ∈ [1, (p-1)/2]` forces `j = j'`
  have hj := Finset.mem_Icc.1 j.2
  have hj' := Finset.mem_Icc.1 j'.2
  have hp0 : (0 : ℝ) < p := by exact_mod_cast (Fact.out : p.Prime).pos
  have hcos : Real.cos (2 * π * (j.1 : ℝ) / p) = Real.cos (2 * π * (j'.1 : ℝ) / p) := by
    have := h1; simp only [Nat.cast_one, mul_one] at this; linarith
  have hmem : ∀ k : ℕ, k ≤ (p - 1) / 2 → 2 * π * (k : ℝ) / p ∈ Set.Icc 0 π := by
    intro k hk
    constructor
    · positivity
    · rw [div_le_iff₀ hp0]
      have : (k : ℝ) * 2 ≤ p := by
        have : k * 2 ≤ p := by omega
        exact_mod_cast this
      nlinarith [Real.pi_pos]
  have := Real.injOn_cos (hmem _ hj.2) (hmem _ hj'.2) hcos
  have h2 : (j.1 : ℝ) = j'.1 := by
    field_simp at this
    nlinarith [Real.pi_pos]
  exact Subtype.ext (by exact_mod_cast h2)

/-- The permutation of `F_p` induced by multiplication by `u`. -/
def mulPerm (hp : 3 ≤ p) (u : ZMod p) (hu : u ≠ 0) : Freq p ≃ Freq p :=
  Equiv.ofBijective _ ((Finite.injective_iff_bijective).1 (mulFreq_injective hp u hu))

/-- The image of an allocation under `j ↦ uj`. -/
def permVec (hp : 3 ≤ p) (u : ZMod p) (hu : u ≠ 0) (v : Freq p → ℝ) : Freq p → ℝ :=
  v ∘ (mulPerm hp u hu).symm

lemma cover_permVec (hp : 3 ≤ p) (u : ZMod p) (hu : u ≠ 0) (v : Freq p → ℝ) (δ : Off p) :
    cover (covP p) (permVec hp u hu v) δ = cover (covP p) v (mulOff u hu δ) := by
  unfold cover permVec
  rw [← Equiv.sum_comp (mulPerm hp u hu)]
  refine Finset.sum_congr rfl fun j _ => ?_
  simp only [Function.comp_apply, Equiv.symm_apply_apply]
  congr 1
  exact covP_mulFreq hp u hu δ j

lemma mem_Poly_permVec (hp : 3 ≤ p) (u : ZMod p) (hu : u ≠ 0) {v : Freq p → ℝ} :
    permVec hp u hu v ∈ PolyP p ↔ v ∈ PolyP p := by
  have hbij := mulOff_bijective u hu
  constructor
  · rintro ⟨h0, h1⟩
    refine ⟨fun j => ?_, fun δ => ?_⟩
    · have := h0 (mulPerm hp u hu j)
      simpa [permVec] using this
    · obtain ⟨δ', rfl⟩ := hbij.2 δ
      rw [← cover_permVec]; exact h1 δ'
  · rintro ⟨h0, h1⟩
    exact ⟨fun j => h0 _, fun δ => by rw [cover_permVec]; exact h1 _⟩

lemma permVec_add_smul (hp : 3 ≤ p) (u : ZMod p) (hu : u ≠ 0) (a b : ℝ) (v w : Freq p → ℝ) :
    permVec hp u hu (a • v + b • w) = a • permVec hp u hu v + b • permVec hp u hu w := rfl

/-- The inverse image. -/
def unpermVec (hp : 3 ≤ p) (u : ZMod p) (hu : u ≠ 0) (v : Freq p → ℝ) : Freq p → ℝ :=
  v ∘ (mulPerm hp u hu)

lemma permVec_unpermVec (hp : 3 ≤ p) (u : ZMod p) (hu : u ≠ 0) (v : Freq p → ℝ) :
    permVec hp u hu (unpermVec hp u hu v) = v := by
  funext j; simp [permVec, unpermVec]

lemma unpermVec_mem (hp : 3 ≤ p) (u : ZMod p) (hu : u ≠ 0) {v : Freq p → ℝ} (hv : v ∈ PolyP p) :
    unpermVec hp u hu v ∈ PolyP p := by
  rw [← mem_Poly_permVec hp u hu, permVec_unpermVec]; exact hv

/-- **`prop:shapes`, parenthetical.** The image of a vertex under `j ↦ uj mod p` is a vertex. -/
theorem isVertex_permVec (hp : 3 ≤ p) (u : ZMod p) (hu : u ≠ 0) {v : Freq p → ℝ}
    (hv : IsVertex (covP p) v) : IsVertex (covP p) (permVec hp u hu v) := by
  rw [IsVertex, mem_extremePoints] at hv ⊢
  obtain ⟨hvP, hext⟩ := hv
  refine ⟨(mem_Poly_permVec hp u hu).2 hvP, fun x₁ hx₁ x₂ hx₂ hseg => ?_⟩
  have hx₁' := unpermVec_mem hp u hu hx₁
  have hx₂' := unpermVec_mem hp u hu hx₂
  obtain ⟨a, b, ha, hb, hab, he⟩ := hseg
  have hseg' : v ∈ openSegment ℝ (unpermVec hp u hu x₁) (unpermVec hp u hu x₂) := by
    refine ⟨a, b, ha, hb, hab, ?_⟩
    funext j
    have := congrFun he (mulPerm hp u hu j)
    simpa [permVec, unpermVec] using this
  obtain ⟨h1, h2⟩ := hext _ hx₁' _ hx₂' hseg'
  constructor
  · rw [← permVec_unpermVec hp u hu x₁, h1]
  · rw [← permVec_unpermVec hp u hu x₂, h2]

/-- ... and has the same multiset of non-zero entries, hence the same `∑_j v_j^{1/s}` for every
`s` (the two vertices are always tied). -/
theorem multiset_permVec (hp : 3 ≤ p) (u : ZMod p) (hu : u ≠ 0) (v : Freq p → ℝ) :
    (Finset.univ.val.map (permVec hp u hu v)).filter (· ≠ 0) =
      (Finset.univ.val.map v).filter (· ≠ 0) := by
  congr 1
  unfold permVec
  rw [← Multiset.map_map, Multiset.map_univ_val_equiv]

theorem budget_permVec (hp : 3 ≤ p) (u : ZMod p) (hu : u ≠ 0) (v : Freq p → ℝ) (s : ℝ) :
    budget s (permVec hp u hu v) = budget s v := by
  unfold budget permVec
  exact Equiv.sum_comp (mulPerm hp u hu).symm (fun j => v j ^ (1 / s))

end Heracles
