import Heracles.Dense
import Heracles.Few

/-!
# `cor:dense`, network sentence: at `b = ν` an optimal network has equal amplitudes
-/

open Real Finset

noncomputable section

namespace Heracles

variable {p n : ℕ} [Fact p.Prime]

/-- **`cor:dense`, network part.** At `b = ν`, any network (generic neuron model with
single-neuron budget constant `C`) attaining the bound of `thm:universal`,
`h(θ) = C Γ_1(p) ∑_i N(θ_i)^ν > 0`, has equal amplitude on every frequency:
`A_j = (∑_k A_k) · 2/(p-1)` for all `j ∈ F_p`. -/
theorem dense_network_equal_amp (hp : 3 ≤ p) {Ω : Type*} (φ : Ω → Table p n) (N : Ω → ℝ)
    {ν C : ℝ} (hν : 0 < ν)
    (hbudget : ∀ ω, ∑ j : Freq p, ‖beta (φ ω) j.1‖ ^ (ν / ν) ≤ C * N ω ^ ν)
    {m : ℕ} (θ : Fin m → Ω)
    (hatt : margin (netTable φ θ) = C * GammaP p 1 * ∑ i, N (θ i) ^ ν)
    (hpos : 0 < margin (netTable φ θ)) :
    ∀ j : Freq p, amp (netTable φ θ) j.1 =
      (∑ k : Freq p, amp (netTable φ θ) k.1) * (2 / ((p : ℝ) - 1)) := by
  set A : Freq p → ℝ := fun j => amp (netTable φ θ) j.1 with hAdef
  have hA0 : ∀ j, 0 ≤ A j := fun j => norm_nonneg _
  set T := ∑ k, A k with hT
  have hbud := amp_budget φ N hν le_rfl hbudget θ
  simp only [div_self hν.ne', Real.rpow_one] at hbud
  -- `h ≤ worst A`
  have hworstA : margin (netTable φ θ) ≤ worst (covP p) A := by
    rw [le_worst_iff]
    intro δ
    have := lem_phase hp (netTable φ θ) δ.1 δ.2
    exact this.1.trans this.2
  have hΓ : 0 < GammaP p 1 := GammaP_pos p hp le_rfl
  have hCN : 0 < C * ∑ i, N (θ i) ^ ν := by
    by_contra h
    push_neg at h
    have : margin (netTable φ θ) ≤ 0 := by
      rw [hatt, mul_comm C, mul_assoc]; exact mul_nonpos_of_nonneg_of_nonpos hΓ.le h
    linarith
  have hTpos : 0 < T := by
    by_contra h
    push_neg at h
    have hA : A = 0 := by
      funext j
      have := (Finset.sum_eq_zero_iff_of_nonneg fun k _ => hA0 k).1
        (le_antisymm h (Finset.sum_nonneg fun k _ => hA0 k)) j (Finset.mem_univ j)
      exact this
    rw [hA, worst_zero] at hworstA
    linarith
  set y := T⁻¹ • A with hy
  have hTi : 0 < T⁻¹ := inv_pos.2 hTpos
  have hfeas : Feasible 1 y := by
    refine ⟨fun k => by simp only [hy, Pi.smul_apply, smul_eq_mul]; exact mul_nonneg hTi.le (hA0 k),
      ?_⟩
    unfold budget
    simp only [div_one, Real.rpow_one, hy, Pi.smul_apply, smul_eq_mul, ← Finset.mul_sum]
    rw [← hT, inv_mul_cancel₀ hTpos.ne']
  have hwy : GammaP p 1 ≤ worst (covP p) y := by
    rw [hy, worst_smul _ hTi.le]
    have h1 : C * GammaP p 1 * ∑ i, N (θ i) ^ ν ≤ worst (covP p) A := hatt ▸ hworstA
    have h2 : T ≤ C * ∑ i, N (θ i) ^ ν := hbud
    have h3 : T * GammaP p 1 ≤ worst (covP p) A := by nlinarith
    rw [le_inv_mul_iff₀ hTpos]
    linarith
  have hmax : IsMaximizer (covP p) 1 y :=
    (isMaximizer_iff (covP_le_two p) le_rfl).2
      ⟨hfeas, le_antisymm (le_Gam (covP_le_two p) le_rfl hfeas) hwy⟩
  have hflat := (dense_maximizer_iff p hp y).1 hmax
  intro j
  have := congrFun hflat j
  simp only [hy, Pi.smul_apply, smul_eq_mul] at this
  have e : A j = T * (T⁻¹ * A j) := by field_simp
  rw [show amp (netTable φ θ) j.1 = A j from rfl, e, this]

end Heracles
