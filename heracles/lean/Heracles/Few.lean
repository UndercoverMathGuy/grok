import Heracles.Scaling
import Heracles.Approach

/-!
# Corollary `cor:few` (few features cannot be optimal)

Paper, Section `sec:cyclic`, proof at the end of appendix `app:scaling`.
-/

open Real Finset Filter Topology

noncomputable section

set_option linter.unusedSectionVars false

namespace Heracles

variable {p : ℕ} [Fact p.Prime]

/-- The function `g_s(x) = (x / (e^{x/(2 a_s)} - 1))^s` of (Kbound). -/
def gK (s x : ℝ) : ℝ := (x / (Real.exp (x / (2 * aS s)) - 1)) ^ s

/-- (Kbound) applied to an arbitrary non-negative amplitude vector: if `h ≤ ∑_j A_j c_j(δ)` for
every `δ ≠ 0` and `A` has at most `K ≥ 1` non-zero entries, then
`h ≤ ‖A‖_{1/s} · (3/2)(ln p)^{1-s} g_s(ln p / K)`, where `‖A‖_{1/s} = (∑_j A_j^{1/s})^s`. -/
theorem few_bound (hp : 3 ≤ p) {s : ℝ} (hs : 1 < s) (A : Freq p → ℝ) (hA0 : ∀ j, 0 ≤ A j)
    {K : ℕ} (hK0 : 0 < K) (hK : (supp A).card ≤ K) (h : ℝ)
    (hh : ∀ δ : Off p, h ≤ cover (covP p) A δ) :
    h ≤ (∑ j, A j ^ (1 / s)) ^ s *
      ((3 / 2) * Real.log p ^ (1 - s) * gK s (Real.log p / K)) := by
  have hs0 : 0 < s := by linarith
  have hlog := log_p_pos hp
  have hKR : (0 : ℝ) < K := by exact_mod_cast hK0
  have hx0 : 0 < Real.log p / K := by positivity
  have hE1 : 1 < Real.exp (Real.log p / K / (2 * aS s)) :=
    Real.one_lt_exp_iff.2 (by unfold aS; positivity)
  have hQ : 0 ≤ (3 / 2) * Real.log p ^ (1 - s) * gK s (Real.log p / K) := by
    unfold gK
    have : 0 ≤ Real.log p / K / (Real.exp (Real.log p / K / (2 * aS s)) - 1) :=
      div_nonneg hx0.le (by linarith)
    positivity
  have hworst : h ≤ worst (covP p) A := le_worst_iff.2 hh
  by_cases hA : A = 0
  · have : worst (covP p) A = 0 := by rw [hA]; exact worst_zero _
    rw [this] at hworst
    exact le_trans hworst (mul_nonneg (Real.rpow_nonneg (Finset.sum_nonneg fun j _ =>
      Real.rpow_nonneg (hA0 j) _) _) hQ)
  · set S := ∑ j, A j ^ (1 / s) with hS
    have hSpos : 0 < S := budget_pos_of_ne hs0.ne' hA0 hA
    set a := S ^ (-s) with ha
    have ha0 : 0 < a := Real.rpow_pos_of_pos hSpos _
    have hfeas : Feasible s (a • A) := by
      refine ⟨fun k => by simp only [Pi.smul_apply, smul_eq_mul]; exact mul_nonneg ha0.le (hA0 k),
        ?_⟩
      rw [budget_smul ha0.le hA0]
      have : budget s A = S := rfl
      rw [this, ha, ← Real.rpow_mul hSpos.le]
      have e : -s * (1 / s) = -1 := by field_simp
      rw [e, Real.rpow_neg_one, inv_mul_cancel₀ hSpos.ne']
    have hsupp : supp (a • A) = supp A := by
      ext j; simp [supp, ha0.ne']
    have hKb := Kbound hp hs hfeas hK0 (by rw [hsupp]; exact hK)
    rw [worst_smul _ ha0.le] at hKb
    have hSs : S ^ s * a = 1 := by
      rw [ha, ← Real.rpow_add hSpos]; simp
    calc h ≤ worst (covP p) A := hworst
      _ = S ^ s * (a * worst (covP p) A) := by rw [← mul_assoc, hSs, one_mul]
      _ ≤ S ^ s * ((3 / 2) * Real.log p ^ (1 - s) * gK s (Real.log p / K)) :=
          mul_le_mul_of_nonneg_left hKb (Real.rpow_nonneg hSpos.le _)

/-- The amplitude budget of a network (the first step of the proof of `thm:universal`):
`∑_j A_j^{b/ν} ≤ C ∑_i N(θ_i)^b`. -/
theorem amp_budget {n : ℕ} {Ω : Type*} (φ : Ω → Table p n) (N : Ω → ℝ)
    {b ν C : ℝ} (hb : 0 < b) (hbν : b ≤ ν)
    (hbudget : ∀ ω, ∑ j : Freq p, ‖beta (φ ω) j.1‖ ^ (b / ν) ≤ C * N ω ^ b)
    {m : ℕ} (θ : Fin m → Ω) :
    ∑ j : Freq p, amp (netTable φ θ) j.1 ^ (b / ν) ≤ C * ∑ i, N (θ i) ^ b := by
  have hν : 0 < ν := lt_of_lt_of_le hb hbν
  have hr0 : 0 < b / ν := div_pos hb hν
  have hr1 : b / ν ≤ 1 := (div_le_one hν).2 hbν
  calc ∑ j : Freq p, amp (netTable φ θ) j.1 ^ (b / ν)
      ≤ ∑ j : Freq p, ∑ i, ‖beta (φ (θ i)) j.1‖ ^ (b / ν) := by
        refine Finset.sum_le_sum fun j _ => ?_
        unfold amp
        rw [beta_netTable]
        calc ‖∑ i, beta (φ (θ i)) j.1‖ ^ (b / ν) ≤ (∑ i, ‖beta (φ (θ i)) j.1‖) ^ (b / ν) :=
              Real.rpow_le_rpow (norm_nonneg _) (norm_sum_le _ _) hr0.le
          _ ≤ ∑ i, ‖beta (φ (θ i)) j.1‖ ^ (b / ν) :=
              rpow_sum_le_sum_rpow' _ _ (fun i => norm_nonneg _) hr0 hr1
    _ = ∑ i, ∑ j : Freq p, ‖beta (φ (θ i)) j.1‖ ^ (b / ν) := Finset.sum_comm
    _ ≤ ∑ i, C * N (θ i) ^ b := Finset.sum_le_sum fun i _ => hbudget (θ i)
    _ = C * ∑ i, N (θ i) ^ b := (Finset.mul_sum _ _ _).symm

/-- **Corollary `cor:few`.** Let `θ` be any network (any neuron model satisfying the
single-neuron budget with constant `C`, any width, interference allowed) whose amplitudes `A_j`
are non-zero on at most `K ≥ 1` frequencies, and `x = ln p / K`.  Then
`h(θ) ≤ C^{ν/b} ‖θ‖_{2,b}^ν · (3/2) (x/(e^{x/(2a_s)} - 1))^s (ln p)^{1-s}` with `s = ν/b`,
i.e. the paper's bound on `h(θ)/‖θ‖_{2,b}^ν`. -/
theorem cor_few (hp : 3 ≤ p) {n : ℕ} {Ω : Type*} (φ : Ω → Table p n) (N : Ω → ℝ)
    {b ν C : ℝ} (hb : 0 < b) (hbν : b < ν) (hC : 0 ≤ C) (hN : ∀ ω, 0 ≤ N ω)
    (hbudget : ∀ ω, ∑ j : Freq p, ‖beta (φ ω) j.1‖ ^ (b / ν) ≤ C * N ω ^ b)
    {m : ℕ} (θ : Fin m → Ω) {K : ℕ} (hK0 : 0 < K)
    (hK : (Finset.univ.filter fun j : Freq p => amp (netTable φ θ) j.1 ≠ 0).card ≤ K) :
    margin (netTable φ θ) ≤ C ^ (ν / b) * (∑ i, N (θ i) ^ b) ^ (ν / b) *
      ((3 / 2) * gK (ν / b) (Real.log p / K) * Real.log p ^ (1 - ν / b)) := by
  have hν : 0 < ν := lt_of_lt_of_le hb hbν.le
  have hs : 1 < ν / b := (one_lt_div hb).2 hbν
  have hrs : 1 / (ν / b) = b / ν := one_div_div ν b
  set A : Freq p → ℝ := fun j => amp (netTable φ θ) j.1 with hAdef
  have hA0 : ∀ j, 0 ≤ A j := fun j => norm_nonneg _
  have hh : ∀ δ : Off p, margin (netTable φ θ) ≤ cover (covP p) A δ := by
    intro δ
    have := (lem_phase hp (netTable φ θ) δ.1 δ.2)
    exact this.1.trans this.2
  have hKA : (supp A).card ≤ K := by
    have : supp A = Finset.univ.filter fun j : Freq p => amp (netTable φ θ) j.1 ≠ 0 := rfl
    rw [this]; exact hK
  have hfb := few_bound hp hs A hA0 hK0 hKA _ hh
  rw [hrs] at hfb
  have hAb := amp_budget φ N hb hbν.le hbudget θ
  have hS0 : 0 ≤ ∑ j, A j ^ (b / ν) := Finset.sum_nonneg fun j _ => Real.rpow_nonneg (hA0 j) _
  have hNb : 0 ≤ ∑ i, N (θ i) ^ b := Finset.sum_nonneg fun i _ => Real.rpow_nonneg (hN _) _
  have hlog := log_p_pos hp
  have hKR : (0 : ℝ) < K := by exact_mod_cast hK0
  have hQ : 0 ≤ (3 / 2) * Real.log p ^ (1 - ν / b) * gK (ν / b) (Real.log p / K) := by
    unfold gK
    have hE1 : 1 < Real.exp (Real.log p / K / (2 * aS (ν / b))) :=
      Real.one_lt_exp_iff.2 (by unfold aS; positivity)
    have : 0 ≤ Real.log p / K / (Real.exp (Real.log p / K / (2 * aS (ν / b))) - 1) :=
      div_nonneg (by positivity) (by linarith)
    positivity
  calc margin (netTable φ θ)
      ≤ (∑ j, A j ^ (b / ν)) ^ (ν / b) *
          ((3 / 2) * Real.log p ^ (1 - ν / b) * gK (ν / b) (Real.log p / K)) := hfb
    _ ≤ (C * ∑ i, N (θ i) ^ b) ^ (ν / b) *
          ((3 / 2) * Real.log p ^ (1 - ν / b) * gK (ν / b) (Real.log p / K)) :=
        mul_le_mul_of_nonneg_right (Real.rpow_le_rpow hS0 hAb (by positivity)) hQ
    _ = C ^ (ν / b) * (∑ i, N (θ i) ^ b) ^ (ν / b) *
          ((3 / 2) * gK (ν / b) (Real.log p / K) * Real.log p ^ (1 - ν / b)) := by
        rw [Real.mul_rpow hC hNb]; ring

/-- `x / (e^{x/(2a)} - 1) → 0` as `x → ∞` (for `a > 0`), hence `g_s(x) → 0`: the ratio bound
`3 (9ν)^{s-1} g_s(x)` of `cor:few` tends to `0`, so `K = o(log p)` attains a vanishing fraction. -/
theorem tendsto_gK {s : ℝ} (hs : 1 < s) : Tendsto (gK s) atTop (𝓝 0) := by
  have ha : 0 < aS s := by unfold aS; linarith
  set c := 1 / (2 * aS s) with hc
  have hc0 : 0 < c := by positivity
  -- `0 ≤ x/(e^{cx}-1) ≤ 2/(c² x)` for `x > 0`
  have hle : ∀ x : ℝ, 0 < x → x / (Real.exp (x / (2 * aS s)) - 1) ≤ 2 / (c ^ 2 * x) := by
    intro x hx
    have hz : x / (2 * aS s) = c * x := by rw [hc]; field_simp
    rw [hz]
    have h1 := exp_sub_one_ge_sq (z := c * x) (by positivity)
    have h2 : 0 < (c * x) ^ 2 / 2 := by positivity
    rw [div_le_div_iff₀ (lt_of_lt_of_le h2 h1) (by positivity)]
    nlinarith
  have hnn : ∀ x : ℝ, 0 < x → 0 ≤ x / (Real.exp (x / (2 * aS s)) - 1) := by
    intro x hx
    have : 1 < Real.exp (x / (2 * aS s)) := Real.one_lt_exp_iff.2 (by positivity)
    exact div_nonneg hx.le (by linarith)
  have hlim1 : Tendsto (fun x : ℝ => x / (Real.exp (x / (2 * aS s)) - 1)) atTop (𝓝 0) := by
    have hup : Tendsto (fun x : ℝ => 2 / (c ^ 2 * x)) atTop (𝓝 0) := by
      have : Tendsto (fun x : ℝ => c ^ 2 * x) atTop atTop :=
        Tendsto.const_mul_atTop (by positivity) tendsto_id
      have h := this.inv_tendsto_atTop.const_mul 2
      rw [mul_zero] at h
      refine h.congr fun x => ?_
      simp only [Pi.inv_apply, div_eq_mul_inv]
    refine tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds hup ?_ ?_
    · filter_upwards [eventually_gt_atTop 0] with x hx using hnn x hx
    · filter_upwards [eventually_gt_atTop 0] with x hx using hle x hx
  have hcont : ContinuousAt (fun t : ℝ => t ^ s) 0 :=
    Real.continuousAt_rpow_const _ _ (Or.inr (by linarith))
  have := hcont.tendsto.comp hlim1
  rw [Real.zero_rpow (by linarith : s ≠ 0)] at this
  exact this

end Heracles
