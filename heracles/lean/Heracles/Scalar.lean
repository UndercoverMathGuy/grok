import Heracles.Defs

/-!
# The scalar bound on `ψ(z) = z - ln I₀(z)` (appendix `app:scaling`, "A scalar bound")

Paper claim: for `z ≥ 0` (and `s > 1`, `r = 1/s`, `a_s = 3 + s/2`),
`0 ≤ ψ(z) ≤ a_s ln(1 + z^r)`.

Here `I0 z = (1/π) ∫_0^π e^{z cos θ} dθ` (defined in `Heracles.Defs`).
-/

open Real

noncomputable section

namespace Heracles

lemma continuous_I0_integrand (z : ℝ) : Continuous fun θ : ℝ => Real.exp (z * Real.cos θ) := by
  fun_prop

/-- `∫_0^π e^{z cos θ} dθ ≥ π` (from `e^x ≥ 1 + x` and `∫_0^π cos = 0`). -/
lemma pi_le_integral_I0 (z : ℝ) : π ≤ ∫ θ in (0 : ℝ)..π, Real.exp (z * Real.cos θ) := by
  have h1 : ∫ θ in (0 : ℝ)..π, (1 + z * Real.cos θ) = π := by
    rw [intervalIntegral.integral_add, intervalIntegral.integral_const_mul, integral_cos]
    · simp
    · exact intervalIntegrable_const
    · exact (continuous_const.mul continuous_cos).intervalIntegrable _ _
  calc π = ∫ θ in (0 : ℝ)..π, (1 + z * Real.cos θ) := h1.symm
    _ ≤ _ := ?_
  apply intervalIntegral.integral_mono_on Real.pi_pos.le
  · exact (continuous_const.add (continuous_const.mul continuous_cos)).intervalIntegrable _ _
  · exact (continuous_I0_integrand z).intervalIntegrable _ _
  · intro θ _
    have := Real.add_one_le_exp (z * Real.cos θ)
    linarith

/-- `I₀(z) ≥ 1` for every real `z`. -/
theorem one_le_I0 (z : ℝ) : 1 ≤ I0 z := by
  unfold I0
  have := pi_le_integral_I0 z
  rw [one_div, ← div_eq_inv_mul, le_div_iff₀ Real.pi_pos]
  linarith

theorem I0_pos (z : ℝ) : 0 < I0 z := lt_of_lt_of_le one_pos (one_le_I0 z)

/-- `I₀(z) ≤ e^z` for `z ≥ 0`. -/
theorem I0_le_exp {z : ℝ} (hz : 0 ≤ z) : I0 z ≤ Real.exp z := by
  unfold I0
  have h : ∫ θ in (0 : ℝ)..π, Real.exp (z * Real.cos θ) ≤ ∫ θ in (0 : ℝ)..π, Real.exp z := by
    apply intervalIntegral.integral_mono_on Real.pi_pos.le
    · exact (continuous_I0_integrand z).intervalIntegrable _ _
    · exact intervalIntegrable_const
    · intro θ _
      apply Real.exp_le_exp.2
      have := mul_le_mul_of_nonneg_left (Real.cos_le_one θ) hz
      linarith
  rw [intervalIntegral.integral_const, smul_eq_mul, sub_zero] at h
  rw [one_div, ← div_eq_inv_mul, div_le_iff₀ Real.pi_pos]
  linarith

/-- For `z ≥ 1`, `I₀(z) ≥ e^{z - 1/2} / (π √z)`: integrate only over `[0, z^{-1/2}]` and use
`cos θ ≥ 1 - θ²/2`. -/
theorem I0_ge_of_one_le {z : ℝ} (hz : 1 ≤ z) :
    Real.exp (z - 1 / 2) / (π * Real.sqrt z) ≤ I0 z := by
  set a := 1 / Real.sqrt z with ha
  have hsq : 0 < Real.sqrt z := Real.sqrt_pos.2 (by linarith)
  have hsq1 : 1 ≤ Real.sqrt z := by
    rw [show (1 : ℝ) = Real.sqrt 1 by simp]
    exact Real.sqrt_le_sqrt hz
  have ha0 : 0 ≤ a := by positivity
  have ha1 : a ≤ π := by
    have : a ≤ 1 := by rw [ha, div_le_one hsq]; exact hsq1
    linarith [Real.pi_gt_three]
  have hsub : ∫ θ in (0 : ℝ)..a, Real.exp (z * Real.cos θ) ≤
      ∫ θ in (0 : ℝ)..π, Real.exp (z * Real.cos θ) := by
    apply intervalIntegral.integral_mono_interval le_rfl ha0 ha1
    · exact Filter.Eventually.of_forall fun θ => (Real.exp_pos _).le
    · exact (continuous_I0_integrand z).intervalIntegrable _ _
  have ha2 : a ^ 2 = 1 / z := by
    rw [ha, div_pow, Real.sq_sqrt (by linarith)]
    ring
  have hlow : ∫ θ in (0 : ℝ)..a, Real.exp (z - 1 / 2) ≤
      ∫ θ in (0 : ℝ)..a, Real.exp (z * Real.cos θ) := by
    apply intervalIntegral.integral_mono_on ha0
    · exact intervalIntegrable_const
    · exact (continuous_I0_integrand z).intervalIntegrable _ _
    · intro θ hθ
      apply Real.exp_le_exp.2
      have hc := Real.one_sub_sq_div_two_le_cos (x := θ)
      have hθa : θ ^ 2 ≤ a ^ 2 := pow_le_pow_left₀ hθ.1 hθ.2 2
      have hzθ : z * θ ^ 2 ≤ 1 := by
        calc z * θ ^ 2 ≤ z * a ^ 2 := mul_le_mul_of_nonneg_left hθa (by linarith)
          _ = 1 := by rw [ha2]; field_simp
      have hzc := mul_le_mul_of_nonneg_left hc (by linarith : (0 : ℝ) ≤ z)
      nlinarith
  rw [intervalIntegral.integral_const, smul_eq_mul, sub_zero] at hlow
  unfold I0
  have e : Real.exp (z - 1 / 2) / (π * Real.sqrt z) = (1 / π) * (a * Real.exp (z - 1 / 2)) := by
    rw [ha]
    field_simp
  rw [e]
  apply mul_le_mul_of_nonneg_left _ (by positivity)
  linarith

/-- `ψ(z) ≥ 0` for `z ≥ 0` (because `I₀(z) ≤ e^z`). -/
theorem psi_nonneg {z : ℝ} (hz : 0 ≤ z) : 0 ≤ psi z := by
  unfold psi
  have := Real.log_le_log (I0_pos z) (I0_le_exp hz)
  rw [Real.log_exp] at this
  linarith

/-- `ψ(z) ≤ z` (because `I₀ ≥ 1`). -/
theorem psi_le_self (z : ℝ) : psi z ≤ z := by
  unfold psi
  have := Real.log_nonneg (one_le_I0 z)
  linarith

theorem psi_zero : psi 0 = 0 := by
  unfold psi I0
  simp [Real.pi_ne_zero]

/-- `ln(1 + x) ≥ x/2` on `[0,1]`. -/
lemma half_le_log_one_add {x : ℝ} (hx0 : 0 ≤ x) (hx1 : x ≤ 1) : x / 2 ≤ Real.log (1 + x) := by
  have h := Real.one_sub_inv_le_log_of_pos (by linarith : (0 : ℝ) < 1 + x)
  have e : 1 - (1 + x)⁻¹ = x / (1 + x) := by
    field_simp
    ring
  have : x / 2 ≤ x / (1 + x) := by
    rw [div_le_div_iff₀ (by norm_num) (by linarith)]
    nlinarith
  linarith

/-- **The scalar bound** (appendix `app:scaling`): for `s > 1` and `z ≥ 0`,
`ψ(z) ≤ a_s ln(1 + z^{1/s})` with `a_s = 3 + s/2`. -/
theorem psi_le {s z : ℝ} (hs : 1 < s) (hz : 0 ≤ z) :
    psi z ≤ aS s * Real.log (1 + z ^ (1 / s)) := by
  have hs0 : 0 < s := by linarith
  have hr0 : 0 < 1 / s := by positivity
  have hsplit : aS s * Real.log (1 + z ^ (1 / s)) =
      3 * Real.log (1 + z ^ (1 / s)) + s / 2 * Real.log (1 + z ^ (1 / s)) := by
    unfold aS; ring
  rcases le_total z 1 with hz1 | hz1
  · -- `z ≤ 1`: `ψ(z) ≤ z ≤ z^r ≤ 2 ln(1 + z^r) ≤ a_s ln(1 + z^r)`
    have hzr : z ≤ z ^ (1 / s) := le_rpow_one_div hs.le hz hz1
    have hzr1 : z ^ (1 / s) ≤ 1 := Real.rpow_le_one hz hz1 hr0.le
    have hzr0 : 0 ≤ z ^ (1 / s) := Real.rpow_nonneg hz _
    have hlog := half_le_log_one_add hzr0 hzr1
    have hlog0 : 0 ≤ Real.log (1 + z ^ (1 / s)) := Real.log_nonneg (by linarith)
    have := psi_le_self z
    have : 0 ≤ s / 2 * Real.log (1 + z ^ (1 / s)) := by positivity
    rw [hsplit]
    linarith
  · -- `z ≥ 1`: `ψ(z) ≤ ln π + 1/2 + (1/2) ln z`
    have hz0 : 0 < z := by linarith
    have hI := I0_ge_of_one_le hz1
    have hpos : 0 < Real.exp (z - 1 / 2) / (π * Real.sqrt z) := by positivity
    have hlogI := Real.log_le_log hpos hI
    have hexpand : Real.log (Real.exp (z - 1 / 2) / (π * Real.sqrt z)) =
        z - 1 / 2 - Real.log π - Real.log z / 2 := by
      rw [Real.log_div (Real.exp_pos _).ne' (by positivity), Real.log_exp,
        Real.log_mul Real.pi_ne_zero (Real.sqrt_pos.2 hz0).ne', Real.log_sqrt hz0.le]
      ring
    have hpsi : psi z ≤ 1 / 2 + Real.log π + Real.log z / 2 := by
      unfold psi
      rw [hexpand] at hlogI
      linarith
    have hzr1 : 1 ≤ z ^ (1 / s) := Real.one_le_rpow hz1 hr0.le
    have hlog2 : Real.log 2 ≤ Real.log (1 + z ^ (1 / s)) :=
      Real.log_le_log (by norm_num) (by linarith)
    -- `3 ln 2 ≥ ln π + 1/2`, from `π ≤ 4` and `1/2 ≤ ln 2`
    have hconst : 1 / 2 + Real.log π ≤ 3 * Real.log 2 := by
      have h1 : Real.log π ≤ Real.log 4 := Real.log_le_log Real.pi_pos Real.pi_le_four
      have h4 : Real.log 4 = 2 * Real.log 2 := by
        rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.log_pow]
        norm_num
      have h2 : 1 / 2 ≤ Real.log 2 := by
        have := Real.log_two_gt_d9
        linarith
      linarith
    -- `(s/2) ln(1 + z^r) ≥ (s/2) ln(z^r) = (1/2) ln z`
    have hlogz : Real.log z / 2 ≤ s / 2 * Real.log (1 + z ^ (1 / s)) := by
      have h1 : Real.log (z ^ (1 / s)) ≤ Real.log (1 + z ^ (1 / s)) :=
        Real.log_le_log (by positivity) (by linarith)
      rw [Real.log_rpow hz0] at h1
      have e : Real.log z / 2 = s / 2 * (1 / s * Real.log z) := by
        field_simp
      rw [e]
      exact mul_le_mul_of_nonneg_left h1 (by linarith)
    rw [hsplit]
    linarith

end Heracles
