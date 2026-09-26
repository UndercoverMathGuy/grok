import Heracles.Cost

/-!
# Lemma `lem:bernstein` with the paper's exact constant

`Cost.lean` proves the Bernstein representation in scaling form
(`bernstein_scaling : ∫_0^∞ log(1+xt) t^{-1-r} dt = x^r K_r`, `K_r = ∫_0^∞ log(1+u) u^{-1-r} du`).
Here we compute `K_r = π / (r sin(π r))` (`Kr_eq`) and deduce the paper's statement
(`lem_bernstein`): for `0 < r < 1` and `x ≥ 0`,
`x^r = (r sin(π r)/π) ∫_0^∞ log(1+xt) t^{-1-r} dt`.

Route (as in the paper): integrate by parts on `(0,∞)` (boundary terms vanish) to get
`K_r = (1/r) ∫_0^∞ u^{-r}/(1+u) du`; substitute `x = u/(1+u)` to get the Beta integral
`B(1-r, r) = Γ(1-r) Γ(r) = π / sin(π r)` (Mathlib: `Complex.Gamma_mul_Gamma_eq_betaIntegral`,
`Complex.Gamma_mul_Gamma_one_sub`).
-/

open Real MeasureTheory Set Filter Topology

noncomputable section

namespace Heracles
namespace Cost

variable {r : ℝ}

/-- The real Beta integral `∫_0^1 x^{-r}(1-x)^{r-1} dx = B(1-r, r) = π / sin(π r)`. -/
lemma beta_real (hr0 : 0 < r) (hr1 : r < 1) :
    ∫ x in (0 : ℝ)..1, x ^ (-r) * (1 - x) ^ (r - 1) = π / Real.sin (π * r) := by
  have hB := Complex.Gamma_mul_Gamma_eq_betaIntegral (s := ((1 - r : ℝ) : ℂ)) (t := (r : ℂ))
    (by rw [Complex.ofReal_re]; linarith) (by rw [Complex.ofReal_re]; linarith)
  rw [show ((1 - r : ℝ) : ℂ) + (r : ℂ) = 1 by push_cast; ring, Complex.Gamma_one, one_mul] at hB
  have hrefl := Complex.Gamma_mul_Gamma_one_sub (r : ℂ)
  rw [show (1 : ℂ) - (r : ℂ) = ((1 - r : ℝ) : ℂ) by push_cast; ring, mul_comm] at hrefl
  have hint : Complex.betaIntegral ((1 - r : ℝ) : ℂ) (r : ℂ) =
      ((∫ x in (0 : ℝ)..1, x ^ (-r) * (1 - x) ^ (r - 1) : ℝ) : ℂ) := by
    unfold Complex.betaIntegral
    rw [← intervalIntegral.integral_ofReal]
    refine intervalIntegral.integral_congr fun x hx => ?_
    rw [Set.uIcc_of_le zero_le_one] at hx
    have hx0 : 0 ≤ x := hx.1
    have hx1 : 0 ≤ 1 - x := by linarith [hx.2]
    have e1 : ((1 - r : ℝ) : ℂ) - 1 = ((-r : ℝ) : ℂ) := by push_cast; ring
    have e2 : (r : ℂ) - 1 = ((r - 1 : ℝ) : ℂ) := by push_cast; ring
    have e3 : (1 : ℂ) - (x : ℂ) = ((1 - x : ℝ) : ℂ) := by push_cast; ring
    rw [e1, e2, e3, ← Complex.ofReal_cpow hx0, ← Complex.ofReal_cpow hx1, Complex.ofReal_mul]
  rw [hint, hrefl] at hB
  have hsin : ((π / Real.sin (π * r) : ℝ) : ℂ) = (π : ℂ) / Complex.sin ((π : ℂ) * (r : ℂ)) := by
    push_cast; rfl
  rw [← hsin] at hB
  exact_mod_cast hB.symm

/-- The map `u ↦ u/(1+u)` sends `(0,∞)` onto `(0,1)`. -/
lemma image_div_one_add : (fun u : ℝ => u / (1 + u)) '' Ioi 0 = Ioo 0 1 := by
  ext y
  constructor
  · rintro ⟨u, hu, rfl⟩
    have hu : (0 : ℝ) < u := hu
    refine ⟨by positivity, ?_⟩
    rw [div_lt_one (by linarith)]; linarith
  · rintro ⟨hy0, hy1⟩
    refine ⟨y / (1 - y), div_pos hy0 (by linarith), ?_⟩
    have h : (1 : ℝ) - y ≠ 0 := by linarith
    simp only
    field_simp
    ring

lemma injOn_div_one_add : InjOn (fun u : ℝ => u / (1 + u)) (Ioi 0) := by
  intro u hu v hv h
  have hu : (0 : ℝ) < u := hu
  have hv : (0 : ℝ) < v := hv
  simp only at h
  rw [div_eq_div_iff (by linarith) (by linarith)] at h
  linarith

lemma hasDerivAt_div_one_add {u : ℝ} (hu : 0 < u) :
    HasDerivAt (fun u : ℝ => u / (1 + u)) (1 / (1 + u) ^ 2) u := by
  have h := (hasDerivAt_id' u).div ((hasDerivAt_id' u).const_add 1) (by linarith : 1 + u ≠ 0)
  convert h using 1
  field_simp
  ring

/-- `∫_0^∞ u^{-r}/(1+u) du = π / sin(π r)`. -/
lemma integral_rpow_div_one_add (hr0 : 0 < r) (hr1 : r < 1) :
    ∫ u in Ioi (0 : ℝ), u ^ (-r) / (1 + u) = π / Real.sin (π * r) := by
  have hcv := integral_image_eq_integral_abs_deriv_smul (s := Ioi (0 : ℝ))
    (f := fun u : ℝ => u / (1 + u)) (f' := fun u => 1 / (1 + u) ^ 2) measurableSet_Ioi
    (fun u hu => (hasDerivAt_div_one_add hu).hasDerivWithinAt) injOn_div_one_add
    (fun x => x ^ (-r) * (1 - x) ^ (r - 1))
  rw [image_div_one_add] at hcv
  rw [← beta_real hr0 hr1, intervalIntegral.integral_of_le zero_le_one, integral_Ioc_eq_integral_Ioo,
    hcv]
  refine setIntegral_congr_fun measurableSet_Ioi fun u hu => ?_
  have hu : (0 : ℝ) < u := hu
  have h1 : (0 : ℝ) < 1 + u := by linarith
  have hA : (u / (1 + u)) ^ (-r) = u ^ (-r) * (1 + u) ^ r := by
    rw [Real.div_rpow hu.le h1.le, Real.rpow_neg h1.le, div_inv_eq_mul]
  have hB : (1 - u / (1 + u)) ^ (r - 1) = (1 + u) ^ (1 - r) := by
    rw [show 1 - u / (1 + u) = (1 + u)⁻¹ by field_simp; ring, Real.inv_rpow h1.le,
      ← Real.rpow_neg h1.le]
    congr 1; ring
  have hC : (1 + u) ^ r * (1 + u) ^ (1 - r) = 1 + u := by
    rw [← Real.rpow_add h1]; simp
  simp only [smul_eq_mul]
  rw [hA, hB, abs_of_pos (by positivity), mul_assoc, hC]
  field_simp

/-- `u ↦ u^{-r}/(1+u)` is integrable on `(0,∞)` for `0 < r < 1`. -/
lemma integrableOn_rpow_div_one_add (hr0 : 0 < r) (hr1 : r < 1) :
    IntegrableOn (fun u : ℝ => u ^ (-r) / (1 + u)) (Ioi 0) := by
  have hcont : ContinuousOn (fun u : ℝ => u ^ (-r) / (1 + u)) (Ioi 0) := by
    refine ContinuousOn.div ?_ (continuousOn_const.add continuousOn_id) fun x hx => ?_
    · exact ContinuousOn.rpow_const continuousOn_id fun x hx => Or.inl (ne_of_gt hx)
    · have : (0 : ℝ) < x := hx
      exact (by linarith : (0 : ℝ) < 1 + x).ne'
  rw [← Ioc_union_Ioi_eq_Ioi zero_le_one]
  refine IntegrableOn.union ?_ ?_
  · have hint : IntegrableOn (fun u : ℝ => u ^ (-r)) (Ioc 0 1) :=
      (intervalIntegrable_iff_integrableOn_Ioc_of_le zero_le_one).1
        (intervalIntegral.intervalIntegrable_rpow' (by linarith))
    refine hint.mono' ((hcont.mono Ioc_subset_Ioi_self).aestronglyMeasurable measurableSet_Ioc) ?_
    refine ae_restrict_of_forall_mem measurableSet_Ioc fun u hu => ?_
    have hu0 : 0 < u := hu.1
    rw [Real.norm_of_nonneg (by positivity)]
    exact div_le_self (Real.rpow_nonneg hu0.le _) (by linarith)
  · have hint : IntegrableOn (fun u : ℝ => u ^ (-1 - r)) (Ioi 1) :=
      integrableOn_Ioi_rpow_of_lt (by linarith) one_pos
    refine hint.mono' ((hcont.mono fun x (hx : 1 < x) => (show (0:ℝ) < x by linarith)).aestronglyMeasurable
      measurableSet_Ioi) ?_
    refine ae_restrict_of_forall_mem measurableSet_Ioi fun u hu => ?_
    have hu1 : (1 : ℝ) < u := hu
    have hu0 : (0 : ℝ) < u := by linarith
    rw [Real.norm_of_nonneg (by positivity)]
    have e : u ^ (-1 - r) = u ^ (-r) / u := by
      rw [show -1 - r = -r - 1 by ring, Real.rpow_sub_one hu0.ne']
    rw [e]
    exact div_le_div_of_nonneg_left (Real.rpow_nonneg hu0.le _) hu0 (by linarith)

/-- `log(1+t) t^{-r} → 0` as `t → 0⁺`. -/
lemma tendsto_log_mul_rpow_zero (hr1 : r < 1) :
    Tendsto (fun t : ℝ => Real.log (1 + t) * t ^ (-r)) (𝓝[>] 0) (𝓝 0) := by
  have hc : ContinuousAt (fun t : ℝ => t ^ (1 - r)) 0 :=
    Real.continuousAt_rpow_const 0 (1 - r) (Or.inr (by linarith))
  have hup : Tendsto (fun t : ℝ => t ^ (1 - r)) (𝓝[>] 0) (𝓝 0) := by
    have := hc.tendsto
    rw [Real.zero_rpow (by linarith)] at this
    exact this.mono_left nhdsWithin_le_nhds
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds hup ?_ ?_
  · filter_upwards [self_mem_nhdsWithin] with t (ht : 0 < t)
    exact mul_nonneg (Real.log_nonneg (by linarith)) (Real.rpow_nonneg ht.le _)
  · filter_upwards [self_mem_nhdsWithin] with t (ht : 0 < t)
    have hlog : Real.log (1 + t) ≤ t := by
      have := Real.log_le_sub_one_of_pos (by linarith : (0 : ℝ) < 1 + t); linarith
    have e : t ^ (1 - r) = t * t ^ (-r) := by
      rw [show 1 - r = 1 + (-r) by ring, Real.rpow_add ht, Real.rpow_one]
    rw [e]
    exact mul_le_mul_of_nonneg_right hlog (Real.rpow_nonneg ht.le _)

/-- `log(1+t) t^{-r} → 0` as `t → ∞`. -/
lemma tendsto_log_mul_rpow_top (hr0 : 0 < r) :
    Tendsto (fun t : ℝ => Real.log (1 + t) * t ^ (-r)) atTop (𝓝 0) := by
  have hup : Tendsto (fun t : ℝ => (2 / r) * (1 + 1) ^ (r / 2) * t ^ (-(r / 2))) atTop (𝓝 0) := by
    have := (tendsto_rpow_neg_atTop (by positivity : 0 < r / 2)).const_mul
      ((2 / r) * (1 + 1 : ℝ) ^ (r / 2))
    rwa [mul_zero] at this
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds hup ?_ ?_
  · filter_upwards [eventually_gt_atTop 0] with t ht
    exact mul_nonneg (Real.log_nonneg (by linarith)) (Real.rpow_nonneg ht.le _)
  · filter_upwards [eventually_ge_atTop 1] with t ht
    have ht0 : (0 : ℝ) < t := by linarith
    have hb := bI_le_near_top hr0 (x := 1) zero_le_one ht
    unfold bI at hb
    rw [one_mul] at hb
    have e1 : Real.log (1 + t) * t ^ (-r) = t * (Real.log (1 + t) * t ^ (-1 - r)) := by
      rw [show -r = 1 + (-1 - r) by ring, Real.rpow_add ht0, Real.rpow_one]; ring
    have e2 : (2 / r) * (1 + 1) ^ (r / 2) * t ^ (-(r / 2)) =
        t * ((2 / r) * (1 + 1) ^ (r / 2) * t ^ (-1 - r / 2)) := by
      rw [show -(r / 2) = 1 + (-1 - r / 2) by ring, Real.rpow_add ht0, Real.rpow_one]; ring
    rw [e1, e2]
    exact mul_le_mul_of_nonneg_left hb ht0.le

/-- Integration by parts: `K_r = (1/r) ∫_0^∞ u^{-r}/(1+u) du`. -/
lemma Kr_eq_integral (hr0 : 0 < r) (hr1 : r < 1) :
    Kr r = (1 / r) * ∫ u in Ioi (0 : ℝ), u ^ (-r) / (1 + u) := by
  have hu : ∀ x ∈ Ioi (0 : ℝ), HasDerivAt (fun t : ℝ => Real.log (1 + t)) (1 / (1 + x)) x := by
    intro x hx
    have hx : (0 : ℝ) < x := hx
    have := ((hasDerivAt_id' x).const_add 1).log (by linarith : (1 : ℝ) + x ≠ 0)
    simpa using this
  have hv : ∀ x ∈ Ioi (0 : ℝ), HasDerivAt (fun t : ℝ => -(t ^ (-r)) / r) (x ^ (-1 - r)) x := by
    intro x hx
    have hx : (0 : ℝ) < x := hx
    have := ((Real.hasDerivAt_rpow_const (p := -r) (Or.inl hx.ne')).neg).div_const r
    convert this using 1
    rw [show -r - 1 = -1 - r by ring]
    field_simp
  have hfun1 : (fun t : ℝ => Real.log (1 + t)) * (fun t : ℝ => t ^ (-1 - r)) = bI r 1 := by
    funext t; simp [bI]
  have hfun2 : (fun t : ℝ => 1 / (1 + t)) * (fun t : ℝ => -(t ^ (-r)) / r) =
      fun t => (-(1 / r)) * (t ^ (-r) / (1 + t)) := by
    funext t; simp only [Pi.mul_apply]; ring
  have huv' : IntegrableOn ((fun t : ℝ => Real.log (1 + t)) * (fun t : ℝ => t ^ (-1 - r)))
      (Ioi 0) := by
    rw [hfun1]; exact integrableOn_bI hr0 hr1 zero_le_one
  have hu'v : IntegrableOn ((fun t : ℝ => 1 / (1 + t)) * (fun t : ℝ => -(t ^ (-r)) / r))
      (Ioi 0) := by
    rw [hfun2]; exact (integrableOn_rpow_div_one_add hr0 hr1).const_mul _
  have hprod : (fun t : ℝ => Real.log (1 + t)) * (fun t : ℝ => -(t ^ (-r)) / r) =
      fun t => (-(1 / r)) * (Real.log (1 + t) * t ^ (-r)) := by
    funext t; simp only [Pi.mul_apply]; ring
  have h0 : Tendsto ((fun t : ℝ => Real.log (1 + t)) * (fun t : ℝ => -(t ^ (-r)) / r))
      (𝓝[>] 0) (𝓝 0) := by
    rw [hprod]
    have := (tendsto_log_mul_rpow_zero hr1).const_mul (-(1 / r))
    rwa [mul_zero] at this
  have hinf : Tendsto ((fun t : ℝ => Real.log (1 + t)) * (fun t : ℝ => -(t ^ (-r)) / r))
      atTop (𝓝 0) := by
    rw [hprod]
    have := (tendsto_log_mul_rpow_top hr0).const_mul (-(1 / r))
    rwa [mul_zero] at this
  have hibp := integral_Ioi_mul_deriv_eq_deriv_mul hu hv huv' hu'v h0 hinf
  have hK : Kr r = ∫ x in Ioi (0 : ℝ), Real.log (1 + x) * x ^ (-1 - r) := by
    unfold Kr bI; simp_rw [one_mul]
  rw [hK, hibp]
  have : ∫ x in Ioi (0 : ℝ), 1 / (1 + x) * (-(x ^ (-r)) / r) =
      ∫ x in Ioi (0 : ℝ), (-(1 / r)) * (x ^ (-r) / (1 + x)) := by
    refine setIntegral_congr_fun measurableSet_Ioi fun x _ => ?_
    ring
  rw [this, integral_const_mul]
  ring

/-- **The Bernstein constant.** `K_r = ∫_0^∞ log(1+u) u^{-1-r} du = π / (r sin(π r))`. -/
theorem Kr_eq (hr0 : 0 < r) (hr1 : r < 1) : Kr r = π / (r * Real.sin (π * r)) := by
  rw [Kr_eq_integral hr0 hr1, integral_rpow_div_one_add hr0 hr1]
  have hs : 0 < Real.sin (π * r) :=
    Real.sin_pos_of_pos_of_lt_pi (by positivity) (by nlinarith [Real.pi_pos])
  field_simp

/-- **Lemma `lem:bernstein`** (paper, appendix `app:cost`). For `0 < r < 1` and `x ≥ 0`,
`x^r = (r sin(π r)/π) ∫_0^∞ log(1+xt) t^{-1-r} dt`. -/
theorem lem_bernstein {x : ℝ} (hr0 : 0 < r) (hr1 : r < 1) (hx : 0 ≤ x) :
    x ^ r = (r * Real.sin (π * r) / π) *
      ∫ t in Set.Ioi (0 : ℝ), Real.log (1 + x * t) * t ^ (-1 - r) := by
  have h := bernstein_scaling hr0 hx
  unfold bI at h
  rw [h, Kr_eq hr0 hr1]
  have hs : 0 < Real.sin (π * r) :=
    Real.sin_pos_of_pos_of_lt_pi (by positivity) (by nlinarith [Real.pi_pos])
  have hsn := hs.ne'
  have hrn := hr0.ne'
  have hpn := Real.pi_pos.ne'
  have key : (r * Real.sin (π * r) / π) * (π / (r * Real.sin (π * r))) = 1 := by
    rw [div_mul_div_comm, mul_comm (r * Real.sin (π * r)) π]
    exact div_self (mul_ne_zero hpn (mul_ne_zero hrn hsn))
  calc x ^ r = x ^ r * ((r * Real.sin (π * r) / π) * (π / (r * Real.sin (π * r)))) := by
        rw [key, mul_one]
    _ = (r * Real.sin (π * r) / π) * (x ^ r * (π / (r * Real.sin (π * r)))) := by ring

end Cost
end Heracles
