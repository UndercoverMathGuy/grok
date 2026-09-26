import Heracles.Defs

/-!
# The positive-Fourier inequality (paper eq. (posfourier), appendix `app:scaling`)

For `z_j ≥ 0`, the average over `δ ∈ ℤ_p` of `∏_j exp(z_j cos(2π j δ / p))` is at least
`∏_j I_0(z_j)`.  Proof:

* (finite Fourier analysis on `ZMod p`) each factor `δ ↦ exp(z cos(2π j δ/p))` has non-negative
  real discrete Fourier coefficients; expanding the product and averaging over `δ` keeps only
  tuples of frequencies summing to `0`, all non-negative, among them the tuple `0`;
* the constant coefficient `(1/p) ∑_δ exp(z cos(2π j δ/p))` dominates `I_0(z)`, comparing the
  power series term by term (Wallis integrals).
-/

open Real Finset

noncomputable section

set_option linter.unusedSectionVars false

namespace Heracles

namespace PF

variable {p : ℕ} [NeZero p]

/-- The standard additive character `e(x) = exp(2π i x / p)` on `ZMod p`. -/
abbrev e (p : ℕ) [NeZero p] : AddChar (ZMod p) ℂ := ZMod.stdAddChar

lemma sum_e_mul (b : ZMod p) :
    ∑ x : ZMod p, e p (x * b) = if b = 0 then (p : ℂ) else 0 := by
  rw [AddChar.sum_mulShift b (ZMod.isPrimitive_stdAddChar p)]
  split_ifs <;> simp [ZMod.card]

lemma e_sum {ι : Type*} (s : Finset ι) (a : ι → ZMod p) :
    e p (∑ i ∈ s, a i) = ∏ i ∈ s, e p (a i) := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | insert i s hi ih => rw [Finset.sum_insert hi, Finset.prod_insert hi, AddChar.map_add_eq_mul, ih]

/-- Fourier inversion on `ZMod p`. -/
lemma inversion (g : ZMod p → ℂ) (δ : ZMod p) :
    g δ = ∑ k : ZMod p, ((1 / (p : ℂ)) * ∑ δ' : ZMod p, g δ' * e p (-(k * δ'))) * e p (k * δ) := by
  have hp : (p : ℂ) ≠ 0 := by exact_mod_cast NeZero.ne p
  symm
  calc ∑ k : ZMod p, ((1 / (p : ℂ)) * ∑ δ' : ZMod p, g δ' * e p (-(k * δ'))) * e p (k * δ)
      = (1 / (p : ℂ)) * ∑ δ' : ZMod p, g δ' * ∑ k : ZMod p, e p (k * (δ - δ')) := by
        simp_rw [Finset.mul_sum, Finset.sum_mul]
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun δ' _ => Finset.sum_congr rfl fun k _ => ?_
        have : k * (δ - δ') = -(k * δ') + k * δ := by ring
        rw [this, AddChar.map_add_eq_mul]
        ring
    _ = (1 / (p : ℂ)) * ∑ δ' : ZMod p, g δ' * (if δ - δ' = 0 then (p : ℂ) else 0) := by
        simp_rw [sum_e_mul]
    _ = g δ := by
        have : ∀ δ' : ZMod p, (δ - δ' = 0) ↔ (δ = δ') := fun δ' => sub_eq_zero
        simp_rw [this, mul_ite, mul_zero]
        rw [Finset.sum_ite_eq]
        simp only [Finset.mem_univ, ite_true]
        field_simp

/-- `cos(2π j δ / p)` in terms of the character. -/
lemma cos_eq_e (j : ℕ) (δ : ZMod p) :
    ((Real.cos (2 * π * (j : ℝ) * (δ.val : ℝ) / p) : ℝ) : ℂ)
      = (e p ((j : ZMod p) * δ) + e p (-((j : ZMod p) * δ))) / 2 := by
  have h1 : (j : ZMod p) * δ = (((j * δ.val : ℕ) : ℤ) : ZMod p) := by
    push_cast
    rw [ZMod.natCast_zmod_val]
  have h2 : -((j : ZMod p) * δ) = ((-((j * δ.val : ℕ) : ℤ) : ℤ) : ZMod p) := by
    rw [h1]; push_cast; ring
  rw [h2, h1, ZMod.stdAddChar_coe, ZMod.stdAddChar_coe, Complex.ofReal_cos]
  rw [Complex.cos]
  congr 2
  · congr 1; push_cast; ring
  · congr 1; push_cast; ring

/-- The index condition for the `m`-th term of the binomial expansion. -/
def cond (p : ℕ) (j : ℕ) (k : ZMod p) (n m : ℕ) : Prop :=
  ((m : ZMod p) - ((n - m : ℕ) : ZMod p)) * (j : ZMod p) - k = 0

instance (j : ℕ) (k : ZMod p) (n m : ℕ) : Decidable (cond p j k n m) := by
  unfold cond; infer_instance

/-- `(1/p) ∑_δ cos^n(2π j δ/p) e(-kδ)`, computed: a non-negative real. -/
def moment (p : ℕ) [NeZero p] (j : ℕ) (k : ZMod p) (n : ℕ) : ℝ :=
  (1 / 2 : ℝ) ^ n * ∑ m ∈ Finset.range (n + 1), (n.choose m : ℝ) * (if cond p j k n m then 1 else 0)

lemma moment_nonneg (j : ℕ) (k : ZMod p) (n : ℕ) : 0 ≤ moment p j k n := by
  unfold moment
  refine mul_nonneg (by positivity) (Finset.sum_nonneg fun m _ => ?_)
  split_ifs <;> positivity

lemma moment_eq (j : ℕ) (k : ZMod p) (n : ℕ) :
    (1 / (p : ℂ)) * ∑ δ : ZMod p,
        ((Real.cos (2 * π * (j : ℝ) * (δ.val : ℝ) / p) : ℝ) : ℂ) ^ n * e p (-(k * δ))
      = ((moment p j k n : ℝ) : ℂ) := by
  have hp : (p : ℂ) ≠ 0 := by exact_mod_cast NeZero.ne p
  -- expand the power
  have hexp : ∀ δ : ZMod p,
      ((Real.cos (2 * π * (j : ℝ) * (δ.val : ℝ) / p) : ℝ) : ℂ) ^ n * e p (-(k * δ))
        = (1 / 2 : ℂ) ^ n * ∑ m ∈ Finset.range (n + 1), (n.choose m : ℂ) *
            e p (δ * (((m : ZMod p) - ((n - m : ℕ) : ZMod p)) * (j : ZMod p) - k)) := by
    intro δ
    rw [cos_eq_e, div_eq_mul_inv, mul_comm _ (2 : ℂ)⁻¹, mul_pow, add_pow, Finset.mul_sum,
      Finset.mul_sum, Finset.sum_mul]
    refine Finset.sum_congr rfl fun m hm => ?_
    have hmn : m ≤ n := Nat.lt_succ_iff.1 (Finset.mem_range.1 hm)
    rw [← AddChar.map_nsmul_eq_pow, ← AddChar.map_nsmul_eq_pow]
    have : δ * (((m : ZMod p) - ((n - m : ℕ) : ZMod p)) * (j : ZMod p) - k)
        = (m • ((j : ZMod p) * δ) + (n - m) • (-((j : ZMod p) * δ))) + -(k * δ) := by
      simp only [nsmul_eq_mul]
      ring
    rw [this, AddChar.map_add_eq_mul, AddChar.map_add_eq_mul]
    simp only [one_div, inv_pow]
    ring
  simp_rw [hexp]
  rw [← Finset.mul_sum, Finset.sum_comm]
  simp_rw [← Finset.mul_sum, sum_e_mul]
  unfold moment
  push_cast
  rw [Finset.mul_sum, Finset.mul_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun m _ => ?_
  unfold cond
  split_ifs with h
  · simp [h]; field_simp
  · simp [h]

/-! ### Fourier coefficients of `δ ↦ exp(z cos(2π j δ / p))` -/

/-- The real exponential series. -/
lemma hasSum_exp_real (x : ℝ) : HasSum (fun n : ℕ => x ^ n / (n.factorial : ℝ)) (Real.exp x) := by
  rw [Real.exp_eq_exp_ℝ]
  exact NormedSpace.expSeries_div_hasSum_exp x

/-- The angle `θ_j(δ) = 2π j δ / p`. -/
def ang (p : ℕ) (j : ℕ) (δ : ZMod p) : ℝ := 2 * π * (j : ℝ) * (δ.val : ℝ) / p

/-- The discrete Fourier coefficient `(1/p) ∑_δ exp(z cos θ_j(δ)) e(-kδ)`. -/
def coef (p : ℕ) [NeZero p] (j : ℕ) (z : ℝ) (k : ZMod p) : ℂ :=
  (1 / (p : ℂ)) * ∑ δ : ZMod p, ((Real.exp (z * Real.cos (ang p j δ)) : ℝ) : ℂ) * e p (-(k * δ))

lemma coef_hasSum (j : ℕ) (z : ℝ) (k : ZMod p) :
    HasSum (fun n : ℕ => (((z ^ n / (n.factorial : ℝ)) * moment p j k n : ℝ) : ℂ)) (coef p j z k) := by
  have h1 : ∀ δ : ZMod p, HasSum
      (fun n : ℕ => ((((z * Real.cos (ang p j δ)) ^ n / (n.factorial : ℝ) : ℝ) : ℂ) * e p (-(k * δ))))
      (((Real.exp (z * Real.cos (ang p j δ)) : ℝ) : ℂ) * e p (-(k * δ))) := by
    intro δ
    exact ((Complex.hasSum_ofReal).2 (hasSum_exp_real _)).mul_right _
  have h2 := (hasSum_sum (s := (Finset.univ : Finset (ZMod p))) fun δ _ => h1 δ).mul_left (1 / (p : ℂ))
  unfold coef
  convert h2 using 1
  funext n
  rw [Complex.ofReal_mul, ← moment_eq, Finset.mul_sum, Finset.mul_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun δ _ => ?_
  simp only [ang, Complex.ofReal_div, Complex.ofReal_pow, Complex.ofReal_mul, Complex.ofReal_natCast]
  ring

/-- The real part of the coefficient, as a real series. -/
lemma coef_re_hasSum (j : ℕ) (z : ℝ) (k : ZMod p) :
    HasSum (fun n : ℕ => (z ^ n / (n.factorial : ℝ)) * moment p j k n) (coef p j z k).re := by
  have h := Complex.hasSum_re (coef_hasSum j z k)
  simp only [Complex.ofReal_re] at h
  exact h

lemma coef_im (j : ℕ) (z : ℝ) (k : ZMod p) : (coef p j z k).im = 0 := by
  have h := Complex.hasSum_im (coef_hasSum j z k)
  simp only [Complex.ofReal_im] at h
  exact h.unique hasSum_zero

lemma coef_eq_re (j : ℕ) (z : ℝ) (k : ZMod p) : coef p j z k = (((coef p j z k).re : ℝ) : ℂ) :=
  Complex.ext (by simp) (by simp [coef_im])

lemma coef_re_nonneg (j : ℕ) {z : ℝ} (hz : 0 ≤ z) (k : ZMod p) : 0 ≤ (coef p j z k).re :=
  (coef_re_hasSum j z k).nonneg fun n => mul_nonneg (by positivity) (moment_nonneg j k n)

/-- Fourier expansion of `exp(z cos θ_j(δ))` with real non-negative coefficients. -/
lemma exp_cos_expansion (j : ℕ) (z : ℝ) (δ : ZMod p) :
    ((Real.exp (z * Real.cos (ang p j δ)) : ℝ) : ℂ)
      = ∑ k : ZMod p, (((coef p j z k).re : ℝ) : ℂ) * e p (k * δ) := by
  rw [inversion (fun δ => ((Real.exp (z * Real.cos (ang p j δ)) : ℝ) : ℂ)) δ]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [← coef_eq_re]
  rfl

/-! ### The product step -/

/-- If each `f i` has a Fourier expansion with non-negative real coefficients `r i k`, then the
average of `∏_i f i` dominates `∏_i r i 0`. -/
lemma prod_step {ι : Type*} [Fintype ι] [DecidableEq ι] (f : ι → ZMod p → ℝ) (r : ι → ZMod p → ℝ)
    (hr : ∀ i k, 0 ≤ r i k) (hf : ∀ i δ, (f i δ : ℂ) = ∑ k : ZMod p, (r i k : ℂ) * e p (k * δ)) :
    ∏ i, r i 0 ≤ (1 / (p : ℝ)) * ∑ δ : ZMod p, ∏ i, f i δ := by
  classical
  have hp : (p : ℂ) ≠ 0 := by exact_mod_cast NeZero.ne p
  set S : ℝ := ∑ κ : ι → ZMod p, (∏ i, r i (κ i)) * (if ∑ i, κ i = 0 then 1 else 0) with hS
  have key : (((1 / (p : ℝ)) * ∑ δ : ZMod p, ∏ i, f i δ : ℝ) : ℂ) = (S : ℂ) := by
    push_cast
    simp_rw [hf]
    have hpu : ∀ δ : ZMod p, ∏ i, ∑ k : ZMod p, (r i k : ℂ) * e p (k * δ)
        = ∑ κ : ι → ZMod p, ∏ i, ((r i (κ i) : ℂ) * e p (κ i * δ)) := by
      intro δ
      rw [Finset.prod_univ_sum]
      simp
    simp_rw [hpu, Finset.prod_mul_distrib]
    rw [Finset.sum_comm, Finset.mul_sum, hS]
    push_cast
    refine Finset.sum_congr rfl fun κ _ => ?_
    rw [← Finset.mul_sum]
    have he : ∀ δ : ZMod p, ∏ i, e p (κ i * δ) = e p (δ * ∑ i, κ i) := by
      intro δ
      rw [Finset.mul_sum, e_sum]
      refine Finset.prod_congr rfl fun i _ => ?_
      rw [mul_comm]
    simp_rw [he, sum_e_mul]
    split_ifs
    · simp; field_simp
    · simp
  have hSr : S = (1 / (p : ℝ)) * ∑ δ : ZMod p, ∏ i, f i δ := by exact_mod_cast key.symm
  rw [← hSr, hS]
  have h0 : (∏ i, r i ((0 : ι → ZMod p) i)) * (if ∑ i, (0 : ι → ZMod p) i = 0 then (1 : ℝ) else 0)
      = ∏ i, r i 0 := by simp
  rw [← h0]
  refine Finset.single_le_sum (f := fun κ : ι → ZMod p =>
    (∏ i, r i (κ i)) * (if ∑ i, κ i = 0 then (1 : ℝ) else 0)) (fun κ _ => ?_) (Finset.mem_univ _)
  exact mul_nonneg (Finset.prod_nonneg fun i _ => hr i _) (by split_ifs <;> norm_num)

/-! ### Wallis integrals and the series of `I_0` -/

lemma J_rec (n : ℕ) :
    ∫ x in (0 : ℝ)..π, Real.cos x ^ (n + 2) = (n + 1) / (n + 2) * ∫ x in (0 : ℝ)..π, Real.cos x ^ n := by
  rw [integral_cos_pow]
  simp

lemma J_even (m : ℕ) :
    ∫ x in (0 : ℝ)..π, Real.cos x ^ (2 * m) = π * (m.centralBinom : ℝ) / 4 ^ m := by
  induction m with
  | zero => simp
  | succ m ih =>
    rw [show 2 * (m + 1) = 2 * m + 2 by ring, J_rec, ih]
    have h := Nat.succ_mul_centralBinom_succ m
    have h' : ((m + 1 : ℕ) : ℝ) * ((m + 1).centralBinom : ℝ) = 2 * (2 * m + 1) * (m.centralBinom : ℝ) := by
      exact_mod_cast h
    have hcb : ((m + 1).centralBinom : ℝ) = 2 * (2 * m + 1) * (m.centralBinom : ℝ) / (m + 1) := by
      rw [eq_div_iff (by positivity)]
      push_cast at h'
      linarith
    rw [hcb]
    push_cast
    field_simp
    ring

lemma J_odd (m : ℕ) : ∫ x in (0 : ℝ)..π, Real.cos x ^ (2 * m + 1) = 0 := by
  induction m with
  | zero => simp
  | succ m ih =>
    rw [show 2 * (m + 1) + 1 = (2 * m + 1) + 2 by ring, J_rec, ih, mul_zero]

/-- `M_n = (1/π) ∫_0^π cos^n`. -/
def Mom (n : ℕ) : ℝ := (1 / π) * ∫ x in (0 : ℝ)..π, Real.cos x ^ n

lemma Mom_even (m : ℕ) : Mom (2 * m) = (m.centralBinom : ℝ) / 4 ^ m := by
  unfold Mom
  rw [J_even]
  field_simp

lemma Mom_odd (m : ℕ) : Mom (2 * m + 1) = 0 := by
  unfold Mom
  rw [J_odd, mul_zero]

lemma Mom_nonneg (n : ℕ) : 0 ≤ Mom n := by
  obtain ⟨m, rfl | rfl⟩ := Nat.even_or_odd' n
  · rw [Mom_even]; positivity
  · rw [Mom_odd]

lemma Mom_le_moment (j : ℕ) (n : ℕ) : Mom n ≤ moment p j 0 n := by
  obtain ⟨m, rfl | rfl⟩ := Nat.even_or_odd' n
  · rw [Mom_even]
    unfold moment
    have hc : cond p j 0 (2 * m) m := by
      unfold cond
      rw [show 2 * m - m = m by omega]
      simp
    have hle : ((2 * m).choose m : ℝ) * (if cond p j 0 (2 * m) m then 1 else 0)
        ≤ ∑ i ∈ Finset.range (2 * m + 1),
            ((2 * m).choose i : ℝ) * (if cond p j 0 (2 * m) i then 1 else 0) := by
      refine Finset.single_le_sum (f := fun i => ((2 * m).choose i : ℝ) *
        (if cond p j 0 (2 * m) i then 1 else 0)) (fun i _ => ?_) (Finset.mem_range.2 (by omega))
      exact mul_nonneg (by positivity) (by split_ifs <;> norm_num)
    simp only [hc, ↓reduceIte, mul_one] at hle
    calc (m.centralBinom : ℝ) / 4 ^ m = (1 / 2 : ℝ) ^ (2 * m) * ((2 * m).choose m : ℝ) := by
          rw [Nat.centralBinom_eq_two_mul_choose, pow_mul,
            show ((1 : ℝ) / 2) ^ 2 = 1 / 4 by norm_num, one_div_pow]
          ring
      _ ≤ _ := mul_le_mul_of_nonneg_left hle (by positivity)
  · rw [Mom_odd]
    exact moment_nonneg j 0 _

/-- The power series of `I_0`. -/
lemma I0_hasSum (z : ℝ) : HasSum (fun n : ℕ => (z ^ n / (n.factorial : ℝ)) * Mom n) (I0 z) := by
  have h := intervalIntegral.hasSum_integral_of_dominated_convergence (μ := MeasureTheory.volume)
    (a := 0) (b := π) (f := fun θ => Real.exp (z * Real.cos θ))
    (F := fun (n : ℕ) (θ : ℝ) => (z * Real.cos θ) ^ n / (n.factorial : ℝ))
    (fun n _ => |z| ^ n / (n.factorial : ℝ))
    (fun n => (by fun_prop : Continuous fun θ : ℝ => (z * Real.cos θ) ^ n / (n.factorial : ℝ)).aestronglyMeasurable)
    (fun n => Filter.Eventually.of_forall fun θ _ => by
      rw [Real.norm_eq_abs, abs_div, abs_pow, abs_mul, Nat.abs_cast]
      gcongr
      calc |z| * |Real.cos θ| ≤ |z| * 1 := by gcongr; exact Real.abs_cos_le_one θ
        _ = |z| := mul_one _)
    (Filter.Eventually.of_forall fun θ _ => Real.summable_pow_div_factorial |z|)
    intervalIntegrable_const
    (Filter.Eventually.of_forall fun θ _ => hasSum_exp_real (z * Real.cos θ))
  have h2 := h.mul_left (1 / π)
  unfold I0
  convert h2 using 1
  funext n
  unfold Mom
  have : (fun θ : ℝ => (z * Real.cos θ) ^ n / (n.factorial : ℝ))
      = fun θ => (z ^ n / (n.factorial : ℝ)) * Real.cos θ ^ n := by
    funext θ; rw [mul_pow]; ring
  rw [this, intervalIntegral.integral_const_mul]
  ring

lemma one_le_I0' {z : ℝ} (hz : 0 ≤ z) : 1 ≤ I0 z := by
  have h := I0_hasSum z
  have h0 : (z ^ 0 / ((0 : ℕ).factorial : ℝ)) * Mom 0 = 1 := by
    unfold Mom; simp [Real.pi_ne_zero]
  rw [← h0]
  exact le_hasSum h 0 fun n _ => mul_nonneg (by positivity) (Mom_nonneg n)

lemma I0_pos' {z : ℝ} (hz : 0 ≤ z) : 0 < I0 z := lt_of_lt_of_le one_pos (one_le_I0' hz)

lemma I0_le_coef (j : ℕ) {z : ℝ} (hz : 0 ≤ z) : I0 z ≤ (coef p j z 0).re :=
  hasSum_le (fun n => mul_le_mul_of_nonneg_left (Mom_le_moment j n) (by positivity))
    (I0_hasSum z) (coef_re_hasSum j z 0)

/-- Core inequality: `∏_j I_0(z_j) ≤ (1/p) ∑_δ ∏_j exp(z_j cos(2π j δ/p))`. -/
lemma prod_I0_le {ι : Type*} [Fintype ι] [DecidableEq ι] (fr : ι → ℕ) (z : ι → ℝ)
    (hz : ∀ i, 0 ≤ z i) :
    ∏ i, I0 (z i) ≤ (1 / (p : ℝ)) * ∑ δ : ZMod p, ∏ i, Real.exp (z i * Real.cos (ang p (fr i) δ)) := by
  calc ∏ i, I0 (z i) ≤ ∏ i, (coef p (fr i) (z i) 0).re :=
        Finset.prod_le_prod₀ (fun i _ => (I0_pos' (hz i)).le) fun i _ => I0_le_coef (fr i) (hz i)
    _ ≤ _ := prod_step (fun i δ => Real.exp (z i * Real.cos (ang p (fr i) δ)))
          (fun i k => (coef p (fr i) (z i) k).re) (fun i k => coef_re_nonneg (fr i) (hz i) k)
          (fun i δ => exp_cos_expansion (fr i) (z i) δ)

end PF

open PF in
/-- **Positive-Fourier inequality** (paper eq. (posfourier)): for `y ≥ 0` and `t ≥ 0`,
`exp(-∑_j ψ(t y_j)) ≤ (1/p) ∑_{δ ∈ ℤ_p} exp(-t F_y(δ))`, where `F_y(δ) = ∑_j y_j (1 - cos(2π j δ/p))`. -/
theorem posfourier_gen (p : ℕ) [NeZero p] (y : Freq p → ℝ) (hy : ∀ j, 0 ≤ y j)
    {t : ℝ} (ht : 0 ≤ t) :
    Real.exp (-∑ j, psi (t * y j))
      ≤ (1 / (p : ℝ)) * ∑ δ : ZMod p, Real.exp (-(t * ∑ j, y j * cov p δ j.1)) := by
  classical
  have hz : ∀ j, 0 ≤ t * y j := fun j => mul_nonneg ht (hy j)
  have hL : Real.exp (-∑ j, psi (t * y j)) = (∏ j, Real.exp (-(t * y j))) * ∏ j, I0 (t * y j) := by
    rw [← Finset.prod_mul_distrib]
    unfold psi
    rw [show -∑ j, (t * y j - Real.log (I0 (t * y j)))
        = ∑ j, (-(t * y j) + Real.log (I0 (t * y j))) by
      rw [← Finset.sum_neg_distrib]; congr 1; funext j; ring]
    rw [Real.exp_sum]
    refine Finset.prod_congr rfl fun j _ => ?_
    rw [Real.exp_add, Real.exp_log (I0_pos' (hz j))]
  have hR : ∀ δ : ZMod p, Real.exp (-(t * ∑ j, y j * cov p δ j.1))
      = (∏ j, Real.exp (-(t * y j))) * ∏ j : Freq p, Real.exp ((t * y j) * Real.cos (ang p j.1 δ)) := by
    intro δ
    rw [← Finset.prod_mul_distrib]
    simp_rw [← Real.exp_add]
    rw [← Real.exp_sum]
    congr 1
    rw [Finset.mul_sum, ← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun j _ => ?_
    unfold cov ang
    ring
  simp_rw [hR]
  rw [hL, ← Finset.mul_sum, ← mul_assoc, mul_comm (1 / (p : ℝ)), mul_assoc]
  refine mul_le_mul_of_nonneg_left ?_ (Finset.prod_nonneg fun j _ => (Real.exp_pos _).le)
  exact prod_I0_le (p := p) (fun j : Freq p => j.1) (fun j => t * y j) hz

/-- **Positive-Fourier inequality** (paper eq. (posfourier)), in the setting of the paper
(`p ≥ 3` prime).  The hypotheses `p` prime and `p ≥ 3` are not needed (see `posfourier_gen`). -/
theorem posfourier (p : ℕ) [Fact p.Prime] (_hp : 3 ≤ p) (y : Freq p → ℝ) (hy : ∀ j, 0 ≤ y j)
    {t : ℝ} (ht : 0 ≤ t) :
    Real.exp (-∑ j, psi (t * y j))
      ≤ (1 / (p : ℝ)) * ∑ δ : ZMod p, Real.exp (-(t * ∑ j, y j * cov p δ j.1)) :=
  posfourier_gen p y hy ht

end Heracles
