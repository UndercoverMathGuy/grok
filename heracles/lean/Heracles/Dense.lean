import Heracles.Lower
import Heracles.Network

/-!
# Corollary `cor:dense` (`b = ν`: every frequency is used)

Paper, Section `sec:approach`, proof in appendix `app:dense`.

`Γ_1(p) = p/(p-1)`, with unique maximizer `y_j = 2/(p-1)` for all `j ∈ F_p`; and for the
quadratic network (`n = 2`, `ν = 3`) the constant `2 B_p · p/(p-1) = √(2/27) / (p^{1/2}(p-1))`.

At `s = 1` the budget is `∑_j y_j^{1/1} = ∑_j y_j`.

Uniqueness: equality forces `∑_j y_j = 1` and every coverage equal to `p/(p-1)`, i.e.
`ζ(d) := ∑_j y_j cos(2π j d/p) = -1/(p-1)` for `d ≠ 0` and `ζ(0) = 1`.  Instead of the paper's
normalized Fourier coefficients we use the (equivalent) finite orthogonality relation
`∑_{d<p} cos(2π j d/p) cos(2π k d/p) = (p/2)[j = k]` for `j, k ∈ F_p`.
-/

open Real Finset

noncomputable section

set_option linter.unusedSectionVars false

namespace Heracles

namespace DenseAux

variable (p : ℕ) [Fact p.Prime]

/-- Sum over the non-zero offsets = full sum minus the `0` term. -/
lemma sum_off_eq (f : ZMod p → ℝ) : ∑ δ : Off p, f δ.1 = ∑ δ : ZMod p, f δ - f 0 := by
  have h := Finset.sum_subtype (F := (inferInstance : Fintype (Off p)))
    (Finset.univ.erase (0 : ZMod p)) (p := fun δ : ZMod p => δ ≠ 0) (by intro x; simp) f
  rw [← h, ← Finset.add_sum_erase _ f (Finset.mem_univ (0 : ZMod p))]
  ring

/-- Sum over `ℤ_p` through `δ.val` = sum over `range p`. -/
lemma sum_zmod_val (g : ℕ → ℝ) : ∑ δ : ZMod p, g δ.val = ∑ d ∈ Finset.range p, g d := by
  refine Finset.sum_bij' (fun δ _ => δ.val) (fun d _ => (d : ZMod p)) ?_ ?_ ?_ ?_ ?_
  · intro δ _; exact Finset.mem_range.2 (ZMod.val_lt δ)
  · intro d _; exact Finset.mem_univ _
  · intro δ _; exact ZMod.natCast_zmod_val δ
  · intro d hd; exact ZMod.val_cast_of_lt (Finset.mem_range.1 hd)
  · intro δ _; rfl

/-- `∑_{d<p} cos(2π m d/p) = 0` for `0 < m < p`. -/
lemma sum_range_cos_nat (m : ℕ) (hm0 : 0 < m) (hmp : m < p) :
    ∑ d ∈ Finset.range p, Real.cos (2 * π * (m : ℝ) * (d : ℝ) / p) = 0 := by
  have h := sum_range_cos_eq_zero p (Fact.out : p.Prime).pos m hm0 hmp
  rw [← h]
  refine Finset.sum_congr rfl fun d _ => ?_
  congr 1; ring

lemma cos_mul_cos' (a b : ℝ) :
    Real.cos a * Real.cos b = (Real.cos (a + b) + Real.cos (a - b)) / 2 := by
  rw [Real.cos_add, Real.cos_sub]; ring

/-- Orthogonality: `∑_{d<p} cos(2π j d/p) cos(2π k d/p) = (p/2) [j = k]` for `j, k ∈ F_p`. -/
lemma orth (j k : ℕ) (hj : j ∈ Finset.Icc 1 ((p - 1) / 2)) (hk : k ∈ Finset.Icc 1 ((p - 1) / 2)) :
    ∑ d ∈ Finset.range p, Real.cos (2 * π * (j : ℝ) * d / p) * Real.cos (2 * π * (k : ℝ) * d / p)
      = if j = k then (p : ℝ) / 2 else 0 := by
  have hp2 : 2 ≤ p := (Fact.out : p.Prime).two_le
  rw [Finset.mem_Icc] at hj hk
  simp_rw [cos_mul_cos']
  rw [← Finset.sum_div, Finset.sum_add_distrib]
  have h1 : ∑ d ∈ Finset.range p,
      Real.cos (2 * π * (j : ℝ) * d / p + 2 * π * (k : ℝ) * d / p) = 0 := by
    rw [← sum_range_cos_nat p (j + k) (by omega) (by omega)]
    refine Finset.sum_congr rfl fun d _ => ?_
    congr 1; push_cast; ring
  have h2 : ∑ d ∈ Finset.range p,
      Real.cos (2 * π * (j : ℝ) * d / p - 2 * π * (k : ℝ) * d / p)
        = if j = k then (p : ℝ) else 0 := by
    split_ifs with hjk
    · subst hjk; simp
    · rcases lt_or_gt_of_ne hjk with hlt | hlt
      · rw [← sum_range_cos_nat p (k - j) (by omega) (by omega)]
        refine Finset.sum_congr rfl fun d _ => ?_
        rw [← Real.cos_neg, Nat.cast_sub hlt.le]
        congr 1; ring
      · rw [← sum_range_cos_nat p (j - k) (by omega) (by omega)]
        refine Finset.sum_congr rfl fun d _ => ?_
        rw [Nat.cast_sub hlt.le]
        congr 1; ring
  rw [h1, h2]
  split_ifs <;> ring

/-- Number of frequencies: `|F_p| = (p-1)/2`. -/
lemma card_freq : Fintype.card (Freq p) = (p - 1) / 2 := by
  rw [Fintype.card_coe, Nat.card_Icc]; omega

lemma budget_one (y : Freq p → ℝ) : budget 1 y = ∑ j, y j := by
  unfold budget; simp

end DenseAux

open DenseAux

/-- `∑_{δ ≠ 0} c_j(δ) = p` for every `j ∈ F_p`. -/
theorem sum_cov_offsets (p : ℕ) [Fact p.Prime] (hp : 3 ≤ p) (j : Freq p) :
    ∑ δ : Off p, covP p δ j = p := by
  have hj := Finset.mem_Icc.1 j.2
  unfold covP cov
  rw [sum_off_eq p (fun δ : ZMod p => 1 - Real.cos (2 * π * (j.1 : ℝ) * (δ.val : ℝ) / p))]
  rw [sum_zmod_val p (fun d => 1 - Real.cos (2 * π * (j.1 : ℝ) * (d : ℝ) / p))]
  rw [Finset.sum_sub_distrib, sum_range_cos_nat p j.1 (by omega) (by omega)]
  simp

/-- The minimum coverage is at most the average: `(p-1) m(y) ≤ p ∑_j y_j`. -/
lemma worst_le_avg (p : ℕ) [Fact p.Prime] (hp : 3 ≤ p) (y : Freq p → ℝ) :
    ((p : ℝ) - 1) * worst (covP p) y ≤ p * ∑ j, y j := by
  have hcard : ∑ _δ : Off p, (1 : ℝ) = (p : ℝ) - 1 := by
    rw [sum_off_eq p (fun _ => (1 : ℝ))]
    simp [ZMod.card]
  have h1 : ∑ δ : Off p, worst (covP p) y ≤ ∑ δ : Off p, cover (covP p) y δ :=
    Finset.sum_le_sum fun δ _ => worst_le_cover _ _ _
  have h2 : ∑ δ : Off p, cover (covP p) y δ = p * ∑ j, y j := by
    unfold cover
    rw [Finset.sum_comm]
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [← Finset.mul_sum, sum_cov_offsets p hp j]
    ring
  have h3 : ∑ _δ : Off p, worst (covP p) y = ((p : ℝ) - 1) * worst (covP p) y := by
    rw [← hcard, Finset.sum_mul, one_mul]
  linarith

/-- The flat allocation `y_j = 2/(p-1)`. -/
def flat (p : ℕ) : Freq p → ℝ := fun _ => 2 / ((p : ℝ) - 1)

lemma flat_facts (p : ℕ) [Fact p.Prime] (hp : 3 ≤ p) :
    Feasible 1 (flat p) ∧ ∀ δ : Off p, cover (covP p) (flat p) δ = p / ((p : ℝ) - 1) := by
  have hprime : p.Prime := Fact.out
  have hodd : p % 2 = 1 := Nat.odd_iff.1 (hprime.odd_of_ne_two (by omega))
  set m := (p - 1) / 2 with hm
  have hpm : p = 2 * m + 1 := by omega
  have hpmR : (p : ℝ) = 2 * m + 1 := by exact_mod_cast hpm
  have hm0 : (0 : ℝ) < m := by
    have : 1 ≤ m := by omega
    exact_mod_cast this
  have hcardR : ∑ _j : Freq p, (1 : ℝ) = m := by
    rw [Finset.sum_const, Finset.card_univ, card_freq p, nsmul_eq_mul, mul_one]
  have hp1 : (0 : ℝ) < (p : ℝ) - 1 := by rw [hpmR]; linarith
  refine ⟨⟨fun j => ?_, ?_⟩, fun δ => ?_⟩
  · unfold flat; exact div_nonneg (by norm_num) hp1.le
  · rw [budget_one]
    unfold flat
    rw [Finset.sum_const, Finset.card_univ, card_freq p, nsmul_eq_mul, ← hm, hpmR]
    field_simp
    ring_nf
    rfl
  · unfold cover flat covP cov
    rw [← Finset.mul_sum, Finset.sum_sub_distrib, sum_cos_freq p hp δ.1 δ.2, hcardR, hpmR]
    field_simp
    ring

/-- `Γ_1(p) = p/(p-1)`. -/
theorem GammaP_one (p : ℕ) [Fact p.Prime] (hp : 3 ≤ p) : GammaP p 1 = p / ((p : ℝ) - 1) := by
  have hp1 : (0 : ℝ) < (p : ℝ) - 1 := by
    have : (3 : ℝ) ≤ p := by exact_mod_cast hp
    linarith
  obtain ⟨hfeas, hcov⟩ := flat_facts p hp
  apply le_antisymm
  · refine Gam_le (by norm_num) fun y hy => ?_
    have hsum : ∑ j, y j ≤ 1 := by
      have := hy.2; rwa [budget_one] at this
    have h := worst_le_avg p hp y
    rw [le_div_iff₀ hp1]
    have hp0 : (0 : ℝ) ≤ p := by positivity
    nlinarith
  · have hw : worst (covP p) (flat p) = p / ((p : ℝ) - 1) := by
      obtain ⟨δ, hδ⟩ := exists_worst_eq (covP p) (flat p)
      rw [hδ, hcov δ]
    rw [← hw]
    exact le_Gam (covP_le_two p) le_rfl hfeas

/-- Uniqueness: the flat allocation is the unique maximizer of `Γ_1(p)`. -/
theorem dense_maximizer_iff (p : ℕ) [Fact p.Prime] (hp : 3 ≤ p) (y : Freq p → ℝ) :
    IsMaximizer (covP p) 1 y ↔ y = fun _ => 2 / ((p : ℝ) - 1) := by
  have hp3R : (3 : ℝ) ≤ p := by exact_mod_cast hp
  have hp1 : (0 : ℝ) < (p : ℝ) - 1 := by linarith
  have hpos : (0 : ℝ) < p := by linarith
  obtain ⟨hfeas, hcov⟩ := flat_facts p hp
  constructor
  · intro hmax
    have hw : worst (covP p) y = p / ((p : ℝ) - 1) := by
      rw [hmax.worst_eq (covP_le_two p) le_rfl]
      exact GammaP_one p hp
    have hsum_le : ∑ j, y j ≤ 1 := by
      have := hmax.1.2; rwa [budget_one] at this
    have havg := worst_le_avg p hp y
    rw [hw, mul_div_cancel₀ _ hp1.ne'] at havg
    have hsum : ∑ j, y j = 1 := by
      apply le_antisymm hsum_le
      nlinarith
    -- every coverage equals the worst coverage
    have hcov_eq : ∀ δ : Off p, cover (covP p) y δ = p / ((p : ℝ) - 1) := by
      have hcard : ∑ _δ : Off p, (1 : ℝ) = (p : ℝ) - 1 := by
        rw [sum_off_eq p (fun _ => (1 : ℝ))]
        simp [ZMod.card]
      have htot : ∑ δ : Off p, cover (covP p) y δ = p * ∑ j, y j := by
        unfold cover
        rw [Finset.sum_comm, Finset.mul_sum]
        refine Finset.sum_congr rfl fun j _ => ?_
        rw [← Finset.mul_sum, sum_cov_offsets p hp j]
        ring
      have hzero : ∑ δ : Off p, (cover (covP p) y δ - p / ((p : ℝ) - 1)) = 0 := by
        rw [Finset.sum_sub_distrib, htot, hsum, Finset.sum_const, Finset.card_univ,
          nsmul_eq_mul]
        have : (Fintype.card (Off p) : ℝ) = (p : ℝ) - 1 := by
          rw [← hcard]; simp
        rw [this]
        field_simp
        ring
      have hnn : ∀ δ ∈ (Finset.univ : Finset (Off p)),
          0 ≤ cover (covP p) y δ - p / ((p : ℝ) - 1) := by
        intro δ _
        have := worst_le_cover (covP p) y δ
        rw [hw] at this
        linarith
      intro δ
      have := (Finset.sum_eq_zero_iff_of_nonneg hnn).1 hzero δ (Finset.mem_univ δ)
      linarith
    -- ζ(d) = ∑_j y_j cos(2π j d/p)
    set ζ : ℕ → ℝ := fun d => ∑ j : Freq p, y j * Real.cos (2 * π * (j.1 : ℝ) * (d : ℝ) / p)
      with hζ
    have hζ0 : ζ 0 = 1 := by
      simp only [hζ, Nat.cast_zero, mul_zero, zero_div, Real.cos_zero, mul_one]
      exact hsum
    have hζd : ∀ d ∈ Finset.Ico 1 p, ζ d = 1 - p / ((p : ℝ) - 1) := by
      intro d hd
      rw [Finset.mem_Ico] at hd
      have hval : ((d : ZMod p)).val = d := ZMod.val_cast_of_lt hd.2
      have hne : (d : ZMod p) ≠ 0 := by
        intro h
        have : ((d : ZMod p)).val = 0 := by rw [h, ZMod.val_zero]
        omega
      have hc := hcov_eq ⟨(d : ZMod p), hne⟩
      unfold cover covP cov at hc
      simp only [hval] at hc
      simp only [mul_sub, mul_one, Finset.sum_sub_distrib, hsum] at hc
      simp only [hζ]
      linarith
    funext k
    have hk := k.2
    -- S = ∑_d ζ(d) cos(2π k d/p), computed two ways
    have hS1 : ∑ d ∈ Finset.range p, ζ d * Real.cos (2 * π * (k.1 : ℝ) * d / p)
        = y k * ((p : ℝ) / 2) := by
      simp only [hζ, Finset.sum_mul]
      rw [Finset.sum_comm]
      have : ∀ j : Freq p, ∑ d ∈ Finset.range p,
          y j * Real.cos (2 * π * (j.1 : ℝ) * (d : ℝ) / p) *
            Real.cos (2 * π * (k.1 : ℝ) * d / p)
          = y j * (if j.1 = k.1 then (p : ℝ) / 2 else 0) := by
        intro j
        rw [← orth p j.1 k.1 j.2 k.2, Finset.mul_sum]
        refine Finset.sum_congr rfl fun d _ => by ring
      rw [Finset.sum_congr rfl fun j _ => this j]
      rw [Finset.sum_eq_single k]
      · simp
      · intro j _ hjk
        have : j.1 ≠ k.1 := fun h => hjk (Subtype.ext h)
        simp [this]
      · simp
    have hS2 : ∑ d ∈ Finset.range p, ζ d * Real.cos (2 * π * (k.1 : ℝ) * d / p)
        = p / ((p : ℝ) - 1) := by
      have hk' := Finset.mem_Icc.1 k.2
      have hsplit : ∀ g : ℕ → ℝ, ∑ d ∈ Finset.range p, g d
          = g 0 + ∑ d ∈ Finset.Ico 1 p, g d := by
        intro g
        rw [Finset.range_eq_Ico, Finset.sum_eq_sum_Ico_succ_bot (by omega : 0 < p)]
      rw [hsplit, hζ0]
      rw [Finset.sum_congr rfl fun d hd => by rw [hζd d hd]]
      rw [← Finset.mul_sum]
      have hcs := sum_range_cos_nat p k.1 (by omega) (by omega)
      rw [hsplit] at hcs
      simp only [Nat.cast_zero, mul_zero, zero_div, Real.cos_zero] at hcs ⊢
      have : ∑ d ∈ Finset.Ico 1 p, Real.cos (2 * π * (k.1 : ℝ) * d / p) = -1 := by linarith
      rw [this]
      field_simp
      ring
    rw [hS1] at hS2
    field_simp at hS2 ⊢
    nlinarith
  · rintro rfl
    refine (isMaximizer_iff (covP_le_two p) le_rfl).2 ⟨hfeas, ?_⟩
    obtain ⟨δ, hδ⟩ := exists_worst_eq (covP p) (flat p)
    have hG := GammaP_one p hp
    unfold GammaP at hG
    change worst (covP p) (flat p) = _
    rw [hδ, hcov δ, hG]

/-- The constant for the quadratic network (`n = 2`):
`2 B_p · p/(p-1) = √(2/27) / (p^{1/2} (p-1))`. -/
theorem dense_constant (p : ℕ) (hp : 0 < p) :
    2 * Bp p 2 * (p / ((p : ℝ) - 1)) = Real.sqrt (2 / 27) / (Real.sqrt p * ((p : ℝ) - 1)) := by
  have hpR : (0 : ℝ) < p := by exact_mod_cast hp
  unfold Bp
  have h3 : ((2 : ℕ) : ℝ) + 1 = 3 := by norm_num
  rw [h3]
  have h6 : (2 * (p : ℝ) * 3) = 6 * p := by ring
  rw [h6]
  have h6p : (0 : ℝ) < 6 * p := by positivity
  have hpow : (6 * (p : ℝ)) ^ (-((3 : ℝ) / 2)) = 1 / (6 * p * Real.sqrt (6 * p)) := by
    rw [Real.rpow_neg h6p.le, Real.sqrt_eq_rpow, ← Real.rpow_one_add' h6p.le (by norm_num)]
    norm_num
  rw [hpow]
  have hs6 : Real.sqrt (6 * p) = Real.sqrt 2 * Real.sqrt 3 * Real.sqrt p := by
    rw [show (6 : ℝ) * p = 2 * 3 * p by ring, Real.sqrt_mul (by positivity),
      Real.sqrt_mul (by positivity)]
  have hs27 : Real.sqrt (2 / 27) = Real.sqrt 2 / (3 * Real.sqrt 3) := by
    rw [Real.sqrt_div (by norm_num), show (27 : ℝ) = 3 ^ 2 * 3 by norm_num,
      Real.sqrt_mul (by positivity), Real.sqrt_sq (by norm_num)]
  rw [hs6, hs27]
  have h2 : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  have h2p : 0 < Real.sqrt 2 := by positivity
  have h3p : 0 < Real.sqrt 3 := by positivity
  have hpp : 0 < Real.sqrt p := Real.sqrt_pos.2 hpR
  by_cases hp1 : (p : ℝ) - 1 = 0
  · rw [hp1]; simp
  · field_simp
    nlinarith [h2]

end Heracles
