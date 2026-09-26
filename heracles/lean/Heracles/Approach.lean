import Heracles.Network
import Heracles.Lower

/-!
# Section `sec:approach`: averaged margin, spectrum, phase lemma, coverage bound,
# universal bound

Paper, Section `sec:approach` and appendix `app:spectrum`.

* `lem_avg` (Lemma `lem:avg`): `h(f) ≤ η(0) - η(δ)` for `δ ≠ 0`.
* `spectrum` (eq. (spectrum)): `η(δ) = η̄ + ∑_{j ∈ F_p} Re(β_j ρ^{jδ})`, and its averaged form
  `spectrum_avg`: `η(0) - (η(δ)+η(-δ))/2 = ∑_{j ∈ F_p} Re(β_j) c_j(δ)`.
* `lem_phase` (Lemma `lem:phase`).
* `prop_coverage` (Proposition `prop:coverage`).
* `beta_netTable`: `β_j` is additive over neurons.
* `thm_universal` (Theorem `thm:universal`) for generic neuron models, with the single-neuron
  budget as a hypothesis, and `thm_universal_Cb` with `C_b` defined as the supremum of the
  paper, under explicit homogeneity hypotheses.
-/

open Real Finset

noncomputable section

set_option linter.unusedSectionVars false

namespace Heracles

/-! ## Characters of `ℤ_p` -/

section Char

variable {p : ℕ} [NeZero p]

lemma rho_add (a b : ZMod p) : rho (a + b) = rho a * rho b :=
  AddChar.map_add_eq_mul _ a b

lemma rho_zero : rho (0 : ZMod p) = 1 := AddChar.map_zero_eq_one _

lemma rho_neg (a : ZMod p) : rho (-a) = starRingEnd ℂ (rho a) :=
  AddChar.map_neg_eq_conj _ a

lemma sum_rho_mul (b : ZMod p) :
    ∑ x : ZMod p, rho (x * b) = if b = 0 then (p : ℂ) else 0 := by
  unfold rho
  rw [AddChar.sum_mulShift b (ZMod.isPrimitive_stdAddChar p)]
  split_ifs <;> simp [ZMod.card]

lemma rho_natCast (m : ℕ) : rho (m : ZMod p) = Complex.exp (2 * π * Complex.I * m / p) := by
  unfold rho
  have := ZMod.stdAddChar_coe (N := p) (m : ℤ)
  simpa using this

lemma re_rho_natCast (m : ℕ) : (rho (m : ZMod p)).re = Real.cos (2 * π * m / p) := by
  rw [rho_natCast]
  have : (2 * π * Complex.I * m / p : ℂ) = ((2 * π * m / p : ℝ) : ℂ) * Complex.I := by
    push_cast; ring
  rw [this, Complex.exp_ofReal_mul_I_re]

/-- `Re ρ^{jδ} = cos(2π j δ / p)`. -/
lemma re_rho_mul (j : ℕ) (δ : ZMod p) :
    (rho ((j : ZMod p) * δ)).re = Real.cos (2 * π * j * δ.val / p) := by
  have : (j : ZMod p) * δ = ((j * δ.val : ℕ) : ZMod p) := by
    push_cast; rw [ZMod.natCast_zmod_val]
  rw [this, re_rho_natCast]
  congr 1
  push_cast
  ring

/-- Fourier inversion on `ℤ_p`. -/
lemma dft_inv (g : ZMod p → ℂ) (δ : ZMod p) :
    g δ = (1 / (p : ℂ)) *
      ∑ k : ZMod p, (∑ δ' : ZMod p, g δ' * rho (-(k * δ'))) * rho (k * δ) := by
  have hp : (p : ℂ) ≠ 0 := by exact_mod_cast NeZero.ne p
  have h1 : ∑ k : ZMod p, (∑ δ' : ZMod p, g δ' * rho (-(k * δ'))) * rho (k * δ)
      = ∑ δ' : ZMod p, g δ' * ∑ k : ZMod p, rho (k * (δ - δ')) := by
    simp_rw [Finset.sum_mul, Finset.mul_sum]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun δ' _ => Finset.sum_congr rfl fun k _ => ?_
    rw [mul_assoc, ← rho_add]
    congr 2
    ring
  rw [h1]
  simp_rw [sum_rho_mul, sub_eq_zero]
  rw [Finset.sum_eq_single δ]
  · simp only [ite_true]
    field_simp
  · intro b _ hb
    simp [Ne.symm hb]
  · simp

end Char

/-! ## Splitting a sum over `ℤ_p` into `0`, `F_p` and `-F_p` -/

lemma sum_zmod_split {M : Type*} [AddCommMonoid M] (p : ℕ) [Fact p.Prime] (hp : 3 ≤ p)
    (h : ZMod p → M) :
    ∑ k : ZMod p, h k = h 0 + ∑ j : Freq p, (h (j.1 : ZMod p) + h (-(j.1 : ZMod p))) := by
  have hprime : p.Prime := Fact.out
  have hodd : p % 2 = 1 := Nat.odd_iff.1 (hprime.odd_of_ne_two (by omega))
  set m := (p - 1) / 2 with hm
  have hpm : p = 2 * m + 1 := by omega
  set g : ℕ → M := fun i => h (i : ZMod p) with hg
  have h1 : ∑ k : ZMod p, h k = ∑ i ∈ Finset.range p, g i := by
    refine Finset.sum_nbij' (fun k => k.val) (fun i => (i : ZMod p)) ?_ ?_ ?_ ?_ ?_
    · intro k _; exact Finset.mem_range.2 (ZMod.val_lt k)
    · intro i _; exact Finset.mem_univ _
    · intro k _; exact ZMod.natCast_zmod_val k
    · intro i hi; exact ZMod.val_cast_of_lt (Finset.mem_range.1 hi)
    · intro k _; simp only [hg]; rw [ZMod.natCast_zmod_val]
  have hsplit : ∑ i ∈ Finset.range p, g i
      = g 0 + (∑ i ∈ Finset.Ico 1 (m + 1), g i + ∑ i ∈ Finset.Ico (m + 1) p, g i) := by
    rw [Finset.range_eq_Ico, ← Finset.sum_Ico_consecutive g (m := 0) (n := 1) (k := p)
      (by omega) (by omega), ← Finset.sum_Ico_consecutive g (m := 1) (n := m + 1) (k := p)
      (by omega) (by omega)]
    simp
  have hrefl : ∑ i ∈ Finset.Ico (m + 1) p, g i
      = ∑ j ∈ Finset.Ico 1 (m + 1), h (-(j : ZMod p)) := by
    have hr := Finset.sum_Ico_reflect g 1 (m := m + 1) (n := p) (by omega)
    have e1 : p + 1 - (m + 1) = m + 1 := by omega
    have e2 : p + 1 - 1 = p := by omega
    rw [e1, e2] at hr
    rw [← hr]
    refine Finset.sum_congr rfl fun j hj => ?_
    have hjp : j ≤ p := by
      have := (Finset.mem_Ico.1 hj).2; omega
    simp only [hg]
    rw [Nat.cast_sub hjp, ZMod.natCast_self, zero_sub]
  have hF : ∑ j : Freq p, (h (j.1 : ZMod p) + h (-(j.1 : ZMod p)))
      = ∑ j ∈ Finset.Ico 1 (m + 1), (h (j : ZMod p) + h (-(j : ZMod p))) := by
    rw [Finset.Ico_add_one_right_eq_Icc]
    exact Finset.sum_coe_sort (Finset.Icc 1 ((p - 1) / 2))
      (fun j => h (j : ZMod p) + h (-(j : ZMod p)))
  rw [h1, hsplit, hrefl, hF, Finset.sum_add_distrib]
  simp [hg]

/-! ## The averaged logit and its spectrum -/

section Spectrum

variable {p n : ℕ} [Fact p.Prime]

/-- `G_k := f̂(k,…,k; -k)`. -/
def Gk (f : Table p n) (k : ZMod p) : ℂ := fhat f (fun _ => k) (-k)

/-- The mean `η̄ = p^{-(n+1)} Re f̂(0,…,0;0)` of eq. (spectrum). -/
def etaBar (f : Table p n) : ℝ := (1 / (p : ℝ) ^ (n + 1)) * (Gk f 0).re

lemma pos_p : (0 : ℝ) < p := by exact_mod_cast (Fact.out : p.Prime).pos

/-- The DFT of `η` is `p^{-n} G_k`. -/
lemma eta_dft (f : Table p n) (k : ZMod p) :
    ∑ δ : ZMod p, (eta f δ : ℂ) * rho (-(k * δ)) = (1 / (p : ℂ) ^ n) * Gk f k := by
  have h1 : ∀ x : Inp p n, ∑ δ : ZMod p, (f x (label x - δ) : ℂ) * rho (-(k * δ)) =
      ∑ c : ZMod p, (f x c : ℂ) * rho (-(∑ t, (fun _ => k) t * x t + -k * c)) := by
    intro x
    refine Fintype.sum_equiv (Equiv.subLeft (label x)) _ _ (fun δ => ?_)
    simp only [Equiv.subLeft_apply]
    congr 2
    unfold label
    rw [← Finset.mul_sum]
    ring
  have h2 : ∀ δ : ZMod p, ((eta f δ : ℝ) : ℂ) * rho (-(k * δ)) =
      (1 / (p : ℂ) ^ n) * ∑ x : Inp p n, (f x (label x - δ) : ℂ) * rho (-(k * δ)) := by
    intro δ
    unfold eta
    push_cast
    rw [mul_assoc, Finset.sum_mul]
  simp_rw [h2]
  rw [← Finset.mul_sum, Finset.sum_comm]
  unfold Gk fhat
  congr 1
  exact Finset.sum_congr rfl fun x _ => h1 x

/-- Fourier expansion of `η`: `η(δ) = p^{-(n+1)} ∑_k G_k ρ^{kδ}` (appendix `app:spectrum`). -/
lemma eta_eq_sum (f : Table p n) (δ : ZMod p) :
    (eta f δ : ℂ) = (1 / (p : ℂ) ^ (n + 1)) * ∑ k : ZMod p, Gk f k * rho (k * δ) := by
  rw [dft_inv (fun δ => (eta f δ : ℂ)) δ]
  simp_rw [eta_dft]
  rw [Finset.mul_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [pow_succ]
  ring

/-- Reality of the table: `G_{-k} = conj G_k`. -/
lemma Gk_neg (f : Table p n) (k : ZMod p) : Gk f (-k) = starRingEnd ℂ (Gk f k) := by
  unfold Gk fhat
  simp only [map_sum, map_mul, Complex.conj_ofReal]
  refine Finset.sum_congr rfl fun x _ => Finset.sum_congr rfl fun c _ => ?_
  congr 1
  rw [← rho_neg]
  congr 1
  simp only [neg_mul, Finset.sum_neg_distrib, neg_neg]
  ring

lemma ofReal_div_pow (a : ℝ) (m : ℕ) :
    (a : ℂ) / (p : ℂ) ^ m = ((a / (p : ℝ) ^ m : ℝ) : ℂ) := by push_cast; ring

/-- **Eq. (spectrum).** `η(δ) = η̄ + ∑_{j ∈ F_p} Re(β_j ρ^{jδ})`. -/
theorem spectrum (hp : 3 ≤ p) (f : Table p n) (δ : ZMod p) :
    eta f δ = etaBar f + ∑ j : Freq p, (beta f j.1 * rho ((j.1 : ZMod p) * δ)).re := by
  have hre : eta f δ = (1 / (p : ℝ) ^ (n + 1)) * ∑ k : ZMod p, (Gk f k * rho (k * δ)).re := by
    have := congrArg Complex.re (eta_eq_sum f δ)
    rw [Complex.ofReal_re] at this
    rw [this, show (1 / (p : ℂ) ^ (n + 1)) = ((1 / (p : ℝ) ^ (n + 1) : ℝ) : ℂ) by push_cast; ring,
      Complex.re_ofReal_mul, Complex.re_sum]
  have hs := sum_zmod_split p hp (fun k => (Gk f k * rho (k * δ)).re)
  rw [hre, hs]
  have h0 : (Gk f 0 * rho (0 * δ)).re = (Gk f 0).re := by simp [rho_zero]
  have hj : ∀ j : Freq p, (Gk f (-(j.1 : ZMod p)) * rho (-(j.1 : ZMod p) * δ)).re =
      (Gk f (j.1 : ZMod p) * rho ((j.1 : ZMod p) * δ)).re := by
    intro j
    rw [Gk_neg, neg_mul, rho_neg, ← map_mul, Complex.conj_re]
  have hb : ∀ j : Freq p, (beta f j.1 * rho ((j.1 : ZMod p) * δ)).re =
      2 / (p : ℝ) ^ (n + 1) * (Gk f (j.1 : ZMod p) * rho ((j.1 : ZMod p) * δ)).re := by
    intro j
    unfold beta
    rw [show (2 : ℂ) / (p : ℂ) ^ (n + 1) = ((2 / (p : ℝ) ^ (n + 1) : ℝ) : ℂ) by push_cast; ring,
      mul_assoc, Complex.re_ofReal_mul]
    rfl
  rw [h0]
  simp_rw [hj, hb]
  unfold etaBar
  rw [mul_add, Finset.mul_sum]
  congr 1
  refine Finset.sum_congr rfl fun j _ => ?_
  ring

/-- `Re(β z) + Re(β conj z) = 2 Re β Re z`. -/
lemma re_mul_add_re_mul_conj (β z : ℂ) :
    (β * z).re + (β * starRingEnd ℂ z).re = 2 * β.re * z.re := by
  simp only [Complex.mul_re, Complex.conj_re, Complex.conj_im]
  ring

/-- **Averaged form of eq. (spectrum)** (appendix `app:spectrum`):
`η(0) - (η(δ)+η(-δ))/2 = ∑_{j ∈ F_p} Re(β_j) c_j(δ)`. -/
theorem spectrum_avg (hp : 3 ≤ p) (f : Table p n) (δ : ZMod p) :
    eta f 0 - (eta f δ + eta f (-δ)) / 2 = ∑ j : Freq p, (beta f j.1).re * cov p δ j.1 := by
  rw [spectrum hp f 0, spectrum hp f δ, spectrum hp f (-δ)]
  have key : ∀ j : Freq p,
      (beta f j.1 * rho ((j.1 : ZMod p) * 0)).re -
        ((beta f j.1 * rho ((j.1 : ZMod p) * δ)).re +
          (beta f j.1 * rho ((j.1 : ZMod p) * -δ)).re) / 2 =
      (beta f j.1).re * cov p δ j.1 := by
    intro j
    rw [mul_zero, rho_zero, mul_one, mul_neg, rho_neg, re_mul_add_re_mul_conj, re_rho_mul]
    unfold cov
    ring
  rw [← Finset.sum_congr rfl fun j _ => key j, Finset.sum_sub_distrib, ← Finset.sum_div,
    Finset.sum_add_distrib]
  ring

end Spectrum

/-! ## Lemma `lem:avg`, Lemma `lem:phase`, Proposition `prop:coverage` -/

section Margins

variable {p n : ℕ} [Fact p.Prime]

/-- **Lemma `lem:avg`** (averaged margin): `h(f) ≤ η(0) - η(δ)` for `δ ≠ 0`. -/
theorem lem_avg (f : Table p n) (δ : ZMod p) (hδ : δ ≠ 0) :
    margin f ≤ eta f 0 - eta f δ := by
  have hp0 : (0 : ℝ) < (p : ℝ) ^ n := pow_pos pos_p n
  have key : ∀ x : Inp p n, margin f ≤ f x (label x - 0) - f x (label x - δ) := by
    intro x
    rw [sub_zero]
    exact margin_le f x _ (fun h => hδ (sub_eq_self.1 h))
  have hsum : (p : ℝ) ^ n * margin f ≤
      ∑ x : Inp p n, (f x (label x - 0) - f x (label x - δ)) := by
    have := Finset.sum_le_sum (fun x (_ : x ∈ (Finset.univ : Finset (Inp p n))) => key x)
    simpa [Finset.sum_const, Finset.card_univ, ZMod.card] using this
  unfold eta
  rw [← mul_sub, ← Finset.sum_sub_distrib]
  calc margin f = (1 / (p : ℝ) ^ n) * ((p : ℝ) ^ n * margin f) := by field_simp
    _ ≤ (1 / (p : ℝ) ^ n) * ∑ x : Inp p n, (f x (label x - 0) - f x (label x - δ)) :=
        mul_le_mul_of_nonneg_left hsum (by positivity)

/-- **Lemma `lem:phase`**: for `δ ≠ 0`,
`h(f) ≤ ∑_{j ∈ F_p} Re(β_j) c_j(δ) ≤ ∑_{j ∈ F_p} A_j c_j(δ)`. -/
theorem lem_phase (hp : 3 ≤ p) (f : Table p n) (δ : ZMod p) (hδ : δ ≠ 0) :
    margin f ≤ ∑ j : Freq p, (beta f j.1).re * cov p δ j.1 ∧
      ∑ j : Freq p, (beta f j.1).re * cov p δ j.1 ≤ ∑ j : Freq p, amp f j.1 * cov p δ j.1 := by
  refine ⟨?_, ?_⟩
  · rw [← spectrum_avg hp f δ]
    have h1 := lem_avg f δ hδ
    have h2 := lem_avg f (-δ) (neg_ne_zero.2 hδ)
    linarith
  · refine Finset.sum_le_sum fun j _ => ?_
    exact mul_le_mul_of_nonneg_right (Complex.re_le_norm _) (cov_nonneg p δ j.1)

/-- **Proposition `prop:coverage`**: `h(f) ≤ Γ_s(p) ‖A‖_{1/s}` with
`‖A‖_{1/s} = (∑_j A_j^{1/s})^s`, for every `s ≥ 1`. -/
theorem prop_coverage (hp : 3 ≤ p) (f : Table p n) {s : ℝ} (hs : 1 ≤ s) :
    margin f ≤ GammaP p s * (∑ j : Freq p, amp f j.1 ^ (1 / s)) ^ s := by
  set A : Freq p → ℝ := fun j => amp f j.1 with hAdef
  have hA : ∀ j, 0 ≤ A j := fun j => norm_nonneg _
  have hs0 : 0 < s := by linarith
  have hphase : ∀ δ : Off p, margin f ≤ ∑ j : Freq p, A j * covP p δ j := fun δ =>
    ((lem_phase hp f δ.1 δ.2).1.trans (lem_phase hp f δ.1 δ.2).2)
  set S := ∑ j : Freq p, A j ^ (1 / s) with hSdef
  have hS0 : 0 ≤ S := Finset.sum_nonneg fun j _ => Real.rpow_nonneg (hA j) _
  rcases eq_or_lt_of_le hS0 with hS | hS
  · have hAz : ∀ j, A j = 0 := by
      intro j
      have := (Finset.sum_eq_zero_iff_of_nonneg
        (fun j _ => Real.rpow_nonneg (hA j) (1 / s))).1 hS.symm j (Finset.mem_univ j)
      exact (Real.rpow_eq_zero (hA j) (one_div_ne_zero hs0.ne')).1 this
    obtain ⟨δ⟩ := (inferInstance : Nonempty (Off p))
    have := hphase δ
    simp only [hAz, zero_mul, Finset.sum_const_zero] at this
    rw [← hS, Real.zero_rpow hs0.ne', mul_zero]
    exact this
  · set Nn := S ^ s with hNn
    have hN : 0 < Nn := Real.rpow_pos_of_pos hS s
    set y : Freq p → ℝ := fun j => A j / Nn with hy
    have hfeas : Feasible s y := by
      refine ⟨fun j => div_nonneg (hA j) hN.le, ?_⟩
      unfold budget
      have hyj : ∀ j, y j ^ (1 / s) = A j ^ (1 / s) / S := by
        intro j
        simp only [hy]
        rw [Real.div_rpow (hA j) hN.le, hNn, ← Real.rpow_mul hS.le, mul_one_div_cancel hs0.ne',
          Real.rpow_one]
      simp_rw [hyj, ← Finset.sum_div]
      rw [← hSdef, div_self hS.ne']
    obtain ⟨δ, hδ⟩ := exists_worst_eq (covP p) y
    have h1 : margin f ≤ Nn * cover (covP p) y δ := by
      refine (hphase δ).trans (le_of_eq ?_)
      unfold cover
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun j _ => ?_
      simp only [hy]
      field_simp
    have h2 : cover (covP p) y δ ≤ GammaP p s := by
      rw [← hδ]; exact le_Gam (covP_le_two p) hs hfeas
    calc margin f ≤ Nn * cover (covP p) y δ := h1
      _ ≤ Nn * GammaP p s := mul_le_mul_of_nonneg_left h2 hN.le
      _ = GammaP p s * (∑ j : Freq p, amp f j.1 ^ (1 / s)) ^ s := mul_comm _ _

end Margins

/-! ## Additivity over neurons and Theorem `thm:universal` -/

section Universal

variable {p n : ℕ} [Fact p.Prime]

/-- `β_j` is additive over the neurons of a network. -/
theorem beta_netTable {Ω : Type*} {m : ℕ} (φ : Ω → Table p n) (θ : Fin m → Ω) (j : ℕ) :
    beta (netTable φ θ) j = ∑ i, beta (φ (θ i)) j := by
  unfold beta fhat netTable
  rw [← Finset.mul_sum]
  congr 1
  simp only [Complex.ofReal_sum, Finset.sum_mul]
  conv_rhs => rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun x _ => ?_
  exact Finset.sum_comm

/-- `β_j` is homogeneous: `β_j(a f) = a β_j(f)`. -/
theorem beta_smul (f : Table p n) (a : ℝ) (j : ℕ) :
    beta (fun x c => a * f x c) j = (a : ℂ) * beta f j := by
  unfold beta fhat
  rw [mul_left_comm (a : ℂ)]
  congr 1
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun c _ => ?_
  push_cast
  ring

/-- Subadditivity of `t ↦ t^r` (`0 < r ≤ 1`) over a finite sum. -/
lemma rpow_sum_le_sum_rpow' {κ : Type*} (s : Finset κ) (g : κ → ℝ) (hg : ∀ i, 0 ≤ g i)
    {r : ℝ} (hr0 : 0 < r) (hr1 : r ≤ 1) :
    (∑ i ∈ s, g i) ^ r ≤ ∑ i ∈ s, g i ^ r := by
  classical
  induction s using Finset.induction_on with
  | empty => simp [Real.zero_rpow hr0.ne']
  | insert a s ha ih =>
    rw [Finset.sum_insert ha, Finset.sum_insert ha]
    calc (g a + ∑ i ∈ s, g i) ^ r ≤ g a ^ r + (∑ i ∈ s, g i) ^ r :=
          Real.rpow_add_le_add_rpow (hg a) (Finset.sum_nonneg fun i _ => hg i) hr0.le hr1
      _ ≤ g a ^ r + ∑ i ∈ s, g i ^ r := by linarith

/-- **Theorem `thm:universal`** (generic neurons, budget form).  For any neuron model
`φ : Ω → Table p n` with a size function `N` (playing `‖ω‖₂`) and any constant `C` satisfying the
single-neuron budget `∑_{j ∈ F_p} |β_j(φ ω)|^{b/ν} ≤ C N(ω)^b`, every network of any width
satisfies `h(θ) ≤ C^{ν/b} Γ_{ν/b}(p) ‖θ‖_{2,b}^ν`, where `‖θ‖_{2,b}^ν = (∑_i N(θ_i)^b)^{ν/b}`. -/
theorem thm_universal (hp : 3 ≤ p) {Ω : Type*} (φ : Ω → Table p n) (N : Ω → ℝ)
    {b ν C : ℝ} (hb : 0 < b) (hbν : b ≤ ν) (hC : 0 ≤ C) (hN : ∀ ω, 0 ≤ N ω)
    (hbudget : ∀ ω, ∑ j : Freq p, ‖beta (φ ω) j.1‖ ^ (b / ν) ≤ C * N ω ^ b)
    {m : ℕ} (θ : Fin m → Ω) :
    margin (netTable φ θ) ≤ C ^ (ν / b) * GammaP p (ν / b) * (∑ i, N (θ i) ^ b) ^ (ν / b) := by
  have hν : 0 < ν := lt_of_lt_of_le hb hbν
  have hr0 : 0 < b / ν := div_pos hb hν
  have hr1 : b / ν ≤ 1 := (div_le_one hν).2 hbν
  have hs1 : 1 ≤ ν / b := (one_le_div hb).2 hbν
  have hrs : 1 / (ν / b) = b / ν := one_div_div ν b
  have hcov := prop_coverage hp (netTable φ θ) hs1
  rw [hrs] at hcov
  have hA : ∑ j : Freq p, amp (netTable φ θ) j.1 ^ (b / ν) ≤ C * ∑ i, N (θ i) ^ b := by
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
  have hΓ : 0 ≤ GammaP p (ν / b) := (GammaP_pos p hp hs1).le
  have hA0 : 0 ≤ ∑ j : Freq p, amp (netTable φ θ) j.1 ^ (b / ν) :=
    Finset.sum_nonneg fun j _ => Real.rpow_nonneg (norm_nonneg _) _
  have hNb : 0 ≤ ∑ i, N (θ i) ^ b := Finset.sum_nonneg fun i _ => Real.rpow_nonneg (hN _) _
  calc margin (netTable φ θ)
      ≤ GammaP p (ν / b) * (∑ j : Freq p, amp (netTable φ θ) j.1 ^ (b / ν)) ^ (ν / b) := hcov
    _ ≤ GammaP p (ν / b) * (C * ∑ i, N (θ i) ^ b) ^ (ν / b) :=
        mul_le_mul_of_nonneg_left (Real.rpow_le_rpow hA0 hA (by positivity)) hΓ
    _ = C ^ (ν / b) * GammaP p (ν / b) * (∑ i, N (θ i) ^ b) ^ (ν / b) := by
        rw [Real.mul_rpow hC hNb]; ring

/-- Corollary: for `‖θ‖_{2,b} ≤ 1`, `h(θ) ≤ C^{ν/b} Γ_{ν/b}(p)` (so `γ*_b ≤ C^{ν/b} Γ_{ν/b}`). -/
theorem thm_universal_unit (hp : 3 ≤ p) {Ω : Type*} (φ : Ω → Table p n) (N : Ω → ℝ)
    {b ν C : ℝ} (hb : 0 < b) (hbν : b ≤ ν) (hC : 0 ≤ C) (hN : ∀ ω, 0 ≤ N ω)
    (hbudget : ∀ ω, ∑ j : Freq p, ‖beta (φ ω) j.1‖ ^ (b / ν) ≤ C * N ω ^ b)
    {m : ℕ} (θ : Fin m → Ω) (hθ : ∑ i, N (θ i) ^ b ≤ 1) :
    margin (netTable φ θ) ≤ C ^ (ν / b) * GammaP p (ν / b) := by
  have hν : 0 < ν := lt_of_lt_of_le hb hbν
  have h := thm_universal hp φ N hb hbν hC hN hbudget θ
  have hs1 : 1 ≤ ν / b := (one_le_div hb).2 hbν
  have hNb : 0 ≤ ∑ i, N (θ i) ^ b := Finset.sum_nonneg fun i _ => Real.rpow_nonneg (hN _) _
  have h1 : (∑ i, N (θ i) ^ b) ^ (ν / b) ≤ 1 :=
    Real.rpow_le_one hNb hθ (by positivity)
  have hK : 0 ≤ C ^ (ν / b) * GammaP p (ν / b) :=
    mul_nonneg (Real.rpow_nonneg hC _) (GammaP_pos p hp hs1).le
  calc margin (netTable φ θ) ≤ C ^ (ν / b) * GammaP p (ν / b) * (∑ i, N (θ i) ^ b) ^ (ν / b) := h
    _ ≤ C ^ (ν / b) * GammaP p (ν / b) * 1 := mul_le_mul_of_nonneg_left h1 hK
    _ = C ^ (ν / b) * GammaP p (ν / b) := mul_one _

/-- The paper's constant `C_b = sup_{‖ω‖ ≤ 1} ∑_{j ∈ F_p} |β_j(ω)|^{b/ν}`. -/
def Cb {Ω : Type*} (φ : Ω → Table p n) (N : Ω → ℝ) (b ν : ℝ) : ℝ :=
  sSup ((fun ω => ∑ j : Freq p, ‖beta (φ ω) j.1‖ ^ (b / ν)) '' {ω | N ω ≤ 1})

/-- Homogeneity gives the single-neuron budget with `C = C_b`: if rescaling a neuron by
`λ > 0` (an arbitrary operation `sc λ` on parameters) multiplies its table by `λ^ν` and its size
by `λ`, and the defining set of `C_b` is bounded, then
`∑_j |β_j(φ ω)|^{b/ν} ≤ C_b N(ω)^b` for every `ω`. -/
theorem budget_le_Cb {Ω : Type*} (φ : Ω → Table p n) (N : Ω → ℝ) (hN : ∀ ω, 0 ≤ N ω)
    (sc : ℝ → Ω → Ω) {b ν : ℝ} (hb : 0 < b) (hν : 0 < ν)
    (hφ : ∀ l : ℝ, 0 < l → ∀ ω, φ (sc l ω) = fun x c => l ^ ν * φ ω x c)
    (hNsc : ∀ l : ℝ, 0 < l → ∀ ω, N (sc l ω) = l * N ω)
    (hbdd : BddAbove ((fun ω => ∑ j : Freq p, ‖beta (φ ω) j.1‖ ^ (b / ν)) '' {ω | N ω ≤ 1}))
    (ω : Ω) :
    ∑ j : Freq p, ‖beta (φ ω) j.1‖ ^ (b / ν) ≤ Cb φ N b ν * N ω ^ b := by
  set F : Ω → ℝ := fun ω => ∑ j : Freq p, ‖beta (φ ω) j.1‖ ^ (b / ν) with hF
  have hF0 : ∀ ω, 0 ≤ F ω := fun ω =>
    Finset.sum_nonneg fun j _ => Real.rpow_nonneg (norm_nonneg _) _
  have hscale : ∀ l : ℝ, 0 < l → ∀ ω, F (sc l ω) = l ^ b * F ω := by
    intro l hl ω
    simp only [hF]
    rw [hφ l hl ω, Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    have hlν : 0 < l ^ ν := Real.rpow_pos_of_pos hl ν
    rw [beta_smul (φ ω) (l ^ ν), norm_mul, Complex.norm_real, Real.norm_eq_abs,
      abs_of_pos hlν, Real.mul_rpow hlν.le (norm_nonneg _), ← Real.rpow_mul hl.le,
      mul_div_cancel₀ b hν.ne']
  have hle : ∀ ω, N ω ≤ 1 → F ω ≤ Cb φ N b ν := fun ω h => le_csSup hbdd ⟨ω, h, rfl⟩
  show F ω ≤ Cb φ N b ν * N ω ^ b
  by_cases hN0 : N ω = 0
  · rw [hN0, Real.zero_rpow hb.ne', mul_zero]
    by_contra hpos
    replace hpos : 0 < F ω := lt_of_not_ge hpos
    set K := |Cb φ N b ν| + 1 with hK
    have hK0 : 0 < K := by positivity
    set l := (K / F ω) ^ (1 / b) with hl
    have hl0 : 0 < l := Real.rpow_pos_of_pos (div_pos hK0 hpos) _
    have h1 := hle (sc l ω) (by rw [hNsc l hl0, hN0, mul_zero]; exact zero_le_one)
    rw [hscale l hl0] at h1
    have hlb : l ^ b = K / F ω := by
      rw [hl, ← Real.rpow_mul (div_nonneg hK0.le hpos.le), one_div_mul_cancel hb.ne',
        Real.rpow_one]
    rw [hlb, div_mul_cancel₀ _ hpos.ne'] at h1
    linarith [le_abs_self (Cb φ N b ν)]
  · have hNpos : 0 < N ω := lt_of_le_of_ne (hN ω) (Ne.symm hN0)
    set l := 1 / N ω with hl
    have hl0 : 0 < l := by positivity
    have h1 := hle (sc l ω) (by rw [hNsc l hl0, hl, one_div_mul_cancel hN0])
    rw [hscale l hl0] at h1
    have hlb : l ^ b * N ω ^ b = 1 := by
      rw [← Real.mul_rpow hl0.le (hN ω), hl, one_div_mul_cancel hN0, Real.one_rpow]
    calc F ω = (l ^ b * N ω ^ b) * F ω := by rw [hlb, one_mul]
      _ = N ω ^ b * (l ^ b * F ω) := by ring
      _ ≤ N ω ^ b * Cb φ N b ν := mul_le_mul_of_nonneg_left h1 (Real.rpow_nonneg (hN ω) _)
      _ = Cb φ N b ν * N ω ^ b := mul_comm _ _

/-- **Theorem `thm:universal`**, paper form: with `C_b` the supremum over unit-size neurons and
`ν`-homogeneous neurons, every network satisfies `h(θ) ≤ C_b^{ν/b} Γ_{ν/b}(p) ‖θ‖_{2,b}^ν`
(equivalently `γ*_b ≤ C_b^{ν/b} Γ_{ν/b}(p)`), for every width. -/
theorem thm_universal_Cb (hp : 3 ≤ p) {Ω : Type*} [Nonempty Ω] (φ : Ω → Table p n)
    (N : Ω → ℝ) (hN : ∀ ω, 0 ≤ N ω) (sc : ℝ → Ω → Ω) {b ν : ℝ} (hb : 0 < b) (hbν : b ≤ ν)
    (hφ : ∀ l : ℝ, 0 < l → ∀ ω, φ (sc l ω) = fun x c => l ^ ν * φ ω x c)
    (hNsc : ∀ l : ℝ, 0 < l → ∀ ω, N (sc l ω) = l * N ω)
    (hbdd : BddAbove ((fun ω => ∑ j : Freq p, ‖beta (φ ω) j.1‖ ^ (b / ν)) '' {ω | N ω ≤ 1}))
    {m : ℕ} (θ : Fin m → Ω) :
    margin (netTable φ θ) ≤
      Cb φ N b ν ^ (ν / b) * GammaP p (ν / b) * (∑ i, N (θ i) ^ b) ^ (ν / b) := by
  have hν : 0 < ν := lt_of_lt_of_le hb hbν
  have hC : 0 ≤ Cb φ N b ν := by
    obtain ⟨ω0⟩ := (inferInstance : Nonempty Ω)
    have hl : 0 < 1 / (N ω0 + 1) := by have := hN ω0; positivity
    have hmem : N (sc (1 / (N ω0 + 1)) ω0) ≤ 1 := by
      rw [hNsc _ hl]
      have := hN ω0
      rw [div_mul_eq_mul_div, one_mul, div_le_one (by linarith)]
      linarith
    exact le_trans (Finset.sum_nonneg fun j _ => Real.rpow_nonneg (norm_nonneg _) _)
      (le_csSup hbdd ⟨_, hmem, rfl⟩)
  exact thm_universal hp φ N hb hbν hC hN
    (budget_le_Cb φ N hN sc hb hν hφ hNsc hbdd) θ

end Universal

end Heracles
