import Heracles.Cyclic
import Heracles.Few

/-!
# Network-level consequences of `thm:scaling` and `cor:few` for product networks

`gammaStarW p n m b` is the maximum `L_{2,b}`-margin of product networks of width `m`
(`γ*_b = max_{‖θ‖_{2,b} ≤ 1} h(θ)`, paper Section `sec:approach`), defined as a supremum.
-/

open Real Finset

noncomputable section

set_option linter.unusedSectionVars false

namespace Heracles

variable {p n : ℕ} [Fact p.Prime]

/-- The maximum `L_{2,b}`-margin over product networks of width `m`:
`γ*_b = sup {h(θ) : ‖θ‖_{2,b}^b ≤ 1}`. -/
def gammaStarW (p n m : ℕ) [Fact p.Prime] (b : ℝ) : ℝ :=
  sSup ((fun θ : Fin m → ProdParam p n => margin (netTable prodNeuron θ)) ''
    {θ | normPow b θ ≤ 1})

/-- The single-neuron budget of the product neuron with constant `B_p` (paper: `C_b^{ν/b} = B_p`),
as a proposition. -/
def ProdBudget (p n : ℕ) [Fact p.Prime] (b : ℝ) : Prop :=
  ∀ ω : ProdParam p n, ∑ j : Freq p, ‖beta (prodNeuron ω) j.1‖ ^ (b / (n + 1)) ≤
    Bp p n ^ (b / (n + 1)) * Real.sqrt (normSq ω) ^ b

lemma Bp_pos : 0 < Bp p n := by
  unfold Bp
  have : (0 : ℝ) < p := by exact_mod_cast (Fact.out : p.Prime).pos
  positivity

/-- Every product network with `‖θ‖_{2,b} ≤ 1` has `h(θ) ≤ B_p Γ_{ν/b}(p)` (`thm:universal`). -/
theorem margin_le_universal (hp : 3 ≤ p) {b : ℝ} (hb : 0 < b) (hbν : b ≤ n + 1)
    (hbud : ProdBudget p n b) {m : ℕ} (θ : Fin m → ProdParam p n) (hθ : normPow b θ ≤ 1) :
    margin (netTable prodNeuron θ) ≤ Bp p n * GammaP p (((n : ℝ) + 1) / b) := by
  have hν : (0 : ℝ) < n + 1 := by positivity
  have h := thm_universal_unit hp prodNeuron (fun ω => Real.sqrt (normSq ω)) hb hbν
    (Real.rpow_nonneg (Bp_pos).le _) (fun ω => Real.sqrt_nonneg _) hbud θ hθ
  rwa [← Real.rpow_mul (Bp_pos).le,
    show b / ((n : ℝ) + 1) * (((n : ℝ) + 1) / b) = 1 by field_simp, Real.rpow_one] at h

lemma bddAbove_gammaStar (hp : 3 ≤ p) {b : ℝ} (hb : 0 < b) (hbν : b ≤ n + 1)
    (hbud : ProdBudget p n b) (m : ℕ) :
    BddAbove ((fun θ : Fin m → ProdParam p n => margin (netTable prodNeuron θ)) ''
      {θ | normPow b θ ≤ 1}) :=
  ⟨_, by rintro _ ⟨θ, hθ, rfl⟩; exact margin_le_universal hp hb hbν hbud θ hθ⟩

/-- Upper bound `γ*_b ≤ B_p Γ_{ν/b}(p)` at every width. -/
theorem gammaStar_le (hp : 3 ≤ p) {b : ℝ} (hb : 0 < b) (hbν : b ≤ n + 1)
    (hbud : ProdBudget p n b) (m : ℕ) :
    gammaStarW p n m b ≤ Bp p n * GammaP p (((n : ℝ) + 1) / b) := by
  apply csSup_le
  · refine ⟨_, fun _ => zeroParam p n, ?_, rfl⟩
    show normPow b (fun _ : Fin m => zeroParam p n) ≤ 1
    unfold normPow
    simp [normSq_zero, Real.zero_rpow hb.ne']
  · rintro _ ⟨θ, hθ, rfl⟩; exact margin_le_universal hp hb hbν hbud θ hθ

/-- Any network with positive norm gives the lower bound `γ*_b ≥ h(θ)/‖θ‖_{2,b}^ν`
(normalize by rescaling). -/
theorem le_gammaStar (hp : 3 ≤ p) {b : ℝ} (hb : 0 < b) (hbν : b ≤ n + 1)
    (hbud : ProdBudget p n b) {m : ℕ} (θ : Fin m → ProdParam p n) (hpos : 0 < normPow b θ) :
    margin (netTable prodNeuron θ) / normPow b θ ^ (((n : ℝ) + 1) / b) ≤ gammaStarW p n m b := by
  set N := normPow b θ
  set lam := N ^ (-(1 / b)) with hlam
  have hlam0 : 0 < lam := Real.rpow_pos_of_pos hpos _
  have hnorm : normPow b (fun i => smulParam lam (θ i)) = 1 := by
    rw [normPow_smul hlam0.le, hlam, ← Real.rpow_mul hpos.le]
    have : -(1 / b) * b = -1 := by field_simp
    rw [this, Real.rpow_neg_one, inv_mul_cancel₀ hpos.ne']
  have hmarg : margin (netTable prodNeuron (fun i => smulParam lam (θ i))) =
      lam ^ (n + 1) * margin (netTable prodNeuron θ) := by
    have : netTable prodNeuron (fun i => smulParam lam (θ i)) =
        fun x c => lam ^ (n + 1) * netTable prodNeuron θ x c := by
      funext x c; exact netTable_smul lam θ x c
    rw [this, margin_smul (pow_nonneg hlam0.le _)]
  have hpow : lam ^ (n + 1) = (N ^ (((n : ℝ) + 1) / b))⁻¹ := by
    rw [hlam, ← Real.rpow_natCast, ← Real.rpow_mul hpos.le, ← Real.rpow_neg hpos.le]
    congr 1; push_cast; ring
  have hmem : margin (netTable prodNeuron (fun i => smulParam lam (θ i))) ∈
      ((fun θ : Fin m → ProdParam p n => margin (netTable prodNeuron θ)) ''
        {θ | normPow b θ ≤ 1}) := ⟨_, hnorm.le, rfl⟩
  have := le_csSup (bddAbove_gammaStar hp hb hbν hbud m) hmem
  rw [hmarg, hpow] at this
  rw [div_eq_inv_mul]
  exact this

/-- `γ*_b` at width `m ≥ ν⌈8 ln p⌉` is at least `ν^{1-s} B_p · ½ ⌈8 ln p⌉^{1-s}`. -/
theorem gammaStar_ge (hp : 3 ≤ p) {b : ℝ} (hb : 0 < b) (hbν : b < n + 1)
    (hbud : ProdBudget p n b) {m : ℕ} (hm : (n + 1) * ⌈8 * Real.log p⌉₊ ≤ m) :
    ((n : ℝ) + 1) ^ (1 - ((n : ℝ) + 1) / b) * Bp p n *
        ((1 / 2) * (⌈8 * Real.log p⌉₊ : ℝ) ^ (1 - ((n : ℝ) + 1) / b)) ≤ gammaStarW p n m b := by
  have hν : (0 : ℝ) < n + 1 := by positivity
  set s := ((n : ℝ) + 1) / b with hsdef
  have hs1 : 1 < s := (one_lt_div hb).2 hbν
  obtain ⟨y, hyf, hsupp, hworst⟩ := exists_alloc_lower p hp hs1
  -- rescale `y` to a full budget
  have hwpos : 0 < worst (covP p) y := lt_of_lt_of_le (by
    have : 0 < ⌈8 * Real.log p⌉₊ := Nat.ceil_pos.2 (by have := log_p_pos hp; positivity)
    have : (0 : ℝ) < ⌈8 * Real.log p⌉₊ := by exact_mod_cast this
    positivity) hworst
  have hyne : y ≠ 0 := by intro h; rw [h, worst_zero] at hwpos; exact lt_irrefl _ hwpos
  have hB := budget_pos_of_ne (s := s) (by linarith) hyf.1 hyne
  set a := budget s y ^ (-s) with ha
  have ha0 : 0 < a := Real.rpow_pos_of_pos hB _
  have ha1 : 1 ≤ a := Real.one_le_rpow_of_pos_of_le_one_of_nonpos hB hyf.2 (by linarith)
  set y' := a • y with hy'
  have hy'0 : ∀ j, 0 ≤ y' j := fun j => by
    simp only [hy', Pi.smul_apply, smul_eq_mul]; exact mul_nonneg ha0.le (hyf.1 j)
  have hbud' : budget s y' = 1 := by
    rw [hy', budget_smul ha0.le hyf.1, ha, ← Real.rpow_mul hB.le]
    have e : -s * (1 / s) = -1 := by field_simp
    rw [e, Real.rpow_neg_one, inv_mul_cancel₀ hB.ne']
  have hsupp' : supp y' = supp y := by ext j; simp [supp, hy', ha0.ne']
  have hworst' : worst (covP p) y ≤ worst (covP p) y' := by
    rw [hy', worst_smul _ ha0.le]; nlinarith
  -- the block network on `y'`, at width `m`
  have hm' : (n + 1) * (supp y').card ≤ m := by
    rw [hsupp']; exact le_trans (Nat.mul_le_mul_left _ hsupp) hm
  obtain ⟨θ, htab, hnorm⟩ := exists_block_width (n := n) y' hb hm'
  have hpos : 0 < normPow b θ := by rw [hnorm]; exact normPow_blockNet_pos y' hy'0 hbud'
  have hval : margin (netTable prodNeuron θ) / normPow b θ ^ s =
      ((n : ℝ) + 1) ^ (-(((n : ℝ) + 1) - b) / b) * Bp p n * worst (covP p) y' := by
    rw [htab, hnorm]; exact blockNet_normalized_margin hb y' hy'0 hbud'
  have hle := le_gammaStar hp hb hbν.le hbud θ hpos
  rw [hval] at hle
  have e : -(((n : ℝ) + 1) - b) / b = 1 - s := by rw [hsdef]; field_simp; ring
  rw [e] at hle
  have hc : 0 ≤ ((n : ℝ) + 1) ^ (1 - s) * Bp p n :=
    mul_nonneg (Real.rpow_nonneg hν.le _) (Bp_pos).le
  calc ((n : ℝ) + 1) ^ (1 - s) * Bp p n * ((1 / 2) * (⌈8 * Real.log p⌉₊ : ℝ) ^ (1 - s))
      ≤ ((n : ℝ) + 1) ^ (1 - s) * Bp p n * worst (covP p) y' :=
        mul_le_mul_of_nonneg_left (hworst.trans hworst') hc
    _ ≤ gammaStarW p n m b := hle

/-- `B_p = 2 (2ν)^{-ν/2} p^{-ν/2}`. -/
lemma Bp_eq (p n : ℕ) : Bp p n = 2 * (2 * ((n : ℝ) + 1)) ^ (-(((n : ℝ) + 1) / 2)) *
    (p : ℝ) ^ (-(((n : ℝ) + 1) / 2)) := by
  unfold Bp
  rw [show 2 * (p : ℝ) * (n + 1) = (2 * ((n : ℝ) + 1)) * p by ring,
    Real.mul_rpow (by positivity) (Nat.cast_nonneg p)]
  ring

/-- **`thm:scaling`, network statement.** For product networks with `2 ≤ b < ν` (only `0 < b`
is used) and every width `m ≥ ν⌈8 ln p⌉`,
`c · p^{-ν/2} (ln p)^{1-ν/b} ≤ γ*_b(p) ≤ C · p^{-ν/2} (ln p)^{1-ν/b}` for every odd prime `p`,
with explicit constants depending only on `ν` and `b`:
`c = ν^{1-s} · 2(2ν)^{-ν/2} · ½ 9^{1-s}`, `C = 2(2ν)^{-ν/2} · (3/2)(2a_s)^s`, `s = ν/b`. -/
theorem thm_scaling_network (hp : 3 ≤ p) {b : ℝ} (hb : 0 < b) (hbν : b < n + 1)
    (hbud : ProdBudget p n b) {m : ℕ} (hm : (n + 1) * ⌈8 * Real.log p⌉₊ ≤ m) :
    let ν : ℝ := (n : ℝ) + 1
    let s : ℝ := ν / b
    let K0 : ℝ := 2 * (2 * ν) ^ (-(ν / 2))
    ν ^ (1 - s) * K0 * ((1 / 2) * 9 ^ (1 - s)) * ((p : ℝ) ^ (-(ν / 2)) * Real.log p ^ (1 - s)) ≤
        gammaStarW p n m b ∧
      gammaStarW p n m b ≤
        K0 * ((3 / 2) * (2 * aS s) ^ s) * ((p : ℝ) ^ (-(ν / 2)) * Real.log p ^ (1 - s)) := by
  intro ν s K0
  have hν : (0 : ℝ) < ν := by positivity
  have hs1 : 1 < s := (one_lt_div hb).2 hbν
  have hlog := log_p_pos hp
  have hBp : Bp p n = K0 * (p : ℝ) ^ (-(ν / 2)) := by rw [Bp_eq]
  constructor
  · -- lower bound
    have h1 := gammaStar_ge hp hb hbν hbud hm
    have h9 : (1 / 2) * (9 * Real.log p) ^ (1 - s) ≤
        (1 / 2) * (⌈8 * Real.log p⌉₊ : ℝ) ^ (1 - s) := by
      have hL : (0 : ℝ) < ⌈8 * Real.log p⌉₊ := by
        have : 0 < ⌈8 * Real.log p⌉₊ := Nat.ceil_pos.2 (by positivity)
        exact_mod_cast this
      have := ceil_le_nine_log p hp
      gcongr (1 / 2) * ?_
      exact Real.rpow_le_rpow_of_nonpos hL this (by linarith)
    have hc : 0 ≤ ν ^ (1 - s) * Bp p n := mul_nonneg (Real.rpow_nonneg hν.le _) (Bp_pos).le
    have h2 := mul_le_mul_of_nonneg_left h9 hc
    refine le_trans (le_of_eq ?_) (h2.trans h1)
    rw [hBp, Real.mul_rpow (by norm_num) hlog.le]
    ring
  · -- upper bound
    have h1 := gammaStar_le hp hb hbν.le hbud m
    have h2 := GammaP_upper hp hs1
    have h3 := mul_le_mul_of_nonneg_left h2 (Bp_pos (p := p) (n := n)).le
    refine (h1.trans h3).trans (le_of_eq ?_)
    rw [hBp]
    ring

/-- **`cor:few`, ratio form for product networks.** For product networks, the normalized margin
of a network with at most `K ≥ 1` frequencies is at most `3 (9ν)^{s-1} g_s(ln p/K)` times
`γ*_b(p)` (at any width `m' ≥ ν⌈8 ln p⌉` for `γ*`).  Together with `tendsto_gK`, `K = o(log p)`
attains a vanishing fraction of `γ*_b(p)`. -/
theorem cor_few_ratio (hp : 3 ≤ p) {b : ℝ} (hb : 0 < b) (hbν : b < n + 1)
    (hbud : ProdBudget p n b) {m m' : ℕ} (hm' : (n + 1) * ⌈8 * Real.log p⌉₊ ≤ m')
    (θ : Fin m → ProdParam p n) (hpos : 0 < normPow b θ) {K : ℕ} (hK0 : 0 < K)
    (hK : (Finset.univ.filter fun j : Freq p => amp (netTable prodNeuron θ) j.1 ≠ 0).card ≤ K) :
    margin (netTable prodNeuron θ) / normPow b θ ^ (((n : ℝ) + 1) / b) ≤
      3 * (9 * ((n : ℝ) + 1)) ^ (((n : ℝ) + 1) / b - 1) *
        gK (((n : ℝ) + 1) / b) (Real.log p / K) * gammaStarW p n m' b := by
  set ν : ℝ := (n : ℝ) + 1 with hνdef
  set s := ν / b with hsdef
  have hν : (0 : ℝ) < ν := by positivity
  have hs1 : 1 < s := (one_lt_div hb).2 hbν
  have hlog := log_p_pos hp
  have hB := Bp_pos (p := p) (n := n)
  have hfew := cor_few hp prodNeuron (fun ω => Real.sqrt (normSq ω)) hb hbν
    (Real.rpow_nonneg hB.le _) (fun ω => Real.sqrt_nonneg _) hbud θ hK0 hK
  rw [← Real.rpow_mul hB.le, show b / ((n : ℝ) + 1) * (ν / b) = 1 by rw [hνdef]; field_simp,
    Real.rpow_one] at hfew
  have hN : 0 < normPow b θ ^ s := Real.rpow_pos_of_pos hpos _
  have hg : 0 ≤ gK s (Real.log p / K) := by
    unfold gK
    have hKR : (0 : ℝ) < K := by exact_mod_cast hK0
    have hE1 : 1 < Real.exp (Real.log p / K / (2 * aS s)) :=
      Real.one_lt_exp_iff.2 (by unfold aS; positivity)
    have : 0 ≤ Real.log p / K / (Real.exp (Real.log p / K / (2 * aS s)) - 1) :=
      div_nonneg (by positivity) (by linarith)
    positivity
  have hlow := gammaStar_ge hp hb hbν hbud hm'
  have h9 : (1 / 2) * (9 * Real.log p) ^ (1 - s) ≤
      (1 / 2) * (⌈8 * Real.log p⌉₊ : ℝ) ^ (1 - s) := by
    have hL : (0 : ℝ) < ⌈8 * Real.log p⌉₊ := by
      have : 0 < ⌈8 * Real.log p⌉₊ := Nat.ceil_pos.2 (by positivity)
      exact_mod_cast this
    have := ceil_le_nine_log p hp
    gcongr (1 / 2) * ?_
    exact Real.rpow_le_rpow_of_nonpos hL this (by linarith)
  have hc : 0 ≤ ν ^ (1 - s) * Bp p n := mul_nonneg (Real.rpow_nonneg hν.le _) hB.le
  have hlow9 : ν ^ (1 - s) * Bp p n * ((1 / 2) * (9 * Real.log p) ^ (1 - s)) ≤ gammaStarW p n m' b :=
    (mul_le_mul_of_nonneg_left h9 hc).trans hlow
  have hk : 0 ≤ 3 * (9 * ν) ^ (s - 1) * gK s (Real.log p / K) :=
    mul_nonneg (mul_nonneg (by norm_num) (Real.rpow_nonneg (by positivity) _)) hg
  rw [div_le_iff₀ hN]
  calc margin (netTable prodNeuron θ)
      ≤ Bp p n * normPow b θ ^ s * ((3 / 2) * gK s (Real.log p / K) * Real.log p ^ (1 - s)) := hfew
    _ = 3 * (9 * ν) ^ (s - 1) * gK s (Real.log p / K) *
          (ν ^ (1 - s) * Bp p n * ((1 / 2) * (9 * Real.log p) ^ (1 - s))) * normPow b θ ^ s := by
        rw [Real.mul_rpow (by norm_num) hlog.le, Real.mul_rpow (by norm_num) hν.le]
        have e1 : (9 : ℝ) ^ (s - 1) * 9 ^ (1 - s) = 1 := by
          rw [← Real.rpow_add (by norm_num)]; simp
        have e2 : ν ^ (s - 1) * ν ^ (1 - s) = 1 := by
          rw [← Real.rpow_add hν]; simp
        linear_combination (-(3 / 2 * Bp p n * normPow b θ ^ s * gK s (Real.log p / K) *
          Real.log p ^ (1 - s))) * e1 + (-(3 / 2 * Bp p n * normPow b θ ^ s *
          gK s (Real.log p / K) * Real.log p ^ (1 - s)) * (9 : ℝ) ^ (s - 1) * 9 ^ (1 - s)) * e2
    _ ≤ 3 * (9 * ν) ^ (s - 1) * gK s (Real.log p / K) * gammaStarW p n m' b *
          normPow b θ ^ s := by gcongr

end Heracles
