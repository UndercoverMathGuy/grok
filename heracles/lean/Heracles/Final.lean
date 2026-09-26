import Heracles.NetScaling
import Heracles.DenseNet
import Heracles.Shapes

/-!
# Final, unconditional statements for product networks

The theorems of `Cyclic.lean` / `NetScaling.lean` take the product single-neuron budget
(`ProdBudget`) as a hypothesis; here it is discharged by `prod_budget` (Product.lean), valid for
`2 ≤ b ≤ ν`.
-/

open Real Finset

noncomputable section

set_option linter.unusedSectionVars false

namespace Heracles

variable {p n : ℕ} [Fact p.Prime]

lemma prodBudget_holds {b : ℝ} (hb2 : 2 ≤ b) (hbν : b ≤ n + 1) : ProdBudget p n b :=
  fun ω => prod_budget ω hb2 hbν

/-- **`thm:universal` for product networks** (`C_b^{ν/b} = B_p`):
`h(θ) ≤ B_p Γ_{ν/b}(p) ‖θ‖_{2,b}^ν` for every product network, `2 ≤ b ≤ ν`. -/
theorem thm_universal_prod (hp : 3 ≤ p) {b : ℝ} (hb2 : 2 ≤ b) (hbν : b ≤ n + 1) {m : ℕ}
    (θ : Fin m → ProdParam p n) :
    margin (netTable prodNeuron θ) ≤
      Bp p n * GammaP p (((n : ℝ) + 1) / b) * normPow b θ ^ (((n : ℝ) + 1) / b) := by
  have hb : 0 < b := by linarith
  have h := thm_universal hp prodNeuron (fun ω => Real.sqrt (normSq ω)) hb hbν
    (Real.rpow_nonneg (Bp_pos (p := p) (n := n)).le _) (fun ω => Real.sqrt_nonneg _)
    (prodBudget_holds hb2 hbν) θ
  rwa [← Real.rpow_mul (Bp_pos (p := p) (n := n)).le,
    show b / ((n : ℝ) + 1) * (((n : ℝ) + 1) / b) = 1 by field_simp, Real.rpow_one] at h

/-- **`thm:cyclic`, upper bound.** Every interference-free product network satisfies
`h(θ) ≤ ν^{-(ν-b)/b} B_p Γ_{ν/b}(p) ‖θ‖_{2,b}^ν`, `2 ≤ b ≤ ν`. -/
theorem thm_cyclic_upper (hp : 3 ≤ p) {b : ℝ} (hb2 : 2 ≤ b) (hbν : b ≤ n + 1) {m : ℕ}
    (θ : Fin m → ProdParam p n) (hIF : InterferenceFree (netTable prodNeuron θ)) :
    margin (netTable prodNeuron θ) ≤ valueIF p n b * normPow b θ ^ (((n : ℝ) + 1) / b) :=
  thm_cyclic_upper_of_budget hp θ hIF (by linarith) hbν (prodBudget_holds hb2 hbν)

lemma betaT_smul [NeZero p] (f : Table p n) (a : ℝ) (j : ℕ) (T : Finset (Fin n)) :
    betaT (fun x c => a * f x c) j T = (a : ℂ) * betaT f j T := by
  unfold betaT fhat
  rw [mul_left_comm (a : ℂ)]
  congr 1
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun c _ => ?_
  push_cast; ring

lemma interferenceFree_smul [NeZero p] {f : Table p n} (hf : InterferenceFree f) (a : ℝ) :
    InterferenceFree (fun x c => a * f x c) := by
  intro j hj T hT
  rw [betaT_smul, hf j hj T hT, mul_zero]

/-- **`thm:cyclic`, value.** For a maximizer `y*` of `Γ_{ν/b}(p)` and width `m ≥ ν |spt(y*)|`,
the maximum of `h(θ)` over interference-free product networks of width `m` with
`‖θ‖_{2,b} ≤ 1` is exactly `γ*_{b,IF} = ν^{-(ν-b)/b} B_p Γ_{ν/b}(p)` (`2 ≤ b ≤ ν`). -/
theorem thm_cyclic_value (hp : 3 ≤ p) {b : ℝ} (hb2 : 2 ≤ b) (hbν : b ≤ n + 1)
    {y : Freq p → ℝ} (hy : IsMaximizer (covP p) (((n : ℝ) + 1) / b) y) {m : ℕ}
    (hm : (n + 1) * (supp y).card ≤ m) :
    IsGreatest ((fun θ : Fin m → ProdParam p n => margin (netTable prodNeuron θ)) ''
      {θ | InterferenceFree (netTable prodNeuron θ) ∧ normPow b θ ≤ 1}) (valueIF p n b) := by
  have hb : 0 < b := by linarith
  have hν : (0 : ℝ) < n + 1 := by positivity
  have hs1 : 1 ≤ ((n : ℝ) + 1) / b := (one_le_div hb).2 hbν
  have hV : 0 < valueIF p n b := by
    unfold valueIF
    have := GammaP_pos p hp hs1
    have := Bp_pos (p := p) (n := n)
    positivity
  constructor
  · -- attained: normalize the network of `thm_cyclic_attain`
    obtain ⟨θ, hIF, hpos, hval⟩ := thm_cyclic_attain hp hb hbν hy hm
    set N := normPow b θ
    set lam := N ^ (-(1 / b)) with hlam
    have hlam0 : 0 < lam := Real.rpow_pos_of_pos hpos _
    have hnorm : normPow b (fun i => smulParam lam (θ i)) = 1 := by
      rw [normPow_smul hlam0.le, hlam, ← Real.rpow_mul hpos.le]
      have : -(1 / b) * b = -1 := by field_simp
      rw [this, Real.rpow_neg_one, inv_mul_cancel₀ hpos.ne']
    have htab : netTable prodNeuron (fun i => smulParam lam (θ i)) =
        fun x c => lam ^ (n + 1) * netTable prodNeuron θ x c := by
      funext x c; exact netTable_smul lam θ x c
    have hpow : lam ^ (n + 1) = (N ^ (((n : ℝ) + 1) / b))⁻¹ := by
      rw [hlam, ← Real.rpow_natCast, ← Real.rpow_mul hpos.le, ← Real.rpow_neg hpos.le]
      congr 1; push_cast; ring
    refine ⟨fun i => smulParam lam (θ i), ⟨?_, hnorm.le⟩, ?_⟩
    · rw [htab]; exact interferenceFree_smul hIF _
    · show margin (netTable prodNeuron (fun i => smulParam lam (θ i))) = valueIF p n b
      rw [htab, margin_smul (pow_nonneg hlam0.le _), hpow, ← hval]
      ring
  · -- upper bound
    rintro _ ⟨θ, ⟨hIF, hθ⟩, rfl⟩
    have h := thm_cyclic_upper hp hb2 hbν θ hIF
    have hN0 : 0 ≤ normPow b θ :=
      Finset.sum_nonneg fun i _ => Real.rpow_nonneg (Real.sqrt_nonneg _) _
    have h1 : normPow b θ ^ (((n : ℝ) + 1) / b) ≤ 1 := Real.rpow_le_one hN0 hθ (by positivity)
    calc margin (netTable prodNeuron θ) ≤ valueIF p n b * normPow b θ ^ (((n : ℝ) + 1) / b) := h
      _ ≤ valueIF p n b * 1 := mul_le_mul_of_nonneg_left h1 hV.le
      _ = valueIF p n b := mul_one _

/-- **`thm:cyclic`, structure (`b < ν`).** Any interference-free product network attaining the
value: (1) has amplitudes proportional to a maximizer `y⋆`, using exactly `spt(y⋆)`;
(2) every used frequency is carried by exactly `ν` neurons with `|β_ij| = A_j/ν` and pairwise
orthogonal chain vectors; no neuron carries an unused frequency. -/
theorem thm_cyclic_structure (hp : 3 ≤ p) {b : ℝ} (hb2 : 2 ≤ b) (hbν : b < n + 1) {m : ℕ}
    (θ : Fin m → ProdParam p n) (hIF : InterferenceFree (netTable prodNeuron θ))
    (hpos : 0 < normPow b θ)
    (heq : margin (netTable prodNeuron θ) = valueIF p n b * normPow b θ ^ (((n : ℝ) + 1) / b)) :
    (∃ y, IsMaximizer (covP p) (((n : ℝ) + 1) / b) y ∧ ∃ c : ℝ, 0 < c ∧
        (∀ j : Freq p, amp (netTable prodNeuron θ) j.1 = c * y j) ∧
        (Finset.univ.filter fun j : Freq p => amp (netTable prodNeuron θ) j.1 ≠ 0) = supp y) ∧
    (∀ j : Freq p, amp (netTable prodNeuron θ) j.1 ≠ 0 →
      ∃ ξ : Fin m → Fin n → ℂ, (∀ i t, ‖ξ i t‖ = 1) ∧
        (∀ i T, betaT (prodNeuron (θ i)) j.1 T = beta (prodNeuron (θ i)) j.1 * ∏ t ∈ T, ξ i t) ∧
        (Finset.univ.filter fun i => beta (prodNeuron (θ i)) j.1 ≠ 0).card = n + 1 ∧
        (∀ i, beta (prodNeuron (θ i)) j.1 ≠ 0 →
          ‖beta (prodNeuron (θ i)) j.1‖ = amp (netTable prodNeuron θ) j.1 / (n + 1)) ∧
        (∀ i i', beta (prodNeuron (θ i)) j.1 ≠ 0 → beta (prodNeuron (θ i')) j.1 ≠ 0 → i ≠ i' →
          ∑ k, star (Cost.chain ξ i k) * Cost.chain ξ i' k = 0)) ∧
    (∀ j : Freq p, amp (netTable prodNeuron θ) j.1 = 0 → ∀ i, beta (prodNeuron (θ i)) j.1 = 0) :=
  thm_cyclic_structure_of_budget hp (by linarith) hbν (prodBudget_holds hb2 hbν.le) θ hIF hpos heq

/-- **`thm:scaling`, network statement (unconditional).** For product networks, `2 ≤ b < ν`,
every odd prime `p` and every width `m ≥ ν⌈8 ln p⌉`:
`c p^{-ν/2}(ln p)^{1-ν/b} ≤ γ*_b(p) ≤ C p^{-ν/2}(ln p)^{1-ν/b}` with
`c = ν^{1-s}·2(2ν)^{-ν/2}·½·9^{1-s}` and `C = 2(2ν)^{-ν/2}·(3/2)(2a_s)^s`, `s = ν/b`. -/
theorem thm_scaling_net (hp : 3 ≤ p) {b : ℝ} (hb2 : 2 ≤ b) (hbν : b < n + 1) {m : ℕ}
    (hm : (n + 1) * ⌈8 * Real.log p⌉₊ ≤ m) :
    let ν : ℝ := (n : ℝ) + 1
    let s : ℝ := ν / b
    let K0 : ℝ := 2 * (2 * ν) ^ (-(ν / 2))
    ν ^ (1 - s) * K0 * ((1 / 2) * 9 ^ (1 - s)) * ((p : ℝ) ^ (-(ν / 2)) * Real.log p ^ (1 - s)) ≤
        gammaStarW p n m b ∧
      gammaStarW p n m b ≤
        K0 * ((3 / 2) * (2 * aS s) ^ s) * ((p : ℝ) ^ (-(ν / 2)) * Real.log p ^ (1 - s)) :=
  thm_scaling_network hp (by linarith) hbν (prodBudget_holds hb2 hbν.le) hm

/-- **`cor:few` for product networks** (`C_b^{ν/b} = B_p`): with at most `K ≥ 1` frequencies,
`h(θ)/‖θ‖_{2,b}^ν ≤ B_p (3/2) g_s(ln p/K) (ln p)^{1-s}`. -/
theorem cor_few_prod (hp : 3 ≤ p) {b : ℝ} (hb2 : 2 ≤ b) (hbν : b < n + 1) {m : ℕ}
    (θ : Fin m → ProdParam p n) {K : ℕ} (hK0 : 0 < K)
    (hK : (Finset.univ.filter fun j : Freq p => amp (netTable prodNeuron θ) j.1 ≠ 0).card ≤ K) :
    margin (netTable prodNeuron θ) ≤ Bp p n * normPow b θ ^ (((n : ℝ) + 1) / b) *
      ((3 / 2) * gK (((n : ℝ) + 1) / b) (Real.log p / K) * Real.log p ^ (1 - ((n : ℝ) + 1) / b)) := by
  have hb : 0 < b := by linarith
  have h := cor_few hp prodNeuron (fun ω => Real.sqrt (normSq ω)) hb hbν
    (Real.rpow_nonneg (Bp_pos (p := p) (n := n)).le _) (fun ω => Real.sqrt_nonneg _)
    (prodBudget_holds hb2 hbν.le) θ hK0 hK
  rwa [← Real.rpow_mul (Bp_pos (p := p) (n := n)).le,
    show b / ((n : ℝ) + 1) * (((n : ℝ) + 1) / b) = 1 by field_simp, Real.rpow_one] at h

/-- **`cor:few`, vanishing fraction for product networks (unconditional).** -/
theorem cor_few_ratio_prod (hp : 3 ≤ p) {b : ℝ} (hb2 : 2 ≤ b) (hbν : b < n + 1) {m m' : ℕ}
    (hm' : (n + 1) * ⌈8 * Real.log p⌉₊ ≤ m') (θ : Fin m → ProdParam p n)
    (hpos : 0 < normPow b θ) {K : ℕ} (hK0 : 0 < K)
    (hK : (Finset.univ.filter fun j : Freq p => amp (netTable prodNeuron θ) j.1 ≠ 0).card ≤ K) :
    margin (netTable prodNeuron θ) / normPow b θ ^ (((n : ℝ) + 1) / b) ≤
      3 * (9 * ((n : ℝ) + 1)) ^ (((n : ℝ) + 1) / b - 1) *
        gK (((n : ℝ) + 1) / b) (Real.log p / K) * gammaStarW p n m' b :=
  cor_few_ratio hp (by linarith) hbν (prodBudget_holds hb2 hbν.le) hm' θ hpos hK0 hK

end Heracles
