import Heracles.Final

/-!
# Power neurons (appendix `app:models`, "Power neuron"; `lem:sign` for power neurons)

The power neuron is `φ({u_1,…,u_n,w}, x) = (u_1(a_1) + ⋯ + u_n(a_n))^n · w(c)`, `ν = n + 1`.
At every mode whose operand frequencies are all non-zero, only the multilinear monomial of the
power survives, with multinomial coefficient `n!`; so `β_j`, `β_{j,T}` are `n!` times those of the
product neuron with the same weights, the sign rule holds with the same unit numbers, and
`C_b^{ν/b} ≤ n! B_p`.
-/

open Real Finset

noncomputable section

set_option linter.unusedSectionVars false

namespace Heracles

variable {p n : ℕ}

/-- The power neuron `(u_1(a_1) + ⋯ + u_n(a_n))^n · w(c)`. -/
def powNeuron (ω : ProdParam p n) : Table p n := fun x c => (∑ t, ω.u t (x t)) ^ n * ω.w c

/-- Multiplicity of `t` in the word `σ`. -/
def mult (σ : Fin n → Fin n) (t : Fin n) : ℕ := (Finset.univ.filter fun i => σ i = t).card

/-- The product neuron with operand weights `u_t^{m_t}` (`m_t = mult σ t`). -/
def powParam (ω : ProdParam p n) (σ : Fin n → Fin n) : ProdParam p n :=
  ⟨fun t a => ω.u t a ^ mult σ t, ω.w⟩

lemma prod_comp_eq_mult (σ : Fin n → Fin n) (g : Fin n → ℝ) :
    ∏ i, g (σ i) = ∏ t, g t ^ mult σ t := by
  classical
  rw [← Finset.prod_fiberwise Finset.univ σ (fun i => g (σ i))]
  refine Finset.prod_congr rfl fun t _ => ?_
  rw [Finset.prod_congr rfl (fun i hi => by rw [(Finset.mem_filter.1 hi).2]), Finset.prod_const]
  rfl

/-- Multinomial expansion: the power neuron is the sum over words `σ : [n] → [n]` of product
neurons with weights `u_t^{m_t(σ)}`. -/
lemma powNeuron_expand (ω : ProdParam p n) (x : Inp p n) (c : ZMod p) :
    powNeuron ω x c = ∑ σ : Fin n → Fin n, prodNeuron (powParam ω σ) x c := by
  classical
  unfold powNeuron prodNeuron powParam
  rw [← Finset.sum_mul]
  congr 1
  rw [show (∑ t, ω.u t (x t)) ^ n = ∏ _i : Fin n, ∑ t, ω.u t (x t) by
    rw [Finset.prod_const, Finset.card_univ, Fintype.card_fin]]
  rw [Finset.prod_univ_sum, Fintype.piFinset_univ]
  exact Finset.sum_congr rfl fun σ _ => prod_comp_eq_mult σ (fun t => ω.u t (x t))

variable [NeZero p]

/-- The transform is additive in the table. -/
lemma fhat_sum {ι : Type*} (s : Finset ι) (g : ι → Table p n) (k : Fin n → ZMod p) (l : ZMod p) :
    fhat (fun x c => ∑ i ∈ s, g i x c) k l = ∑ i ∈ s, fhat (g i) k l := by
  unfold fhat
  simp only [Complex.ofReal_sum, Finset.sum_mul]
  conv_rhs => rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun x _ => ?_
  exact Finset.sum_comm

/-- `∑_a ρ^{-ka} = 0` for `k ≠ 0`. -/
lemma dft_one_eq_zero {k : ZMod p} (hk : k ≠ 0) : dft (fun _ => (1 : ℝ)) k = 0 := by
  unfold dft
  have h : ∀ a : ZMod p, ((1 : ℝ) : ℂ) * rho (-(k * a)) = ZMod.stdAddChar (a * (-k)) := by
    intro a
    rw [Complex.ofReal_one, one_mul]
    unfold rho
    congr 1
    ring
  rw [Finset.sum_congr rfl fun a _ => h a,
    AddChar.sum_mulShift _ (ZMod.isPrimitive_stdAddChar p), ite_eq_right (neg_ne_zero.2 hk),
    Nat.cast_zero]

lemma card_bijective_self (n : ℕ) :
    (Finset.univ.filter fun σ : Fin n → Fin n => Function.Bijective σ).card = n.factorial := by
  classical
  rw [← Fintype.card_subtype]
  let e : {σ : Fin n → Fin n // Function.Bijective σ} ≃ Equiv.Perm (Fin n) :=
    { toFun := fun σ => Equiv.ofBijective σ.1 σ.2
      invFun := fun e => ⟨e, e.bijective⟩
      left_inv := fun σ => rfl
      right_inv := fun e => Equiv.ext fun _ => rfl }
  rw [Fintype.card_congr e, Fintype.card_perm, Fintype.card_fin]

/-- The term of word `σ` at a mode with all operand frequencies non-zero: it vanishes unless `σ`
is a bijection, in which case it is the product-neuron transform. -/
lemma fhat_powParam (ω : ProdParam p n) (σ : Fin n → Fin n) (k : Fin n → ZMod p)
    (hk : ∀ t, k t ≠ 0) (l : ZMod p) :
    fhat (prodNeuron (powParam ω σ)) k l =
      if Function.Bijective σ then fhat (prodNeuron ω) k l else 0 := by
  classical
  rw [fhat_prodNeuron, fhat_prodNeuron]
  split_ifs with hσ
  · have hm : ∀ t, mult σ t = 1 := by
      intro t
      obtain ⟨i0, hi0⟩ := hσ.2 t
      unfold mult
      rw [Finset.card_eq_one]
      refine ⟨i0, ?_⟩
      ext i
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton]
      constructor
      · intro hi; exact hσ.1 (hi.trans hi0.symm)
      · rintro rfl; exact hi0
    have hu : ∀ t, (powParam ω σ).u t = ω.u t := by
      intro t; funext a; simp [powParam, hm t]
    simp only [hu]
    rfl
  · -- `σ` is not surjective: some operand slot `t` has multiplicity zero
    have hns : ¬ Function.Surjective σ := by
      intro hs
      exact hσ ⟨Finite.injective_iff_surjective.2 hs, hs⟩
    obtain ⟨t, ht⟩ : ∃ t, ∀ i, σ i ≠ t := by
      by_contra h
      push Not at h
      exact hns h
    have hm : mult σ t = 0 := by
      unfold mult
      rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
      intro i _; exact ht i
    have hzero : dft ((powParam ω σ).u t) (k t) = 0 := by
      have : (powParam ω σ).u t = fun _ => (1 : ℝ) := by
        funext a; simp [powParam, hm]
      rw [this]
      exact dft_one_eq_zero (hk t)
    rw [Finset.prod_eq_zero (Finset.mem_univ t) hzero, zero_mul]

/-- **Power neuron transform.** At every mode `(k; l)` with all operand frequencies `k_t ≠ 0`,
`f̂_{pow}(k; l) = n! · f̂_{prod}(k; l)` (same weights). -/
theorem fhat_powNeuron (ω : ProdParam p n) (k : Fin n → ZMod p) (hk : ∀ t, k t ≠ 0)
    (l : ZMod p) :
    fhat (powNeuron ω) k l = (n.factorial : ℂ) * fhat (prodNeuron ω) k l := by
  classical
  have e : powNeuron ω = fun x c => ∑ σ : Fin n → Fin n, prodNeuron (powParam ω σ) x c := by
    funext x c; exact powNeuron_expand ω x c
  rw [e, fhat_sum, Finset.sum_congr rfl fun σ _ => fhat_powParam ω σ k hk l,
    Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const, card_bijective_self,
    nsmul_eq_mul]

/-- **`app:models`, power neuron.** `β_j(pow) = n! · β_j(prod)` for `j ≢ 0 (mod p)`. -/
theorem beta_powNeuron (ω : ProdParam p n) {j : ℕ} (hj : (j : ZMod p) ≠ 0) :
    beta (powNeuron ω) j = (n.factorial : ℂ) * beta (prodNeuron ω) j := by
  unfold beta
  rw [fhat_powNeuron ω _ (fun _ => hj)]
  ring

/-- `β_{j,T}(pow) = n! · β_{j,T}(prod)` for `j ≢ 0 (mod p)`. -/
theorem betaT_powNeuron (ω : ProdParam p n) {j : ℕ} (hj : (j : ZMod p) ≠ 0)
    (T : Finset (Fin n)) :
    betaT (powNeuron ω) j T = (n.factorial : ℂ) * betaT (prodNeuron ω) j T := by
  unfold betaT
  rw [fhat_powNeuron ω _ (fun t => by split_ifs; exacts [neg_ne_zero.2 hj, hj])]
  ring

/-- Frequencies in `F_p` are non-zero in `ℤ_p`. -/
lemma freq_cast_ne_zero (j : Freq p) : (j.1 : ZMod p) ≠ 0 := by
  have hj := Finset.mem_Icc.1 j.2
  rw [Ne, ZMod.natCast_eq_zero_iff]
  intro hdvd
  have := Nat.le_of_dvd (by omega) hdvd
  omega

/-- **`lem:sign` for power neurons**, with the same unit numbers as for the product neuron with
the same weights: whenever `ξ` is a sign rule for `prodNeuron ω` at `j`, it is one for
`powNeuron ω`. -/
theorem lem_sign_pow_same (ω : ProdParam p n) {j : ℕ} (hj : (j : ZMod p) ≠ 0) (ξ : Fin n → ℂ)
    (hξ : ∀ T : Finset (Fin n), betaT (prodNeuron ω) j T = beta (prodNeuron ω) j * ∏ t ∈ T, ξ t) :
    ∀ T : Finset (Fin n), betaT (powNeuron ω) j T = beta (powNeuron ω) j * ∏ t ∈ T, ξ t := by
  intro T
  rw [betaT_powNeuron ω hj, beta_powNeuron ω hj, hξ T]
  ring

/-- **Lemma `lem:sign` (power neurons).** For each power neuron and frequency `j ≢ 0` there are
unit complex numbers `ξ_t` with `β_{j,T} = β_j ∏_{t∈T} ξ_t` for every `T ⊆ [n]`. -/
theorem lem_sign_pow (ω : ProdParam p n) {j : ℕ} (hj : (j : ZMod p) ≠ 0) :
    ∃ ξ : Fin n → ℂ, (∀ t, ‖ξ t‖ = 1) ∧
      ∀ T : Finset (Fin n), betaT (powNeuron ω) j T = beta (powNeuron ω) j * ∏ t ∈ T, ξ t := by
  obtain ⟨ξ, h1, h2⟩ := lem_sign ω j
  exact ⟨ξ, h1, lem_sign_pow_same ω hj ξ h2⟩

variable [Fact p.Prime]

/-- **Single-neuron budget of the power neuron** (`C_b^{ν/b} ≤ n! B_p`), `2 ≤ b ≤ ν`. -/
theorem pow_budget (ω : ProdParam p n) {b : ℝ} (hb2 : 2 ≤ b) (hbν : b ≤ n + 1) :
    ∑ j : Freq p, ‖beta (powNeuron ω) j.1‖ ^ (b / (n + 1)) ≤
      ((n.factorial : ℝ) * Bp p n) ^ (b / (n + 1)) * Real.sqrt (normSq ω) ^ b := by
  have hr : 0 ≤ b / ((n : ℝ) + 1) := by
    have : (0 : ℝ) < n + 1 := by positivity
    exact div_nonneg (by linarith) this.le
  have hF : (0 : ℝ) ≤ n.factorial := Nat.cast_nonneg _
  have hB : 0 ≤ Bp p n := (Bp_pos (p := p) (n := n)).le
  calc ∑ j : Freq p, ‖beta (powNeuron ω) j.1‖ ^ (b / (n + 1))
      = ∑ j : Freq p, (n.factorial : ℝ) ^ (b / (n + 1)) *
          ‖beta (prodNeuron ω) j.1‖ ^ (b / (n + 1)) := by
        refine Finset.sum_congr rfl fun j _ => ?_
        rw [beta_powNeuron ω (freq_cast_ne_zero j), norm_mul, Complex.norm_natCast,
          Real.mul_rpow hF (norm_nonneg _)]
    _ = (n.factorial : ℝ) ^ (b / (n + 1)) *
          ∑ j : Freq p, ‖beta (prodNeuron ω) j.1‖ ^ (b / (n + 1)) := (Finset.mul_sum _ _ _).symm
    _ ≤ (n.factorial : ℝ) ^ (b / (n + 1)) *
          (Bp p n ^ (b / (n + 1)) * Real.sqrt (normSq ω) ^ b) :=
        mul_le_mul_of_nonneg_left (prod_budget ω hb2 hbν) (Real.rpow_nonneg hF _)
    _ = ((n.factorial : ℝ) * Bp p n) ^ (b / (n + 1)) * Real.sqrt (normSq ω) ^ b := by
        rw [Real.mul_rpow hF hB]; ring

/-- **`thm:universal` for power networks.** Every power network satisfies
`h(θ) ≤ n! B_p Γ_{ν/b}(p) ‖θ‖_{2,b}^ν`, `2 ≤ b ≤ ν`. -/
theorem thm_universal_pow (hp : 3 ≤ p) {b : ℝ} (hb2 : 2 ≤ b) (hbν : b ≤ n + 1) {m : ℕ}
    (θ : Fin m → ProdParam p n) :
    margin (netTable powNeuron θ) ≤
      (n.factorial : ℝ) * Bp p n * GammaP p (((n : ℝ) + 1) / b) *
        normPow b θ ^ (((n : ℝ) + 1) / b) := by
  have hb : 0 < b := by linarith
  have hC0 : 0 ≤ (n.factorial : ℝ) * Bp p n :=
    mul_nonneg (Nat.cast_nonneg _) (Bp_pos (p := p) (n := n)).le
  have h := thm_universal hp powNeuron (fun ω => Real.sqrt (normSq ω)) hb hbν
    (Real.rpow_nonneg hC0 _) (fun ω => Real.sqrt_nonneg _) (fun ω => pow_budget ω hb2 hbν) θ
  rw [← Real.rpow_mul hC0,
    show b / ((n : ℝ) + 1) * (((n : ℝ) + 1) / b) = 1 by field_simp, Real.rpow_one] at h
  exact h

end Heracles
