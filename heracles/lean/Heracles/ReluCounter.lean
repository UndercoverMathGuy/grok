import Heracles.Product

/-!
# The ReLU single-neuron budget claimed in appendix `app:models` fails for `n = 2`

Paper, appendix `app:models`, paragraph "ReLU neuron": for
`φ({u_1,…,u_n,w}, x) = σ(∑_t u_t(a_t)) w(c)` (σ = ReLU, `ν = 2`), with `h(x) = σ(∑_t u_t(a_t))`
and `β_j = 2 p^{-(n+1)} ĥ(j,…,j) conj(ŵ(j))`, the paper claims
`∑_{j∈F_p} |ĥ(j,…,j) ŵ(j)| ≤ (p/4) ‖ω‖²`, i.e. `C_2 ≤ p^{-n}/2`.

We give an explicit counterexample at `p = 5`, `n = 2`:
`u_1 = u_2 = (−2, 2, −2, 2, 0)`, `w = (−2, 0, 3, −3, 1)` (values at `a = 0,…,4`), so
`‖ω‖² = 16 + 16 + 23 = 55`.  Then `∑_{j∈F_5} |ĥ(j,j)| |ŵ(j)| ≥ 72 > 68.75 = (5/4)·55`
(`relu_diag_claim_false`) and `∑_{j∈F_5} |β_j| > (5^{-2}/2)·55` (`relu_budget_claim_false`).

Proof: `∑_{j∈F_5} |G_j| ≥ Re ∑_{j∈F_5} G_j = ½ (Re ∑_{j∈ℤ_5} G_j − G_0)` with
`G_j = f̂(j,j;−j)` (conjugate symmetry), `∑_{j∈ℤ_p} G_j = p ∑_x f(x, y(x))` (orthogonality of
characters) and `G_0 = ∑_{x,c} f(x,c)`; the two finite integer sums are evaluated exactly.
-/

open Real Finset

noncomputable section

set_option linter.unusedSectionVars false

namespace Heracles

/-- The ReLU neuron `σ(∑_t u_t(a_t)) · w(c)` as a logit table. -/
def reluNeuron {p n : ℕ} (u : Fin n → ZMod p → ℝ) (w : ZMod p → ℝ) : Table p n :=
  fun x c => max (∑ t, u t (x t)) 0 * w c

section General

variable {p n : ℕ} [NeZero p]

/-- `G_k = f̂(k,…,k; −k)`, so that `β_j = 2 p^{-(n+1)} G_j`. -/
def Gd (f : Table p n) (k : ZMod p) : ℂ := fhat f (fun _ => k) (-k)

lemma beta_eq_Gd (f : Table p n) (j : ℕ) : beta f j = 2 / (p : ℂ) ^ (n + 1) * Gd f (j : ZMod p) :=
  rfl

/-- Conjugate symmetry `G_{-k} = conj G_k` for a real table. -/
lemma Gd_neg (f : Table p n) (k : ZMod p) : Gd f (-k) = (starRingEnd ℂ) (Gd f k) := by
  unfold Gd fhat
  rw [map_sum]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [map_sum]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [map_mul, Complex.conj_ofReal, ← pn_rho_neg]
  congr 2
  simp only [neg_mul, Finset.sum_neg_distrib, neg_neg]
  ring

/-- Orthogonality: `∑_{k ∈ ℤ_p} G_k = p ∑_x f(x, y(x))`. -/
lemma sum_Gd (f : Table p n) :
    ∑ k : ZMod p, Gd f k = (p : ℂ) * ∑ x : Inp p n, (f x (label x) : ℂ) := by
  classical
  unfold Gd fhat
  have harg : ∀ (k : ZMod p) (x : Inp p n) (c : ZMod p),
      -(∑ t, (fun _ => k) t * x t + -k * c) = k * (c - label x) := by
    intro k x c
    simp only [label, ← Finset.mul_sum]
    ring
  simp_rw [harg]
  rw [Finset.sum_comm]
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [Finset.sum_comm]
  have hc : ∀ c : ZMod p, ∑ k : ZMod p, (f x c : ℂ) * rho (k * (c - label x)) =
      (f x c : ℂ) * (if c - label x = 0 then (p : ℂ) else 0) := by
    intro c
    rw [← Finset.mul_sum, pn_sum_rho_mul]
  simp_rw [hc]
  rw [Finset.sum_eq_single (label x)]
  · simp; ring
  · intro c _ hc'
    have : c - label x ≠ 0 := sub_ne_zero.2 hc'
    simp [this]
  · simp

/-- `G_0 = ∑_{x,c} f(x,c)`. -/
lemma Gd_zero (f : Table p n) : Gd f 0 = ∑ x : Inp p n, ∑ c : ZMod p, (f x c : ℂ) := by
  unfold Gd fhat
  refine Finset.sum_congr rfl fun x _ => Finset.sum_congr rfl fun c _ => ?_
  simp [rho_zero']

/-- The diagonal transform of the hidden activation, `ĥ(k,…,k) = ∑_x h(x) ρ^{-k(a_1+⋯+a_n)}`. -/
def hdiag (u : Fin n → ZMod p → ℝ) (k : ZMod p) : ℂ :=
  ∑ x : Inp p n, ((max (∑ t, u t (x t)) 0 : ℝ) : ℂ) * rho (-(∑ t, k * x t))

/-- Factorization for the ReLU neuron: `G_k = ĥ(k,…,k) · conj(ŵ(k))`. -/
lemma Gd_relu (u : Fin n → ZMod p → ℝ) (w : ZMod p → ℝ) (k : ZMod p) :
    Gd (reluNeuron u w) k = hdiag u k * (starRingEnd ℂ) (dft w k) := by
  rw [← dft_neg]
  unfold Gd fhat hdiag dft reluNeuron
  rw [Finset.sum_mul]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun c _ => ?_
  have : -(∑ t, (fun _ => k) t * x t + -k * c) = -(∑ t, k * x t) + -(-k * c) := by ring
  rw [this, pn_rho_add]
  push_cast
  ring

end General

/-! ## The witness at `p = 5`, `n = 2` -/

/-- `u_1 = u_2 = (−2, 2, −2, 2, 0)` as integers. -/
def uZ : ZMod 5 → ℤ := fun a => ![-2, 2, -2, 2, 0] (a : Fin 5)

/-- `w = (−2, 0, 3, −3, 1)` as integers. -/
def wZ : ZMod 5 → ℤ := fun a => ![-2, 0, 3, -3, 1] (a : Fin 5)

/-- The two operand weight vectors `u_1 = u_2`. -/
def uCE : Fin 2 → ZMod 5 → ℝ := fun _ a => (uZ a : ℝ)

/-- The output weight vector `w`. -/
def wCE : ZMod 5 → ℝ := fun c => (wZ c : ℝ)

lemma sum_inp_two {M : Type*} [AddCommMonoid M] (F : Inp 5 2 → M) :
    ∑ x : Inp 5 2, F x = ∑ a : ZMod 5, ∑ b : ZMod 5, F ![a, b] := by
  rw [← (piFinTwoEquiv fun _ => ZMod 5).symm.sum_comp, Fintype.sum_prod_type]
  rfl

lemma relu_table_eq (x : Inp 5 2) (c : ZMod 5) :
    reluNeuron uCE wCE x c = ((max (uZ (x 0) + uZ (x 1)) 0 * wZ c : ℤ) : ℝ) := by
  simp [reluNeuron, uCE, wCE, Fin.sum_univ_two]

/-- `∑_x f(x, y(x)) = 24`. -/
lemma S_value : ∑ x : Inp 5 2, reluNeuron uCE wCE x (label x) = 24 := by
  have h : ∀ x : Inp 5 2, reluNeuron uCE wCE x (label x) =
      ((max (uZ (x 0) + uZ (x 1)) 0 * wZ (x 0 + x 1) : ℤ) : ℝ) := by
    intro x
    rw [relu_table_eq]
    simp [label, Fin.sum_univ_two]
  simp_rw [h]
  rw [← Int.cast_sum, sum_inp_two]
  have : ∑ a : ZMod 5, ∑ b : ZMod 5,
      max (uZ ((![a, b] : Inp 5 2) 0) + uZ ((![a, b] : Inp 5 2) 1)) 0 *
        wZ ((![a, b] : Inp 5 2) 0 + (![a, b] : Inp 5 2) 1) = 24 := by
    decide
  rw [this]; norm_num

/-- `∑_{x,c} f(x, c) = −24`. -/
lemma T_value : ∑ x : Inp 5 2, ∑ c : ZMod 5, reluNeuron uCE wCE x c = -24 := by
  simp_rw [relu_table_eq]
  simp_rw [← Int.cast_sum]
  rw [sum_inp_two]
  have : ∑ a : ZMod 5, ∑ b : ZMod 5, ∑ c : ZMod 5,
      max (uZ ((![a, b] : Inp 5 2) 0) + uZ ((![a, b] : Inp 5 2) 1)) 0 * wZ c = -24 := by
    decide
  exact_mod_cast this

/-- `‖ω‖² = ∑_t ‖u_t‖² + ‖w‖² = 55`. -/
lemma normSq_value : (∑ t, ∑ a, uCE t a ^ 2 + ∑ c, wCE c ^ 2) = 55 := by
  have h1 : ∑ t : Fin 2, ∑ a : ZMod 5, uCE t a ^ 2 = ((∑ t : Fin 2, ∑ a : ZMod 5, uZ a ^ 2 : ℤ) : ℝ) := by
    simp [uCE]
  have h2 : ∑ c : ZMod 5, wCE c ^ 2 = ((∑ c : ZMod 5, wZ c ^ 2 : ℤ) : ℝ) := by
    simp [wCE]
  have e1 : (∑ t : Fin 2, ∑ a : ZMod 5, uZ a ^ 2 : ℤ) = 32 := by decide
  have e2 : (∑ c : ZMod 5, wZ c ^ 2 : ℤ) = 23 := by decide
  rw [h1, h2, e1, e2]; norm_num

/-- The key lower bound `|G_1| + |G_2| ≥ ½ (5·24 − (−24)) = 72`. -/
lemma G12_ge : (72 : ℝ) ≤ ‖Gd (reluNeuron uCE wCE) 1‖ + ‖Gd (reluNeuron uCE wCE) 2‖ := by
  set f := reluNeuron uCE wCE with hf
  have h3 : Gd f 3 = (starRingEnd ℂ) (Gd f 2) := by
    rw [← Gd_neg]; congr 1
  have h4 : Gd f 4 = (starRingEnd ℂ) (Gd f 1) := by
    rw [← Gd_neg]; congr 1
  have hsum : ∑ k : ZMod 5, Gd f k = Gd f 0 + Gd f 1 + Gd f 2 + Gd f 3 + Gd f 4 := by
    have huniv : (Finset.univ : Finset (ZMod 5)) = {0, 1, 2, 3, 4} := by decide
    rw [huniv, Finset.sum_insert (by decide), Finset.sum_insert (by decide),
      Finset.sum_insert (by decide), Finset.sum_insert (by decide), Finset.sum_singleton]
    ring
  have hS := sum_Gd f
  rw [hsum] at hS
  have hS' : (∑ x : Inp 5 2, (f x (label x) : ℂ)) = ((24 : ℝ) : ℂ) := by
    rw [← Complex.ofReal_sum, S_value]
  have hT : Gd f 0 = ((-24 : ℝ) : ℂ) := by
    rw [Gd_zero]
    rw [show (∑ x : Inp 5 2, ∑ c : ZMod 5, (f x c : ℂ)) =
      ((∑ x : Inp 5 2, ∑ c : ZMod 5, f x c : ℝ) : ℂ) by push_cast; rfl, T_value]
  rw [hS', hT, h3, h4] at hS
  have hre := congrArg Complex.re hS
  simp only [Complex.add_re, Complex.conj_re, Complex.ofReal_re, Complex.mul_re,
    Complex.ofReal_im, Complex.natCast_re, Complex.natCast_im] at hre
  have n1 := Complex.re_le_norm (Gd f 1)
  have n2 := Complex.re_le_norm (Gd f 2)
  norm_num at hre
  linarith

/-- **Counterexample, paper's intermediate form.** For the ReLU neuron at `p = 5`, `n = 2`:
`∑_{j∈F_5} |ĥ(j,j)| |ŵ(j)| > (p/4) ‖ω‖²`, contradicting the displayed inequality of
appendix `app:models` ("ReLU neuron"). -/
theorem relu_diag_claim_false :
    (5 : ℝ) / 4 * (∑ t, ∑ a, uCE t a ^ 2 + ∑ c, wCE c ^ 2) <
      ∑ j : Freq 5, ‖hdiag uCE (j.1 : ZMod 5)‖ * ‖dft wCE (j.1 : ZMod 5)‖ := by
  rw [normSq_value]
  have hfreq : ∀ g : ℕ → ℝ, ∑ j : Freq 5, g j.1 = g 1 + g 2 := by
    intro g
    rw [Finset.sum_coe_sort (Finset.Icc 1 ((5 - 1) / 2)) g]
    have : Finset.Icc 1 ((5 - 1) / 2) = ({1, 2} : Finset ℕ) := by decide
    rw [this, Finset.sum_pair (by norm_num)]
  rw [hfreq (fun j => ‖hdiag uCE (j : ZMod 5)‖ * ‖dft wCE (j : ZMod 5)‖)]
  have e : ∀ k : ZMod 5, ‖hdiag uCE k‖ * ‖dft wCE k‖ = ‖Gd (reluNeuron uCE wCE) k‖ := by
    intro k
    rw [Gd_relu, norm_mul, Complex.norm_conj]
  simp only [e]
  have := G12_ge
  push_cast
  linarith

/-- **Counterexample to `C_2 ≤ p^{-n}/2`.** For the ReLU neuron at `p = 5`, `n = 2`, with the
repository's `β_j` (paper conventions): `∑_{j∈F_5} |β_j| > (p^{-n}/2) ‖ω‖²`.  Hence the claimed
single-neuron budget `C_2 ≤ p^{-n}/2` of appendix `app:models` is false. -/
theorem relu_budget_claim_false :
    (5 : ℝ) ^ (-(2 : ℤ)) / 2 * (∑ t, ∑ a, uCE t a ^ 2 + ∑ c, wCE c ^ 2) <
      ∑ j : Freq 5, ‖beta (reluNeuron uCE wCE) j.1‖ := by
  rw [normSq_value]
  have hfreq : ∀ g : ℕ → ℝ, ∑ j : Freq 5, g j.1 = g 1 + g 2 := by
    intro g
    rw [Finset.sum_coe_sort (Finset.Icc 1 ((5 - 1) / 2)) g]
    have : Finset.Icc 1 ((5 - 1) / 2) = ({1, 2} : Finset ℕ) := by decide
    rw [this, Finset.sum_pair (by norm_num)]
  rw [hfreq (fun j => ‖beta (reluNeuron uCE wCE) j‖)]
  simp only [beta_eq_Gd, norm_mul]
  have hc : ‖(2 : ℂ) / (5 : ℂ) ^ (2 + 1)‖ = 2 / 125 := by
    rw [norm_div]; norm_num
  have hc' : ‖(2 : ℂ) / ((5 : ℕ) : ℂ) ^ (2 + 1)‖ = 2 / 125 := by
    push_cast; exact hc
  rw [hc']
  have := G12_ge
  push_cast
  norm_num
  linarith

end Heracles
