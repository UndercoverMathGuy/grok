import Heracles.Network

/-!
# Product neurons: the `ν`-block (`prop:block`), the sign rule (`lem:sign`) and the
single-neuron budget (appendix `app:models`)
-/

open Real Finset

noncomputable section

set_option linter.unusedSectionVars false

namespace Heracles

/-! ## Characters of `ℤ_p` -/

section Char

variable {p : ℕ} [NeZero p]

lemma pn_rho_add (a b : ZMod p) : rho (a + b) = rho a * rho b :=
  AddChar.map_add_eq_mul _ a b

lemma rho_zero' : rho (0 : ZMod p) = 1 := AddChar.map_zero_eq_one _

lemma rho_ne_one {m : ZMod p} (hm : m ≠ 0) : rho m ≠ 1 := by
  intro h
  apply hm
  apply ZMod.injective_stdAddChar
  rw [AddChar.map_zero_eq_one]
  exact h

/-- Shifting operand `t0` by `1` and the candidate by `1` permutes the sum over the dataset. -/
lemma sum_shift {n : ℕ} {M : Type*} [AddCommMonoid M] (t0 : Fin n)
    (F : Inp p n → ZMod p → M) :
    ∑ x : Inp p n, ∑ c : ZMod p, F x c =
      ∑ x : Inp p n, ∑ c : ZMod p, F (x + Pi.single t0 1) (c + 1) := by
  classical
  calc ∑ x : Inp p n, ∑ c : ZMod p, F x c = ∑ x : Inp p n, ∑ c : ZMod p, F x (c + 1) :=
        Finset.sum_congr rfl fun x _ =>
          (Fintype.sum_equiv (Equiv.addRight (1 : ZMod p)) (fun c => F x (c + 1)) (F x)
            (fun c => rfl)).symm
    _ = ∑ x : Inp p n, ∑ c : ZMod p, F (x + Pi.single t0 1) (c + 1) :=
        (Fintype.sum_equiv (Equiv.addRight (Pi.single t0 (1 : ZMod p)))
          (fun x => ∑ c : ZMod p, F (x + Pi.single t0 1) (c + 1))
          (fun x => ∑ c, F x (c + 1)) (fun x => rfl)).symm

lemma label_add_single {n : ℕ} (x : Inp p n) (t0 : Fin n) :
    label (x + Pi.single t0 1) = label x + 1 := by
  classical
  unfold label
  simp [Finset.sum_add_distrib]

/-- **Translation-invariant tables have vanishing off-diagonal Fourier modes.**  If
`f x c = g(y(x) - c)` then `f̂(k; l) = 0` as soon as `k_{t0} + l ≠ 0` for some operand `t0`. -/
lemma fhat_eq_zero_of_invariant {n : ℕ} (f : Table p n) (g : ZMod p → ℝ)
    (hf : ∀ x c, f x c = g (label x - c)) (k : Fin n → ZMod p) (l : ZMod p) (t0 : Fin n)
    (hkl : k t0 + l ≠ 0) : fhat f k l = 0 := by
  classical
  have hshift : fhat f k l = rho (-(k t0 + l)) * fhat f k l := by
    unfold fhat
    conv_lhs => rw [sum_shift t0]
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun x _ => ?_
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun c _ => ?_
    have hk : ∑ t, k t * (x + (Pi.single t0 (1 : ZMod p) : Inp p n)) t =
        ∑ t, k t * x t + k t0 := by
      simp [mul_add, Finset.sum_add_distrib, Pi.single_apply]
    rw [hf (x + Pi.single t0 1), hf x, label_add_single, hk]
    have h1 : label x + 1 - (c + 1) = label x - c := by ring
    have h2 : -(∑ t, k t * x t + k t0 + l * (c + 1)) =
        -(k t0 + l) + -(∑ t, k t * x t + l * c) := by ring
    rw [h1, h2, pn_rho_add]
    ring
  have h1 : (1 - rho (-(k t0 + l))) * fhat f k l = 0 := by
    rw [sub_mul, one_mul, ← hshift, sub_self]
  rcases mul_eq_zero.1 h1 with h | h
  · exact absurd (sub_eq_zero.1 h).symm (rho_ne_one (neg_ne_zero.2 hkl))
  · exact h

end Char

/-- A table of the form `f x c = g(y(x) - c)` is interference-free (for odd prime `p`). -/
theorem interferenceFree_of_invariant {p n : ℕ} [Fact p.Prime] (hp : 3 ≤ p) (f : Table p n)
    (g : ZMod p → ℝ) (hf : ∀ x c, f x c = g (label x - c)) : InterferenceFree f := by
  intro j hj T hT
  obtain ⟨t0, ht0⟩ := hT
  unfold betaT
  rw [fhat_eq_zero_of_invariant f g hf _ _ t0, mul_zero]
  simp only [ht0, ite_true]
  intro h
  have h2 : ((2 * j : ℕ) : ZMod p) = 0 := by
    push_cast
    linear_combination -h
  rw [ZMod.natCast_eq_zero_iff] at h2
  have hj' := Finset.mem_Icc.1 hj
  have hpos : 0 < 2 * j := by omega
  have := Nat.le_of_dvd hpos h2
  omega

/-! ## Roots-of-unity filter -/

lemma sum_pow_roots {ν : ℕ} {ζ : ℂ} (hζ : IsPrimitiveRoot ζ ν) (k : ℕ) :
    ∑ l ∈ range ν, ζ ^ (l * k) = if ν ∣ k then (ν : ℂ) else 0 := by
  have h : ∀ l, ζ ^ (l * k) = (ζ ^ k) ^ l := fun l => by rw [← pow_mul, mul_comm]
  simp_rw [h]
  split_ifs with hk
  · have : ζ ^ k = 1 := (hζ.pow_eq_one_iff_dvd k).2 hk
    simp [this]
  · have hne : ζ ^ k ≠ 1 := fun h' => hk ((hζ.pow_eq_one_iff_dvd k).1 h')
    rw [geom_sum_eq hne, ← pow_mul, mul_comm, pow_mul, hζ.pow_eq_one, one_pow, sub_self,
      zero_div]

/-- The algebraic core of `prop:block`: averaging the polynomial
`∏_t (b_t + a_t X) (a_c + b_c X)` of degree `ν = n+1` over the `ν`-th roots of unity keeps
only its constant and top coefficients. -/
lemma root_filter_prod (n : ℕ) {ζ : ℂ} (hζ : IsPrimitiveRoot ζ (n + 1)) (a b : Fin n → ℂ)
    (ac bc : ℂ) :
    ∑ l ∈ range (n + 1), (∏ t, (b t + a t * ζ ^ l)) * (ac + bc * ζ ^ l) =
      (n + 1 : ℂ) * ((∏ t, b t) * ac + (∏ t, a t) * bc) := by
  classical
  set A : Finset (Fin n) → ℂ := fun S => (∏ t ∈ S, a t) * ∏ t ∈ univ \ S, b t with hA
  have hexp : ∀ l, ∏ t, (b t + a t * ζ ^ l) =
      ∑ S ∈ (univ : Finset (Fin n)).powerset, A S * ζ ^ (l * S.card) := by
    intro l
    rw [show (∏ t, (b t + a t * ζ ^ l)) = ∏ t, (a t * ζ ^ l + b t) from
      Finset.prod_congr rfl fun t _ => add_comm _ _]
    rw [Finset.prod_add]
    refine Finset.sum_congr rfl fun S _ => ?_
    rw [Finset.prod_mul_distrib, Finset.prod_const, ← pow_mul, hA]
    ring
  have hcardS : ∀ S : Finset (Fin n), (n + 1 ∣ S.card ↔ S = ∅) := by
    intro S
    constructor
    · intro h
      have hlt : S.card < n + 1 := by
        have := Finset.card_le_univ S; simp at this; omega
      exact Finset.card_eq_zero.1 (Nat.eq_zero_of_dvd_of_lt h hlt)
    · intro h; simp [h]
  have hcardS1 : ∀ S : Finset (Fin n), (n + 1 ∣ S.card + 1 ↔ S = univ) := by
    intro S
    constructor
    · intro h
      have hle : S.card ≤ n := by have := Finset.card_le_univ S; simpa using this
      have := Nat.le_of_dvd (by omega) h
      apply Finset.eq_univ_of_card
      simp; omega
    · intro h; simp [h]
  calc ∑ l ∈ range (n + 1), (∏ t, (b t + a t * ζ ^ l)) * (ac + bc * ζ ^ l)
      = ∑ l ∈ range (n + 1), ∑ S ∈ (univ : Finset (Fin n)).powerset,
          (A S * ac * ζ ^ (l * S.card) + A S * bc * ζ ^ (l * (S.card + 1))) := by
        refine Finset.sum_congr rfl fun l _ => ?_
        rw [hexp, Finset.sum_mul]
        refine Finset.sum_congr rfl fun S _ => ?_
        rw [show l * (S.card + 1) = l * S.card + l by ring, pow_add]
        ring
    _ = ∑ S ∈ (univ : Finset (Fin n)).powerset,
          (A S * ac * ∑ l ∈ range (n + 1), ζ ^ (l * S.card) +
            A S * bc * ∑ l ∈ range (n + 1), ζ ^ (l * (S.card + 1))) := by
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun S _ => ?_
        rw [Finset.sum_add_distrib, Finset.mul_sum, Finset.mul_sum]
    _ = ∑ S ∈ (univ : Finset (Fin n)).powerset,
          ((if S = ∅ then A S * ac * (n + 1 : ℂ) else 0) +
            (if S = univ then A S * bc * (n + 1 : ℂ) else 0)) := by
        refine Finset.sum_congr rfl fun S _ => ?_
        rw [sum_pow_roots hζ, sum_pow_roots hζ]
        simp only [hcardS, hcardS1]
        push_cast
        split_ifs <;> simp
    _ = (n + 1 : ℂ) * ((∏ t, b t) * ac + (∏ t, a t) * bc) := by
        rw [Finset.sum_add_distrib, Finset.sum_ite_eq', Finset.sum_ite_eq']
        simp only [Finset.mem_powerset, Finset.subset_univ, ite_true, hA,
          Finset.prod_empty, Finset.sdiff_empty, Finset.sdiff_self, one_mul, mul_one]
        ring

/-! ## The `ν`-block identity (appendix `app:block`, "Identity") -/

/-- `E(x) = e^{i x}`. -/
def Ex (x : ℝ) : ℂ := Complex.exp (x * Complex.I)

lemma Ex_add (x y : ℝ) : Ex (x + y) = Ex x * Ex y := by
  unfold Ex; push_cast; rw [add_mul, Complex.exp_add]

lemma Ex_zero : Ex 0 = 1 := by simp [Ex]

lemma Ex_pow (x : ℝ) (m : ℕ) : Ex x ^ m = Ex (m * x) := by
  unfold Ex; rw [← Complex.exp_nat_mul]; push_cast; ring_nf

lemma Ex_sum {ι : Type*} (s : Finset ι) (f : ι → ℝ) : ∏ i ∈ s, Ex (f i) = Ex (∑ i ∈ s, f i) := by
  unfold Ex; push_cast; rw [Finset.sum_mul, Complex.exp_sum]

lemma two_cos_eq_Ex (x : ℝ) : ((2 * Real.cos x : ℝ) : ℂ) = Ex x + Ex (-x) := by
  unfold Ex; push_cast; rw [Complex.two_cos]

lemma two_cos_add_u (θ α : ℝ) :
    ((2 * Real.cos (θ + α) : ℝ) : ℂ) = Ex (-α) * (Ex (-θ) + Ex θ * Ex (2 * α)) := by
  rw [two_cos_eq_Ex, mul_add, ← Ex_add, ← mul_assoc, ← Ex_add, ← Ex_add]
  rw [add_comm (Ex _) (Ex _)]
  congr 2 <;> ring

lemma two_cos_add_w (θ β : ℝ) :
    ((2 * Real.cos (θ + β) : ℝ) : ℂ) = Ex β * (Ex θ + Ex (-θ) * Ex (-2 * β)) := by
  rw [two_cos_eq_Ex, mul_add, ← Ex_add, ← mul_assoc, ← Ex_add, ← Ex_add]
  congr 2 <;> ring

/-- The phases `α_l = π l / ν`. -/
def alphaL (n l : ℕ) : ℝ := π * l / (n + 1)

/-- A primitive `ν`-th root of unity, `ζ = e^{2π i/ν}`. -/
def zetaNu (n : ℕ) : ℂ := Complex.exp (2 * π * Complex.I / (n + 1 : ℕ))

lemma zetaNu_prim (n : ℕ) : IsPrimitiveRoot (zetaNu n) (n + 1) :=
  Complex.isPrimitiveRoot_exp (n + 1) (Nat.succ_ne_zero n)

lemma Ex_two_alpha (n l : ℕ) : Ex (2 * alphaL n l) = zetaNu n ^ l := by
  unfold Ex zetaNu alphaL
  rw [← Complex.exp_nat_mul]
  congr 1
  push_cast
  field_simp

lemma Ex_neg_two_n_alpha (n l : ℕ) : Ex (-2 * (n * alphaL n l)) = zetaNu n ^ l := by
  have h : (-2 * (n * alphaL n l) : ℝ) = 2 * alphaL n l + (-((l : ℝ) * (2 * π))) := by
    unfold alphaL
    field_simp
    ring
  rw [h, Ex_add, Ex_two_alpha]
  have h1 : Ex (-((l : ℝ) * (2 * π))) = 1 := by
    unfold Ex
    push_cast
    rw [show -((l : ℂ) * (2 * (π : ℂ))) * Complex.I = -((l : ℂ) * (2 * π * Complex.I)) by ring,
      Complex.exp_neg, Complex.exp_nat_mul_two_pi_mul_I, inv_one]
  rw [h1, mul_one]

/-- One neuron of the block, in exponential form. -/
lemma block_neuron_Ex (n l : ℕ) (θ : Fin n → ℝ) (θc : ℝ) :
    (∏ t, ((2 * Real.cos (θ t + alphaL n l) : ℝ) : ℂ)) *
        ((2 * Real.cos (θc + n * alphaL n l) : ℝ) : ℂ) =
      (∏ t, (Ex (-θ t) + Ex (θ t) * zetaNu n ^ l)) * (Ex θc + Ex (-θc) * zetaNu n ^ l) := by
  rw [two_cos_add_w]
  simp_rw [two_cos_add_u]
  simp_rw [Ex_two_alpha, Ex_neg_two_n_alpha]
  rw [Finset.prod_mul_distrib, Finset.prod_const, Finset.card_univ, Fintype.card_fin, Ex_pow]
  have h1 : Ex ((n : ℝ) * -alphaL n l) * Ex (n * alphaL n l) = 1 := by
    rw [← Ex_add, show (n : ℝ) * -alphaL n l + n * alphaL n l = 0 by ring, Ex_zero]
  calc Ex (↑n * -alphaL n l) * (∏ t, (Ex (-θ t) + Ex (θ t) * zetaNu n ^ l)) *
        (Ex (↑n * alphaL n l) * (Ex θc + Ex (-θc) * zetaNu n ^ l))
      = (Ex ((n : ℝ) * -alphaL n l) * Ex (n * alphaL n l)) *
          ((∏ t, (Ex (-θ t) + Ex (θ t) * zetaNu n ^ l)) * (Ex θc + Ex (-θc) * zetaNu n ^ l)) := by
        ring
    _ = _ := by rw [h1, one_mul]

/-- **The trigonometric identity behind `prop:block`**:
`∑_{l=0}^{n} ∏_t cos(θ_t + α_l) · cos(θ_c + n α_l) = ν 2^{-n} cos(∑_t θ_t − θ_c)`. -/
theorem block_trig (n : ℕ) (θ : Fin n → ℝ) (θc : ℝ) :
    ∑ l ∈ range (n + 1), (∏ t, Real.cos (θ t + alphaL n l)) * Real.cos (θc + n * alphaL n l) =
      (n + 1) / 2 ^ n * Real.cos (∑ t, θ t - θc) := by
  have key : ∑ l ∈ range (n + 1), (∏ t, ((2 * Real.cos (θ t + alphaL n l) : ℝ) : ℂ)) *
        ((2 * Real.cos (θc + n * alphaL n l) : ℝ) : ℂ) =
      (((n + 1) * (2 * Real.cos (∑ t, θ t - θc)) : ℝ) : ℂ) := by
    simp_rw [block_neuron_Ex]
    rw [root_filter_prod n (zetaNu_prim n) (fun t => Ex (θ t)) (fun t => Ex (-θ t))]
    rw [Ex_sum, Ex_sum, ← Ex_add, ← Ex_add, Complex.ofReal_mul, two_cos_eq_Ex]
    have h1 : ∑ i, -θ i + θc = -(∑ t, θ t - θc) := by rw [Finset.sum_neg_distrib]; ring
    have h1' : ∑ i, θ i + -θc = ∑ t, θ t - θc := by ring
    rw [h1, h1']
    push_cast
    ring
  have h2 : ∀ l : ℕ, (∏ t, ((2 * Real.cos (θ t + alphaL n l) : ℝ) : ℂ)) *
        ((2 * Real.cos (θc + n * alphaL n l) : ℝ) : ℂ) =
      (((2 : ℝ) ^ (n + 1) * ((∏ t, Real.cos (θ t + alphaL n l)) *
        Real.cos (θc + n * alphaL n l)) : ℝ) : ℂ) := by
    intro l
    push_cast
    rw [Finset.prod_mul_distrib, Finset.prod_const, Finset.card_univ, Fintype.card_fin]
    ring
  simp_rw [h2] at key
  rw [← Complex.ofReal_sum, Complex.ofReal_inj, ← Finset.mul_sum] at key
  have h4 : (0 : ℝ) < 2 ^ n := by positivity
  rw [div_mul_eq_mul_div, eq_div_iff h4.ne']
  rw [pow_succ] at key
  linear_combination key / 2

/-! ## The block neurons -/

/-- Neuron `l` of the `ν`-block at frequency `j` with scale `λ`:
`u_t(a) = λ cos(2π j a/p + α_l)`, `w(c) = λ cos(2π j c/p + n α_l)`. -/
def blockNeuron (p n j : ℕ) (lam : ℝ) (l : ℕ) : ProdParam p n where
  u := fun _ a => lam * Real.cos (2 * π * j * (a.val : ℝ) / p + alphaL n l)
  w := fun c => lam * Real.cos (2 * π * j * (c.val : ℝ) / p + n * alphaL n l)

/-- `cos(∑_t 2πj a_t/p − 2πj c/p) = cos(2πj (y(x) − c)/p)`: the phase only depends on
`y(x) - c ∈ ℤ_p`. -/
lemma cos_label {p n : ℕ} [NeZero p] (j : ℕ) (x : Inp p n) (c : ZMod p) :
    Real.cos (∑ t, 2 * π * j * ((x t).val : ℝ) / p - 2 * π * j * (c.val : ℝ) / p) =
      Real.cos (2 * π * j * ((label x - c).val : ℝ) / p) := by
  have hp : (p : ℝ) ≠ 0 := by exact_mod_cast NeZero.ne p
  set A : ℤ := ∑ t, ((x t).val : ℤ) - (c.val : ℤ) with hA
  set B : ℤ := ((label x - c).val : ℤ) with hB
  have hAB : ((A : ℤ) : ZMod p) = ((B : ℤ) : ZMod p) := by
    simp [hA, hB, label]
  obtain ⟨m, hm⟩ := (ZMod.intCast_eq_intCast_iff_dvd_sub A B p).1 hAB
  have hBA : (B : ℝ) = A + p * m := by
    have : B = A + p * m := by linarith
    exact_mod_cast this
  have hL : ∑ t, 2 * π * j * ((x t).val : ℝ) / p - 2 * π * j * (c.val : ℝ) / p =
      2 * π * j * (A : ℝ) / p := by
    rw [hA]
    push_cast
    rw [mul_sub, sub_div, Finset.mul_sum, Finset.sum_div]
  have hR : 2 * π * j * ((label x - c).val : ℝ) / p =
      2 * π * j * (A : ℝ) / p + ((j * m : ℤ) : ℝ) * (2 * π) := by
    have : ((label x - c).val : ℝ) = (B : ℝ) := by rw [hB]; push_cast; rfl
    rw [this, hBA]
    push_cast
    field_simp
  rw [hL, hR, Real.cos_add_int_mul_two_pi]

lemma prodNeuron_block {p n : ℕ} (j : ℕ) (lam : ℝ) (l : ℕ) (x : Inp p n) (c : ZMod p) :
    prodNeuron (blockNeuron p n j lam l) x c =
      lam ^ (n + 1) * ((∏ t, Real.cos (2 * π * j * ((x t).val : ℝ) / p + alphaL n l)) *
        Real.cos (2 * π * j * (c.val : ℝ) / p + n * alphaL n l)) := by
  unfold prodNeuron blockNeuron
  simp only
  rw [Finset.prod_mul_distrib, Finset.prod_const, Finset.card_univ, Fintype.card_fin]
  ring

/-- **`prop:block`, the identity.**  The total table of the `ν` neurons of the block is
`ν λ^ν 2^{-n} cos(2π j (a_1 + ⋯ + a_n − c)/p)`. -/
theorem block_table {p n : ℕ} [NeZero p] (j : ℕ) (lam : ℝ) (x : Inp p n) (c : ZMod p) :
    ∑ l ∈ range (n + 1), prodNeuron (blockNeuron p n j lam l) x c =
      (n + 1) * lam ^ (n + 1) / 2 ^ n * Real.cos (2 * π * j * ((label x - c).val : ℝ) / p) := by
  simp_rw [prodNeuron_block]
  rw [← Finset.mul_sum, block_trig, cos_label]
  ring

/-- Reindexing a sum over `ℤ_p` by `a ↦ a.val ∈ [0, p)`. -/
lemma pn_sum_zmod_val {p : ℕ} [NeZero p] {M : Type*} [AddCommMonoid M] (F : ℕ → M) :
    ∑ a : ZMod p, F a.val = ∑ k ∈ range p, F k := by
  refine Finset.sum_bij' (fun a _ => a.val) (fun k _ => (k : ZMod p)) ?_ ?_ ?_ ?_ ?_
  · intro a _; exact Finset.mem_range.2 (ZMod.val_lt a)
  · intro k _; exact Finset.mem_univ _
  · intro a _; exact ZMod.natCast_zmod_val a
  · intro k hk; exact ZMod.val_cast_of_lt (Finset.mem_range.1 hk)
  · intro a _; rfl

lemma sum_cos_zmod {p : ℕ} [NeZero p] (m : ℕ) (hm : ¬ p ∣ m) (β : ℝ) :
    ∑ a : ZMod p, Real.cos (2 * π * m * (a.val : ℝ) / p + β) = 0 := by
  rw [pn_sum_zmod_val (fun k => Real.cos (2 * π * m * (k : ℝ) / p + β))]
  set ζ : ℂ := Complex.exp (2 * π * Complex.I / (p : ℕ)) with hζ
  have hprim : IsPrimitiveRoot ζ p := Complex.isPrimitiveRoot_exp p (NeZero.ne p)
  have hterm : ∀ k : ℕ, Real.cos (2 * π * m * (k : ℝ) / p + β) =
      (Ex β * ζ ^ (k * m)).re := by
    intro k
    have : Ex β * ζ ^ (k * m) = Ex (2 * π * m * (k : ℝ) / p + β) := by
      rw [Ex_add, mul_comm]
      congr 1
      unfold Ex
      rw [hζ, ← Complex.exp_nat_mul]
      congr 1
      push_cast
      ring
    rw [this, Ex, Complex.exp_ofReal_mul_I_re]
  simp_rw [hterm]
  rw [← Complex.re_sum, ← Finset.mul_sum, sum_pow_roots hprim, ite_eq_right_iff.2 (fun h => absurd h hm), mul_zero,
    Complex.zero_re]

lemma sum_cos_sq_zmod {p : ℕ} [NeZero p] (j : ℕ) (hj : ¬ p ∣ 2 * j) (β : ℝ) :
    ∑ a : ZMod p, Real.cos (2 * π * j * (a.val : ℝ) / p + β) ^ 2 = p / 2 := by
  have h : ∀ a : ZMod p, Real.cos (2 * π * j * (a.val : ℝ) / p + β) ^ 2 =
      1 / 2 + Real.cos (2 * π * ((2 * j : ℕ) : ℝ) * (a.val : ℝ) / p + 2 * β) / 2 := by
    intro a
    rw [Real.cos_sq]
    congr 3
    push_cast
    ring
  simp_rw [h]
  rw [Finset.sum_add_distrib, ← Finset.sum_div, ← Finset.sum_div, sum_cos_zmod (2 * j) hj]
  simp [ZMod.card]

/-- The norm in `prop:block`: `‖ω^{(l)}‖_2^2 = ν λ^2 p / 2` (for `p ∤ 2j`). -/
theorem block_normSq {p n : ℕ} [NeZero p] (j : ℕ) (hj : ¬ p ∣ 2 * j) (lam : ℝ) (l : ℕ) :
    normSq (blockNeuron p n j lam l) = (n + 1) * lam ^ 2 * p / 2 := by
  have hu : ∀ β : ℝ, ∑ a : ZMod p, (lam * Real.cos (2 * π * j * (a.val : ℝ) / p + β)) ^ 2 =
      lam ^ 2 * (p / 2) := by
    intro β
    simp_rw [mul_pow]
    rw [← Finset.mul_sum, sum_cos_sq_zmod j hj]
  unfold normSq blockNeuron
  simp only
  rw [Finset.sum_congr rfl fun t _ => hu (alphaL n l), hu]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  ring

/-! ## The block network (attainment in `prop:block` / `thm:cyclic`) -/

/-- The scale `λ_j = (2^n y_j / ν)^{1/ν}`, so that `ν λ_j^ν 2^{-n} = y_j`. -/
def lamOf (n : ℕ) (yj : ℝ) : ℝ := (2 ^ n * yj / (n + 1)) ^ (((n + 1 : ℕ) : ℝ)⁻¹)

lemma lamOf_nonneg (n : ℕ) {yj : ℝ} (hy : 0 ≤ yj) : 0 ≤ lamOf n yj :=
  Real.rpow_nonneg (by positivity) _

lemma lamOf_pow (n : ℕ) {yj : ℝ} (hy : 0 ≤ yj) : lamOf n yj ^ (n + 1) = 2 ^ n * yj / (n + 1) :=
  Real.rpow_inv_natCast_pow (by positivity) (Nat.succ_ne_zero n)

lemma lamOf_rpow (n : ℕ) {yj : ℝ} (hy : 0 ≤ yj) (b : ℝ) :
    lamOf n yj ^ b = (2 ^ n / (n + 1)) ^ (b / (n + 1)) * yj ^ (b / (n + 1)) := by
  unfold lamOf
  rw [← Real.rpow_mul (by positivity),
    show (2 : ℝ) ^ n * yj / (n + 1) = (2 ^ n / (n + 1)) * yj by ring,
    Real.mul_rpow (by positivity) hy]
  have : ((n + 1 : ℕ) : ℝ)⁻¹ * b = b / (n + 1) := by push_cast; ring
  rw [this]

/-- Index set of the block network: one neuron per frequency `j ∈ F_p` and `l ∈ {0,…,n}`. -/
abbrev BIdx (p n : ℕ) := Freq p × Fin (n + 1)

/-- The neurons of the block network, indexed by `(j, l)`. -/
def blockNetQ (p n : ℕ) (y : Freq p → ℝ) (q : BIdx p n) : ProdParam p n :=
  blockNeuron p n q.1.1 (lamOf n (y q.1)) q.2

/-- The network of `prop:block`: one `ν`-block per `j ∈ F_p` with `λ_j^ν = 2^n y_j / ν`
(width `ν |F_p|`; the blocks with `y_j = 0` are zero neurons). -/
def blockNet (p n : ℕ) (y : Freq p → ℝ) : Fin (Fintype.card (BIdx p n)) → ProdParam p n :=
  blockNetQ p n y ∘ (Fintype.equivFin (BIdx p n)).symm

lemma sum_blockNet {p n : ℕ} {M : Type*} [AddCommMonoid M] (y : Freq p → ℝ)
    (F : ProdParam p n → M) :
    ∑ i, F (blockNet p n y i) = ∑ j : Freq p, ∑ l ∈ range (n + 1),
      F (blockNeuron p n j.1 (lamOf n (y j)) l) := by
  unfold blockNet
  simp only [Function.comp_apply]
  rw [Equiv.sum_comp (Fintype.equivFin (BIdx p n)).symm (fun q => F (blockNetQ p n y q)),
    Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun j _ => ?_
  exact Fin.sum_univ_eq_sum_range (fun l => F (blockNeuron p n j.1 (lamOf n (y j)) l)) (n + 1)

/-- The table of the block network is `∑_j y_j cos(2π j (y(x) − c)/p)`. -/
theorem blockNet_table {p n : ℕ} [NeZero p] (y : Freq p → ℝ) (hy : ∀ j, 0 ≤ y j)
    (x : Inp p n) (c : ZMod p) :
    netTable prodNeuron (blockNet p n y) x c =
      ∑ j : Freq p, y j * Real.cos (2 * π * j.1 * ((label x - c).val : ℝ) / p) := by
  unfold netTable
  rw [sum_blockNet y (fun ω => prodNeuron ω x c)]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [block_table, lamOf_pow n (hy j)]
  field_simp

/-- The block network is interference-free (its table depends only on `y(x) - c`). -/
theorem blockNet_interferenceFree {p n : ℕ} [Fact p.Prime] (hp : 3 ≤ p) (y : Freq p → ℝ)
    (hy : ∀ j, 0 ≤ y j) : InterferenceFree (netTable prodNeuron (blockNet p n y)) :=
  interferenceFree_of_invariant hp _
    (fun δ => ∑ j : Freq p, y j * Real.cos (2 * π * j.1 * (δ.val : ℝ) / p))
    (fun x c => blockNet_table y hy x c)

/-- The gap between the label and a wrong candidate `c` is the coverage of `δ = y(x) - c`. -/
lemma blockNet_gap {p n : ℕ} [Fact p.Prime] (y : Freq p → ℝ) (hy : ∀ j, 0 ≤ y j)
    (x : Inp p n) (c : ZMod p) (hc : c ≠ label x) :
    netTable prodNeuron (blockNet p n y) x (label x) - netTable prodNeuron (blockNet p n y) x c =
      cover (covP p) y ⟨label x - c, sub_ne_zero.2 (Ne.symm hc)⟩ := by
  rw [blockNet_table y hy, blockNet_table y hy, ← Finset.sum_sub_distrib]
  unfold cover covP cov
  refine Finset.sum_congr rfl fun j _ => ?_
  dsimp only
  simp only [sub_self, ZMod.val_zero, Nat.cast_zero, mul_zero, zero_div, Real.cos_zero, mul_one]
  ring

/-- The margin of the block network is the worst coverage `m(y)`. -/
theorem blockNet_margin {p n : ℕ} [Fact p.Prime] (y : Freq p → ℝ) (hy : ∀ j, 0 ≤ y j) :
    margin (netTable prodNeuron (blockNet p n y)) = worst (covP p) y := by
  apply le_antisymm
  · rw [le_worst_iff]
    intro δ
    have hc : -δ.1 ≠ label (0 : Inp p n) := by
      simp [label]; exact δ.2
    have h1 := margin_le (netTable prodNeuron (blockNet p n y)) 0 (-δ.1) hc
    rw [blockNet_gap y hy 0 (-δ.1) hc] at h1
    have : (⟨label (0 : Inp p n) - -δ.1, sub_ne_zero.2 (Ne.symm hc)⟩ : Off p) = δ := by
      apply Subtype.ext; simp [label]
    rwa [this] at h1
  · apply le_margin
    intro x c hc
    rw [blockNet_gap y hy x c hc]
    exact worst_le_cover _ _ _

lemma freq_not_dvd {p : ℕ} (j : Freq p) : ¬ p ∣ 2 * j.1 := by
  have hj := Finset.mem_Icc.1 j.2
  intro h
  have := Nat.le_of_dvd (by omega) h
  omega

lemma sqrt_normSq_block {p n : ℕ} [NeZero p] (j : ℕ) (hj : ¬ p ∣ 2 * j) {lam : ℝ}
    (hlam : 0 ≤ lam) (l : ℕ) :
    Real.sqrt (normSq (blockNeuron p n j lam l)) = lam * Real.sqrt ((n + 1) * p / 2) := by
  rw [block_normSq j hj, show (n + 1 : ℝ) * lam ^ 2 * p / 2 = lam ^ 2 * ((n + 1) * p / 2) by ring,
    Real.sqrt_mul' _ (by positivity), Real.sqrt_sq hlam]

/-- `‖θ‖_{2,b}^b` of the block network built on `y` with `∑_j y_j^{b/ν} = 1`. -/
theorem normPow_blockNet {p n : ℕ} [NeZero p] {b : ℝ} (y : Freq p → ℝ) (hy : ∀ j, 0 ≤ y j)
    (hbud : budget ((n + 1) / b) y = 1) :
    normPow b (blockNet p n y) =
      (n + 1) * Real.sqrt ((n + 1) * p / 2) ^ b * (2 ^ n / (n + 1)) ^ (b / (n + 1)) := by
  unfold normPow
  rw [sum_blockNet y (fun ω => Real.sqrt (normSq ω) ^ b)]
  have hK : 0 ≤ Real.sqrt ((n + 1) * p / 2) := Real.sqrt_nonneg _
  have hterm : ∀ j : Freq p, ∑ l ∈ range (n + 1),
      Real.sqrt (normSq (blockNeuron p n j.1 (lamOf n (y j)) l)) ^ b =
      (n + 1) * Real.sqrt ((n + 1) * p / 2) ^ b * (2 ^ n / (n + 1)) ^ (b / (n + 1)) *
        y j ^ (b / (n + 1)) := by
    intro j
    simp_rw [sqrt_normSq_block j.1 (freq_not_dvd j) (lamOf_nonneg n (hy j)),
      Real.mul_rpow (lamOf_nonneg n (hy j)) hK, lamOf_rpow n (hy j)]
    rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
    push_cast
    ring
  rw [Finset.sum_congr rfl fun j _ => hterm j, ← Finset.mul_sum]
  have : ∑ j : Freq p, y j ^ (b / (n + 1)) = budget ((n + 1) / b) y := by
    unfold budget; rw [one_div_div]
  rw [this, hbud, mul_one]

/-- The constant identity behind the attainment value:
`ν^{-(ν-b)/b} · B_p · ν^{ν/b} (ν p/2)^{ν/2} (2^n/ν) = 1`. -/
lemma block_const_identity (p n : ℕ) [NeZero p] {b : ℝ} (hb : 0 < b) :
    (n + 1 : ℝ) ^ (-((n + 1 : ℝ) - b) / b) * Bp p n *
      ((n + 1 : ℝ) ^ ((n + 1 : ℝ) / b) * ((n + 1) * p / 2 : ℝ) ^ ((n + 1 : ℝ) / 2) *
        (2 ^ n / (n + 1))) = 1 := by
  have hν : (0 : ℝ) < n + 1 := by positivity
  have hP : (0 : ℝ) < p := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne p)
  have h1 : (n + 1 : ℝ) ^ (-((n + 1 : ℝ) - b) / b) * (n + 1 : ℝ) ^ ((n + 1 : ℝ) / b) =
      n + 1 := by
    rw [← Real.rpow_add hν]
    have : -((n + 1 : ℝ) - b) / b + (n + 1) / b = 1 := by field_simp; ring
    rw [this, Real.rpow_one]
  have h2 : (2 * (p : ℝ) * (n + 1)) ^ (-((n + 1 : ℝ) / 2)) * ((n + 1) * p / 2 : ℝ) ^ ((n + 1 : ℝ) / 2)
      = (1 / 2) ^ (n + 1) := by
    rw [Real.rpow_neg (by positivity), ← div_eq_inv_mul, ← Real.div_rpow (by positivity)
      (by positivity)]
    have : ((n + 1) * p / 2 : ℝ) / (2 * p * (n + 1)) = (1 / 2) ^ 2 := by field_simp
    rw [this, ← Real.rpow_natCast (1 / 2 : ℝ) 2, ← Real.rpow_mul (by norm_num),
      show ((2 : ℕ) : ℝ) * ((n + 1 : ℝ) / 2) = ((n + 1 : ℕ) : ℝ) by push_cast; ring,
      Real.rpow_natCast]
  unfold Bp
  calc (n + 1 : ℝ) ^ (-((n + 1 : ℝ) - b) / b) *
        (2 * (2 * (p : ℝ) * (n + 1)) ^ (-((n + 1 : ℝ) / 2))) *
        ((n + 1 : ℝ) ^ ((n + 1 : ℝ) / b) * ((n + 1) * p / 2 : ℝ) ^ ((n + 1 : ℝ) / 2) *
          (2 ^ n / (n + 1)))
      = ((n + 1 : ℝ) ^ (-((n + 1 : ℝ) - b) / b) * (n + 1 : ℝ) ^ ((n + 1 : ℝ) / b)) *
          ((2 * (p : ℝ) * (n + 1)) ^ (-((n + 1 : ℝ) / 2)) *
            ((n + 1) * p / 2 : ℝ) ^ ((n + 1 : ℝ) / 2)) * (2 * (2 ^ n / (n + 1))) := by ring
    _ = 1 := by
        rw [h1, h2]
        field_simp
        have h3 : (1 / 2 : ℝ) ^ (n + 1) * 2 ^ (n + 1) = 1 := by rw [← mul_pow]; norm_num
        linear_combination h3

/-- **`prop:block`, attainment value.**  For `y ≥ 0` with `∑_j y_j^{b/ν} = 1`, the block
network has normalized margin `h / ‖θ‖_{2,b}^ν = ν^{-(ν-b)/b} B_p m(y)`. -/
theorem blockNet_normalized_margin {p n : ℕ} [Fact p.Prime] {b : ℝ} (hb : 0 < b)
    (y : Freq p → ℝ) (hy : ∀ j, 0 ≤ y j) (hbud : budget ((n + 1) / b) y = 1) :
    margin (netTable prodNeuron (blockNet p n y)) / normPow b (blockNet p n y) ^ ((n + 1 : ℝ) / b)
      = (n + 1 : ℝ) ^ (-((n + 1 : ℝ) - b) / b) * Bp p n * worst (covP p) y := by
  have hν : (0 : ℝ) < n + 1 := by positivity
  have hK : 0 ≤ Real.sqrt ((n + 1) * p / 2) := Real.sqrt_nonneg _
  have hC : (0 : ℝ) ≤ 2 ^ n / (n + 1) := by positivity
  rw [blockNet_margin y hy, normPow_blockNet y hy hbud]
  have hX : ((n + 1 : ℝ) * Real.sqrt ((n + 1) * p / 2) ^ b * (2 ^ n / (n + 1)) ^ (b / (n + 1))) ^
      ((n + 1 : ℝ) / b) = (n + 1 : ℝ) ^ ((n + 1 : ℝ) / b) *
        ((n + 1) * p / 2 : ℝ) ^ ((n + 1 : ℝ) / 2) * (2 ^ n / (n + 1)) := by
    rw [Real.mul_rpow (by positivity) (by positivity), Real.mul_rpow hν.le (by positivity),
      ← Real.rpow_mul hK, ← Real.rpow_mul hC, Real.sqrt_eq_rpow, ← Real.rpow_mul (by positivity)]
    congr 2
    · congr 1; field_simp
    · rw [show b / (n + 1) * ((n + 1) / b) = (1 : ℝ) by field_simp, Real.rpow_one]
  rw [hX]
  have hid := block_const_identity p n hb
  have hXpos : (0 : ℝ) < (n + 1 : ℝ) ^ ((n + 1 : ℝ) / b) *
      ((n + 1) * p / 2 : ℝ) ^ ((n + 1 : ℝ) / 2) * (2 ^ n / (n + 1)) := by
    have hP : (0 : ℝ) < p := by exact_mod_cast (Fact.out : p.Prime).pos
    positivity
  rw [div_eq_iff hXpos.ne']
  linear_combination (worst (covP p) y) * hid.symm

/-! ## Fourier coefficients of a product neuron and the sign rule (`lem:sign`) -/

section Sign

variable {p n : ℕ} [NeZero p]

/-- The one-dimensional transform `v̂(k) = ∑_a v(a) ρ^{-k a}`. -/
def dft (v : ZMod p → ℝ) (k : ZMod p) : ℂ := ∑ a : ZMod p, (v a : ℂ) * rho (-(k * a))

lemma rho_sum {ι : Type*} (s : Finset ι) (f : ι → ZMod p) :
    rho (∑ i ∈ s, f i) = ∏ i ∈ s, rho (f i) := by
  classical
  induction s using Finset.induction_on with
  | empty => simp [rho_zero']
  | insert i s hi ih => rw [Finset.sum_insert hi, Finset.prod_insert hi, pn_rho_add, ih]

lemma pn_rho_neg (x : ZMod p) : rho (-x) = (starRingEnd ℂ) (rho x) :=
  AddChar.map_neg_eq_conj _ x

/-- `v̂(-k) = conj(v̂(k))` for real `v`. -/
lemma dft_neg (v : ZMod p → ℝ) (k : ZMod p) : dft v (-k) = (starRingEnd ℂ) (dft v k) := by
  unfold dft
  rw [map_sum]
  refine Finset.sum_congr rfl fun a _ => ?_
  rw [map_mul, Complex.conj_ofReal, pn_rho_neg, neg_mul]

/-- The transform of a product neuron factorizes:
`f̂(k; l) = ∏_t û_t(k_t) · ŵ(l)`. -/
theorem fhat_prodNeuron (ω : ProdParam p n) (k : Fin n → ZMod p) (l : ZMod p) :
    fhat (prodNeuron ω) k l = (∏ t, dft (ω.u t) (k t)) * dft ω.w l := by
  classical
  unfold fhat prodNeuron dft
  have hsum : ∀ (x : Inp p n) (c : ZMod p),
      ((((∏ t, ω.u t (x t)) * ω.w c : ℝ)) : ℂ) * rho (-(∑ t, k t * x t + l * c)) =
        (∏ t, ((ω.u t (x t) : ℂ) * rho (-(k t * x t)))) * ((ω.w c : ℂ) * rho (-(l * c))) := by
    intro x c
    rw [neg_add, pn_rho_add, ← Finset.sum_neg_distrib, rho_sum]
    push_cast
    rw [Finset.prod_mul_distrib]
    ring
  simp_rw [hsum, ← Finset.mul_sum, ← Finset.sum_mul]
  congr 1
  rw [Finset.prod_univ_sum (t := fun _ : Fin n => (univ : Finset (ZMod p)))
    (f := fun t a => (ω.u t a : ℂ) * rho (-(k t * a))), Fintype.piFinset_univ]

/-- **`app:models`, product neuron.** `β_j = 2 p^{-ν} ∏_t û_t(j) · conj(ŵ(j))`. -/
theorem beta_prodNeuron (ω : ProdParam p n) (j : ℕ) :
    beta (prodNeuron ω) j =
      2 / (p : ℂ) ^ (n + 1) * (∏ t, dft (ω.u t) j) * (starRingEnd ℂ) (dft ω.w j) := by
  unfold beta
  rw [fhat_prodNeuron, dft_neg]
  ring

/-- The sign-flipped coefficient of a product neuron:
`β_{j,T} = 2 p^{-ν} ∏_{t∉T} û_t(j) ∏_{t∈T} conj(û_t(j)) · conj(ŵ(j))`. -/
theorem betaT_prodNeuron (ω : ProdParam p n) (j : ℕ) (T : Finset (Fin n)) :
    betaT (prodNeuron ω) j T =
      2 / (p : ℂ) ^ (n + 1) *
        (∏ t, if t ∈ T then (starRingEnd ℂ) (dft (ω.u t) j) else dft (ω.u t) j) *
          (starRingEnd ℂ) (dft ω.w j) := by
  have h : ∀ t : Fin n, dft (ω.u t) (if t ∈ T then -(j : ZMod p) else (j : ZMod p)) =
      if t ∈ T then (starRingEnd ℂ) (dft (ω.u t) j) else dft (ω.u t) j := by
    intro t
    split_ifs
    · exact dft_neg _ _
    · rfl
  unfold betaT
  rw [fhat_prodNeuron, dft_neg, Finset.prod_congr rfl (fun t _ => h t)]
  ring

/-- **Lemma `lem:sign` (product neurons).**  For each product neuron and frequency `j` there are
unit complex numbers `ξ_1, …, ξ_n` with `β_{j,T} = β_j ∏_{t∈T} ξ_t` for every `T ⊆ [n]`. -/
theorem lem_sign (ω : ProdParam p n) (j : ℕ) :
    ∃ ξ : Fin n → ℂ, (∀ t, ‖ξ t‖ = 1) ∧
      ∀ T : Finset (Fin n), betaT (prodNeuron ω) j T = beta (prodNeuron ω) j * ∏ t ∈ T, ξ t := by
  classical
  set U : Fin n → ℂ := fun t => dft (ω.u t) j with hU
  let ξ : Fin n → ℂ := fun t => if U t = 0 then 1 else (starRingEnd ℂ) (U t) / U t
  have hξ : ∀ t, ‖ξ t‖ = 1 := by
    intro t
    by_cases h : U t = 0
    · simp [ξ, h]
    · simp only [ξ, h, ite_false, norm_div, Complex.norm_conj]
      exact div_self (norm_ne_zero_iff.2 h)
  have hconj : ∀ t, (starRingEnd ℂ) (U t) = U t * ξ t := by
    intro t
    by_cases h : U t = 0
    · simp [ξ, h]
    · simp only [ξ, h, ite_false]
      field_simp
  refine ⟨ξ, hξ, fun T => ?_⟩
  rw [betaT_prodNeuron, beta_prodNeuron]
  have hprod : (∏ t, if t ∈ T then (starRingEnd ℂ) (dft (ω.u t) j) else dft (ω.u t) j) =
      (∏ t, dft (ω.u t) j) * ∏ t ∈ T, ξ t := by
    rw [show (∏ t ∈ T, ξ t) = ∏ t, (if t ∈ T then ξ t else 1) by
      rw [Finset.prod_ite_mem, Finset.univ_inter], ← Finset.prod_mul_distrib]
    refine Finset.prod_congr rfl fun t _ => ?_
    split_ifs
    · exact hconj t
    · ring
  rw [hprod]
  ring

end Sign

/-! ## The single-neuron budget of a product neuron (appendix `app:models`) -/

section Budget

variable {p n : ℕ}

/-- **Parseval on `ℤ_p`**: `∑_k |v̂(k)|² = p ∑_a v(a)²`. -/
theorem dft_parseval [NeZero p] (v : ZMod p → ℝ) :
    ∑ k : ZMod p, ‖dft v k‖ ^ 2 = p * ∑ a, v a ^ 2 := by
  classical
  have hterm : ∀ k : ZMod p, dft v k * (starRingEnd ℂ) (dft v k) =
      ∑ a : ZMod p, ∑ b : ZMod p, ((v a * v b : ℝ) : ℂ) * rho (k * (b - a)) := by
    intro k
    unfold dft
    rw [map_sum, Finset.sum_mul_sum]
    refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ => ?_
    rw [map_mul, Complex.conj_ofReal]
    have hc : (starRingEnd ℂ) (rho (-(k * b))) = rho (k * b) := by
      rw [pn_rho_neg, Complex.conj_conj]
    rw [hc, show k * (b - a) = -(k * a) + k * b by ring, pn_rho_add]
    push_cast
    ring
  have horth : ∀ d : ZMod p, ∑ k : ZMod p, rho (k * d) = if d = 0 then (p : ℂ) else 0 := by
    intro d
    have := AddChar.sum_mulShift d (ZMod.isPrimitive_stdAddChar p)
    unfold rho
    rw [this]
    split_ifs <;> simp [ZMod.card]
  have key : ∑ k : ZMod p, dft v k * (starRingEnd ℂ) (dft v k) =
      (p : ℂ) * ∑ a, ((v a ^ 2 : ℝ) : ℂ) := by
    simp_rw [hterm]
    calc ∑ k : ZMod p, ∑ a : ZMod p, ∑ b : ZMod p, ((v a * v b : ℝ) : ℂ) * rho (k * (b - a))
        = ∑ a : ZMod p, ∑ b : ZMod p, ((v a * v b : ℝ) : ℂ) * ∑ k : ZMod p, rho (k * (b - a)) := by
          rw [Finset.sum_comm]
          refine Finset.sum_congr rfl fun a _ => ?_
          rw [Finset.sum_comm]
          refine Finset.sum_congr rfl fun b _ => ?_
          rw [Finset.mul_sum]
      _ = ∑ a : ZMod p, ∑ b : ZMod p, (if b = a then ((v a * v b : ℝ) : ℂ) * p else 0) := by
          simp_rw [horth, sub_eq_zero, mul_ite, mul_zero]
      _ = (p : ℂ) * ∑ a, ((v a ^ 2 : ℝ) : ℂ) := by
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun a _ => ?_
          rw [Finset.sum_ite_eq']
          simp only [Finset.mem_univ, ite_true]
          push_cast
          ring
  have hre : ∀ k, dft v k * (starRingEnd ℂ) (dft v k) = ((‖dft v k‖ ^ 2 : ℝ) : ℂ) := by
    intro k
    rw [Complex.mul_conj, Complex.normSq_eq_norm_sq]
  simp_rw [hre] at key
  rw [← Complex.ofReal_sum, ← Complex.ofReal_sum, ← Complex.ofReal_natCast,
    ← Complex.ofReal_mul, Complex.ofReal_inj] at key
  exact key

lemma freq_lt {p : ℕ} (j : Freq p) : j.1 < p := by
  have hj := Finset.mem_Icc.1 j.2
  omega

/-- **Half Parseval**: `∑_{j ∈ F_p} |v̂(j)|² ≤ (p/2) ‖v‖²` for real `v`. -/
theorem dft_half_parseval [Fact p.Prime] (v : ZMod p → ℝ) :
    ∑ j : Freq p, ‖dft v (j.1 : ZMod p)‖ ^ 2 ≤ (p / 2) * ∑ a, v a ^ 2 := by
  classical
  let g : Freq p ⊕ Freq p → ZMod p := Sum.elim (fun j => (j.1 : ZMod p)) (fun j => -(j.1 : ZMod p))
  have hcast : ∀ j j' : Freq p, (j.1 : ZMod p) = (j'.1 : ZMod p) → j = j' := by
    intro j j' h
    apply Subtype.ext
    have := congrArg ZMod.val h
    rwa [ZMod.val_cast_of_lt (freq_lt j), ZMod.val_cast_of_lt (freq_lt j')] at this
  have hmix : ∀ j j' : Freq p, (j.1 : ZMod p) ≠ -(j'.1 : ZMod p) := by
    intro j j' h
    have h2 : ((j.1 + j'.1 : ℕ) : ZMod p) = 0 := by push_cast; rw [h]; ring
    rw [ZMod.natCast_eq_zero_iff] at h2
    have hj := Finset.mem_Icc.1 j.2
    have hj' := Finset.mem_Icc.1 j'.2
    have := Nat.le_of_dvd (by omega) h2
    omega
  have hg : Function.Injective g := by
    rintro (j | j) (j' | j') h
    · simp only [g, Sum.elim_inl] at h; rw [hcast j j' h]
    · exact absurd h (hmix j j')
    · exact absurd h.symm (hmix j' j)
    · simp only [g, Sum.elim_inr, neg_inj] at h; rw [hcast j j' h]
  have hsum : ∑ i : Freq p ⊕ Freq p, ‖dft v (g i)‖ ^ 2 =
      2 * ∑ j : Freq p, ‖dft v (j.1 : ZMod p)‖ ^ 2 := by
    rw [Fintype.sum_sum_type]
    simp only [g, Sum.elim_inl, Sum.elim_inr, dft_neg, Complex.norm_conj]
    ring
  have hle : ∑ i : Freq p ⊕ Freq p, ‖dft v (g i)‖ ^ 2 ≤ ∑ k : ZMod p, ‖dft v k‖ ^ 2 := by
    rw [← Finset.sum_image (f := fun k => ‖dft v k‖ ^ 2) (fun i _ j _ h => hg h)]
    exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
      (fun k _ _ => sq_nonneg _)
  rw [dft_parseval] at hle
  linarith

/-- `B_p = 2 p^{-ν} (p / (2ν))^{ν/2}`. -/
lemma Bp_eq_prod (p n : ℕ) [NeZero p] :
    Bp p n = 2 / (p : ℝ) ^ (n + 1) * ((p : ℝ) / (2 * (n + 1))) ^ ((n + 1 : ℝ) / 2) := by
  have hP : (0 : ℝ) < p := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne p)
  have hν : (0 : ℝ) < 2 * (n + 1) := by positivity
  unfold Bp
  have e1 : (2 * (p : ℝ) * (n + 1)) ^ (-((n + 1 : ℝ) / 2)) =
      ((p : ℝ) ^ ((n + 1 : ℝ) / 2) * (2 * (n + 1)) ^ ((n + 1 : ℝ) / 2))⁻¹ := by
    rw [Real.rpow_neg (by positivity), show (2 * (p : ℝ) * (n + 1)) = p * (2 * (n + 1)) by ring,
      Real.mul_rpow hP.le hν.le]
  have e2 : (p : ℝ) ^ (n + 1) = (p : ℝ) ^ ((n + 1 : ℝ) / 2) * (p : ℝ) ^ ((n + 1 : ℝ) / 2) := by
    rw [← Real.rpow_add hP, add_halves, ← Real.rpow_natCast]
    push_cast; ring_nf
  have e3 : ((p : ℝ) / (2 * (n + 1))) ^ ((n + 1 : ℝ) / 2) =
      (p : ℝ) ^ ((n + 1 : ℝ) / 2) / (2 * (n + 1)) ^ ((n + 1 : ℝ) / 2) :=
    Real.div_rpow hP.le hν.le _
  have h1 : (0 : ℝ) < (p : ℝ) ^ ((n + 1 : ℝ) / 2) := Real.rpow_pos_of_pos hP _
  have h2 : (0 : ℝ) < (2 * (n + 1) : ℝ) ^ ((n + 1 : ℝ) / 2) := Real.rpow_pos_of_pos hν _
  rw [e1, e3, e2]
  field_simp

/-- AM–GM for the `ν` factors: `(∏ X_i)^{b/ν} ≤ (∑ X_i² / ν)^{b/2}`. -/
lemma amgm_rpow (X : Option (Fin n) → ℝ) (hX : ∀ i, 0 ≤ X i) {b : ℝ} (hb : 0 ≤ b) :
    (∏ i, X i) ^ (b / (n + 1)) ≤ ((∑ i, X i ^ 2) / (n + 1)) ^ (b / 2) := by
  have hν : (0 : ℝ) < n + 1 := by positivity
  have h1 := Real.geom_mean_le_arith_mean_weighted (s := univ) (fun _ => 1 / (n + 1 : ℝ))
    (fun i => X i ^ 2) (fun _ _ => by positivity)
    (by simp [Finset.card_univ, Fintype.card_option]; field_simp) (fun i _ => sq_nonneg _)
  rw [Real.finsetProd_rpow _ _ (fun i _ => sq_nonneg _), Finset.prod_pow, ← Finset.mul_sum] at h1
  have hP : 0 ≤ ∏ i, X i := Finset.prod_nonneg fun i _ => hX i
  have h2 := Real.rpow_le_rpow (by positivity) h1 (by linarith : 0 ≤ b / 2)
  have hL : (((∏ i, X i) ^ 2) ^ (1 / (n + 1 : ℝ))) ^ (b / 2) = (∏ i, X i) ^ (b / (n + 1)) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hP, ← Real.rpow_mul hP]
    congr 1
    push_cast
    field_simp
  rw [hL, show 1 / (n + 1 : ℝ) * ∑ i, X i ^ 2 = (∑ i, X i ^ 2) / (n + 1) by ring] at h2
  exact h2

/-- Superadditivity of `t ↦ t^q` for `q ≥ 1` on non-negative reals. -/
lemma sum_rpow_le_rpow_sum {ι : Type*} (s : Finset ι) (f : ι → ℝ) (hf : ∀ i ∈ s, 0 ≤ f i)
    {q : ℝ} (hq : 1 ≤ q) : ∑ i ∈ s, f i ^ q ≤ (∑ i ∈ s, f i) ^ q := by
  have hS : 0 ≤ ∑ i ∈ s, f i := Finset.sum_nonneg hf
  have hq0 : (1 : ℝ) + (q - 1) ≠ 0 := by linarith
  have hsplit : ∀ x : ℝ, 0 ≤ x → x ^ q = x * x ^ (q - 1) := by
    intro x hx
    rw [show q = 1 + (q - 1) by ring, Real.rpow_add' hx hq0, Real.rpow_one]
    ring_nf
  calc ∑ i ∈ s, f i ^ q = ∑ i ∈ s, f i * f i ^ (q - 1) :=
        Finset.sum_congr rfl fun i hi => hsplit _ (hf i hi)
    _ ≤ ∑ i ∈ s, f i * (∑ i ∈ s, f i) ^ (q - 1) := by
        refine Finset.sum_le_sum fun i hi => mul_le_mul_of_nonneg_left ?_ (hf i hi)
        exact Real.rpow_le_rpow (hf i hi) (Finset.single_le_sum hf hi) (by linarith)
    _ = (∑ i ∈ s, f i) ^ q := by rw [← Finset.sum_mul, hsplit _ hS]

/-- The norm of `β_j` for a product neuron. -/
lemma norm_beta_prodNeuron [NeZero p] (ω : ProdParam p n) (j : ℕ) :
    ‖beta (prodNeuron ω) j‖ =
      2 / (p : ℝ) ^ (n + 1) * ((∏ t, ‖dft (ω.u t) j‖) * ‖dft ω.w j‖) := by
  rw [beta_prodNeuron, norm_mul, norm_mul, norm_div, norm_prod, Complex.norm_conj, norm_pow,
    Complex.norm_natCast, Complex.norm_ofNat]
  ring

/-- **The single-neuron budget of a product neuron** (appendix `app:models`): for `b ≥ 2`,
`∑_{j∈F_p} |β_j(ω)|^{b/ν} ≤ B_p^{b/ν} ‖ω‖_2^b`, i.e. `C_b^{ν/b} ≤ B_p`. -/
theorem prod_budget_two_le [Fact p.Prime] (ω : ProdParam p n) {b : ℝ} (hb : 2 ≤ b) :
    ∑ j : Freq p, ‖beta (prodNeuron ω) j.1‖ ^ (b / (n + 1)) ≤
      Bp p n ^ (b / (n + 1)) * Real.sqrt (normSq ω) ^ b := by
  have hν : (0 : ℝ) < n + 1 := by positivity
  have hP : (0 : ℝ) < p := by exact_mod_cast (Fact.out : p.Prime).pos
  have hb0 : 0 ≤ b := by linarith
  set c0 : ℝ := 2 / (p : ℝ) ^ (n + 1) with hc0
  have hc0pos : 0 ≤ c0 := by positivity
  let X : Freq p → Option (Fin n) → ℝ := fun j i =>
    Option.elim i ‖dft ω.w (j.1 : ZMod p)‖ (fun t => ‖dft (ω.u t) (j.1 : ZMod p)‖)
  have hX : ∀ j i, 0 ≤ X j i := by
    intro j i; cases i <;> simp [X]
  have hprodX : ∀ j : Freq p, ∏ i, X j i =
      (∏ t, ‖dft (ω.u t) (j.1 : ZMod p)‖) * ‖dft ω.w (j.1 : ZMod p)‖ := by
    intro j; rw [Fintype.prod_option]; simp [X]; ring
  set M : Freq p → ℝ := fun j => ∑ i, X j i ^ 2 with hM
  have hMsum : ∑ j, M j ≤ (p / 2) * normSq ω := by
    have : ∀ j : Freq p, M j = ‖dft ω.w (j.1 : ZMod p)‖ ^ 2 +
        ∑ t, ‖dft (ω.u t) (j.1 : ZMod p)‖ ^ 2 := by
      intro j; simp only [hM]; rw [Fintype.sum_option]; simp [X]
    simp_rw [this]
    rw [Finset.sum_add_distrib, Finset.sum_comm]
    unfold normSq
    have h1 := dft_half_parseval ω.w
    have h2 : ∀ t, ∑ j : Freq p, ‖dft (ω.u t) (j.1 : ZMod p)‖ ^ 2 ≤ (p / 2) * ∑ a, ω.u t a ^ 2 :=
      fun t => dft_half_parseval (ω.u t)
    have h3 := Finset.sum_le_sum fun t (_ : t ∈ (univ : Finset (Fin n))) => h2 t
    rw [← Finset.mul_sum] at h3
    linarith
  have hM0 : ∀ j, 0 ≤ M j := fun j => Finset.sum_nonneg fun i _ => sq_nonneg _
  have hstep : ∀ j : Freq p, ‖beta (prodNeuron ω) j.1‖ ^ (b / (n + 1)) ≤
      c0 ^ (b / (n + 1)) * (M j / (n + 1)) ^ (b / 2) := by
    intro j
    rw [norm_beta_prodNeuron, ← hprodX, Real.mul_rpow hc0pos
      (Finset.prod_nonneg fun i _ => hX j i)]
    exact mul_le_mul_of_nonneg_left (amgm_rpow (X j) (hX j) hb0) (by positivity)
  calc ∑ j : Freq p, ‖beta (prodNeuron ω) j.1‖ ^ (b / (n + 1))
      ≤ ∑ j : Freq p, c0 ^ (b / (n + 1)) * (M j / (n + 1)) ^ (b / 2) :=
        Finset.sum_le_sum fun j _ => hstep j
    _ = c0 ^ (b / (n + 1)) * ∑ j : Freq p, (M j / (n + 1)) ^ (b / 2) := by rw [Finset.mul_sum]
    _ ≤ c0 ^ (b / (n + 1)) * (∑ j : Freq p, M j / (n + 1)) ^ (b / 2) := by
        refine mul_le_mul_of_nonneg_left ?_ (by positivity)
        exact sum_rpow_le_rpow_sum _ _ (fun j _ => div_nonneg (hM0 j) hν.le) (by linarith)
    _ ≤ c0 ^ (b / (n + 1)) * ((p / (2 * (n + 1))) * normSq ω) ^ (b / 2) := by
        refine mul_le_mul_of_nonneg_left ?_ (by positivity)
        refine Real.rpow_le_rpow (Finset.sum_nonneg fun j _ => div_nonneg (hM0 j) hν.le) ?_
          (by linarith)
        rw [← Finset.sum_div, div_le_iff₀ hν]
        have : p / (2 * (n + 1)) * normSq ω * (n + 1) = (p / 2) * normSq ω := by field_simp
        rw [this]; exact hMsum
    _ = Bp p n ^ (b / (n + 1)) * Real.sqrt (normSq ω) ^ b := by
        have hN : 0 ≤ normSq ω := by
          unfold normSq
          exact add_nonneg (Finset.sum_nonneg fun t _ => Finset.sum_nonneg fun a _ => sq_nonneg _)
            (Finset.sum_nonneg fun c _ => sq_nonneg _)
        rw [Bp_eq_prod, Real.mul_rpow hc0pos (by positivity), ← Real.rpow_mul (by positivity),
          Real.mul_rpow (by positivity) hN, Real.sqrt_eq_rpow, ← Real.rpow_mul hN]
        rw [show (n + 1 : ℝ) / 2 * (b / (n + 1)) = b / 2 by field_simp,
          show (1 / 2 : ℝ) * b = b / 2 by ring]
        ring

/-- `prod_budget` in the shape used by the network-level glue (`2 ≤ b ≤ ν`). -/
theorem prod_budget [Fact p.Prime] (ω : ProdParam p n) {b : ℝ} (hb2 : 2 ≤ b)
    (_hbν : b ≤ n + 1) :
    ∑ j : Freq p, ‖beta (prodNeuron ω) j.1‖ ^ (b / (n + 1)) ≤
      Bp p n ^ (b / (n + 1)) * Real.sqrt (normSq ω) ^ b :=
  prod_budget_two_le ω hb2

end Budget

/-! ## Attainment of the single-neuron budget: `C_b^{ν/b} = B_p` -/

section Attain

variable {p n : ℕ}

lemma pn_sum_rho_mul [NeZero p] (d : ZMod p) :
    ∑ a : ZMod p, rho (a * d) = if d = 0 then (p : ℂ) else 0 := by
  classical
  have := AddChar.sum_mulShift d (ZMod.isPrimitive_stdAddChar p)
  unfold rho
  rw [this]
  split_ifs <;> simp [ZMod.card]

lemma rho_val [NeZero p] (a : ZMod p) :
    rho a = Complex.exp (2 * π * Complex.I * (a.val : ℝ) / p) := by
  have h := ZMod.stdAddChar_coe (N := p) (a.val : ℤ)
  rw [Int.cast_natCast, ZMod.natCast_zmod_val] at h
  unfold rho
  rw [h]
  push_cast
  ring_nf

lemma cos_eq_rho [NeZero p] (a : ZMod p) :
    ((Real.cos (2 * π * (a.val : ℝ) / p) : ℝ) : ℂ) = (rho a + rho (-a)) / 2 := by
  have h1 : rho a = Ex (2 * π * (a.val : ℝ) / p) := by
    rw [rho_val]; unfold Ex; congr 1; push_cast; ring
  have h2 : rho (-a) = Ex (-(2 * π * (a.val : ℝ) / p)) := by
    rw [pn_rho_neg, h1]
    unfold Ex
    rw [← Complex.exp_conj]
    congr 1
    simp only [map_mul, Complex.conj_ofReal, Complex.conj_I]
    push_cast
    ring
  rw [h1, h2, ← two_cos_eq_Ex]
  push_cast
  ring

/-- The one-frequency cosine neuron: every weight vector is `λ cos(2π a/p)`. -/
def cosNeuron (p n : ℕ) (lam : ℝ) : ProdParam p n where
  u := fun _ a => lam * Real.cos (2 * π * (a.val : ℝ) / p)
  w := fun c => lam * Real.cos (2 * π * (c.val : ℝ) / p)

lemma dft_cos [NeZero p] (lam : ℝ) (k : ZMod p) :
    dft (fun a => lam * Real.cos (2 * π * (a.val : ℝ) / p)) k =
      lam / 2 * ((if 1 - k = 0 then (p : ℂ) else 0) + (if -1 - k = 0 then (p : ℂ) else 0)) := by
  unfold dft
  have h : ∀ a : ZMod p, ((lam * Real.cos (2 * π * (a.val : ℝ) / p) : ℝ) : ℂ) * rho (-(k * a)) =
      lam / 2 * (rho (a * (1 - k)) + rho (a * (-1 - k))) := by
    intro a
    rw [Complex.ofReal_mul, cos_eq_rho]
    rw [show a * (1 - k) = a + -(k * a) by ring, show a * (-1 - k) = -a + -(k * a) by ring,
      pn_rho_add, pn_rho_add]
    ring
  simp_rw [h]
  rw [← Finset.mul_sum, Finset.sum_add_distrib, pn_sum_rho_mul, pn_sum_rho_mul]

lemma dft_cos_freq [Fact p.Prime] (hp : 3 ≤ p) (lam : ℝ) (j : Freq p) :
    dft (fun a => lam * Real.cos (2 * π * (a.val : ℝ) / p)) (j.1 : ZMod p) =
      if j.1 = 1 then ((lam * p / 2 : ℝ) : ℂ) else 0 := by
  rw [dft_cos]
  have hj := Finset.mem_Icc.1 j.2
  have hneg : ¬ ((-1 : ZMod p) - (j.1 : ZMod p) = 0) := by
    intro h
    have h2 : ((j.1 + 1 : ℕ) : ZMod p) = 0 := by push_cast; linear_combination -h
    rw [ZMod.natCast_eq_zero_iff] at h2
    have := Nat.le_of_dvd (by omega) h2
    omega
  rw [ite_eq_right hneg, add_zero]
  by_cases h1 : j.1 = 1
  · rw [ite_eq_left h1, ite_eq_left (by rw [h1]; simp)]
    push_cast; ring
  · rw [ite_eq_right h1, ite_eq_right, mul_zero]
    intro h
    apply h1
    have h' : ((j.1 : ℕ) : ZMod p) = ((1 : ℕ) : ZMod p) := by push_cast; linear_combination -h
    have := congrArg ZMod.val h'
    rwa [ZMod.val_cast_of_lt (freq_lt j), ZMod.val_cast_of_lt (by omega)] at this

lemma cosNeuron_normSq [Fact p.Prime] (hp : 3 ≤ p) (lam : ℝ) :
    normSq (cosNeuron p n lam) = (n + 1) * lam ^ 2 * p / 2 := by
  have hpd : ¬ p ∣ 2 * 1 := by
    intro h; have := Nat.le_of_dvd (by norm_num) h; omega
  have hu : ∑ a : ZMod p, (lam * Real.cos (2 * π * (a.val : ℝ) / p)) ^ 2 = lam ^ 2 * (p / 2) := by
    simp_rw [mul_pow]
    rw [← Finset.mul_sum]
    have := sum_cos_sq_zmod (p := p) 1 hpd 0
    simp only [Nat.cast_one, mul_one, add_zero] at this
    rw [this]
  unfold normSq cosNeuron
  simp only
  rw [Finset.sum_congr rfl fun t _ => hu, hu]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  ring

/-- The coefficients of the cosine neuron: only frequency `1` is present. -/
lemma norm_beta_cosNeuron [Fact p.Prime] (hp : 3 ≤ p) {lam : ℝ} (hlam : 0 ≤ lam) (j : Freq p) :
    ‖beta (prodNeuron (cosNeuron p n lam)) j.1‖ =
      if j.1 = 1 then 2 / (p : ℝ) ^ (n + 1) * (lam * p / 2) ^ (n + 1) else 0 := by
  rw [norm_beta_prodNeuron]
  unfold cosNeuron
  simp only
  simp_rw [dft_cos_freq hp lam j]
  have hP : (0 : ℝ) ≤ p := Nat.cast_nonneg p
  split_ifs with h
  · simp only [Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg (by positivity :
      (0 : ℝ) ≤ lam * p / 2), Finset.prod_const, Finset.card_univ, Fintype.card_fin]
    ring
  · simp

/-- **Attainment of the budget.** The cosine neuron with `‖ω‖_2 = 1` has
`∑_{j∈F_p} |β_j|^{b/ν} = B_p^{b/ν}`. -/
theorem prod_budget_attained [Fact p.Prime] (hp : 3 ≤ p) {b : ℝ} (hb : 0 < b) :
    ∃ ω : ProdParam p n, normSq ω = 1 ∧
      ∑ j : Freq p, ‖beta (prodNeuron ω) j.1‖ ^ (b / (n + 1)) = Bp p n ^ (b / (n + 1)) := by
  have hP : (0 : ℝ) < p := by exact_mod_cast (Fact.out : p.Prime).pos
  have hν : (0 : ℝ) < n + 1 := by positivity
  set lam : ℝ := Real.sqrt (2 / ((n + 1) * p)) with hlam
  have hlam0 : 0 ≤ lam := Real.sqrt_nonneg _
  have hlam2 : lam ^ 2 = 2 / ((n + 1) * p) := Real.sq_sqrt (by positivity)
  refine ⟨cosNeuron p n lam, ?_, ?_⟩
  · rw [cosNeuron_normSq hp, hlam2]; field_simp
  · have h1 : (1 : ℕ) ∈ Finset.Icc 1 ((p - 1) / 2) := Finset.mem_Icc.2 ⟨le_rfl, by omega⟩
    rw [Finset.sum_eq_single (⟨1, h1⟩ : Freq p)]
    · rw [norm_beta_cosNeuron hp hlam0, ite_eq_left rfl]
      congr 1
      rw [Bp_eq_prod]
      congr 1
      rw [show (lam * p / 2) ^ (n + 1) = ((lam * p / 2) ^ 2) ^ ((n + 1 : ℝ) / 2) by
        rw [← Real.rpow_natCast, ← Real.rpow_natCast, ← Real.rpow_mul (by positivity)]
        congr 1; push_cast; ring]
      congr 1
      rw [div_pow, mul_pow, hlam2]
      field_simp
    · intro j _ hj
      rw [norm_beta_cosNeuron hp hlam0, ite_eq_right, Real.zero_rpow (by positivity)]
      intro h; apply hj; exact Subtype.ext h
    · simp

/-- The single-neuron budget constant `C_b = max_{‖ω‖_2 ≤ 1} ∑_{j∈F_p} |β_j(ω)|^{b/ν}` of
Section `sec:approach`, for product neurons. -/
def CbProd (p n : ℕ) [Fact p.Prime] (b : ℝ) : ℝ :=
  sSup ((fun ω : ProdParam p n => ∑ j : Freq p, ‖beta (prodNeuron ω) j.1‖ ^ (b / (n + 1))) ''
    {ω | normSq ω ≤ 1})

/-- **`C_b^{ν/b} = B_p` for the product neuron** (Section `sec:approach`, appendix
`app:models`), for every `b ≥ 2`. -/
theorem CbProd_eq [Fact p.Prime] (hp : 3 ≤ p) {b : ℝ} (hb : 2 ≤ b) :
    CbProd p n b = Bp p n ^ (b / (n + 1)) ∧ CbProd p n b ^ ((n + 1) / b) = Bp p n := by
  have hb0 : 0 < b := by linarith
  have hν : (0 : ℝ) < n + 1 := by positivity
  have hBp : 0 ≤ Bp p n := by unfold Bp; positivity
  have hup : ∀ ω : ProdParam p n, normSq ω ≤ 1 →
      ∑ j : Freq p, ‖beta (prodNeuron ω) j.1‖ ^ (b / (n + 1)) ≤ Bp p n ^ (b / (n + 1)) := by
    intro ω hω
    have hN : 0 ≤ normSq ω := by
      unfold normSq
      exact add_nonneg (Finset.sum_nonneg fun t _ => Finset.sum_nonneg fun a _ => sq_nonneg _)
        (Finset.sum_nonneg fun c _ => sq_nonneg _)
    have h1 := prod_budget_two_le ω hb
    have h2 : Real.sqrt (normSq ω) ^ b ≤ 1 :=
      Real.rpow_le_one (Real.sqrt_nonneg _) (Real.sqrt_le_one.2 hω) hb0.le
    have h3 : 0 ≤ Bp p n ^ (b / (n + 1)) := Real.rpow_nonneg hBp _
    nlinarith
  obtain ⟨ω0, hω0, heq⟩ := prod_budget_attained (n := n) hp hb0
  have hC : CbProd p n b = Bp p n ^ (b / (n + 1)) := by
    apply le_antisymm
    · exact csSup_le ⟨_, ω0, hω0.le, rfl⟩ (by rintro _ ⟨ω, hω, rfl⟩; exact hup ω hω)
    · rw [← heq]
      exact le_csSup ⟨_, by rintro _ ⟨ω, hω, rfl⟩; exact hup ω hω⟩ ⟨ω0, hω0.le, rfl⟩
  refine ⟨hC, ?_⟩
  rw [hC, ← Real.rpow_mul hBp, show b / (n + 1) * ((n + 1) / b) = (1 : ℝ) by field_simp,
    Real.rpow_one]

end Attain

/-! ## `prop:block`, bundled -/

/-- **Proposition `prop:block`** (the `ν`-block), identity part: for `j ∈ F_p` and any `λ`,
the `ν` neurons `u_t^{(l)}(a) = λ cos(2πja/p + α_l)`, `w^{(l)}(c) = λ cos(2πjc/p + nα_l)`,
`α_l = πl/ν`, have total table `ν λ^ν 2^{-n} cos(2πj(a_1+⋯+a_n−c)/p)` and
`‖ω^{(l)}‖_2^2 = ν λ^2 p / 2`. -/
theorem prop_block {p n : ℕ} [Fact p.Prime] (j : Freq p) (lam : ℝ) :
    (∀ (x : Inp p n) (c : ZMod p),
      ∑ l ∈ range (n + 1), prodNeuron (blockNeuron p n j.1 lam l) x c =
        (n + 1) * lam ^ (n + 1) / 2 ^ n * Real.cos (2 * π * j.1 * ((label x - c).val : ℝ) / p)) ∧
    (∀ l : ℕ, normSq (blockNeuron p n j.1 lam l) = (n + 1) * lam ^ 2 * p / 2) :=
  ⟨fun x c => block_table j.1 lam x c, fun l => block_normSq j.1 (freq_not_dvd j) lam l⟩

/-- **`prop:block`, attainment of `thm:cyclic`.**  If `y` maximizes the programme at
`s = ν/b` and spends the whole budget (`∑_j y_j^{b/ν} = 1`, true for every maximizer), the
block network built on `y` is interference-free and has normalized margin
`ν^{-(ν-b)/b} B_p Γ_{ν/b}(p)`. -/
theorem block_attains {p n : ℕ} [Fact p.Prime] (hp : 3 ≤ p) {b : ℝ} (hb : 2 ≤ b)
    (hbν : b ≤ n + 1) (y : Freq p → ℝ) (hy : IsMaximizer (covP p) ((n + 1) / b) y)
    (hbud : budget ((n + 1) / b) y = 1) :
    InterferenceFree (netTable prodNeuron (blockNet p n y)) ∧
    margin (netTable prodNeuron (blockNet p n y)) / normPow b (blockNet p n y) ^ ((n + 1 : ℝ) / b)
      = (n + 1 : ℝ) ^ (-((n + 1 : ℝ) - b) / b) * Bp p n * GammaP p ((n + 1) / b) := by
  have hs : (1 : ℝ) ≤ (n + 1) / b := by rw [le_div_iff₀ (by linarith)]; linarith
  refine ⟨blockNet_interferenceFree hp y hy.1.1, ?_⟩
  rw [blockNet_normalized_margin (by linarith) y hy.1.1 hbud,
    hy.worst_eq (covP_le_two p) hs]
  rfl

end Heracles
