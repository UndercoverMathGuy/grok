import Mathlib

/-!
# Lemma `lem:cost` (cost of a feature), appendix `app:cost`

Setting: `n` operands, `ν = n + 1`, `N` neurons, coefficients `β : Fin N → ℂ`, unit phases
`ξ : Fin N → Fin n → ℂ` (`‖ξ l t‖ = 1`), total `B`, and the cancellation system (JF):
`∑_l β_l ∏_{t ∈ T} ξ_{l t} = [T = ∅] B` for every `T ⊆ [n]`.
Chain vectors `m_l ∈ ℂ^ν`, `m_l(i) = ∏_{t < i} ξ_{l t}` (`i = 0, …, n`).

Main results:
* `chain_lemma` (lem:chain): `‖m_l‖² = ν`, `H = ∑ β_l m_l m_l^*` is upper triangular with
  diagonal `B`.
* `det_one_add_smul_expand` (lem:minor, principal-minor expansion) and
  `det_submatrix_diagonal_mul` (factor `∏ d_l`); Sylvester is Mathlib's
  `Matrix.det_one_add_mul_comm`.
* `norm_det_le_of_posSemidef` (lem:hadamard, the inequality part, for PSD matrices with
  constant diagonal; proved by AM–GM on the eigenvalues).
* `prop_gen` (prop:gen): `∏_l (1 + x_l t) ≥ (1+t)^ν` for `t ≥ 0`, `x_l = ν |β_l| / |B|`.
* `bernstein_scaling` (lem:bernstein, in scaling form):
  `∫_0^∞ log(1+xt) t^{-1-r} dt = x^r K_r` with `0 < K_r < ∞`.
* `lem_cost`: `∑_l |β_l|^r ≥ ν^{1-r} |B|^r` for `r ∈ (0,1]`.
* `lem_cost_eq`: equality case for `r < 1`.
-/

open Finset Matrix MeasureTheory Set
open scoped ComplexOrder

noncomputable section

namespace Heracles
namespace Cost

variable {n N : ℕ}

/-- Chain vector `m_l(i) = ∏_{t < i} ξ_{l t}` for `i = 0, …, n`. -/
def chain (ξ : Fin N → Fin n → ℂ) (l : Fin N) (i : Fin (n + 1)) : ℂ :=
  ∏ t ∈ Finset.univ.filter (fun t : Fin n => (t : ℕ) < (i : ℕ)), ξ l t

/-- The cancellation system (JF). -/
def JF (β : Fin N → ℂ) (ξ : Fin N → Fin n → ℂ) (B : ℂ) : Prop :=
  ∀ T : Finset (Fin n), ∑ l, β l * ∏ t ∈ T, ξ l t = if T = ∅ then B else 0

/-- `H = ∑_l β_l m_l m_l^*`. -/
def Hmat (β : Fin N → ℂ) (ξ : Fin N → Fin n → ℂ) : Matrix (Fin (n + 1)) (Fin (n + 1)) ℂ :=
  Matrix.of fun i j => ∑ l, β l * (chain ξ l i * star (chain ξ l j))

/-- The `ν × N` matrix whose columns are the chain vectors. -/
def Umat (ξ : Fin N → Fin n → ℂ) : Matrix (Fin (n + 1)) (Fin N) ℂ :=
  Matrix.of fun i l => chain ξ l i

/-- The Gram matrix of the chain vectors, `G_{l l'} = ⟨m_l, m_{l'}⟩`. -/
def Gram (ξ : Fin N → Fin n → ℂ) : Matrix (Fin N) (Fin N) ℂ := (Umat ξ)ᴴ * Umat ξ

section Chain

variable {ξ : Fin N → Fin n → ℂ}

lemma norm_chain (hξ : ∀ l t, ‖ξ l t‖ = 1) (l : Fin N) (i : Fin (n + 1)) :
    ‖chain ξ l i‖ = 1 := by
  unfold chain
  rw [norm_prod]
  exact Finset.prod_eq_one fun t _ => hξ l t

lemma mul_star_eq_one {z : ℂ} (hz : ‖z‖ = 1) : z * star z = 1 := by
  rw [Complex.star_def, Complex.mul_conj, Complex.normSq_eq_norm_sq, hz]
  simp

lemma star_mul_eq_one {z : ℂ} (hz : ‖z‖ = 1) : star z * z = 1 := by
  rw [mul_comm]; exact mul_star_eq_one hz

lemma chain_mul_star_self (hξ : ∀ l t, ‖ξ l t‖ = 1) (l : Fin N) (i : Fin (n + 1)) :
    chain ξ l i * star (chain ξ l i) = 1 :=
  mul_star_eq_one (norm_chain hξ l i)

/-- For `j < i`, `m_l(i) conj(m_l(j)) = ∏_{j ≤ t < i} ξ_{l t}`. -/
lemma chain_mul_star_of_lt (hξ : ∀ l t, ‖ξ l t‖ = 1) (l : Fin N) {i j : Fin (n + 1)}
    (hji : j < i) :
    chain ξ l i * star (chain ξ l j) =
      ∏ t ∈ Finset.univ.filter (fun t : Fin n => (j : ℕ) ≤ t ∧ (t : ℕ) < i), ξ l t := by
  have hji' : (j : ℕ) < i := hji
  set P := ∏ t ∈ Finset.univ.filter (fun t : Fin n => (t : ℕ) < (j : ℕ)), ξ l t with hP
  set Q := ∏ t ∈ Finset.univ.filter (fun t : Fin n => (j : ℕ) ≤ t ∧ (t : ℕ) < i), ξ l t
    with hQ
  have hsplit : Finset.univ.filter (fun t : Fin n => (t : ℕ) < (i : ℕ)) =
      Finset.univ.filter (fun t : Fin n => (t : ℕ) < (j : ℕ)) ∪
        Finset.univ.filter (fun t : Fin n => (j : ℕ) ≤ t ∧ (t : ℕ) < i) := by
    ext t
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_union]
    omega
  have hdisj : Disjoint (Finset.univ.filter (fun t : Fin n => (t : ℕ) < (j : ℕ)))
      (Finset.univ.filter (fun t : Fin n => (j : ℕ) ≤ t ∧ (t : ℕ) < i)) := by
    rw [Finset.disjoint_filter]
    intro t _ h1 h2
    omega
  have hci : chain ξ l i = P * Q := by
    unfold chain
    rw [hsplit, Finset.prod_union hdisj]
  have hcj : chain ξ l j = P := rfl
  have hPP : P * star P = 1 := by
    rw [← hcj]; exact chain_mul_star_self hξ l j
  rw [hci, hcj]
  calc P * Q * star P = Q * (P * star P) := by ring
    _ = Q := by rw [hPP, mul_one]

/-- **lem:chain.** `‖m_l‖² = ν`; `H` is upper triangular with every diagonal entry `B`. -/
theorem chain_lemma {β : Fin N → ℂ} {B : ℂ} (hξ : ∀ l t, ‖ξ l t‖ = 1) (hJF : JF β ξ B) :
    (∀ l, ∑ i, ‖chain ξ l i‖ ^ 2 = (n + 1 : ℝ)) ∧
      (∀ i j : Fin (n + 1), j < i → Hmat β ξ i j = 0) ∧
      (∀ i : Fin (n + 1), Hmat β ξ i i = B) := by
  refine ⟨fun l => ?_, fun i j hji => ?_, fun i => ?_⟩
  · simp [norm_chain hξ]
  · simp only [Hmat, Matrix.of_apply]
    simp_rw [chain_mul_star_of_lt hξ _ hji]
    have hne : (Finset.univ.filter
        (fun t : Fin n => (j : ℕ) ≤ t ∧ (t : ℕ) < i)) ≠ ∅ := by
      have hjn : (j : ℕ) < n := by have := i.2; have : (j : ℕ) < i := hji; omega
      intro h
      have hmem : (⟨j, hjn⟩ : Fin n) ∈ Finset.univ.filter
          (fun t : Fin n => (j : ℕ) ≤ t ∧ (t : ℕ) < i) := by
        simp only [Finset.mem_filter, Finset.mem_univ, true_and]
        exact ⟨le_rfl, hji⟩
      rw [h] at hmem
      simp at hmem
    rw [hJF, ite_eq_right_iff.2 (fun h => absurd h hne)]
  · simp only [Hmat, Matrix.of_apply]
    simp_rw [chain_mul_star_self hξ]
    have := hJF ∅
    simpa using this

/-- `H = U diag(β) U^*`. -/
lemma Hmat_eq (β : Fin N → ℂ) (ξ : Fin N → Fin n → ℂ) :
    Hmat β ξ = Umat ξ * (diagonal β * (Umat ξ)ᴴ) := by
  ext i j
  rw [Matrix.mul_apply]
  simp_rw [Matrix.diagonal_mul]
  simp only [Hmat, Umat, Matrix.of_apply, Matrix.conjTranspose_apply]
  refine Finset.sum_congr rfl fun l _ => ?_
  ring

/-- The diagonal of the Gram matrix: `‖m_l‖² = ν`. -/
lemma Gram_diag (hξ : ∀ l t, ‖ξ l t‖ = 1) (l : Fin N) : Gram ξ l l = (n + 1 : ℂ) := by
  simp only [Gram, Umat, Matrix.mul_apply, Matrix.conjTranspose_apply, Matrix.of_apply]
  simp_rw [star_mul_eq_one (norm_chain hξ l _)]
  simp

end Chain

/-! ## Principal-minor expansion (lem:minor) -/

section Minor

variable {ι : Type*} [Fintype ι] [DecidableEq ι] {R : Type*} [CommRing R]

/-- **lem:minor (expansion by principal minors).**
`det (1 + t M) = ∑_{L} t^{|L|} det M_{L,L}`. -/
theorem det_one_add_smul_expand (M : Matrix ι ι R) (t : R) :
    det (1 + t • M) =
      ∑ s : Finset ι, t ^ s.card * det (M.submatrix (Subtype.val : s → ι) Subtype.val) := by
  simp only [det]
  let D := (detRowAlternating : (ι → R) [⋀^ι]→ₗ[R] R)
  rw [add_comm]
  change D (fun i => (t • M) i + (1 : Matrix ι ι R) i) = _
  rw [show (fun i => (t • M) i + (1 : Matrix ι ι R) i) =
      (fun i => (t • M) i) + (fun i => (1 : Matrix ι ι R) i) from rfl, D.map_add_univ]
  refine Finset.sum_congr rfl fun s _ => ?_
  have h_smul : s.piecewise (fun i => (t • M) i) (fun i => (1 : Matrix ι ι R) i) =
      fun i => (if i ∈ s then t else 1) •
        s.piecewise (fun i => M i) (fun i => (1 : Matrix ι ι R) i) i := by
    funext i j
    simp only [Finset.piecewise, Pi.smul_apply, smul_eq_mul]
    split_ifs <;> simp
  rw [h_smul, D.map_smul_univ]
  simp only [Finset.prod_ite_mem, Finset.univ_inter, Finset.prod_const, smul_eq_mul]
  congr 1
  exact det_piecewise_one_eq_submatrix_det M s

/-- `det (diag(β) G)_{L,L} = (∏_{l ∈ L} β_l) det G_{L,L}`. -/
lemma det_submatrix_diagonal_mul (β : ι → R) (G : Matrix ι ι R) (s : Finset ι) :
    det ((diagonal β * G).submatrix (Subtype.val : s → ι) Subtype.val) =
      (∏ l ∈ s, β l) * det (G.submatrix (Subtype.val : s → ι) Subtype.val) := by
  have : (diagonal β * G).submatrix (Subtype.val : s → ι) (Subtype.val : s → ι) =
      diagonal (fun l : s => β l) * G.submatrix (Subtype.val : s → ι) (Subtype.val : s → ι) := by
    ext i j
    simp [Matrix.diagonal_mul]
  rw [this, det_mul, det_diagonal]
  congr 1
  exact Finset.prod_coe_sort s β

end Minor

/-! ## Hadamard-type bound (lem:hadamard) -/

/-- **lem:hadamard (inequality).** A positive semidefinite matrix whose diagonal entries all
equal `c` has `|det| ≤ c^k`.  (Proof: the eigenvalues are `≥ 0` with sum `k c`; AM–GM.) -/
theorem norm_det_le_of_posSemidef {κ : Type*} [Fintype κ] [DecidableEq κ]
    (A : Matrix κ κ ℂ) (hA : A.PosSemidef) (c : ℝ) (hdiag : ∀ i, A i i = c) :
    ‖A.det‖ ≤ c ^ Fintype.card κ := by
  have hev0 : ∀ i, 0 ≤ hA.1.eigenvalues i := hA.eigenvalues_nonneg
  have hdet : ‖A.det‖ = ∏ i, hA.1.eigenvalues i := by
    rw [hA.1.det_eq_prod_eigenvalues, norm_prod]
    refine Finset.prod_congr rfl fun i _ => ?_
    rw [RCLike.norm_ofReal, abs_of_nonneg (hev0 i)]
  have htr : ∑ i, hA.1.eigenvalues i = Fintype.card κ * c := by
    have h1 := hA.1.trace_eq_sum_eigenvalues
    have h2 : A.trace = (Fintype.card κ : ℂ) * c := by
      simp [Matrix.trace, Matrix.diag, hdiag]
    have h3 := congrArg Complex.re (h1.symm.trans h2)
    simpa using h3
  rw [hdet]
  rcases Nat.eq_zero_or_pos (Fintype.card κ) with hk0 | hkpos
  · have : IsEmpty κ := Fintype.card_eq_zero_iff.1 hk0
    simp
  · set k := Fintype.card κ with hk
    have hkR : (0 : ℝ) < k := by exact_mod_cast hkpos
    have hamgm := Real.geom_mean_le_arith_mean_weighted (s := Finset.univ)
      (fun _ => (1 : ℝ) / k) (fun i => hA.1.eigenvalues i) (fun _ _ => by positivity)
      (by rw [Finset.sum_const, Finset.card_univ, ← hk, nsmul_eq_mul]; field_simp)
      (fun i _ => hev0 i)
    rw [← Finset.mul_sum, htr] at hamgm
    have hmean : (1 : ℝ) / k * (k * c) = c := by field_simp
    rw [hmean] at hamgm
    have hprod : ∏ i, hA.1.eigenvalues i =
        (∏ i, hA.1.eigenvalues i ^ ((1 : ℝ) / k)) ^ k := by
      rw [← Finset.prod_pow]
      refine Finset.prod_congr rfl fun i _ => ?_
      rw [one_div, Real.rpow_inv_natCast_pow (hev0 i) (by omega)]
    rw [hprod]
    exact pow_le_pow_left₀ (Finset.prod_nonneg fun i _ => Real.rpow_nonneg (hev0 i) _) hamgm k

/-- Equality in the Hadamard-type bound: a positive semidefinite matrix with unit diagonal and
determinant `1` is the identity. -/
theorem eq_one_of_posSemidef_det_eq_one {κ : Type*} [Fintype κ] [DecidableEq κ]
    (A : Matrix κ κ ℂ) (hA : A.PosSemidef) (hdiag : ∀ i, A i i = 1) (hdet : A.det = 1) :
    A = 1 := by
  have hev0 : ∀ i, 0 ≤ hA.1.eigenvalues i := hA.eigenvalues_nonneg
  have hprod : ∏ i, hA.1.eigenvalues i = 1 := by
    have h : ‖A.det‖ = ∏ i, hA.1.eigenvalues i := by
      rw [hA.1.det_eq_prod_eigenvalues, norm_prod]
      refine Finset.prod_congr rfl fun i _ => ?_
      rw [RCLike.norm_ofReal, abs_of_nonneg (hev0 i)]
    rw [← h, hdet, norm_one]
  have htr : ∑ i, hA.1.eigenvalues i = Fintype.card κ := by
    have h1 := hA.1.trace_eq_sum_eigenvalues
    have h2 : A.trace = (Fintype.card κ : ℂ) := by
      simp [Matrix.trace, Matrix.diag, hdiag]
    have h3 := congrArg Complex.re (h1.symm.trans h2)
    simpa using h3
  rcases isEmpty_or_nonempty κ with hκ | hκ
  · ext i; exact isEmptyElim i
  set k := Fintype.card κ with hk
  have hkpos : 0 < k := Fintype.card_pos
  have hkR : (0 : ℝ) < k := by exact_mod_cast hkpos
  have hw : ∑ _i : κ, (1 : ℝ) / k = 1 := by
    rw [Finset.sum_const, Finset.card_univ, ← hk, nsmul_eq_mul]; field_simp
  have hgeo : ∏ i, hA.1.eigenvalues i ^ ((1 : ℝ) / k) = 1 := by
    rw [Real.finsetProd_rpow _ _ (fun i _ => hev0 i), hprod, Real.one_rpow]
  have hari : ∑ i, (1 : ℝ) / k * hA.1.eigenvalues i = 1 := by
    rw [← Finset.mul_sum, htr]; field_simp
  have heqAM := (Real.geom_mean_eq_arith_mean_weighted_iff_of_pos (s := Finset.univ)
    (fun _ => (1 : ℝ) / k) (fun i => hA.1.eigenvalues i) (fun _ _ => by positivity) hw
    (fun i _ => hev0 i)).1 (by rw [hgeo, hari])
  have hall : ∀ i, hA.1.eigenvalues i = 1 := by
    obtain ⟨i0⟩ := hκ
    have hc : ∀ i, hA.1.eigenvalues i = hA.1.eigenvalues i0 := fun i =>
      heqAM i (Finset.mem_univ _) i0 (Finset.mem_univ _)
    have hs : ∑ i, hA.1.eigenvalues i = k * hA.1.eigenvalues i0 := by
      simp_rw [hc]; simp [Finset.sum_const, hk]
    rw [hs] at htr
    intro i
    rw [hc i]
    have hk0 : (k : ℝ) ≠ 0 := hkR.ne'
    field_simp at htr
    linarith
  rw [hA.1.spectral_theorem]
  have hd : (diagonal (RCLike.ofReal ∘ hA.1.eigenvalues) : Matrix κ κ ℂ) = 1 := by
    ext i j
    simp [Matrix.diagonal, Matrix.one_apply, hall]
  rw [hd, map_one]

/-- **lem:hadamard.** For unit vectors `u_l`, the Gram determinant satisfies
`0 ≤ det Gram ≤ 1`, with `det Gram = 1` iff the vectors are orthonormal. -/
theorem lem_hadamard {κ m : Type*} [Fintype κ] [DecidableEq κ] [Fintype m]
    (u : κ → m → ℂ) (hu : ∀ l, ∑ i, ‖u l i‖ ^ 2 = 1) :
    let G : Matrix κ κ ℂ := Matrix.of fun l l' => ∑ i, star (u l i) * u l' i
    0 ≤ G.det ∧ ‖G.det‖ ≤ 1 ∧
      (G.det = 1 ↔ ∀ l l', ∑ i, star (u l i) * u l' i = if l = l' then 1 else 0) := by
  intro G
  let U : Matrix m κ ℂ := Matrix.of fun i l => u l i
  have hGU : G = Uᴴ * U := by
    ext l l'
    simp [G, U, Matrix.mul_apply, Matrix.conjTranspose_apply]
  have hpsd : G.PosSemidef := hGU ▸ Matrix.posSemidef_conjTranspose_mul_self U
  have hdiag : ∀ l, G l l = 1 := by
    intro l
    simp only [G, Matrix.of_apply]
    have : ∀ i, star (u l i) * u l i = ((‖u l i‖ ^ 2 : ℝ) : ℂ) := by
      intro i
      rw [mul_comm, Complex.star_def, Complex.mul_conj, Complex.normSq_eq_norm_sq]
    simp_rw [this]
    rw [← Complex.ofReal_sum, hu l, Complex.ofReal_one]
  refine ⟨hpsd.det_nonneg, ?_, ?_⟩
  · have := norm_det_le_of_posSemidef G hpsd 1 (fun l => by rw [hdiag]; simp)
    simpa using this
  · constructor
    · intro hdet l l'
      have h1 := eq_one_of_posSemidef_det_eq_one G hpsd hdiag hdet
      have := congrFun (congrFun h1 l) l'
      simp only [G, Matrix.of_apply] at this
      rw [this, Matrix.one_apply]
    · intro h
      have : G = 1 := by
        ext l l'
        simp only [G, Matrix.of_apply, h l l', Matrix.one_apply]
      rw [this, det_one]

/-! ## The generating inequality (prop:gen) -/

section Gen

variable {β : Fin N → ℂ} {ξ : Fin N → Fin n → ℂ}

lemma det_one_add_smul_Hmat {B : ℂ} (hξ : ∀ l t, ‖ξ l t‖ = 1) (hJF : JF β ξ B) (t : ℂ) :
    det (1 + t • Hmat β ξ) = (1 + t * B) ^ (n + 1) := by
  obtain ⟨-, hlow, hdiag⟩ := chain_lemma hξ hJF
  have hup : (1 + t • Hmat β ξ).IsUpperTriangular := by
    intro i j hji
    change j < i at hji
    simp only [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]
    rw [hlow i j hji, Matrix.one_apply_ne (ne_of_gt hji)]
    simp
  rw [Matrix.det_of_isUpperTriangular hup]
  simp [hdiag, Finset.prod_const]

/-- `det(1 + t H) = det(1 + t diag(β) G)` (Sylvester). -/
lemma det_one_add_smul_Hmat_eq_Gram (t : ℂ) :
    det (1 + t • Hmat β ξ) = det (1 + t • (diagonal β * Gram ξ)) := by
  have e1 : t • Hmat β ξ = Umat ξ * (t • (diagonal β * (Umat ξ)ᴴ)) := by
    rw [Hmat_eq, Matrix.mul_smul]
  rw [e1, Matrix.det_one_add_mul_comm (Umat ξ) (t • (diagonal β * (Umat ξ)ᴴ)),
    Matrix.smul_mul, Matrix.mul_assoc]
  rfl

/-- The expansion of `det(1 + t H)` over subsets of neurons. -/
lemma det_one_add_smul_Hmat_expand (t : ℂ) :
    det (1 + t • Hmat β ξ) = ∑ s : Finset (Fin N), t ^ s.card *
      ((∏ l ∈ s, β l) * det ((Gram ξ).submatrix (Subtype.val : s → Fin N) Subtype.val)) := by
  rw [det_one_add_smul_Hmat_eq_Gram, det_one_add_smul_expand]
  simp_rw [det_submatrix_diagonal_mul]

lemma Gram_posSemidef (ξ : Fin N → Fin n → ℂ) : (Gram ξ).PosSemidef :=
  Matrix.posSemidef_conjTranspose_mul_self _

/-- `|det G_{L,L}| ≤ ν^{|L|}`. -/
lemma norm_det_Gram_sub_le (hξ : ∀ l t, ‖ξ l t‖ = 1) (s : Finset (Fin N)) :
    ‖det ((Gram ξ).submatrix (Subtype.val : s → Fin N) Subtype.val)‖ ≤ (n + 1 : ℝ) ^ s.card := by
  have h := norm_det_le_of_posSemidef ((Gram ξ).submatrix (Subtype.val : s → Fin N)
    (Subtype.val : s → Fin N)) ((Gram_posSemidef ξ).submatrix _) (n + 1 : ℝ)
    (fun i => by simp [Gram_diag hξ])
  simpa using h

/-- The first half of the chain: `(1+t)^ν ≤ ∑_L t^{|L|} (∏_{l∈L} |β_l|) |det G_{L,L}|`. -/
lemma gen_terms (hξ : ∀ l t, ‖ξ l t‖ = 1) (hJF : JF β ξ 1) {t : ℝ} (ht : 0 ≤ t) :
    (1 + t) ^ (n + 1) ≤ ∑ s : Finset (Fin N), t ^ s.card * ((∏ l ∈ s, ‖β l‖) *
      ‖det ((Gram ξ).submatrix (Subtype.val : s → Fin N) Subtype.val)‖) := by
  have h1 := det_one_add_smul_Hmat hξ hJF (t : ℂ)
  rw [det_one_add_smul_Hmat_expand] at h1
  have h2 : (1 + t) ^ (n + 1) = ‖(1 + (t : ℂ) * 1) ^ (n + 1)‖ := by
    rw [mul_one, norm_pow]
    congr 1
    rw [show (1 + (t : ℂ)) = ((1 + t : ℝ) : ℂ) by push_cast; ring, Complex.norm_real,
      Real.norm_of_nonneg (by linarith)]
  rw [h2, ← h1]
  refine le_trans (norm_sum_le _ _) (le_of_eq (Finset.sum_congr rfl fun s _ => ?_))
  rw [norm_mul, norm_mul, norm_pow, Complex.norm_real, Real.norm_of_nonneg ht, norm_prod]

/-- `∑_L t^{|L|} (∏_{l∈L} |β_l|) ν^{|L|} = ∏_l (1 + t ν |β_l|)`. -/
lemma sum_expand_eq_prod (t : ℝ) :
    ∑ s : Finset (Fin N), t ^ s.card * ((∏ l ∈ s, ‖β l‖) * (n + 1 : ℝ) ^ s.card) =
      ∏ l, (1 + t * ((n + 1) * ‖β l‖)) := by
  rw [Finset.prod_one_add, Finset.powerset_univ]
  refine Finset.sum_congr rfl fun s _ => ?_
  rw [Finset.prod_mul_distrib, Finset.prod_mul_distrib, Finset.prod_const,
    Finset.prod_const]
  ring

/-- **prop:gen** (normalised to `B = 1`): for every `t ≥ 0`,
`(1+t)^ν ≤ ∏_l (1 + t ν |β_l|)`.  (The paper derives it from prop:esym; we bound the whole
expansion of `det(1 + tH)` at once, which gives the same inequality directly.) -/
theorem prop_gen (hξ : ∀ l t, ‖ξ l t‖ = 1) (hJF : JF β ξ 1) {t : ℝ} (ht : 0 ≤ t) :
    (1 + t) ^ (n + 1) ≤ ∏ l, (1 + t * ((n + 1) * ‖β l‖)) := by
  refine le_trans (gen_terms hξ hJF ht) (le_of_le_of_eq ?_ (sum_expand_eq_prod t))
  refine Finset.sum_le_sum fun s _ => ?_
  gcongr
  exact norm_det_Gram_sub_le hξ s

/-- The polynomial identity `∑_L t^{|L|} (∏_{l∈L} β_l) det G_{L,L} = (1+t)^ν` (`B = 1`). -/
lemma expand_eq_pow (hξ : ∀ l t, ‖ξ l t‖ = 1) (hJF : JF β ξ 1) (t : ℂ) :
    ∑ s : Finset (Fin N), t ^ s.card *
      ((∏ l ∈ s, β l) * det ((Gram ξ).submatrix (Subtype.val : s → Fin N) Subtype.val)) =
      (1 + t) ^ (n + 1) := by
  rw [← det_one_add_smul_Hmat_expand, det_one_add_smul_Hmat hξ hJF, mul_one]

/-- **prop:esym.** Under (JF) with `B = 1` and `x_l = ν |β_l|`, the elementary symmetric
polynomials satisfy `e_k(x) ≥ C(ν, k)` for every `k`. -/
theorem prop_esym (hξ : ∀ l t, ‖ξ l t‖ = 1) (hJF : JF β ξ 1) (k : ℕ) :
    ((n + 1).choose k : ℝ) ≤
      ∑ s ∈ (Finset.univ : Finset (Fin N)).powersetCard k, ∏ l ∈ s, ((n + 1 : ℝ) * ‖β l‖) := by
  classical
  set a : Finset (Fin N) → ℂ := fun s =>
    (∏ l ∈ s, β l) * det ((Gram ξ).submatrix (Subtype.val : s → Fin N) Subtype.val) with ha
  let P : Polynomial ℂ := ∑ s : Finset (Fin N), Polynomial.C (a s) * Polynomial.X ^ s.card
  have hP : P = (1 + Polynomial.X) ^ (n + 1) := by
    refine Polynomial.funext fun t => ?_
    have := expand_eq_pow hξ hJF t
    simp only [P, Polynomial.eval_finsetSum, Polynomial.eval_mul, Polynomial.eval_C,
      Polynomial.eval_pow, Polynomial.eval_X, Polynomial.eval_add, Polynomial.eval_one]
    rw [← this]
    exact Finset.sum_congr rfl fun s _ => by simp only [ha]; ring
  have hcoeff := congrArg (fun q : Polynomial ℂ => q.coeff k) hP
  simp only [P, Polynomial.finsetSum_coeff, Polynomial.coeff_C_mul_X_pow,
    Polynomial.coeff_one_add_X_pow] at hcoeff
  rw [← Finset.sum_filter] at hcoeff
  have hfilt : (Finset.univ.filter fun s : Finset (Fin N) => k = s.card) =
      (Finset.univ : Finset (Fin N)).powersetCard k := by
    ext s; simp [Finset.mem_powersetCard, eq_comm]
  rw [hfilt] at hcoeff
  calc ((n + 1).choose k : ℝ) = ‖(((n + 1).choose k : ℕ) : ℂ)‖ := by
        rw [Complex.norm_natCast]
    _ = ‖∑ s ∈ (Finset.univ : Finset (Fin N)).powersetCard k, a s‖ := by rw [hcoeff]
    _ ≤ ∑ s ∈ (Finset.univ : Finset (Fin N)).powersetCard k, ‖a s‖ := norm_sum_le _ _
    _ ≤ ∑ s ∈ (Finset.univ : Finset (Fin N)).powersetCard k,
          ∏ l ∈ s, ((n + 1 : ℝ) * ‖β l‖) := by
        refine Finset.sum_le_sum fun s _ => ?_
        simp only [ha, norm_mul, norm_prod]
        rw [Finset.prod_mul_distrib, Finset.prod_const, mul_comm]
        gcongr
        exact norm_det_Gram_sub_le hξ s

/-- Consequence of prop:esym at `k = ν`: at least `ν` of the `β_l` are non-zero
(for any `B ≠ 0`). -/
theorem card_support_ge (hξ : ∀ l t, ‖ξ l t‖ = 1) {B : ℂ} (hB : B ≠ 0) (hJF : JF β ξ B) :
    n + 1 ≤ (Finset.univ.filter fun l => β l ≠ 0).card := by
  classical
  have hJF' : JF (fun l => β l / B) ξ 1 := by
    intro T
    have := hJF T
    simp_rw [div_mul_eq_mul_div, ← Finset.sum_div, this]
    split_ifs <;> simp [hB]
  have h := prop_esym hξ hJF' (n + 1)
  rw [Nat.choose_self, Nat.cast_one] at h
  obtain ⟨s, hs, hne⟩ : ∃ s ∈ (Finset.univ : Finset (Fin N)).powersetCard (n + 1),
      ∏ l ∈ s, ((n + 1 : ℝ) * ‖β l / B‖) ≠ 0 := by
    by_contra hcon
    push Not at hcon
    rw [Finset.sum_eq_zero hcon] at h
    linarith
  rw [Finset.mem_powersetCard] at hs
  rw [← hs.2]
  refine Finset.card_le_card fun l hl => ?_
  simp only [Finset.mem_filter, Finset.mem_univ, true_and]
  intro hβ
  exact hne (Finset.prod_eq_zero hl (by simp [hβ]))

/-- `det` of the `2 × 2` principal submatrix on `{l, l'}`. -/
lemma det_submatrix_pair (G : Matrix (Fin N) (Fin N) ℂ) {l l' : Fin N} (hll : l ≠ l') :
    det (G.submatrix (Subtype.val : ({l, l'} : Finset (Fin N)) → Fin N) Subtype.val) =
      G l l * G l' l' - G l l' * G l' l := by
  let f : Fin 2 → ({l, l'} : Finset (Fin N)) := fun i =>
    ⟨![l, l'] i, by fin_cases i <;> simp⟩
  have hf : Function.Bijective f := by
    constructor
    · intro i j h
      simp only [f, Subtype.mk.injEq] at h
      fin_cases i <;> fin_cases j <;> simp_all [eq_comm]
    · rintro ⟨a, ha⟩
      simp only [Finset.mem_insert, Finset.mem_singleton] at ha
      rcases ha with rfl | rfl
      · exact ⟨0, rfl⟩
      · exact ⟨1, rfl⟩
  let e := Equiv.ofBijective f hf
  rw [← Matrix.det_submatrix_equiv_self e, Matrix.det_fin_two]
  simp [e, f]

end Gen

/-! ## The Bernstein-type integral (lem:bernstein, scaling form) -/

section Bernstein

/-- The integrand `log(1 + x t) t^{-1-r}`. -/
def bI (r x t : ℝ) : ℝ := Real.log (1 + x * t) * t ^ (-1 - r)

variable {r : ℝ}

lemma bI_nonneg {x t : ℝ} (hx : 0 ≤ x) (ht : 0 ≤ t) : 0 ≤ bI r x t :=
  mul_nonneg (Real.log_nonneg (by nlinarith)) (Real.rpow_nonneg ht _)

lemma continuousOn_bI {x : ℝ} (hx : 0 ≤ x) : ContinuousOn (bI r x) (Ioi 0) := by
  unfold bI
  refine ContinuousOn.mul ?_ ?_
  · refine ContinuousOn.log (by fun_prop) fun t ht => ?_
    have : (0 : ℝ) < t := ht
    nlinarith
  · exact ContinuousOn.rpow_const continuousOn_id fun t ht => Or.inl (ne_of_gt ht)

lemma bI_le_near_zero {x t : ℝ} (hx : 0 ≤ x) (ht : 0 < t) : bI r x t ≤ x * t ^ (-r) := by
  unfold bI
  have h1 : Real.log (1 + x * t) ≤ x * t := by
    have := Real.log_le_sub_one_of_pos (by nlinarith : (0 : ℝ) < 1 + x * t)
    linarith
  have h2 : t * t ^ (-1 - r) = t ^ (-r) := by
    have h := Real.rpow_add ht 1 (-1 - r)
    rw [Real.rpow_one] at h
    rw [← h]; congr 1; ring
  calc Real.log (1 + x * t) * t ^ (-1 - r) ≤ x * t * t ^ (-1 - r) :=
        mul_le_mul_of_nonneg_right h1 (Real.rpow_nonneg ht.le _)
    _ = x * t ^ (-r) := by rw [mul_assoc, h2]

lemma bI_le_near_top (hr0 : 0 < r) {x t : ℝ} (hx : 0 ≤ x) (ht : 1 ≤ t) :
    bI r x t ≤ (2 / r) * (1 + x) ^ (r / 2) * t ^ (-1 - r / 2) := by
  unfold bI
  have ht0 : 0 < t := by linarith
  have h1 : Real.log (1 + x * t) ≤ (1 + x * t) ^ (r / 2) / (r / 2) :=
    Real.log_le_rpow_div (by nlinarith) (by linarith)
  have h2 : (1 + x * t) ^ (r / 2) ≤ ((1 + x) * t) ^ (r / 2) :=
    Real.rpow_le_rpow (by nlinarith) (by nlinarith) (by linarith)
  have h3 : ((1 + x) * t) ^ (r / 2) = (1 + x) ^ (r / 2) * t ^ (r / 2) :=
    Real.mul_rpow (by linarith) ht0.le
  have h4 : t ^ (r / 2) * t ^ (-1 - r) = t ^ (-1 - r / 2) := by
    rw [← Real.rpow_add ht0]; congr 1; ring
  have hpos : 0 ≤ t ^ (-1 - r) := Real.rpow_nonneg ht0.le _
  calc Real.log (1 + x * t) * t ^ (-1 - r)
      ≤ (1 + x * t) ^ (r / 2) / (r / 2) * t ^ (-1 - r) := mul_le_mul_of_nonneg_right h1 hpos
    _ ≤ ((1 + x) * t) ^ (r / 2) / (r / 2) * t ^ (-1 - r) := by
        gcongr
    _ = (2 / r) * (1 + x) ^ (r / 2) * (t ^ (r / 2) * t ^ (-1 - r)) := by
        rw [h3]; field_simp
    _ = (2 / r) * (1 + x) ^ (r / 2) * t ^ (-1 - r / 2) := by rw [h4]

lemma integrableOn_bI_Ioc (hr1 : r < 1) {x : ℝ} (hx : 0 ≤ x) :
    IntegrableOn (bI r x) (Ioc 0 1) := by
  refine IntegrableOn.congr_set_ae (t := Ioo 0 1) ?_ (Filter.EventuallyEq.symm Ioo_ae_eq_Ioc)
  refine ⟨ContinuousOn.aestronglyMeasurable ((continuousOn_bI hx).mono Ioo_subset_Ioi_self)
    measurableSet_Ioo, ?_⟩
  refine HasFiniteIntegral.mono' (g := fun t => x * t ^ (-r)) ?_ ?_
  · apply Integrable.hasFiniteIntegral
    refine Integrable.const_mul ?_ _
    rw [← IntegrableOn, intervalIntegral.integrableOn_Ioo_rpow_iff zero_lt_one]
    linarith
  · refine ae_restrict_of_forall_mem measurableSet_Ioo fun t ht => ?_
    rw [Real.norm_of_nonneg (bI_nonneg hx ht.1.le)]
    exact bI_le_near_zero hx ht.1

lemma integrableOn_bI_Ioi_one (hr0 : 0 < r) {x : ℝ} (hx : 0 ≤ x) :
    IntegrableOn (bI r x) (Ioi 1) := by
  refine ⟨ContinuousOn.aestronglyMeasurable ((continuousOn_bI hx).mono
    (Set.Ioi_subset_Ioi zero_le_one)) measurableSet_Ioi, ?_⟩
  refine HasFiniteIntegral.mono'
    (g := fun t => (2 / r) * (1 + x) ^ (r / 2) * t ^ (-1 - r / 2)) ?_ ?_
  · apply Integrable.hasFiniteIntegral
    refine Integrable.const_mul ?_ _
    exact integrableOn_Ioi_rpow_of_lt (by linarith) zero_lt_one
  · refine ae_restrict_of_forall_mem measurableSet_Ioi fun t (ht : 1 < t) => ?_
    rw [Real.norm_of_nonneg (bI_nonneg hx (by linarith))]
    exact bI_le_near_top hr0 hx ht.le

lemma integrableOn_bI (hr0 : 0 < r) (hr1 : r < 1) {x : ℝ} (hx : 0 ≤ x) :
    IntegrableOn (bI r x) (Ioi 0) := by
  rw [← Set.Ioc_union_Ioi_eq_Ioi zero_le_one]
  exact (integrableOn_bI_Ioc hr1 hx).union (integrableOn_bI_Ioi_one hr0 hx)

/-- The constant `K_r = ∫_0^∞ log(1+u) u^{-1-r} du`. -/
def Kr (r : ℝ) : ℝ := ∫ t in Ioi 0, bI r 1 t

lemma Kr_pos (hr0 : 0 < r) (hr1 : r < 1) : 0 < Kr r := by
  have hsub : ∫ t in Ioc 1 2, bI r 1 t ≤ Kr r := by
    refine setIntegral_mono_set (integrableOn_bI hr0 hr1 zero_le_one) ?_ ?_
    · exact ae_restrict_of_forall_mem measurableSet_Ioi fun t ht => bI_nonneg zero_le_one
        (le_of_lt ht)
    · exact Filter.Eventually.of_forall fun t ht => lt_of_lt_of_le zero_lt_one (le_of_lt ht.1)
  have hpos : 0 < ∫ t in (1 : ℝ)..2, bI r 1 t := by
    refine intervalIntegral.intervalIntegral_pos_of_pos_on ?_ ?_ one_lt_two
    · exact ((continuousOn_bI zero_le_one).mono fun t ht => by
        simp only [Set.mem_Icc] at ht
        exact lt_of_lt_of_le zero_lt_one ht.1).intervalIntegrable_of_Icc one_le_two
    · intro t ht
      unfold bI
      have ht0 : 0 < t := by linarith [ht.1]
      exact mul_pos (Real.log_pos (by linarith)) (Real.rpow_pos_of_pos ht0 _)
  rw [intervalIntegral.integral_of_le one_le_two] at hpos
  linarith

/-- **lem:bernstein (scaling form).** For `0 < r < 1` and `x ≥ 0`,
`∫_0^∞ log(1+xt) t^{-1-r} dt = x^r K_r`, with `0 < K_r < ∞` (`Kr_pos`, `integrableOn_bI`).
(The paper's closed form `K_r = π / (r sin(π r))` is not needed: the constant cancels.) -/
theorem bernstein_scaling (hr0 : 0 < r) {x : ℝ} (hx : 0 ≤ x) :
    ∫ t in Ioi 0, bI r x t = x ^ r * Kr r := by
  rcases hx.eq_or_lt with rfl | hx
  · simp [bI, Real.zero_rpow hr0.ne']
  have hcv := MeasureTheory.integral_comp_mul_left_Ioi (bI r 1) 0 hx
  rw [mul_zero] at hcv
  have heq : ∀ t ∈ Ioi (0 : ℝ), bI r 1 (x * t) = x ^ (-1 - r) * bI r x t := by
    intro t ht
    unfold bI
    rw [one_mul, Real.mul_rpow hx.le (le_of_lt ht)]
    ring
  rw [setIntegral_congr_fun measurableSet_Ioi heq, integral_const_mul, smul_eq_mul] at hcv
  unfold Kr
  have hxr : x ^ (-1 - r) ≠ 0 := (Real.rpow_pos_of_pos hx _).ne'
  apply mul_left_cancel₀ hxr
  rw [hcv, ← mul_assoc, ← Real.rpow_add hx]
  congr 1
  rw [show -1 - r + r = -1 by ring, Real.rpow_neg_one]

/-- From the generating inequality to the power sums: if `(1+t)^k ≤ ∏_l (1 + x_l t)` for all
`t > 0`, then `∑_l x_l^r ≥ k` for `0 < r < 1` (integrate `log` against `t^{-1-r} dt`). -/
theorem sum_rpow_ge_of_prod {ι : Type*} [Fintype ι] (x : ι → ℝ) (hx : ∀ l, 0 ≤ x l) (k : ℕ)
    (h : ∀ t : ℝ, 0 < t → (1 + t) ^ k ≤ ∏ l, (1 + t * x l)) (hr0 : 0 < r) (hr1 : r < 1) :
    (k : ℝ) ≤ ∑ l, x l ^ r := by
  have hK := Kr_pos hr0 hr1
  have hsum : ∑ l, x l ^ r * Kr r = ∫ t in Ioi 0, ∑ l, bI r (x l) t := by
    rw [integral_finsetSum _ fun l _ => integrableOn_bI hr0 hr1 (hx l)]
    exact Finset.sum_congr rfl fun l _ => (bernstein_scaling hr0 (hx l)).symm
  have hle : ∫ t in Ioi 0, (k : ℝ) * bI r 1 t ≤ ∫ t in Ioi 0, ∑ l, bI r (x l) t := by
    refine setIntegral_mono_on ((integrableOn_bI hr0 hr1 zero_le_one).const_mul _)
      (integrable_finsetSum _ fun l _ => integrableOn_bI hr0 hr1 (hx l)) measurableSet_Ioi ?_
    intro t ht
    have ht0 : (0 : ℝ) < t := ht
    unfold bI
    rw [← Finset.sum_mul, one_mul, ← mul_assoc]
    refine mul_le_mul_of_nonneg_right ?_ (Real.rpow_nonneg ht0.le _)
    have hpos : ∀ l ∈ (Finset.univ : Finset ι), 1 + x l * t ≠ 0 := fun l _ => by
      have := hx l; nlinarith
    rw [← Real.log_prod hpos, ← Real.log_pow]
    refine Real.log_le_log (by positivity) ?_
    have := h t ht0
    simpa [mul_comm] using this
  rw [integral_const_mul] at hle
  have : (k : ℝ) * Kr r ≤ (∑ l, x l ^ r) * Kr r := by
    rw [Finset.sum_mul, hsum]; exact hle
  exact le_of_mul_le_mul_right this hK

end Bernstein

/-! ## Lemma `lem:cost` -/

section Cost

variable {β : Fin N → ℂ} {ξ : Fin N → Fin n → ℂ}

lemma JF_normalize {B : ℂ} (hB : B ≠ 0) (hJF : JF β ξ B) : JF (fun l => β l / B) ξ 1 := by
  intro T
  have := hJF T
  simp_rw [div_mul_eq_mul_div, ← Finset.sum_div, this]
  split_ifs <;> simp [hB]

/-- **Lemma `lem:cost`.** Under (JF), `∑_l |β_l|^r ≥ ν^{1-r} |B|^r` for every `r ∈ (0,1]`. -/
theorem lem_cost (β : Fin N → ℂ) (ξ : Fin N → Fin n → ℂ) (hξ : ∀ l t, ‖ξ l t‖ = 1) (B : ℂ)
    (hJF : JF β ξ B) {r : ℝ} (hr0 : 0 < r) (hr1 : r ≤ 1) :
    ((n : ℝ) + 1) ^ (1 - r) * ‖B‖ ^ r ≤ ∑ l, ‖β l‖ ^ r := by
  rcases hr1.lt_or_eq with hr1 | rfl
  · rcases eq_or_ne B 0 with rfl | hB
    · rw [norm_zero, Real.zero_rpow hr0.ne', mul_zero]
      exact Finset.sum_nonneg fun l _ => Real.rpow_nonneg (norm_nonneg _) _
    have hBpos : 0 < ‖B‖ := norm_pos_iff.2 hB
    have hν : (0 : ℝ) < n + 1 := by positivity
    set x : Fin N → ℝ := fun l => (n + 1) * ‖β l / B‖ with hxdef
    have hx : ∀ l, 0 ≤ x l := fun l => by positivity
    have hgen := fun t (ht : (0 : ℝ) < t) =>
      prop_gen hξ (JF_normalize hB hJF) (t := t) ht.le
    have hsum := sum_rpow_ge_of_prod x hx (n + 1) hgen hr0 hr1
    -- unwind `x_l = ν |β_l| / |B|`
    have hxl : ∀ l, x l ^ r = ((n + 1 : ℝ) / ‖B‖) ^ r * ‖β l‖ ^ r := by
      intro l
      simp only [hxdef, norm_div]
      rw [← Real.mul_rpow (by positivity) (norm_nonneg _)]
      congr 1; field_simp
    simp_rw [hxl] at hsum
    rw [← Finset.mul_sum] at hsum
    have hc : 0 < ((n + 1 : ℝ) / ‖B‖) ^ r := Real.rpow_pos_of_pos (by positivity) _
    have hkey : ((n : ℝ) + 1) ^ (1 - r) * ‖B‖ ^ r * ((n + 1 : ℝ) / ‖B‖) ^ r = (n + 1 : ℝ) := by
      rw [Real.div_rpow hν.le hBpos.le, Real.rpow_sub hν, Real.rpow_one]
      have h1 : ‖B‖ ^ r ≠ 0 := (Real.rpow_pos_of_pos hBpos r).ne'
      have h2 : ((n : ℝ) + 1) ^ r ≠ 0 := (Real.rpow_pos_of_pos hν r).ne'
      field_simp
    have : ((n : ℝ) + 1) ^ (1 - r) * ‖B‖ ^ r * ((n + 1 : ℝ) / ‖B‖) ^ r ≤
        (∑ l, ‖β l‖ ^ r) * ((n + 1 : ℝ) / ‖B‖) ^ r := by
      rw [hkey, mul_comm]; exact_mod_cast hsum
    exact le_of_mul_le_mul_right this hc
  · -- `r = 1`: the triangle inequality
    simp only [sub_self, Real.rpow_zero, one_mul, Real.rpow_one]
    have := hJF ∅
    simp at this
    rw [← this]
    exact norm_sum_le _ _

end Cost

/-! ## The equality case of `lem:cost` -/

section Equality

/-- The integrand of the equality argument, written through the logarithm of the product. -/
lemma sum_bI_sub_eq {ι : Type*} [Fintype ι] (x : ι → ℝ) (hx : ∀ l, 0 ≤ x l) (k : ℕ) (r : ℝ)
    {t : ℝ} (ht : 0 < t) :
    ∑ l, bI r (x l) t - k * bI r 1 t =
      (Real.log (∏ l, (1 + t * x l)) - Real.log ((1 + t) ^ k)) * t ^ (-1 - r) := by
  unfold bI
  have hpos : ∀ l ∈ (Finset.univ : Finset ι), 1 + t * x l ≠ 0 := fun l _ => by
    have := hx l; nlinarith
  rw [← Finset.sum_mul, Real.log_prod hpos, Real.log_pow, one_mul]
  simp_rw [mul_comm (x _) t]
  ring

/-- Equality in `sum_rpow_ge_of_prod` forces `∏_l (1 + t x_l) = (1+t)^k` for all `t > 0`
(a continuous non-negative integrand with zero integral against `t^{-1-r} dt` vanishes). -/
theorem prod_eq_of_sum_rpow_eq {ι : Type*} [Fintype ι] (x : ι → ℝ) (hx : ∀ l, 0 ≤ x l)
    (k : ℕ) (h : ∀ t : ℝ, 0 < t → (1 + t) ^ k ≤ ∏ l, (1 + t * x l)) {r : ℝ} (hr0 : 0 < r)
    (hr1 : r < 1) (heq : ∑ l, x l ^ r = k) :
    ∀ t : ℝ, 0 < t → ∏ l, (1 + t * x l) = (1 + t) ^ k := by
  set F : ℝ → ℝ := fun t => ∑ l, bI r (x l) t - k * bI r 1 t with hF
  have hI1 : IntegrableOn (fun t => ∑ l, bI r (x l) t) (Ioi 0) :=
    integrable_finsetSum _ fun l _ => integrableOn_bI hr0 hr1 (hx l)
  have hI2 : IntegrableOn (fun t => (k : ℝ) * bI r 1 t) (Ioi 0) :=
    (integrableOn_bI hr0 hr1 zero_le_one).const_mul _
  have hFi : IntegrableOn F (Ioi 0) := hI1.sub hI2
  have hFnn : ∀ t ∈ Ioi (0 : ℝ), 0 ≤ F t := by
    intro t ht
    have ht0 : (0 : ℝ) < t := ht
    simp only [hF]
    rw [sum_bI_sub_eq x hx k r ht0]
    refine mul_nonneg ?_ (Real.rpow_nonneg ht0.le _)
    rw [sub_nonneg]
    exact Real.log_le_log (by positivity) (h t ht0)
  have hFint : ∫ t in Ioi 0, F t = 0 := by
    simp only [hF]
    rw [integral_sub hI1 hI2, integral_finsetSum _ fun l _ => integrableOn_bI hr0 hr1 (hx l),
      integral_const_mul]
    simp_rw [bernstein_scaling hr0 (hx _)]
    rw [← Finset.sum_mul, heq]
    unfold Kr
    ring
  have hae : F =ᵐ[volume.restrict (Ioi 0)] 0 :=
    (setIntegral_eq_zero_iff_of_nonneg_ae (ae_restrict_of_forall_mem measurableSet_Ioi hFnn)
      hFi).1 hFint
  have hcont : ContinuousOn F (Ioi 0) :=
    (continuousOn_finsetSum _ fun l _ => continuousOn_bI (hx l)).sub
      (continuousOn_const.mul (continuousOn_bI zero_le_one))
  have hzero := Measure.eqOn_open_of_ae_eq hae isOpen_Ioi hcont continuousOn_const
  intro t ht
  have hFt : F t = 0 := hzero ht
  simp only [hF] at hFt
  rw [sum_bI_sub_eq x hx k r ht] at hFt
  have hr : t ^ (-1 - r) ≠ 0 := (Real.rpow_pos_of_pos ht _).ne'
  have hlog : Real.log (∏ l, (1 + t * x l)) = Real.log ((1 + t) ^ k) := by
    have := (mul_eq_zero.1 hFt).resolve_right hr
    linarith
  have hp1 : 0 < ∏ l, (1 + t * x l) := Finset.prod_pos fun l _ => by
    have := hx l; nlinarith
  have hp2 : (0 : ℝ) < (1 + t) ^ k := by positivity
  exact Real.log_injOn_pos (Set.mem_Ioi.2 hp1) (Set.mem_Ioi.2 hp2) hlog

/-- If `∏_l (1 + t x_l) = (1+t)^k` for all `t > 0`, then every non-zero `x_l` equals `1` and
exactly `k` of them are non-zero (unique factorisation). -/
theorem eq_one_of_prod_eq {ι : Type*} [Fintype ι] [DecidableEq ι] (x : ι → ℝ) (k : ℕ)
    (hprod : ∀ t : ℝ, 0 < t → ∏ l, (1 + t * x l) = (1 + t) ^ k) :
    (∀ l, x l ≠ 0 → x l = 1) ∧ (Finset.univ.filter fun l => x l ≠ 0).card = k := by
  classical
  let P : Polynomial ℝ := ∏ l, (1 + Polynomial.C (x l) * Polynomial.X)
  let Q : Polynomial ℝ := (1 + Polynomial.X) ^ k
  have hevalP : ∀ t, P.eval t = ∏ l, (1 + t * x l) := by
    intro t
    simp only [P, Polynomial.eval_prod, Polynomial.eval_add, Polynomial.eval_one,
      Polynomial.eval_mul, Polynomial.eval_C, Polynomial.eval_X]
    exact Finset.prod_congr rfl fun l _ => by ring
  have hevalQ : ∀ t, Q.eval t = (1 + t) ^ k := by
    intro t; simp [Q]
  have hPQ : P = Q := by
    refine Polynomial.eq_of_infinite_eval_eq P Q (Set.Infinite.mono ?_ (Set.Ioi_infinite 0))
    intro t ht
    simp only [Set.mem_ofPred_eq, hevalP, hevalQ]
    exact hprod t ht
  have hone : ∀ l, x l ≠ 0 → x l = 1 := by
    intro l hl
    have h0 : P.eval (-1 / x l) = 0 := by
      rw [hevalP]
      refine Finset.prod_eq_zero (Finset.mem_univ l) ?_
      field_simp
      ring
    rw [hPQ, hevalQ] at h0
    have h1 : 1 + -1 / x l = 0 := by
      by_contra hne; exact pow_ne_zero _ hne h0
    field_simp at h1
    linarith
  refine ⟨hone, ?_⟩
  have h2 := hprod 1 one_pos
  have hsplit : ∏ l, (1 + 1 * x l) =
      ∏ l, (if x l ≠ 0 then (2 : ℝ) else 1) := by
    refine Finset.prod_congr rfl fun l _ => ?_
    by_cases hl : x l = 0
    · simp [hl]
    · rw [ite_eq_left_iff.2 (fun h => absurd hl h), hone l hl]; norm_num
  rw [hsplit, Finset.prod_ite, Finset.prod_const_one, mul_one, Finset.prod_const] at h2
  norm_num at h2
  exact h2

/-- The Gram matrix is Hermitian: `G_{l' l} = conj G_{l l'}`. -/
lemma Gram_symm (ξ : Fin N → Fin n → ℂ) (l l' : Fin N) :
    Gram ξ l' l = star (Gram ξ l l') := by
  simp only [Gram, Matrix.mul_apply, Matrix.conjTranspose_apply, star_sum, star_mul,
    star_star]

/-- A pair `{l, l'}` whose principal Gram determinant is maximal, `|det| = ν²`, has
orthogonal chain vectors. -/
lemma Gram_offdiag_eq_zero {ξ : Fin N → Fin n → ℂ} (hξ : ∀ l t, ‖ξ l t‖ = 1) {l l' : Fin N}
    (hll : l ≠ l')
    (hdet : ‖det ((Gram ξ).submatrix (Subtype.val : ({l, l'} : Finset (Fin N)) → Fin N)
      Subtype.val)‖ = (n + 1 : ℝ) ^ 2) :
    Gram ξ l l' = 0 := by
  have hnn := ((Gram_posSemidef ξ).submatrix
    (Subtype.val : ({l, l'} : Finset (Fin N)) → Fin N)).det_nonneg
  rw [det_submatrix_pair _ hll, Gram_diag hξ, Gram_diag hξ, Gram_symm ξ l l'] at hdet hnn
  set a := Gram ξ l l'
  have ha : a * star a = ((‖a‖ ^ 2 : ℝ) : ℂ) := by
    rw [Complex.star_def, Complex.mul_conj, Complex.normSq_eq_norm_sq]
  have hval : (n + 1 : ℂ) * (n + 1) - a * star a = (((n + 1 : ℝ) ^ 2 - ‖a‖ ^ 2 : ℝ) : ℂ) := by
    rw [ha]; push_cast; ring
  rw [hval] at hdet hnn
  rw [Complex.norm_real, Real.norm_eq_abs] at hdet
  have hnn' : 0 ≤ (n + 1 : ℝ) ^ 2 - ‖a‖ ^ 2 := Complex.zero_le_real.1 hnn
  rw [abs_of_nonneg hnn'] at hdet
  have : ‖a‖ ^ 2 = 0 := by linarith
  exact norm_eq_zero.1 (pow_eq_zero_iff (by norm_num) |>.1 this)

variable {β : Fin N → ℂ} {ξ : Fin N → Fin n → ℂ}

/-- The equality case in the generating inequality: if moreover `∏_l(1 + ν|β_l|) = 2^ν`
(the value at `t = 1`), then every pair of neurons with non-zero coefficient has orthogonal
chain vectors. -/
theorem orthogonal_of_gen_eq (hξ : ∀ l t, ‖ξ l t‖ = 1) (hJF : JF β ξ 1)
    (heq1 : ∏ l, (1 + 1 * ((n + 1) * ‖β l‖)) = (1 + 1) ^ (n + 1)) {l l' : Fin N}
    (hl : β l ≠ 0) (hl' : β l' ≠ 0) (hll : l ≠ l') :
    ∑ i, star (chain ξ l i) * chain ξ l' i = 0 := by
  classical
  have h1 := gen_terms hξ hJF (t := 1) zero_le_one
  rw [← heq1, ← sum_expand_eq_prod] at h1
  -- termwise slack
  set a : Finset (Fin N) → ℝ := fun s => 1 ^ s.card * ((∏ l ∈ s, ‖β l‖) *
      ‖det ((Gram ξ).submatrix (Subtype.val : s → Fin N) Subtype.val)‖) with ha
  set b : Finset (Fin N) → ℝ := fun s => 1 ^ s.card * ((∏ l ∈ s, ‖β l‖) *
      (n + 1 : ℝ) ^ s.card) with hb
  have hab : ∀ s ∈ (Finset.univ : Finset (Finset (Fin N))), a s ≤ b s := by
    intro s _
    simp only [ha, hb]
    gcongr
    exact norm_det_Gram_sub_le hξ s
  have hzero : ∀ s ∈ (Finset.univ : Finset (Finset (Fin N))), b s - a s = 0 := by
    have hsum : ∑ s, (b s - a s) = 0 := by
      rw [Finset.sum_sub_distrib]
      have : ∑ s, a s ≤ ∑ s, b s := Finset.sum_le_sum hab
      linarith
    exact (Finset.sum_eq_zero_iff_of_nonneg fun s hs => sub_nonneg.2 (hab s hs)).1 hsum
  have hs := hzero {l, l'} (Finset.mem_univ _)
  simp only [ha, hb, one_pow, one_mul] at hs
  have hprodpos : 0 < ∏ k ∈ ({l, l'} : Finset (Fin N)), ‖β k‖ :=
    Finset.prod_pos fun k hk => by
      simp only [Finset.mem_insert, Finset.mem_singleton] at hk
      rcases hk with rfl | rfl
      · exact norm_pos_iff.2 hl
      · exact norm_pos_iff.2 hl'
  have hcard : ({l, l'} : Finset (Fin N)).card = 2 := Finset.card_pair hll
  rw [hcard, ← mul_sub] at hs
  have hdet := (mul_eq_zero.1 hs).resolve_left hprodpos.ne'
  have hG := Gram_offdiag_eq_zero hξ hll (by linarith)
  simpa [Gram, Umat, Matrix.mul_apply, Matrix.conjTranspose_apply] using hG

/-- **Lemma `lem:cost`, equality case** (`0 < r < 1`, `B ≠ 0`): equality forces exactly
`ν` non-zero coefficients, all of modulus `|B|/ν`, with pairwise orthogonal chain vectors. -/
theorem lem_cost_eq (β : Fin N → ℂ) (ξ : Fin N → Fin n → ℂ) (hξ : ∀ l t, ‖ξ l t‖ = 1) (B : ℂ)
    (hJF : JF β ξ B) {r : ℝ} (hr0 : 0 < r) (hr1 : r < 1) (hB : B ≠ 0)
    (heq : ∑ l, ‖β l‖ ^ r = ((n : ℝ) + 1) ^ (1 - r) * ‖B‖ ^ r) :
    (Finset.univ.filter fun l => β l ≠ 0).card = n + 1 ∧
      (∀ l, β l ≠ 0 → ‖β l‖ = ‖B‖ / (n + 1)) ∧
      (∀ l l', β l ≠ 0 → β l' ≠ 0 → l ≠ l' → ∑ i, star (chain ξ l i) * chain ξ l' i = 0) := by
  classical
  have hBpos : 0 < ‖B‖ := norm_pos_iff.2 hB
  have hν : (0 : ℝ) < n + 1 := by positivity
  set β' : Fin N → ℂ := fun l => β l / B with hβ'
  have hJF' : JF β' ξ 1 := JF_normalize hB hJF
  set x : Fin N → ℝ := fun l => (n + 1) * ‖β' l‖ with hxdef
  have hx : ∀ l, 0 ≤ x l := fun l => by positivity
  have hxl : ∀ l, x l ^ r = ((n + 1 : ℝ) / ‖B‖) ^ r * ‖β l‖ ^ r := by
    intro l
    simp only [hxdef, hβ', norm_div]
    rw [← Real.mul_rpow (by positivity) (norm_nonneg _)]
    congr 1; field_simp
  have hsumx : ∑ l, x l ^ r = ((n + 1 : ℕ) : ℝ) := by
    simp_rw [hxl]
    rw [← Finset.mul_sum, heq, Real.div_rpow hν.le hBpos.le, Real.rpow_sub hν, Real.rpow_one]
    have h1 : ‖B‖ ^ r ≠ 0 := (Real.rpow_pos_of_pos hBpos r).ne'
    have h2 : ((n : ℝ) + 1) ^ r ≠ 0 := (Real.rpow_pos_of_pos hν r).ne'
    push_cast
    field_simp
  have hgen : ∀ t : ℝ, 0 < t → (1 + t) ^ (n + 1) ≤ ∏ l, (1 + t * x l) :=
    fun t ht => prop_gen hξ hJF' ht.le
  have hprod := prod_eq_of_sum_rpow_eq x hx (n + 1) hgen hr0 hr1 hsumx
  obtain ⟨hone, hcard⟩ := eq_one_of_prod_eq x (n + 1) hprod
  have hxne : ∀ l, x l ≠ 0 ↔ β l ≠ 0 := by
    intro l
    simp only [hxdef, hβ', ne_eq, mul_eq_zero, norm_eq_zero, div_eq_zero_iff, not_or]
    constructor
    · exact fun h => h.2.1
    · exact fun h => ⟨by positivity, h, hB⟩
  refine ⟨?_, ?_, ?_⟩
  · rw [← hcard]
    congr 1
    ext l
    simp [hxne l]
  · intro l hl
    have h1 := hone l ((hxne l).2 hl)
    simp only [hxdef, hβ', norm_div] at h1
    field_simp at h1 ⊢
    linarith
  · intro l l' hl hl' hll
    have hβ'l : β' l ≠ 0 := div_ne_zero hl hB
    have hβ'l' : β' l' ≠ 0 := div_ne_zero hl' hB
    exact orthogonal_of_gen_eq hξ hJF' (hprod 1 one_pos) hβ'l hβ'l' hll

/-- Converse of the equality case: `ν` non-zero coefficients of modulus `|B|/ν` give
equality `∑_l |β_l|^r = ν^{1-r}|B|^r`. -/
theorem lem_cost_eq_converse (β : Fin N → ℂ) (B : ℂ) {r : ℝ} (hr0 : 0 < r)
    (hcard : (Finset.univ.filter fun l => β l ≠ 0).card = n + 1)
    (hmod : ∀ l, β l ≠ 0 → ‖β l‖ = ‖B‖ / (n + 1)) :
    ∑ l, ‖β l‖ ^ r = ((n : ℝ) + 1) ^ (1 - r) * ‖B‖ ^ r := by
  classical
  have hν : (0 : ℝ) < n + 1 := by positivity
  rw [← Finset.sum_filter_add_sum_filter_not Finset.univ (fun l => β l ≠ 0)]
  have h0 : ∑ l ∈ Finset.univ.filter (fun l => ¬ β l ≠ 0), ‖β l‖ ^ r = 0 := by
    refine Finset.sum_eq_zero fun l hl => ?_
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, not_not] at hl
    rw [hl, norm_zero, Real.zero_rpow hr0.ne']
  rw [h0, add_zero, Finset.sum_congr rfl fun l hl => by
    rw [hmod l (Finset.mem_filter.1 hl).2], Finset.sum_const, hcard, nsmul_eq_mul,
    Real.div_rpow (norm_nonneg _) hν.le, Real.rpow_sub hν, Real.rpow_one]
  push_cast
  have h2 : ((n : ℝ) + 1) ^ r ≠ 0 := (Real.rpow_pos_of_pos hν r).ne'
  field_simp

end Equality

end Cost
end Heracles
