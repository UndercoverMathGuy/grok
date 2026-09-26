import Mathlib

/-!
# Common definitions: the budget-splitting programme (paper eq. (programme))

Paper: "Explaining Feature Sparsity Through the Margin–Regularization Trade-off",
Section `sec:programme`.

We first work with an *abstract* programme: a finite type `ι` of frequencies, a finite
non-empty type `Δ` of offsets and a coverage matrix `c : Δ → ι → ℝ`.  Then we instantiate
it with the concrete data of the paper:

* frequencies `Freq p = {1, …, (p-1)/2}` (natural numbers),
* offsets `Off p = {δ : ZMod p // δ ≠ 0}`,
* coverage `cov p δ j = 1 - cos(2π j δ / p)`.

`Γ_s(p)` is defined as a supremum (`sSup`) of the objective over the feasible set; we prove
(`exists_maximizer`) that it is attained, so it is the maximum of the paper.
-/

open Real Finset

noncomputable section

set_option linter.unusedSectionVars false

namespace Heracles

section Abstract

variable {ι Δ : Type*} [Fintype ι] [Fintype Δ] [Nonempty Δ]

/-- Coverage of the offset `δ` by the allocation `y`: `F_y(δ) = ∑_j y_j c_j(δ)`. -/
def cover (c : Δ → ι → ℝ) (y : ι → ℝ) (δ : Δ) : ℝ := ∑ j, y j * c δ j

/-- Worst coverage `m(y) = min_δ F_y(δ)`. -/
def worst (c : Δ → ι → ℝ) (y : ι → ℝ) : ℝ :=
  Finset.univ.inf' Finset.univ_nonempty (cover c y)

/-- The budget `∑_j y_j^{1/s}` (real power; `0^{1/s} = 0`). -/
def budget (s : ℝ) (y : ι → ℝ) : ℝ := ∑ j, y j ^ (1 / s)

/-- Feasibility for the programme: `y ≥ 0` and `∑_j y_j^{1/s} ≤ 1`. -/
def Feasible (s : ℝ) (y : ι → ℝ) : Prop := (∀ j, 0 ≤ y j) ∧ budget s y ≤ 1

/-- `Γ_s := sup { m(y) : y feasible }`.  It is attained (`exists_maximizer`). -/
def Gam (c : Δ → ι → ℝ) (s : ℝ) : ℝ := sSup (worst c '' {y | Feasible s y})

/-- `y` is a maximizer of the programme. -/
def IsMaximizer (c : Δ → ι → ℝ) (s : ℝ) (y : ι → ℝ) : Prop :=
  Feasible s y ∧ ∀ y', Feasible s y' → worst c y' ≤ worst c y

/-- The support `spt(y) = {j : y_j ≠ 0}`. -/
def supp (y : ι → ℝ) : Finset ι := Finset.univ.filter (fun j => y j ≠ 0)

lemma worst_le_cover (c : Δ → ι → ℝ) (y : ι → ℝ) (δ : Δ) : worst c y ≤ cover c y δ :=
  Finset.inf'_le _ (Finset.mem_univ δ)

lemma le_worst_iff {c : Δ → ι → ℝ} {y : ι → ℝ} {a : ℝ} :
    a ≤ worst c y ↔ ∀ δ, a ≤ cover c y δ := by
  unfold worst
  simp [Finset.le_inf'_iff]

lemma exists_worst_eq (c : Δ → ι → ℝ) (y : ι → ℝ) : ∃ δ, worst c y = cover c y δ := by
  obtain ⟨δ, -, h⟩ := Finset.exists_mem_eq_inf' (s := Finset.univ) Finset.univ_nonempty (cover c y)
  exact ⟨δ, h⟩

lemma budget_zero {s : ℝ} (hs : s ≠ 0) : budget s (0 : ι → ℝ) = 0 := by
  unfold budget
  simp [Real.zero_rpow (inv_ne_zero hs)]

lemma feasible_zero {s : ℝ} (hs : s ≠ 0) : Feasible s (0 : ι → ℝ) :=
  ⟨fun _ => le_rfl, by rw [budget_zero hs]; norm_num⟩

lemma rpow_one_div_nonneg {s x : ℝ} (hx : 0 ≤ x) : 0 ≤ x ^ (1 / s) := Real.rpow_nonneg hx _

/-- Each coordinate of a feasible point has `y_j^{1/s} ≤ 1`. -/
lemma Feasible.rpow_le_one {s : ℝ} {y : ι → ℝ} (hy : Feasible s y) (j : ι) :
    y j ^ (1 / s) ≤ 1 := by
  have h := hy.2
  unfold budget at h
  calc y j ^ (1 / s) ≤ ∑ k, y k ^ (1 / s) :=
        Finset.single_le_sum (fun k _ => Real.rpow_nonneg (hy.1 k) _) (Finset.mem_univ j)
    _ ≤ 1 := h

lemma Feasible.le_one {s : ℝ} (hs : 0 < s) {y : ι → ℝ} (hy : Feasible s y) (j : ι) :
    y j ≤ 1 := by
  have h := hy.rpow_le_one j
  by_contra hcon
  replace hcon := lt_of_not_ge hcon
  have : 1 < y j ^ (1 / s) := Real.one_lt_rpow hcon (by positivity)
  linarith

/-- For `s ≥ 1` and `0 ≤ x ≤ 1`, `x ≤ x^{1/s}`. -/
lemma le_rpow_one_div {s x : ℝ} (hs : 1 ≤ s) (hx0 : 0 ≤ x) (hx1 : x ≤ 1) : x ≤ x ^ (1 / s) := by
  have h1 : 1 / s ≤ 1 := by rw [div_le_one (by linarith)]; exact hs
  have := Real.rpow_le_rpow_of_exponent_ge' hx0 hx1 (by positivity) h1
  simpa using this

lemma Feasible.sum_le_one {s : ℝ} (hs : 1 ≤ s) {y : ι → ℝ} (hy : Feasible s y) :
    ∑ j, y j ≤ 1 := by
  calc ∑ j, y j ≤ ∑ j, y j ^ (1 / s) :=
        Finset.sum_le_sum fun j _ => le_rpow_one_div hs (hy.1 j) (hy.le_one (by linarith) j)
    _ ≤ 1 := hy.2

lemma cover_le {c : Δ → ι → ℝ} (hc2 : ∀ δ j, c δ j ≤ 2) {y : ι → ℝ} (hy : ∀ j, 0 ≤ y j)
    (δ : Δ) : cover c y δ ≤ 2 * ∑ j, y j := by
  unfold cover
  rw [Finset.mul_sum]
  exact Finset.sum_le_sum fun j _ => by nlinarith [hc2 δ j, hy j]

lemma worst_le_two {c : Δ → ι → ℝ} (hc2 : ∀ δ j, c δ j ≤ 2) {s : ℝ} (hs : 1 ≤ s)
    {y : ι → ℝ} (hy : Feasible s y) : worst c y ≤ 2 := by
  obtain ⟨δ⟩ := (inferInstance : Nonempty Δ)
  calc worst c y ≤ cover c y δ := worst_le_cover c y δ
    _ ≤ 2 * ∑ j, y j := cover_le hc2 hy.1 δ
    _ ≤ 2 * 1 := by gcongr; exact hy.sum_le_one hs
    _ = 2 := by ring

lemma bddAbove_worst {c : Δ → ι → ℝ} (hc2 : ∀ δ j, c δ j ≤ 2) {s : ℝ} (hs : 1 ≤ s) :
    BddAbove (worst c '' {y | Feasible s y}) :=
  ⟨2, by rintro _ ⟨y, hy, rfl⟩; exact worst_le_two hc2 hs hy⟩

lemma le_Gam {c : Δ → ι → ℝ} (hc2 : ∀ δ j, c δ j ≤ 2) {s : ℝ} (hs : 1 ≤ s)
    {y : ι → ℝ} (hy : Feasible s y) : worst c y ≤ Gam c s :=
  le_csSup (bddAbove_worst hc2 hs) ⟨y, hy, rfl⟩

lemma Gam_le {c : Δ → ι → ℝ} {s : ℝ} (hs : s ≠ 0) {a : ℝ}
    (h : ∀ y, Feasible s y → worst c y ≤ a) : Gam c s ≤ a :=
  csSup_le ⟨_, 0, feasible_zero hs, rfl⟩ (by rintro _ ⟨y, hy, rfl⟩; exact h y hy)

lemma continuous_cover (c : Δ → ι → ℝ) (δ : Δ) : Continuous fun y : ι → ℝ => cover c y δ := by
  unfold cover
  fun_prop

lemma continuous_worst (c : Δ → ι → ℝ) : Continuous fun y : ι → ℝ => worst c y := by
  unfold worst
  exact Continuous.finset_inf'_apply _ fun δ _ => continuous_cover c δ

lemma continuous_budget {s : ℝ} (hs : 0 < s) : Continuous fun y : ι → ℝ => budget s y := by
  unfold budget
  refine continuous_finsetSum _ fun j _ => ?_
  exact (continuous_apply j).rpow_const fun _ => Or.inr (by positivity)

lemma isCompact_feasible {s : ℝ} (hs : 1 ≤ s) : IsCompact {y : ι → ℝ | Feasible s y} := by
  have hsub : {y : ι → ℝ | Feasible s y} ⊆ Set.univ.pi fun _ => Set.Icc (0 : ℝ) 1 := by
    intro y hy j _
    exact ⟨hy.1 j, hy.le_one (by linarith) j⟩
  refine (isCompact_univ_pi fun _ => isCompact_Icc).of_isClosed_subset ?_ hsub
  have h1 : IsClosed {y : ι → ℝ | ∀ j, 0 ≤ y j} := by
    simp only [Set.ofPred_forall]
    exact isClosed_iInter fun j => isClosed_le continuous_const (continuous_apply j)
  have h2 : IsClosed {y : ι → ℝ | budget s y ≤ 1} :=
    isClosed_le (continuous_budget (by linarith)) continuous_const
  exact h1.inter h2

/-- The programme has a maximizer (so `Γ_s` is a maximum, as in the paper). -/
theorem exists_maximizer (c : Δ → ι → ℝ) {s : ℝ} (hs : 1 ≤ s) :
    ∃ y, IsMaximizer c s y := by
  obtain ⟨y, hy, hmax⟩ := (isCompact_feasible (ι := ι) hs).exists_isMaxOn
    ⟨0, feasible_zero (by linarith)⟩ (continuous_worst c).continuousOn
  exact ⟨y, hy, fun y' hy' => hmax hy'⟩

lemma IsMaximizer.worst_eq {c : Δ → ι → ℝ} (hc2 : ∀ δ j, c δ j ≤ 2) {s : ℝ} (hs : 1 ≤ s)
    {y : ι → ℝ} (hy : IsMaximizer c s y) : worst c y = Gam c s :=
  le_antisymm (le_Gam hc2 hs hy.1) (Gam_le (by linarith) hy.2)

lemma isMaximizer_iff {c : Δ → ι → ℝ} (hc2 : ∀ δ j, c δ j ≤ 2) {s : ℝ} (hs : 1 ≤ s)
    {y : ι → ℝ} : IsMaximizer c s y ↔ Feasible s y ∧ worst c y = Gam c s := by
  constructor
  · intro h; exact ⟨h.1, h.worst_eq hc2 hs⟩
  · rintro ⟨hf, he⟩
    exact ⟨hf, fun y' hy' => he ▸ le_Gam hc2 hs hy'⟩

end Abstract

/-! ## The concrete programme of the paper -/

/-- The distinct non-zero frequencies `F_p = {1, …, (p-1)/2}`. -/
abbrev Freq (p : ℕ) := {j : ℕ // j ∈ Finset.Icc 1 ((p - 1) / 2)}

/-- The non-zero offsets `δ ∈ ℤ_p \ {0}`. -/
abbrev Off (p : ℕ) := {δ : ZMod p // δ ≠ 0}

/-- The coverage `c_j(δ) = 1 - cos(2π j δ / p)` (with `δ` represented by `δ.val ∈ [0,p)`). -/
def cov (p : ℕ) (δ : ZMod p) (j : ℕ) : ℝ := 1 - Real.cos (2 * π * j * (δ.val : ℝ) / p)

/-- The coverage matrix of the paper, indexed by non-zero offsets and frequencies. -/
def covP (p : ℕ) (δ : Off p) (j : Freq p) : ℝ := cov p δ.1 j.1

instance instNeZeroOfPrime (p : ℕ) [hp : Fact p.Prime] : NeZero p := ⟨hp.out.ne_zero⟩

instance instNonemptyOff (p : ℕ) [Fact p.Prime] : Nonempty (Off p) := ⟨⟨1, one_ne_zero⟩⟩

/-- `Γ_s(p)` of eq. (programme). -/
def GammaP (p : ℕ) [Fact p.Prime] (s : ℝ) : ℝ := Gam (covP p) s

lemma cov_nonneg (p : ℕ) (δ : ZMod p) (j : ℕ) : 0 ≤ cov p δ j := by
  unfold cov; linarith [Real.cos_le_one (2 * π * j * (δ.val : ℝ) / p)]

lemma cov_le_two (p : ℕ) (δ : ZMod p) (j : ℕ) : cov p δ j ≤ 2 := by
  unfold cov; linarith [Real.neg_one_le_cos (2 * π * j * (δ.val : ℝ) / p)]

lemma covP_nonneg (p : ℕ) (δ : Off p) (j : Freq p) : 0 ≤ covP p δ j := cov_nonneg _ _ _

lemma covP_le_two (p : ℕ) (δ : Off p) (j : Freq p) : covP p δ j ≤ 2 := cov_le_two _ _ _

/-- The frequency `1 ∈ F_p` (for `p ≥ 3`). -/
def freqOne (p : ℕ) (hp : 3 ≤ p) : Freq p :=
  ⟨1, Finset.mem_Icc.2 ⟨le_rfl, by omega⟩⟩

lemma cov_one_pos (p : ℕ) [Fact p.Prime] (δ : ZMod p) (hδ : δ ≠ 0) : 0 < cov p δ 1 := by
  unfold cov
  have hp : (0 : ℝ) < p := by exact_mod_cast (Fact.out : p.Prime).pos
  have hv0 : 0 < δ.val := by
    rcases Nat.eq_zero_or_pos δ.val with h | h
    · exact absurd ((ZMod.val_eq_zero δ).1 h) hδ
    · exact h
  have hvp : δ.val < p := ZMod.val_lt δ
  have hx0 : 0 < 2 * π * ((1 : ℕ) : ℝ) * (δ.val : ℝ) / p := by
    have : (0 : ℝ) < δ.val := by exact_mod_cast hv0
    positivity
  have hx1 : 2 * π * ((1 : ℕ) : ℝ) * (δ.val : ℝ) / p < 2 * π := by
    rw [div_lt_iff₀ hp]
    have : (δ.val : ℝ) < p := by exact_mod_cast hvp
    have := Real.pi_pos
    push_cast
    nlinarith
  have hne : Real.cos (2 * π * ((1 : ℕ) : ℝ) * (δ.val : ℝ) / p) ≠ 1 := by
    intro h
    rw [Real.cos_eq_one_iff_of_lt_of_lt (by linarith [Real.pi_pos]) hx1] at h
    linarith
  have := Real.cos_le_one (2 * π * ((1 : ℕ) : ℝ) * (δ.val : ℝ) / p)
  have : Real.cos (2 * π * ((1 : ℕ) : ℝ) * (δ.val : ℝ) / p) < 1 := lt_of_le_of_ne this hne
  linarith

/-- `Γ_s(p) > 0` for `p ≥ 3` prime and `s ≥ 1`. -/
lemma GammaP_pos (p : ℕ) [Fact p.Prime] (hp : 3 ≤ p) {s : ℝ} (hs : 1 ≤ s) :
    0 < GammaP p s := by
  classical
  let e : Freq p → ℝ := fun j => if j = freqOne p hp then 1 else 0
  have hfe : Feasible s e := by
    refine ⟨fun j => by simp only [e]; split_ifs <;> norm_num, ?_⟩
    unfold budget
    rw [Finset.sum_eq_single (freqOne p hp)]
    · simp [e]
    · intro j _ hj
      simp [e, hj, Real.zero_rpow (inv_ne_zero (by linarith : s ≠ 0))]
    · simp
  have hpos : 0 < worst (covP p) e := by
    obtain ⟨δ, hδ⟩ := exists_worst_eq (covP p) e
    rw [hδ]
    unfold cover
    rw [Finset.sum_eq_single (freqOne p hp)]
    · simp only [e, ite_true, one_mul]
      exact cov_one_pos p δ.1 δ.2
    · intro j _ hj
      simp [e, hj]
    · simp
  exact lt_of_lt_of_le hpos (le_Gam (covP_le_two p) hs hfe)

/-! ## Bessel function and `ψ` (appendix `app:scaling`) -/

/-- The modified Bessel function `I_0(z) = (1/π) ∫_0^π e^{z cos θ} dθ`. -/
def I0 (z : ℝ) : ℝ := (1 / π) * ∫ θ in (0 : ℝ)..π, Real.exp (z * Real.cos θ)

/-- `ψ(z) = z - ln I_0(z)`. -/
def psi (z : ℝ) : ℝ := z - Real.log (I0 z)

/-- The constant `a_s = 3 + s/2`. -/
def aS (s : ℝ) : ℝ := 3 + s / 2

end Heracles
