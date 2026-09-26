import Heracles.Vertex
import Heracles.ExpSum

/-!
# `prop:shapes` and `prop:bottleneck` for the paper's programme `Γ_s(p)`

Concrete instances of the abstract results of `Heracles/Vertex.lean` and `Heracles/ExpSum.lean`
for the coverage matrix `covP p` (frequencies `F_p = {1,…,(p-1)/2}`, offsets `δ ≠ 0` in `ℤ_p`).
-/

open Real Finset

noncomputable section

set_option linter.unusedSectionVars false

namespace Heracles

variable {p : ℕ} [Fact p.Prime]

/-- The polyhedron `\curlyp_p`. -/
abbrev PolyP (p : ℕ) [Fact p.Prime] : Set (Freq p → ℝ) := Poly (covP p)

/-! ### `prop:shapes` -/

/-- **`prop:shapes`, first sentence.** Let `s > 1`. Every maximizer of (programme) is
`y = Γ_s(p) v` for a vertex `v` of `\curlyp_p` that minimizes `∑_j v_j^{1/s}` over it. -/
theorem prop_shapes_vertex (hp : 3 ≤ p) {s : ℝ} (hs : 1 < s) {y : Freq p → ℝ}
    (hy : IsMaximizer (covP p) s y) :
    ∃ v, y = GammaP p s • v ∧ IsVertex (covP p) v ∧ ∀ w ∈ PolyP p, budget s v ≤ budget s w :=
  maximizer_eq_vertex (covP_le_two p) hs (GammaP_pos p hp hs.le) hy

/-- The polyhedron has finitely many vertices. -/
theorem finite_verticesP (p : ℕ) [Fact p.Prime] : ((PolyP p).extremePoints ℝ).Finite :=
  finite_vertices (covP p)

/-- The set of vertices minimizing `φ_s = ∑_j v_j^{1/s}` over the polyhedron. -/
def minVertices (p : ℕ) [Fact p.Prime] (s : ℝ) : Set (Freq p → ℝ) :=
  {v | IsVertex (covP p) v ∧ ∀ w ∈ PolyP p, budget s v ≤ budget s w}

/-- A vertex minimizes `φ_s` over the polyhedron iff it minimizes it over the vertex set
(the minimum over the polyhedron is attained at a vertex). -/
lemma mem_minVertices_iff (hp : 3 ≤ p) {s : ℝ} (hs : 1 < s) (v : Freq p → ℝ) :
    v ∈ minVertices p s ↔
      IsVertex (covP p) v ∧ ∀ w, IsVertex (covP p) w → budget s v ≤ budget s w := by
  constructor
  · rintro ⟨hv, hmin⟩
    exact ⟨hv, fun w hw => hmin w (mem_extremePoints.1 hw).1⟩
  · rintro ⟨hv, hmin⟩
    refine ⟨hv, fun w hw => ?_⟩
    obtain ⟨v0, hv0, hmin0⟩ := exists_minimizer (covP_le_two p) hs (GammaP_pos p hp hs.le)
    have hvert0 : IsVertex (covP p) v0 := minimizer_isVertex hs hv0 hmin0
    exact (hmin v0 hvert0).trans (hmin0 w hw)

/-- **`prop:shapes`, finitely many switches.** There is a finite set `Z` of values of `s` such
that the set of minimizing vertices is the same at any two `s, s' > 1` with no point of `Z`
between them (inclusive).  In other words the set of minimizing vertices is constant on each
of finitely many intervals covering `(1,∞)`.  (Vertices whose non-zero entries agree as
multisets are always tied, `ties_of_multiset_eq`, so they enter and leave together; the
statement needs no "up to ties".) -/
theorem prop_shapes_switches (hp : 3 ≤ p) :
    ∃ Z : Finset ℝ, ∀ s s' : ℝ, 1 < s → 1 < s' → (∀ z ∈ Z, z ∉ Set.uIcc s s') →
      minVertices p s = minVertices p s' := by
  classical
  set V := (finite_verticesP p).toFinset with hV
  have hV0 : ∀ v ∈ V, ∀ j, 0 ≤ v j := by
    intro v hv j
    rw [hV, Set.Finite.mem_toFinset] at hv
    exact (mem_extremePoints.1 hv).1.1 j
  obtain ⟨Z, hZ⟩ := switches_finite V hV0
  refine ⟨Z, fun s s' hs hs' hZs => ?_⟩
  have key := hZ s s' hs hs' hZs
  have hconv : ∀ σ, 1 < σ → minVertices p σ =
      ↑(V.filter fun v => ∀ w ∈ V, budget σ v ≤ budget σ w) := by
    intro σ hσ
    ext v
    rw [mem_minVertices_iff hp hσ, Finset.coe_filter, Set.mem_ofPred_eq, hV,
      Set.Finite.mem_toFinset]
    constructor
    · rintro ⟨hv, hmin⟩
      exact ⟨hv, fun w hw => hmin w ((Set.Finite.mem_toFinset _).1 hw)⟩
    · rintro ⟨hv, hmin⟩
      exact ⟨hv, fun w hw => hmin w ((Set.Finite.mem_toFinset _).2 hw)⟩
  rw [hconv s hs, hconv s' hs', key]

/-- Maximizers are exactly `Γ_s(p)` times minimizing vertices. -/
theorem isMaximizer_iff_minVertex (hp : 3 ≤ p) {s : ℝ} (hs : 1 < s) (y : Freq p → ℝ) :
    IsMaximizer (covP p) s y ↔ ∃ v ∈ minVertices p s, y = GammaP p s • v := by
  constructor
  · intro hy
    obtain ⟨v, hyv, hv, hmin⟩ := prop_shapes_vertex hp hs hy
    exact ⟨v, ⟨hv, hmin⟩, hyv⟩
  · rintro ⟨v, ⟨hv, hmin⟩, rfl⟩
    exact isMaximizer_of_minimizer (covP_le_two p) hs (GammaP_pos p hp hs.le)
      (mem_extremePoints.1 hv).1 hmin

/-- **`prop:shapes`, consequence.** Between switches, the maximizers at `s'` are exactly the
maximizers at `s` rescaled by the positive factor `Γ_{s'}(p)/Γ_s(p)`: the support and the ratios
`y_j/y_k` of a maximizer are piecewise constant in `s`. -/
theorem maximizers_between_switches (hp : 3 ≤ p) :
    ∃ Z : Finset ℝ, ∀ s s' : ℝ, 1 < s → 1 < s' → (∀ z ∈ Z, z ∉ Set.uIcc s s') →
      ∀ y', IsMaximizer (covP p) s' y' ↔
        ∃ y, IsMaximizer (covP p) s y ∧ y' = (GammaP p s' / GammaP p s) • y := by
  obtain ⟨Z, hZ⟩ := prop_shapes_switches (p := p) hp
  refine ⟨Z, fun s s' hs hs' hZs y' => ?_⟩
  have hM := hZ s s' hs hs' hZs
  have hΓ := GammaP_pos p hp hs.le
  rw [isMaximizer_iff_minVertex hp hs']
  constructor
  · rintro ⟨v, hv, rfl⟩
    refine ⟨GammaP p s • v, (isMaximizer_iff_minVertex hp hs _).2 ⟨v, hM ▸ hv, rfl⟩, ?_⟩
    rw [smul_smul, div_mul_cancel₀ _ hΓ.ne']
  · rintro ⟨y, hy, rfl⟩
    obtain ⟨v, hv, rfl⟩ := (isMaximizer_iff_minVertex hp hs _).1 hy
    refine ⟨v, hM ▸ hv, ?_⟩
    rw [smul_smul, div_mul_cancel₀ _ hΓ.ne']

/-- **`prop:shapes`, budget shares.** For a maximizer `y = Γ_s(p) v`,
`y_j^{1/s} = v_j^{1/s} / ∑_k v_k^{1/s}`. -/
theorem prop_shapes_shares (hp : 3 ≤ p) {s : ℝ} (hs : 1 < s) {y v : Freq p → ℝ}
    (hy : IsMaximizer (covP p) s y) (hyv : y = GammaP p s • v) (hv : ∀ j, 0 ≤ v j)
    (j : Freq p) : y j ^ (1 / s) = v j ^ (1 / s) / budget s v :=
  share_formula (covP_le_two p) hs (GammaP_pos p hp hs.le) hy hyv hv j

/-! ### `prop:bottleneck` -/

lemma cos_reflect (j v : ℕ) (hv : v ≤ p) :
    Real.cos (2 * π * j * ((p - v : ℕ) : ℝ) / p) = Real.cos (2 * π * j * (v : ℝ) / p) := by
  have hp0 : (p : ℝ) ≠ 0 := by exact_mod_cast (Fact.out : p.Prime).ne_zero
  rw [Nat.cast_sub hv]
  have : 2 * π * (j : ℝ) * ((p : ℝ) - v) / p = (j : ℝ) * (2 * π) - 2 * π * j * v / p := by
    field_simp
  rw [this, Real.cos_nat_mul_two_pi_sub]

/-- The coverage is symmetric: `c_j(-δ) = c_j(δ)`. -/
lemma covP_neg (δ : Off p) (j : Freq p) :
    covP p ⟨-δ.1, neg_ne_zero.2 δ.2⟩ j = covP p δ j := by
  unfold covP cov
  have : NeZero δ.1 := ⟨δ.2⟩
  simp only
  rw [ZMod.val_neg_of_ne_zero, cos_reflect _ _ (ZMod.val_lt δ.1).le]

/-- **Proposition `prop:bottleneck`** (bottleneck offsets).  Let `s > 1` and `y` a maximizer
with `K = |spt(y)|`.  There are `K` distinct offsets `δ_1,…,δ_K ∈ {1,…,(p-1)/2}` with
`∑_j y_j c_j(δ_i) = Γ_s(p)`, whose coverage vectors restricted to `spt(y)` are linearly
independent; and `y` is the unique vector supported on `spt(y)` solving these `K` equations. -/
theorem prop_bottleneck (hp : 3 ≤ p) {s : ℝ} (hs : 1 < s) {y : Freq p → ℝ}
    (hy : IsMaximizer (covP p) s y) :
    ∃ δs : Fin (supp y).card → Off p, Function.Injective δs ∧
      (∀ i, 1 ≤ (δs i).1.val ∧ (δs i).1.val ≤ (p - 1) / 2) ∧
      (∀ i, cover (covP p) y (δs i) = GammaP p s) ∧
      LinearIndependent ℝ (fun i => fun j : supp y => covP p (δs i) j) ∧
      ∀ z : Freq p → ℝ, (∀ j, j ∉ supp y → z j = 0) →
        (∀ i, cover (covP p) z (δs i) = GammaP p s) → z = y := by
  classical
  have hΓ := GammaP_pos p hp hs.le
  set A : Finset (Off p) := Finset.univ.filter fun δ =>
    cover (covP p) y δ = GammaP p s ∧ δ.1.val ≤ (p - 1) / 2 with hA
  have hAact : ∀ δ ∈ A, cover (covP p) y δ = GammaP p s := fun δ hδ =>
    (Finset.mem_filter.1 hδ).2.1
  have hArows : ∀ δ, cover (covP p) y δ = GammaP p s →
      ∃ δ' ∈ A, ∀ j, covP p δ' j = covP p δ j := by
    intro δ hδ
    by_cases hle : δ.1.val ≤ (p - 1) / 2
    · exact ⟨δ, Finset.mem_filter.2 ⟨Finset.mem_univ _, hδ, hle⟩, fun _ => rfl⟩
    · set δ' : Off p := ⟨-δ.1, neg_ne_zero.2 δ.2⟩
      have hrow : ∀ j, covP p δ' j = covP p δ j := covP_neg δ
      have hcov : cover (covP p) y δ' = cover (covP p) y δ := by
        unfold cover; exact Finset.sum_congr rfl fun j _ => by rw [hrow]
      have hval : δ'.1.val ≤ (p - 1) / 2 := by
        have : NeZero δ.1 := ⟨δ.2⟩
        show (-δ.1).val ≤ (p - 1) / 2
        rw [ZMod.val_neg_of_ne_zero]
        have hodd : p % 2 = 1 :=
          Nat.odd_iff.1 ((Fact.out : p.Prime).odd_of_ne_two (by omega))
        have := ZMod.val_lt δ.1
        omega
      exact ⟨δ', Finset.mem_filter.2 ⟨Finset.mem_univ _, hcov.trans hδ, hval⟩, hrow⟩
  obtain ⟨δs, hinj, hmem, hact, hli, huniq⟩ :=
    bottleneck_abstract (covP_le_two p) hs hΓ hy A hAact hArows
  refine ⟨δs, hinj, fun i => ⟨?_, (Finset.mem_filter.1 (hmem i)).2.2⟩, hact, hli, huniq⟩
  have h0 : (δs i).1.val ≠ 0 := fun h => (δs i).2 ((ZMod.val_eq_zero _).1 h)
  omega

/-- The logit table `T(x, c) = ∑_j y_j cos(2π j (y(x) - c)/p)` of the network of
`prop:block` (as a function of the label `a = y(x)` and the candidate `c`). -/
def blockTable (y : Freq p → ℝ) (a c : ZMod p) : ℝ :=
  ∑ j, y j * Real.cos (2 * π * (j.1 : ℝ) * ((a - c).val : ℝ) / p)

/-- In a table of the form `blockTable`, the gap between the label `a` and the candidate
`δ` below it is `∑_j y_j c_j(δ)` on every input. -/
theorem blockTable_gap (y : Freq p → ℝ) (a : ZMod p) (δ : Off p) :
    blockTable y a a - blockTable y a (a - δ.1) = cover (covP p) y δ := by
  unfold blockTable cover covP cov
  rw [← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun j _ => ?_
  simp only [sub_self, ZMod.val_zero, Nat.cast_zero, mul_zero, zero_div, Real.cos_zero,
    sub_sub_cancel]
  ring

/-- **`prop:bottleneck`, network statement.** In a network whose table is `blockTable y` for a
maximizer `y` (the network of `prop:block`), on every input the `2K` candidates at offsets
`±δ_i` from the label are all hardest wrong answers: their gap is `Γ_s(p)`, the minimum gap over
all wrong candidates. -/
theorem prop_bottleneck_network (_hp : 3 ≤ p) {s : ℝ} (hs : 1 < s) {y : Freq p → ℝ}
    (hy : IsMaximizer (covP p) s y) (δs : Fin (supp y).card → Off p)
    (hact : ∀ i, cover (covP p) y (δs i) = GammaP p s) (a : ZMod p) (i : Fin (supp y).card) :
    blockTable y a a - blockTable y a (a - (δs i).1) = GammaP p s ∧
      blockTable y a a - blockTable y a (a + (δs i).1) = GammaP p s ∧
      ∀ δ : Off p, blockTable y a a - blockTable y a (a - (δs i).1) ≤
        blockTable y a a - blockTable y a (a - δ.1) := by
  have hw : worst (covP p) y = GammaP p s := hy.worst_eq (covP_le_two p) hs.le
  refine ⟨by rw [blockTable_gap, hact], ?_, fun δ => ?_⟩
  · have e : a + (δs i).1 = a - (⟨-(δs i).1, neg_ne_zero.2 (δs i).2⟩ : Off p).1 := by
      simp [sub_neg_eq_add]
    rw [e, blockTable_gap, ← hact i]
    unfold cover
    exact Finset.sum_congr rfl fun j _ => by rw [covP_neg]
  · rw [blockTable_gap, blockTable_gap, hact, ← hw]
    exact worst_le_cover _ _ _

end Heracles
