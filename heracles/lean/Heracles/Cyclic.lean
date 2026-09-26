import Heracles.Cost
import Heracles.Product
import Heracles.NetTools
import Heracles.Approach
import Heracles.Vertex

/-!
# Theorem `thm:cyclic` (interference-free product networks)

Paper, Section `sec:cyclic`, proof in appendix `app:thm`.

Ingredients: `lem_sign` (Product.lean), `Cost.lem_cost`, `Cost.lem_cost_eq` (Cost.lean),
the single-neuron budget of the product neuron (Product.lean), `prop_coverage`, `lem_phase`
(Approach.lean), the `ν`-block network `blockNet` (Product.lean).
-/

open Real Finset

noncomputable section

set_option linter.unusedSectionVars false

namespace Heracles

variable {p n : ℕ}

lemma betaT_netTable [NeZero p] {Ω : Type*} {m : ℕ} (φ : Ω → Table p n) (θ : Fin m → Ω)
    (j : ℕ) (T : Finset (Fin n)) :
    betaT (netTable φ θ) j T = ∑ i, betaT (φ (θ i)) j T := by
  unfold betaT fhat netTable
  rw [← Finset.mul_sum]
  congr 1
  simp only [Complex.ofReal_sum, Finset.sum_mul]
  conv_rhs => rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun x _ => ?_
  exact Finset.sum_comm

lemma betaT_empty [NeZero p] (f : Table p n) (j : ℕ) : betaT f j ∅ = beta f j := by
  unfold betaT beta
  simp

variable [Fact p.Prime]

/-- The coefficients `(β_{ij})_i` of an interference-free product network at a frequency
`j ∈ F_p` satisfy the cancellation system (JF) with total `β_j`, for the unit numbers of the sign
rule (`lem_sign`). -/
theorem JF_of_interferenceFree {m : ℕ} (θ : Fin m → ProdParam p n)
    (hIF : InterferenceFree (netTable prodNeuron θ)) (j : Freq p)
    (ξ : Fin m → Fin n → ℂ)
    (hξ : ∀ i T, betaT (prodNeuron (θ i)) j.1 T = beta (prodNeuron (θ i)) j.1 * ∏ t ∈ T, ξ i t) :
    Cost.JF (fun i => beta (prodNeuron (θ i)) j.1) ξ (beta (netTable prodNeuron θ) j.1) := by
  intro T
  have h1 : ∑ i, beta (prodNeuron (θ i)) j.1 * ∏ t ∈ T, ξ i t = betaT (netTable prodNeuron θ) j.1 T := by
    rw [betaT_netTable]
    exact Finset.sum_congr rfl fun i _ => (hξ i T).symm
  rw [h1]
  split_ifs with hT
  · rw [hT, betaT_empty]
  · exact hIF j.1 j.2 T (Finset.nonempty_iff_ne_empty.2 hT)

/-- Choice of the sign-rule unit numbers for every neuron at frequency `j`. -/
lemma exists_signs {m : ℕ} (θ : Fin m → ProdParam p n) (j : ℕ) :
    ∃ ξ : Fin m → Fin n → ℂ, (∀ i t, ‖ξ i t‖ = 1) ∧
      ∀ i T, betaT (prodNeuron (θ i)) j T = beta (prodNeuron (θ i)) j * ∏ t ∈ T, ξ i t := by
  choose ξ hξ1 hξ2 using fun i => lem_sign (θ i) j
  exact ⟨ξ, hξ1, hξ2⟩

/-- **Per-frequency cost** (`lem:cost` applied through `lem:sign`): in an interference-free
product network, `ν^{1-r} A_j^r ≤ ∑_i |β_{ij}|^r` for `r ∈ (0,1]`. -/
theorem freq_cost {m : ℕ} (θ : Fin m → ProdParam p n)
    (hIF : InterferenceFree (netTable prodNeuron θ)) (j : Freq p) {r : ℝ} (hr0 : 0 < r)
    (hr1 : r ≤ 1) :
    ((n : ℝ) + 1) ^ (1 - r) * amp (netTable prodNeuron θ) j.1 ^ r ≤
      ∑ i, ‖beta (prodNeuron (θ i)) j.1‖ ^ r := by
  obtain ⟨ξ, hξ1, hξ2⟩ := exists_signs θ j.1
  exact Cost.lem_cost _ ξ hξ1 _ (JF_of_interferenceFree θ hIF j ξ hξ2) hr0 hr1

/-- The key chain (eq. (chain)) with an abstract single-neuron budget `C`:
`ν^{1-r} ∑_j A_j^r ≤ ∑_{i,j} |β_{ij}|^r ≤ C ∑_i ‖ω_i‖^b`. -/
theorem chain_ineq {m : ℕ} (θ : Fin m → ProdParam p n)
    (hIF : InterferenceFree (netTable prodNeuron θ)) {b C : ℝ} (hb : 0 < b)
    (hbν : b ≤ n + 1)
    (hbudget : ∀ ω : ProdParam p n,
      ∑ j : Freq p, ‖beta (prodNeuron ω) j.1‖ ^ (b / (n + 1)) ≤ C * Real.sqrt (normSq ω) ^ b) :
    ((n : ℝ) + 1) ^ (1 - b / (n + 1)) *
        ∑ j : Freq p, amp (netTable prodNeuron θ) j.1 ^ (b / (n + 1)) ≤
      C * normPow b θ := by
  have hν : (0 : ℝ) < n + 1 := by positivity
  have hr0 : 0 < b / (n + 1) := div_pos hb hν
  have hr1 : b / (n + 1) ≤ 1 := (div_le_one hν).2 hbν
  rw [Finset.mul_sum]
  calc ∑ j : Freq p, ((n : ℝ) + 1) ^ (1 - b / (n + 1)) *
          amp (netTable prodNeuron θ) j.1 ^ (b / (n + 1))
      ≤ ∑ j : Freq p, ∑ i, ‖beta (prodNeuron (θ i)) j.1‖ ^ (b / (n + 1)) :=
        Finset.sum_le_sum fun j _ => freq_cost θ hIF j hr0 hr1
    _ = ∑ i, ∑ j : Freq p, ‖beta (prodNeuron (θ i)) j.1‖ ^ (b / (n + 1)) := Finset.sum_comm
    _ ≤ ∑ i, C * Real.sqrt (normSq (θ i)) ^ b := Finset.sum_le_sum fun i _ => hbudget (θ i)
    _ = C * normPow b θ := by rw [normPow, Finset.mul_sum]

/-- The interference-free value `ν^{-(ν-b)/b} B_p Γ_{ν/b}(p)`. -/
def valueIF (p n : ℕ) [Fact p.Prime] (b : ℝ) : ℝ :=
  ((n : ℝ) + 1) ^ (-(((n : ℝ) + 1) - b) / b) * Bp p n * GammaP p (((n : ℝ) + 1) / b)

/-- Amplitude bound: `‖A‖_{1/s} ≤ ν^{-(ν-b)/b} B_p ‖θ‖_{2,b}^ν` for interference-free networks,
given the product budget `∑_j |β_j(ω)|^{b/ν} ≤ B_p^{b/ν} ‖ω‖^b`. -/
theorem ampNorm_le {m : ℕ} (θ : Fin m → ProdParam p n)
    (hIF : InterferenceFree (netTable prodNeuron θ)) {b : ℝ} (hb : 0 < b) (hbν : b ≤ n + 1)
    (hbudget : ∀ ω : ProdParam p n,
      ∑ j : Freq p, ‖beta (prodNeuron ω) j.1‖ ^ (b / (n + 1)) ≤
        Bp p n ^ (b / (n + 1)) * Real.sqrt (normSq ω) ^ b) :
    (∑ j : Freq p, amp (netTable prodNeuron θ) j.1 ^ (b / (n + 1))) ^ (((n : ℝ) + 1) / b) ≤
      ((n : ℝ) + 1) ^ (-(((n : ℝ) + 1) - b) / b) * Bp p n * normPow b θ ^ (((n : ℝ) + 1) / b) := by
  have hν : (0 : ℝ) < n + 1 := by positivity
  have hB : 0 ≤ Bp p n := by unfold Bp; positivity
  have hch := chain_ineq θ hIF hb hbν hbudget
  set S := ∑ j : Freq p, amp (netTable prodNeuron θ) j.1 ^ (b / (n + 1)) with hS
  have hS0 : 0 ≤ S := Finset.sum_nonneg fun j _ => Real.rpow_nonneg (norm_nonneg _) _
  have hN0 : 0 ≤ normPow b θ := Finset.sum_nonneg fun i _ => Real.rpow_nonneg (Real.sqrt_nonneg _) _
  have hc : 0 < ((n : ℝ) + 1) ^ (1 - b / (n + 1)) := Real.rpow_pos_of_pos hν _
  -- `S ≤ ν^{r-1} B^r N`
  have hS : S ≤ ((n : ℝ) + 1) ^ (b / (n + 1) - 1) * (Bp p n ^ (b / (n + 1)) * normPow b θ) := by
    have e : ((n : ℝ) + 1) ^ (b / (n + 1) - 1) = (((n : ℝ) + 1) ^ (1 - b / (n + 1)))⁻¹ := by
      rw [← Real.rpow_neg hν.le]; ring_nf
    rw [e, inv_mul_eq_div, le_div_iff₀ hc]
    linarith
  have hs : 0 < ((n : ℝ) + 1) / b := div_pos hν hb
  calc S ^ (((n : ℝ) + 1) / b)
      ≤ (((n : ℝ) + 1) ^ (b / (n + 1) - 1) * (Bp p n ^ (b / (n + 1)) * normPow b θ)) ^
          (((n : ℝ) + 1) / b) := Real.rpow_le_rpow hS0 hS hs.le
    _ = ((n : ℝ) + 1) ^ (-(((n : ℝ) + 1) - b) / b) * Bp p n * normPow b θ ^ (((n : ℝ) + 1) / b) := by
      rw [Real.mul_rpow (Real.rpow_nonneg hν.le _) (mul_nonneg (Real.rpow_nonneg hB _) hN0),
        Real.mul_rpow (Real.rpow_nonneg hB _) hN0, ← Real.rpow_mul hν.le, ← Real.rpow_mul hB]
      have e1 : (b / (↑n + 1) - 1) * ((↑n + 1) / b) = -((↑n + 1) - b) / b := by
        field_simp; ring
      have e2 : b / (↑n + 1) * ((↑n + 1) / b) = 1 := by field_simp
      rw [e1, e2, Real.rpow_one]
      ring

/-- **`thm:cyclic`, upper bound** (budget form): every interference-free product network
satisfies `h(θ) ≤ ν^{-(ν-b)/b} B_p Γ_{ν/b}(p) ‖θ‖_{2,b}^ν`. -/
theorem thm_cyclic_upper_of_budget (hp : 3 ≤ p) {m : ℕ} (θ : Fin m → ProdParam p n)
    (hIF : InterferenceFree (netTable prodNeuron θ)) {b : ℝ} (hb : 0 < b) (hbν : b ≤ n + 1)
    (hbudget : ∀ ω : ProdParam p n,
      ∑ j : Freq p, ‖beta (prodNeuron ω) j.1‖ ^ (b / (n + 1)) ≤
        Bp p n ^ (b / (n + 1)) * Real.sqrt (normSq ω) ^ b) :
    margin (netTable prodNeuron θ) ≤ valueIF p n b * normPow b θ ^ (((n : ℝ) + 1) / b) := by
  have hν : (0 : ℝ) < n + 1 := by positivity
  have hs1 : 1 ≤ ((n : ℝ) + 1) / b := (one_le_div hb).2 hbν
  have hrs : 1 / (((n : ℝ) + 1) / b) = b / (n + 1) := one_div_div _ _
  have hcov := prop_coverage hp (netTable prodNeuron θ) hs1
  rw [hrs] at hcov
  have hA := ampNorm_le θ hIF hb hbν hbudget
  have hΓ : 0 ≤ GammaP p (((n : ℝ) + 1) / b) := (GammaP_pos p hp hs1).le
  calc margin (netTable prodNeuron θ)
      ≤ GammaP p (((n : ℝ) + 1) / b) *
          (∑ j : Freq p, amp (netTable prodNeuron θ) j.1 ^ (b / (n + 1))) ^ (((n : ℝ) + 1) / b) := hcov
    _ ≤ GammaP p (((n : ℝ) + 1) / b) * (((n : ℝ) + 1) ^ (-(((n : ℝ) + 1) - b) / b) * Bp p n *
          normPow b θ ^ (((n : ℝ) + 1) / b)) := mul_le_mul_of_nonneg_left hA hΓ
    _ = valueIF p n b * normPow b θ ^ (((n : ℝ) + 1) / b) := by unfold valueIF; ring

/-- A maximizer spends its whole budget, for `s ≥ 1` (version of `IsMaximizer.budget_eq_one`). -/
theorem IsMaximizer.budget_eq_one' {ι Δ : Type*} [Fintype ι] [Fintype Δ] [Nonempty Δ]
    [DecidableEq ι] {c : Δ → ι → ℝ} (hc2 : ∀ δ j, c δ j ≤ 2) {s : ℝ}
    (hs : 1 ≤ s) (hΓ : 0 < Gam c s) {y : ι → ℝ} (hy : IsMaximizer c s y) :
    budget s y = 1 := by
  have hs0 : s ≠ 0 := by linarith
  have hw : worst c y = Gam c s := hy.worst_eq hc2 hs
  have hne : y ≠ 0 := by
    intro h; rw [h, worst_zero] at hw; linarith
  have hb := budget_pos_of_ne hs0 hy.1.1 hne
  by_contra hne1
  have hb1 : budget s y < 1 := lt_of_le_of_ne hy.1.2 hne1
  set b := budget s y
  set a := b ^ (-s) with ha
  have ha0 : 0 < a := Real.rpow_pos_of_pos hb _
  have ha1 : 1 < a := Real.one_lt_rpow_of_pos_of_lt_one_of_neg hb hb1 (by linarith)
  have hfeas : Feasible s (a • y) := by
    refine ⟨fun k => by simp only [Pi.smul_apply, smul_eq_mul]; exact mul_nonneg ha0.le (hy.1.1 k),
      ?_⟩
    rw [budget_smul ha0.le hy.1.1, ha, ← Real.rpow_mul hb.le]
    have : -s * (1 / s) = -1 := by field_simp
    rw [this, Real.rpow_neg_one, inv_mul_cancel₀ hb.ne']
  have h := hy.2 _ hfeas
  rw [worst_smul c ha0.le, hw] at h
  nlinarith

lemma blockNeuron_zero (j l : ℕ) : blockNeuron p n j 0 l = zeroParam p n := by
  simp [blockNeuron, zeroParam]

lemma lamOf_zero : lamOf n 0 = 0 := by
  unfold lamOf
  rw [mul_zero, zero_div, Real.zero_rpow]
  positivity

/-- The block network, compressed to any width `m ≥ ν |spt(y)|`. -/
theorem exists_block_width [NeZero p] (y : Freq p → ℝ) {b : ℝ} (hb : 0 < b) {m : ℕ}
    (hm : (n + 1) * (supp y).card ≤ m) :
    ∃ θ : Fin m → ProdParam p n, netTable prodNeuron θ = netTable prodNeuron (blockNet p n y) ∧
      normPow b θ = normPow b (blockNet p n y) := by
  classical
  set e := Fintype.equivFin (BIdx p n)
  set Q : Finset (BIdx p n) := Finset.univ.filter fun q => y q.1 ≠ 0
  set Z : Finset (Fin (Fintype.card (BIdx p n))) := Q.map e.toEmbedding
  have hQ : Q = supp y ×ˢ (Finset.univ : Finset (Fin (n + 1))) := by
    ext q; simp [Q, supp]
  have hZcard : Z.card = (n + 1) * (supp y).card := by
    rw [Finset.card_map, hQ, Finset.card_product, Finset.card_univ, Fintype.card_fin, mul_comm]
  refine exists_compress (blockNet p n y) Z ?_ (hZcard ▸ hm) hb
  intro i hi
  have hq : y (e.symm i).1 = 0 := by
    by_contra h
    apply hi
    refine Finset.mem_map.2 ⟨e.symm i, ?_, by simp⟩
    simp [Q, h]
  show blockNetQ p n y (e.symm i) = zeroParam p n
  unfold blockNetQ
  rw [hq, lamOf_zero, blockNeuron_zero]

lemma normPow_blockNet_pos [NeZero p] {b : ℝ} (y : Freq p → ℝ) (hy : ∀ j, 0 ≤ y j)
    (hbud : budget ((n + 1) / b) y = 1) : 0 < normPow b (blockNet p n y) := by
  rw [normPow_blockNet y hy hbud]
  have hp0 : (0 : ℝ) < p := by exact_mod_cast (NeZero.pos p)
  have : 0 < Real.sqrt ((n + 1) * p / 2) := Real.sqrt_pos.2 (by positivity)
  positivity

/-- **`thm:cyclic`, attainment.** For a maximizer `y*` of `Γ_{ν/b}(p)` and every width
`m ≥ ν |spt(y*)|`, there is an interference-free product network of width `m` with
`h(θ)/‖θ‖_{2,b}^ν = ν^{-(ν-b)/b} B_p Γ_{ν/b}(p)`. -/
theorem thm_cyclic_attain (hp : 3 ≤ p) {b : ℝ} (hb : 0 < b) (hbν : b ≤ n + 1)
    {y : Freq p → ℝ} (hy : IsMaximizer (covP p) (((n : ℝ) + 1) / b) y) {m : ℕ}
    (hm : (n + 1) * (supp y).card ≤ m) :
    ∃ θ : Fin m → ProdParam p n, InterferenceFree (netTable prodNeuron θ) ∧
      0 < normPow b θ ∧
      margin (netTable prodNeuron θ) / normPow b θ ^ (((n : ℝ) + 1) / b) = valueIF p n b := by
  have hs1 : 1 ≤ ((n : ℝ) + 1) / b := (one_le_div hb).2 hbν
  have hΓ := GammaP_pos p hp hs1
  have hbud := IsMaximizer.budget_eq_one' (covP_le_two p) hs1 hΓ hy
  have hw : worst (covP p) y = GammaP p (((n : ℝ) + 1) / b) := hy.worst_eq (covP_le_two p) hs1
  obtain ⟨θ, htab, hnorm⟩ := exists_block_width (n := n) y hb hm
  refine ⟨θ, ?_, ?_, ?_⟩
  · rw [htab]; exact blockNet_interferenceFree hp y hy.1.1
  · rw [hnorm]; exact normPow_blockNet_pos y hy.1.1 hbud
  · rw [htab, hnorm, blockNet_normalized_margin hb y hy.1.1 hbud, hw]
    unfold valueIF
    ring

/-- **`thm:cyclic`, structure of optimal interference-free networks** (budget form, `b < ν`).
If an interference-free product network attains the value, then (1) its amplitude vector is a
positive multiple of a maximizer `y⋆` of (programme) (so its set of used frequencies is
`spt(y⋆)`); (2) every used frequency `j` is carried by exactly `ν` neurons, each with
`|β_{ij}| = A_j/ν`, with pairwise orthogonal chain vectors (for the unit numbers of the sign
rule); and no neuron carries an unused frequency. -/
theorem thm_cyclic_structure_of_budget (hp : 3 ≤ p) {b : ℝ} (hb : 0 < b) (hbν : b < n + 1)
    (hbudget : ∀ ω : ProdParam p n,
      ∑ j : Freq p, ‖beta (prodNeuron ω) j.1‖ ^ (b / (n + 1)) ≤
        Bp p n ^ (b / (n + 1)) * Real.sqrt (normSq ω) ^ b)
    {m : ℕ} (θ : Fin m → ProdParam p n) (hIF : InterferenceFree (netTable prodNeuron θ))
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
    (∀ j : Freq p, amp (netTable prodNeuron θ) j.1 = 0 → ∀ i, beta (prodNeuron (θ i)) j.1 = 0) := by
  classical
  set ν : ℝ := (n : ℝ) + 1 with hνdef
  have hν : (0 : ℝ) < ν := by positivity
  set s := ν / b with hsdef
  set r := b / ν with hrdef
  have hs1 : 1 < s := (one_lt_div hb).2 hbν
  have hr0 : 0 < r := div_pos hb hν
  have hr1 : r < 1 := (div_lt_one hν).2 hbν
  have hrs : 1 / s = r := one_div_div ν b
  have hsr : s * r = 1 := by rw [hsdef, hrdef]; field_simp
  set f := netTable prodNeuron θ with hf
  set A : Freq p → ℝ := fun j => amp f j.1 with hAdef
  have hA0 : ∀ j, 0 ≤ A j := fun j => norm_nonneg _
  set S := ∑ j, A j ^ r with hSdef
  have hS0 : 0 ≤ S := Finset.sum_nonneg fun j _ => Real.rpow_nonneg (hA0 j) _
  set N := normPow b θ with hNdef
  have hΓ := GammaP_pos p hp hs1.le
  have hB : 0 < Bp p n := by unfold Bp; positivity
  have hcoefpos : 0 < ν ^ (-(ν - b) / b) := Real.rpow_pos_of_pos hν _
  have hV : 0 < valueIF p n b := by unfold valueIF; positivity
  -- the two inequalities
  have hcov : margin f ≤ GammaP p s * S ^ s := by
    have := prop_coverage hp f hs1.le
    rwa [hrs] at this
  have hAn : S ^ s ≤ ν ^ (-(ν - b) / b) * Bp p n * N ^ s := ampNorm_le θ hIF hb hbν.le hbudget
  have hmpos : 0 < margin f := by rw [heq]; exact mul_pos hV (Real.rpow_pos_of_pos hpos _)
  -- equality `S^s = ν^{..} B_p N^s`
  have hE1 : S ^ s = ν ^ (-(ν - b) / b) * Bp p n * N ^ s := by
    apply le_antisymm hAn
    have h1 : GammaP p s * (ν ^ (-(ν - b) / b) * Bp p n * N ^ s) ≤ GammaP p s * S ^ s := by
      have : valueIF p n b * N ^ s = GammaP p s * (ν ^ (-(ν - b) / b) * Bp p n * N ^ s) := by
        unfold valueIF; ring
      rw [← this, ← heq]; exact hcov
    exact le_of_mul_le_mul_left h1 hΓ
  have hSpos : 0 < S := by
    rcases hS0.lt_or_eq with h | h
    · exact h
    · exfalso
      rw [← h, Real.zero_rpow (by linarith)] at hcov
      linarith
  -- per-frequency equality `∑_i |β_ij|^r = ν^{1-r} A_j^r`
  have hsum2 : ∑ j : Freq p, ∑ i, ‖beta (prodNeuron (θ i)) j.1‖ ^ r ≤ Bp p n ^ r * N := by
    rw [Finset.sum_comm, hNdef, normPow, Finset.mul_sum]
    exact Finset.sum_le_sum fun i _ => hbudget (θ i)
  have hE2 : ν ^ (1 - r) * S = Bp p n ^ r * N := by
    have h := congrArg (fun z => z ^ r) hE1
    rw [← Real.rpow_mul hS0, hsr, Real.rpow_one,
      Real.mul_rpow (mul_nonneg hcoefpos.le hB.le) (Real.rpow_nonneg hpos.le _),
      Real.mul_rpow hcoefpos.le hB.le, ← Real.rpow_mul hν.le, ← Real.rpow_mul hpos.le, hsr,
      Real.rpow_one] at h
    have e : -(ν - b) / b * r = r - 1 := by rw [hrdef]; field_simp; ring
    rw [e] at h
    rw [h, ← mul_assoc, ← mul_assoc, ← Real.rpow_add hν]
    simp
  have hper : ∀ j : Freq p, ∑ i, ‖beta (prodNeuron (θ i)) j.1‖ ^ r = ν ^ (1 - r) * A j ^ r := by
    have hd : ∀ j ∈ (Finset.univ : Finset (Freq p)),
        0 ≤ ∑ i, ‖beta (prodNeuron (θ i)) j.1‖ ^ r - ν ^ (1 - r) * A j ^ r :=
      fun j _ => sub_nonneg.2 (freq_cost θ hIF j hr0 hr1.le)
    have htot : ∑ j : Freq p, (∑ i, ‖beta (prodNeuron (θ i)) j.1‖ ^ r - ν ^ (1 - r) * A j ^ r) = 0 := by
      apply le_antisymm _ (Finset.sum_nonneg hd)
      rw [Finset.sum_sub_distrib, ← Finset.mul_sum]
      have : ν ^ (1 - r) * ∑ j, A j ^ r = ν ^ (1 - r) * S := rfl
      linarith
    intro j
    have := (Finset.sum_eq_zero_iff_of_nonneg hd).1 htot j (Finset.mem_univ j)
    linarith
  refine ⟨?_, ?_, ?_⟩
  · -- (1) the normalized amplitude vector is a maximizer
    set a := S ^ (-s) with ha
    have ha0 : 0 < a := Real.rpow_pos_of_pos hSpos _
    set y := a • A with hy
    have hfeas : Feasible s y := by
      refine ⟨fun k => by simp only [hy, Pi.smul_apply, smul_eq_mul]; exact mul_nonneg ha0.le (hA0 k),
        ?_⟩
      rw [budget_smul ha0.le hA0]
      have : budget s A = S := by rw [budget, hrs]
      rw [this, ha, ← Real.rpow_mul hSpos.le]
      have e : -s * (1 / s) = -1 := by field_simp
      rw [e, Real.rpow_neg_one, inv_mul_cancel₀ hSpos.ne']
    have hSs : S ^ s * a = 1 := by rw [ha, ← Real.rpow_add hSpos]; simp
    have hworstA : margin f ≤ worst (covP p) A := by
      rw [le_worst_iff]
      intro δ
      have := lem_phase hp f δ.1 δ.2
      exact this.1.trans this.2
    have hmΓ : margin f = GammaP p s * S ^ s := by
      rw [heq, hE1]; unfold valueIF; ring
    have hwy : GammaP p s ≤ worst (covP p) y := by
      rw [hy, worst_smul _ ha0.le]
      have h2 : GammaP p s * S ^ s ≤ worst (covP p) A := hmΓ ▸ hworstA
      have h3 : GammaP p s = a * (GammaP p s * S ^ s) := by
        rw [mul_comm (GammaP p s), ← mul_assoc, mul_comm a, hSs, one_mul]
      rw [h3]
      exact mul_le_mul_of_nonneg_left h2 ha0.le
    have hmax : IsMaximizer (covP p) s y :=
      (isMaximizer_iff (covP_le_two p) hs1.le).2
        ⟨hfeas, le_antisymm (le_Gam (covP_le_two p) hs1.le hfeas) hwy⟩
    refine ⟨y, hmax, S ^ s, Real.rpow_pos_of_pos hSpos _, fun j => ?_, ?_⟩
    · simp only [hy, Pi.smul_apply, smul_eq_mul]
      rw [← mul_assoc, hSs, one_mul]
    · ext j
      simp [supp, hy, ha0.ne', A]
  · -- (2) equality case of `lem:cost` at every used frequency
    intro j hj
    obtain ⟨ξ, hξ1, hξ2⟩ := exists_signs θ j.1
    have hJF := JF_of_interferenceFree θ hIF j ξ hξ2
    have hBne : beta f j.1 ≠ 0 := fun h => hj (by show ‖beta f j.1‖ = 0; rw [h, norm_zero])
    have heqj := hper j
    obtain ⟨h1, h2, h3⟩ := Cost.lem_cost_eq _ ξ hξ1 _ hJF hr0 hr1 hBne heqj
    exact ⟨ξ, hξ1, hξ2, h1, h2, h3⟩
  · -- (3) no neuron carries an unused frequency
    intro j hj i
    have h0 := hper j
    have hAj : A j = 0 := hj
    rw [hAj, Real.zero_rpow hr0.ne', mul_zero] at h0
    have := (Finset.sum_eq_zero_iff_of_nonneg
      (fun i _ => Real.rpow_nonneg (norm_nonneg (beta (prodNeuron (θ i)) j.1)) r)).1 h0 i
      (Finset.mem_univ i)
    have := (Real.rpow_eq_zero (norm_nonneg _) hr0.ne').1 this
    exact norm_eq_zero.1 this

end Heracles
