import Heracles.Network

/-!
# Tools for product networks: zero neurons, padding/compression, rescaling
-/

open Real Finset

noncomputable section

namespace Heracles

variable {p n : ℕ}

/-- The zero product neuron. -/
def zeroParam (p n : ℕ) : ProdParam p n := ⟨fun _ _ => 0, fun _ => 0⟩

lemma prodNeuron_zero (x : Inp p n) (c : ZMod p) : prodNeuron (zeroParam p n) x c = 0 := by
  simp [prodNeuron, zeroParam]

lemma normSq_zero [NeZero p] : normSq (zeroParam p n) = 0 := by
  simp [normSq, zeroParam]

/-- Rescaling all weights of a neuron by `λ`. -/
def smulParam (lam : ℝ) (ω : ProdParam p n) : ProdParam p n :=
  ⟨fun t a => lam * ω.u t a, fun c => lam * ω.w c⟩

lemma prodNeuron_smul (lam : ℝ) (ω : ProdParam p n) (x : Inp p n) (c : ZMod p) :
    prodNeuron (smulParam lam ω) x c = lam ^ (n + 1) * prodNeuron ω x c := by
  simp only [prodNeuron, smulParam, Finset.prod_mul_distrib, Finset.prod_const,
    Finset.card_univ, Fintype.card_fin, pow_succ]
  ring

lemma normSq_smul [NeZero p] (lam : ℝ) (ω : ProdParam p n) :
    normSq (smulParam lam ω) = lam ^ 2 * normSq ω := by
  simp only [normSq, smulParam, mul_pow, ← Finset.mul_sum, mul_add]

lemma normPow_smul [NeZero p] {m : ℕ} {b : ℝ} {lam : ℝ} (hlam : 0 ≤ lam)
    (θ : Fin m → ProdParam p n) :
    normPow b (fun i => smulParam lam (θ i)) = lam ^ b * normPow b θ := by
  unfold normPow
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [normSq_smul, Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq hlam,
    Real.mul_rpow hlam (Real.sqrt_nonneg _)]

lemma netTable_smul {m : ℕ} (lam : ℝ) (θ : Fin m → ProdParam p n) (x : Inp p n) (c : ZMod p) :
    netTable prodNeuron (fun i => smulParam lam (θ i)) x c =
      lam ^ (n + 1) * netTable prodNeuron θ x c := by
  unfold netTable
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun i _ => prodNeuron_smul lam (θ i) x c

lemma margin_smul [Fact p.Prime] {a : ℝ} (ha : 0 ≤ a) (f : Table p n) :
    margin (fun x c => a * f x c) = a * margin f := by
  apply le_antisymm
  · obtain ⟨xc, hxc, heq⟩ := Finset.exists_mem_eq_inf' (wrongPairs_nonempty p n)
      (fun xc : Inp p n × ZMod p => f xc.1 (label xc.1) - f xc.1 xc.2)
    have hmem : xc.2 ≠ label xc.1 := (Finset.mem_filter.1 hxc).2
    have h1 := margin_le (fun x c => a * f x c) xc.1 xc.2 hmem
    have h2 : margin f = f xc.1 (label xc.1) - f xc.1 xc.2 := heq
    rw [h2]
    linarith
  · apply le_margin
    intro x c hc
    have := margin_le f x c hc
    show a * margin f ≤ a * f x (label x) - a * f x c
    nlinarith

/-- **Padding / compression.** If all non-zero neurons of `θ` lie in a set of at most `m`
indices, there is a network of width `m` with the same table and the same `normPow`. -/
theorem exists_compress [NeZero p] {W m : ℕ} (θ : Fin W → ProdParam p n) (Z : Finset (Fin W))
    (hZ : ∀ i, i ∉ Z → θ i = zeroParam p n) (hm : Z.card ≤ m) {b : ℝ} (hb : 0 < b) :
    ∃ θ' : Fin m → ProdParam p n, netTable prodNeuron θ' = netTable prodNeuron θ ∧
      normPow b θ' = normPow b θ := by
  classical
  -- an embedding of `Z` into `Fin m`
  have hcard : Fintype.card Z ≤ Fintype.card (Fin m) := by simpa using hm
  obtain ⟨e⟩ := Function.Embedding.nonempty_of_card_le hcard
  let θ' : Fin m → ProdParam p n := fun i =>
    if h : ∃ z : Z, e z = i then θ (Classical.choose h).1 else zeroParam p n
  have hθ'e : ∀ z : Z, θ' (e z) = θ z.1 := by
    intro z
    have h : ∃ z' : Z, e z' = e z := ⟨z, rfl⟩
    simp only [θ', dif_pos h]
    rw [e.injective (Classical.choose_spec h)]
  have hθ'0 : ∀ i, (¬ ∃ z : Z, e z = i) → θ' i = zeroParam p n := by
    intro i hi; simp only [θ', dif_neg hi]
  -- general sum transfer for `g` with `g 0 = 0`
  have transfer : ∀ {M : Type} [AddCommMonoid M] (g : ProdParam p n → M), g (zeroParam p n) = 0 →
      ∑ i, g (θ' i) = ∑ i, g (θ i) := by
    intro M _ g hg
    have h1 : ∑ i, g (θ' i) = ∑ z : Z, g (θ' (e z)) := by
      rw [← Finset.sum_subset (Finset.subset_univ (Finset.univ.map e))]
      · rw [Finset.sum_map]
      · intro i _ hi
        have : ¬ ∃ z : Z, e z = i := by
          rintro ⟨z, rfl⟩; exact hi (Finset.mem_map_of_mem _ (Finset.mem_univ z))
        rw [hθ'0 i this, hg]
    have h2 : ∑ i, g (θ i) = ∑ z : Z, g (θ z.1) := by
      rw [← Finset.sum_subset (Finset.subset_univ Z)]
      · exact (Finset.sum_coe_sort Z (fun i => g (θ i))).symm
      · intro i _ hi; rw [hZ i hi, hg]
    rw [h1, h2]
    exact Finset.sum_congr rfl fun z _ => by rw [hθ'e]
  refine ⟨θ', ?_, ?_⟩
  · funext x c
    unfold netTable
    exact transfer (fun ω => prodNeuron ω x c) (prodNeuron_zero x c)
  · unfold normPow
    exact transfer (fun ω => Real.sqrt (normSq ω) ^ b)
      (by rw [normSq_zero, Real.sqrt_zero, Real.zero_rpow hb.ne'])

end Heracles
