import Heracles.Vertex
import Heracles.Scalar
import Heracles.PosFourier
import Heracles.Lower

/-!
# Theorem `thm:scaling` and Corollary `cor:few` (logarithmic scaling)

Paper, Section `sec:cyclic`, proof in appendix `app:scaling`.

Ingredients proved elsewhere:
* `posfourier` (eq. (posfourier)) — `Heracles/PosFourier.lean`;
* `psi_le` (the scalar bound) — `Heracles/Scalar.lean`;
* `GammaP_lower`, `GammaP_lower9`, `exists_alloc_lower` (random-frequency lower bound) —
  `Heracles/Lower.lean`;
* `prop_ablation` — `Heracles/Ablation.lean`; `IsMaximizer.budget_eq_one` — `Heracles/Vertex.lean`.

Here: (psisum), (twoterms), the two choices of `t`, (Kbound), the bounds on the number of
frequencies and the share floor, and the network-level consequences.
-/

open Real Finset

noncomputable section

namespace Heracles

set_option linter.unusedSectionVars false

variable {p : ℕ} [Fact p.Prime]

lemma log_p_pos (hp : 3 ≤ p) : 0 < Real.log p :=
  Real.log_pos (by exact_mod_cast (by omega : 1 < p))

/-! ### (psisum) -/

/-- `log (1 + λ x) ≥ λ log (1 + x)` for `λ ∈ [0,1]`, `x ≥ 0` (concavity of `log`). -/
lemma log_one_add_mul_ge {lam x : ℝ} (hl0 : 0 ≤ lam) (hl1 : lam ≤ 1) (hx : 0 ≤ x) :
    lam * Real.log (1 + x) ≤ Real.log (1 + lam * x) := by
  have hc := (strictConcaveOn_log_Ioi).concaveOn
  have h := hc.2 (Set.mem_Ioi.2 (by linarith : (0 : ℝ) < 1 + x)) (Set.mem_Ioi.2 one_pos)
    hl0 (by linarith : 0 ≤ 1 - lam) (by ring)
  simp only [smul_eq_mul, Real.log_one, mul_zero, add_zero] at h
  have : lam * (1 + x) + (1 - lam) * 1 = 1 + lam * x := by ring
  rwa [this] at h

/-- Jensen for `log`: `∑_{j∈S} log(1+u_j) ≤ |S| log(1 + (∑_{j∈S} u_j)/|S|)`. -/
lemma sum_log_le_card_mul {κ : Type*} (S : Finset κ) (hS : S.Nonempty) (u : κ → ℝ)
    (hu : ∀ j ∈ S, 0 ≤ u j) :
    ∑ j ∈ S, Real.log (1 + u j) ≤ S.card * Real.log (1 + (∑ j ∈ S, u j) / S.card) := by
  have hc := (strictConcaveOn_log_Ioi).concaveOn
  have hcard : (0 : ℝ) < S.card := by exact_mod_cast hS.card_pos
  have h := hc.le_map_sum (t := S) (w := fun _ => 1 / (S.card : ℝ)) (p := fun j => 1 + u j)
    (fun _ _ => by positivity) (by rw [Finset.sum_const, nsmul_eq_mul]; field_simp)
    (fun j hj => Set.mem_Ioi.2 (by linarith [hu j hj]))
  simp only [smul_eq_mul] at h
  rw [← Finset.mul_sum] at h
  have e : ∑ i ∈ S, 1 / (S.card : ℝ) * (1 + u i) = 1 + (∑ j ∈ S, u j) / S.card := by
    rw [← Finset.mul_sum, Finset.sum_add_distrib, Finset.sum_const, nsmul_eq_mul]
    field_simp
  rw [e] at h
  rw [← div_le_iff₀' hcard]
  calc (∑ j ∈ S, Real.log (1 + u j)) / S.card
      = 1 / (S.card : ℝ) * ∑ j ∈ S, Real.log (1 + u j) := by ring
    _ ≤ _ := h

/-- `k log(1 + c/k) ≤ K log(1 + c/K)` for `0 < k ≤ K`, `c ≥ 0`. -/
lemma mul_log_mono {k K c : ℝ} (hk : 0 < k) (hkK : k ≤ K) (hc : 0 ≤ c) :
    k * Real.log (1 + c / k) ≤ K * Real.log (1 + c / K) := by
  have hK : 0 < K := lt_of_lt_of_le hk hkK
  have h := log_one_add_mul_ge (lam := k / K) (x := c / k) (by positivity)
    ((div_le_one hK).2 hkK) (by positivity)
  have e : k / K * (c / k) = c / K := by field_simp
  rw [e] at h
  have := mul_le_mul_of_nonneg_left h hK.le
  calc k * Real.log (1 + c / k) = K * (k / K * Real.log (1 + c / k)) := by field_simp
    _ ≤ K * Real.log (1 + c / K) := this

/-- **(psisum), first two inequalities.** For feasible `y` with at most `K ≥ 1` non-zero
entries and `t ≥ 0`: `∑_j ψ(t y_j) ≤ a_s ∑_j ln(1 + t^r y_j^r) ≤ a_s K ln(1 + t^r/K)`. -/
theorem psisum_K {s : ℝ} (hs : 1 < s) {y : Freq p → ℝ} (hy : Feasible s y) {t : ℝ} (ht : 0 ≤ t)
    {K : ℕ} (hK0 : 0 < K) (hK : (supp y).card ≤ K) :
    ∑ j, psi (t * y j) ≤ aS s * ∑ j, Real.log (1 + t ^ (1 / s) * y j ^ (1 / s)) ∧
      aS s * ∑ j, Real.log (1 + t ^ (1 / s) * y j ^ (1 / s)) ≤
        aS s * K * Real.log (1 + t ^ (1 / s) / K) := by
  classical
  have hs0 : s ≠ 0 := by linarith
  have ha : 0 < aS s := by unfold aS; linarith
  have htr : 0 ≤ t ^ (1 / s) := Real.rpow_nonneg ht _
  constructor
  · rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun j _ => ?_
    have := psi_le hs (mul_nonneg ht (hy.1 j))
    rwa [Real.mul_rpow ht (hy.1 j)] at this
  · -- restrict to the support
    set u : Freq p → ℝ := fun j => t ^ (1 / s) * y j ^ (1 / s) with hu
    have hu0 : ∀ j, 0 ≤ u j := fun j => mul_nonneg htr (Real.rpow_nonneg (hy.1 j) _)
    have hsupp : ∑ j, Real.log (1 + u j) = ∑ j ∈ supp y, Real.log (1 + u j) := by
      symm
      refine Finset.sum_subset (Finset.subset_univ _) fun j _ hj => ?_
      have : y j = 0 := by simpa [supp] using hj
      simp [hu, this, Real.zero_rpow (inv_ne_zero hs0)]
    have hsumu : ∑ j ∈ supp y, u j ≤ t ^ (1 / s) := by
      calc ∑ j ∈ supp y, u j ≤ ∑ j, u j :=
            Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _) fun j _ _ => hu0 j
        _ = t ^ (1 / s) * budget s y := by rw [budget, Finset.mul_sum]
        _ ≤ t ^ (1 / s) * 1 := mul_le_mul_of_nonneg_left hy.2 htr
        _ = t ^ (1 / s) := mul_one _
    have hKR : (0 : ℝ) < K := by exact_mod_cast hK0
    have hmain : ∑ j, Real.log (1 + u j) ≤ K * Real.log (1 + t ^ (1 / s) / K) := by
      rw [hsupp]
      rcases (supp y).eq_empty_or_nonempty with hE | hE
      · rw [hE, Finset.sum_empty]
        exact mul_nonneg hKR.le (Real.log_nonneg (by have := div_nonneg htr hKR.le; linarith))
      · have hc : (0 : ℝ) < (supp y).card := by exact_mod_cast hE.card_pos
        calc ∑ j ∈ supp y, Real.log (1 + u j)
            ≤ (supp y).card * Real.log (1 + (∑ j ∈ supp y, u j) / (supp y).card) :=
              sum_log_le_card_mul _ hE u fun j _ => hu0 j
          _ ≤ (supp y).card * Real.log (1 + t ^ (1 / s) / (supp y).card) := by
              gcongr
              · have := Finset.sum_nonneg fun j (_ : j ∈ supp y) => hu0 j
                have := div_nonneg this hc.le
                linarith
          _ ≤ K * Real.log (1 + t ^ (1 / s) / K) :=
              mul_log_mono hc (by exact_mod_cast hK) htr
    calc aS s * ∑ j, Real.log (1 + u j) ≤ aS s * (K * Real.log (1 + t ^ (1 / s) / K)) :=
          mul_le_mul_of_nonneg_left hmain ha.le
      _ = aS s * K * Real.log (1 + t ^ (1 / s) / K) := by ring

/-- `K ln(1 + c/K) ≤ c`. -/
lemma mul_log_one_add_div_le {K c : ℝ} (hK : 0 < K) (hc : 0 ≤ c) : K * Real.log (1 + c / K) ≤ c := by
  have h := Real.log_le_sub_one_of_pos (by positivity : (0 : ℝ) < 1 + c / K)
  have := mul_le_mul_of_nonneg_left h hK.le
  calc K * Real.log (1 + c / K) ≤ K * (1 + c / K - 1) := this
    _ = c := by field_simp; ring

/-- **(psisum)**, full chain: `∑_j ψ(t y_j) ≤ a_s t^r` for every feasible `y`. -/
theorem psisum (hp : 3 ≤ p) {s : ℝ} (hs : 1 < s) {y : Freq p → ℝ} (hy : Feasible s y)
    {t : ℝ} (ht : 0 ≤ t) : ∑ j, psi (t * y j) ≤ aS s * t ^ (1 / s) := by
  have hK0 : 0 < Fintype.card (Freq p) := Fintype.card_pos_iff.2 ⟨freqOne p hp⟩
  have hK : (supp y).card ≤ Fintype.card (Freq p) := Finset.card_le_univ _
  obtain ⟨h1, h2⟩ := psisum_K hs hy ht hK0 hK
  have ha : 0 < aS s := by unfold aS; linarith
  have h3 := mul_log_one_add_div_le (K := (Fintype.card (Freq p) : ℝ)) (c := t ^ (1 / s))
    (by exact_mod_cast hK0) (Real.rpow_nonneg ht _)
  calc ∑ j, psi (t * y j) ≤ aS s * (Fintype.card (Freq p)) *
        Real.log (1 + t ^ (1 / s) / (Fintype.card (Freq p))) := h1.trans h2
    _ = aS s * ((Fintype.card (Freq p)) *
        Real.log (1 + t ^ (1 / s) / (Fintype.card (Freq p)))) := by ring
    _ ≤ aS s * t ^ (1 / s) := mul_le_mul_of_nonneg_left h3 ha.le

/-! ### (twoterms) -/

lemma cov_zero (j : ℕ) : cov p 0 j = 0 := by
  unfold cov; simp

lemma cover_eq_sum_cov (y : Freq p → ℝ) (δ : Off p) :
    cover (covP p) y δ = ∑ j, y j * cov p δ.1 j.1 := rfl

/-- **(twoterms)**: `(1/p) ∑_δ e^{-t F_y(δ)} ≤ 1/p + e^{-t m(y)}` for `t ≥ 0`, `y ≥ 0`. -/
theorem twoterms (y : Freq p → ℝ) {t : ℝ} (ht : 0 ≤ t) :
    (1 / (p : ℝ)) * ∑ δ : ZMod p, Real.exp (-(t * ∑ j, y j * cov p δ j.1)) ≤
      1 / p + Real.exp (-(t * worst (covP p) y)) := by
  classical
  have hp0 : (0 : ℝ) < p := by exact_mod_cast (Fact.out : p.Prime).pos
  rw [← Finset.add_sum_erase _ _ (Finset.mem_univ (0 : ZMod p))]
  simp only [cov_zero, mul_zero, Finset.sum_const_zero, neg_zero, Real.exp_zero]
  have hbound : ∀ δ ∈ (Finset.univ : Finset (ZMod p)).erase 0,
      Real.exp (-(t * ∑ j, y j * cov p δ j.1)) ≤ Real.exp (-(t * worst (covP p) y)) := by
    intro δ hδ
    have hδ0 : δ ≠ 0 := Finset.ne_of_mem_erase hδ
    have := worst_le_cover (covP p) y ⟨δ, hδ0⟩
    rw [cover_eq_sum_cov] at this
    apply Real.exp_le_exp.2
    have := mul_le_mul_of_nonneg_left this ht
    linarith
  have hsum := Finset.sum_le_card_nsmul _ _ _ hbound
  rw [Finset.card_erase_of_mem (Finset.mem_univ _), Finset.card_univ, ZMod.card,
    nsmul_eq_mul] at hsum
  have hE : 0 ≤ Real.exp (-(t * worst (covP p) y)) := (Real.exp_pos _).le
  have hp1 : (1 : ℝ) ≤ p := by exact_mod_cast (Fact.out : p.Prime).one_lt.le
  have hcast : ((p - 1 : ℕ) : ℝ) = p - 1 := by
    rw [Nat.cast_sub (Fact.out : p.Prime).one_lt.le]; simp
  rw [hcast] at hsum
  rw [mul_add, one_div]
  have : (p : ℝ)⁻¹ * ∑ x ∈ Finset.univ.erase 0, Real.exp (-(t * ∑ j, y j * cov p x j.1)) ≤
      Real.exp (-(t * worst (covP p) y)) := by
    calc (p : ℝ)⁻¹ * ∑ x ∈ Finset.univ.erase 0, Real.exp (-(t * ∑ j, y j * cov p x j.1))
        ≤ (p : ℝ)⁻¹ * ((p - 1) * Real.exp (-(t * worst (covP p) y))) :=
          mul_le_mul_of_nonneg_left hsum (by positivity)
      _ ≤ Real.exp (-(t * worst (covP p) y)) := by
          rw [← mul_assoc]
          have : (p : ℝ)⁻¹ * (p - 1) ≤ 1 := by
            rw [inv_mul_le_iff₀ hp0]; linarith
          nlinarith
  linarith [this]

/-! ### The core estimate: a bound on `m(y)` from a choice of `t` -/

/-- If `∑_j ψ(t y_j) ≤ ½ ln p` for some `t > 0`, then `m(y) ≤ (3/2) ln p / t`. -/
theorem worst_le_of_psi (hp : 3 ≤ p) {y : Freq p → ℝ} (hy : ∀ j, 0 ≤ y j) {t : ℝ} (ht : 0 < t)
    (hψ : ∑ j, psi (t * y j) ≤ Real.log p / 2) :
    worst (covP p) y ≤ (3 / 2) * Real.log p / t := by
  have hp0 : (0 : ℝ) < p := by exact_mod_cast (Fact.out : p.Prime).pos
  have hlog := log_p_pos hp
  have hpf := posfourier p hp y hy ht.le
  have htt := twoterms y ht.le
  set m := worst (covP p) y
  -- `e^{-Φ} ≥ p^{-1/2}`
  set q := Real.sqrt p with hq
  have hq0 : 0 < q := Real.sqrt_pos.2 hp0
  have hqq : q * q = p := Real.mul_self_sqrt hp0.le
  have hq3 : (3 : ℝ) / 2 ≤ q := by
    rw [hq, Real.le_sqrt (by norm_num) hp0.le]
    have : (3 : ℝ) ≤ p := by exact_mod_cast hp
    linarith
  have hlogq : Real.log q = Real.log p / 2 := by
    rw [hq, Real.log_sqrt hp0.le]
  have h1 : q⁻¹ ≤ Real.exp (-∑ j, psi (t * y j)) := by
    rw [← Real.exp_log (inv_pos.2 hq0), Real.log_inv, hlogq]
    exact Real.exp_le_exp.2 (by linarith)
  have h2 : q⁻¹ ≤ 1 / p + Real.exp (-(t * m)) := h1.trans (hpf.trans htt)
  -- `e^{-tm} ≥ 1/(3q)`
  have h3 : (3 * q)⁻¹ ≤ Real.exp (-(t * m)) := by
    have : 1 / (p : ℝ) = q⁻¹ * q⁻¹ := by rw [← hqq]; field_simp
    rw [this] at h2
    have hqi : q⁻¹ ≤ 2 / 3 := by
      rw [inv_le_comm₀ hq0 (by norm_num)]; linarith
    have e : (3 * q)⁻¹ = q⁻¹ * (1 / 3) := by field_simp
    rw [e]
    have hqi0 : 0 < q⁻¹ := inv_pos.2 hq0
    nlinarith
  have h4 := Real.log_le_log (by positivity) h3
  rw [Real.log_exp, Real.log_inv, Real.log_mul (by norm_num) hq0.ne', hlogq] at h4
  have hl3 : Real.log 3 ≤ Real.log p :=
    Real.log_le_log (by norm_num) (by exact_mod_cast hp)
  -- `t m ≤ ln 3 + ½ ln p ≤ (3/2) ln p`
  rw [le_div_iff₀ ht]
  linarith

/-! ### Upper bound on `Γ_s(p)` -/

/-- **`thm:scaling`, upper bound**: `Γ_s(p) ≤ (3/2)(2 a_s)^s (ln p)^{1-s}`. -/
theorem GammaP_upper (hp : 3 ≤ p) {s : ℝ} (hs : 1 < s) :
    GammaP p s ≤ (3 / 2) * (2 * aS s) ^ s * Real.log p ^ (1 - s) := by
  have hs0 : 0 < s := by linarith
  have ha : 0 < aS s := by unfold aS; linarith
  have hlog := log_p_pos hp
  set x := Real.log p / (2 * aS s) with hx
  have hx0 : 0 < x := by positivity
  set t := x ^ s with ht
  have ht0 : 0 < t := Real.rpow_pos_of_pos hx0 _
  have htr : t ^ (1 / s) = x := by
    rw [ht, ← Real.rpow_mul hx0.le, mul_one_div_cancel hs0.ne', Real.rpow_one]
  apply Gam_le hs0.ne'
  intro y hy
  have hψ : ∑ j, psi (t * y j) ≤ Real.log p / 2 := by
    have := psisum hp hs hy ht0.le
    rw [htr, hx] at this
    have e : aS s * (Real.log p / (2 * aS s)) = Real.log p / 2 := by field_simp
    linarith
  have h := worst_le_of_psi hp hy.1 ht0 hψ
  calc worst (covP p) y ≤ (3 / 2) * Real.log p / t := h
    _ = (3 / 2) * (2 * aS s) ^ s * Real.log p ^ (1 - s) := by
      rw [ht, hx, Real.div_rpow hlog.le (by positivity), Real.rpow_sub hlog, Real.rpow_one]
      field_simp

/-! ### (Kbound) -/

/-- **(Kbound)**: for feasible `y` with at most `K ≥ 1` non-zero entries,
`m(y) ≤ (3/2)(ln p)^{1-s} (x/(e^{x/(2a_s)} - 1))^s` with `x = ln p / K`. -/
theorem Kbound (hp : 3 ≤ p) {s : ℝ} (hs : 1 < s) {y : Freq p → ℝ} (hy : Feasible s y)
    {K : ℕ} (hK0 : 0 < K) (hK : (supp y).card ≤ K) :
    worst (covP p) y ≤ (3 / 2) * Real.log p ^ (1 - s) *
      ((Real.log p / K) / (Real.exp ((Real.log p / K) / (2 * aS s)) - 1)) ^ s := by
  have hs0 : 0 < s := by linarith
  have ha : 0 < aS s := by unfold aS; linarith
  have hlog := log_p_pos hp
  have hKR : (0 : ℝ) < K := by exact_mod_cast hK0
  set x := Real.log p / K with hx
  have hx0 : 0 < x := by positivity
  set E := Real.exp (x / (2 * aS s)) with hE
  have hE1 : 1 < E := Real.one_lt_exp_iff.2 (by positivity)
  set t := (K * (E - 1)) ^ s with ht
  have hKE : 0 < K * (E - 1) := by nlinarith
  have ht0 : 0 < t := Real.rpow_pos_of_pos hKE _
  have htr : t ^ (1 / s) = K * (E - 1) := by
    rw [ht, ← Real.rpow_mul hKE.le, mul_one_div_cancel hs0.ne', Real.rpow_one]
  have hψ : ∑ j, psi (t * y j) ≤ Real.log p / 2 := by
    obtain ⟨h1, h2⟩ := psisum_K hs hy ht0.le hK0 hK
    rw [htr] at h1 h2
    have e1 : 1 + (K : ℝ) * (E - 1) / K = E := by field_simp; ring
    rw [e1, hE, Real.log_exp] at h2
    have e2 : aS s * K * (x / (2 * aS s)) = Real.log p / 2 := by
      rw [hx]; field_simp
    linarith
  have h := worst_le_of_psi hp hy.1 ht0 hψ
  calc worst (covP p) y ≤ (3 / 2) * Real.log p / t := h
    _ = (3 / 2) * Real.log p ^ (1 - s) * (x / (E - 1)) ^ s := by
      rw [ht, Real.div_rpow hx0.le (by linarith), Real.mul_rpow hKR.le (by linarith),
        Real.rpow_sub hlog, Real.rpow_one, hx, Real.div_rpow hlog.le hKR.le]
      have h1 : 0 < (E - 1) ^ s := Real.rpow_pos_of_pos (by linarith) _
      have h2 : 0 < (K : ℝ) ^ s := Real.rpow_pos_of_pos hKR _
      have h3 : 0 < Real.log p ^ s := Real.rpow_pos_of_pos hlog _
      field_simp

/-! ### Number of frequencies -/

/-- `e^z - 1 ≥ z²/2` for `z ≥ 0`. -/
lemma exp_sub_one_ge_sq {z : ℝ} (hz : 0 ≤ z) : z ^ 2 / 2 ≤ Real.exp z - 1 := by
  have := Real.quadratic_le_exp_of_nonneg hz
  nlinarith

/-- The constant `c_s = (8 a_s² (3 · 9^{s-1})^{1/s})^{-1}`. -/
def cS (s : ℝ) : ℝ := (8 * aS s ^ 2 * (3 * 9 ^ (s - 1)) ^ (1 / s))⁻¹

/-- **`thm:scaling`, lower bound on the number of frequencies**: every maximizer uses at
least `c_s ln p` frequencies. -/
theorem card_supp_ge (hp : 3 ≤ p) {s : ℝ} (hs : 1 < s) {y : Freq p → ℝ}
    (hy : IsMaximizer (covP p) s y) : cS s * Real.log p ≤ (supp y).card := by
  have hs0 : 0 < s := by linarith
  have ha : 0 < aS s := by unfold aS; linarith
  have hlog := log_p_pos hp
  have hΓ := GammaP_pos p hp hs.le
  have hw : worst (covP p) y = GammaP p s := hy.worst_eq (covP_le_two p) hs.le
  -- the support is non-empty
  have hK0 : 0 < (supp y).card := by
    rw [Finset.card_pos]
    by_contra hE
    rw [Finset.not_nonempty_iff_eq_empty] at hE
    have hy0 : y = 0 := by
      funext j
      have : j ∉ supp y := by rw [hE]; simp
      simpa [supp] using this
    rw [hy0, worst_zero] at hw
    linarith
  set K := (supp y).card
  have hKR : (0 : ℝ) < K := by exact_mod_cast hK0
  set x := Real.log p / K with hx
  have hx0 : 0 < x := by positivity
  have hKb := Kbound hp hs hy.1 hK0 le_rfl
  have hlow := GammaP_lower9 p hp hs
  rw [← hw] at hlow
  -- combine: ½ 9^{1-s} (ln p)^{1-s} ≤ (3/2)(ln p)^{1-s} (x/(E-1))^s
  set E := Real.exp (x / (2 * aS s)) with hE
  have hE1 : x ^ 2 / (8 * aS s ^ 2) ≤ E - 1 := by
    have := exp_sub_one_ge_sq (z := x / (2 * aS s)) (by positivity)
    have e : (x / (2 * aS s)) ^ 2 / 2 = x ^ 2 / (8 * aS s ^ 2) := by field_simp; ring
    linarith
  have hEpos : 0 < E - 1 := lt_of_lt_of_le (by positivity) hE1
  have hratio : x / (E - 1) ≤ 8 * aS s ^ 2 / x := by
    rw [div_le_div_iff₀ hEpos hx0]
    have := mul_le_mul_of_nonneg_left hE1 (by positivity : (0 : ℝ) ≤ 8 * aS s ^ 2)
    have e : 8 * aS s ^ 2 * (x ^ 2 / (8 * aS s ^ 2)) = x * x := by field_simp
    nlinarith
  have hL : 0 < Real.log p ^ (1 - s) := Real.rpow_pos_of_pos hlog _
  have h9 : (1 / 2) * (9 * Real.log p) ^ (1 - s) =
      (1 / 2) * 9 ^ (1 - s) * Real.log p ^ (1 - s) := by
    rw [Real.mul_rpow (by norm_num) hlog.le]; ring
  rw [h9] at hlow
  have key : (1 / 2) * (9 : ℝ) ^ (1 - s) ≤ (3 / 2) * (8 * aS s ^ 2 / x) ^ s := by
    have h1 := hlow.trans hKb
    have h2 : (x / (E - 1)) ^ s ≤ (8 * aS s ^ 2 / x) ^ s :=
      Real.rpow_le_rpow (by positivity) hratio hs0.le
    have h3 : (1 / 2) * (9 : ℝ) ^ (1 - s) ≤ (3 / 2) * (x / (E - 1)) ^ s := by
      have : (1 / 2) * 9 ^ (1 - s) * Real.log p ^ (1 - s) ≤
          (3 / 2) * (x / (E - 1)) ^ s * Real.log p ^ (1 - s) := by linarith
      exact le_of_mul_le_mul_right this hL
    linarith
  -- so `x ≤ 8 a_s² (3·9^{s-1})^{1/s}`
  set M := 3 * (9 : ℝ) ^ (s - 1) with hM
  have hM0 : 0 < M := by positivity
  have hkey2 : M⁻¹ ≤ (8 * aS s ^ 2 / x) ^ s := by
    have e : M⁻¹ = (1 / 3) * (9 : ℝ) ^ (1 - s) := by
      rw [hM, mul_inv, ← Real.rpow_neg (by norm_num)]; ring_nf
    rw [e]; linarith
  have hkey3 : (M⁻¹) ^ (1 / s) ≤ 8 * aS s ^ 2 / x := by
    have := Real.rpow_le_rpow (by positivity) hkey2 (by positivity : (0 : ℝ) ≤ 1 / s)
    rwa [← Real.rpow_mul (by positivity), mul_one_div_cancel hs0.ne', Real.rpow_one] at this
  have hMs : 0 < M ^ (1 / s) := Real.rpow_pos_of_pos hM0 _
  have hxle : x ≤ 8 * aS s ^ 2 * M ^ (1 / s) := by
    rw [Real.inv_rpow hM0.le] at hkey3
    rw [le_div_iff₀ hx0] at hkey3
    have := mul_le_mul_of_nonneg_left hkey3 hMs.le
    rw [← mul_assoc, mul_inv_cancel₀ hMs.ne', one_mul] at this
    linarith
  -- `K = ln p / x ≥ c_s ln p`
  unfold cS
  rw [← hM]
  have hden : 0 < 8 * aS s ^ 2 * M ^ (1 / s) := by positivity
  rw [inv_mul_eq_div, div_le_iff₀ hden]
  have : Real.log p = K * x := by rw [hx]; field_simp
  rw [this]
  exact mul_le_mul_of_nonneg_left hxle hKR.le

/-- The share floor `4^{-1/(s-1)} / (9 ln p)`. -/
def shareFloor (s : ℝ) (p : ℕ) : ℝ := (4 : ℝ) ^ (-(1 / (s - 1))) / (9 * Real.log p)

/-- **`thm:scaling`, share floor**: every maximizer spends at least `4^{-1/(s-1)}/(9 ln p)` of
its budget on each frequency it uses. -/
theorem share_ge (hp : 3 ≤ p) {s : ℝ} (hs : 1 < s) {y : Freq p → ℝ}
    (hy : IsMaximizer (covP p) s y) (j : Freq p) (hj : j ∈ supp y) :
    shareFloor s p ≤ y j ^ (1 / s) := by
  classical
  have hs0 : 0 < s := by linarith
  have hs1 : 0 < s - 1 := by linarith
  have hlog := log_p_pos hp
  have hΓ := GammaP_pos p hp hs.le
  have hw : worst (covP p) y = GammaP p s := hy.worst_eq (covP_le_two p) hs.le
  have hbud := hy.budget_eq_one (covP_le_two p) hs hΓ
  set m := worst (covP p) y
  have hm0 : 0 < m := by rw [hw]; exact hΓ
  have hyj : 0 < y j := lt_of_le_of_ne (hy.1.1 j) (Ne.symm (by simpa [supp] using hj))
  set w := y j ^ (1 / s) with hwdef
  have hw0 : 0 < w := Real.rpow_pos_of_pos hyj _
  have hyw : y j = w ^ s := by
    rw [hwdef, ← Real.rpow_mul hyj.le, one_div_mul_cancel hs0.ne', Real.rpow_one]
  -- ablation with ε = 0
  obtain ⟨h1, h2, h3, -⟩ := prop_ablation p hs le_rfl (by norm_num) hy.1 hbud
    (by rw [sub_zero, one_mul]; exact hw.symm.le) j hj
  simp only [sub_zero, div_one] at h2
  -- `m w ≤ 2 y_j = 2 w^s`, so `m/2 ≤ w^{s-1}`
  have hmw : m * w ≤ 2 * w ^ s := by nlinarith
  have hws : w ^ s = w ^ (s - 1) * w := by
    rw [Real.rpow_sub hw0, Real.rpow_one]; field_simp
  have hm2 : m / 2 ≤ w ^ (s - 1) := by
    rw [hws] at hmw
    rw [div_le_iff₀ (by norm_num : (0 : ℝ) < 2)]
    nlinarith
  -- raise to `1/(s-1)`
  have hwge : (m / 2) ^ (1 / (s - 1)) ≤ w := by
    have := Real.rpow_le_rpow (by positivity) hm2 (by positivity : (0 : ℝ) ≤ 1 / (s - 1))
    rwa [← Real.rpow_mul hw0.le, mul_one_div_cancel hs1.ne', Real.rpow_one] at this
  -- `m/2 ≥ (1/4) (9 ln p)^{1-s}`
  have hlow := GammaP_lower9 p hp hs
  rw [← hw] at hlow
  have hq : 0 < 9 * Real.log p := by positivity
  have hm4 : (1 / 4) * (9 * Real.log p) ^ (1 - s) ≤ m / 2 := by linarith
  have hfl : shareFloor s p = ((1 / 4) * (9 * Real.log p) ^ (1 - s)) ^ (1 / (s - 1)) := by
    unfold shareFloor
    rw [Real.mul_rpow (by norm_num) (Real.rpow_nonneg hq.le _), ← Real.rpow_mul hq.le]
    have e1 : (1 - s) * (1 / (s - 1)) = -1 := by field_simp; ring
    rw [e1, Real.rpow_neg_one, Real.div_rpow (by norm_num) (by norm_num), Real.one_rpow,
      Real.rpow_neg (by norm_num)]
    field_simp
  rw [hfl]
  exact le_trans (Real.rpow_le_rpow (by positivity) hm4 (by positivity)) hwge

/-- **`thm:scaling`, upper bound on the number of frequencies**: every maximizer uses at most
`9 · 4^{1/(s-1)} ln p` frequencies. -/
theorem card_supp_le (hp : 3 ≤ p) {s : ℝ} (hs : 1 < s) {y : Freq p → ℝ}
    (hy : IsMaximizer (covP p) s y) :
    ((supp y).card : ℝ) ≤ 9 * (4 : ℝ) ^ (1 / (s - 1)) * Real.log p := by
  classical
  have hlog := log_p_pos hp
  have hΓ := GammaP_pos p hp hs.le
  have hbud := hy.budget_eq_one (covP_le_two p) hs hΓ
  have hfl0 : 0 < shareFloor s p := by unfold shareFloor; positivity
  have hsum : ((supp y).card : ℝ) * shareFloor s p ≤ 1 := by
    calc ((supp y).card : ℝ) * shareFloor s p = ∑ j ∈ supp y, shareFloor s p := by
          rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ ∑ j ∈ supp y, y j ^ (1 / s) := Finset.sum_le_sum fun j hj => share_ge hp hs hy j hj
      _ ≤ ∑ j, y j ^ (1 / s) := Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
          fun j _ _ => Real.rpow_nonneg (hy.1.1 j) _
      _ = 1 := hbud
  have e : 9 * (4 : ℝ) ^ (1 / (s - 1)) * Real.log p = (shareFloor s p)⁻¹ := by
    unfold shareFloor
    rw [Real.rpow_neg (by norm_num)]
    field_simp
  rw [e]
  calc ((supp y).card : ℝ) = ((supp y).card : ℝ) * shareFloor s p * (shareFloor s p)⁻¹ := by
        field_simp
    _ ≤ 1 * (shareFloor s p)⁻¹ := mul_le_mul_of_nonneg_right hsum (inv_nonneg.2 hfl0.le)
    _ = (shareFloor s p)⁻¹ := one_mul _

/-- Share floor in terms of `Γ` itself: `w_j ≥ (Γ_s(p)/2)^{1/(s-1)}` for `j ∈ spt(y)`. -/
theorem share_ge_gamma (hp : 3 ≤ p) {s : ℝ} (hs : 1 < s) {y : Freq p → ℝ}
    (hy : IsMaximizer (covP p) s y) (j : Freq p) (hj : j ∈ supp y) :
    (GammaP p s / 2) ^ (1 / (s - 1)) ≤ y j ^ (1 / s) := by
  have hs0 : 0 < s := by linarith
  have hs1 : 0 < s - 1 := by linarith
  have hΓ := GammaP_pos p hp hs.le
  have hw : worst (covP p) y = GammaP p s := hy.worst_eq (covP_le_two p) hs.le
  have hbud := hy.budget_eq_one (covP_le_two p) hs hΓ
  have hyj : 0 < y j := lt_of_le_of_ne (hy.1.1 j) (Ne.symm (by simpa [supp] using hj))
  set w := y j ^ (1 / s) with hwdef
  have hw0 : 0 < w := Real.rpow_pos_of_pos hyj _
  have hyw : y j = w ^ s := by
    rw [hwdef, ← Real.rpow_mul hyj.le, one_div_mul_cancel hs0.ne', Real.rpow_one]
  obtain ⟨h1, h2, h3, -⟩ := prop_ablation p hs le_rfl (by norm_num) hy.1 hbud
    (by rw [sub_zero, one_mul]; exact hw.symm.le) j hj
  simp only [sub_zero, div_one] at h2
  rw [hw] at h1 h2
  have hmw : GammaP p s * w ≤ 2 * w ^ s := by nlinarith
  have hws : w ^ s = w ^ (s - 1) * w := by
    rw [Real.rpow_sub hw0, Real.rpow_one]; field_simp
  have hm2 : GammaP p s / 2 ≤ w ^ (s - 1) := by
    rw [hws] at hmw
    rw [div_le_iff₀ (by norm_num : (0 : ℝ) < 2)]
    nlinarith
  have := Real.rpow_le_rpow (by positivity) hm2 (by positivity : (0 : ℝ) ≤ 1 / (s - 1))
  rwa [← Real.rpow_mul hw0.le, mul_one_div_cancel hs1.ne', Real.rpow_one] at this

/-- Number of frequencies of a maximizer, sharper form used in the paper's remark on the
dense regime: `|spt(y)| ≤ 4^{1/(s-1)} ⌈8 ln p⌉`. -/
theorem card_supp_le_ceil (hp : 3 ≤ p) {s : ℝ} (hs : 1 < s) {y : Freq p → ℝ}
    (hy : IsMaximizer (covP p) s y) :
    ((supp y).card : ℝ) ≤ (4 : ℝ) ^ (1 / (s - 1)) * (⌈8 * Real.log p⌉₊ : ℝ) := by
  classical
  have hs1 : 0 < s - 1 := by linarith
  have hΓ := GammaP_pos p hp hs.le
  have hbud := hy.budget_eq_one (covP_le_two p) hs hΓ
  have hlog := log_p_pos hp
  set L := (⌈8 * Real.log p⌉₊ : ℝ) with hL
  have hL0 : 0 < L := by
    have : 0 < ⌈8 * Real.log p⌉₊ := Nat.ceil_pos.2 (by positivity)
    rw [hL]; exact_mod_cast this
  set f := (GammaP p s / 2) ^ (1 / (s - 1)) with hf
  have hf0 : 0 < f := Real.rpow_pos_of_pos (by positivity) _
  have hsum : ((supp y).card : ℝ) * f ≤ 1 := by
    calc ((supp y).card : ℝ) * f = ∑ j ∈ supp y, f := by rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ ∑ j ∈ supp y, y j ^ (1 / s) :=
          Finset.sum_le_sum fun j hj => share_ge_gamma hp hs hy j hj
      _ ≤ ∑ j, y j ^ (1 / s) := Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
          fun j _ _ => Real.rpow_nonneg (hy.1.1 j) _
      _ = 1 := hbud
  -- `f ≥ 4^{-1/(s-1)} / L`
  have hlow := GammaP_lower p hp hs
  have hm4 : (1 / 4) * L ^ (1 - s) ≤ GammaP p s / 2 := by rw [← hL] at hlow; linarith
  have hfl : ((1 / 4) * L ^ (1 - s)) ^ (1 / (s - 1)) = ((4 : ℝ) ^ (1 / (s - 1)) * L)⁻¹ := by
    rw [Real.mul_rpow (by norm_num) (Real.rpow_nonneg hL0.le _), ← Real.rpow_mul hL0.le]
    have e1 : (1 - s) * (1 / (s - 1)) = -1 := by field_simp; ring
    rw [e1, Real.rpow_neg_one, Real.div_rpow (by norm_num) (by norm_num), Real.one_rpow,
      mul_inv]
    field_simp
  have hfge : ((4 : ℝ) ^ (1 / (s - 1)) * L)⁻¹ ≤ f := by
    rw [← hfl]; exact Real.rpow_le_rpow (by positivity) hm4 (by positivity)
  have hD : 0 < (4 : ℝ) ^ (1 / (s - 1)) * L := by positivity
  have := le_trans (mul_le_mul_of_nonneg_left hfge (by positivity)) hsum
  rw [← div_eq_mul_inv, div_le_one hD] at this
  exact this

/-- **Remark after `cor:few` (narrowness of the dense regime).** If a maximizer uses all
`(p-1)/2` frequencies and `(p-1)/(2⌈8 ln p⌉) > 1`, then `s - 1 ≤ ln 4 / ln((p-1)/(2⌈8 ln p⌉))`. -/
theorem dense_only_if (hp : 3 ≤ p) {s : ℝ} (hs : 1 < s) {y : Freq p → ℝ}
    (hy : IsMaximizer (covP p) s y) (hall : (supp y).card = (p - 1) / 2)
    (hbig : 1 < ((p : ℝ) - 1) / (2 * (⌈8 * Real.log p⌉₊ : ℝ))) :
    s - 1 ≤ Real.log 4 / Real.log (((p : ℝ) - 1) / (2 * (⌈8 * Real.log p⌉₊ : ℝ))) := by
  have hs1 : 0 < s - 1 := by linarith
  have hlog := log_p_pos hp
  set L := (⌈8 * Real.log p⌉₊ : ℝ) with hL
  have hL0 : 0 < L := by
    have : 0 < ⌈8 * Real.log p⌉₊ := Nat.ceil_pos.2 (by positivity)
    rw [hL]; exact_mod_cast this
  have hK := card_supp_le_ceil hp hs hy
  rw [hall] at hK
  have hodd : p % 2 = 1 := Nat.odd_iff.1 ((Fact.out : p.Prime).odd_of_ne_two (by omega))
  have hcast : (((p - 1) / 2 : ℕ) : ℝ) = ((p : ℝ) - 1) / 2 := by
    have h2 : (p - 1) / 2 * 2 = p - 1 := by omega
    have h3 : ((((p - 1) / 2 * 2 : ℕ)) : ℝ) = (p : ℝ) - 1 := by
      rw [h2, Nat.cast_sub (by omega : 1 ≤ p)]; simp
    push_cast at h3
    linarith
  rw [hcast] at hK
  have hq : ((p : ℝ) - 1) / (2 * L) ≤ (4 : ℝ) ^ (1 / (s - 1)) := by
    rw [div_le_iff₀ (by positivity)]; nlinarith
  have hlq := Real.log_le_log (by linarith) hq
  rw [Real.log_rpow (by norm_num)] at hlq
  have hpos : 0 < Real.log (((p : ℝ) - 1) / (2 * L)) := Real.log_pos hbig
  rw [le_div_iff₀ hpos]
  have := mul_le_mul_of_nonneg_left hlq hs1.le
  have e : (s - 1) * (1 / (s - 1) * Real.log 4) = Real.log 4 := by field_simp
  linarith

end Heracles
