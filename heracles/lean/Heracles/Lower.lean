import Heracles.Defs

/-!
# Random-frequency lower bound for `Γ_s(p)` (Theorem `thm:scaling`, lower half)

Paper, appendix `app:scaling`, paragraph "Lower bound":
`Γ_s(p) ≥ ½ ⌈8 ln p⌉^{1-s}` and `Γ_s(p) ≥ ½ (9 ln p)^{1-s}` for odd primes `p` and `s > 1`,
attained by an allocation with at most `⌈8 ln p⌉` frequencies.

The paper draws `L = ⌈8 ln p⌉` frequencies independently and uniformly and uses Hoeffding's
inequality plus a union bound.  We give the same argument in its deterministic counting form:
the uniform probability of a set of tuples `J ∈ F_p^L` is its cardinality divided by `|F_p|^L`,
Hoeffding's bound is proved by the Chernoff/Markov argument with the elementary form of
Hoeffding's lemma (`e^{λx} ≤ cosh λ + x sinh λ` on `[-1,1]`, `cosh λ ≤ e^{λ²/2}`), and the union
bound is `card (⋃ B_δ) ≤ ∑ card B_δ`.
-/

open Real Finset

noncomputable section

namespace Heracles

/-! ## The trigonometric sum `∑_{j ∈ F_p} cos(2π j δ / p) = -1/2` -/

/-- `∑_{j < p} cos(2π j d / p) = 0` for `0 < d < p`. -/
lemma sum_range_cos_eq_zero (p : ℕ) (hp : 0 < p) (d : ℕ) (hd0 : 0 < d) (hdp : d < p) :
    ∑ j ∈ Finset.range p, Real.cos (2 * π * (j : ℝ) * (d : ℝ) / p) = 0 := by
  set ζ : ℂ := Complex.exp (2 * π * Complex.I * d / p) with hζ
  have hp' : (p : ℂ) ≠ 0 := by exact_mod_cast hp.ne'
  have h2πI : (2 * π * Complex.I : ℂ) ≠ 0 := by
    simp [Real.pi_ne_zero, Complex.I_ne_zero]
  have hζp : ζ ^ p = 1 := by
    rw [hζ, ← Complex.exp_nat_mul, Complex.exp_eq_one_iff]
    refine ⟨d, ?_⟩
    field_simp
    push_cast
    ring
  have hζ1 : ζ ≠ 1 := by
    intro h
    rw [hζ, Complex.exp_eq_one_iff] at h
    obtain ⟨n, hn⟩ := h
    have hd : (d : ℂ) = n * p := by
      field_simp at hn
      linear_combination hn
    have hdz : (d : ℤ) = n * p := by exact_mod_cast hd
    have hn0 : 0 < n := by
      by_contra hcon
      have : n ≤ 0 := not_lt.1 hcon
      have : (d : ℤ) ≤ 0 := by rw [hdz]; nlinarith
      omega
    have : (p : ℤ) ≤ d := by rw [hdz]; nlinarith
    omega
  have hsum : ∑ j ∈ Finset.range p, ζ ^ j = 0 := by
    rw [geom_sum_eq hζ1, hζp]; simp
  have hre := congrArg Complex.re hsum
  rw [Complex.re_sum, Complex.zero_re] at hre
  rw [← hre]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [hζ, ← Complex.exp_nat_mul]
  have : (j : ℂ) * (2 * π * Complex.I * d / p) = ((2 * π * j * d / p : ℝ) : ℂ) * Complex.I := by
    push_cast; ring
  rw [this, Complex.exp_ofReal_mul_I_re]

/-- The coverage sum `∑_{j ∈ F_p} cos(2π j δ / p) = -1/2` for `δ ≠ 0`. -/
theorem sum_cos_freq (p : ℕ) [Fact p.Prime] (hp : 3 ≤ p) (δ : ZMod p) (hδ : δ ≠ 0) :
    ∑ j : Freq p, Real.cos (2 * π * (j.1 : ℝ) * (δ.val : ℝ) / p) = -1 / 2 := by
  have hprime : p.Prime := Fact.out
  have hodd : p % 2 = 1 := Nat.odd_iff.1 (hprime.odd_of_ne_two (by omega))
  set m := (p - 1) / 2 with hm
  have hpm : p = 2 * m + 1 := by omega
  set d := δ.val with hd
  have hd0 : 0 < d := by
    rcases Nat.eq_zero_or_pos d with h | h
    · exact absurd ((ZMod.val_eq_zero δ).1 h) hδ
    · exact h
  have hdp : d < p := ZMod.val_lt δ
  set g : ℕ → ℝ := fun j => Real.cos (2 * π * (j : ℝ) * (d : ℝ) / p) with hg
  have hzero := sum_range_cos_eq_zero p (by omega) d hd0 hdp
  -- the sum over `F_p` is the sum over `Ico 1 (m+1)`
  have hF : ∑ j : Freq p, Real.cos (2 * π * (j.1 : ℝ) * (δ.val : ℝ) / p)
      = ∑ j ∈ Finset.Ico 1 (m + 1), g j := by
    rw [Finset.Ico_add_one_right_eq_Icc]
    exact Finset.sum_coe_sort (Finset.Icc 1 ((p - 1) / 2)) g
  -- reflection `j ↦ p - j`
  have hrefl : ∑ j ∈ Finset.Ico (m + 1) p, g j = ∑ j ∈ Finset.Ico 1 (m + 1), g j := by
    have h := Finset.sum_Ico_reflect g 1 (m := m + 1) (n := p) (by omega)
    have e1 : p + 1 - (m + 1) = m + 1 := by omega
    have e2 : p + 1 - 1 = p := by omega
    rw [e1, e2] at h
    rw [← h]
    refine Finset.sum_congr rfl fun j hj => ?_
    have hjp : j ≤ p := by
      have := (Finset.mem_Ico.1 hj).2; omega
    simp only [hg]
    rw [Nat.cast_sub hjp]
    have : 2 * π * ((p : ℝ) - j) * d / p = (d : ℕ) * (2 * π) - 2 * π * j * d / p := by
      have hp0 : (p : ℝ) ≠ 0 := by exact_mod_cast (by omega : p ≠ 0)
      field_simp
    rw [this, Real.cos_nat_mul_two_pi_sub]
  have hsplit : ∑ j ∈ Finset.range p, g j
      = g 0 + ∑ j ∈ Finset.Ico 1 (m + 1), g j + ∑ j ∈ Finset.Ico (m + 1) p, g j := by
    rw [Finset.range_eq_Ico, ← Finset.sum_Ico_consecutive g (m := 0) (n := 1) (k := p)
      (by omega) (by omega), ← Finset.sum_Ico_consecutive g (m := 1) (n := m + 1) (k := p)
      (by omega) (by omega)]
    simp
    ring
  have hg0 : g 0 = 1 := by simp [hg]
  rw [hF]
  rw [hsplit, hrefl, hg0] at hzero
  linarith

/-! ## Chernoff bound for one offset (Hoeffding's lemma, elementary form) -/

/-- Convexity of `exp` on `[-λ, λ]`: `e^{λx} ≤ cosh λ + x sinh λ` for `x ∈ [-1, 1]`. -/
lemma exp_mul_le_convex (lam x : ℝ) (hx1 : -1 ≤ x) (hx2 : x ≤ 1) :
    Real.exp (lam * x) ≤ (Real.exp lam + Real.exp (-lam)) / 2
      + x * ((Real.exp lam - Real.exp (-lam)) / 2) := by
  have h := convexOn_exp.2 (Set.mem_univ lam) (Set.mem_univ (-lam))
    (by linarith : (0 : ℝ) ≤ (1 + x) / 2) (by linarith : (0 : ℝ) ≤ (1 - x) / 2) (by ring)
  simp only [smul_eq_mul] at h
  have e : (1 + x) / 2 * lam + (1 - x) / 2 * -lam = lam * x := by ring
  rw [e] at h
  linarith

/-- The cosines `cos(2π j δ / p)` of the frequencies at offset `δ`. -/
def fcos (p : ℕ) (δ : ZMod p) (j : Freq p) : ℝ := Real.cos (2 * π * (j.1 : ℝ) * (δ.val : ℝ) / p)

lemma covP_eq_one_sub (p : ℕ) (δ : Off p) (j : Freq p) : covP p δ j = 1 - fcos p δ.1 j := rfl

/-- Moment generating bound: `∑_{j ∈ F_p} e^{λ cos(2π j δ/p)} ≤ |F_p| e^{λ²/2}` (uses that the
mean of the cosines is `-1/(p-1) ≤ 0`). -/
lemma sum_exp_fcos_le (p : ℕ) [Fact p.Prime] (hp : 3 ≤ p) (δ : Off p) {lam : ℝ}
    (hlam : 0 ≤ lam) :
    ∑ j : Freq p, Real.exp (lam * fcos p δ.1 j)
      ≤ (Fintype.card (Freq p) : ℝ) * Real.exp (lam ^ 2 / 2) := by
  have hab : Real.exp (-lam) ≤ Real.exp lam := Real.exp_le_exp.2 (by linarith)
  have hsum : ∑ j : Freq p, fcos p δ.1 j = -1 / 2 := sum_cos_freq p hp δ.1 δ.2
  calc ∑ j : Freq p, Real.exp (lam * fcos p δ.1 j)
      ≤ ∑ j : Freq p, ((Real.exp lam + Real.exp (-lam)) / 2
          + fcos p δ.1 j * ((Real.exp lam - Real.exp (-lam)) / 2)) :=
        Finset.sum_le_sum fun j _ =>
          exp_mul_le_convex lam _ (Real.neg_one_le_cos _) (Real.cos_le_one _)
    _ = (Fintype.card (Freq p) : ℝ) * ((Real.exp lam + Real.exp (-lam)) / 2)
          + (∑ j : Freq p, fcos p δ.1 j) * ((Real.exp lam - Real.exp (-lam)) / 2) := by
        rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul,
          Finset.sum_mul]
    _ = (Fintype.card (Freq p) : ℝ) * Real.cosh lam
          - (Real.exp lam - Real.exp (-lam)) / 4 := by
        rw [hsum, Real.cosh_eq]; ring
    _ ≤ (Fintype.card (Freq p) : ℝ) * Real.cosh lam := by linarith
    _ ≤ (Fintype.card (Freq p) : ℝ) * Real.exp (lam ^ 2 / 2) := by
        gcongr; exact Real.cosh_le_exp_half_sq lam

/-- Chernoff/Hoeffding for one offset: at most `|F_p|^L e^{-L/8}` tuples `J ∈ F_p^L` have
`∑_l cos(2π J_l δ/p) > L/2`.  (In probabilistic language: for `L` i.i.d. uniform frequencies,
`Pr[(1/L) ∑_l cos(2π J_l δ/p) > 1/2] ≤ e^{-L/8}`.) -/
lemma card_bad_le (p : ℕ) [Fact p.Prime] (hp : 3 ≤ p) (L : ℕ) (δ : Off p) :
    ((Finset.univ.filter
        (fun J : Fin L → Freq p => (L : ℝ) / 2 < ∑ l, fcos p δ.1 (J l))).card : ℝ)
      ≤ (Fintype.card (Freq p) : ℝ) ^ L * Real.exp (-(L : ℝ) / 8) := by
  classical
  have step1 : ((Finset.univ.filter
        (fun J : Fin L → Freq p => (L : ℝ) / 2 < ∑ l, fcos p δ.1 (J l))).card : ℝ)
      ≤ ∑ J : Fin L → Freq p, Real.exp ((1 / 2) * (∑ l, fcos p δ.1 (J l) - L / 2)) := by
    rw [Finset.card_eq_sum_ones, Nat.cast_sum, Nat.cast_one]
    calc ∑ J ∈ Finset.univ.filter
          (fun J : Fin L → Freq p => (L : ℝ) / 2 < ∑ l, fcos p δ.1 (J l)), (1 : ℝ)
        ≤ ∑ J ∈ Finset.univ.filter
          (fun J : Fin L → Freq p => (L : ℝ) / 2 < ∑ l, fcos p δ.1 (J l)),
            Real.exp ((1 / 2) * (∑ l, fcos p δ.1 (J l) - L / 2)) :=
          Finset.sum_le_sum fun J hJ => by
            have := (Finset.mem_filter.1 hJ).2
            exact Real.one_le_exp (mul_nonneg (by norm_num) (by linarith))
      _ ≤ ∑ J : Fin L → Freq p, Real.exp ((1 / 2) * (∑ l, fcos p δ.1 (J l) - L / 2)) :=
          Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _)
            (fun _ _ _ => (Real.exp_pos _).le)
  have step2 : ∑ J : Fin L → Freq p, Real.exp ((1 / 2) * (∑ l, fcos p δ.1 (J l) - L / 2))
      = Real.exp (-(L : ℝ) / 4) * (∑ j : Freq p, Real.exp ((1 / 2) * fcos p δ.1 j)) ^ L := by
    have h1 : ∀ J : Fin L → Freq p,
        Real.exp ((1 / 2) * (∑ l, fcos p δ.1 (J l) - L / 2))
          = Real.exp (-(L : ℝ) / 4) * ∏ l, Real.exp ((1 / 2) * fcos p δ.1 (J l)) := by
      intro J
      rw [← Real.exp_sum, ← Real.exp_add, ← Finset.mul_sum]
      congr 1
      ring
    simp_rw [h1]
    rw [← Finset.mul_sum]
    congr 1
    rw [← Fintype.prod_sum (fun (_ : Fin L) (j : Freq p) => Real.exp ((1 / 2) * fcos p δ.1 j))]
    rw [Finset.prod_const, Finset.card_univ, Fintype.card_fin]
  have step3 := sum_exp_fcos_le p hp δ (lam := 1 / 2) (by norm_num)
  calc _ ≤ _ := step1
    _ = _ := step2
    _ ≤ Real.exp (-(L : ℝ) / 4)
          * ((Fintype.card (Freq p) : ℝ) * Real.exp ((1 / 2) ^ 2 / 2)) ^ L := by
        gcongr
    _ = (Fintype.card (Freq p) : ℝ) ^ L * Real.exp (-(L : ℝ) / 8) := by
        rw [mul_pow, ← Real.exp_nat_mul, mul_comm, mul_assoc, ← Real.exp_add]
        congr 2
        ring

/-! ## Union bound: a good draw exists -/

/-- There is a tuple of `L ≥ 8 ln p` frequencies covering every non-zero offset by at least
`L/2`. -/
theorem exists_good_draw (p : ℕ) [Fact p.Prime] (hp : 3 ≤ p) (L : ℕ)
    (hL : 8 * Real.log p ≤ L) :
    ∃ J : Fin L → Freq p, ∀ δ : Off p, (L : ℝ) / 2 ≤ ∑ l, covP p δ (J l) := by
  classical
  by_contra hcon
  simp only [not_exists, not_forall, not_le] at hcon
  set B : Off p → Finset (Fin L → Freq p) := fun δ =>
    Finset.univ.filter (fun J : Fin L → Freq p => (L : ℝ) / 2 < ∑ l, fcos p δ.1 (J l)) with hB
  have hcover : (Finset.univ : Finset (Fin L → Freq p)) ⊆ Finset.univ.biUnion B := by
    intro J _
    obtain ⟨δ, hδ⟩ := hcon J
    rw [Finset.mem_biUnion]
    refine ⟨δ, Finset.mem_univ _, ?_⟩
    rw [hB, Finset.mem_filter]
    refine ⟨Finset.mem_univ _, ?_⟩
    have : ∑ l, covP p δ (J l) = L - ∑ l, fcos p δ.1 (J l) := by
      simp only [covP_eq_one_sub, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ,
        Fintype.card_fin, nsmul_eq_mul, mul_one]
    linarith
  have hcard := (Finset.card_le_card hcover).trans Finset.card_biUnion_le
  have hR : ((Finset.univ : Finset (Fin L → Freq p)).card : ℝ) ≤ ∑ δ, ((B δ).card : ℝ) := by
    exact_mod_cast hcard
  rw [Finset.card_univ, Fintype.card_fun, Fintype.card_fin] at hR
  have hN : 0 < Fintype.card (Freq p) := Fintype.card_pos_iff.2 ⟨freqOne p hp⟩
  have hA : (0 : ℝ) < (Fintype.card (Freq p) : ℝ) ^ L := by positivity
  have hp0 : (0 : ℝ) < p := by exact_mod_cast (by omega : 0 < p)
  have hexp : Real.exp (-(L : ℝ) / 8) ≤ 1 / p := by
    have : -(L : ℝ) / 8 ≤ -Real.log p := by linarith
    calc Real.exp (-(L : ℝ) / 8) ≤ Real.exp (-Real.log p) := Real.exp_le_exp.2 this
      _ = 1 / p := by rw [Real.exp_neg, Real.exp_log hp0, one_div]
  have hOff : Fintype.card (Off p) < p := by
    have := Fintype.card_subtype_lt (p := fun δ : ZMod p => δ ≠ 0) (x := 0) (by simp)
    rwa [ZMod.card] at this
  have hOffR : (Fintype.card (Off p) : ℝ) < p := by exact_mod_cast hOff
  have hsumB : ∑ δ, ((B δ).card : ℝ)
      ≤ (Fintype.card (Off p) : ℝ) * ((Fintype.card (Freq p) : ℝ) ^ L * (1 / p)) := by
    calc ∑ δ, ((B δ).card : ℝ)
        ≤ ∑ _δ : Off p, (Fintype.card (Freq p) : ℝ) ^ L * (1 / p) :=
          Finset.sum_le_sum fun δ _ =>
            (card_bad_le p hp L δ).trans (mul_le_mul_of_nonneg_left hexp hA.le)
      _ = _ := by rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  have hlt : (Fintype.card (Off p) : ℝ) * ((Fintype.card (Freq p) : ℝ) ^ L * (1 / p))
      < (p : ℝ) * ((Fintype.card (Freq p) : ℝ) ^ L * (1 / p)) :=
    mul_lt_mul_of_pos_right hOffR (by positivity)
  have heq : (p : ℝ) * ((Fintype.card (Freq p) : ℝ) ^ L * (1 / p))
      = (Fintype.card (Freq p) : ℝ) ^ L := by
    field_simp
  push_cast at hR
  linarith

/-! ## The allocation and the lower bound on `Γ_s(p)` -/

lemma natCast_rpow_le_self (k : ℕ) {r : ℝ} (hr0 : 0 < r) (hr1 : r ≤ 1) :
    (k : ℝ) ^ r ≤ k := by
  rcases Nat.eq_zero_or_pos k with h | h
  · subst h; simp [Real.zero_rpow hr0.ne']
  · have hk : (1 : ℝ) ≤ k := by exact_mod_cast h
    calc (k : ℝ) ^ r ≤ (k : ℝ) ^ (1 : ℝ) := Real.rpow_le_rpow_of_exponent_le hk hr1
      _ = k := Real.rpow_one _

/-- The random allocation: with `L = ⌈8 ln p⌉` and a good draw `J`, the allocation
`y_j = n_j L^{-s}` (`n_j` the multiplicity of `j` in `J`) is feasible, uses at most `L`
frequencies and has worst coverage at least `½ L^{1-s}`. -/
theorem exists_alloc_lower (p : ℕ) [Fact p.Prime] (hp : 3 ≤ p) {s : ℝ} (hs : 1 < s) :
    ∃ y : Freq p → ℝ, Feasible s y ∧ (supp y).card ≤ ⌈8 * Real.log p⌉₊ ∧
      (1 / 2) * (⌈8 * Real.log p⌉₊ : ℝ) ^ (1 - s) ≤ worst (covP p) y := by
  classical
  set L := ⌈8 * Real.log p⌉₊ with hLdef
  have hlogp : 0 < Real.log p := Real.log_pos (by exact_mod_cast (by omega : 1 < p))
  have hLpos : 0 < L := Nat.ceil_pos.2 (by positivity)
  have hLR : (0 : ℝ) < L := by exact_mod_cast hLpos
  obtain ⟨J, hJ⟩ := exists_good_draw p hp L (Nat.le_ceil _)
  set n : Freq p → ℕ := fun j => (Finset.univ.filter (fun l => J l = j)).card with hn
  set y : Freq p → ℝ := fun j => (n j : ℝ) / (L : ℝ) ^ s with hy
  have hLs : (0 : ℝ) < (L : ℝ) ^ s := Real.rpow_pos_of_pos hLR s
  have hsum_n : ∑ j, (n j : ℝ) = L := by
    have := Finset.card_eq_sum_card_fiberwise (f := J) (s := (Finset.univ : Finset (Fin L)))
      (t := Finset.univ) (fun _ _ => Finset.mem_coe.2 (Finset.mem_univ _))
    rw [Finset.card_univ, Fintype.card_fin] at this
    rw [hn]
    exact_mod_cast this.symm
  have hs0 : s ≠ 0 := by linarith
  refine ⟨y, ⟨fun j => div_nonneg (Nat.cast_nonneg _) hLs.le, ?_⟩, ?_, ?_⟩
  · -- the budget
    unfold budget
    calc ∑ j, y j ^ (1 / s) = ∑ j, (n j : ℝ) ^ (1 / s) / L := by
          refine Finset.sum_congr rfl fun j _ => ?_
          rw [hy]
          dsimp only
          rw [Real.div_rpow (Nat.cast_nonneg _) hLs.le, one_div, Real.rpow_rpow_inv hLR.le hs0]
      _ ≤ ∑ j, (n j : ℝ) / L := by
          gcongr with j
          exact natCast_rpow_le_self _ (by positivity) (by rw [div_le_one (by linarith)]; linarith)
      _ = 1 := by rw [← Finset.sum_div, hsum_n, div_self hLR.ne']
  · -- the support
    calc (supp y).card ≤ (Finset.univ.image J).card := by
          refine Finset.card_le_card fun j hj => ?_
          simp only [supp, Finset.mem_filter, Finset.mem_univ, true_and] at hj
          rw [Finset.mem_image]
          have hn0 : n j ≠ 0 := by
            intro h0
            apply hj
            simp [hy, h0]
          obtain ⟨l, hl⟩ := Finset.card_pos.1 (Nat.pos_of_ne_zero hn0)
          exact ⟨l, Finset.mem_univ _, (Finset.mem_filter.1 hl).2⟩
      _ ≤ (Finset.univ : Finset (Fin L)).card := Finset.card_image_le
      _ = L := by simp
  · -- the coverage
    rw [le_worst_iff]
    intro δ
    have hcov : cover (covP p) y δ = (∑ l, covP p δ (J l)) / (L : ℝ) ^ s := by
      unfold cover
      rw [← Finset.sum_fiberwise Finset.univ J (fun l => covP p δ (J l)), Finset.sum_div]
      refine Finset.sum_congr rfl fun j _ => ?_
      have : ∑ l ∈ Finset.univ.filter (fun l => J l = j), covP p δ (J l)
          = ∑ l ∈ Finset.univ.filter (fun l => J l = j), covP p δ j :=
        Finset.sum_congr rfl fun l hl => by rw [(Finset.mem_filter.1 hl).2]
      rw [this, Finset.sum_const, nsmul_eq_mul, hy]
      ring
    rw [hcov, Real.rpow_sub hLR, Real.rpow_one]
    calc 1 / 2 * ((L : ℝ) / (L : ℝ) ^ s) = ((L : ℝ) / 2) / (L : ℝ) ^ s := by ring
      _ ≤ (∑ l, covP p δ (J l)) / (L : ℝ) ^ s := by gcongr; exact hJ δ

/-- Theorem `thm:scaling`, lower bound: `½ ⌈8 ln p⌉^{1-s} ≤ Γ_s(p)`. -/
theorem GammaP_lower (p : ℕ) [Fact p.Prime] (hp : 3 ≤ p) {s : ℝ} (hs : 1 < s) :
    (1 / 2) * (⌈8 * Real.log p⌉₊ : ℝ) ^ (1 - s) ≤ GammaP p s := by
  obtain ⟨y, hf, -, hw⟩ := exists_alloc_lower p hp hs
  exact hw.trans (le_Gam (covP_le_two p) hs.le hf)

/-- `⌈8 ln p⌉ ≤ 9 ln p` for `p ≥ 3`. -/
lemma ceil_le_nine_log (p : ℕ) (hp : 3 ≤ p) :
    (⌈8 * Real.log p⌉₊ : ℝ) ≤ 9 * Real.log p := by
  have h3 : (1 : ℝ) < Real.log 3 := by
    rw [Real.lt_log_iff_exp_lt (by norm_num)]
    have := Real.exp_one_lt_d9
    linarith
  have hlog : Real.log 3 ≤ Real.log p :=
    Real.log_le_log (by norm_num) (by exact_mod_cast hp)
  have hceil := Nat.ceil_lt_add_one (a := 8 * Real.log p) (by linarith)
  linarith

/-- Theorem `thm:scaling`, lower bound in the form used later: `½ (9 ln p)^{1-s} ≤ Γ_s(p)`. -/
theorem GammaP_lower9 (p : ℕ) [Fact p.Prime] (hp : 3 ≤ p) {s : ℝ} (hs : 1 < s) :
    (1 / 2) * (9 * Real.log p) ^ (1 - s) ≤ GammaP p s := by
  have hlogp : 0 < Real.log p := Real.log_pos (by exact_mod_cast (by omega : 1 < p))
  have hLpos : 0 < ⌈8 * Real.log p⌉₊ := Nat.ceil_pos.2 (by positivity)
  have hLR : (0 : ℝ) < ⌈8 * Real.log p⌉₊ := by exact_mod_cast hLpos
  refine le_trans ?_ (GammaP_lower p hp hs)
  gcongr (1 / 2) * ?_
  exact Real.rpow_le_rpow_of_nonpos hLR (ceil_le_nine_log p hp) (by linarith)

end Heracles
