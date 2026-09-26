import Heracles.Defs

/-!
# Proposition `prop:shapes` (ii) and (iii): exponential sums and finitely many switches

Paper, appendix `app:shape`, proof of Proposition `prop:shapes`:

* (ii) a non-trivial exponential sum `∑_{k<N} c_k e^{λ_k u}` with distinct `λ_k` has at most
  `N - 1` real zeros (`expsum_card_zeros_le`);
* (iii) for a finite set `V` of non-negative vectors, the set of minimizers of
  `φ_s(v) = ∑_j v_j^{1/s}` over `V` is constant between consecutive points of a finite set of
  values of `s` (`switches_finite`); vectors whose non-zero entries agree as multisets are tied
  for every `s` (`ties_of_multiset_eq`).
-/

open Real Finset

noncomputable section

namespace Heracles

/-- The exponential sum `u ↦ ∑_k c_k e^{l_k u}` over a finite index type. -/
def expSum {κ : Type*} [Fintype κ] (c l : κ → ℝ) (u : ℝ) : ℝ := ∑ k, c k * Real.exp (l k * u)

lemma hasDerivAt_expSum {κ : Type*} [Fintype κ] (c l : κ → ℝ) (u : ℝ) :
    HasDerivAt (expSum c l) (expSum (fun k => c k * l k) l u) u := by
  have h : ∀ k ∈ (univ : Finset κ), HasDerivAt (fun u => c k * Real.exp (l k * u))
      (c k * l k * Real.exp (l k * u)) u := by
    intro k _
    have h1 : HasDerivAt (fun u => l k * u) (l k) u := by
      simpa using (hasDerivAt_id u).const_mul (l k)
    have h3 := h1.exp.const_mul (c k)
    convert h3 using 1
    ring
  have := HasDerivAt.sum h
  unfold expSum
  convert this using 1
  ext u
  simp

lemma differentiable_expSum {κ : Type*} [Fintype κ] (c l : κ → ℝ) :
    Differentiable ℝ (expSum c l) := fun u => (hasDerivAt_expSum c l u).differentiableAt

lemma continuous_expSum {κ : Type*} [Fintype κ] (c l : κ → ℝ) :
    Continuous (expSum c l) := (differentiable_expSum c l).continuous

/-- Rolle counting: between two consecutive zeros of a differentiable `f` there is a zero of
`f'`, so a finite set of zeros of `f` has at most one more element than some finite set of
zeros of `f'`. -/
lemma card_zeros_le_card_zeros_deriv (f : ℝ → ℝ) (hf : Differentiable ℝ f) (Z : Finset ℝ)
    (hZ : ∀ u ∈ Z, f u = 0) :
    ∃ T : Finset ℝ, (∀ u ∈ T, deriv f u = 0) ∧ Z.card ≤ T.card + 1 := by
  classical
  let F : ℝ × ℝ → ℝ := fun q => if h : q.1 < q.2 ∧ f q.1 = f q.2 then
      Classical.choose (exists_deriv_eq_zero h.1 hf.continuous.continuousOn h.2) else 0
  refine ⟨((Z ×ˢ Z).image F).filter (fun z => deriv f z = 0),
    fun u hu => (Finset.mem_filter.1 hu).2, ?_⟩
  refine Finset.card_le_of_interleaved fun x hx y hy hxy _ => ?_
  have h : x < y ∧ f x = f y := ⟨hxy, by rw [hZ x hx, hZ y hy]⟩
  have hspec := Classical.choose_spec (exists_deriv_eq_zero h.1 hf.continuous.continuousOn h.2)
  have hF : F (x, y) = Classical.choose
      (exists_deriv_eq_zero h.1 hf.continuous.continuousOn h.2) := dite_eq_left h
  refine ⟨F (x, y), ?_, ?_, ?_⟩
  · rw [Finset.mem_filter]
    refine ⟨Finset.mem_image.2 ⟨(x, y), Finset.mem_product.2 ⟨hx, hy⟩, rfl⟩, ?_⟩
    rw [hF]; exact hspec.2
  · rw [hF]; exact hspec.1.1
  · rw [hF]; exact hspec.1.2

/-- Splitting off the first exponent: `∑_k c_k e^{l_k u} = e^{l_0 u} ∑_k c_k e^{(l_k - l_0) u}`. -/
lemma expSum_eq_exp_mul {N : ℕ} (c l : Fin (N + 1) → ℝ) (u : ℝ) :
    ∑ k, c k * Real.exp (l k * u) = Real.exp (l 0 * u) * expSum c (fun k => l k - l 0) u := by
  unfold expSum
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [mul_left_comm, ← Real.exp_add]
  congr 2
  ring

lemma deriv_expSum_shift {N : ℕ} (c l : Fin (N + 1) → ℝ) (u : ℝ) :
    deriv (expSum c (fun k => l k - l 0)) u =
      ∑ k : Fin N, (c k.succ * (l k.succ - l 0)) * Real.exp ((l k.succ - l 0) * u) := by
  rw [(hasDerivAt_expSum c _ u).deriv]
  unfold expSum
  rw [Fin.sum_univ_succ]
  simp

/-- **Proposition `prop:shapes` (ii).** A non-trivial exponential sum `∑_{k<N} c_k e^{l_k u}` with
pairwise distinct exponents `l_k` has at most `N - 1` real zeros. -/
theorem expsum_card_zeros_le : ∀ (N : ℕ) (c l : Fin N → ℝ), Function.Injective l → c ≠ 0 →
    ∀ Z : Finset ℝ, (∀ u ∈ Z, ∑ k, c k * Real.exp (l k * u) = 0) → Z.card ≤ N - 1
  | 0, c, _, _, hc, _, _ => absurd (Subsingleton.elim c 0) hc
  | N + 1, c, l, hl, hc, Z, hZ => by
    set g : ℝ → ℝ := expSum c (fun k => l k - l 0) with hg
    have hZg : ∀ u ∈ Z, g u = 0 := fun u hu => by
      have := hZ u hu
      rw [expSum_eq_exp_mul] at this
      exact (mul_eq_zero.1 this).resolve_left (Real.exp_pos _).ne'
    by_cases hc' : (fun k : Fin N => c k.succ * (l k.succ - l 0)) = 0
    · -- all the coefficients but the first vanish: no zeros at all
      have hcs : ∀ k : Fin N, c k.succ = 0 := by
        intro k
        have h1 := congrFun hc' k
        have hne : l k.succ - l 0 ≠ 0 :=
          sub_ne_zero.2 (fun h => Fin.succ_ne_zero k (hl h))
        simp only [Pi.zero_apply, mul_eq_zero] at h1
        exact h1.resolve_right hne
      have hc0 : c 0 ≠ 0 := by
        intro h0
        apply hc
        funext k
        refine Fin.cases ?_ (fun k => ?_) k
        · simpa using h0
        · simpa using hcs k
      have hZe : Z = ∅ := by
        rw [Finset.eq_empty_iff_forall_notMem]
        intro u hu
        have := hZ u hu
        rw [Fin.sum_univ_succ] at this
        simp only [hcs, zero_mul, Finset.sum_const_zero, add_zero, mul_eq_zero] at this
        rcases this with h | h
        · exact hc0 h
        · exact (Real.exp_pos _).ne' h
      simp [hZe]
    · have hN : 1 ≤ N := by
        rcases Nat.eq_zero_or_pos N with h | h
        · subst h; exact absurd (Subsingleton.elim _ _) hc'
        · exact h
      have hinj : Function.Injective (fun k : Fin N => l k.succ - l 0) := by
        intro a b h
        exact Fin.succ_injective _ (hl (by simpa using h))
      obtain ⟨T, hT, hcard⟩ :=
        card_zeros_le_card_zeros_deriv g (differentiable_expSum _ _) Z hZg
      have hTc := expsum_card_zeros_le N _ _ hinj hc' T (fun u hu => by
        rw [← deriv_expSum_shift c l u]; exact hT u hu)
      simp only [Nat.add_sub_cancel]
      omega

/-- **Proposition `prop:shapes` (ii), set form.** The zero set of a non-trivial exponential sum
with `N` pairwise distinct exponents is finite and has at most `N - 1` elements. -/
theorem expsum_zeros_finite (N : ℕ) (c l : Fin N → ℝ) (hl : Function.Injective l) (hc : c ≠ 0) :
    {u : ℝ | ∑ k, c k * Real.exp (l k * u) = 0}.Finite ∧
      {u : ℝ | ∑ k, c k * Real.exp (l k * u) = 0}.ncard ≤ N - 1 := by
  have hfin : {u : ℝ | ∑ k, c k * Real.exp (l k * u) = 0}.Finite := by
    by_contra hinf
    obtain ⟨t, ht, hcard⟩ := Set.Infinite.exists_subset_card_eq hinf N
    have := expsum_card_zeros_le N c l hl hc t (fun u hu => ht hu)
    have hN : 1 ≤ N := by
      rcases Nat.eq_zero_or_pos N with h | h
      · subst h; exact absurd (Subsingleton.elim c 0) hc
      · exact h
    omega
  refine ⟨hfin, ?_⟩
  rw [Set.ncard_eq_toFinset_card _ hfin]
  exact expsum_card_zeros_le N c l hl hc _ (fun u hu => (Set.Finite.mem_toFinset hfin).1 hu)

/-- Dichotomy for arbitrary (possibly repeated) exponents over `Fin N`: either the exponential sum
vanishes identically, or every finite set of its zeros has at most `N - 1` elements. -/
theorem expsum_zero_or_card_le_fin : ∀ (N : ℕ) (c l : Fin N → ℝ),
    (∀ u, ∑ k, c k * Real.exp (l k * u) = 0) ∨
      ∀ Z : Finset ℝ, (∀ u ∈ Z, ∑ k, c k * Real.exp (l k * u) = 0) → Z.card ≤ N - 1
  | 0, c, l => Or.inl fun u => by simp
  | N + 1, c, l => by
    set g : ℝ → ℝ := expSum c (fun k => l k - l 0) with hg
    have hfg : ∀ u, ∑ k, c k * Real.exp (l k * u) = 0 ↔ g u = 0 := fun u => by
      rw [expSum_eq_exp_mul]
      constructor
      · intro h; exact (mul_eq_zero.1 h).resolve_left (Real.exp_pos _).ne'
      · intro h
        have h' : expSum c (fun k => l k - l 0) u = 0 := h
        rw [h', mul_zero]
    have hgd : Differentiable ℝ g := differentiable_expSum _ _
    by_cases hd : ∀ u, deriv g u = 0
    · -- `g` is constant
      have hconst := is_const_of_deriv_eq_zero hgd hd
      by_cases h0 : g 0 = 0
      · left
        intro u
        rw [hfg, hconst u 0, h0]
      · right
        intro Z hZ
        have hZe : Z = ∅ := by
          rw [Finset.eq_empty_iff_forall_notMem]
          intro u hu
          have := (hfg u).1 (hZ u hu)
          rw [hconst u 0] at this
          exact h0 this
        simp [hZe]
    · right
      intro Z hZ
      have hZg : ∀ u ∈ Z, g u = 0 := fun u hu => (hfg u).1 (hZ u hu)
      obtain ⟨T, hT, hcard⟩ := card_zeros_le_card_zeros_deriv g hgd Z hZg
      rcases expsum_zero_or_card_le_fin N (fun k => c k.succ * (l k.succ - l 0))
          (fun k => l k.succ - l 0) with hIH | hIH
      · exact absurd (fun u => by rw [deriv_expSum_shift c l u]; exact hIH u) hd
      · have hTc := hIH T (fun u hu => by rw [← deriv_expSum_shift c l u]; exact hT u hu)
        have hN : 1 ≤ N := by
          rcases Nat.eq_zero_or_pos N with h | h
          · subst h
            exact absurd (fun u => by rw [deriv_expSum_shift c l u]; simp) hd
          · exact h
        simp only [Nat.add_sub_cancel]
        omega

/-- The dichotomy for an arbitrary finite index type: either the exponential sum vanishes
identically, or its zero set is finite. -/
theorem expSum_zero_or_finite {κ : Type*} [Fintype κ] (c l : κ → ℝ) :
    (∀ u, expSum c l u = 0) ∨ {u | expSum c l u = 0}.Finite := by
  classical
  let e := Fintype.equivFin κ
  have hsum : ∀ u, expSum c l u = ∑ k : Fin (Fintype.card κ),
      (c ∘ e.symm) k * Real.exp ((l ∘ e.symm) k * u) := fun u => by
    unfold expSum
    exact (Fintype.sum_equiv e _ _ (fun k => by simp)).trans rfl
  rcases expsum_zero_or_card_le_fin (Fintype.card κ) (c ∘ e.symm) (l ∘ e.symm) with h | h
  · left; intro u; rw [hsum]; exact h u
  · right
    by_contra hinf
    obtain ⟨t, ht, hcard⟩ := Set.Infinite.exists_subset_card_eq hinf (Fintype.card κ + 1)
    have := h t (fun u hu => by rw [← hsum]; exact ht hu)
    omega

/-! ## (iii) Finitely many switches -/

/-- `E_v(u) = ∑_{j : v_j ≠ 0} e^{u log v_j}`; for `u > 0` it equals `∑_j v_j^u`. -/
def Ev {ι : Type*} [Fintype ι] (v : ι → ℝ) (u : ℝ) : ℝ :=
  ∑ j, if v j = 0 then 0 else Real.exp (Real.log (v j) * u)

lemma budget_eq_Ev {ι : Type*} [Fintype ι] {v : ι → ℝ} (hv : ∀ j, 0 ≤ v j) {s : ℝ}
    (hs : s ≠ 0) : budget s v = Ev v (1 / s) := by
  unfold budget Ev
  refine Finset.sum_congr rfl fun j _ => ?_
  split_ifs with h
  · rw [h, Real.zero_rpow (one_div_ne_zero hs)]
  · rw [Real.rpow_def_of_pos (lt_of_le_of_ne (hv j) (Ne.symm h))]

/-- The difference `E_v - E_w` is an exponential sum (indexed by `ι ⊕ ι`). -/
lemma Ev_sub_eq_expSum {ι : Type*} [Fintype ι] (v w : ι → ℝ) (u : ℝ) :
    Ev v u - Ev w u =
      expSum (Sum.elim (fun j => if v j = 0 then (0 : ℝ) else 1)
                       (fun j => if w j = 0 then (0 : ℝ) else -1))
             (Sum.elim (fun j => Real.log (v j)) (fun j => Real.log (w j))) u := by
  unfold Ev expSum
  rw [Fintype.sum_sum_type, sub_eq_add_neg, ← Finset.sum_neg_distrib]
  congr 1
  · refine Finset.sum_congr rfl fun j _ => ?_
    by_cases h : v j = 0 <;> simp [h]
  · refine Finset.sum_congr rfl fun j _ => ?_
    by_cases h : w j = 0 <;> simp [h]

/-- A continuous function without zeros on `[[a, b]]` has the same sign at `a` and `b`. -/
lemma sign_const_of_no_zero {f : ℝ → ℝ} {a b : ℝ} (hf : ContinuousOn f (Set.uIcc a b))
    (h0 : ∀ x ∈ Set.uIcc a b, f x ≠ 0) : f a ≤ 0 ↔ f b ≤ 0 := by
  have key : ¬ (0 : ℝ) ∈ Set.uIcc (f a) (f b) := by
    intro hmem
    obtain ⟨x, hx, hfx⟩ := intermediate_value_uIcc hf hmem
    exact h0 x hx hfx
  rw [Set.mem_uIcc] at key
  constructor
  · intro ha
    by_contra hb
    exact key (Or.inl ⟨ha, (lt_of_not_ge hb).le⟩)
  · intro hb
    by_contra ha
    exact key (Or.inr ⟨hb, (lt_of_not_ge ha).le⟩)

lemma one_div_mem_uIcc {s s' u : ℝ} (hs : 0 < s) (hs' : 0 < s')
    (hu : u ∈ Set.uIcc (1 / s) (1 / s')) : 1 / u ∈ Set.uIcc s s' := by
  rw [Set.mem_uIcc] at hu ⊢
  rcases hu with ⟨h1, h2⟩ | ⟨h1, h2⟩
  · have hu0 : 0 < u := lt_of_lt_of_le (by positivity) h1
    right
    exact ⟨(le_one_div hu0 hs').1 h2, (one_div_le hs hu0).1 h1⟩
  · have hu0 : 0 < u := lt_of_lt_of_le (by positivity) h1
    left
    exact ⟨(le_one_div hu0 hs).1 h2, (one_div_le hs' hu0).1 h1⟩

/-- **Proposition `prop:shapes` (iii), finiteness of switches.** For a finite set `V` of
non-negative vectors there is a finite set `Z` of values of `s` such that the set of minimizers of
`φ_s(v) = ∑_j v_j^{1/s}` over `V` is the same at any two `s, s' > 1` with no point of `Z` between
them. -/
theorem switches_finite {ι : Type*} [Fintype ι] (V : Finset (ι → ℝ))
    (hV : ∀ v ∈ V, ∀ j, 0 ≤ v j) :
    ∃ Z : Finset ℝ, ∀ s s' : ℝ, 1 < s → 1 < s' → (∀ z ∈ Z, z ∉ Set.uIcc s s') →
      V.filter (fun v => ∀ w ∈ V, budget s v ≤ budget s w) =
        V.filter (fun v => ∀ w ∈ V, budget s' v ≤ budget s' w) := by
  classical
  let D : (ι → ℝ) × (ι → ℝ) → ℝ → ℝ := fun q u => Ev q.1 u - Ev q.2 u
  have hD : ∀ q, D q = expSum _ _ := fun q => funext fun u => Ev_sub_eq_expSum q.1 q.2 u
  have hdich : ∀ q, (∀ u, D q u = 0) ∨ {u | D q u = 0}.Finite := fun q => by
    rw [hD q]; exact expSum_zero_or_finite _ _
  have hcont : ∀ q, Continuous (D q) := fun q => by rw [hD q]; exact continuous_expSum _ _
  let Zq : (ι → ℝ) × (ι → ℝ) → Finset ℝ := fun q =>
    if h : {u | D q u = 0}.Finite then h.toFinset.image (fun u => 1 / u) else ∅
  refine ⟨(V ×ˢ V).biUnion Zq, ?_⟩
  intro s s' hs hs' hZ
  have hs0 : s ≠ 0 := by linarith
  have hs0' : s' ≠ 0 := by linarith
  -- pairwise comparisons do not change
  have hpair : ∀ v ∈ V, ∀ w ∈ V, (budget s v ≤ budget s w ↔ budget s' v ≤ budget s' w) := by
    intro v hv w hw
    have e1 : budget s v ≤ budget s w ↔ D (v, w) (1 / s) ≤ 0 := by
      simp only [D, budget_eq_Ev (hV v hv) hs0, budget_eq_Ev (hV w hw) hs0, sub_nonpos]
    have e2 : budget s' v ≤ budget s' w ↔ D (v, w) (1 / s') ≤ 0 := by
      simp only [D, budget_eq_Ev (hV v hv) hs0', budget_eq_Ev (hV w hw) hs0', sub_nonpos]
    rw [e1, e2]
    rcases hdich (v, w) with h | h
    · simp [h]
    · refine sign_const_of_no_zero (hcont (v, w)).continuousOn ?_
      intro x hx hDx
      have hxZ : 1 / x ∈ (V ×ˢ V).biUnion Zq := by
        rw [Finset.mem_biUnion]
        refine ⟨(v, w), Finset.mem_product.2 ⟨hv, hw⟩, ?_⟩
        simp only [Zq, dite_eq_left h]
        exact Finset.mem_image.2 ⟨x, (Set.Finite.mem_toFinset h).2 hDx, rfl⟩
      exact hZ _ hxZ (one_div_mem_uIcc (by linarith) (by linarith) hx)
  refine Finset.filter_congr fun v hv => ?_
  constructor
  · intro h w hw; exact (hpair v hv w hw).1 (h w hw)
  · intro h w hw; exact (hpair v hv w hw).2 (h w hw)

/-! ## Ties: equal multisets of non-zero entries -/

lemma multiset_sum_map_filter_ne_zero (m : Multiset ℝ) (g : ℝ → ℝ) (hg : g 0 = 0) :
    (m.map g).sum = ((m.filter (· ≠ 0)).map g).sum := by
  induction m using Multiset.induction_on with
  | empty => simp
  | cons a m ih =>
    by_cases ha : a = 0
    · subst ha
      simp [hg, ih]
    · simp [ha, ih]

/-- **Ties (`prop:shapes`).** If the non-zero entries of `v` and `w` agree as multisets, then
`φ_s(v) = φ_s(w)` for every `s ≠ 0`; in particular two such vertices are tied for every `s`. -/
theorem ties_of_multiset_eq {ι : Type*} [Fintype ι] {v w : ι → ℝ}
    (h : (Finset.univ.val.map v).filter (· ≠ 0) = (Finset.univ.val.map w).filter (· ≠ 0))
    {s : ℝ} (hs : s ≠ 0) : budget s v = budget s w := by
  have hg : (0 : ℝ) ^ (1 / s) = 0 := Real.zero_rpow (one_div_ne_zero hs)
  have key : ∀ x : ι → ℝ, budget s x =
      (((Finset.univ.val.map x).filter (· ≠ 0)).map (fun t => t ^ (1 / s))).sum := by
    intro x
    unfold budget
    rw [Finset.sum_eq_multiset_sum, ← multiset_sum_map_filter_ne_zero _ _ hg, Multiset.map_map]
    rfl
  rw [key v, key w, h]

end Heracles
