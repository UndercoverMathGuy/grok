import Heracles.Defs

/-!
# Networks, margins and the Fourier spectrum (paper Sections `sec:prelim`, `sec:approach`)

Shared definitions for the network-level results.

* Inputs `x : Fin n → ZMod p` (the `n` operands), label `y(x) = ∑_t x_t`.
* A logit table `f : Inp p n → ZMod p → ℝ` (`f x c` is the logit of candidate `c`).
* The dataset margin `h(f) = min_{x, c ≠ y(x)} (f x (y x) - f x c)` (full dataset `D_p`).
* `ρ^m` is `ZMod.stdAddChar m = exp(2π i m / p)`.
* The multidimensional transform `f̂(k; l) = ∑_{x,c} f x c ρ^{-(∑_t k_t x_t + l c)}`.
* `β_j(f) = 2 p^{-(n+1)} f̂(j,…,j; -j)` and the amplitude `A_j = |β_j|`.
* The averaged logit `η_f(δ) = E_x f x (y(x) - δ)`.
-/

open Real Finset

noncomputable section

namespace Heracles

/-- Inputs: `n` operands in `ℤ_p`. -/
abbrev Inp (p n : ℕ) := Fin n → ZMod p

/-- The label `y(x) = a_1 + ⋯ + a_n`. -/
def label {p n : ℕ} (x : Inp p n) : ZMod p := ∑ t, x t

/-- A logit table. -/
abbrev Table (p n : ℕ) := Inp p n → ZMod p → ℝ

/-- The wrong (input, candidate) pairs of the full dataset `D_p`. -/
def wrongPairs (p n : ℕ) [NeZero p] : Finset (Inp p n × ZMod p) :=
  Finset.univ.filter fun xc => xc.2 ≠ label xc.1

lemma wrongPairs_nonempty (p n : ℕ) [Fact p.Prime] : (wrongPairs p n).Nonempty :=
  ⟨(0, 1), Finset.mem_filter.2 ⟨Finset.mem_univ _, by simp [label]⟩⟩

/-- The margin of the dataset, `h = min_{(x,y) ∈ D_p} (f[y] - max_{y' ≠ y} f[y'])`,
written as the minimum over all wrong pairs. -/
def margin {p n : ℕ} [Fact p.Prime] (f : Table p n) : ℝ :=
  (wrongPairs p n).inf' (wrongPairs_nonempty p n) fun xc => f xc.1 (label xc.1) - f xc.1 xc.2

lemma margin_le {p n : ℕ} [Fact p.Prime] (f : Table p n) (x : Inp p n) (c : ZMod p)
    (hc : c ≠ label x) : margin f ≤ f x (label x) - f x c :=
  Finset.inf'_le (f := fun xc : Inp p n × ZMod p => f xc.1 (label xc.1) - f xc.1 xc.2)
    (Finset.mem_filter.2 ⟨Finset.mem_univ (x, c), hc⟩)

lemma le_margin {p n : ℕ} [Fact p.Prime] (f : Table p n) {a : ℝ}
    (h : ∀ x c, c ≠ label x → a ≤ f x (label x) - f x c) : a ≤ margin f := by
  unfold margin
  rw [Finset.le_inf'_iff]
  rintro ⟨x, c⟩ hxc
  simp only [wrongPairs, Finset.mem_filter, Finset.mem_univ, true_and] at hxc
  exact h x c hxc

/-- `ρ^m = exp(2π i m/p)` for `m ∈ ℤ_p`. -/
def rho {p : ℕ} [NeZero p] (m : ZMod p) : ℂ := ZMod.stdAddChar m

/-- The multidimensional Fourier transform of a logit table. -/
def fhat {p n : ℕ} [NeZero p] (f : Table p n) (k : Fin n → ZMod p) (l : ZMod p) : ℂ :=
  ∑ x : Inp p n, ∑ c : ZMod p, (f x c : ℂ) * rho (-(∑ t, k t * x t + l * c))

/-- `β_j = 2 p^{-(n+1)} f̂(j,…,j; -j)` (paper eq. (spectrum)). -/
def beta {p n : ℕ} [NeZero p] (f : Table p n) (j : ℕ) : ℂ :=
  2 / (p : ℂ) ^ (n + 1) * fhat f (fun _ => (j : ZMod p)) (-(j : ZMod p))

/-- The sign-flipped coefficient `β_{j,T} = 2 p^{-(n+1)} f̂(k; -j)` with `k_t = -j` for `t ∈ T`
and `k_t = j` otherwise (Section `sec:cyclic`); `β_{j,∅} = β_j`. -/
def betaT {p n : ℕ} [NeZero p] (f : Table p n) (j : ℕ) (T : Finset (Fin n)) : ℂ :=
  2 / (p : ℂ) ^ (n + 1) *
    fhat f (fun t => if t ∈ T then -(j : ZMod p) else (j : ZMod p)) (-(j : ZMod p))

/-- Definition `def:if`: a table is interference-free if `β_{j,T} = 0` for every `j ∈ F_p`
and every non-empty `T ⊆ [n]`. -/
def InterferenceFree {p n : ℕ} [NeZero p] (f : Table p n) : Prop :=
  ∀ j ∈ Finset.Icc 1 ((p - 1) / 2), ∀ T : Finset (Fin n), T.Nonempty → betaT f j T = 0

/-- The amplitude `A_j = |β_j|`. -/
def amp {p n : ℕ} [NeZero p] (f : Table p n) (j : ℕ) : ℝ := ‖beta f j‖

/-- The averaged logit `η(δ) = E_x f x (y(x) - δ)`. -/
def eta {p n : ℕ} [NeZero p] (f : Table p n) (δ : ZMod p) : ℝ :=
  (1 / (p : ℝ) ^ n) * ∑ x : Inp p n, f x (label x - δ)

/-- The table of a network: the sum of its neurons' tables. -/
def netTable {p n : ℕ} {Ω : Type*} {m : ℕ} (φ : Ω → Table p n) (θ : Fin m → Ω) : Table p n :=
  fun x c => ∑ i, φ (θ i) x c

/-! ## Product neurons -/

/-- Parameters of a product neuron: `u_1, …, u_n, w : ℤ_p → ℝ`. -/
structure ProdParam (p n : ℕ) where
  u : Fin n → ZMod p → ℝ
  w : ZMod p → ℝ

/-- The product neuron `u_1(a_1) ⋯ u_n(a_n) w(c)`. -/
def prodNeuron {p n : ℕ} (ω : ProdParam p n) : Table p n :=
  fun x c => (∏ t, ω.u t (x t)) * ω.w c

/-- Squared Euclidean norm `‖ω‖_2^2` of a product neuron's weights. -/
def normSq {p n : ℕ} [NeZero p] (ω : ProdParam p n) : ℝ :=
  ∑ t, ∑ a, ω.u t a ^ 2 + ∑ c, ω.w c ^ 2

/-- `‖θ‖_{2,b}^b = ∑_i ‖ω_i‖_2^b` for a product network. -/
def normPow {p n m : ℕ} [NeZero p] (b : ℝ) (θ : Fin m → ProdParam p n) : ℝ :=
  ∑ i, (Real.sqrt (normSq (θ i))) ^ b

/-- `B_p = 2 (2 p ν)^{-ν/2}` with `ν = n + 1`. -/
def Bp (p n : ℕ) : ℝ := 2 * (2 * (p : ℝ) * (n + 1)) ^ (-((n + 1 : ℝ) / 2))

end Heracles
