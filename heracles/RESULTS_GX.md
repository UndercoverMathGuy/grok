# GX — the group axis on the exact families (quadratic + bilinear, ν = 3, plain l2)   (2026-09-18)
[group_exact.py, kstar_gx.json, FINAL_RUNS/GX; pods gx-2 / gx-3, RTX 5090 secure]

Families (one hidden layer, one-hot inputs, no biases, no pre-layer, width m = 500), task a·b = c on G:
quad z = Σ_i w_i(c)(e_i(a) + f_i(b))² (Morwani's net), bil z = Σ_i w_i(c) e_i(a) f_i(b) (gated / CP). Every symbol:
e_i, f_i, w_i are functions on G (rows of the two input tables and the output table), |G| = N, ν = 3 = homogeneity
degree, q = 2 = plain weight decay (λ‖θ‖² added to the loss), d_ρ = dimension of the real irrep ρ, χ_ρ its character,
cov_ρ(x) = 1 − χ_ρ(x)/d_ρ ("coverage" of the wrong answer at offset x by irrep ρ; the mod-add 1 − cos).

## 0. What is theorem and what is ansatz at ν = 3 on a non-abelian group   (written BEFORE any GX run was read)

Symmetry: (a, b, c) → (g a k⁻¹, k b h, g c h) preserves the task, and its only invariant is the conjugacy class of
D = a b c⁻¹. So the orbit average of the logit table ("clean part") is a class function f(D) = Σ_ρ A_ρ χ_ρ(D)/d_ρ.

- **T1 (junk lemma, theorem).** margin(z) ≤ margin(f) = min_{D≠e} Σ_ρ A_ρ cov_ρ(D) (needs all A_ρ ≥ 0, else smaller).
  Same orbit-average proof as THEORY_nsum L1; the orbit group is transitive on each class level set.
- **T2 (assembly, theorem).** Write ê_i(ρ) = Σ_a e_i(a) ρ(a) (d×d Fourier block). Then for both families
  A_ρ = c_fam · (d_ρ / N³) Σ_i tr( ê_i(ρ) f̂_i(ρ) ŵ_i(ρ)ᵀ ),  c_bil = 1, c_quad = 2.
  For quad only the cross term 2 e f w reaches the clean part: e² w and f² w depend on (a, c) or (b, c) alone, and the
  orbit average of such a function is constant in c. So per-frequency decoupling (abelian) becomes per-irrep
  decoupling: only the ρ-blocks of e, f, w talk to A_ρ. This part carries over exactly.
- **T3 (per-irrep AM-GM, theorem, upper bound).** |tr(EFWᵀ)| ≤ ‖E‖_F ‖F‖_F ‖W‖_F ≤ (mass/3)^{3/2}, and Parseval gives
  ‖ê(ρ)‖_F² = (N/d_ρ)‖e_ρ‖². Hence A_ρ ≤ c_fam d_ρ^{-1/2} N^{-3/2} 3^{-3/2} Σ_i m_{iρ}^{3/2} with m_{iρ} the squared weight
  norm neuron i puts on irrep ρ, and (superadditivity) Σ_ρ d_ρ^{1/3} A_ρ^{2/3} ≤ C ‖θ‖². With T1:
      margin ≤ C' ‖θ‖³ · max_A  min_{D≠e} Σ_ρ A_ρ cov_ρ(D) / ( Σ_ρ κ_ρ A_ρ^{2/3} )^{3/2},   κ_ρ = d_ρ^{1/3}   ("bound" ladder).
  Unrestricted (any net in the family). For abelian G every 2-dim real irrep has the same κ, and this IS the old
  ladder M_K / K^{3/2}; order-2 characters (Z_120's sgn60) are 2^{1/3} cheaper.
- **Where exactness stops (ansatz).** Equality in T3 needs rank-1 aligned blocks in ONE neuron (E = uvᵀ, F = vwᵀ,
  W = uwᵀ): that neuron reaches the bound for A_ρ but is not junk-free. The junk-free matrix-coefficient construction
  (d³ neurons ρ_ij ⊗ ρ_jk ⊗ ρ_ik: the matrix-multiplication tensor ⟨d,d,d⟩ as a CP sum) costs κ_ρ = d_ρ^{4/3} — a factor
  d_ρ more. For d = 1 the bound is attained exactly; for abelian 2-dim irreps the gap is the known constant
  (κ = 2 vs 2^{1/3}, the same for every frequency so the argmax does not move). For a non-abelian group with mixed
  dimensions the true price of irrep ρ is the l2-CP cost of ⟨d,d,d⟩ (unknown in closed form, in [d^{1/3}, d^{4/3}],
  and junk-tolerant solutions may sit between): **the irrep-dependent constant is NOT fixed by the theorem.**
  So: theorem = T1–T3 + the abelian argmax; ansatz = which κ_ρ the non-abelian optimum pays. Both ladders are
  registered below ("bound" κ = d^{1/3}; "naive" κ = d^{4/3}, complex-type 2-dim: 2); where they disagree the run
  discriminates between them. Also not theorem: that GD reaches the max-margin point (as everywhere in the programme).

## 1. Predictions (kstar_gx.json; every support enumerated with free amplitudes for S5/S4/A5; abelian = kstar_ab
equal-share sets + free amplitudes; γ̂ relative to the argmax; plateau = rungs ≥ 0.74 of the top)

| G | bound ladder (κ = d^{1/3}) | naive ladder (κ = d^{4/3}) | discriminating content |
|---|---|---|---|
| S5 | **K 1: std_sign** 1.00; 5a = 5b 0.95; 6 0.91; std+sign(0.125) 0.80; std 0.67 | **K 1: std_sign** 1.00; std+sign(0.125) 0.94; 5a = 5b 0.68; std 0.67; 6 0.49 | one faithful irrep; ν-3 argmax is std_sign with NO sign patch (std_sign self-covers transpositions: min cov 0.75 vs std 0.5). Patch only on std-led runs, at 0.125 (ν 4 had 0.17). bound ⇒ 5/6-dim leaders nearly as good; naive ⇒ 4-dim only |
| S4 | **K 2: std+sign or std_sign+sign, patch 0.167** 1.00; std+std_sign 0.94; single std/std_sign 0.89; 2+std 0.83 | same argmax 1.00; 2+std 0.85; std+std_sign 0.78; single 0.74 | leader 3-dim + sign patch 0.167; the non-faithful 2 never alone |
| A5 | **K 1: 4** 1.00; 5 0.95; 3a+3b 0.91; 3a+4 0.88; 3a alone 0.71 | **K 2: 3a+3b (equal)** 1.00; 3a alone 0.78; 3a+4 0.76; 4 0.72; 5 0.49 | THE κ test: bound ⇒ single 4 (or 5); naive ⇒ the Galois pair 3a+3b |
| Z_120 | K* 7 (5: 0.995, 8: 0.99, 6: 0.96, 4: 0.90); plateau 4..16 | same (all κ equal) | count in plateau, mode 5–8; non-faithful (gcd > 1) frequencies expected as in RESULTS_Z |
| Z_11×Z_11 | K* 5 (8: 0.99, 9: 0.98, 6–7: 0.97, 4: 0.92); plateau 4..16 | same | as cyclic: ν 3 cannot separate Z_11² from Z_121-like cyclic (it did at ν 6) |
| Z_113 (control) | K* 6 (5: 0.99, 7: 0.97); plateau 4..16 | same | must reproduce RESULTS_QM: K_5pct 6–8, K_2pct 7–9 |
| Z_71 (control) | K* 5 (6: 0.95, 7: 0.92, 4: 0.89); plateau 3..15 | same | must reproduce RESULTS_QM / PROD: quad 4–6, bil 5–8 |

Registered statements (both families, full data, per cell, separated runs only):
- (a) non-abelian count: S5 and A5-bound: exactly one faithful irrep ≥ 0.9 of faithful energy in ≥ 75% of runs;
  kill if ≥ 2 faithful irreps above 0.2 relative amplitude in ≥ 1/3 of runs (then the ν-3 exponent 2/3 is too weak to
  make K = 1 and the honest reading is "plateau {1, 2}").  S4: sign present (sign_rel ≥ 0.05) in ≥ 75%.
- (b) identity is a lottery (as at ν 4/6): no leader-frequency claim; but the MARGIN ORDER by leader should follow
  the ladder: S5 std_sign-led > std-led; A5: decides bound (4/5-led on top) vs naive (3a+3b on top).
- (c) sign patch amplitude: S5 std-led 0.125, std_sign-led 0; S4 0.167.
- (d) abelian: K_abl mode inside the plateau and within K* ± 2; controls reproduce QM/PROD.
- ν = 3 is the flattest rung of the ladder (plateaus 4..16 abelian, {1, 2} non-abelian): weak discrimination is
  expected and is part of the prediction, not an excuse added afterwards.

Disclosure: before this section was written I saw ONE local smoke run (S4, bil, m 100, 3000 steps, 3 seeds, not
converged, acc 0.998–1.0: counts 3, 4, –). The ladders above are the mechanical output of `group_exact.py predict`
and were not adjusted to it.

## 2. Measured (551 runs in FINAL_RUNS/GX; 3 pods, all deleted after checksummed pulls; `group_exact.py table`)

Recipe: m 500, full-batch GD on mean CE + 1e-4 Σθ² (penalty in the loss), lr 0.05 doubled at 1e3..10e3, cap 3.2, no clip,
init ×4, 40k steps. quad trains as is. **bil at |G| 113–121 collapses to zero weights** under that recipe (S5 16/16,
Z_120/Z_11² most seeds; kept as `*_st40000` without `_wdw`): it needs the decay-free warm-up of RESULTS_PROD (penalty
switched on per seed once CE < 0.8 ln|G|, step ≈ 6.5k; 50k steps). A5/S4/Z_71 bil train either way, same counts.
Instruments: K_5pct / K_2pct = irreps holding ≥ 5% / 2% of clean-logit energy (the RESULTS_QM instrument); K_abl =
interface ablation (score.py); γ = min margin / ‖θ‖³; sign_rel = A_sign / A_leader; clean = clean-logit energy fraction.

### 2a. Free runs, full data (16 seeds per cell, all acc 1.0, all separated unless noted)

| G | family | supports found | K_5pct | K_abl | sign_rel | γ med (max) | clean |
|---|---|---|---|---|---|---|---|
| S5 | quad | 16/16 std + std_sign near-equal (leader std 12, std_sign 4; lead share 0.53) + sign + 5a/5b tails; 6 never | {3: 9, 4: 7} | 6 (16/16) | 0.38–0.41 | 3.2e-6 (4.0e-6) | 0.49 |
| S5 | bil | same (std 11, std_sign 5) | {3: 6, 4: 7, 5: 3} | 6 (16/16) | 0.44–0.48 | 1.5e-6 (2.0e-6) | 0.50 |
| A5 | quad | 16/16 **3a + 3b** (share 0.57), 4 and 5 at 0.00 amplitude | 2 (16/16) | 4 (16/16) | – | 2.7e-5 (3.3e-5) | 0.78 |
| A5 | bil | 16/16 3a + 3b (4 at ≤ 0.19 in some) | 2 (16/16) | {3: 1, 4: 15} | – | 1.3e-5 (1.6e-5) | 0.66 |
| S4 | quad | 16/16 all four irreps (std or std_sign leads 5/11, share 0.64–0.79) | {3: 1, 4: 15} | 4 | 0.47–0.51 | 2.3e-4 (2.5e-4) | 0.67 |
| S4 | bil | same (6/10) | {3: 1, 4: 15} | 4 | 0.63–0.64 | 1.4e-4 (1.6e-4) | 0.64 |
| Z_120 | quad | sgn60 member 15/16, gcd > 1 members 16/16 | {3: 1, 5: 6, 6: 6, 7: 1, 8: 1, 9: 1} | {14: 2, 16: 14} | | 6.3e-6 | 0.49 |
| Z_120 | bil | sgn60 15/16 | {2: 1, 3: 2, 4: 2, 5: 7, 6: 3, 7: 1} | 10–16 | | 3.7e-6 | 0.57 |
| Z_11² | quad | | {4: 2, 5: 3, 6: 4, 7: 6, 8: 1} | 5–16, mode 16 | | 4.3e-6 | 0.40 |
| Z_11² | bil | 13/16 separated | {3: 2, 4: 2, 5: 5, 6: 1, 7: 2, 8: 1} | 10–16 | | 9.2e-7 | 0.56 |
| Z_113 | quad | | {5: 1, 6: 5, 7: 2, 8: 7, 9: 1} (K_2pct 7–10) | 13–16 | | 6.4e-6 | 0.44 |
| Z_113 | bil | | {3: 1, 4: 3, 5: 5, 6: 5, 7: 1, 8: 1} | 14–16 | | 2.4e-6 | 0.53 |
| Z_71 | quad | | {4: 2, 5: 5, 6: 2, 7: 4, 8: 3} | 11–16 | | 2.4e-5 | 0.63 |
| Z_71 | bil | | {3: 1, 4: 2, 5: 5, 6: 2, 7: 4, 8: 1, 9: 1} | 9–16 | | 1.1e-5 | 0.69 |

Neurons are single-irrep everywhere (energy-weighted top-irrep share 0.98–1.00).

### 2b. Fixed-support ladder (added after the S5 free runs were read; weights projected on the named irreps every
step, warm-up recipe, 4 seeds, all acc 1.0). γ best / median, ×1e-6; "rel" = median / median of the top support.

| S5 support | quad | rel | bil | rel | predicted rel: naive / bound |
|---|---|---|---|---|---|
| std_sign + sign | 10.4 / 8.6 | 1.06 | 4.7 / 3.6 | 1.20 | (not enumerated: optimiser put sign at 0) |
| std_sign | 10.3 / 8.1 | 1.00 | 4.2 / 3.0 | 1.00 | 1.00 / 1.00 |
| std + sign | 11.9 / 7.8 | 0.96 | 5.6 / 4.2 | 1.42 | 0.94 / 0.80 |
| std | 8.4 / 5.8 | 0.72 | 2.9 / 2.5 | 0.82 | 0.67 / 0.67 |
| std + std_sign + sign | 5.4 / 4.9 | 0.61 | 2.6 / 2.2 | 0.75 | – |
| std + std_sign + sign + 5a + 5b (the free runs' own support) | 4.4 / 4.1 | 0.51 | 1.9 / 1.6 | 0.52 | – |
| 5b | 4.5 / 3.7 | 0.46 | 1.6 / 1.3 | 0.45 | 0.68 / 0.95 |
| 5a | 3.5 / 3.2 | 0.40 | 1.4 / 1.2 | 0.39 | 0.68 / 0.95 |
| 6 | 2.9 / 2.3 | 0.29 | 1.1 / 0.9 | 0.30 | 0.49 / 0.91 |
| **free run** | 4.0 / 3.2 | 0.40 | 2.0 / 1.5 | 0.51 | |

| A5 support | quad | rel | bil | rel | naive / bound |
|---|---|---|---|---|---|
| 3a + 3b | 53 / 49 | 1.00 | 30 / 20 | 1.00 | 1.00 / 0.91 |
| 3a | 39 / 39 | 0.79 | 23 / 23 | 1.14 | 0.78 / 0.71 |
| 3a + 3b + 4 | 37 / 34 | 0.68 | 20 / 16 | 0.81 | – |
| 4 | 34 / 30 | 0.61 | 20 / 17 | 0.86 | 0.72 / 1.00 |
| 5 | 14 / 14 | 0.28 | 7.3 / 6.9 | 0.34 | 0.49 / 0.95 |
| **free run** | 33 / 27 | 0.55 | 17 / 14 | 0.71 | |

S4 (×1e-4, quad | bil medians): std+sign 2.72 | 1.87 (sign_rel 0.25 | 0.29); 2+std+sign 2.42 | 1.58; std+std_sign+sign
2.33 | 1.45; std 2.11 | 1.19; 2+std 2.11 | 1.22; std+std_sign 2.05 | 1.15; **free 2.27 | 1.42–1.47** (0.83 | 0.77 of top).
Fixed-support sign patch: S5 std-led 0.20 (predicted 0.125), std_sign-led 0.13–0.19 (predicted 0); S4 0.25–0.29 (0.167).

Measured irrep price in the free nets, κ_meas = (‖θ_ρ‖²)/(3|G|(A_ρ/c_fam)^{2/3}): sign 1.00–1.06 (theorem: exactly 1);
std 3.7–3.9, std_sign 3.4–3.8, 5a 3.4–3.7, 5b 3.6–4.0 (S5); 3a/3b 3.3–3.7 (A5). Bound d^{1/3} = 1.4–1.7, naive
d^{4/3} = 4.3–8.6: the nets pay between the two, closer to naive for d = 3, and A5 uses 13–20 live neurons per 3-dim
irrep against the naive 27.

### 2c. frac 0.8 arm (8 seeds): generalises in every cell (test acc med 0.93 S4-quad, ≥ 0.997 elsewhere) but the
full-table min margin is often ≤ 0 at 40–50k, so few runs are scoreable: S5 bil 2/8 (same mixture, sign_rel 0.40),
S5 quad 0/8, A5 7/8 + 2/8 (3a + 3b, K_5pct 2), S4 8/8 + 1/8 (K 4), Z_120 8/8 + 6/8 (K_5pct 4–7, sgn60 14/14),
Z_11² 1/8 + 2/8, Z_113 3/8 + 1/8, Z_71 5/8 + 7/8 (K_5pct 4–8). No count differs from the full-data cell.

## 3. Scorecard against §1 (as registered)

- (a) **S5 K = 1: FAILS, kill fires** (≥ 2 faithful irreps above 0.2 in 32/32). **S4: sign present 32/32 (holds) but
  support is all four irreps, not std + sign.** A5: decided below.
- **A5 κ test: naive ladder wins outright, bound ladder refuted.** Free runs 32/32 on 3a + 3b with 4 and 5 at zero;
  fixed-support quad order 3a+3b > 3a > 4 > 5 at 1.00 / 0.79 / 0.61 / 0.28 against naive 1.00 / 0.78 / 0.72 / 0.49 and
  bound 0.91 / 0.71 / 1.00 / 0.95. (bil: 3a+3b top by best seed, 3a top by median.) The AM-GM upper bound is a theorem
  but it is NOT the functional the optimum sees on a non-abelian group: the junk-free (matrix-multiplication-tensor)
  price, growing like d^{4/3}, is. The S5 ladder says the same: 5a/5b/6 measure at 0.40–0.46 / 0.29 (naive 0.68 / 0.49,
  bound 0.95 / 0.91) — high-dimensional irreps are even more expensive than naive.
- (b) margin order: the FIXED-SUPPORT ladder follows the naive ladder on all three groups (S5: std_sign(+sign) ≳
  std+sign > std > 5 > 6; every added irrep lowers γ: 1.00 → 0.61 → 0.51). Within free runs there is no order to read:
  every run is the same mixture.
- (c) sign patch: fixed-support 0.13–0.20 (S5) and 0.25–0.29 (S4), i.e. 1.5× the prediction; free runs 0.4–0.6, 3×.
- (d) abelian: K_5pct mode 5–6 (Z_120, K* 7), 5–7 (Z_11², K* 5), 5–8 (Z_113, K* 6), 5–7 (Z_71, K* 5): inside the
  plateau and within K* ± 2 in every cell; controls reproduce RESULTS_QM (Z_113 quad K_2pct 7–10 vs 7–9; Z_71 4–8 vs
  4–6) and RESULTS_PROD (Z_71 bil 5–8). Z_120: sign character 30/32, gcd > 1 members 32/32. ν 3 does not separate
  Z_11² from cyclic (as registered). **K_abl does not reproduce this**: 13–16 on every abelian cell (it saturates at
  its 16-candidate cap) and 4 / 6 on A5 / S5 where irreps holding < 1% of weight energy clear the fixed 2% floor. At
  ν 3 the nets are ~half junk (clean 0.4–0.7) and cutting any irrep from the weights disturbs the junk cancellation;
  the ablation count was calibrated on clean ν 4/6 nets and should not be quoted for these families.

## 4. Verdict

1. **The ladder is right about WHERE the maximum margin is, on the exact families, for every group tested** — with
   the junk-free irrep price (naive κ), not the AM-GM bound. S5: one 4-dim irrep (± sign); A5: the Galois pair; S4:
   std + sign. This is a fixed-support statement (the measured-ladder instrument of RESULTS_S5 E0).
2. **Free GD at ν = 3 does not go there on S5 and S4.** It lands on a broad mixture at 0.40–0.51 (S5), 0.55–0.71 (A5),
   0.77–0.83 (S4) of the best fixed-support margin, and the mixture's own support, trained alone, is also at 0.51:
   the deficit is the support, not under-training. The registered count claim "one irrep on S5" is dead at ν 3.
   A5 is the exception that shows the mechanism: its argmax support IS the two cheapest irreps, and 32/32 find it.
3. Consistent reading: ν 3 is the flattest exponent (A^{2/3}); selection pressure towards sparsity is weakest here,
   exactly as on the abelian side where the plateau is 4..16 and nets sit anywhere in it. The ν 4/6 ReLU results
   (one irrep on S5) are not contradicted, but the exact-theory family cannot be used to claim them: the paper can
   say "the functional's optimum is sparse on every group (fixed-support ladder), GD reaches it at ν ≥ 4, and at ν 3
   stays in a mixed state at half the margin".
4. New theory content worth keeping: per-irrep decoupling is exact (T2), the unrestricted bound has κ = d^{1/3} (T3),
   and the data say the attained price is the junk-free CP cost of ⟨d,d,d⟩ (κ_meas ≈ 3.3–4 for d = 3–5 in mixtures;
   ladder ratios ≥ naive). Open: the l2-CP cost of the matrix-multiplication tensor as the closed-form κ_d.

Caveats: fixed-support runs are 4 seeds, best-of-4 is noisy (±25%); bil ladder has two order inversions by median
(std+sign > std_sign on S5, 3a > 3a+3b on A5); fixed-support runs used warm-up + 50k steps vs 40k for quad free runs
(A5/S4/Z_71 bil free runs at both recipes agree to 5%); K_abl unusable here (above); frac arm mostly unscoreable.
Spend ≈ $6.6 (of which ≈ $3.3 is two pods idling through a session rate-limit outage 15:52–17:32 UTC).
