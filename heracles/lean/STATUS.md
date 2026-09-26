# Lean formalization status

Project: `heracles/lean/`, Lean `v4.34.1`, Mathlib `v4.34.1` (git `d13f23b723b8a846827a245b89c10fc7d3f11612`).
Compiled only on the RunPod CPU pod (`/root/work/Heracles`). This directory is the canonical copy;
`.lake` build artifacts are not kept here. The local and pod sources were checked identical (md5).

Legend. **FORMALIZED**: no `sorry` / `admit` / new `axiom` anywhere in the dependency cone, and
`#print axioms` shows only `propext, Classical.choice, Quot.sound`. **PARTIAL**: some sub-lemmas
`sorry`'d (listed). **NOT ATTEMPTED**.

`#print axioms`: `scripts/Axioms.lean` prints the axioms of 117 main theorems; the output of the
last run is in `axioms_output.txt`. Every one of the 117 lines reads
`depends on axioms: [propext, Classical.choice, Quot.sound]`.
`grep -n "sorry\|admit\|^axiom" Heracles/*.lean` finds nothing.
The whole project (`lake build`, 24 modules) builds with no errors.

See `STATEMENTS.md` for the statement-by-statement fidelity notes (all differences flagged).

## Priority 1–3

| Paper label | Lean name(s) | File | Status |
|---|---|---|---|
| `prop:ablation` | `prop_ablation` (+ `ablation`, `ablation_fraction`, `share_bounds`) | Ablation.lean | FORMALIZED |
| `prop:bottleneck` | `prop_bottleneck`, `prop_bottleneck_network` (+ `bottleneck_abstract`) | Shapes.lean, Vertex.lean | FORMALIZED |
| `thm:scaling`: `Γ` bounds | `GammaP_lower`, `GammaP_lower9`, `GammaP_upper` | Lower.lean, Scaling.lean | FORMALIZED |
| `thm:scaling`: number of frequencies, share floor | `card_supp_ge`, `card_supp_le`, `share_ge` (+ `card_supp_le_ceil`, `dense_only_if`) | Scaling.lean | FORMALIZED |
| `thm:scaling`: network `Θ` | `thm_scaling_net` (+ `gammaStar_le`, `gammaStar_ge`, `thm_scaling_network`) | Final.lean, NetScaling.lean | FORMALIZED |
| scalar bound `0 ≤ ψ ≤ a_s ln(1+z^r)` | `psi_nonneg`, `psi_le` (+ `one_le_I0`, `I0_le_exp`, `I0_ge_of_one_le`) | Scalar.lean | FORMALIZED |
| (psisum) | `psisum_K`, `psisum` | Scaling.lean | FORMALIZED |
| (posfourier) | `posfourier` | PosFourier.lean | FORMALIZED |
| (twoterms), the two choices of `t`, (Kbound) | `twoterms`, `worst_le_of_psi`, `GammaP_upper`, `Kbound` | Scaling.lean | FORMALIZED |
| `K` lower bound from (Kbound), `e^z−1 ≥ z²/2` | `card_supp_ge`, `exp_sub_one_ge_sq` | Scaling.lean | FORMALIZED |
| deletion / share-floor `K` upper bound | `share_ge`, `share_ge_gamma`, `card_supp_le` | Scaling.lean | FORMALIZED |
| random-frequency lower bound | `exists_good_draw`, `exists_alloc_lower`, `sum_cos_freq` | Lower.lean | FORMALIZED (counting form of Hoeffding + union bound) |
| `cor:few` | `cor_few` (generic), `cor_few_prod`, `cor_few_ratio_prod`, `tendsto_gK` (+ `few_bound`) | Few.lean, Final.lean | FORMALIZED |

## Priority 4: `prop:shapes`

| Part | Lean name(s) | File | Status |
|---|---|---|---|
| every maximizer is `Γ·v`, `v` a minimizing vertex | `prop_shapes_vertex` (+ `maximizer_eq_vertex`, `isMaximizer_iff_minVertex`) | Shapes.lean, Vertex.lean | FORMALIZED |
| (i) minimizers of a strictly concave function are vertices | `mem_extremePoints_of_isMin`, `minimizer_isVertex`, `exists_minimizer` | Vertex.lean | FORMALIZED |
| vertex rank property; finitely many vertices | `vertex_eq_zero_of_active`, `finite_vertices` | Vertex.lean | FORMALIZED |
| (ii) exponential sum has ≤ N−1 real zeros | `expsum_card_zeros_le`, `expsum_zeros_finite` | ExpSum.lean | FORMALIZED |
| (iii) finitely many switches; ties; shares | `prop_shapes_switches`, `switches_finite`, `ties_of_multiset_eq`, `maximizers_between_switches`, `prop_shapes_shares` | Shapes.lean, ExpSum.lean | FORMALIZED |
| "(such as images under `j ↦ uj mod p`), which are always tied" | `isVertex_permVec`, `mem_Poly_permVec`, `multiset_permVec`, `budget_permVec` | Symmetry.lean | FORMALIZED |

## Priority 5: earlier results

| Paper label | Lean name(s) | File | Status |
|---|---|---|---|
| `lem:avg` | `lem_avg` | Approach.lean | FORMALIZED |
| eq. (spectrum) | `spectrum`, `spectrum_avg` | Approach.lean | FORMALIZED |
| `lem:phase` | `lem_phase` | Approach.lean | FORMALIZED |
| `prop:coverage` | `prop_coverage` | Approach.lean | FORMALIZED |
| `thm:universal` | `thm_universal`, `thm_universal_unit`, `thm_universal_Cb` (generic); `thm_universal_prod` (product) | Approach.lean, Final.lean | FORMALIZED |
| `cor:dense` | `GammaP_one`, `dense_maximizer_iff`, `dense_constant`, `dense_network_equal_amp` (+ `sum_cov_offsets`) | Dense.lean, DenseNet.lean | FORMALIZED (the identification with Morwani et al.'s value is external; only the arithmetic identity is formalized) |
| `lem:cost` | `Cost.lem_cost`, `Cost.lem_cost_eq`, `Cost.lem_cost_eq_converse` | Cost.lean | FORMALIZED |
| `lem:chain` | `Cost.chain_lemma` | Cost.lean | FORMALIZED |
| `lem:minor` | `Cost.det_one_add_smul_expand` (+ `det_one_add_smul_Hmat_expand`) | Cost.lean | FORMALIZED (for the chain-vector matrix; general principal-minor expansion) |
| `lem:hadamard` | `Cost.lem_hadamard` | Cost.lean | FORMALIZED |
| `prop:esym` | `Cost.prop_esym`, `Cost.card_support_ge` | Cost.lean | FORMALIZED |
| `prop:gen` | `Cost.prop_gen` | Cost.lean | FORMALIZED |
| `lem:bernstein` | `Cost.lem_bernstein`, `Cost.Kr_eq` (+ `bernstein_scaling`) | Bernstein.lean, Cost.lean | FORMALIZED (exact constant) |
| `lem:sign` (product and power neurons) | `lem_sign`; `lem_sign_pow`, `lem_sign_pow_same` | Product.lean, Power.lean | FORMALIZED |
| product budget `C_b^{ν/b} = B_p` | `prod_budget`, `prod_budget_attained`, `CbProd_eq` | Product.lean | FORMALIZED (the equality characterization is NOT ATTEMPTED) |
| `prop:block` | `prop_block`, `blockNet_table`, `blockNet_interferenceFree`, `blockNet_margin`, `blockNet_normalized_margin`, `block_attains` | Product.lean | FORMALIZED |
| `thm:cyclic` | `thm_cyclic_upper`, `thm_cyclic_attain`, `thm_cyclic_value`, `thm_cyclic_structure` | Final.lean, Cyclic.lean | FORMALIZED |
| power neuron: `β_j = n!·β_j^{prod}`, `C_b^{ν/b} ≤ n! B_p`, `thm:universal` for power networks | `beta_powNeuron`, `betaT_powNeuron`, `pow_budget`, `thm_universal_pow` | Power.lean | FORMALIZED (the remark that the ν-block does not attain the bound for power networks is NOT ATTEMPTED) |
| ReLU neuron budget (`C_2 ≤ p^{-n}/2`) | `relu_budget_claim_false`, `relu_diag_claim_false` | ReluCounter.lean | **claim REFUTED** (formal counterexample, `p = 5`, `n = 2`; see "Errors found") |

## Errors / gaps found in the paper

1. **Appendix `app:models`, "ReLU neuron" paragraph (not a priority target).** The claimed bound
   `∑_{j∈F_p}|ĥ(j,…,j) ŵ(j)| ≤ (p/4)‖ω‖²`, i.e. `C_2 ≤ p^{-n}/2`, is false for `n = 2`.
   Integer counterexample at `p = 5`, `n = 2`: `u_1 = u_2 = (−2, 2, −2, 2, 0)`,
   `w = (−2, 0, 3, −3, 1)`: `‖ω‖² = 55`, `(p/4)‖ω‖² = 68.75`, but
   `∑_{j∈F_5}|ĥ(j,j)ŵ(j)| ≥ ½(p∑_a H(a)w(a) − Ĥ(0)ŵ(0)) = 72` (numerically ≈ 76.9).
   **Formally verified**: `relu_diag_claim_false` (`(5/4)·‖ω‖² < ∑_{j∈F_5}|ĥ(j,j)|·|ŵ(j)|`) and
   `relu_budget_claim_false` (`(5^{-2}/2)‖ω‖² < ∑_{j∈F_5}|β_j|` with the repo's `beta`, i.e. the
   paper's `C_2 ≤ p^{-n}/2` fails), both with standard axioms only (integer sums by kernel `decide`).
   The faulty step is the diagonal Parseval step: `ĥ(j,…,j)` sums over `p^n` inputs, so
   `∑_{j∈F_p}|ĥ(j,…,j)|² ≤ (p^{2n−1}/2)∑_t‖u_t‖²`, not `O(p)`.
   With `u_t = w = cos(2πa/p)` the ratio `∑|ĥŵ|/‖ω‖²` grows like `0.045 p²`
   (5.41, 43.2, 167.5 at `p = 11, 31, 61`), far above `p/4`.
   The Cauchy–Schwarz/Parseval/variance argument sketched in the paper gives `(p^n/4)‖ω‖²`, i.e.
   `C_2 ≤ 1/(2p)`; it agrees with the paper only at `n = 1`. The qualitative conclusion
   (ReLU weight decay is the `b = ν = 2` endpoint; `thm:universal` and `cor:dense` apply) is
   unaffected, only the constant changes.
2. No error was found in any priority result. Minor points (not errors): `prop:coverage`'s proof
   omits the case `A = 0` (trivial); `cor:few` needs `K ≥ 1` (for `K = 0`, `x = ln p/K` is
   undefined); in `prop:ablation` the hypotheses `j ∈ spt(y)` and `ε ≥ 0` are not needed.

## How to rebuild

```
# on a machine with elan:
cd heracles/lean
lake exe cache get          # prebuilt Mathlib v4.34.1 (never build Mathlib from source)
lake build                  # builds all modules (45 s wall on 8 vCPU, from a clean .lake/build)
lake env lean scripts/Axioms.lean   # prints the axioms of every main theorem
```
