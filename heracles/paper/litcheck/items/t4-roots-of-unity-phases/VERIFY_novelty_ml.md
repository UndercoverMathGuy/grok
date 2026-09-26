# VERIFY (novelty_ml): T4 roots-of-unity phases (prop:block and the appendix remark)

Verifier verdict: PARTIALLY_KNOWN still holds, but more of it is known than the checker said. The ML literature already has the nu=3 case. **Not upheld as written.** The checker missed the closest ML prior art. Its "what is ours" statement and its recommended text both need to change.

## Missed ML prior art (fetched and read this session)

### 1. Tian, "Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets", NeurIPS 2025 poster
- arXiv 2410.01779 v4, comment "NeurIPS 2025 camera ready". OpenReview forum tD7MLq0dbZ, venue "NeurIPS 2025 poster".
- Setting: 2-layer quadratic-activation nets on Abelian group multiplication, including modular addition. Loss is MSE, not margin.
- Sum potentials:
  - R_c = {r_{k1 k2 k}, not all equal} are exactly the sign-flipped copies (ābc, ab̄c, abc̄) of the trilinear term.
  - R_g = {r_kkk} is the wanted term.
- Table 1: u_syn = [ω3, ω3, ω3] with ω3 = e^{2πi/3}. Section 5: z_syn := ρ(u_syn) = u² + u + 1, i.e. 3 hidden nodes per frequency with coefficients ω3^j on a, b and c.
- The ābc / ab̄c / abc̄ potentials evaluate to ω3, and the polynomial kills them.
- Fig. 2 (bottom): "Now each frequency has exactly 3 hidden nodes, corresponding to the solution z_syn". This holds when the self-interaction constraint R_* is dropped.
- Lemma 6 (App.) says: "With small L2 regularization, all per-frequency order-3 canonical solutions ... that satisfy R_c and R_g are [1, α_p ω3, β_p ω̄3]", with α_p, β_p = ±1. Eqs. (91)-(95) are a minimal-norm problem whose optimum is equal magnitudes m_j = 1/ε with phases 2π/3 and 4π/3.
- The proof of Lemma 6 (Eq. 85) says "all cross terms are gone since Σ_j ω3^j = 0". This is the roots-of-unity filter.
- Numerical check (scratchpad chk.py):
  - Tian's z_syn trilinear part, Σ_j cos(θa+2πj/3) cos(θb+2πj/3) cos(θc−2πj/3), equals (3/4) cos(θa+θb−θc).
  - This is the same table as the paper's ν=3 block with α_l = πl/3.
  - The two blocks are identical up to the trivial sign symmetry (flip two inputs by π).
- The global order-6 solution z_F6 = z_syn * z_ν (Corollary 2) handles the quadratic self-terms. It is a cube-root × sign product: 6 nodes per frequency.
- The perfect-memorisation solution (Corollary 3) uses d-th roots of unity.
- **Consequences:**
  - The ν=3 roots-of-unity block is known in ML.
  - Its sign-flip cancellation is known.
  - A minimal-norm characterisation within the 3-node class is known.
  - The checker's claim "I found no ML paper that uses the rank-nu separated form" is false at ν=3.
  - The paper does not cite Tian anywhere (grep of refs.bib and sections).
  - The sweep (LIT_ml.md l.64) lists Tian but did not connect it to this item.

### 2. Shutman, Louidor, Tessler, "Learning words in groups: fusion algebras, tensor ranks and grokking", arXiv 2509.06931 (Sep 2025, cs.LG)
- The Hadamard (HD) model f = C^T(Ax ⊙ By) is a two-operand product network.
- Prop. 5.2 gives rank(δ_{G,ϕ⊗3}) ≤ 3 m_{d/2} for type-II bscs. For Z_p (2-dim rotation bsc) this is rank ≤ 3 per frequency, so 3 product neurons per frequency suffice.
- The proof uses the Gauss/Karatsuba three-multiplication trick (γ, δ, ε), not equal-amplitude roots of unity. It is not norm-optimal.
- Relevance:
  - It is ML prior art for "ν neurons per frequency suffice in a product network" at ν=3.
  - The roots-of-unity form is not in it.
  - The norm-optimality / equality-case claim is not in it.

### 3. Doshi, He, Das, Gromov, "Grokking Modular Polynomials", ICLR 2024 Workshop BGPT, arXiv 2406.03495
- Analytic solutions for many-term modular addition with power activations.
- It still relies on random i.i.d. phases: "all other terms will vanish due to random phase approximation".
- It is a contrast citation alongside Gromov, not prior art for exact cancellation.

## Other searches (no hit)
- Kunin et al., AGF (NeurIPS 2025, arXiv 2506.06489) cancels square terms by polarisation sign pairs, z1 z2 = ¼(z1+z2)² − ¼(z1−z2)². No roots of unity.
- He, Wang, Chen, Yang, arXiv 2602.16849, Def. 4.1(iii): Σ_{m∈N_k} exp(iιϕ_m) = 0 for ι ∈ {2,4}. The checker described this correctly. It is quadratic-net majority voting, and the 2x/2y peaks are left uncancelled (Prop. 4.2).
- Kam et al., arXiv 2607.13749 uses roots of unity as the input encoding for holomorphic activations. That is a different use.
- Huang and Li, arXiv 2511.23443 (sine activations, width 2). Not relevant.
- arXiv API queries, all with no further hits:
  - "roots of unity" AND "modular addition"
  - "roots of unity" AND grokking
  - modular addition AND phase AND neurons
  - modular addition AND margin
  - modular addition AND "quadratic activation"
  - grokking AND "tensor rank"
  - "sum potentials"
- WebSearch budget was exhausted, so the searches ran through the arXiv API and direct fetches.

## The checker's confirmed citations, re-verified
| Citation | Check | Result |
|---|---|---|
| mohlenkamp2005trig | Crossref: Math. Intelligencer 27(2):65-69, 2005, doi 10.1007/BF02985795, authors Mohlenkamp and Monzón. Preprint sine.pdf read. | Confirmed |
| gromov2023grokking | arXiv 2301.02679: Eq. (12) φ1+φ2=φ3, Eq. (15) random uniform phases, and "each frequency appears several times with different random phases". | Confirmed |
| morwani2024feature | arXiv 2311.07568 App. F.2: 8 neurons per frequency with the listed phase triples. | Confirmed |
| li2025nsum | PMLR v258 li25b, pp. 523-531, authors confirmed. App. E.3 Lemma E.3: 2^{2k−1} neurons per frequency. | Confirmed |

Details for each:
- **mohlenkamp2005trig.** Thm 2 is the rank-n identity for sin of a sum. Fig. 1 (right) uses α = 0, π/3, 2π/3 with "amplitude equidistributed". The Remarks say it extends to cos.
  - By rephasing, the equally spaced cos case has a uniform shift within each term. This matches the paper's block up to signs, so the checker's claim is correct.
- **morwani2024feature.** Its network is (u(a)+v(b))² w(c), a power network. The sign pairs also cancel the u² w and v² w self-terms.
- **li2025nsum.** 2^{2k−1} with ν = k+1 gives 2^{2ν−3}. This is also a power network (sum)^k w.

## Issue with the recommended contrast
Morwani (8) and Li (2^{2ν−3}) count neurons for power networks, which must also cancel self-interaction monomials. The paper itself says the product ν-block does NOT give an attaining power network (appendix, "Power neuron"). "8 or 2^{2ν−3} vs ν" is therefore an apples-to-oranges comparison unless the text names the model classes. Tian's two cases make the fair comparison:
- 3 nodes (z_syn) when the self-terms are dropped;
- 6 nodes for the full quadratic net.

## What remains ours
- **General ν:** ν-th roots of unity for n = ν−1 operands. Tian covers two operands only.
- **Uniqueness over all widths:** the block is the unique equality case of Lemma cost among ANY number of neurons, so ν neurons per feature are forced, not just sufficient. Tian's Lemma 6 fixes the node count at 3.
- **Objective:** the block is the exact L_{2,b} max-margin optimum within the interference-free class. Tian's setting is MSE plus small L2.

## Revised recommended text (cyclic.tex after l.58)
"For $\nu=3$ this block is the three-node solution $z_{\rm syn}$ that \citet{tian2025composing} builds from cube roots of unity for quadratic networks. Tian also shows it has minimal norm among three-node solutions. It is the equally spaced case of the rank-$\nu$ separated representation of a trigonometric function of a sum \citep{mohlenkamp2005trig}.

Constructions for power networks must also cancel self-interaction terms, and they use sign patterns:
- $6$ neurons per frequency \citep{tian2025composing};
- $8$ \citep{morwani2024feature};
- $2^{2\nu-3}$ \citep{li2025nsum}.

\citet{gromov2023grokking} and \citet{doshi2024modpoly} cancel only approximately, with random phases. What is new here is general $\nu$, and that the block is the unique equality case of Lemma~\ref{lem:cost} over any number of neurons."

Add the bib entries tian2025composing (and optionally doshi2024modpoly and shutman2025words). Do not claim the identity or the ν=3 block as new.
