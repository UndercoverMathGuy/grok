# Brief: close the remaining holes in the global-optimality proof (state as of 2026-09-22)

You are a research mathematician with full access to this repository. Everything lives in `heracles/`.
Read this file first, then the files it points to, in the order given in §1. Do not re-derive anything
marked PROVEN and do not retry anything marked DEAD. Your job is the holes in §3, in priority order.

Paper: ICLR 2027, "Explaining Feature Sparsity Through the Margin–Regularization Trade-off"
(`ABSTRACT.md`), full paper due 2026-09-25. Paper logic: (i) a theorem locates the max-margin network
and its feature count `K*`; (ii) exact closed-form evaluation of constructed networks exhibits it
(`LADDER.md`); (iii) Wei et al. 2019 Thm 4.1 (global minimiser of the regularised loss → max normalized
margin as `lambda → 0`) puts training dynamics out of scope, exactly as Morwani et al. 2311.07568 do;
(iv) empirics are trends only. The theory leg is now the paper's spine, so every hole below is a
sentence the paper has to hedge.

---

## 0. Setting and the target (unchanged; full detail in `GOPT_STATE.md` §"Setting")

- Task: full-table `n`-operand modular addition mod an odd prime `p` (all `p^n` rows, label = sum mod `p`).
- Network ("gated chain"): one hidden layer, no biases; neuron `i` has operand embeddings
  `E_j[i,·] ∈ R^p` (`j = 1..n`) and readout `W[·,i] ∈ R^p`. Logit `z_c(x) = Σ_i W[c,i] Π_j E_j[i,x_j]`.
  Homogeneity degree `nu = n+1`.
- Penalty `P_q(θ) = Σ_i ‖θ_i‖^q` (Euclidean norm of all of neuron `i`'s weights). `q = 2` is plain weight
  decay; `q = nu` is Morwani's `L_{2,nu}`. `2 ≤ q ≤ nu`, `r := q/nu ∈ (0,1]`, `s := nu/q = 1/r`.
- Normalized margin `gamma_q = margin / P_q^{nu/q}`; margin = min over rows and wrong classes of
  `z_correct − z_wrong`.
- Programme `Γ_s(p) = max { min_{D≠0} Σ_k b_k^s (1 − cos 2πkD/p) : b ≥ 0, Σ_k b_k = 1 }` (free shares).
  `K*` := number of non-zero shares at the optimum (see hole H3 for the definitional trouble).
- Atoms (no purity needed, `ELEGANT.md` §3): neuron `i` at frequency `k` contributes
  `c_ik = Π_j ê_ij(k) · conj(ŵ_i(k))` with slot phases `v_ijk = conj(ê_ij(k))/ê_ij(k)`. The clean
  Fourier coefficient is `G_k = Σ_i c_ik`; the same-frequency junk is `J_{k,T} = Σ_i c_ik Π_{j∈T} v_ijk`,
  `T ≠ ∅`. Per frequency, `h(u) = Σ_l c_l Π_j (1 + u_j v_lj) = Σ_T C_T u^T` is the generating polynomial;
  junk-free ⟺ `h` constant.
- Hypothesis `(H)`: `J_{k,T} = 0` for every `k` and every non-empty `T` ("no same-frequency junk").
- **TARGET (T)**: for every gated-chain network, every `2 ≤ q < nu`,
  `gamma_q ≤ (n+1)^{−(nu−q)/q} · B · Γ_{nu/q}(p)`, `B = 2(2p(n+1))^{−nu/2}`, with equality at the
  share-weighted `(n+1)`-neuron common-phase block (frequency `k`'s block scaled by `b_k^{1/q}`;
  neuron `l = 0..n` has operand phases a rotation of the `(n+1)`-st roots of unity and readout phase
  `n` times that).

---

## 1. Reading order

1. `GOPT_STATE.md`, `GOPT_STATE2.md`, `GOPT_STATE3.md` — the programme's history, agents 1–4, the
   7-agent fan-out, dead routes with breaking points.
2. `ELEGANT.md` — Theorem 1 (Feature Cost) and Theorem 2 (Margin Ceiling): the (H)-class result,
   PROVEN, paper-ready, lemma graph in §1.4, naming scheme in §4.2.
3. `VERIFY.md` — hostile referee on the (H)-class chain (no fatal issue; repair list R1–R7, C1–C10;
   the `K*` measurement in "X3 (continued)"; the equality-case identity in "LINK 7 (final)").
4. `ELEGANT2.md` — junk with `N = n+1` atoms (Theorem A), the `hmin`/aimed-output variable, the
   slot-conjugation twist, Corollary N1 (`K = 1`), and the refutations of every determinantal
   extension (§6).
5. `DEPTHA.md` — depth-1/2 aimed-output cost at `nu = 3`, every `N`, computer-assisted (279 cells).
6. `P3C.md`, `P3A.md`, `P3B.md` — junk at `K ≥ 2`: THEOREM P3C-1 reduction, `A*` (aiming depth),
   Riesz certificate on dissociate supports, Theorem R (no support-independent junk lemma), the
   duality barrier.
7. `ARCH.md` — the operator/Schatten architecture, the absolute chain matrix `M_+`, Conjecture (E+),
   Theorem UT, the `n = 2` admissibility LMI. `PROMPT_SOL.md` is a self-contained statement of the
   `n = 2` problem (E+) ⟺ (PW₂).
8. `P2.md`, `P2B.md`, `GLOBAL_OPT.md`, `GLOBAL_OPT3.md`, `GLOBAL_OPT4.md`, `LEMMA_ATTEMPT.md`,
   `ADVERSARY.md`, `P1A.md`, `P1B.md`, `P1C.md` — superseded routes; consult only for the dead-route
   lists and the numerical evidence.
9. `THEORY_nsum.md`, `PROOF_AUDIT.md` — the unrestricted upper bound (junk lemma L1, phase lemma L2,
   assembly L3, per-frequency AM-GM, `q`-budget) and its audit; the `x^n` family.
10. `LADDER.md` — the exact constructed ladders, the two `K*` definitions (§3), what the paper can
    say (§8), referee caveats (§9).

Scripts: `fvf_obj.py`, `nu4_program.py`, `kstar.py`/`kstar.json`, `lemma_*.py`, `gopt*_*.py`,
`adv_*.py`, `p1[abc]_*.py`, `p2_*.py`, `p2b_*.py`, `p3[abc]_*.py`, `verify_*.py`, `eleg_*.py`,
`eleg2_*.py`, `deptha_*.py`, `arch_*.py`, `ladder_*.py`. Outputs in `FINAL_RUNS/<NAME>/`.
Run with `uv run python ...` (add `--with scipy` / `--with cvxpy` / `--with mpmath` as needed).

---

## 2. What is PROVEN (do not redo; build on it)

| # | statement | where | verification status |
|---|---|---|---|
| P-1 | **Unrestricted upper bound** `gamma_q ≤ B Γ_{nu/q}(p)` for `2 ≤ q ≤ nu`, full table, any network. Two-sided only at `q = nu`. | `THEORY_nsum.md` §1, §3, §5; `PROOF_AUDIT.md` F8; `VERIFY.md` link 1 | re-derived by two hostile passes; brute-force checked |
| P-2 | **Theorem 1 (Feature Cost).** Junk-free `(KL)` system at one frequency, any `N`, any `n`, any `r ∈ (0,1]`: `Σ_l |c_l|^r ≥ (n+1)^{1−r} |A|^r`; equality (for `r < 1`) iff exactly `n+1` non-zero atoms of equal modulus with orthonormal chain vectors. | `ELEGANT.md` §2 (chain matrix, Cauchy–Binet, Hadamard, Bernstein); one-line alternative via Weyl + Rotfel'd in `ARCH.md` §1 | coordinator hand-verified; fp64 |
| P-3 | **Theorem 2 (Margin Ceiling).** Full table, any width, impure neurons allowed, `(H)` assumed, every `nu`, every `q ∈ [2, nu]`: (T) holds; the share-weighted block attains it; attaining networks have exactly `K*` features on `n+1` neurons each (Cor 2.1). | `ELEGANT.md` §3 (Props 6, 7, 8; Cor 2.1) | `VERIFY.md` links 1, 5, 7: no fatal issue; equality is an algebraic identity (50 digits) |
| P-4 | **Theorem A (aimed-output cost, `N = n+1`, any junk):** `Σ_l |c_l|^r ≥ (n+1)^{1−r} hmin^r`, `hmin = min_{T^n} |h|`. Lemmas A0 (collapse), A1 (chain identity), A2 (polydisc extension), A3 (`|C_T| ≤ 2(Re A − m)`), A5 (`hmin ≤ max_T |C_T|`, every `N`). Theorem B: C1/C2 at `n = 1`, every `N`. | `ELEGANT2.md` §3–§6 | written out, fp64-checked; not separately hand-verified by the coordinator |
| P-5 | **Corollary N1:** `K = 1`, pure single-frequency neurons, width `≤ n+1`, any junk: (T) up to `(1 − eps_p)^{−1}`, `eps_p = (n+1)^{(nu−q)/q}(2^n − (1 + cos π/p)^n)`; `eps_p < 1` from `p ≥ 7/17/43` at `nu = 3/4/5`, `q = 2`. | `ELEGANT2.md` §5 | as above |
| P-6 | **THEOREM P3C-1 + PROP P3C-2 (reduction):** for pure networks, every `p`, `K`, junk level: if a mixture `lam` over slot partitions satisfies `Σ_P lam_P R_P(pop, A*)^r ≤ 1` for every one-frequency population, then (T) holds. Block is a strict local max of the criterion. | `P3C.md` §3–§4 | verified end to end on 36 networks, exact at the block; **NOT hand-verified by the coordinator** |
| P-7 | **`A* ≥ 1/2` on dissociate supports** (Riesz product), `A*` = reciprocal one-sided Sidon constant of `{2k : k ∈ S}`. Theorem R: no support-independent multi-mode junk lemma (Dirichlet profile, `A* ≤ 1/(p−1)` on the full support). Theorem S: max-junk (one atom per frequency) closed for `(n+1)^{(nu−q)/q} ≤ 2^n`, i.e. `n ≤ 4` at `q = 2`. | `P3A.md` §4, §5 | hand-proved |
| P-8 | **Theorem DA (DEPTH-A at `nu = 3`):** `n = 2`, every `N`, any junk, depth `A = 1/2`, `r = 2/3` and every `r ∈ {0.67..0.99}`: `Σ_l |c_l|^r ≥ 3^{1−r} (m_A)_+^r`, `m_A = min_{|t_j| = 1/2} Re h`. Passes down to `A = 0.47`, fails at `0.46`. Cor DA-1: torus version (`A = 1`) for `r ∈ {0.6, 2/3, 0.75, 0.9}`, fails to certify at `r ≤ 0.55`. Cor DA-2: with P-6 + P-7, (T) for every pure network at `nu = 3`, `q = 2` on every support with `A* ≥ 0.47`. | `DEPTHA.md` | lemmas D1–D5 by hand; assembly step (ii) is an **fp64 279-cell monotone certificate** (`numpy.roots`, tol `1e-6` vs margin `2.7e-3`), **not interval arithmetic**; corner `R0` by hand |
| P-9 | **P3B duality barrier:** `conv(K_q) = K_nu`; any fixed test measure on rows gives a `q`-independent value; the weight-decay vs `L_{2,nu}` gap IS a duality gap. Consequence: the probe/measure must depend on the network. | `P3B.md` | hand-proved |
| P-10 | **ARCH:** Prop S0 (`D^r Σ|c_l|^r ≥ tr(M_+^r) ≥ ‖M‖_{S_r}^r ≥ Σ|λ_j(M)|^r`), Cor S1 (impure neurons are free on the **cost** side), (E+) ⟺ `det M_+ ≥ m^D` (Maclaurin), **(E+) at `n = 1` every `N`**, (E+) ⟺ (F) ⟺ (PW) (Sion twice), **Theorem UT** (`max{Re tr Y : Y upper-tri, [[B,Y],[Y*,B]] ⪰ 0} ≥ D det(B)^{1/D}`; its dual is the junk-free (E+)), **`n = 2` admissibility is the elliptope LMI** `R_ω ⪰ 0`. | `ARCH.md` §1–§3, §10 | written out; fp64 / cvxpy checks |
| P-11 | **P2B (`n = 2`, `N = 3`):** `27|c_1c_2c_3| ≥ exp(−5.2247 η²)` for `η ≤ 0.15` (Lemma M4 by hand, Lemma M5 by a 129,179-box interval branch-and-bound). Superseded in spirit by Theorem A (P-4) but is the only **quadratic** junk-relief statement proved. | `P2B.md` §4–§6 | computer-assisted |
| P-12 | Exact constructed ladders at the paper cells; the share-weighted block is junk-free to `1.2e−13` on the full table; `gamma_con(K)` closed form; both `K*` definitions tabulated; branch-and-bound certificate for the **share** problem on a given support. | `LADDER.md` §2–§5, T1–T7 | exact numerics |

---

## 3. THE HOLES, in priority order

Each hole: exact statement; what is known; where it breaks; what would close it; what to write if it
cannot be closed. "Closing" means a proof written out step by step and fp64-checked, OR a verified
counterexample, OR a strictly simpler equivalent statement with the reduction proved.

### H1 — Removing hypothesis (H): junk on `K ≥ 2` frequencies (the paper's main hedge)

This is the gap between P-3 ("in the class with no same-frequency junk") and (T) unrestricted. It
splits into five sub-holes, ordered by how much of the paper each one unlocks.

**H1a. Harden `nu = 3` so Cor DA-2 can be stated as a proposition.** Two mechanical items:
  (i) replace DEPTHA's fp64 step (ii) with interval arithmetic (`mpmath.iv`, or exact rational
  evaluation of the cubic's roots via a Sturm/positivity argument — the tools `chain`, `peel`, `S^r`
  are explicit; `deptha_gridcert.py` is the reference). The margin is `≥ 0.27 %` at the corner's edge
  and `≥ 1.2 %` elsewhere (`deptha_dense.py`, `deptha_cert4.py`).
  (ii) hand-verify THEOREM P3C-1's six steps (`P3C.md` §3.3) and PROP P3C-2 (§3.4) — nobody has.
  Watch for: the sign convention `W_{kD} = 1 − ω^{−kD}` (P3C's own fix), the profile identity
  (`P3A` P0, verified to `2.9e−15`), the claim that block depths can be chosen independently per
  (block, frequency), and that the Sion step is legal (compactness of the mixture simplex, continuity
  in the population).

**H1b. Certify `A*` on the paper's supports (non-dissociate).** `A* ≥ 1/2` is proved only for
dissociate supports (`P3A` §4.1). The `Γ_{3/2}`-optimal supports at the paper's primes are NOT
dissociate. Measured `A*` (upper bounds from an optimiser, `FINAL_RUNS/P3C/astar_papers.json`):
`p = 59: 0.386`, `71: 0.511`, `97: 0.496`, `113: 0.503`, `127: 0.497` at `K*`; range `0.381–0.614`
across `K* ± 1`. DEPTHA needs `A* ≥ 0.47` (so `p = 59` is NOT covered by DEPTHA and needs the
mixture threshold `A_lam = 0.304`, see H1c); P3C's mixture needs `A* ≥ 0.3044`.
  Known tool: a Lipschitz-mesh certificate works at `K = 2` (`A*(11,{1,4}) ≥ 0.5643` vs measured
  `0.6018`) but the mesh is `(2K−1)`-dimensional and does not scale to `K = 5..7` (`P3C` §7 item 3).
  Suggested route (`P3A` §7 item 2, "looks routine, not done"): a **perturbed Riesz product** —
  solve `ŵ(2k) = A e^{iφ_k}` by contraction, the collision terms being `O(A²)` times the number of
  additive representations of `2k` in `±{2k'}`; count the representations on each paper support
  explicitly (they are few) and get an explicit `A* ≥ 1/2 − c·(#collisions)·A²`-type bound. Even a
  bound of `0.35` on all five supports would let the paper state (T) unconditionally at `nu = 3`,
  `q = 2` on its own cells once H1c is proved; a bound of `0.47` does it with DEPTHA alone.
  A dual certificate is also available: `A* = min_{z≠0} [max_a Re Σ_i z_i ω^{−m_i·a}] / Σ_i|z_i|`, so
  any explicit non-negative density with the right moments is a lower bound — that is what must be
  exhibited, exactly or in interval arithmetic.

**H1c. Prove (PFP-lam) at `A = 1/2` (or at `A_lam`) for every `N`, `nu = 3` and `nu = 4`.**
Statement (`P3C.md` §9): there is a probability vector `lam` on the partitions of `{1..n}` with
`Σ_P lam_P R_P(pop, A)^r ≤ 1` for every one-frequency population,
`R_P(pop, A) = (n+1)^{(1−r)/r} (min_{|t_i| = A} Re Σ_U C_{B_U} Π_{i∈U} t_i)_+ / (Σ_l|c_l|^r)^{1/r}`.
Measured thresholds: `nu3q2: A_lam ∈ [0.30344, 0.30438]` (`M_m` family gives `0.30042`, nearly
extremal); `nu4q2: (0.45, 0.50]` (`M_m: 0.48106`); `nu5q2: M_m 0.5535 > 1/2` (fails, see H1d).
  - Singleton partition alone = (DEPTH-A) = `Σ_l|c_l|^r ≥ (n+1)^{1−r}(m_A)_+^r`. Threshold for
    singleton alone: `0.4284` at `nu3q2` (binding at the two-atom family `M_2`), `0.6300` at `nu4q2`
    (binding at `M_3`). So `nu = 4` **needs the one-block partition in the mixture** (`P3C` §5.1,
    `DEPTHA` §3, §6).
  - What exists: DEPTHA proves the singleton at `A = 1/2` for `n = 2` (computer-assisted). Nothing is
    proved at `n = 3`. `DEPTHA` §6: lemmas D1, D2, D3, D5 hold verbatim for every `n`; D4 needs
    `|E_k(I + L + U)|` bounds for the `4 × 4` chain matrix with interval junk `L` and mirror `U`
    (D2: `|U_I| ≤ |C_I| + 2w`, `w` = phase-misalignment excess — this is the ingredient that made the
    mirror controllable; `ELEGANT2` §6.1 had declared it "unconstrained").
  - Why the kill-point proof does not deform (`ELEGANT2` §7): at depth `A` the point
    `u_j = −A conj(v_{σ(j),j})` multiplies the targeted atom by `(1 − A)` instead of `0`; the linear
    system is no longer diagonal; no chain identity. Any proof must use `r` (the `e_k` ladder at depth
    `A` is FALSE: `h = 1 + u_1` has `m_{1/2} = 1/2 > 0` and `e_3 = 0`, `DEPTHA` §5.3).
  - Route hint: at `n = 3` extend DEPTHA's D4 (`E_2, E_3, E_4` of `I + L' + U'` with `|U'_I| ≤
    |C_I| + 2w`, `|C_T| ≤ d/A^{|T|}`), the 2-peel D5 to a 3-peel, and re-run the monotone cell
    certificate in `(d, w)`. The mixture over partitions then enters as: the one-block partition
    aims all three slots jointly (`R_{one-block}` uses `C_{123}`-type products only). Expect the
    certificate to be the deliverable, not a hand proof.

**H1d. `nu = 5, q = 2` (and `q ≥ 2.5` at `nu ≥ 4`).** The dissociate certificate expires
(`A_M(4, 2/5) = 0.5535 > 1/2`), exactly where Theorem S's boundary `(n+1)^{(nu−q)/q} ≤ 2^n` fails
(`P3A` §4.2, `P3C` §5.1). Unknown whether this is a real obstruction or an artefact of the partition
family. Also: at `nu = 5` the paper's own trained nets have negative quality (`LADDER` §8 F), so this
cell can be stated as `(H)`-class only. Low priority unless a uniform idea appears; do not spend more
than a token effort here.

**H1e. Impure neurons with junk.** Cost side is free (`ARCH` Cor S1: the neuron budget never used
purity). Margin side: P3C step 1 (the profile identity) FAILS for impure neurons because
cross-frequency modes appear (`P3C` §6, "IMPURE neurons"). `ELEGANT` Theorem 2 handles impure neurons
only under `(H)`, where the cross-frequency terms are killed by the junk lemma L1. Needed: an
aiming/test-density argument in which the modes `2k·1_T` of different frequencies do not collide, or
a way to charge for cross-frequency junk. Note Theorem R (`P3A` §5): the dilution by `K` is real for
support-independent lemmas, so the density must be support-adapted (product/Riesz over the support).
Open at every `nu`. If not closed, the paper's (T)-with-junk statements stay "pure single-frequency
neurons".

### H2 — Conjecture C1/C2: junk with `N > n+1` atoms on the full torus; (E+) at `n ≥ 2`

Statement (`ELEGANT2` §6): for every `N`, `n`, `r`: `Σ_l|c_l|^r ≥ (n+1)^{1−r} hmin^r` (C1), and the
ladder `e_k(|c|) ≥ C(n+1,k)(hmin/(n+1))^k` (C2). Numerically tight, min `1.000000009`, and every
adversarial minimiser collapses to `n+1` atoms of equal modulus.
  - Proved: `n = 1` every `N` (Theorem B); `N = n+1` every `n` (Theorem A); `k = 1` rung every `n`,
    `N` (Lemma A5); Re-form at `n = 2` for `r ∈ {0.6, 2/3, 0.75, 0.9}` by DEPTHA's certificate at
    `A = 1` (Cor DA-1) — fails to certify at `r ≤ 0.55` because the method lacks the "≥ 3 atoms"
    rigidity.
  - The clean route is `ARCH` Conjecture (E+): `det M_+ ≥ m_+^D` for the absolute chain matrix
    `M_+ = Σ_l |c_l| m_l m_l^*`, `m = min_{T^n} Re h`. It implies C1 (Re form) for all `N`, `n`, `r`
    in one line (S0 + Bernstein). (E+) ⟺ (PW): for every `D × D` PD `B` with `det B = 1` there is a
    probability measure `ρ` on `T^n` with `|Λ_ρ(v)| ≤ q_B(v)/D` on `T^n`. Theorem UT proves the
    phase-blind half (some interval polynomial `Λ` with constant term 1 works); **the whole remaining
    content is admissibility** (`Λ` must be a moment functional). At `n = 2` that is one LMI + one
    pointwise inequality in three complex unknowns (`ARCH` §10.7.6; self-contained in
    `PROMPT_SOL.md` §4). Second-order (PW) at `B = I + εH` is exact and tight (`arch_local2.py`).
  - Refuted probes (do not retry): `ARCH` §10.5, §10.7.1–10.7.4, `PROMPT_SOL.md` §5. The `hmin`
    version of (E+) is FALSE at `n = 2` (twisted block); only the `Re` version can be true.
  - Untried ideas: `PROMPT_SOL.md` §6 (Fejér–Riesz/Bochner on `q_B² ≥ 9|Λ|²`; Bernstein–Szegő measure
    of `B`; Schur-complement / matrix Cauchy–Schwarz; Cauchy–Binet over triples with `m ≤ averaged
    Gram`; homotopy in `B` from `I`).
  - Payoff: (E+) at `n = 2` gives C1 at `nu = 3` with no certificate and removes the width restriction
    from Cor N1. It does NOT give the depth-`A` version P3C needs (`ARCH` §10.8), so it does not
    close H1 by itself. Medium priority: elegant if it lands, but H1a–H1c unlock more of the paper.

### H3 — `K*` is computed, not certified; and it has two definitions

  - `Γ_s(p)` is a non-convex max–min. The **share** problem on a given support has a branch-and-bound
    certificate (`LADDER` §3 step 5, T7). The **support** choice is certified only where the
    enumeration is exhaustive: `K ≤ 8` at `p = 59`, `K ≤ 7` at `p = 71`, `K ≤ 5` at `p = 113`
    (`LADDER` §3 step 2). The `p = 113, nu = 3, q = 2` cell has `K*_free = 7 > 5`: **uncertified**.
    Everything on the large-prime axis (`PRIME_AXIS.md`, 69 primes to 1069) is heuristic. Direction of
    error: a missed support raises `Γ_s` and raises `K*_free`, never lowers it.
  - Two definitions: `K*_eq = argmax_K M_K(p) K^{−s}` (equal-amplitude ladder, what `kstar.py`
    computes and `THEORY_nsum` tables quote) vs `K*_free` = smallest number of active shares among
    rungs reaching `Γ_s` (thresholds `t_a`, `t_v` in `LADDER` §3). The theorem (Cor 2.1) pins
    `K*_free`. They differ at `p = 11, nu = 3` (2 vs 5) and agree at `p = 23, 31` (`VERIFY` X3
    continued). Adjacent rungs are within 1–8 % (`LADDER` T2), so `K*` is often a **band**.
  - What would close it: (i) an exact or certified global bound on `Γ_s` at `p = 113, nu = 3`
    (e.g. an LP relaxation over ALL supports, or a symmetry argument that bounds the value of any
    `K ≥ 6` support), or (ii) a structural theorem about the optimal support (e.g. that the maximiser
    is a maximal-coverage set in the sense of `M(S)`), or (iii) at minimum a clean statement of the
    band and the threshold dependence for the paper. Also (iv): is `K*_free` well-defined, i.e. is
    the maximiser's support size unique? `VERIFY` X3 item 3 says this is needed for "the feature
    count" to be a single number; nobody has addressed it.

### H4 — Scope items that are NOT holes but must be stated (verify and record; no proof expected)

  - **Full table only.** Prop 6's orbit averaging needs every row (`ELEGANT` §3, `PROOF_AUDIT` F6,
    `VERIFY` link 1). Every `frac < 1` cohort in the repo is outside the theorem. The `LADDER` trends
    are full-table. If you see a way to extend L1 to a training subset (e.g. a random subset with
    high probability, via the orbit structure), say so; otherwise state the scope.
  - **Wei et al. 2019 Thm 4.1** applies (`(Σ_i ‖θ_i‖^q)^{1/q}` is a norm for `q ≥ 1`, `f` is
    `nu`-homogeneous, no biases; `PROOF_AUDIT` F8) but only to the **global minimiser** as
    `lambda → 0`. Lyu–Li does NOT apply for `q ≠ 2` and at `q = 2` only to KKT points. `LADDER` §5.2:
    trained endpoints at `q = 2` are local, non-global minima in 6 of 7 cells. So the abstract's
    "we empirically show that the normalized margin attained by these networks peaks at the predicted
    optimum" is true for constructed nets, not for trained nets — flag it.
  - **`q = nu`** is the only two-sided unrestricted theorem (Morwani's endpoint; `PROOF_AUDIT` F8).
  - **`q > nu`**: unbounded only with unbounded width (`PROOF_AUDIT` F8).

### H5 — The `x^n` family has no attainment result

`THEORY_nsum.md` §3: the unrestricted upper bound holds for `x^n` (the sparse programme is
unrestricted for `nu = n+1`). But the best known junk-free `x^n` block attains only `0.408` of the
bound at `nu = 3` vs the flat solution's `0.507–0.571` (`PROOF_AUDIT` F1 table), so **the flat
(Morwani) solution is not excluded for `x^n`**, and Theorem 1 does not apply as-is: `x^n` junk is not
multilinear (`u²v`-type terms, `THEORY_nsum` §3, `PROOF_AUDIT` F7). Tied `x^n` has no junk-free
population at all (`PROOF_AUDIT` F7). `LADDER` §9 item 5: the cyclic blocks `N = 6, 18, 54` are the
smallest cyclic ones, smaller non-cyclic ones may exist.
  - Needed: either a Feature Cost theorem for the `x^n` junk system (the analogue of `(KL)` with the
    non-multilinear patterns), or a better construction, or a clean statement that the `x^n` results
    are upper-bound + ladder-shape only ("the count is a property of the norm and degree, not the
    activation" — `LADDER` §8 A is proved algebra and survives).
  - Low priority for the proof; high priority for the paper's wording.

### H6 — Small open items, each one paragraph if closed

  - Equality class of Theorem A (`ELEGANT2` §3.4): is it exactly the block's orbit under the `2^n`
    slot-conjugation twists and gauge? Not needed by anything; nice to know.
  - `VERIFY` C2: the Key Lemma's "same rotation pattern across slots" equality case is proved only at
    `n = 2`. Superseded by Theorem 1's equality case (Cor 1.2), which is for every `n` — confirm that
    Cor 1.2's proof (`ELEGANT` §2.5) really does not use it.
  - P2B F2B-4: the quadratic junk-relief `n ≥ 3`. Superseded by Theorem A for `N = n+1`; only
    relevant if some route needs an `η`-parametrised curve. Skip unless needed.
  - Theorem 1' (`ELEGANT` §2.8): the interval sub-system version — only `C(n+1,2)` interval patterns
    are used, so Theorem 1 is true under the weaker hypothesis "interval junk vanishes". Check whether
    the paper should state it that way (it strictly enlarges the `(H)` class: `(H_int)` has
    `(p−1)·C(n+1,2)/2` conditions instead of `(p−1)(2^n−1)/2`).
  - Constants and names (`VERIFY` X1, C1–C10; `ELEGANT` §4.1): `Γ_{3/2}(23) = 0.402933388119`;
    block phase rule is "a rotation of the `(n+1)`-st roots of unity" (not `2πl/(n+1)` at even `nu`);
    two different "Lemma T"; `build_chain` equal-amplitude does NOT attain (T) when shares are unequal
    (`ADVERSARY` §5). Do not introduce new names; use `ELEGANT` §4.2's scheme.

---

## 4. DEAD ROUTES — consolidated, with breaking points (do not repeat)

From `GOPT_STATE.md` §"Dead routes": Morwani-style dual certificate (lossy by exactly
`N^{(nu−q)/q}`); domination `Σ|c| ≥ (n+1) max|c|` (max/sum → 1/2); capped-mass / weak-majorisation
(FALSE at `θ = 1/3`, so the proof must use `r`); peeling with sup-norm test polynomials (`D_1(2) <
2/3`); one averaged test measure (Theorem O); product test measures; gauge symmetrisation (circular
at every level, `ELEGANT2` §6.4 measured to `1e−15`); induction on `n` by shifting a slot;
Gram/Carathéodory–Fejér; Mahler; Hadamard crowding; Carathéodory support counting.

From `GOPT_STATE2.md`: product-of-three-largest; plain `H ≥ log(n+1)`; monomial-rotation symmetry;
diagonal relaxation (gets 2 not `n+1`); mixtures of `(n+1)`-atom solutions; equal-moduli rigidity;
six-angle Lipschitz B&B; the chord criterion (REFUTED in closed form at `nu = 4, 5`, `P3C` §2;
knife-edge tautology at `nu = 3`); support-independent multi-mode junk lemma (Theorem R);
hypercontractivity (dead by `√K`).

From `ELEGANT2.md` §6: `|det(I + sM)| ≥ (1 + sm)^D` for the complex chain matrix (infimum 0);
`Re spec(M) ≥ m` (ratio −350); `Herm(M) ⪰ mI` (−1500); the diagonal restriction `u_1 = … = u_n`
(short by `2^{1/3}` vs `3^{1/3}`); Theorem A does not improve the `η` curve (it replaces the variable).

From `DEPTHA.md` §5: depth via splitting `1 + Auv` into two torus atoms (costs `2^{n(1−r)}`); bridge
from Theorem A `m ≥ m_A − E` (no `r`-free bridge); the `e_k` ladder at depth `A` (FALSE); any finite
family of modulus-linear constraints with depth-admissible probes (cannot be tight); Hermitian
variants; three-circles/log-convexity in radius (wrong direction).

From `ARCH.md` §7, §10.5, §10.7: fixed or Haar-averaged Hermitian compressions (not sharp at
`n = 1`); non-chain character spaces or shift `≠ 1/2`; `h`-determined rank-one kernels (mirror is
essential); `2^n`-dimensional full monomial space; `(S)` with free `U` at `r < 0.44`; `λ_min(M_+) ≥ m`
(ratio `1e−5`); `hmin`-form of (E+) at `n = 2`; diagonal similarity; the naive `n = 1` probe
`2B_I/tr B` at `D ≥ 3`; geometric-mean/Cholesky probes; LS chain probe; "feasible + `tr Y ≥ D` ⇒
admissible" (`|ω| = 1.25`); "UT-optimal `Y*` is admissible" (fails at cond 193/527); the
elliptope-constrained matrix programme (`T_ell → 0.9` at cond `5e8`); Riesz/half densities;
interval-supported SOS densities; `R_ω = (1−a)I + a·corr(B)` and `−corr(B^{−1})` mixtures.

From `P3C.md` §6: single partition alone; mixing densities rather than strategies; aiming at a
sub-support; the profile identity for impure neurons.

From `P3A.md` §6 and `P3B.md`: fixed row measures (duality gap); variance/fourth-moment junk lemma
(provably ≤ half of Lemma Q); `(x, η)` bookkeeping beyond `Ṽ(p)`.

---

## 5. Evidence that (T) is TRUE unrestricted (so counterexample hunting is low-yield)

Whole-network smooth NLP over raw weights, no restrictions, `p = 5, 7` (`n = 2`), `p = 5` (`n = 3`):
ratio `1.000000000000` in 16/16 cells, never above (`GOPT_STATE` §"Evidence"). `ADVERSARY`: 40
whole-network cells up to `p = 23`, block excluded from every start, ~2,400 ascents, max ratio
`0.999999999990`. `VERIFY`: two ascents inside `(H)` with the block excluded, best `0.9999999998`;
from-scratch (KL) counterexample search over `N = n+1 .. 2^n+3`, min exactly `1.0000000000`.
`ELEGANT2` §8: C1 min `1.000000009`, C2 min `1.000000179`, minimisers collapse to `n+1` equal atoms.
`ARCH` §10.4: `det M_+/m^D` min `1.0000118` at `n = 2` (at the block), `1.11–1.56` at `n = 3` (weak
search). `eleg2_net.py`: junk-carrying `K = 1` networks never exceed the unrelaxed target, equality at
the block. Free CE + weight-decay training lands at `0.4–0.6` of the target with 40–60 % junk but at
a strictly higher training objective than the construction (local, non-global minima, `LADDER` §5.2).

If you do find a violation anywhere, it is the most important result in this file: verify it with a
brute-force margin over all `p^n` rows and all `p` classes, in fp64 and mpmath, before reporting.

---

## 6. Rules

- Code: `uv run python ...`. LOCAL CPU ONLY. No pods, no GPU, `$0`.
- Do not edit any existing file; do not commit. Write only new files: `HOLES_<yourtag>.md` for the
  write-up and `holes_<yourtag>_*.py` for scripts; outputs in `FINAL_RUNS/HOLES_<yourtag>/`.
- No background processes left running at hand-back.
- Never call something proven unless every step is written out and fp64-checked (interval or exact
  arithmetic where the margin is thin). A verified counterexample or a precise "this route dies
  here, at this number" is a valid result. If you find an error in inherited material, say so, with
  the line.
- Write-up style: define every symbol, plain-language reading with each formula, no bare ML notation.
  Use `ELEGANT.md` §4.2's naming scheme; do not coin new names for old objects.
- Priority: H1a → H1b → H1c → H3 → H2 → H1e → H5 → H1d → H6. Stop and write up as soon as a hole is
  closed or provably blocked; do not silently move to the next one.
- Final message: exactly what is PROVEN / REFUTED / PARTIAL / OPEN per hole, the file each result is
  in, what the paper can now say verbatim, and confirmation that no background jobs remain.
