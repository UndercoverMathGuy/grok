# Global-optimality programme: addendum, state as of 2026-09-21 midday

Read `GOPT_STATE.md` FIRST (setting, atom picture, agents 1–4, problems P1/P2/P3, dead routes, rules). This file adds the results of the 7-agent fan-out of 09-21. Write-ups (all in heracles/): ADVERSARY.md, P1A.md, P1B.md, P1C.md, P2.md, P3A.md, P3B.md; scripts adv_*.py, p1a_*.py, p1b_*.py, p1c_*.py, p2_*.py, p3a_*.py, p3b_*.py; outputs FINAL_RUNS/{ADV,P1A,P1B,P1C,P2,P3A,P3B}/. NONE of the fan-out results has been hand-verified by the coordinator (only the Key Lemma and Lemma Q have).

## Code correction (ADVERSARY)
Equal-amplitude `build_chain` does NOT attain the target when the programme's optimal shares b_k are unequal (ratio 0.677 at p=11). The equality case of (T) is the SHARE-WEIGHTED block: frequency k's neurons scaled by b_k^{1/q}. Γ_{3/2}(23) = 0.402933388.

## P1 (sharp constant, every N) — claimed CLOSED at q=2 for ν=3,4,5
Three independent derivations of the same argument:
- Split atoms into the top n+1 (set S) and a tail of total modulus τ (or residual R). Robust Key Lemma on S — P1C's "Lemma T": Σ_{l∈S}|c_l|^r ≥ (n+1)^{1−r} max(|A| − 2^n R, 0)^r (no Theorem B needed). ADVERSARY's version uses Theorem B on the top n+1.
- Peeling (Theorem E, κ_1=1/2, κ_2=3√3/8, κ_3=49/72) forces the tail to be non-trivial unless the top atoms already pay; the tail's own price is max(τ^r, τβ^{r−1}) (β = largest tail modulus) — P1A calls ITS staircase tail price "Lemma T" (NAMING CLASH with P1C's Lemma T).
- Result: a 2-variable programme G(β,τ) ≥ target. ADVERSARY: margin ×1.022 at n=2,3. P1C: float64 cell-wise certificate, ratios 1.032 (n=2), 1.034 (n=3), 1.011 (n=4); n=5 short by 1.3%. P1A: RIGOROUS interval-arithmetic certificate (mpmath iv, 30 digits; 101 boxes n=2, 63 boxes n=3), conditional on inherited Theorems B and E. P1B Prop B1: cost ≥ (n+1)^{1−r}(1−2^n E)_+^r + E^r; r-windows r ≤ 0.84333 (n=2), 0.62678 (n=3), 0.43487 (n=4, i.e. q ≤ 2.174) ⇒ P1 for every N at q=2, ν=3,4,5 ⇒ Theorem D's F = 1 there.
- P1B correction to GLOBAL_OPT3: D_h(n) is a max over Σd_i ≤ n, so only κ_1..κ_3 appear and all constants are exact algebraic.
- Certified r*: P1A r*(2) ∈ [0.84,0.86), r*(3) ∈ [0.60,0.667); P1C r* ≥ 0.797/0.613/0.423.
- OPEN window r ∈ (r*,1): reduces (P1A) to H(q) + (r_0/(1−r_0)) log S ≥ log(n+1) — at S=1: a probability measure on T^n killing every non-trivial multi-affine character has Shannon entropy ≥ log(n+1). P1B §11: closes iff a*(n) ≤ r*/(1−r*) = 5.383 (n=2), 1.679 (n=3); measured a* ≈ 0.13 / 0.12–0.14 (sampled, NOT a certificate). Residue: H(x/S) ≥ log 3 − 5.383 log S for 0 < S−1 < 0.1472; any lemma Σ(x_l−1/3)² ≤ C(S−1) finishes it. Max-weight route DEAD ((max−1/3) is Θ(√δ), const ≈ 0.222).
- Structure facts: P1A Prop A: n=2, N=4 is an affine dependency of 4 torus points in C², |c_l| ∝ triangle areas; symmetry group only 96; constant 2^n is ATTAINED for a free probe. P1B: (KL) ⇔ complex cubature rule exact for multi-affine polynomials; multiplicative structure essential (free fourth row gives 0.925); Prop A1: on the aligned class max x_l ≤ A_d (A_1=1/2, A_2=1/3) ⇒ P1 all r on aligned class n ≤ 2. P1C: value identity F = (Σ|Λ(v_l)|^{−σ})^{1−r}; an unequal stationary branch exists (0.3368,0.3368,0.1636,0.1636).
- Refuted: product-of-three-largest conjecture (1/64 vs 1/27); plain H ≥ log(n+1); monomial-rotation symmetry; diagonal relaxation (gets 2 not n+1); mixtures of (n+1)-atom solutions; equal-moduli rigidity; six-angle Lipschitz branch-and-bound cannot terminate. Earlier "1.0000009" was optimiser noise (best 1.00000005).

## ADVERSARY: (T) not broken
40 whole-network cells up to p=23, block excluded from every start, ~2,400 ascents: max ratio 0.999999999990. Refuted "purification never lowers the margin" (but impurity is cost-free exactly at q=2).

## P3B (duality)
- Thm P3-1: conv(K_q) = K_ν. Thm P3-2 DUALITY BARRIER: for any fixed measure μ on (row, wrong class), V_q(μ) is q-independent and a width-1 junk neuron wins it; inf_μ V_q = B·Γ_1. So the gap between weight decay and L_{2,ν} IS a duality gap; any averaging costs.
- True maximisers at p=5,7 have zero same-|k| junk, but KKT certifies nothing (the maximiser realises 41% of the Lagrangian's value). "Junk-free at the same cost" formalisation is false.

## P3A (large junk)
- Profile reduction (exact). P3 reduces to ONE constant: ρ(η) := [1 − A(η)/κ0]/η ≤ ρ_need = (1−(n+1)^{−(ν−q)/q})/√(2^n−1) on (0, η*], where A = Cl2_r^{1/r}, κ0 = (n+1)^{(1−r)/r}, η* = √(2^n−1) ("the chord criterion": A above the chord from (0,κ0) to (η*,1)). ρ_need: 0.2440 (ν3,q2), 0.1139 (ν3,q2.5), 0.2835 (ν4,q2), 0.1399 (ν4,q3), 0.2351 (ν5,q2).
- **THE CHORD CRITERION IS REFUTED AT ν=4** (P2's `FINAL_RUNS/P2/chordcheck.json`, measured Cl2_r is an upper bound on the true curve so this is a genuine refutation of the sufficient condition, NOT of (T)): ν4,q2: sup ρ = 0.3837 > 0.2835 at η = 1.24; ν4,q3: 0.1500 > 0.1399 at η = 1.35. At ν=3 (q=2, 2.5) sup ρ equals ρ_need to 11 digits, attained at the endpoint η* — i.e. zero room. So the K-free route is dead at ν=4 and knife-edge at ν=3.
- Thm R: a support-independent multi-mode junk lemma is FALSE: ρ ≤ 1/(p−1) on the full support; extremal junk = Dirichlet kernel; the 1/K dilution is real; hypercontractivity dead by √K.
- Thm S: at max junk (one pure neuron per frequency) γ_q/B ≤ (1−A*)^n·clean(S); needs A* ≥ 0.2402 at ν3,q2; A* ≥ 1/2 proved only for dissociate supports (p=11); measured A* 0.34–0.60 for p=11..23. Condition (n+1)^{(ν−q)/q} ≤ 2^n holds at q=2 for n=2,3,4, fails from n=5. Max-junk nets at p=11,13 have margin exactly 0 (conjecture).
- OPEN: general junk at K ≥ 2; rigorous A* for non-dissociate supports.

## P2 (junk cost curve)
- Thm P2-1 (proved, N=n+1, all η): moduli satisfy (C1) Σp_l ≥ 1, (C2) Πp_l ≥ ((1−δ)_+/(n+1))^{n+1}, (C3) p_l ≥ (1−δ)_+/2^n, δ = √(2^n−1)η. Tight at η=0 but LINEAR: sup ρ = 0.709 vs needed 0.244 (ν3,q2) ⇒ proven junk threshold is 0.
- Thm P2-2 (proved) quantitative Hadamard: V² ≤ (n+1)^{n+1} exp(−(n+1)^{−3} Σ_{k=1}^n (n+1−k)|P_k|²), P_k = Σ_l v_l^k.
- Thm P2-3 reduction + QPL (CONJECTURE): Π(η) := min (n+1)^{n+1}Π|c_l| ≥ exp(−c_n η²), c_n = (n+1)²/(4n) (Prop P2-4: exact at n=2 by hand, eigenvalue-verified n ≤ 6). r-free, p-free, K-free. Measured (−log Π)/η² at n=2: 1.127 (η=.003) → 1.20 (.1) → 1.83 (.44) → 2.74 (.55); Π = 0 by η ≈ 0.58 (n=2), ≈0.4 (n=3).
- Exact chain identity: log[(n+1)^{n+1}Π|c_l|] = (n+1)H + Δ, H = kill-point average of log|h|, Δ = Hadamard defect ≥ 0. Kill-point moments: m_{j} = −conj(S_j)/(n+1); m_{12} = (conj(S_1)conj(S_2) − Z)/6 at n=2, Z = Σ_l conj(v_l1 v_l2). MISSING LEMMA: Σ_{|T|≥2}|m_T|² ≤ const·(Δ + η²). Obstruction example v_l = (w^{−l}, w^{l}): Δ=0, |m_12| = 1/2, but forces η ≥ 1 identically.
- Cor P2-5 (from QPL): Cl2_r ≥ (n+1)^{1−r}exp(−rCη²/(n+1)). Conditional (JT) thresholds from measured Π at ν3,q2, K=2..6: 0.511/0.426/0.369/0.312/0.284.
- F2-2: P2 at N>n+1 reduces to N=n+1 (min at τ=0) — bookkeeping unfinished. Dead: chord/concavity via mixing (F2-1), ℓ2 peeling (F2-3, slack at block), improving R ≥ 1−δ (F2-4, sharp), Jensen over kill points (F2-5, not Haar).

## Net position
On the class with no same-|k| junk, (T) is exact at q=2, ν=3,4,5 (modulo verification). Open: junk in the mid-range at K ≥ 2 (P3) — needs a new idea; QPL's cross-slot lemma (P2); the r→1 window; ν ≥ 6; x^n family.
