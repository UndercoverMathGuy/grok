# Prompt for an external solver (self-contained; no repository access assumed)

You are a research mathematician. Below is a sharply posed open inequality from a machine-learning theory paper (max-margin analysis of a small neural network on modular addition). It has resisted several days of attack by strong automated provers, which have (a) proved it in special cases, (b) reduced it to an equivalent finite-dimensional matrix statement, and (c) refuted every "natural" proof route. Your job is to prove it, refute it, or reduce it further. Everything you need is here. Everything marked PROVED has a written proof and has been numerically stress-tested; everything marked REFUTED has an explicit counterexample. Please do not re-derive refuted routes.

## 1. Objects

Fix an integer n ≥ 1 and write D = n+1. A **population** is a finite list of N ≥ 1 **atoms**: complex numbers c_1, …, c_N and, for each atom l, n unimodular complex numbers v_{l1}, …, v_{ln} (|v_{lj}| = 1). N is arbitrary (any N ≥ 1).

- **Generating polynomial** (multi-affine in u = (u_1,…,u_n)):
  h(u) = Σ_l c_l Π_{j=1}^n (1 + u_j v_{lj}).
  Writing C_T = Σ_l c_l Π_{j∈T} v_{lj} for T ⊆ {1..n}, h(u) = Σ_T C_T u^T. C_∅ = Σ_l c_l is the "clean" coefficient; the others are "junk".
- **Aimed output**: m = min over the torus u ∈ T^n (|u_j| = 1) of Re h(u). (m ≤ Re C_∅. If all junk vanishes, h ≡ C_∅ and m = Re C_∅.)
- **Chain vector** of atom l: m_l = (1, v_{l1}, v_{l1}v_{l2}, …, v_{l1}v_{l2}⋯v_{ln}) ∈ C^D. All entries unimodular, ‖m_l‖² = D. Note (m_l m_l^*)_{a,b} = Π_{j=b+1}^{a} v_{lj} for a > b: the entries of the rank-one matrix m_l m_l^* are exactly the monomials of the **interval** patterns {b+1,…,a}.
- **Absolute chain matrix**: M_+ = Σ_l |c_l| m_l m_l^*. Hermitian positive semidefinite, D×D, trace D·S with S = Σ_l |c_l|. Its diagonal is S; its (a,b) entry for a > b is Σ_l |c_l| Π_{j=b+1}^a v_{lj}, the interval-pattern moment of the absolute population {|c_l|, v_l}.
- For a matrix X, E_k(X) = coefficient of t^k in det(I + tX) (k-th elementary symmetric function of eigenvalues).

## 2. The conjecture

**(E+)** For every n, every population, with m_+ = max(m, 0):

  det(I + t M_+) ≥ (1 + t m_+)^D  for all t ≥ 0,  equivalently E_k(M_+) ≥ C(D,k) m_+^k for k = 1..D.

PROVED: (E+) is equivalent to its top rung **det M_+ ≥ m_+^D** (Maclaurin's inequality on the PSD matrix M_+ gives the other rungs).

Why it matters: it implies, in one line, the cost bound Σ_l |c_l|^r ≥ D^{1−r} m_+^r for every r ∈ (0,1] (via D^r Σ|c_l|^r ≥ tr(M_+^r) — proved by Rotfel'd subadditivity of Schatten quasi-norms — and the Bernstein representation x^r = c_r ∫_0^∞ log(1+xt) t^{−1−r} dt), which is the sharp "junk cannot lower representation cost" statement the paper needs.

Sharpness: at the **block** — N = D atoms, c_l = A/D, v_{lj} = ω^{−l} with ω = e^{2πi/D} for all j — M_+ = A·I and m = A, equality in every rung. Numerically, the ratio det M_+ / m^D has been minimised by adversarial multi-start at n = 1, 2 (N up to 12): never below 1, minimum 1.0000118 attained at the block; second-order expansion around the block is exactly tight (so any proof must be sharp to second order at the block).

## 3. What is PROVED

P1. **n = 1, every N.** Here M_+ = [[S, P̄],[P, S]] with P = Σ|c_l| v_l, so det M_+ = S² − |P|². Proof of m² ≤ S² − |P|²: for any probability measure ρ on T and Λ_ρ(u) = its "probe", one has m ≤ Σ_l |c_l| |Λ_ρ(v_l)|-type bounds; aiming the probe at the reflected centroid z = −P̄/S and applying Cauchy–Schwarz gives exactly m² ≤ S² − |P|². Equality class is larger than the block.

P2. **Equivalent population-free form (two applications of Sion's minimax).** (E+) for all populations at a given n ⇔ **(PW)**: for every Hermitian positive-definite D×D matrix B with det B = 1 there exists a probability measure ρ on T^n such that, with q_B(v) := m(v)^* B m(v) (m(v) the chain vector of a point v ∈ T^n) and Λ_ρ(v) := ∫ (chain-monomial kernel) dρ (explicitly: Λ_ρ(v) = Σ_{intervals I} ω_I v^I where ω_I are the interval moments of ρ, normalised so the constant term is 1),

  D · |Λ_ρ(v)| ≤ q_B(v)  for all v ∈ T^n.

P3. **Theorem UT (torus-free, the "phase-blind half").** For every B ≻ 0,
  max{ Re tr Y : Y upper-triangular, [[B, Y],[Y^*, B]] ⪰ 0 } ≥ D · det(B)^{1/D}.
Proof: SDP duality; the dual is min over unit-lower-triangular K of ‖B^{1/2} K B^{1/2}‖_{S_1}, and AM–GM with det K = 1 finishes. Optimal Y^* = B^{1/2} U B^{1/2} with U unitary. Consequence: for every population there is an interval polynomial Λ with constant term 1 and Σ_l |c_l| |Λ(v_l)| ≤ det(M_+)^{1/D}. The whole remaining content of (E+) is that Λ can be taken to be a **moment functional** of a probability measure (an "admissible" Λ).

P4. **n = 2: admissibility is one LMI.** An interval polynomial 1 + ω_1 v_1 + ω_2 v_2 + ω_{12} v_1 v_2 is a moment functional of a probability measure on T² iff the 3×3 Toeplitz-like matrix
  R_ω = [[1, ω̄_1, ω̄_{12}],[ω_1, 1, ω̄_2],[ω_{12}, ω_2, 1]] ⪰ 0
(the complex 3×3 elliptope has only rank-one extreme points). Necessary, but not sufficient, at every n ≥ 3.

## 4. The concrete open statement (n = 2, i.e. D = 3)

**(PW₂)** For every 3×3 Hermitian B ≻ 0 with det B = 1, find ω = (ω_1, ω_2, ω_{12}) ∈ C³ with R_ω ⪰ 0 and

  |1 + ω_1 v_1 + ω_2 v_2 + ω_{12} v_1 v_2| ≤ q_B(v)/3  for all (v_1, v_2) ∈ T²,  q_B(v) = m(v)^* B m(v), m(v) = (1, v_1, v_1 v_2).

Proving (PW₂) proves (E+) at n = 2 for every N (the paper's lead case, ν = 3). Known forced structure: at B = I + εH the first-order solution is ω_I = (2/3) ε H_I; the second-order correction is NOT the naive one and the inequality is exactly tangent at second order. Numerically (PW₂) holds with positive slack at every tested B (condition numbers up to ~580), found by optimising over 10-atom measures.

## 5. What is REFUTED (do not retry)

- λ_min(M_+) ≥ m: FALSE (ratio down to 1e−5 at n=1, 0.002 at n=2). (E+) is about det / elementary symmetric functions, not the bottom eigenvalue.
- The variant with hmin = min |h| in place of m = min Re h: FALSE at n = 2 (a "twisted" block with chain vectors (1, ω^{−l}, 1) has degenerate frame; det M_+ / hmin³ → 0). The sign structure Re h is essential.
- Naive n=1-style probes at D ≥ 3: the probe 2B_I/tr B, all geometric-mean / Cholesky-product probes, the least-squares chain probe — all fail at second order at the block.
- "Feasible for UT with tr Y ≥ D ⇒ admissible": FALSE (|ω| = 1.25 > 1 occurs).
- "The UT-optimal Y^* = B^{1/2}UB^{1/2} is admissible": FALSE at condition numbers ≈ 200–500 (LP and LMI agree).
- The elliptope-constrained matrix programme (testing all vectors rather than only chain vectors): value → 0 for B with a near-null generic direction; (PW) only quantifies over chain vectors m(v), which is what saves it.
- Candidate closed-form measures R_ω = (1−a) I + a·corr(B), corr(B) = diag(B)^{−1/2} B diag(B)^{−1/2} (the n=1 answer is a = 1): fails at second order for every a (a = 2/3 is forced, slack −0.02 … −0.6); mixtures with −corr(B^{−1}) fail too. Riesz-product and half-densities, interval and degree-1 SOS densities: fail.
- On the original problem, before this architecture: dual (Lagrangian) certificates with a fixed test measure on rows are provably impossible for the network statement (a duality gap), so the measure ρ in (PW) must depend on B — which it does.

## 6. Ideas not yet tried (suggestions only)

- Fejér–Riesz / Bochner: q_B(v) − 3|Λ_ρ(v)| ≥ 0 on T² is a non-negativity statement about a trigonometric polynomial of degree ≤ 1 in each variable once |Λ| is handled (e.g. via |Λ| ≤ Re(e^{iθ(v)}Λ) with θ chosen, or by squaring: q_B² ≥ 9|Λ|² is a degree-2 non-negativity on T², where SOS = non-negativity for multi-affine-squared forms is false in general but may hold here).
- Choose ρ from B by a canonical construction: e.g. ρ = the "Bernstein–Szegő" measure of B (the measure whose chain-moment matrix is a scalar multiple of B^{−1}, or of the inverse of a Cholesky factor), since for Toeplitz/Bernstein–Szegő measures moment matrices and their inverses are tied by the Gohberg–Semencul formula; test whether R_ω ∝ B^{−1} normalised, or ∝ (B^{-1} with unit diagonal via a diagonal congruence), satisfies (PW₂).
- A Schur-complement / matrix-Cauchy–Schwarz proof: (PW₂) says the rank-one PSD matrix m(v)m(v)^* pairs with B at least as well as with 3·(a rank-one unit-modulus phase matrix); look for a decomposition B = R_ω^{1/2} C R_ω^{1/2} with C ⪰ I-type.
- Log-concavity / mixed discriminants: det M_+ = Σ_{|L|=3} Π_{l∈L}|c_l| · |det[m_l]_{l∈L}|² (Cauchy–Binet, all terms ≥ 0); bound the Gram determinants of triples of chain vectors from below by the aimed output m — an "m ≤ (something averaged over triples)" inequality.
- A continuity / homotopy in B from I (where ρ = the block's uniform measure works exactly) tracking the optimal ρ(B), with a monotonicity argument along a path that preserves det B = 1.

## 7. Deliverable

A proof of (PW₂) (equivalently (E+) at n = 2), or a counterexample (a specific B with det B = 1 for which no admissible ω exists — please give B explicitly and a certificate, e.g. a point set on T² and a separating argument), or a further reduction to a strictly simpler statement with the reduction proved. State every step; flag anything you have not fully verified. If you find a proof at n = 2, say whether it generalises to n ≥ 3 (where admissibility requires that all non-interval moments of ρ be consistent, not just the elliptope LMI).
