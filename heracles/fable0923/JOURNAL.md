# fable0923 journal — (T) for the gated chain, open-ended pass (2026-09-23)

Local CPU, fp64, no pods. New files only, in this directory. Nothing inherited is edited.

## 0. What I read (state as inherited)
GOPT_STATE{,2,3}, ELEGANT, THEORY_nsum, claude0923/NOTES, astra/{README,ARGUMENT,LOCAL_THEOREM,
OPTIMIZER_ATTACK,P3_ANALYTIC,GENERAL_DEPTH,PROJECTION_OBSTRUCTION}, opus55/{README,IMPURE,P3_THEOREM},
P3A, P3B, P3C, HOLES_logic/aiming, DEPTHA, ARCH (status sections), code fvf_obj/kstar/lmax_junk/exact_gamma.

Net position as I understand it (n=2, q=2 main case):
- (T) exact under (H). Unrestricted bound off by sqrt 3. Own-share bound.
- Pure class: P3C reduction to a ONE-population inequality at aiming depth A, with the
  partition-product density family. Threshold A_lam ~ 0.304 (mixture of singleton + fibre
  partitions, numerically), singleton alone 0.4284, fibre alone 0.4226. Theorem DA proves
  the singleton inequality at A = 1/2 (exact certificate), HOLES_aiming at 0.47. Paper
  supports have A* in 0.38..0.61; p=113 has A* < 0.4222 so the singleton route is refuted there.
  The mixture inequality at ~0.31 is unproved.
- Impure: blind family 1.30 at p=11; charging lemma open.
- P3_ANALYTIC proves (analytically, all r in [2/3,1]) the ONE-population inequality
  sum|c|^r >= 3^{1-r}(|G| - delta/2)^r with delta = max_T |J_T|.

## 1. The idea I am pushing (new, as far as I can tell)
Two changes to the P3C reduction, both of which strictly enlarge what it can certify:

(a) Replace the partition-PRODUCT test densities by GENERAL densities on Z_p^2. Then the
    three junk moments of frequency k, at modes (2k,0), (0,2k), (2k,2k), are independent
    (constrained only by positivity of the density), instead of (t1, t2, t1*t2) [singleton]
    or (0, 0, t) [fibre]. The mixture-of-partitions device becomes unnecessary.

(b) Match the aiming to the eta-form of the population inequality. P3_ANALYTIC's PSD/Hoelder
    method actually controls tr M^2 through eta^2 = |J_1|^2+|J_2|^2+|J_12|^2 (it only
    weakens to 3 delta^2 at the end). So the natural population inequality is
        (ETA-A)   sum_l |c_l|^r >= 3^{1-r} (|G| - A * eta)_+^r ,  eta = ||J||_2 ,
    and the aiming it needs is an l2-POLYBALL, not a polydisc: for every k, a moment vector
    t^{(k)} with Re sum_T t_T J_T <= -A ||J^{(k)}||_2. Since the l2 ball sits INSIDE the
    polydisc of the same radius, the polyball inradius A_ball(S) is >= the polydisc inradius,
    and (ETA-A) at the M_1 population (single atom, J=(1,1,1), eta=sqrt3) needs only
    A >= (1 - 3^{-1/2})/sqrt 3 = 0.2440 (this is P3A's rho_need, reappearing).

So the criterion becomes:  (T) on the pure class on support S  <=  A_ball(S) >= A_eta,
with A_eta the threshold of a PROVED (ETA-A). Both sides are to be computed now.

Risk noted up front: the 3K-mode set {(2k,0),(0,2k),(2k,2k)} has the intrinsic relation
m_12 = m_1 + m_2, so a Riesz-product lower bound on the inradius is capped at a - a^2 <= 1/4,
uncomfortably close to 0.244. Whether real supports do better is an empirical question first.

## 2. Log

### 2.1 First numerics (eta_psd.py, aball.py)
- PSD route (P3_ANALYTIC with delta -> eta): proved (ETA-A) holds for A >= A_PSD = 0.2701
  (binding at e ~ 1.03, eta ~ 1.3, i.e. slightly misaligned atoms; at e = 1 alone it would be 0.244).
- True (ETA-A) threshold: N=1 gives 0.2440 (M_1 = single atom); optimizer for N >= 2 unreliable
  (bisection returned lower values but the re-check at 'hi' violated); monotone in N, so the
  true threshold is >= 0.244, most likely = 0.244 (endpoint population, as P3A's rho_need).
- ||J||_1 (sum-form) true threshold ~ 0.1835, binding at M_2 (two antipodal atoms, junk only on {12}).
- Torus limit, one frequency, modes (1,0),(0,1),(1,1): polydisc inradius 0.5000 (= the classical
  cos a + cos b + cos(a+b) >= -3/2), l2-ball inradius 0.8660 = sqrt3/2.
- Paper supports (upper bounds from LP ascent, 6 starts):
    p   K  A*(S)   Adisc(3K)  Aball(3K)
    59  6  0.3828  0.3256     0.3828
    71  5  0.4796  0.3583     0.5478
    97  6  0.4612  0.3568     0.4612
    113 6  0.4749  0.3574     0.4752
    127 7  0.4348  0.3481     0.4420
  Aball >= A* in every case (worst polyball direction lives on a single pattern, all-{1} or all-{12},
  where the problem degenerates to the K-mode one). Adisc(3K) ~ 0.75-0.85 A*.
  Small p: 11: A* .6018 / .3749 / .6460;  13: .3369/.2919/.3369;  17: .4781/.3541/.4781;
  23: .3277/.2894/.3277;  31: .4009/.3242/.4277.
- So numerically A_ball(S) >= 0.38 >> 0.27 on every paper support: the pure-class (T) would follow on
  all of them, INCLUDING p = 113 where the singleton route is refuted.
- BUT: certification. A_ball and Adisc(3K) need a HOLES_aiming-type enumeration over 6K-1 phases:
  infeasible at K >= 4. Black-box use of the certified K-mode A* loses a factor 3 (mixture of the
  three fibre families). Conjecture (numerical, 10/10 supports): A_ball(S) >= A*(S). No proof.
- Alternative that IS certifiable today: the partition-product family (depths (A,A,A^2) and (0,0,A))
  with HOLES' certified A* -- needs the MIXTURE criterion with the PSD-proved Psi. Computing its
  threshold next (eta_mix.py).

### 2.2 Transfer lemma: certified K-mode depth A* => l2-polyball depth of the 3K body (cavg.py)
Write F(a,b) = f_1(a) + f_2(b) + f_3(a+b), f_i(t) = Re sum_k z^i_k zeta_k^t, zeta_k = omega^{-2k}.
Since (a, c := a+b) is a bijection of Z_p^2:  max F = max_a [ f_1(a) + max_c ( f_3(c) + f_2(c-a) ) ].
Inner: c -> f_3(c) + f_2(c-a) is a K-mode function with coefficients z^3_k + z^2_k zeta_k^{-a}, so by the
K-mode certificate its max is >= A* sum_k |z^3_k + z^2_k zeta_k^{-a}| >= A* Re sum_k conj(d_k)(z^3_k + z^2_k zeta_k^{-a})
for any unit d (chosen before a). Outer: max_a Re sum_k [ z^1_k + A* d_k conj(z^2_k) ] zeta_k^a >= A* sum_k |z^1_k + A* d_k conj(z^2_k)|.
Hence for each of the 6 role assignments (outer, full, coupled) and every unit d:
   h(z) >= A* [ Re<d, z^full> + sum_k |z^outer_k + A* d_k conj(z^coupled_k)| ]   (*)
Per k, with moduli (x,y,w) = (|full|,|outer|,|coupled|): beta >= y + |x - A* w|  (d = x/|x| gives x + y - A*w; aligning gives y + A*w - x).
Summing over the 6 assignments: sum_sigma beta_sigma >= 2(a+b+c) + sum_{i != j} |a_i - A* a_j|, a = (|z^1_k|,|z^2_k|,|z^3_k|).
Numerically (exact beta, Nelder-Mead over C^3): c(A) := min (1/6) sum_sigma beta_sigma / ||.||_2 = (2 + A)/3 EXACTLY
for A in [0.30, 0.50], worst direction = single pattern.  So
   A_ball(S) >= A*(S) (2 + A*(S)) / 3      (rigorous once the 3-variable inequality
   2(a+b+c) + sum_{i!=j}|a_i - A a_j| >= 2(2+A) sqrt(a^2+b^2+c^2), a,b,c >= 0, 0 <= A <= 1/2, is proved; to do).
Consequence with HOLES_aiming's certified K-mode depths: LADDER 59 (A* >= 0.400911) -> A_ball >= 0.3209;
LADDER 71 (0.473741) -> 0.3906; LADDER 113 (0.373393) -> 0.2954. All above A_PSD = 0.2701.
=> candidate THEOREM: pure class, n=2, q=2, (T) on S whenever A*(S) >= A_0 with A_0(2+A_0)/3 = A_eta,
   where A_eta is any depth at which (ETA-A) is PROVED. With A_eta = 0.29: A_0 = 0.3535 -- covers p=71, p=59
   ladder supports and p=113 LADDER (0.3734). Still to do: (i) rigorous proof of (ETA-A) at A <= 0.29;
   (ii) the 3-variable inequality; (iii) write the reduction out in full (general density, eta-aim).

### 2.3 Certificates done
- (3V) 3-variable inequality: PROVED exactly on A' in [0, 47/100] (tri_ineq2.py: 21 symbolic vertices, cell-wise
  concavity, exact root counting). Fails from A' ~ 0.4975 (vertex (1,A',A')). Cap A' = min(A*, 0.47).
- Claim(A) for the eta-form population inequality: PROVED at A = 0.272 / 0.275 / 0.28 (eta_cert3.py): corner lemma
  analytic on [1,1.2]x[0,0.25] (Delta >= 9x^2 + 6A eta - eta^2 bracket, bracket <= 6.4699), remainder 5279 / 2075 / 1064
  interval boxes, 0 failures. Corner algebra verified symbolically (corner_verify.py).
- Reduction identity verified to 1.9e-13 (reduce_verify.py).
- Threshold: A*(S) >= 0.34759 suffices at A_eta = 0.272. All eight HOLES-certified supports pass (min 0.358249).
- Inherited during the session (coordinator): astra EXPLICIT_ENDPOINT / PHASE_FRAME_EXTENSION (unrestricted near q = nu,
  p=5 down to q = 5/2, and a matrix-only obstruction at q=2). Not in conflict: my population step is matrix-based but the
  row side is priced by a junk-aware density first; pure networks have no discarded components.
- FINDINGS.md written. No background jobs.
