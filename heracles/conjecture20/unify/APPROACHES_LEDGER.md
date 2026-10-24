# Ledger: every approach tried on the feature count (C), and why it failed

Sections A-E were mostly agent rounds run under orchestration. Section F is the solo path on RT_n.
Legend:
- **Dead**: refuted, or provably unable to give the general statement.
- **Partial**: proved something real but stops at a class or range.
- **Unfinished**: cut off, not refuted.

## A. The original target

1. **Conjecture 20 as stated** (IF value at every p, n, b). Dead: false. There are certified counterexamples at p = 3, b = 2,
   n >= 11 (concentrated winners using constant input components). Replaced by the price-programme form 20' and the count (C).
   (`conjecture20_disproof.tex`)

## B. Direct attack: per-feature payment against detection (mechanism note, unified note)

2. **Master inequality / Theorem A**: charge each cheap feature's gain to its own interference and detect it by moduli. Dead
   as a general route:
   - cost-free compensation defeats it, so costs must enter;
   - moduli-only arguments provably cannot reach the relations case with its constant.
3. **Pointwise payment (P)**: each feature pays pointwise through its own sign-flipped coefficients. Dead: false at n = 4,
   b = 2 (certified constant >= 1.127 > 1).
4. **One-sided relations bound (O1, first form)**. Dead: false. It was corrected to the oscillation form, which is still open.
5. **Parseval / L2 detection of the cheap set**. Partial: it closes small cheap sets, but loses sqrt(number of cheap
   features) and provably runs out once K* >~ 55.
6. **Quadratic / circularity payment** (Companion Thm 14, constant k*). Dead beyond small n: k* grows with n, and Thm 14
   fails at (7, 28, 2).
7. **Riesz products / Sidon detection**. Partial: it works on dissociated supports. But optimal supports have
   K* ~ 1.3-1.5 ln p > log_3 p features, so additive relations are forced by pigeonhole.
8. **Linear gain bounds for compensation (O2)**. Dead: they provably cannot close O2. The gain is quadratic near the IF block
   and has to be treated that way.
9. **Local optimality / KKT to decide the count**. Dead: a local maximiser at p = 5, n = 6 has both cheap and spurious
   features.
10. **"Every single feature needs nu neurons."** Dead: there is a width-6 network at (5, 6).
11. **Efficiency-Sidon reduction**. Partial: it closes inefficient allocations at every p. The content of the problem is in
    near-efficient allocations.
12. **Local + global ingredient split** (`unified.tex`). Dead:
    - the global ingredient fails at p = 3, and is itself the open problem;
    - its clean lemma with order-2 size is false (Gauss-sum thin layer).
13. **Thin / costly / blind trichotomy**. Partial: thin and costly blocks are closed. Blind blocks are open: blocks with
    blind spots that the rest of the allocation must over-cover, with a numerical window near p ~ 1e12.
14. **Single-cheap-feature compensation game at n = 2** (Chebotarev dichotomy). Partial: numerically won for p <= 41, with
    no general proof.

## C. Agent rounds 1-3 (case-wise, before the all-p rule)

15. **Frequency split + sign-flip averaging (SPLIT)**. It gives (C) <= (class-C count) + (MM_opt). Canonical-split
    decoupling for all networks is false (certified (5,2,2) violator), so the mixed-mode step needs global optimality.
    That step is open.
16. **Temperature identity / global optimality (GLOBAL)**. At every K*-transition (every p >= 11) the count becomes an
    exact value statement. Global optimality only yields lower bounds where an upper bound is needed.
17. **Large n (LARGEN)**. Dead: every payment constant (circularity, pointwise, polydisc) grows with n, and the polydisc
    payment is refuted.
18. **Energy-only route, class C at n = 2 (GIBBS)**. Partial: it closes p <= 31 at b = 2. It provably stops there because both
    constants are sharp (sawtooth detection, single-neuron payment). Going beyond needs phases.
19. **Arithmetic relations core (ARITH2)**. Partial: p <= 59 at b = 2; every prime for b > 3 - O(p^{-1/2}); AP cheap sets at
    every prime. It stops because:
    - the remaining cheap sets are large, incompressible and flat;
    - the phase-free condition TDP* must fail near K* ~ 55;
    - it needs a structure theorem for near-optimal allocations, and none exists.
20. **Case-by-case computer-assisted proofs (p52)**. Partial: (5,2) is proved for every b. (7,2) is stuck for b < 2.538:
    - phase-blind cap;
    - the free relaxation fails;
    - shared neurons are free at b = 2.

    The all-p rule then excluded this style.
21. **Literature import (KNOWN)**. Dead: no existing theorem implies any piece. The nearest analogue
    (minimum-output-entropy additivity) is false in high dimension.
22. **n = 3 payments (N3); channel-resolved ledger at (7,2) (N2MIX)**. Unfinished: cut off by session limits, then superseded
    by the all-p rule.

## D. Round (E): five general routes, refereed

23. **Reduction (E_reduce)**. Worked: Theorem T. But value plus first-order data cannot rule out straddlers (dilation
    obstruction), so FE is unavoidable. FE is false at non-global local maxima, so it needs a genuinely global proof.
24. **Split / replication / superadditivity (E_split)**. Dead:
    - blind to decoupling;
    - trivial on class C;
    - the own-data relaxation overshoots by 32% at (5,2,2).
25. **KKT / critical values (E_kkt)**. Dead:
    - KKT bounds entropy only from above;
    - co-maximality adds nothing;
    - entropy-dropping level crossings really occur.
26. **Entropic width (E_width)**. Dead: the pointwise-in-s entropy inequality fails in both signs, even on interference-free
    networks.
27. **Duality (E_dual)**. Dead:
    - linear duality is blind to s;
    - no spectral certificate exists at s > 1;
    - the Forster-type inequality is false (D8);
    - the restricted identity on pinned profiles is false (D9).

## E. Core round (convex + correction)

28. **Convex + correction identity**. Exact, but bookkeeping: it rewrites RD as a factorisation statement without proving
    it.
29. **Per-feature charging (CC-FACTOR)**. Dead:
    - the KKT charge is circular (each piece is RD again);
    - the argmin and Shapley charges are refuted at (5,2,2);
    - second-moment detection is blind to antipodal and evenly spread interference (it fails in 135 of 144 binding sets).
30. **Exact factorisation CD_C** (payment at the network's own allocation). Dead: Legendre networks violate it for
    p >= 127 at (2,2). A proof must use the global efficiency Gamma_s.
31. **Stability around the convex case (CC-STAB)**. Partial: the local regime is proved (Theorem L), and large deficits are
    trivial. The middle regime is open:
    - fixed-concentration slice certificates are refuted;
    - no reduction to b = 2 is possible, since b = 2 is not the hardest exponent.
32. **Field-row joint detection (FR)**. Worked on average, with no sqrt(K) loss. Its pointwise residual PW_C is false
    (Legendre-type networks at p = 347, 383, 443, 467, and an optimised one at (43,2,2)).
33. **Every efficiency-blind detection inequality** (Riesz, Lambda(p), Khintchine, Bonami, Paley-Zygmund, tilted measures).
    Dead, all at once: the Legendre network is flat at every offset (depth 2 against sqrt p).
34. **Offset-translation rigidity**. Dead: single-set families are balanced, so the offsets are just translations. There is
    no rigidity where the hard core lives.
35. **Parseval closure / unified closure / gain cap** (J7, J8, joint2). Partial: up to 3 cheap features always (more under
    balance), p < 2/c_n^2, and large n. It cannot handle many cheap features at moderate n.
36. **Per-feature projection (FIELDMAP)**. Dead: Legendre networks refute it, and second-moment detection provably stops at
    45 degrees.
37. **Counterexample hunt (C_HUNT)**. Found nothing. The one candidate, (11,2,2.5), was an artifact.

## F. Solo path on RT_n (`X_core/`)

38. **Field-row averaging**. Partial: it proves RT_n for any interference when m_r <= lambda. That means roughly n >= 3 near
    s = 1, and n >~ 10 at b = 2. It fails at small n, which is exactly where the core lives.
39. **QR half (Legendre, p = 1 mod 4)**. Worked, for one family.
40. **Sine-weighted QR (Legendre, p = 3 mod 4)**. Worked, except for one two-variable calculus inequality. That inequality is
    only checked on a grid (real p up to 1e9) plus asymptotics, so it still needs a real proof.
41. **Positive-part compression at the best w0 (general phases)**. It certifies every configuration numerically. The proof
    breaks because the k >= 2 harmonics of the phase field are not controlled by the depth.
42. **w0 at the extremes of the second harmonic**. Dead: it makes the k = 2 term favourable but clusters phases where the
    cost is high. The trade-off is intrinsic.
43. **Random w0**. Dead: the union bound over offsets loses sqrt(log p).
44. **Phase-free L2 version**. First declared false. That was wrong: the Gamma estimate was weak. It holds numerically, but
    heuristically fails on optimal supports at astronomically large p, so it cannot be the final statement.
45. **Linear vs even-part dichotomy with the depth-free condition (Q)**. Partial: (Q) fails when the coverage profile is not
    flat (few features), so the depth has to come back in.
46. **Real-phase theorem**. Proved for all p, n, s, but only for interference symmetric about a point, where the hard term
    is zero.
47. **Uniform cheap part on all of F, any phases**. Proved for p >= 81. The bounds are too lossy in windows for
    29 <= p <= 80, and a p >= 81 statement is excluded by the rules anyway.
48. **Realification / derandomised w0**. Current. No proof, and not yet tested for truth.

## The five things that keep killing approaches

1. **The Legendre network** is flat at every offset. It killed:
   - 12 (the clean lemma);
   - 19 (support-only flatness);
   - 30, 32, 33, 36.

   Approaches 39-48 exist because of it. Any argument that ignores the global efficiency of the programme dies on it.
2. **p = 3 and mixed modes** kill every inequality stated for all networks (1, 12, 15, 29). A proof must use global
   optimality and p >= 5.
3. **Sharp constants in the wrong place.** Energy/L2 methods stop at exact thresholds (5, 18, 19, 44) because their
   constants are attained.
4. **Local, linear and first-order information is provably insufficient** (9, 23, 25, 27).
5. **Per-feature localisation is circular or false** (3, 29, 30, 36). Payment adds up over features; detection does not.

## What survives

- Theorem T: (C) <= RD + FE.
- The exact identities.
- The closed classes: real phases, Legendre, up to 3 cheap features, small p at n = 2, (5,2).

Open:
- RT_n with complex phases;
- multi-set / non-aligned reduction to the core;
- mixed networks to class C (MM_opt);
- FE.
