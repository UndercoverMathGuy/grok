# Round 1 digest (orchestrator; claims are UNREFEREED unless marked)

Full reports: unify/split/REPORT.md, unify/global/REPORT.md, unify/largen/REPORT.md (the harness forbids subagents from
writing report files, so the orchestrator saved their structured outputs there). Partial work, cut off by a session limit
before any report: unify/arith/, unify/gibbs/, unify/matrix/.

## Structural facts claimed PROVED (to be refereed in round 2)
- SPLIT R1-R9.
  - The frequency split theta -> theta_split (class C) never raises cost and keeps every atom, feature, cheap set, Q_l,
    a_l, G and E_x g.
  - Sigma (sign flips) is the maximal cost-preserving group fixing own data; its average is the split table.
  - Mixed modes reach a feature l's sign-flipped character only through 2|F|-1 channels: partner |3l|, foreign |2l +- j|,
    DC via output |2l|. Free mixing needs b=2 and an F2 relation.
  - Span transfer (R7): mixed modes off the span of the class-C interference never lower a directional minimum.
    Mixing never improves an IF core.
  - Reduction (R9): (C) <= (C_C) + (MM_opt), where (MM_opt) means every global maximiser has gamma(theta*_split) >=
    gamma(theta*). At (5,2) and (7,2), b in [2,3], this is: (C) for all networks iff every global maximiser is IF.
- GLOBAL.
  - R1-R2 (temperature identity): d/ds log gamma(theta;s) = -H(cost shares), and the slopes of log gamma* are the
    entropies of the maximisers (Danskin).
  - R3-R4 (entropy criteria): the price / IF value can first fail only through a "concentrated winner", one with
    entropy H < log nu (resp. < log nu + H(z*)).
  - R5: decoupling (D) <=> [gamma* = pi B_p Gamma_s] + support identification.
  - R6 (slack principle): approximate decoupling within eps_tol(s) gives the lower half of (C).
  - R7 (transition identity): (C) on both sides of a K*-transition s_c forces W_{K<}(s_c) = gamma*(s_c), an exact
    value statement. Transitions exist for all p >= 11 and for p = 5, 7 when n >= 3.
  - R9: direction-uniform IF price at n=2 for single-frequency single features.
- LARGEN.
  - R1: width >= ~(n/2)(1-1/p) for slot-symmetric single features with any sign pattern.
  - R2: higher-order circularity E(Phi-A)^k = 0 for k <= p-1 (class C), k <= (p-1)/2 (DC).
  - R3: slice subordination via Rogosinski: cos(pi/p)^k |sum_{|T|=k} beta_T zeta^T| <= (4/p)|A-c|.
  - R4: robust width; nu neurons are forced when p >= 4n+5 (slot-symmetric class C).
  - R5: certified k*(n, nu/2) grows (k*(32) >= 1.906 > k_7), so Companion Thm 14 is not n-uniform.
  - R12: (Q1) <=> c/L <= (K_r/nu)^{s-1}.

## Refutations
- SPLIT R10 (certified): decoupling for ALL networks with the canonical split is false at (5,2,2) (mixed violator at
  gamma = 0.79 gamma_IF, ratio 1.033). So any proof of (C) must use global optimality, not an all-network inequality.
- LARGEN R8a: the n-uniform polydisc payment (seed S4) is false.
- LARGEN R10b: the Note's "every positive-margin single feature at p>=5 needs nu neurons" is false (p=5, n=6, width 6).

## Numerics (evidence)
- Every optimum found at p = 5, 7, 11 (n = 2, 3) is IF with count = K*, including both sides of six K* transitions.
- Compensation gains exactly nothing in penalised searches at (5,2) and (7,2).
- Single features are direction-uniformly priced at IF at n = 2, 3, 4 and small p.
- At p = 5 (n <= 12) and p = 7 (n <= 7) the best single feature is the IF block; spurious clean amplitudes <= 1e-11.

## Where the problem now stands
1. (5,2) and (7,2), all networks, all b: (C) reduces to ONE compensation bound with slack (GLOBAL R12).
   - At (5,2): W_{1}(s) < 1.279 * 3^{1-s} B_5 kappa_min(5) on s in (1.331, 1.5].
   - SPLIT's "first lemma" targets the same thing via the master inequality with uniform mu: delta-Fourier separation
     charges (1/2)|beta_{l,T}| per cheap feature against c_1 = 0.4226.
   - MATRIX (cut off) was computing the same V_1 object with proved payments (unify/matrix/payment_bound.py,
     freesub.py).
2. At transitions the count is a value statement (GLOBAL R7). No count-only shortcut exists at p >= 11.
3. Large n: an n-uniform payment in the subordination currency is the missing lemma (LARGEN).
