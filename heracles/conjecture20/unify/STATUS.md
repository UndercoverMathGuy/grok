# Feature count = K*: status of the unification push

Target (C): for an odd prime p, n >= 2 and b in [2, nu]: every global maximiser of the L_{2,b} margin of product
networks on modular addition uses exactly |spt y*| features, for some maximiser y* of the programme Gamma_s(p).

Labels used below:
- **Confirmed** = proved by an agent, then a hostile referee rebuilt the proof and ran independent checks.
- **Proved** = an agent's proof that has not yet been refereed.

Agent reports and referee verdicts are in `unify/<angle>/REPORT*.md` and `VERIFY*.md`. The p=5, n=2 proof is
written up in `unify/p52/p52_count_theorem_proof.tex`.

## 1. Where (C) is now a theorem (all networks)

| case | b range | status | source |
|---|---|---|---|
| p = 3, every n | all | trivial (one frequency) | Note Prop count(a) |
| **p = 5, n = 2** | **every b in [2,3]** | **Confirmed twice** (mathematics referee + certificate referee; all four interval-arithmetic chunks re-run, fail = 0) | p52 R1-R10 |
| p = 7, n = 2 | b in [2.5381, 3] | Confirmed (certificate re-run by two referees) | p52 R11-R14 |
| p = 7, n = 2, single-feature networks | every b | Confirmed | p52 R11 |
| any p, n | dense regime: nu^{s-1} Gamma_s^{(F minus j)} < Gamma_s | proved earlier | Note Thm dense |
| any p, n | b in [b_0, nu] | proved earlier | earlier notes, Part II |

How the (5,2) proof works:
- Reduction: K* = 2 at every b, so (C) follows if no network with one feature reaches the interference-free (IF)
  value.
- Three ledgers price everything:
  - margin certificate: an exact pairing with the feature's own coefficients plus one l2 number;
  - mode budget: non-own mode mass is converted into cost by a Pythagorean inequality;
  - twisted Rotfel'd payment.
- The compensation mass R is eliminated, b = 2 is the worst case, and Hermitian-square certificates over 2115
  boxes in interval arithmetic give gamma <= 0.995 gamma_IF. The true value is 0.78.

## 2. Where (C) holds inside a class of networks

Class C means frequency-pure neurons with no input DC. Every class-C maximiser is IF and uses K* features:
- n = 2, **every p <= 31 and every b in [2,3]**. The range grows with b: p <= 61 for b >= 2.25, p <= 661 for
  b >= 2.75. Confirmed (gibbs R7).
- The ingredients are two sharp constants:
  - plane Bhatia-Davis via Cauchy-Davenport;
  - an L2 single-feature payment ((1 - 3^{1-s})/sqrt3, computer-assisted).
- At b = 2 this energy-only route provably stops at p = 31.

Relations case: n = 2, class C, every feature's interference on one slot-set.
- Closed for p <= 59 at every b. The range grows with b: p <= 10369 at b = 2.9.
- Closed at **every prime** for b > b_p = 3 - O(p^{-1/2}).
- Interference never gains more than 10.8% over IF.
- At b = 2, cheap sets made of constant-price arithmetic progressions are closed at every prime.
- Status: Confirmed (arith2), minor wording gaps.

Everything above transfers to all networks once every global maximiser is shown to be IF (SPLIT R9, corrected).

## 3. Structural results that organise the problem

- **Frequency split** (Confirmed).
  - theta -> theta_split is in class C, never costs more for b >= 2, and keeps every atom, feature, cheap set and
    mean gap.
  - It is the average over the cost-preserving sign-flip group Sigma.
  - It is the pinching inequality of the l_r tensor quasi-norm; b = 2 is the exact threshold (KM1).
  - So the cost side never causes non-additivity.
- **Mixed modes** (Confirmed). These are the Sigma-odd part.
  - They reach a feature's sign-flipped characters only through 2|F| - 1 explicit channels.
  - They are free only at b = 2, and then only through F2 relations 2l = +-j +- q.
  - They never improve an IF core.
- **delta-Fourier isolation** (Confirmed). For every p, every n and every network, there is a test that pairs only
  with the clean coefficients and ONE chosen sign-flipped coefficient. Mixing and compensation are invisible to it.
- **Temperature identity** (Confirmed): d/ds log gamma = -H(cost shares). Consequences:
  - The IF value or the price can first fail only through a concentrated winner. The p = 3 counterexamples are
    exactly that.
  - Approximate decoupling within an explicit tolerance eps_tol(s) gives the lower half of (C) (slack principle).
- **Transitions** (Confirmed).
  - Every prime p >= 11 has a K*-transition in s in (1, 1.5).
  - At a transition, (C) is equivalent to the existence of a maximiser with the smaller count. This is a count
    statement, not a value identity.
- **Lemma 7 equality is rigid** (KM2, Confirmed). The paper's block is the unique cost-optimal IF feature, up to
  symmetries.

## 4. What blocks a general proof (precise residuals)

1. **Relations core** (n = 2, b = 2, p > 59). Interference can pay only on a cheap set that is at once:
   - large (kappa-mass > 29.7);
   - incompressible (its coverage cannot be bought more cheaply);
   - flat (osc close to 2 x L2).

   The missing inverse theorem is "flat + large implies compressible" (arith2 R11).
   - Support-only flatness bounds are refuted (Legendre construction).
   - Parseval-only detection fails once K* >~ 55.
2. **Compensation with several features at b = 2.** Neurons shared by two frequencies are budget-tight, yet carry
   non-own mode mass 3e^2 (p52 R17). So the Pythagorean ledger behind (5,2) has no two-feature version at b = 2.
   A channel-resolved ledger is needed. Round 3 (n2mix) is attacking (7,2), b in [2, 2.538).
3. **Class C beyond p = 31 at b = 2.** Both energy constants are sharp, so a proof must use phases.
4. **n >= 3 single-feature constants (O3).** Round 3 (n3) is working on n = 3 and (5,3).
5. **Large n at fixed p (O4).**
   - The circularity route provably fails (Thm 14 fails at (7, 28, 2)).
   - The missing lemma is an n-uniform payment in the subordination currency.
   - Numerically the IF block is optimal at p = 5 for n <= 12, with no spurious features.

## 5. Refuted along the way

| claim | where refuted |
|---|---|
| Canonical-split decoupling for all networks | (5,2,2) certified violator at gamma = 0.79 gamma_IF. The Note's existential (D) survives. |
| n-uniform polydisc payment | LARGEN R8a |
| The Note's "every single feature needs nu neurons at p >= 5" | (5,6) width-6 network |
| Plane Bhatia-Davis with constant 1 | sawtooth: the constant is (p-1)/(p+1) |
| Support-only Sidon/flatness lemmas | Legendre symbol |
| The "count needs the value at transitions" overclaim | corrected by referee |
