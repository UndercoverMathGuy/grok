# X_CHARGE: neurons that mix frequencies (chain 3a.9 / f.7), p = 5 first

Fork report, 2026-09-25. Scripts in `xcharge/` (run from that directory with `uv run --with numpy --with scipy python <script>`).
Labels: PROVED, CERTIFIED (exact or interval arithmetic), NUMERICAL, FALSE.
Setting throughout: n = 2 operands, q = 2, full table. Conventions of `common.py` (candidate s - D).

## Headline

1. **Single-frequency networks at p = 5 are closed.** Every such network (any width, any junk) has gamma <= gamma_sym,
   with equality only when it is junk-free. PROVED, with one CERTIFIED constant.
   More generally, at every odd prime, any single-frequency network whose junk sits on at most 2 frequencies is covered.
2. **The chain's per-frequency form is FALSE for mixing networks.** The p = 5 junky local maximum beats every
   per-frequency ceiling (by 1.8% and 0.9%). So no charging lemma of the form "X leak <= something per frequency,
   then the programme" can exist. X must be paid for out of *coverage* slack (a global, whole-spectrum quantity).
3. **X-blind test distributions are one-sided**, so at p = 5 they are too weak (1.115x, opus55). NUMERICAL, 1e-16.
4. What remains for p = 5 unrestricted is a coverage-aware charging lemma; the precise target is stated at the end.

---

## 1. Single-frequency networks at p = 5: closed

**Elementary depth bounds (PROVED, every support S of size K, every odd prime).** Let f_z(x) = Re sum_{k in S} z_k w^{-2kx}.
- *One-sided depth A\* >= 1/(2K).* If f_z <= 1 everywhere, then g = 1 - f_z >= 0 has mean 1. Its Fourier coefficient
  at w^{-2kx} is -z_k/2, so |z_k|/2 <= mean(g) = 1. Hence sum|z_k| <= 2K, and the depth is 1/max sum|z_k| >= 1/(2K).
  (The characters w^{-2kx}, k in S, and their conjugates are distinct because p is odd.)
- *Two-sided depth A_s >= 1/sqrt(2K).* If |f_z| <= 1, Parseval gives mean(f_z^2) = sum|z_k|^2/2 <= 1, so
  sum|z_k| <= sqrt(2K).
At p = 5, S = {1,2}: A* >= 1/4 and A_s >= 1/2. Both are sharp or near-sharp: the grid LP (`p5_depths.py`) gives
A* = 0.2500 (Dirichlet) and A_s = 0.5590 = sqrt5/4.
Note: the cyclic-symmetry reduction (fix the first phase) used in HOLES_aiming and in the parent's scripts needs grid
size L <= p. At p = 5 I used the full grid instead.

**Transfer constant (CERTIFIED, `cert_transfer.py`).** With the flip-enhanced transfer lemma (VERIFY.md "New lemma"),
assignment weights lam = (4/9, 7/18, 0, 1/6, 0, 0), A* = 1/4 and A_s = 1/2:
sum lam beta >= c |a|_2 holds on the whole positive octant for c = 29/100. Exact rational check at all 26 vertices of
the kink arrangement. On each cell the left side is linear, so left minus c|a| is concave and its minimum sits at a
vertex. The check fails at c = 73/250 = 0.292, matching the numerical optimum 0.2917.
So the flip-enlarged aim body contains the l2 polyball of radius 0.29 > 0.272.

**Theorem (p = 5, single-frequency networks).** For every network of single-frequency neurons at p = 5 (any width,
any junk): gamma <= gamma_sym, strictly unless the table is junk-free. At equality the network is the share-weighted
block on {1,2} (K* = 2).
*Proof.* The junky set is a subset of {1,2}, so the aim depth is >= 0.29 (3a.7 plus the certificate). The cost side
eta-form holds at depth 0.272 (fable0923 Lemma 3, interval certificate re-run in VERIFY.md V3b), hence at 0.29,
strictly when eta_k > 0. Assembly is V3c. Equality forces eta_k = 0; then theory_modadd Cor. 7.2 applies.
Dependencies: flip lemma (VERIFY Vf.1), the transfer derivation (VERIFY "New lemma"), fable's certificate.
**Same argument, any odd prime:** a single-frequency network whose junk sits on at most 2 frequencies satisfies the bound,
since the elementary depths depend only on K. For K = 3 the elementary bounds give only c = 0.214 < 0.272
(`cvals7.py`), so p = 7 needs actual depth computations.

---

## 2. What X does, and why a per-frequency charge cannot work

**Local max anatomy (`p5_anatomy.py`, NUMERICAL).** 5 active neurons, each mostly one frequency with a minority
admixture of the other (e.g. |e1^, e2^, w^| = (0.43, 0.43, 0.44) at k = 1 and (0.10, 0.10, 0.05) at k = 2).
Budget slack (DC + imbalance) is only 1.1% of P. Chain pure bound at aim 0.272: 0.004642 < true margin 0.005199.

**Mechanism (NUMERICAL, with an exact count).** A minority admixture eps of frequency l into a neuron dominated by
frequency k costs O(eps^2) budget. It creates X at triple (l,k,k) of size eps x (the k-amplitudes), which is *first
order* in eps. At p = 5 that X lands on k's junk row modes: row mode (2,0) receives the triples (2,1,1), (1,2,2) and
the slot-1-DC triple (0,2,2). So X can cancel junk dips at first order for second-order cost.

**Per-frequency ceilings (FALSE for mixing networks; `p5_pfc.py`).**
- PFC: margin <= (2/p^3) min_D sum_k cov_k(D) (cost_k/3^{1/3})^{3/2}, cost_k = sum_i |c_ik|^{2/3}.
  Local max: ratio **1.0180**.
- PFC-b: the same with cost_k replaced by (p/6) x (frequency-k share of the weight norm), which includes the imbalance
  slack. Ratio **1.0092**.
Every per-D form of the chain (pure bound + X leak <= per-frequency charge, then the programme) implies PFC-b. So
**3a.9 cannot be closed frequency by frequency.** Also: at D = 1 the true margin exceeds the pure bound, so no test
distribution at all achieves the aimed pure bound there. This is not a problem of choosing the weighting.
The network still loses (0.837) because its spectrum is unbalanced: coverage efficiency tau = 0.836. X buys margin at
the one binding offset only because the other offset has slack.

**Offset-profile independence (PROVED, all odd p).** The functions D -> 1 - w^{-gD}, g = 1..p-1, are linearly
independent on Z_p \ {0}. (If sum a_g (1 - w^{-gD}) = 0 for all D != 0, then sum_g a_g w^{-gD} = s := sum a_g for all
D, including D = 0, so by Fourier uniqueness every a_g = 0.) A junk component at row mode (-2m, 0) carries candidate
frequency g = -m, and X components at the same row mode carry other g. So **X can never cancel a junk component at
every offset**, only on a proper set of offsets. This is the mechanism behind "X helps only where coverage has slack".

## 3. X-blind aiming is one-sided (NUMERICAL, `xblind_dirs.py`)

With D uniform on Z_p \ {0} and x2 uniform: the line x1 = s + D/2 gives exactly the same average gap for a mixing net
and for its diagonal part (difference <= 2e-16, p = 5, 7). The reversed line x1 = s - D/2 does **not** (differences
0.09-0.43): it sees an X component. So the X-blind family (opus55 Theorem 1) can only push the junk one way. It also
averages the coverage over D. At p = 5 it caps at 1.115x the block (opus55/xblind_p5.log), so X-blindness alone is
FALSE as a route at p = 5. The flip trick restores two-sidedness but, with offsets concentrated as coverage needs, it
leaks X with weights lam^(-m) - lam^(-m-g) (VERIFY.md).

## 4. Unrestricted p = 5 numerics

`p5_ca_search.py T` (SLSQP, true margin / gamma_sym, width 9, 24 starts): 20 starts reach **1.000000** (blocks). The
4 starts seeded at the local max stay at 0.837046. Nothing above 1 (NUMERICAL, consistent with claude0923).
Diagonal-budget ceiling (CA: margin <= (2/p^3) Gamma (sum_k cost_k/3^{1/3})^{3/2}, stronger than (T) because it
ignores DC and imbalance slack): best found 0.903 from the local-max basin. Random starts did not converge
(nonsmooth |c|^{2/3}). Inconclusive, NUMERICAL.

## 5. The remaining gap, stated precisely

For mixing networks, the proof must bound X's lift at the binding offsets using the network's **coverage slack**
Gamma (sum Y^{2/3})^{3/2} - min_D sum_k cov_k(D) Y_k, together with its budget slack (DC, imbalance). It cannot be
done offset by offset. Target inequality (equivalent to (T) given the chain):

> **(CA+) For every network:** margin <= (2/p^3) Gamma(p) ((p/6) P / 3^{1/3})^{3/2},
> proved by exhibiting an *effective spectrum* Y~ with sum Y~^{2/3} <= (p/6) P / 3^{1/3} and
> margin <= (2/p^3) min_D sum_k cov_k(D) Y~_k.

The local max satisfies this with Y~ != Y: X acts as a *reallocation* of amplitude toward the frequency that covers
the binding offset, not a creation of it. A proof strategy consistent with every finding above:
(i) offset-profile independence forces X's net contribution to average out over any set of offsets on which the
spectrum is balanced;
(ii) the flip + plain variants give two moments per row mode, generically enough to aim at junk *and* zero the X leak
at that mode. Whether the needed moments are realisable (depth) is a finite question at p = 5, not yet checked.
Neither is proved. p = 7 was not attempted.

**Status summary:** PROVED/CERTIFIED: single-frequency theorem at p = 5 (and junk on <= 2 frequencies, any p);
elementary depth bounds; offset-profile independence. FALSE: per-frequency charging (PFC, PFC-b); X-blind route at
p = 5. OPEN: coverage-aware charging (CA+) for mixing networks, p = 5 unrestricted.

---

# Round 2 (2026-09-25): the global form

Scripts: `xcharge/zerosum.py`, `global_eval.py`, `global_slsqp.py`, `xblind_bound.py`, `caseB.py`.

## R2.1 The trade-off, made exact: X is zero-sum over offsets (PROVED)

**Zero-sum identity.** Fix any network (mixing neurons allowed) and any density rho for a start point s.
For each offset class +-D, use the flip pairing: offset D at rows (s + D/2, x2), offset -D at rows (s - D/2, x2),
with x2 uniform. Call L_D(rho) the part of the resulting average gap that comes from X. Then
sum over offset classes of L_D(rho) = 0.
*Proof.* Summing the pairings over all classes is exactly the half-offset line density x1 = s + D/2 with D uniform on
Z_p \ {0}. That density sees each row mode u through exactly one candidate frequency, g = u/2 (the D-average of
w^{u D/2}(1 - w^{-gD}) vanishes unless g = u/2). That component is same-|k| junk, never X. This is opus55's
X-blindness (Theorem 1), repackaged as a sum over offset classes. The same holds with slots swapped and for the
a + b line.
*Check:* sums are 1e-16 at p = 5, 7, 11 (`zerosum.py`). The plain pairing (same row for D and -D) is NOT zero-sum
(sums 0.02-0.8).
**Reading.** Under a shared aim, X can lift the bound at one offset only by lowering it by exactly as much at the others.
So X cannot help "on average over offsets"; it can only move margin between offsets.

**With the programme's dual.** At p = 5 the optimal spectrum is dense and flat, and the dual (binding-offset) measure
is uniform on the two classes. So the dual-weighted flipped bound is *exactly X-free*:
margin <= sum_D lambda*_D B_D = sum_D lambda*_D P_D, with P_D the pure aimed part.
This is tight at the optimal spectrum. Away from it, the dual weights over-credit coverage, and the minimum of this bound
and the clean min-offset bound is exactly opus55's X-blind bound. Its known failure (1.115x at p = 5) occurs on heavily
junky pure profiles whose true margin is negative.
So the dual-measure quantification reproduces, and cannot beat, the X-blind barrier (PROVED equivalence; the barrier
value is NUMERICAL).

## R2.2 Candidate global inequalities (all NUMERICAL, adversarial)

For every network define (`global_eval.py`):
- T = gamma/gamma_sym (the target);
- CA = margin / [(2/p^3) Gamma (sum_k cost_k/3^{1/3})^{3/2}], using diagonal-atom budgets (ignores DC and imbalance
  slack, so it is stronger than T);
- CAb = the same with per-frequency weight shares;
- **CAoff** = the smallest l^{2/3} budget of a clean spectrum whose coverage dominates the network's per-offset margins
  m_D at *every* offset, divided by the diagonal budget, to the power 3/2. This is the "effective spectrum" statement:
  CAoff <= 1 means X only *reallocates* coverage across offsets and never creates it. It is exact for every balanced
  junk-free network (CAoff = 1 there), so it is the natural target.

SLSQP exact max-min searches (`global_slsqp.py`; seeds at junky local maxima + perturbations, blocks + random
mixing/DC perturbations, random):

| p | objective | runs | max found | where |
|---|---|---|---|---|
| 5 | T | 34 | 1.000000 | blocks; local-max seeds either stay at 0.837046 or escape to 1 |
| 5 | CA | 9 | 1.000000 | blocks (random CA starts did not converge) |
| 7 | T | 15 | 1.000000 | blocks; junky local maxima 0.750841, 0.750819, 0.841127 (new), and the junk-free vertex 0.980055 |
| 7 | CA | 5 | 1.000000 | blocks; the local-max basin gives 0.769 |

Nothing exceeds 1 for T, CA or CAoff. CAoff = 1.000000 on every junk-free endpoint, including the non-optimal p = 7
vertex (T = 0.98). This is consistent with CAoff being true, with equality exactly on balanced junk-free nets.

## R2.3 Reduction of unrestricted p = 5 to one region (PROVED reduction + NUMERICAL evidence)

The X-blind family (clean row-uniform bounds at each offset, plus the three types of half-offset lines) gives, for
**every** network, a bound B_XB that depends only on the diagonal profile (G, J). This is PROVED: those lines are X-blind.
- **Case A:** B_XB <= gamma_sym P^{3/2}. Then (T) holds, X-free. Both known junky local maxima are here:
  p = 5: B_XB = 0.857 (true 0.837); p = 7: B_XB = 0.926 (true 0.751). Over 300 random block + mixing nets at p = 5 the
  largest B_XB is 0.972 (`xblind_bound.py`).
- **Case B:** B_XB > gamma_sym P^{3/2}. Since P >= (6/p) sum_k cost_k, and the population inequalities bound cost_k below
  in terms of (G_k, J_k), case B is contained in a set of *diagonal profiles* where the X-blind relaxation exceeds 1.
  Those are the heavily junky profiles of opus55's relaxation, whose pure realisations have negative margin.
  **The remaining lemma for unrestricted p = 5:** a network in case B has T < 1. X would have to lift a profile whose
  pure realisations have negative margin all the way to the ansatz value.

**Case B searched (`caseB3.py`, NUMERICAL).**
- Phase 1 pushes pure networks deep into case B (B_XB = 1.074-1.114 x target at unit cost, reproducing opus55's
  relaxation). Their true scores are -0.54 to -0.68.
- Phase 2 then maximises the *true* score over unrestricted networks (mixing + DC, width 9) while keeping
  B_XB >= (1 + delta) x target:

| delta | best true T found in case B (5 starts) |
|---|---|
| 0 (boundary allowed) | 0.398 |
| 0.01 | 0.000 |
| 0.03 | 0.190 |
| 0.08 | -0.104 |

- Separately, SLSQP maximising the true score from random starts never enters case B: it ends at the block, which
  sits exactly on the boundary (B_XB = target, T = 1).
- So case B consists of the block's boundary neighbourhood plus a far region whose networks score <= 0.4 (all found
  points; local search, not a certificate).

## R2.4 What the unrestricted p = 5 proof now needs (three pieces, one open)

1. **Case A** (X-free bound suffices): PROVED; no X accounting needed.
2. **Case B near a block:** astra/LOCAL_THEOREM.md proves every optimal ansatz is a strict *local* maximum, unrestricted
   (mixing and extra neurons allowed), n = 2, q = 2, every odd prime. What is missing is a *quantitative* radius:
   "within distance r of the block (in weight space, modulo symmetries), T <= 1". OPEN, but a local, finite computation
   at p = 5: the Hessian certificate made explicit, plus a bound on the cubic remainder.
3. **Case B far from blocks:** numerically T <= 0.4. A crude rigorous bound would do, because the gap to 1 is large.
   The natural one is: pure chain bound (flip-aimed, depth 0.29, CERTIFIED) + |X leak| <= (X mass at the aimed row
   modes) x (aim depth). Because the room is ~0.6 x target, even a lossy X-mass bound in terms of cost may close it.
   OPEN: not yet attempted.
The pieces are glued by continuity (case A/B is decided by the X-free bound, which is continuous in the weights), so the
radius in (2) must cover every case-B network with T > 0.4.

## R2.5 Status (round 2)

- **PROVED:** zero-sum identity: under a shared aim, X leaks cancel across offset classes, so X only moves margin
  between offsets and never creates it on average. At p = 5 the programme's dual measure is uniform, so the
  dual-weighted flipped bound is exactly X-free; it coincides with the X-blind bound.
- **PROVED:** reduction of unrestricted p = 5 to case B (profiles where the X-free bound exceeds the target).
- **NUMERICAL:** no violation of T, CA or CAoff at p = 5 (43 runs) or p = 7 (20 runs). CAoff ("X only reallocates
  coverage across offsets") is tight, = 1, on every balanced junk-free net and never exceeded.
  Case-B networks reach T <= 0.40, except at the block's boundary.
- **OPEN (for unrestricted p = 5):** (2) a quantitative local radius around the block, and (3) a crude X-mass bound on
  the far part of case B. Both are finite, p = 5-specific computations, not new ideas. The general-p analogue of case B
  is untouched.

---

# Round 3 (2026-09-25): the two finite p = 5 computations

Scripts: `xcharge/local_F.py`, `caseB_room.py`, `caseB_closest.py`, `Ftilde_check.py`, `Ftilde_max.py`.
**Outcome: the full unrestricted p = 5 theorem is NOT reached.** Piece (1) has a better formulation than a Hessian
radius and is numerically settled. Piece (2), as planned (a crude X bound plus the proved flip bound), is **FALSE as a
route**: the room is ~0.1-0.35x, not ~0.6x.

## R3.1 The neighbourhood of the block: use the X-free functional, not a Hessian of T

A weight-space radius for T is awkward: arbitrary width, extra small neurons and mixing all live in the neighbourhood.
There is a cleaner object. Let **F := B_XB / (gamma_sym P^{3/2})**, with B_XB the X-blind bound (R2.3). Then T <= F for
*every* network (PROVED), and F sees only the diagonal profile (G, J) and the cost. It is blind to X, DC and width.

**Numerics (`local_F.py`, p = 5, 200 directions from the block):** F < 1 on a punctured neighbourhood of the block, in
every direction family tested:

| step size | generic (junk + mixing + DC) | phase only | extra small neurons | shrink one neuron | pure DC |
|---|---|---|---|---|---|
| 1e-3 | -4.9e-4 | -8.6e-5 | -9.3e-7 | -3.2e-4 | -8.9e-7 |
| 1e-2 | -5.2e-3 | -8.6e-4 | -9.3e-5 | -3.1e-3 | -8.9e-5 |
| 0.1 | -7.8e-2 | -9.0e-3 | -9.3e-3 | -2.9e-2 | -8.9e-3 |
| 0.3 | -3.0e-1 | -3.0e-2 | -8.1e-2 | -7.2e-2 | -7.5e-2 |

(Entries are max over directions of F - 1.) F drops *linearly* along junk-creating and spectrum-changing directions
(the one-sided X-blind aim penalises junk at first order, while the cost relief from junk is only second order), and
*quadratically* along mixing / DC / extra-neuron directions (cost rises at second order, the diagonal profile moves
only at third order).
So near the block every network is in case A. **No Hessian of T and no width argument is needed.**

**How far case A extends (`caseB_closest.py`, NUMERICAL).** Minimising the worst relative junk max_k eta_k/|G_k| over
networks with F >= 1 + 1e-4 (unrestricted, width 9, started from case-B points): every feasible end point has
relative junk ~0.71 at **both** frequencies. Runs pushed toward junk-free end up infeasible, approaching the block
from inside case A. **Case B (numerically) needs relative junk >= ~0.7 at every frequency.**

**Making R3.1 rigorous: a population inequality is the bottleneck.** A rigorous version bounds P from below by the
diagonal costs, cost_k >= Phi(G_k, J_k), and certifies Ftilde := B_XB / (gamma_sym ((6/p) sum_k Phi_k)^{3/2}) <= 1.
- With the proved linear eta-form Phi = 3^{1/3}(|G| - 0.272 eta)_+^{2/3}, block phase perturbations stay below 1
  (`Ftilde_check.py`: max 0.9984, 0.9930, 0.9833, 0.9493 at phase noise 0.01, 0.03, 0.1, 0.3).
- But at the edge of the relevant range it fails: a 2-atom-per-frequency profile at relative junk 0.7 has
  **Ftilde = 1.110** (`Ftilde_max.py`), though its true F is below 1.
- So certifying case A up to relative junk ~0.7 needs a sharper population inequality, quadratic in junk and valid
  for all atom counts (P2B proved one for 3 atoms, junk <= 0.15). OPEN.

## R3.2 Case B: the planned "crude X bound + flip bound" route is FALSE (quantitatively)

The plan assumed ~0.6x room: case-B networks score T <= 0.4. But the room that matters is between the target and the
**proved** single-frequency bound on the network's diagonal profile, not the true score. `caseB_room.py` computes the
flip-chain bound PB = (2/p^3) min_D sum_k cov_k(D) (|G_k| - 0.29 eta_k)_+ on all 19 case-B end points:

| case-B points | true T | PB / target | room left for X |
|---|---|---|---|
| lone-atom-like at one frequency (relative junk ~1.73 there, 0.35-0.85 at the other) | -0.10 ... +0.40 | 0.65 - 0.72 | 0.28 - 0.35 |
| both frequencies at relative junk ~0.8 | -0.08 ... 0.00 | 0.86 - 0.90 | **0.10 - 0.14** |

(For comparison, the p = 5 junky local max, which is in case A: T 0.837, PB 0.740.)
The proved flip bound is loose on these profiles (their true margins are <= 0.4x, often negative), but it is what is
proved. The X leak under its aim must fit in 0.10-0.35x of the target, and no magnitude bound on X can do that. The
fractional-matching budget allows one X coefficient up to ~1.96x the target at p = 5 (R2), and a network's X gap
content can reach tens of times the target in sup norm. **A crude X bound cannot close case B.**
What case B needs instead is one of:
- (a) a much sharper *pure* bound on heavy-junk profiles, near their true (negative) values, so that the room becomes
  large; or
- (b) a *sign* argument for X on heavy-junk profiles, of the zero-sum kind (R2.1), but for single-offset aims.
Neither exists yet.

## R3.3 Status of unrestricted p = 5

- **PROVED:** T <= F for every network (F is the X-free functional).
- **NUMERICAL:** F < 1 on a punctured neighbourhood of the block, extending to relative junk ~0.7 at every
  frequency. Case B lies beyond that; there T <= 0.40 in all local searches.
- **OPEN, rigour of R3.1:** a population inequality with quadratic junk relief, valid for all atom counts up to
  relative junk ~0.7. P2B proves the 3-atom case only up to 0.15. The linear eta-form fails at 0.7 (Ftilde = 1.110).
- **OPEN, case B (the real gap):** an X-aware bound on heavy-junk profiles. The crude route is quantitatively
  impossible: 0.10-0.35x room against X coefficients up to ~2x.
- **Net:** unrestricted p = 5 is reduced to two statements about heavy-junk profiles (relative junk >= 0.7), plus a
  sharper population inequality for moderate junk. Not closed.

(Run note: the Ftilde searches with 3-4 atoms per frequency (`Ftilde_max.py`, caps 0.15 / 0.4 / 0.7) were stopped
after SLSQP returned infeasible end points; only the 2-atom result above (1.110 at cap 0.7) is usable. Inconclusive.)
