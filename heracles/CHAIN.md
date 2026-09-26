# The chain: the max-margin network for modular addition is junk-free

Started 2026-09-25. Running log of the lemma chain; update statuses in place.
Core case: two operands (nu = 3), plain weight decay (q = 2), p an odd prime.
Tags: PROVED (source), OPEN-easy / OPEN-medium / OPEN-hard, DEAD.

## Verification status (09-25, details in chain/VERIFY.md)

| step | verdict |
|---|---|
| Goal <= Step 1 (maximiser exists, block attains gamma_sym) | VERIFIED |
| Step 2 (rho tau eta split) | VERIFIED (exact identity; not load-bearing) |
| 3a.1-3a.4 (formula, bins, D/-D) | VERIFIED |
| 3a.5 (depth cos(pi/p)), 3a.6 (transfer), 3a.7 (subsets) | VERIFIED |
| 3a.8(a) Riesz | VERIFIED **only with distinct signed sums** (the weaker "dissociate" reading fails 3397/3683) |
| 3a.8(b) certified depths | VERIFIED (independent fp64 reproduction to the last digit) |
| f.1 flip lemma, f.2 two-sided exposure | VERIFIED (single-frequency nets only) |
| f.3 flipped cost side | **FALSE as stated** (A2 = 0); corrected region measured, not proved |
| f.4 two-sided >= one-sided | VERIFIED; the flip triples depth on dense sets, +5-20% on paper supports |
| f.5, f.6 | OPEN; p = 11 dense not covered by v1 or v2 |
| 3a.9 / f.7 (X) | OPEN; **load-bearing**; per-frequency form **FALSE** (p = 5 local max beats the own-spectrum per-frequency bound, ratio 1.018); needs a global coverage-slack argument |
| p = 7, 11, single-frequency nets | **THEOREM** (pair-offset floor (2/3)/sqrt(p/2) > 0.272; also any p with junk on <= 3 frequencies) |
| p = 5, single-frequency nets | **THEOREM** (depth floors 1/(2K), 1/sqrt(2K); transfer 0.29 > 0.272); also every p when junk sits on <= 2 frequencies |
| 3b cost side (0.272) | VERIFIED (certificate replay: 5279 boxes, 0 failures) |
| 3c assembly | VERIFIED (algebra + end-to-end on 160 junky nets) |

## Words

- **score** gamma = margin / cost^{3/2}, cost = sum of squared weights. Does not change if all weights are scaled.
- **block**: 3 neurons per frequency, phases 0, pi/3, 2pi/3, amplitudes set by the optimal shares
  (theory_modadd Prop 7.3). Junk-free. Its score is **gamma_sym**.
- **clean_k** = |G_k|: amplitude at frequency k of the part of the table that depends only on (answer - candidate).
- **junk_k** = eta_k: size (l2 over the three sign-flip patterns T = {1}, {2}, {1,2}) of the sign-flipped copies
  of feature k. Junk-free means eta_k = 0 for every k.
- **X**: Fourier content that mixes different frequencies. Only neurons that mix frequencies produce it.
- **coverage_k(D)** = 1 - cos(2 pi k D / p): how well frequency k separates the right answer from the one D steps away.
- **weighting w**: a probability distribution over inputs (a, b).
- **aim depth A**: the fraction of each junk term that a weighting turns into a penalty on the margin.
- **junky set T**: the frequencies with eta_k > 0.
- **share beta_k**: fraction of the cost budget spent at frequency k.
- **Gamma\***: value of the share programme (theory_modadd sec 2); gamma_sym = 3^{-1/2} B Gamma\*.

## The chain (each line is enough to prove the line above)

**Goal.** Every network with maximal score is junk-free.

- <= **Step 1.** Every junky network scores strictly below gamma_sym.
  Equivalent: a maximiser exists at finite width (astra/ARGUMENT.md sec 4); the block attains gamma_sym (Prop 7.3).
- <= **Step 2.** For every junky network, margin lost to junk >= cost saved by junk.
  Equivalent: exact split gamma/gamma_sym = rho * tau * eta^{3/2} (astra/ARGUMENT.md eq 3.1).
- <= **Step 3** = 3a + 3b + 3c, both sides measured in "aimed junk" at one common depth A:
  - **3a margin side**: margin <= sum_k coverage_k(D) (clean_k - A junk_k), for every D. Broken down below.
  - **3b cost side**: cost at k >= 3^{1/3} (clean_k - A junk_k)_+^{2/3}, any number of neurons, any junk.
    PROVED at A = 0.272 (fable0923/FINDINGS.md Lemma 3, interval certificate).
    Cannot hold below A = (sqrt3 - 1)/3 = 0.244 (a lone neuron breaks it).
  - **3c assembly**: 3a + 3b + budget + share programme => gamma <= gamma_sym, strict when junk != 0
    and A is above the 3b threshold. PROVED (fable0923 Theorem F).

## 3a in pieces

**Part 1: the margin as a formula (no choices yet)**

- **3a.1** margin <= w-average of the gap at offset D, for any weighting w. PROVED (worst <= average).
- **3a.2** that average = sum over the table's Fourier components of
  (w's coefficient at the component's row pattern) x (a factor depending on D). PROVED (exact identity).
- **3a.3** components sort into three bins: clean (constant over inputs), same-frequency junk
  (row patterns (2k,0), (0,2k), (2k,2k)), and X (everything else). Single-frequency neurons give X = 0.
  PROVED (fable0923 Lemma 1, checked to 1e-13 in reduce_verify.py).
- **3a.4** average offsets D and -D: the D-factor becomes coverage_k(D), and clean <= |G_k|. PROVED (phase lemma).

After Part 1:
margin <= sum_k coverage_k(D) [ |G_k| + Re( junk_k x w's coefficients at k's junk patterns ) ] + X-term.

**Part 2: aim w at the junk**

- **3a.5** one frequency, one pattern: any phase, depth up to cos(pi/p). PROVED.
  Uses that p is an odd prime (no +-1-valued character). The Z_12 / D_3 / S_4 counterexamples use exactly such a character.
- **3a.6** one frequency, all three patterns: costs a factor (2 + A)/3. PROVED (fable0923 Lemma 2).
- **3a.7** frequencies with no junk need no aim, and a subset is never harder to aim at. PROVED (trivial).
- **3a.8** all of T at once:
  - (a) the 3^K signed sums of the doubled frequencies 2T are all distinct ("dissociate" in the standard sense: no relation with coefficients in {-2..2}) => Riesz product, depth 1/2. PROVED (HOLES_logic.md, P3A.md; VERIFY V3a.8a). The weaker "no {-1,0,1} relation" is NOT enough.
  - (b) specific sets by computer. PROVED for 8 paper supports (HOLES_aiming.md). All sets look feasible up to p ~ 30.
  - (c) T with small total share: don't aim, use the crude price. OPEN-easy. Absorbed into (d) v2 (weighted Sidon, f.4).
  - (d) T large or structured, with a real share. OPEN-hard. Reworked as "flip, then Sidon" (v2) below.

**Part 3: charge the cross-frequency term X**

- **3a.9**
  - (a) the X components a weighting can see are those whose row pattern is a signed sum of junk patterns. OPEN-easy.
  - (b) neurons whose slots use different frequencies: leak <= their wasted budget (Hoelder slack). OPEN-medium.
  - (c) neurons spreading one profile over several frequencies act as a lone junky neuron at each;
    leak <= the penalty that junk already pays in Part 2. OPEN-medium.
  - (d) test case: the p = 5 junky local maximum (0.837 of the block; claude0923/endpoint_p5_s3.npy) must pass. OPEN-easy.

## Dead routes (do not retry)

- A fixed weighting, chosen before seeing the network: P3B duality barrier (one junky neuron beats it).
- Only weightings blind to X: tops out at 1.30x the block at p = 11 (opus55/IMPURE.md).
- Cleaning the network's own spectrum: astra/PROJECTION_OBSTRUCTION.md.
- Local / KKT classification of maxima: junky local maxima exist at p = 5, 7 (claude0923/NOTES.md sec 2, 5).
- A junk lemma that ignores the support: P3A Theorem R (Dirichlet junk hides on the dense support).
- Matrix-only relaxation at q = 2: astra/PHASE_FRAME_EXTENSION.md sec 7.
- The same aim depth on every junky frequency, and one-sided aiming generally: see "Why v1 was replaced" in 3a.8(d) v2.

## 3a.8(d), v2: flip, then Sidon (current)

**Why v1 was replaced** (v1 = aim one weighting over inputs at all junky frequencies, then a heavy/light split):
1. One common depth on every junky frequency is dead: the block plus a tiny junky neuron on every other frequency
   gives depth <= 1/(p-1) (Dirichlet kernel), yet that network obviously loses.
2. The key quantity was a *one-sided* depth (how low a sum of cosines can be forced). Junk hides from it by
   Dirichlet-type profiles: slightly negative everywhere, one big positive spike. Few tools exist for this.
3. Its last step (constants at large p) had no argument and might be false.

**The lever (f.1).** The margin may be tested at *any* (input, offset) pairs, so pair offset D at input
(x1 - D/2, x2) with offset -D at input (x1 + D/2, x2). Junk reaches the gap through the same factor
(1 - w^{-kD}) as the clean part does. Shifting the input by D/2 multiplies the junk by w^{kD}, and that turns
+coverage into -coverage. So:

> **Flip lemma.** For single-frequency neurons, this shifted pairing gives the same clean part as the unshifted
> pairing, and the *negative* of the slot-1 junk (patterns {1} and {1,2}; in terms of h_k it replaces u1 by -u1).
> Slot-2 junk is unchanged. The same holds with the slots swapped. Flipping both slots at once does NOT work:
> pattern {1,2} then gets a different factor.
> Checked: shifted + unshifted = 2 x (clean part) on every input, error 1e-16 at p = 7, 11, 13 (`chain/flip_check.py`).
> Fails for frequency-mixing neurons (error 0.2): X interacts with the flip.

Consequence: in the flipped slot the weighting can push the junk profile *either way*, so what matters is the
**sup norm** of the junk profile (two-sided). A Dirichlet spike of the profile becomes a dip after the flip.

- **f.1 Flip lemma.** PROVED (one line of algebra; numerically verified). Write it out properly. OPEN-easy.
- **f.2 Two-sided exposure.** For every offset D and every row x2 averaged out:
  margin <= clean(D) - || sum_k coverage_k(D) Re(J_{k,1} w^{-2k y}) ||_sup, the sup taken over y.
  The same for slot 2. PROVED (f.1 + worst <= average).
  - Universal floor: the sup norm is at least the l2 norm, so exposure >= (sum_k coverage_k(D)^2 |J_{k,1}|^2 / 2)^{1/2},
    with no conditions on the support.
- **f.3 Asymmetric population inequality.** CORRECTED 09-25 (chain/VERIFY.md Vf.3). The version with slot 2 unaimed
  (A2 = 0) is FALSE: two atoms with junk only in slot 2 are invisible to slot-1 aiming. Both slots must be aimed.
  Measured region (search, 2-atom populations bind): A1 needed = 0.918 / 0.612 / 0.458 / 0.367 at
  A2 = 0.2 / 0.3 / 0.4 / 0.5; symmetric; equal-depth threshold ~0.43.
  Consequence: v2 covers p = 113 (0.502, 0.413) but not p = 59 LADDER (0.429, 0.401), which v1 covers.
  RESOLVED as a detour: 3b already prices all patterns; the flip just enlarges the aim body. Corrected f.3 = l2-ball depth of hull(body, flipped bodies) >= 0.272. Search: dense p = 11 goes 0.100 -> 0.386, dense p = 13 0.083 -> 0.359 (need 0.272). Needs certificates. OPEN-medium (computation).
- **f.4 Sidon step.** Two-sided depth on a frequency set T = 1 / Sid(2T), where Sid is the classical Sidon constant
  (sup norm vs l1 of coefficients). Dissociate 2T => Sid <= 2 via Riesz products, depth >= 1/2 >= 0.423.
  Pisier: Sid bounded <=> every subset has a proportional dissociate subset. Weighted version: only heavy junk needs
  depth (light tail enters by its l1 mass). KNOWN THEORY; OPEN-medium to extract explicit constants.
- **f.5 Structure vs coverage.** Sets with Sid(2T) > 1/0.423 = 2.37 carry additive structure (no proportional
  dissociate subset), and structured sets cover some offset badly (simultaneous Dirichlet approximation), or are
  large and lose to the concentration price. OPEN-hard, but in standard structure-vs-randomness form.
  - Heuristic room: adversarial (Spencer-type) phases give sup ~ sqrt(K log(p/K)) against l1 ~ K. At competitive
    sizes K ~ 1.3 ln p the depth ratio is ~0.7-0.9, well above 0.423.
- **f.6 Small p.** p = 7...23 by computer (including the dense set at p = 11, where one slot's Sidon depth ~0.33 is
  below 0.423: needs both slots or the actual junk structure). Feasible.
- **f.7 X for mixing neurons.** Redo 3a.9 in flipped form. OPEN-medium.

**Order of attack:** f.1 write-up -> f.3 (constants for the cost side) -> f.6 at p = 11 (does it close where v1
failed?) -> f.4 explicit constants -> f.5.

## Large p: the additive-combinatorics part, broken down (09-25)

**What has to be shown.** For every competitive network (single-frequency neurons; X handled separately), its junk
can be aimed at depth >= 0.272 on the frequencies that matter, or the network loses for coverage/size reasons.
"Competitive" = passes the proven own-share bound: V_s(beta) > Gamma*/sqrt3.

- **L0 Gamma* is not too small.** Gamma*(p) >= c / sqrt(log p). Proof: a random set of K ~ 8 log p frequencies has
  max_D |sum cos(2 pi k D/p)| <= sqrt(2 K log p) (union bound); take equal shares. EASY.
- **L1 Competitiveness is carried by a small heavy set.** Light frequencies (share < eps) contribute at most sqrt(eps) to
  sum beta^{3/2} (since beta^{3/2} <= sqrt(eps) beta). Averaging over offsets gives sum beta^{3/2} >= ((p-1)/p) Gamma*/sqrt3.
  So the heavy set H (share >= eps) has <= 1/eps elements and carries the competitiveness. EASY.
- **L2 Light junk cannot pay.** Aim with zero moments on light junk modes (automatic outside the aim's spectrum), so light
  frequencies get at most the sqrt3 discount. Spreading share sigma over light frequencies buys at most ~sigma sqrt(eps)
  coverage and loses ~sigma Gamma* on the heavy part, so it is harmless for eps <= c Gamma*^2. Must handle: forced moments
  (light modes that are signed sums of heavy ones, size <= 2^{-j}); a better coverage constant than 2 (average over
  binding offsets). OUTPUT: |H| = O(1/Gamma*^2) = O(log p). MEDIUM.
- **L3 Almost-dissociate sets are aimable.** If 2H has a distinct-sums subset carrying most of the junk, and every other
  element needs a signed sum of >= j of them, then Riesz on the subset + flip gives depth >= 1/2 - O(r 2^{-j}) on H.
  Random sets of size ~1.3 ln p are like this (short relations are rare). Deliver both one-sided and two-sided depths for
  the transfer lemma. MEDIUM, explicit.
- **L4 Structure gives a coverage defect.** If H has low "dissociated dimension" d (H lies in signed sums of d generators),
  simultaneous Dirichlet approximation gives an offset D with every generator near 0 mod p, so H covers D poorly:
  sum_{k in H} beta_k^{3/2} cov_k(D) <= f(d, p) sum beta^{3/2}. Arithmetic progressions are the model case
  (D = inverse of the step). MEDIUM, explicit.
- **L5 A coverage defect loses.** [09-25: whole-set version PROVED but vacuous; the structured-subset version is killed by the zero-slack lemma; L4+L5 replaced by the share-floor critical-discount requirement, see chain/VERIFY.md "L5 fork results".] If H covers some offset poorly, that offset must be covered by light frequencies, which
  L2 bounds, so the network is not competitive once the defect is large enough. Pure programme inequality. EASY-MEDIUM.
- **L6 The middle regime.** Sets neither almost-dissociate (L3) nor low-dimensional (L4): short relations but high
  dimension. This is the actual research core. First step: quantify the gap between what L3 and L4 cover (as functions
  of d, j, p), and test the gap numerically (random sets with a few planted short relations, unions of short progressions
  with random elements) for depth and coverage. Only then look for the proof, likely through additive energy of 2H.
  HARD.

**Calibration** (`chain/ac_calib.py`, p = 1009, two-sided K-mode depth, search = upper estimates):
- progressions {1..L}: 0.87, 0.60, 0.51, 0.44, 0.37, 0.35, 0.34 for L = 2, 3, 4, 6, 8, 10, 12 (~1/sqrt(L));
- random sets: median 0.85 / 0.75 / 0.61 / 0.51 for K = 4 / 6 / 9 / 12 (min 0.70 / 0.58 / 0.52 / 0.49).
So low depth really does come from structure (progressions), and structure is exactly what spoils coverage.
