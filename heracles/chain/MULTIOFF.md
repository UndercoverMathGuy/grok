# MULTIOFF: multi-offset aiming for single-frequency networks (gap 1, all p)

Fork report, 2026-09-25. Scripts: `MULTIOFF/` (library `mo.py`; runs `run1.py`, logs `batch*.log`). Local CPU, no commits.
Tags: PROVED / CERTIFIED / NUMERICAL / FALSE.

## 0. Words

- **offset D**: how far the wrong answer sits from the right one (candidate s - D). **row** (a, b): one input.
- **junk profile at offset D**: the part of the gap at offset D that comes from junk, as a function of the row:
  junk_D(a,b) = Re sum_{k in T} (1 - w^{kD}) [J_{k,1} w^{-2ka} + J_{k,2} w^{-2kb} + J_{k,12} w^{-2k(a+b)}]  (x 2/p^3).
  T = the junky frequencies; J_{k,1}, J_{k,2}, J_{k,12} = the three sign-flip junk coefficients of frequency k.
- **dip at D**: dip_D(J) = max over rows of (-junk_D): how far junk pushes the worst row at offset D below the clean level.
- **offset weights lam**: a probability distribution on offsets, symmetric under D <-> -D.
  **W_k(lam)** = sum_D lam(D) (1 - cos(2 pi k D/p)) = frequency k's coverage averaged over lam.
- **eta_k** = (|J_{k,1}|^2 + |J_{k,2}|^2 + |J_{k,12}|^2)^{1/2}: junk size at k.
- **multi-offset depth** A(T, lam) = min over junk J on T of  sum_D lam(D) dip_D(J) / sum_k W_k(lam) eta_k.
  In words: the fraction of the junk that the best test distribution with offset weights lam exposes.
- **pair-offset depth** at D: the same with lam = half on D, half on -D.
- **spectrum z**: nonnegative amplitudes z_k; coverage C_z(D) = sum_k z_k (1 - cos(2 pi k D/p)); ||z|| = (sum z_k^{2/3})^{3/2}.

## 1. The multi-offset theorem template (PROVED)

**Exact exposure identity.** For a single-frequency network and offset weights lam, the best test distribution that
uses offset D with probability lam(D) and an arbitrary row distribution at each offset gives exactly
  sum_D lam(D) min_rows gap_D = (2/p^3) [ sum_k W_k(lam) Re G_k - sum_D lam(D) dip_D(J) ].
Proof: at each offset the gap is clean(D) + junk_D(row) (VERIFY V3a.2, X = 0 for single-frequency nets). The best row
at D attains clean(D) - dip_D. Averaging clean(D) over a symmetric lam gives W_k Re G_k (the sin terms cancel between D
and -D). Checked numerically: pointwise reconstruction 1e-14, min-gap identity 7e-15 (`sanity.py`).
So "aiming" has no slack left: N_lam(J) = sum lam dip_D is the best any test distribution with these offset
weights can do.

**Template.** Suppose, for the network at hand, there are offset weights lam with
 (i) aim: sum_D lam(D) dip_D(J) >= A sum_k W_k(lam) eta_k, with A >= 0.272, and
 (ii) admissibility: sum_k W_k(lam) z_k <= Gamma* ||z||, where z_k = (|G_k| - A eta_k)_+.
Then gamma <= gamma_sym, strictly if some eta_k > 0 and A > 0.272.
Proof: margin <= (2/p^3)[sum W_k |G_k| - A sum W_k eta_k] <= (2/p^3) sum W_k z_k <= (2/p^3) Gamma* ||z||.
The eta-form cost side (VERIFY V3b) gives cost_k >= 3^{1/3} z_k^{2/3}, and then the budget (B) finishes as in V3c.
(ii) holds automatically when lam sits on binding offsets of z (argmin of C_z), since then sum W z = min_D C_z <= Gamma*||z||.

**What (ii) allows (stability; PROVED to first order, sketch).** Let y* be a vertex optimum of the programme with
binding offsets D_1..D_K and KKT dual lam*, so W_k(lam*) = Gamma* y*_k^{-1/3} on its support. For z = y* + dz
(normalised), write e_i = C_z(D_i) - min_j C_z(D_j) >= 0 and S(z) = Gamma*||z|| - min_D C_z (slack).
- lam is admissible iff sum lam(D) e_D <= S(z).
- To first order, Gamma*||z|| = Gamma* + sum_i lam*_i (A dz)_i by KKT. So S(z) = sum_i lam*_i e_i + O(|dz|^2):
  lam* is admissible up to a second-order defect. The defect comes from the concavity of ||.|| on the orthant.
- Mixing in a small mass t = O(|dz|^2 / S) at the argmin offset restores admissibility.
- S(z) >= c dist(z/||z||, y*) except along the scaling direction (the binding-coverage matrix A_{D_i, k} is invertible
  at a vertex), so t = O(dist).
Reading: **a competitive network's offset weights can be taken to be (essentially) the dual lam* of the vertex it sits
near**, spread over all K binding offsets. Single-offset aiming is never forced on a near-optimal network.

## 2. Universal pair-offset floors (PROVED)

For offset D and pattern T in {1, 2, 12}, let F_{T,D} be the pattern-T part of junk_D (a function of a, b or a + b).
(F0) *Pattern isolation.* Averaging rows over the line that kills the other two patterns (b uniform for T = {1};
 a uniform for {2}; a + b fixed with a uniform for {12}) gives dip_D >= max(-F_{T,D}) and dip_{-D} >= max F_{T,D}.
 The second holds because junk_{-D} restricted to pattern T is -F_{T,D} shifted by D/2 (increment identity, Vf.1).
(F1) *Bhatia-Davis.* A mean-zero function has variance <= (max)(-min) <= (osc/2)^2. So
 (dip_D + dip_{-D})/2 >= osc(F_{T,D})/2 >= ||F_{T,D}||_2 = (sum_k cov_k(D) |J_{k,T}|^2)^{1/2}.
 (Parseval: the characters w^{-2ku}, k in T, are distinct and pairwise non-conjugate.)
(F2) *Floors.* By Cauchy-Schwarz, with c_k = cov_k(D) <= 2:
 - junk in a single pattern: pair-offset depth >= 1/sqrt(sum_{k in T} c_k) >= 1/sqrt(2|T|);
 - general junk (take the best pattern; max_T >= l2/sqrt3): pair-offset depth >= 1/sqrt(3 sum_{k in T} c_k) >= 1/sqrt(6|T|).
Checked on 240 random (T, D, J): exposure/floor >= 1.21 (single pattern) and >= 1.69 (three patterns) (`floors.py`).
**Corollary (PROVED; uses only the proven 0.272 cost side and (ii) with lam on the argmin pair):**
- at every odd p, a single-frequency network with general junk on at most 2 frequencies obeys gamma <= gamma_sym
  (1/sqrt 12 = 0.2887 > 0.272; independent of X_CHARGE's transfer-lemma proof of the same statement);
- a network whose junk lies in one pattern only, on at most 6 frequencies, obeys it too.
Multi-offset averaging does not improve these floors (Jensen goes the wrong way for sqrt). The gain has to come
from structure: twisted profiles failing to be flat.

## 3. Obstruction: dense sets are flat under every twist (PROVED)

On the dense set T = {1..h}, take single-pattern junk J_k = chi(k), where chi is the Legendre symbol (times i when
p = 3 mod 4). Gauss sums give P(y) = Re sum_k J_k w^{-2ky} = (sqrt p / 2) chi(2y) for y != 0, and 0 at y = 0.
So at every offset D the junk profile P(y) - P(y + D/2) takes values in {0, +-sqrt p/2, +-sqrt p}. Its oscillation is
2 sqrt p: both sign patterns of (chi(y), chi(y + D/2)) occur, since each occurs ~p/4 times.
Also, for every D, sum_{k=1}^h cov_k(D) |J_k| = h + 1/2 = p/2. So the pair-offset exposure at every D is
exactly sqrt p / (p/2) = 2/sqrt p, and **for every offset distribution lam the multi-offset depth of the dense set is
<= 2/sqrt p**.
Numerically exact (`legendre_all.py`, uniform over all offsets): 0.2604, 0.1881, 0.1377, 0.0630, 0.0316 at
p = 59, 113, 211, 1009, 4001 (= 2/sqrt p to 4 digits).
**Consequence:** no choice of offset weights gives a p-uniform depth bound over all sets. The dense set (and by the
same mechanism any set carrying a function flat under all twists, e.g. Rudin-Shapiro-type profiles on progressions)
must be excluded by competitiveness or by the share-floor race, not by aiming. Single-pattern junk has discount
<= sqrt(3/2) (L6 shape cap). Dense sets are competitive under that discount only while
sqrt(3/2) Gamma_dense / Gamma* > 1, i.e. p <~ 23-31 (claude0923 NOTES sec 6 table): a finite range.

## 4. Numbers on the hard cases (NUMERICAL; searches give UPPER estimates of depth)

Setup (`run1.py`): the adversary's effective spectrum puts every frequency of T at share >= eps (heavy junk) and may add
junk-free helper frequencies (pool = optimal support + 30 random). Floored optimum by multistart SLSQP; binding
offsets = those within 1e-5 of the minimum coverage; "dual" = KKT weights. Depth searches: Adam on a softmax-smoothed
exposure with annealed temperature, from structured starts (Legendre / i*Legendre / chirp / alternating / constant per
pattern, lone-atom-shaped variants) plus random starts, re-evaluated exactly. Modes: '1' = junk in one pattern;
'lone' = lone-atom shape (J = c (v1, v2, v1 v2)); '3' = general. The general-mode minimum is at most the other two;
where the '3' search ends above the '1' search, the '3' search under-explored.
Requirements: general 0.272 (proven eta-form); lone-atom shape 0.244 (sharp for the lone atom); single-pattern shape
~0.184 (2-atom population; the matching cost bound is NUMERICAL only, fable0923).

| case | eps | value/Gamma* | binding offsets | mode | pair-offset (at each binding D) | multi: uniform on binding | multi: dual | uniform over all D (reference) |
|---|---|---|---|---|---|---|---|---|
| dense p = 17 | .01 | .944 | 4 | 1 | .356-.360 | .373 | .370 | .483 |
| dense p = 17 | .01 | .944 | 4 | lone | .434-.487 | .486 | .483 | .524 |
| dense p = 17 | .01 | .944 | 4 | 3 | .365-.368 (<= .356 via mode 1) | .382 | .377 | .490 |
| dense p = 23 | .008 | .932 | 4 | 1 | .306-.310 | .350 | .331 | .410 |
| AP{1..8}, p = 113 (+helpers) | .0063 | .887 | 7 | 1 | .428-.464 | .568 | .545 | .600 |

Rigorous floors for comparison (sec 2): dense p = 17, single pattern: 1/sqrt(p/2) = 0.343 (so the pair minimum is in
[0.343, 0.356]); general junk: 1/sqrt(3p/2) = 0.198.
**Readings.**
- **Pair-offset aiming alone fixes dense p = 17 numerically:** general depth ~0.36 > 0.272, whereas single-weighting flip
  aims are capped at 0.258 there (Legendre, VERIFY Vf.3 correction). For single-pattern junk the proven floor 0.343 >
  0.272 already makes this rigorous: **PROVED: dense p = 17 with single-pattern junk (and any p with
  1/sqrt(sum_{k in T} cov_k(D)) > 0.272)**.
- Multi-offset over the binding offsets adds +4% (dense p = 17), +13% (dense p = 23), +25% (AP8 at p = 113) over the
  best single pair.
- Lone-atom-shaped junk is exposed far beyond its 0.244 requirement.

## 5. New pair-offset floors and what they prove outright

Notation: at offset D, let F_{T,D} be the D-twisted pattern-T junk function (junk_D = F_1(a) + F_2(b) + F_12(a+b)),
delta = D/2 mod p, c_k = cov_k(D), eta_k = l2 junk size, and Sigma_T(D) = sum_{k in T} c_k (T = junky set).
Key identity (from the per-pattern relation (1 - w^{-kD}) w^{-2ku} = -(1 - w^{kD}) w^{-2k(u + D/2)}):
  -junk_{-D}(a', b') = F_1(a' + delta) + F_2(b' + delta) + F_12(a' + b' + delta).
Every row choice at -D is allowed independently of the row at D (pair-offset aiming).

**Lemma P2 (two-pattern junk, PROVED).** Suppose one pattern vanishes at every frequency (J_{k,P} = 0 for all k, for
some fixed P). Shift the -D rows by (-delta, 0) if P = {2}, by (0, -delta) if P = {1}, by (-delta, -delta) if P = {1,2}.
Then -junk_{-D}(shifted row) = junk_D(row) exactly, so (dip_D + dip_{-D})/2 = osc(junk_D)/2 >= ||junk_D||_2 (Bhatia-Davis)
= (sum_k c_k eta_k^2)^{1/2} (Parseval on Z_p^2; the characters (2k,0), (0,2k), (2k,2k) are distinct).
By Cauchy-Schwarz, **depth >= 1/sqrt(Sigma_T(D))**.

**Lemma P3 (general junk, PROVED).** For the pattern pair {1, 12}, shift the -D rows by (-delta, 0). Then
-junk_{-D}(a - delta, b) = junk_D(a, b) + [F_2(b + delta) - F_2(b)]. Fix b, maximise over a in both terms, and
average over b; the F_2 bracket averages to 0. This gives (dip_D + dip_{-D}) >= E_b osc_a(g_b), where
g_b(a) = F_1(a) + F_12(a + b). The function g_b is a mean-zero trigonometric polynomial in a with
||g_b||_2^2 = Q(b) = sum_k c_k |J_{k,1} + J_{k,12} w^{-2kb}|^2. Bhatia-Davis gives osc >= 2 sqrt(Q(b)).
Q has mean M = sum_k c_k (|J_{k,1}|^2 + |J_{k,12}|^2) over b, and its fluctuation has variance
2 sum_k c_k^2 |J_{k,1}|^2 |J_{k,12}|^2 <= M^2/2. Hoelder (E Q)^3 <= (E sqrt Q)^2 E Q^2 then gives
E_b sqrt Q >= sqrt(M/1.5). The same holds for the pairs {2, 12} (shift (0, -delta)) and {1, 2} (shift (-delta, -delta),
averaging over a + b fixed). The best pair carries >= 2/3 of sum_k c_k eta_k^2. So
  (dip_D + dip_{-D})/2 >= (2/3) (sum_k c_k eta_k^2)^{1/2},  and **depth >= (2/3)/sqrt(Sigma_T(D))**.
Checks: `thm_check.py` (the resulting margin bound on 450 block + junk nets at p = 5, 7, 11, every D: never violated,
max relative slack -3.4e-4); adversarial minimisation of the ratio in `pairfloor.py` (sec 5b).

**Consequences (PROVED, using only the proven eta-form cost side at 0.272, VERIFY V3b, and the assembly V3c with
lam = the pair at the argmin offset of z = (|G| - A eta)_+, which is admissible).** Since Sigma_T(D) <= sum_{k=1}^h c_k = p/2
for every D, and Sigma_T(D) <= 2|T|:

| statement | condition | floor |
|---|---|---|
| **every single-frequency network, any junk, any support, is <= gamma_sym, equality only junk-free** | **p in {3, 5, 7, 11}** | (2/3)/sqrt(p/2) = 0.544 / 0.422 / 0.356 / **0.284** |
| same, any p | junk on <= 3 frequencies (Sigma <= 6) | (2/3)/sqrt 6 = 0.2722 |
| same, any p | junk avoids one pattern, on <= 6 frequencies | 1/sqrt 12 = 0.2887 |
| same | junk avoids one pattern, **p <= 23** | 1/sqrt(p/2) >= 0.2949 |
| p = 13 (general junk) | **fails by 4%**: floor 0.2615 < 0.272 | needs a better constant or a lower cost threshold |

p = 5 was already known (X_CHARGE); **p = 7 and p = 11 (all single-frequency networks) are new**, as are the
size-3 and one-pattern-free statements. Rigorous floors on the paper supports (worst D):
general 0.232 / 0.203 / 0.166 and two-pattern 0.349 / 0.305 / 0.250 at p = 59 / 113 / 211. So two-pattern junk is
covered on the p = 59 and 113 optimal supports; general junk is not covered by floors alone.

### 5b. Checks and the p = 13 near-miss
- Adversarial minimisation of (dip_D + dip_{-D})/2 / (sum_k c_k eta_k^2)^{1/2} over general junk (`pairfloor.py`; p = 5..31,
  dense and random T, random D, structured + random starts): **min 0.8275 >= 2/3**. Lemma P3's constant is valid with room
  (the true constant looks ~0.83).
- **p = 13 (NUMERICAL, not certified).** Keeping Lemma P3's exact variance and the actual best pair (instead of the
  "2/3" and "V <= M^2/2" shortcuts), the floor becomes the explicit function
  max_pair M^{3/2}/sqrt(M^2 + V) / sum_k c_k eta_k. Its searched minimum over all junk shapes, with c_k in [0, 2] and
  sum c_k <= p/2 (`p13_const.py`), is 0.3229 (p = 11), **0.3018 (p = 13)**, 0.2678 (p = 17, fails by 1.5%).
  A simpler provable variant (best pair >= 2/3; V <= (1/2) sum (c eta^2)^2) gives 0.2920 (p = 11) and **0.2753 (p = 13)**
  (`p13_simple.py`, search). My attempt to certify the p = 13 maximisation with a crude 1-D envelope (`p13_proof.py`)
  over-shoots (2.08 vs the needed 2.0012), so **p = 13 remains open but within reach of a finite certificate**: a
  5-dimensional simplex maximisation, with c restricted to the fixed multiset {1 - cos(2 pi j/13)}.

## 4b. Remaining numbers (NUMERICAL, upper estimates)

| case | value/Gamma* | binding | mode | pair-offset range | multi: uniform on binding | multi: dual | gain of multi over worst pair |
|---|---|---|---|---|---|---|---|
| p = 59 optimal support {1,7,11,23,27} | 1.000 | 5 | 3 | .738-.994 | 1.028 | 1.023 | +39% |
| p = 113 optimal support (K = 7) | 1.000 | 7 | 3 | .744-.844 | .936 | .928 | +26% |
| p = 211 optimal support (K = 11) | .982* | 8 | 1 | .549-.847 | .752 | .777 | +37% |
| p = 211 optimal support | .982* | 8 | lone | .749-1.107 | .963 | .992 | +29% |
| p = 211 optimal support | .982* | 8 | 3 | .569-.885 | .776 | .782 | +36% |
| AP{1..8}, p = 59 (+helpers) | .904 | 8 | lone | .563-.630 | .651 | .651 | +16% |
| AP{1..8}, p = 59 (+helpers) | .904 | 8 | 3 | .454-.491 | .578 | .577 | +27% |
| AP{1..8}, p = 113 (+helpers) | .887 | 7 | 1 | .428-.464 | .568 | .545 | +33% |
| AP{1..8}, p = 113 (+helpers) | .887 | 7 | lone | .565-.616 | .649 | .651 | +15% |
(*) the floored optimiser stopped at 0.982 of Gamma* at p = 211, i.e. a nearby vertex with 8 binding offsets.

Readings:
- **Every hard case is far above 0.272 once rows may differ between D and -D (pair-offset), and further above with
  multi-offset.** In particular the p = 211 optimal support, where the proven transfer gave 0.251 (L6), searches at
  >= 0.55 (pair) and ~0.78 (multi). The old failures were artefacts of single-weighting aims.
- The multi-offset gain over the worst single pair is 15-39%.
- These are searches (upper estimates). The rigorous floors on the same sets are 0.17-0.23 (general junk), so certifying
  them needs pair-body certificates (a transfer lemma for the difference body, or LP/SDP), not the Parseval floors.

## 6. Verdict on the directive

1. **Multi-offset aim body, formalised (PROVED).** With offset weights lam and independent rows per offset, the best test
   distribution gives exactly sum_D lam(D) (clean(D) - dip_D). The template "aim depth >= 0.272 in lam-weighted units +
   lam admissible for z = (|G| - A eta)_+" proves gamma <= gamma_sym. Near a vertex optimum the dual lam* is admissible up
   to a mixing of size O(distance) (first-order sketch, sec 1). So near-optimal networks can be aimed at over all their
   binding offsets, not just one.
2. **Numbers (NUMERICAL).** Pair-offset aiming (independent rows at +-D) already clears 0.272 by a wide margin on every
   hard case tested: dense p = 17 and 23; progressions of length 8 at p = 59 and 113; the p = 59, 113 and 211 optimal
   supports. Multi-offset over the binding offsets adds 15-39%. The p = 211 optimal support goes from 0.251 (transfer) to
   ~0.55 pair / ~0.78 multi.
3. **No p-uniform bound (PROVED obstruction).** On the dense set the Legendre direction is flat under every twist, so for
   every offset distribution the multi-offset depth is <= 2/sqrt p (exact: 0.2604 at p = 59 ... 0.0316 at p = 4001).
   Progressions carry the same mechanism (Rudin-Shapiro / chirp directions; over well-covered offsets their pair ratio is
   <= ~3/sqrt L, `ap_obstruction.py`). **Gap 1 therefore cannot be closed by aiming alone.** The closing argument must be
   the share-floor race: depth ~ c/sqrt|T| against the critical discount of |T|-frequency sets. Multi-offset raises c.
4. **New rigorous results from the pair-offset floors (PROVED):**
   - **all single-frequency networks at p = 7 and p = 11** (new; p = 5 was known);
   - any p, junk on <= 3 frequencies;
   - any p, junk avoiding one pattern on <= 6 frequencies, and at every p <= 23.
   p = 13 is a certifiable 5-dimensional maximisation away (search: 0.2753 simple / 0.3018 refined vs 0.272).
   p >= 17 needs pair-body certificates beyond Parseval.

**Remaining gap, precisely:** (a) certified lower bounds for the pair/multi-offset body on general junk (a difference-body
transfer lemma, or LP/SDP certificates), to turn sec 4's searches into proofs at the paper primes; (b) the share-floor race
for growing structured sets (dense sets excluded by competitiveness beyond small p; progressions via chain/APPROG.md),
now with depth constants raised by pair/multi-offset aiming; (c) the stability lemma (sec 1) made rigorous with explicit
constants.
