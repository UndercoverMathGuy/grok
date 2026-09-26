# Verification log for CHAIN.md

One entry per chain step, in chain order. Each entry gives the claim, the proof as I checked it, the numerical check
(script in this directory, run with `uv run --with numpy python <script>`), and a verdict.
Conventions are those of `common.py` (theory_modadd.md conventions; G_k = Zhat[k,k,-k], gap uses candidate s - D).
Verdicts: VERIFIED (proof checked + numerics agree), VERIFIED-NUM (numerics only), GAP (something missing), FALSE.

---

## V0. Goal <= Step 1

**Claim.** If every junky network scores strictly below gamma_sym (Step 1), then every maximiser of the score is
junk-free (Goal).

**Proof.** Three facts.
1. *A maximiser exists.* At fixed width the score is continuous on the compact set {P_2 = 1}, so it attains its
   maximum M(m). Padding with zero neurons shows M(m) is nondecreasing in m. Compression (fact 2) gives
   sup over all widths = M(d), d = p^2 (p-1). So the supremum over all networks is attained.
2. *Compression* (astra ARGUMENT sec 4). (a) Centring each readout w_i -> w_i - mean(w_i) changes every logit on
   a row by the same amount, so all gaps are unchanged; the cost does not increase. (b) A neuron with tensor
   a_i U_i (U_i a unit rank-one tensor, a_i = |e1||e2||w|) costs at least 3 a_i^{2/3} (AM-GM), with equality when
   balanced. (c) With the U_i fixed, move the coefficients a along a null direction beta of the active columns.
   f(t) = sum (a + t beta)^{2/3} is concave on the feasible segment, and the segment has at least one finite end
   (if beta has one sign, the side where coordinates shrink is finite; a concave function bounded below on a
   half-line is nondecreasing, so that end is no worse). So one end is no worse than t = 0, and there a coordinate
   is 0. Repeat until the active columns are independent: at most d of them. (d) Rebalance.
   Gaps unchanged, cost not increased, so score not decreased (for positive margin).
3. *The block attains gamma_sym* (Prop 7.3). So max >= gamma_sym. A maximiser has score >= gamma_sym, so by Step 1
   it is not junky.

**Numerics.**
- `v0_block.py`: the programme solver (vertex enumeration) reproduces astra's certified Gamma at p = 3, 5, 7, 11, 13, 17
  (difference <= 4e-16). The share-weighted block has table = sum_k y_k cos(2 pi k (a+b-c)/p) (error <= 4e-15),
  junk/clean <= 8e-16, margin = Gamma (error <= 4e-15), and gamma/gamma_sym = 1.000000000000 at every p.
- `v0_compress.py`: random mixed-frequency nets, p = 3 width 40 -> 8 neurons (bound 18), p = 5 width 140 -> 64
  (bound 100). Gaps unchanged to 1e-14, cost 159.3 -> 44.8 and 922.1 -> 450.7.

**Note.** Only the direction Step 1 => Goal is needed. The converse also holds but uses the symmetric-class
ceiling (Thm 7.1), which the chain does not need.

**Verdict: VERIFIED.**

---

## V1. Step 1 <= Step 2 (the rho * tau * eta split)

**Claim.** gamma / gamma_sym = rho * tau * eta^{3/2} exactly, where
- rho = margin / ((2/p^3) F), F = min_D sum_k |G_k| (1 - cos(2 pi k D/p)): worst-case vs average-case margin;
- tau = F / (Gamma H^{3/2}), H = sum_k |G_k|^{2/3}: coverage of the spectrum vs the best spectrum;
- eta = 3^{1/3} H / ((p/6) P_2): cost efficiency.
Also: rho <= 1 (Lemma 2), tau <= 1 (definition of Gamma), eta <= 3^{1/3} always (budget (B) + subadditivity),
eta <= 1 for junk-free nets (Thm 7.1).

**Proof of the identity.** The product telescopes:
rho tau eta^{3/2} = margin p^3 3^{1/2} / (2 Gamma (p/6)^{3/2} P^{3/2}) = margin (6p)^{3/2} 3^{1/2} / (2 Gamma P^{3/2}),
and gamma_sym = 3^{-1/2} 2 (6p)^{-3/2} Gamma. Equal.

**Numerics** (`v1_split.py`; 240 nets at p = 5, 7, 11, 13: blocks plus junk, random junk-free blocks, fully random):
identity error 7e-16; max rho = 1.000, max tau = 1.000, max (B) ratio = 1.000 (tight for balanced single-frequency
neurons), max eta / 3^{1/3} = 0.73, max eta over junk-free nets = 1.000.

**Status in the chain.** This is an exact restatement; it is not load-bearing. Step 3 (3a + 3b + 3c) proves Step 1
directly. Step 2 is the plain-words reading: junk can only win by pushing eta above 1, and must then lose it in rho * tau.

**Verdict: VERIFIED.**

---

## V2. Step 2 <= Step 3 (structure)

Step 3 proves Step 1 directly: 3a (margin side) + 3b (cost side) + 3c (assembly) give gamma <= gamma_sym, strictly
when junk != 0. Its pieces are verified individually below (V3a.*, V3b, V3c). Nothing further to check here.

---

## V3a.1-V3a.4. The margin as a formula

**3a.1 (worst <= average).** margin <= sum_x w(x) gap_D(x) for every weighting w and every D.
Proof: every gap_D(x) is a gap to a wrong answer, so each is >= margin. **VERIFIED** (numerics: max(margin - average)
= -0.96 < 0 over 80 random nets and all D).

**3a.2 + 3a.3 (Fourier formula, three bins).** Writing z = p^{-3} sum Zhat(al,be,ga) w^{al a + be b + ga c} and putting
c = s - D:
gap_D(a,b) = p^{-3} sum_{al,be,ga} Zhat(al,be,ga) w^{(al+ga) a + (be+ga) b} (1 - w^{-ga D}).
Components with ga = 0 drop out. Grouping by |al| = |be| = |ga| = k (8 sign patterns = clean, 3 junk patterns,
and their conjugates):
gap_D(a,b) = (2/p^3) sum_{k=1..h} Re[(1 - w^{kD}) (G_k + J_{k,1} w^{-2ka} + J_{k,2} w^{-2kb} + J_{k,12} w^{-2k(a+b)})] + X_D(a,b),
where X_D collects every component whose three frequencies do not share one |k|. A single-frequency neuron has
Fourier support only on (+-k, +-k, +-k), so X = 0 for single-frequency nets.
**VERIFIED**: `v3a_formula.py` (80 nets, p = 5, 7, 11, 13): single-frequency nets match the formula to 9e-15;
mixed-frequency nets miss it by up to 1.1 (X is present), and adding back X rebuilt from the leftover components
closes it to 7e-15. So the three bins partition everything.

**3a.4 (average D with -D).** (1 - w^{kD}) + (1 - w^{-kD}) = 2(1 - cos(2 pi k D/p)) = 2 cov_k(D), real. So, on each row,
(gap_D + gap_{-D})/2 = (2/p^3) sum_k cov_k(D) Re[G_k + junk_k(row)] + X-average, and the clean part is
<= (2/p^3) sum_k cov_k(D) |G_k|. **VERIFIED**: pointwise match 8e-15; clean-part inequality holds (max excess 3e-17 = rounding).

Note for later: without the D/-D averaging the clean part has an extra Im(G_k) sin(2 pi k D/p) term. The averaging
is needed; the flip lemma (f.1) pairs D and -D *at different rows*, which is still a valid average of gaps.

---

## V3a.5. One frequency, one junk pattern: depth cos(pi/p)

**Claim.** For one junk pattern of one frequency, a weighting of inputs can give that pattern's character
(e.g. w^{-2ka}) an average equal to any complex number of modulus <= cos(pi/p).
**Proof.** Since 2k is invertible mod p, w^{-2ka} runs over all p-th roots of unity as a runs over Z_p. The set of
averages is their convex hull, a regular p-gon inscribed in the unit circle, with inradius cos(pi/p).
Odd prime matters here: in a group with a +-1-valued character that character's average is forced real, so it
cannot be aimed at imaginary junk. The Z_12 / D_3 / S_4 counterexamples use exactly this.
**Numerics** (`v3a5_onefreq.py`, LP over 721 directions): the minimum reachable depth equals cos(pi/p) to 6 digits
at p = 3, 5, 7, 11, 13. **VERIFIED.**

---

## V3a.6. One frequency, three patterns at once: factor (2 + A)/3 (fable0923 Lemma 2)

**Claim.** If one weighting can reach depth A* on every coefficient vector of the support S (one character per
frequency, the K-mode depth), then weightings over inputs (a, b) can reach every point of the product of l2-balls of
radius A'(2 + A')/3 in the 3K junk coordinates, A' = min(A*, 0.47).

**Proof as checked.**
- *Support-function form.* Ball inclusion <=> h(z) >= A sum_k |z_k|_2 for every direction z, where
  h(z) = max over inputs of f1(a) + f2(b) + f3(a + b).
- *Step 1.* Write c = a + b. For fixed a, the inner max over c is a K-mode sum with coefficients
  u_k = z3_k + z2_k zeta_k^{-a}. So it is >= A* sum|u_k| >= A' Re sum conj(d_k) u_k for any |d_k| <= 1
  (per coordinate, chosen freely per k). The a-dependent part joins f1 as another K-mode sum, with coefficients
  z1_k + A' d_k conj(z2_k), and the certificate applies again. Result:
  h >= A'[Re sum conj(d_k) z3_k + sum |z1_k + A' d_k conj(z2_k)|].
  The six role assignments (which pattern is the outer variable, and which of the other two is left unconjugated)
  come from the six ways to eliminate variables.
- *Step 2.* Per frequency, the best d_k gives >= outer + |full - A' coupled|.
- *Step 3.* Averaging the six assignments: each coordinate is outer twice, and each ordered pair is
  (full, coupled) once. The three-variable inequality (3V) finishes it:
  2 sum a_i + sum_{i != j} |a_i - A' a_j| >= 2(2 + A') |a|_2 for a >= 0 and A' <= 0.47.

**Numerics.**
- Re-ran fable0923/tri_ineq2.py: "21 distinct symbolic vertices; PROVED on [0, 47/100]".
- Independent sampling of (3V) (`v3a6_transfer.py`, 2e5 samples): min slack +1.9e-12 (tight only at the A' = 0
  corner); fails at A' = 0.5 at a = (1, .5, .5) (-0.124), matching the stated cap.
- End-to-end, searched min of h(z)/sum|z_k| vs the guaranteed bound:
  p = 13, S = {1}: 0.843 >= 0.387; {1,2}: 0.489 >= 0.387; {1,3,4}: 0.463 >= 0.296; p = 31, {1,5,7}: 0.709 >= 0.387.

**Verdict: VERIFIED.** The transfer is lossy; true ball depths sit well above the guarantee.

---

## V3a.7. Only junky frequencies need aim; subsets are never harder

**Claim.** (i) If eta_k = 0, frequency k's junk term vanishes for every weighting. (ii) The depth of a subset S' of S
is >= the depth of S.
**Proof.** (i) From the formula in V3a.2: the junk term is J_{k,T} x (character average), which is 0 when J = 0.
(ii) The depth of S is a minimum over coefficient vectors supported on S; those supported on S' are a sub-family,
and a minimum over fewer vectors is at least as large.
**Numerics** (`v3a8a_riesz.py`, p = 13, random-direction search): depth {1,3,4} ~ 0.386 <= {1,3} ~ 0.656 <= {1} = 0.971.
**Verdict: VERIFIED.**

---

## V3a.8(a). Riesz products give depth 1/2 — PRECISE HYPOTHESIS CORRECTED

**Claim as needed.** If the 3^K signed sums sum_k eps_k (2k), eps in {-1, 0, 1}, are pairwise distinct mod p, then
for any lambda_k with |lambda_k| <= 1/2 the density w(x) = p^{-1} prod_k (1 + 2 Re(lambda_k w^{-2kx})) is
non-negative, has mass 1, and has average of w^{2kx} exactly lambda_k. Hence depth >= 1/2.
**Proof.** Each factor is 1 + 2|lambda| cos(.) >= 0. Expanding the product gives one term per eps. Mass 1: only
eps = 0 lands on mode 0. Moment lambda_k: only eps = e_k lands on mode 2k. Both use distinctness of the signed sums.
(This is the standard harmonic-analysis meaning of "dissociate": no nontrivial relation with coefficients in
{-2, ..., 2}.)
**Numerics** (`v3a8a_riesz.py`, all supports of size 2-4 at p = 11...37):
- distinct-sums supports: 577, Riesz construction exact (moments to 1e-12, w >= 0) on all 577;
- supports with merely no {-1, 0, 1} relation but some coincident sums: 3683, **moments wrong on 3397**.
**Verdict: VERIFIED under the distinct-sums hypothesis.** CHAIN.md said "dissociate" loosely; corrected there.
The weaker reading is false as a route. HOLES_logic.md already flagged this.

---

## V3a.8(b). Certified depths on the paper supports

**Claim** (HOLES_aiming.md sec 6). One-sided K-mode depths: p = 59 old {1,4,12,18,25,27} >= 0.358249; p = 59 LADDER
{1,6,14,16,19} >= 0.400911; p = 113 old {1,5,12,15,38,46} >= 0.413104; others listed there (all >= 0.358).
**Method as checked.** Polar form: depth = 1/M, M = max sum|z_k| over P = {z : f_z <= 1 on every row}. The row
shift z_k -> z_k w^{-4bk} preserves P and can bring z_1's phase within pi/p of 0, so directions with d_1 = 1 and the
other phases on an L-grid suffice: depth >= cos(pi/L) / max_d U_d. U_d is the LP value; in polar terms
U_d = 1/(largest radius reachable in direction d). Their certificates repair LP duals exactly (integer arithmetic,
70-digit trig enclosures).
**Independent re-derivation** (`v3a8b_depth.py`, my own LP code, fp64, same grid):
p = 59 LADDER, L = 6: 0.400911; p = 113 old, L = 6: 0.413105; p = 59 old, L = 8: 0.358250. These match the certified
lower bounds to the last digit (theirs are rounded down). The stored exact certificates (FINAL_RUNS/HOLES_aiming,
SHA-256 hashed) were not re-replayed.
**Verdict: VERIFIED** (independent fp64 reproduction; rigour rests on their interval / integer replay).
Side check: fixing theta_1 = 0 by the cyclic symmetry is essential. Enumerating theta_1 too (not needed) gives the
weaker 0.393 at p = 59; still valid, just looser.

---

## V3a.8(c). Small total share: absorbed into (d) v2 (weighted depth). Nothing separate to verify.

---

## Vf.1. Flip lemma

**Claim** (theory convention, candidate s - D). For single-frequency nets, put
U(a,b) = (gap_D(a,b) + gap_{-D}(a,b))/2 and S(a,b) = (gap_D(a + D/2, b) + gap_{-D}(a - D/2, b))/2. Then
U = C(D) + j1(a) + j2(b) + j12(a+b),   S = C(D) - j1(a) + j2(b) - j12(a+b),
where C(D) = (2/p^3) sum_k cov_k(D) Re G_k and j_T(u) = (2/p^3) sum_k cov_k(D) Re[J_{k,T} w^{-2ku}].
**Proof.** From the V3a.2 formula, the {1} junk at offset D on row a is Re[(1 - w^{kD}) J w^{-2ka}]. At row a + D/2
this becomes Re[(w^{-kD} - 1) J w^{-2ka}]. At offset -D and row a - D/2 it becomes Re[(w^{kD} - 1) J w^{-2ka}].
Their average is -cov_k(D) Re[J w^{-2ka}]: the sign flips. The {1,2} junk depends on a + b, which shifts the same
way, so it flips too. The {2} junk and the clean part do not depend on a, so they are unchanged.
In h_k terms: S evaluates h_k at (-t1, t2).
**Numerics** (`f1_flip.py`, 64 nets, p = 5, 7, 11, 13, every D, every row): error 6.5e-15 for single-frequency nets;
0.99 for mixing nets (X is present, and the lemma is not claimed there). **Verdict: VERIFIED.**

---

## Vf.2. Two-sided exposure

**Claim.** margin <= C(D) - max_a |j1(a)| (and the same with slot 2), for every D. Also
max|j1| >= ||j1||_2 = (2/p^3) (sum_k cov_k(D)^2 |J_{k,1}|^2 / 2)^{1/2}.
**Proof.** margin <= average over b of U(a, .) and of S(a, .), which equal C +- j1(a) (the j2 and j12 terms average
to 0 over b). Take the smaller, then the worst a. The l2 floor is Parseval over Z_p: the characters w^{-2ka},
k = 1..h, and their conjugates are distinct.
**Mixtures.** Mixing the flipped and unflipped weightings with a common slot-2 density realises h_k(tau1, t2) for any
tau1 in the symmetric hull of slot 1's moment body. So slot 1 gets the two-sided depth and slot 2 keeps the
one-sided depth.
**Numerics** (`f2_twosided.py`, 120 block + junk nets, p = 7, 11, 13): bound never violated (max margin - bound
= -4.7e-3); l2 floor never violated. Ratio of two-sided to one-sided exposure: median 1.01, max 2.04.
**Verdict: VERIFIED.** On generic junk the flip gains little (median 1.01). Its value is against structured
(spike / Dirichlet) hiding, where one-sided exposure collapses.

---

## Vf.3. Cost side for the flipped form — AS STATED IN CHAIN.md IT IS FALSE; corrected region measured

**Claim as stated.** cost_k >= 3^{1/3} (min over |t1| = A1, |t2| = A2 of Re h_k)_+^{2/3}, with slot 1 aimed two-sided at
depth A1 and slot 2 one-sided at depth A2, "A2 may be 0".
**Refuted at A2 = 0.** Two equal atoms with opposite slot-1 phases and the same slot-2 phase (v1 = (1, -1),
v2 = (1, 1)) have J1 = J12 = 0 and J2 = G: all their junk sits in slot 2. Aiming slot 1 sees nothing, so
m = G at every A1. Their cost is 2|c|^{2/3} < 3^{1/3} (2|c|)^{2/3} = 2.289|c|^{2/3}. The inequality fails for every A1
(`f3_confirm.py`). **Both slots must be aimed.**
**Measured region** (`f3_threshold.py`: for each population, the smallest A1 at which the inequality holds; maximised over
populations of 1-4 atoms, 60 restarts each; confirmed on a 16384-point grid in `f3_confirm.py`). The binding
populations are always 2 atoms. Smallest A1 needed, given A2:

| A2 | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|
| A1 needed (search) | none works | none works | 0.918 | 0.612 | 0.458 | 0.367 |
| lone-atom bound 1 - 3^{-1/2}/(1-A2) | 0.423 | 0.359 | 0.278 | 0.175 | 0.038 | 0 |

The region is symmetric in (A1, A2), so the equal-depth threshold is ~0.43 (consistent with DA proved at 1/2, and
with DEPTHA's certificate reaching 0.47). These are search values: true thresholds can only be higher.
**Consequences for v2.**
- p = 113 old support: (two-sided, one-sided) = (>= 0.502, >= 0.413); needs ~0.446 at A2 = 0.413, so it passes.
- p = 59 LADDER: (0.429, 0.401); needs ~0.457, so it fails. v1's l2-ball route (transfer lemma + eta-form 0.272)
  covers it. **v2 as posed is not uniformly stronger than v1.** The fix is to put the flip *inside* v1's l2-ball
  construction, not to replace it. OPEN.
- p = 11, dense set: (0.33, 0.10), which is outside the region. Neither v1 nor v2 covers it.
**Verdict: FALSE as stated; corrected statement (region) measured numerically, not proved.**

---

## Vf.4. Sidon step: two-sided vs one-sided depth

**Claim.** Two-sided depth (inradius of the rows' moment body together with its negative) >= one-sided depth, and
>= 1/2 whenever the signed sums are distinct (V3a.8a).
**Proof.** The symmetric hull contains the one-sided body. The two-sided depth is 1/Sid, the reciprocal Sidon constant.
**Numerics** (`f4_sidon.py`, grid-LP lower and upper bounds, fp64):

| p | support | one-sided depth | two-sided depth |
|---|---|---|---|
| 11 | all {1..5} | [0.095, 0.103] (Dirichlet ~ 1/(p-1)) | [0.313, 0.339] (~ sqrt(p)/(p-1)) |
| 13 | all {1..6} | [0.076, 0.088] | [0.260, 0.300] |
| 13 | {1,3,4} | [0.363, 0.375] | [0.559, 0.579] |
| 59 | LADDER {1,6,14,16,19} | [0.401, 0.463] | [0.429, 0.496] |
| 59 | old {1,4,12,18,25,27} | [0.336, 0.388] | [0.352, 0.406] |
| 113 | old {1,5,12,15,38,46} | [0.413, 0.477] | [0.502, 0.580] |

**Reading.** The flip triples the depth on dense sets (the Dirichlet obstruction is gone; ~sqrt(p)/(p-1) remains, a
Gauss-sum flatness). On the paper supports it gains 5-20%. **Verdict: VERIFIED** (the claim itself); the table is
exploration for f.5 and f.6.

---

## V3b. Cost side (v1 form): cost_k >= 3^{1/3} (|G_k| - 0.272 eta_k)_+^{2/3}

Re-ran fable0923/eta_cert3.py 0.272: "Claim(0.272) PROVED (5279 boxes + corner lemma on [1,1.2]x[0,0.25])",
0 uncovered boxes, 0 interval failures (log: v3b_etacert.log). Re-ran corner_verify.py: residual polynomial has all
coefficients nonnegative (log: v3b_corner.log). The reduction to the 2-variable claim (P3_ANALYTIC sec 1.1 matrix
argument) was not re-derived here. **Verdict: VERIFIED (certificate replay).**

---

## V3c. Assembly

**Claim.** If (margin side) margin <= (2/p^3) min_D sum_k cov_k(D) y_k and (cost side) cost_k >= 3^{1/3} y_k^{2/3},
with y_k >= 0 and every neuron single-frequency, then gamma <= gamma_sym, strictly when some y_k-inequality is strict.
**Proof.** min_D sum_k cov_k(D) y_k <= Gamma (sum_k y_k^{2/3})^{3/2} (definition of Gamma).
sum_k y_k^{2/3} <= 3^{-1/3} sum_k cost_k <= 3^{-1/3} (p/6) P_2 (budget (B), V1). So
margin <= (2/p^3) Gamma (p/(6 3^{1/3}))^{3/2} P^{3/2} = gamma_sym P^{3/2}.
Strictness: if eta_k > 0 and the cost inequality holds at a depth strictly below the one used, the cost inequality
at k is strict, so the whole chain is strict.
The v2 form is the same with y_k = (m_k)_+, m_k = min over the torus of radii (A1, A2) of Re h_k.
**Numerics** (`v3c_endtoend.py`): the constant chain matches gamma_sym P^{3/2} to 12 digits. End-to-end margin side,
m_k at radii (1/2, 1/2), on 160 positive-margin block + junk nets at p = 11 {1,4}, 13 {1,3}, 17 {1,2}, 19 {1,5,7}:
never violated, max relative (margin - bound) = -2.1e-4, so nearly tight. Depth 1/2 is *proved* only for the
distinct-sums supports (p = 11 {1,4}, p = 13 {1,3}); the other two are empirical. **Verdict: VERIFIED.**

---

## Vf.5. Structure vs coverage — OPEN (no claim to verify yet). Exploration data in Vf.4.

---

## Vf.6. Small p

Using the Vf.3 region and the Vf.4 depths: at p = 11 the dense support has (two-sided, one-sided) ~ (0.33, 0.10),
outside the region; v1 fails there too (one-sided 0.10). Small-p closure needs either the flip inside the l2-ball
route or the network's actual junk structure (Vf.3 binding populations are 2-atom and slot-concentrated).
**Verdict: OPEN; the obvious small-p plan does not close p = 11.**

---

## V3a.9 / Vf.7. Frequency-mixing neurons (X) — OPEN, and load-bearing

**3a.9(a)** (which X a weighting sees): immediate from the V3a.2 identity, which is linear in the weighting. Components
whose row mode lies outside the weighting's Fourier support average to zero. **VERIFIED** (via V3a.2's X rebuild).
**3a.9(d) test case** (`v3a9_p5.py`, claude0923/endpoint_p5_s3.npy, width 9): score 0.837046 of gamma_sym, X share 0.135.
Pure-part bounds, computed from the clean + same-|k| junk content alone as if X were absent:

| D | true margin | clean C(D) | pure one-sided bound | flip slot 1 | flip slot 2 |
|---|---|---|---|---|---|
| 1 | 0.00520 | 0.00532 | **0.00185** | **0.00419** | **0.00419** |
| 2 | 0.00520 | 0.00751 | 0.00573 | 0.00691 | 0.00691 |

At D = 1 the true margin **exceeds** every pure-part bound. So X actively cancels junk dips in this network: the
pure-class bounds are false for mixing nets, and 3a.9 must genuinely charge X for its cost. (It still loses overall,
0.837, but only after paying for X.) **Verdict: OPEN; confirmed load-bearing.**

---

## Vf.3 revisited: the product-torus detour was unnecessary

**CORRECTION (L6 fork, checked by me with `check_l6.py`): the dense-set values in the table below (0.386, 0.359) were
search misses.** Single-pattern directions cap the flip-enlarged depth at the two-sided depth. Legendre-symbol
(Gauss-sum) directions give <= 0.3317 (p = 11), <= 0.3005 (p = 13), <= 0.2577 (p = 17). **Dense p = 17 is below 0.272**, so
single-offset flip aiming plus the proven cost side cannot certify it; the new levers (pair-offset difference body,
shape caps, graded cost side) are needed there.

The proven cost side 3b (eta-form, 0.272) already prices all three junk patterns jointly, for any aim that can reach
the l2-ball of radius A in the 3K junk coordinates. The flip lemma holds pointwise for any weighting, so it simply
*enlarges the aim body*: body' = hull(body, F1 body, F2 body), where F1 negates patterns {1},{1,2} and F2 negates
{2},{1,2}. No new population inequality is needed. The corrected f.3 is:
**the l2-ball depth of the enlarged body is >= 0.272 on the network's junky frequencies.**
Search (`flip_ball.py v2`, structured + random starts; UPPER estimates, i.e. explicit bad directions):

| p | support | plain body | flip-enlarged | need |
|---|---|---|---|---|
| 11 | dense {1..5} | 0.100 | 0.386 | 0.272 |
| 13 | dense {1..6} | 0.083 | 0.359 | 0.272 |
| 59 | LADDER {1,6,14,16,19} | 0.481 | 0.497 | 0.272 |
| 59 | old {1,4,12,18,25,27} | 0.433 | 0.433 | 0.272 |

The dense sets, previously hopeless, now sit ~0.1 above the requirement. Lower-bound certificates are still needed.

---

## New lemma (09-25): flip-enhanced transfer — PROVED (derivation); constant computed numerically

**Statement.** Let A* be the one-sided and A_s the two-sided K-mode depth of S. Then the flip-enlarged aim body contains
the l2 polyball of radius c(A*, A_s), where
c = max over weights lam on the 6 elimination orders of min over (x, y, w) >= 0 with unit norm of sum_sigma lam_sigma beta_sigma,
beta_sigma = max( A_in x + A_out |y - A_in w| , A_out (y + A_in w) - A_in x ),
with (A_in, A_out) = (A_s, A*) when the outer (eliminated-last) variable is slot 1 or slot 2 (4 orders), and (A*, A*) when
it is a + b (2 orders).
**Proof.** Same as fable0923 Lemma 2 (V3a.6), with one change. With slot 1 outer, the inner function of c is
g_a(c) = f3(c) + f2(c - a), and the slot-2 flip F2 negates exactly f2 and f3. So
max(h_plain, h_F2) >= max_a [ f1(a) + max|g_a| ] >= max_a [ f1(a) + A_s sum_k |u_k(a)| ].
The same holds with the slots swapped (F1). Inner-over-b elimination: coefficients z2 + z3 zeta^a; conjugation handles
the twist. With a + b outer, the needed flip (negate {1} and {2}, keep {1,2}) is not available, so those two orders
stay one-sided. The support function of a hull is the max of the support functions, and an average over orders
lower-bounds that max.
**Numerics** (`transfer_flip.py`, sphere grid, not yet an exact certificate): reproduces fable exactly when A_s = A*
(0.3200 at 0.40; 0.2720 at 0.3476). With flips: p = 113 old 0.332 -> 0.372; p = 59 LADDER 0.321 -> 0.334;
**dense p = 11: 0.066 -> 0.146; dense p = 13: 0.053 -> 0.119 (need 0.272)**.
**Reading.** A real improvement, but the transfer is lossy exactly on dense sets: the outer step stays one-sided
(Dirichlet-limited). The true enlarged depth there is ~0.36-0.39 by search. So dense small p needs a sharper certificate
than transfer-from-1-D.

## X leak formula (exact, for planning 3a.9)

For a test distribution with offsets D ~ lam (symmetric), and rows (a + sD, b), a ~ w, the component (al, be, ga) with
row mode (u, v) = (al + ga, be + ga) enters with weight w^(u,v) [lam^(u s) - lam^(u s - ga)], lam^(t) := E w^{tD}.
At a slot-1 junk row mode (-2m, 0), with s = 1/2: junk (ga = -m) gets -(1 - lam^(-m)) (flipped); an X component
(ga != +-m) gets lam^(-m) - lam^(-m - ga). Uniform lam makes every X weight vanish (opus55's X-blind family) but also
averages the coverage; lam concentrated on binding offsets keeps coverage but leaks X with these weights.
The charging lemma must control sum_ga X_ga [lam^(-m) - lam^(-m - ga)], where X_ga = sum_i e1^_i(-2m - ga) e2^_i(-ga) w^_i(ga).

---

## Piece 2a (light junk, all p): first pass — crude bounds are far too weak

Setting: single-frequency net, budget normalised so sum_k cost_k <= 1. Heavy set H is aimed at depth >= 0.272;
light frequencies (share <= eps each) are not aimed.
- Worst-case light bound: the weighting's moments at unaimed modes have modulus <= 1, so y_k <= |G_k| + ||J_k||_1
  <= 4 sum|c_l| <= 4 cost_k^{3/2} (vs the block's cost^{3/2}/sqrt3), and coverage <= 2. Result: light junk is harmless
  only if eps <= Gamma*^2/192 ~ 5e-4.
- If the aim has zero moments at the light junk modes (true for light frequencies outside the aim's spectrum), then
  y_k <= |G_k| <= cost_k^{3/2}, a sqrt3 discount. Harmless if eps <= Gamma*^2/16 ~ 5e-3.
- Replacing max_D coverage <= 2 by the average over the heavy part's binding offsets could give ~2e-2.
Even the last means "heavy" = share >= 2%, i.e. up to ~50 frequencies: no restriction at paper primes. 2a is not a
quick lemma. It must use where the light frequencies sit relative to the binding offsets, and it is entangled with 2b.
**Status: OPEN; crude routes quantified and insufficient.**

---

## L0. Gamma*(p) >= 1 / (4 sqrt(2 ln h)), h = (p-1)/2 — PROVED

Pick K = ceil(8 ln h) frequencies uniformly from {1..h} (repeats allowed; a repeated frequency just gets a larger share,
which is still feasible because m^{2/3} <= m). For a fixed offset D != 0, S_D = sum_i cos(2 pi k_i D/p) is a sum of K
independent terms in [-1, 1] with mean (1/h) sum_{k=1}^h cos(2 pi k D/p) = -1/(2h) <= 0. Hoeffding:
P(S_D > t) <= exp(-t^2/(2K)). Union bound over the h offsets (D and -D coincide): with t = sqrt(2 K ln h), some choice has
S_D <= t for every D. With equal shares y = K^{-3/2} (sum y^{2/3} = 1), min_D sum y (1 - cos) >= K^{-1/2}(1 - sqrt(2 ln h/K))
= K^{-1/2}/2 at K = 8 ln h. (Weak constant: 0.088 at p = 113 vs the true 0.27. The order 1/sqrt(log p) is what L2 needs.)

## L1. Competitive networks are carried by a small heavy set — PROVED

Own-share bound (proved, claude0923/NOTES sec 8): gamma <= B V_s(beta), V_s(beta) = min_D sum_k beta_k^{3/2} cov_k(D). The block
scores 3^{-1/2} B Gamma*, so a network can only win if V_s(beta) > 3^{-1/2} Gamma* ("competitive"). Split into heavy H
(beta_k >= eps) and light L. Since beta^{3/2} <= sqrt(eps) beta on L and cov <= 2,
max_D sum_{k in L} beta_k^{3/2} cov_k(D) <= 2 sqrt(eps) sigma_L <= 2 sqrt(eps). Hence the heavy part alone satisfies
V_s(beta restricted to H) >= 3^{-1/2} Gamma* - 2 sqrt(eps): **it covers every offset almost as well as the whole network**,
and |H| <= 1/eps. Taking eps = c Gamma*^2 and L0 gives |H| = O(log p).

---

## X_CHARGE fork results (09-25) — checked

**PROVED, checked by me:**
- *General depth floors for any K-set:* one-sided A* >= 1/(2K). (Polar normalisation f_z <= 1: g = 1 - f_z >= 0 has mean 1,
  and a nonnegative function's Fourier coefficients are bounded by its mean, so |z_k|/2 <= 1 and sum|z_k| <= 2K.)
  Two-sided A_s >= 1/sqrt(2K) (Parseval under |f_z| <= 1).
- *Transfer constant at (A*, A_s) = (1/4, 1/2) is >= 0.29.* Their exact check covers 26 vertices; my independent sphere-grid
  computation (`check_xcharge.py`, `transfer_flip.py`) gives 0.2917. It is > 0.272.
- **THEOREM (p = 5, single-frequency networks, any width, any junk): gamma <= gamma_sym, equality only for the junk-free block.**
  Chain: junky set within {1,2} (K <= 2), so flip-enlarged depth >= 0.29 > 0.272. Then the eta-form cost side (V3b)
  and the assembly (V3c) give the bound, strict because 0.29 > 0.272.
  *Corollary (all odd p):* the same holds for single-frequency networks whose junk sits on at most 2 frequencies.
- *Offset-profile independence:* the functions D -> 1 - w^{-gD}, g = 1..p-1, are linearly independent on the nonzero
  offsets. (A vanishing combination makes D -> sum a_g w^{-gD} constant on all of Z_p, and Fourier uniqueness gives
  a_g = 0.) So X can cancel a junk component on a proper subset of offsets only.

**FALSE, checked by me:** the per-frequency form of the chain for mixing networks.
At the p = 5 junky local max, margin 0.005199 > (2/p^3) min_D sum_k cov_k(D) (cost_k/3^{1/3})^{3/2} = 0.005107
(ratio 1.0180, `check_xcharge.py`). So no argument of the form "pure bound + per-frequency X charge, then the programme"
can handle mixing networks. The network still loses overall (0.837) because of coverage slack (tau = 0.836): X buys
margin only at the one binding offset. **3a.9 must be global:** X's help must be paid for from coverage slack across
offsets, not frequency by frequency.

---

## L5 fork results (09-25) — checked

- **Whole-heavy-set defect loses — PROVED, but vacuous.** margin <= B V_s(beta) and
  V_s <= sum_H beta^{3/2} cov(D0) + 2 sqrt(eps) sigma_L <= delta + 2 sqrt(eps). So delta + 2 sqrt(eps) < Gamma*/sqrt3 means
  the network loses. Correct (own-share bound only), but it is the contrapositive of L1: a competitive network's heavy
  set never has such a defect.
- **Zero-slack lemma — PROVED (checked).** In the share programme, give any single frequency k0 an amplitude discount
  delta > 1 (same budget buys delta times the amplitude) and the value strictly exceeds Gamma*.
  Proof: dilations k -> lambda k permute the frequencies and the offsets, so they preserve the programme. They act
  transitively on frequencies up to sign, so some optimal support contains k0. Keep those shares; k0's amplitude grows,
  and cov_{k0}(D) >= 1 - cos(2 pi/p) > 0 at every offset. So every binding coverage strictly increases.
  **Consequence: "structured sets have a coverage defect, so junk on them loses" (L4 + L5 as planned) cannot work.** Any
  uncancelled discount on any heavy frequency pays in the programme. What survives is the share floor: each heavy
  frequency carries >= eps of the budget, which limits how the programme can re-balance. Their numerics: progressions of
  length 5-8 at p = 23, 59 with floors 3-6% have critical discount ~1.12-1.38, lowering the needed depth on them from
  0.244 to ~0.12-0.20. Full discount (sqrt3) still pays (1.19-1.53x). NUMERICAL.
- **Revised requirement for all p** (replaces L4 + L5): for every heavy junky set T with share floor eps, the discount
  left at T's achievable aim depth must be <= T's critical discount under the floor. Structure lowers the achievable
  depth and raises the critical discount; the question is which wins.
- Untested lever (flagged by the fork): junk must hide at every near-binding offset at once. Increments at different
  offsets have different coefficient profiles (cf. the flip lemma), which may force exposure beyond single-offset
  aiming.

---

## L2 fork results (09-25) — checked

**LEMMA L2 — PROVED (proof read line by line).** Single-frequency network; heavy frequencies satisfy the chain's
heavy-side conclusion (y_k with cost_k >= 3^{1/3} y_k^{2/3}); the aim's per-atom factor on light frequencies is bounded
by R. Then score < gamma_sym for every network with light share sigma > 0, as soon as eps < Gamma*^2/(12 R^2).
Checked steps:
- light atoms: Re sum c Psi <= R sum|c| <= R cost^{3/2} (l1 <= l_{2/3});
- test at the heavy part's worst offset D0: f_H(D0) <= Gamma* (sum_H cost/3^{1/3})^{3/2};
- light coverage at D0 is a permutation of {1 - cos(2 pi j/p)}; the convex maximum sits at a vertex; rearrangement;
- chord bound 1 - (1 - sigma)^{3/2} >= sigma.
R values checked: R = 1 (no aim mass at light junk modes); R = (1 + tau)^2 for product aims, so R = 25/16 for Riesz aims
on distinct-sums heavy sets (a forced moment needs >= 2 factors of size <= 1/2); R = 4 in general.
**Constants:** clean 0.083/R^2; a 4000-cell floating-point refinement gives ~0.18/R^2; the numerical truth is ~0.35-0.49
(R = 1).
**Interface problem (real):** small R needs aims whose Fourier support misses the light junk modes. Riesz aims give that
only on distinct-sums heavy sets (size <= ~log_3 p). The transfer-lemma aims have R = 4 (eps < Gamma*^2/192). And
|H| <= 12 R^2/Gamma*^2 is larger than h at paper primes. So L2 helps only asymptotically, and only if L3 supplies heavy
aims with controlled Fourier support.

---

## L4 fork results (09-25) — checked where marked

**PROVED (checked):**
- *Fejer-product aiming* generalises Riesz: if 2H has no nontrivial relation with integer coefficients of size <= N+1,
  the product of Fejer kernels (nonnegative, coefficient N/(N+1) at the first harmonic) realises depth N/(N+1). Standard.
  Contrapositive: depth < 1/2 forces a relation with coefficients in {-2..2}.
- *Progressions of length L:* one-sided depth <= 1/L (Fejer coefficients with a minus sign: max = 1/2, l1 = L/2);
  two-sided depth >= 1/sqrt(2L) (Parseval) and <= 2/sqrt(L) (flat polynomial).
- *Aims are capped by the two-sided depth:* restricting to single-pattern directions, the flip-enlarged polyball depth
  (and hence the transfer) is at most the two-sided depth. So on a progression every such aim is <= 2/sqrt(L).
- *Coverage defects:* progressions (D = step^{-1}), generalised progressions (box-Dirichlet), small dissociated
  dimension (useful only while d <~ ln p / ln(4j)). "Low depth => global coverage defect" is false as a route (depth only
  falls when frequencies are added; coverage does not), and zero-slack makes defects irrelevant on their own anyway.

**NUMERICAL (race: achievable depth vs depth required under the share floor):**
- small p: {1,2,3} at p = 23, 59 and {1..5} at p = 59 are won by the *proved* transfer bound (0.200 vs 0.168);
  {1..5} at p = 23 misses by 0.009 (search 0.43 clears).
- p = 113, progressions L = 4..16: required 0.12 / 0.09 / 0.08 / 0.04 (L = 4 / 8 / 12 / 16). Achievable with searched
  two-sided depth: wins everywhere. With proved inputs: roughly a tie (+-0.03), losing at L = 4, 8, 12.
- The critical discount under the forced share floor rises toward sqrt3 as L grows (1.73 at forced share 0.64).
- p = 1009: every set with two-sided depth <= 0.4 contained a progression or a 2-D generalised progression.

**New reduction for progressions (all p):** it suffices that
delta_crit(AP_L, eps) >= sqrt3 (1 - sqrt3 a_L), where a_L is a *certified* depth for AP_L (Parseval gives
1/sqrt(2L), i.e. sqrt3 - 2.12/sqrt L; the truth is ~ sqrt3 - 6/sqrt L with a = 2/sqrt L). That is a statement purely
about the share programme with discounted progression frequencies. Supported numerically at p = 113 only.

**Open (from L4):** certified two-sided depth beating Parseval on progressions / generalised progressions (the
certification bottleneck, a constant factor); the programme statement above for all L and p; the same for 2-D
generalised progressions; the lone-atom discount curve below 0.272 (only its endpoints are proved).

---

## X_CHARGE round 2 (09-25) — logic checked

- *Zero-sum identity:* averaged uniformly over offsets, the flipped pairing's X-leak weights vanish. This is opus55's
  X-blindness restated; consistent with the leak formula (uniform lam gives lam^(us) - lam^(us - ga) = 0). OK.
- *Reduction of unrestricted p = 5 (logic checked):* B_XB (the X-blind bound: opus55's family, a PROVED bound for every
  network) depends only on the per-frequency (G, J). Networks with B_XB <= target are done with no X accounting. **Both
  known junky local maxima are in this case (B_XB = 0.857 at p = 5, 0.926 at p = 7), so they are rigorously below
  gamma_sym.** Remaining "case B": profiles whose X-blind relaxation exceeds the target (up to 1.115x at p = 5). The block
  sits exactly on the boundary (its X-blind bound equals the target).
- NUMERICAL: no violation of gamma <= gamma_sym, of CA, or of CAoff (every network's per-offset margins are dominated by
  some clean spectrum with no more budget) in 63 targeted searches at p = 5, 7. Case-B networks optimised for true score
  reach <= 0.40. New junky local max at p = 7: 0.841.
- **Open, finite, p = 5 only:** (1) a quantitative neighbourhood around the block where it is the maximum (astra's local
  theorem made explicit: Hessian certificate + cubic remainder); (2) a crude X bound for case-B networks far from the
  block, added to the proved single-frequency flip bound (room ~0.6x of the target).

---

## L6 fork results (09-25) — checked where marked

**PROVED (checked):**
- *One clean pattern caps the discount at sqrt(3/2).* If a population's junk vanishes on any one pattern, the matching
  2x2 chain matrix is triangular: chain vector (1, v1) for {1}, (1, v2) for {2}, (1, v1 v2) for {1,2}. Theorem 6.1 with
  n = 1 then gives cost >= 2^{1/3} |G|^{2/3}, a discount of at most (3/2)^{1/2}. The full sqrt3 discount needs the
  lone-atom shape (all three patterns present and tied).
- *Pair-offset lever:* independent aims at +D and -D reach i diag(s)(M - M), s_k = Im w^{kD/2}, i.e. the difference body.
  So depth >= A_osc/(2 max|s_k|) and >= A_s. (Derivation consistent with the leak formula; not re-derived in full.)
- *Universal floors:* max(A_s/3, 1/sqrt(6K)) <= flip-enlarged depth <= A_s.
**CORRECTIONS:**
- Vf.3 revisited (above; confirmed).
- L5's floored-programme optimiser under-reported the value by up to ~10% (0.88-0.97 vs the true 1.000), so L5's
  delta_crit and the required depths quoted in L4/L5 are biased **low**. `L6/floor_opt.py` reproduces 1.000.
- The proven transfer bound drops below 0.272 even on the optimal support at p = 211 (0.251); one-sided depths fall
  with p (0.45 -> 0.29 -> 0.25). So the v1 route (Theorem F style) fails at larger paper-type primes without levers.
**NUMERICAL:** in the gap scan (p = 59, 113, 211) the only cases where accounting fails (ratio > 1) are the length-8
progression at p = 59, an adversarial competitive 10-set at p = 59, and dense p = 59 at eps = 0.005, all under
single-offset accounting. Every one closes once a lever is applied: pair-offset depth, lone-atom-shape depth (0.38 vs
0.14 dense), or the sqrt(3/2) cap. Real junky dense networks at p = 59 reach only 0.14-0.52 of the ansatz.
**Proposed closing statement (SG, shape-graded exposure):** for every heavy junky set T and junk shape sigma, the
shape's discount at the shape-restricted difference-body depth is <= delta_crit(T, eps). Open pieces: the shape-cap
interpolation for mixed shapes (computable); lone-atom directions deeper than single-pattern ones (numerical); the
three-pattern difference-body depth; floor waste for single-pattern junk at tiny floors.

---

## L3 fork results (09-25) — logic checked

**PROVED (logic checked; per-set numbers are exact integer relation counts):**
- *L3-A:* a Riesz product over H with ratio rho <= 1/2 has depth >= min_k (rho - E_k)/(1 + E_0) one-sided, and
  min_k (rho - E_k^odd)/(1 + E_0^even) two-sided. E_0 and E_k sum rho^{support size} over the signed relations
  (resp. alternative representations of 2k). Proof: support function. Choose phases aligned with z; the error terms
  are bounded by triangle inequalities. The two-sided version sees only even-support relations in the mass and
  odd-support ones in the moments, so a single Schur triple a + b = c is invisible to it.
- *L3-B:* strongly dissociate B plus r extra elements needs minimal error-pattern length j >= 4 / 6 / 7 / 9
  (r = 1..4) for one-sided depth >= 0.3476.
- *L3-C:* random sets of size c ln p with c < 1.44 have depth -> 1/2, but the rigorous threshold is huge
  (ln p >= 33 at c = 1.3).
- *L3-D (for L2):* the leak at light junk modes is tau = rho^2 one-sided (R = 25/16) and rho^3 two-sided
  (R = 81/64) on strongly dissociate sets. The flip deepens the aim and also cuts the leak.
**NUMERICAL:** at the competitive size K ~ 1.3 ln p the lemma does NOT reach the requirement for p <= 1e5. The two-sided
median is 0.351 at p = 1e5, polyball 0.258 < 0.272. It loses real depth to the triangle inequality: 0.375 vs a true
>= 0.540 on an example at p = 1009.
**Structural gap (important):** low-leak aims are per-slot Riesz *product* aims, whose cost side is the product-torus
inequality. That is proved only at (1/2, 1/2); the measured equal-depth requirement is ~0.43 (Vf.3 region). The
l2-ball aims matching the proven 0.272 cost side come from the transfer lemma, with uncontrolled leak (R = 4).
**To close:** certify the product-torus region below 1/2; or build a low-leak l2-ball aim (a transfer from Riesz
pieces with tracked leak); or use a better density than a single Riesz product.

---

## X_CHARGE round 3 (09-25)

- PROVED (checked): gamma/gamma_sym <= F for every network, where F = X-free bound / target. F depends only on the
  per-frequency clean/junk profile and the cost, so width, mixing neurons and DC never enter it.
- NUMERICAL: F < 1 in 200 directions around the block (linear decrease along junk/spectrum directions, quadratic
  along mixing/DC/extra-neuron directions). So near the block every network is in case A without any Hessian.
  The nearest case-B network found has relative junk ~0.71 at both frequencies.
- To make this rigorous: a population inequality with *quadratic* junk relief, valid for any number of atoms, up to
  relative junk ~0.7. The linear eta-form is too weak (a 2-atom profile at junk 0.7 gives F~ = 1.110 with it, although
  its true F < 1). P2B's quadratic version covers only 3 atoms and junk <= 0.15.
- **Case B (heavy junk): the planned route FAILS.** The proved single-frequency flip bound on case-B endpoints is
  0.65-0.90 of the target, leaving only 0.10-0.35 of room, while X coefficients can reach ~2x the target per unit cost.
  So no magnitude bound on X fits. It needs a sharper pure bound for heavy-junk profiles, or a sign / zero-sum
  argument for X under single-offset aims. **p = 5 unrestricted remains open.**

---

## CAOFF fork (09-25) — checked where marked

- **Conservation law — PROVED (checked by hand).** At a junk row mode of frequency m, the gap's coefficient as a function
  of the offset is Phi(D) = J (1 - w^{mD}) + sum_ga X_ga (1 - w^{-ga D}), with ga != +-m for X. Summing against w^{-mD}
  over all D kills every X term and leaves -pJ, so J = -(1/p) sum_D Phi(D) w^{-mD}. Consequences:
  - sum_D |Phi| >= p|J|: X can cut junk's total (l1) visibility across offsets by at most 1 - pi/4 ~ 21%;
  - min over X of sum_D |Phi|^2 = p^2 |J|^2/(p-1), against 2p|J|^2 without X: at most ~50% in squared terms;
  - hiding junk on all offsets but one forces a spike of size p|J| there.
- FALSE (numerical, exact re-evaluation): CAoff with the diagonal-atom budget, in every pairing (up to 1.22 at p = 5).
  A neuron whose three slots sit on different frequencies has zero diagonal atoms but moves margins.
- SURVIVES: unpaired offset margins with the FULL weight-norm budget. No violation in 106 adversarial ascents at
  p = 5-13. Implies the conjecture; holds for all junk-free networks; holds for single-frequency nets wherever the
  chain works. Its dual (a non-convex union of vertex pieces; the convex relaxation is provably too weak) was worked
  out. At p = 5 it still meets opus55's 1.115 barrier on heavy junk.
- So a mixing-neuron proof must charge X through the full weight norm. Proposed route (not done): exposure >= depth x
  visibility per offset + the conservation law + a charge for X's visibility reshaping.

---

## MULTIOFF fork (09-25) — proofs of Lemmas P2/P3 checked line by line

- *Key identity (checked):* (1 - w^{-kD}) w^{-2ku} = -(1 - w^{kD}) w^{-2k(u + D/2)}, so junk at -D is minus the junk at D
  shifted by delta = D/2 in every argument. Rows at +D and -D may be chosen independently (margin <= average of gaps).
- **Lemma P2 (PROVED):** if one junk pattern vanishes everywhere, a suitable shift makes junk_{-D}(shifted) = -junk_D,
  so the two dips add up to the oscillation of junk_D, which is >= 2||junk_D||_2 (Bhatia-Davis for mean-zero functions).
  Parseval and Cauchy-Schwarz then give **depth >= 1/sqrt(Sigma_T(D))**, Sigma_T(D) = sum_{k in T} cov_k(D).
- **Lemma P3 (PROVED):** general junk. Shift one slot; fix the other variable; the leftover bracket averages to 0.
  osc >= 2 sqrt(Q(b)); Var Q <= M^2/2 (AM-GM per frequency); Hoelder (EQ)^3 <= (E sqrt Q)^2 E Q^2 gives E sqrt Q >= sqrt(M/1.5);
  the best of the three pattern pairs carries >= 2/3 of sum c eta^2. Result: **depth >= (2/3)/sqrt(Sigma_T(D))**.
  Numerical true constant ~0.83 (adversarial min 0.8275).
- **THEOREMS (PROVED; with the proven eta-form cost side V3b at 0.272 and the assembly V3c; D-dependent pair aims are
  allowed per offset). Since Sigma_T(D) <= p/2:**
  - **every single-frequency network (any width, any junk) is <= gamma_sym, equality only junk-free, for p = 3, 5, 7, 11**
    (floors 0.544 / 0.422 / 0.356 / 0.284 > 0.272). p = 7 and p = 11 are new.
  - any p: junk on <= 3 frequencies ((2/3)/sqrt6 = 0.27217 > 0.272);
  - any p: junk avoiding one pattern on <= 6 frequencies (1/sqrt12); or avoiding one pattern at any p <= 23.
- p = 13: fails by 4% with the clean constant (0.2615). Keeping the exact variance and the actual best pair gives a
  searched 0.3018. Closing it is a finite 5-dimensional certificate. Not done.
- **Obstruction (PROVED per the fork; logic plausible, not re-derived):** Legendre-shaped junk on the dense set stays flat
  under every twist, so for ANY offset weighting the aim depth is <= 2/sqrt(p). **Aiming alone cannot give a p-uniform
  depth.** Gap 1 must go through the share-floor race (depth ~ c/sqrt|T| against the critical discount of competitive,
  i.e. small, heavy sets).
- NUMERICAL: pair aiming clears 0.272 widely on the hard cases (dense p = 17 ~0.36; progressions of length 8 ~0.43-0.49;
  optimal supports p = 59-211 0.55-0.99). Certified floors are far below (0.17-0.23 on paper supports).
