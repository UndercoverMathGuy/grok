# CAOFF: the global "effective spectrum" form of the conjecture (all networks, mixing neurons included)

2026-09-25. Local CPU only, no commits, CHAIN.md / VERIFY.md untouched. Scripts and logs are in `CAOFF/`.
Setting: n = 2 operands, q = 2, full table, conventions of `chain/common.py`.

## 0. Words

- **gap_D(x)**: on input row x = (a, b), the right answer's logit minus the logit of the answer D steps below it.
  The network's **margin** is the minimum over all rows and all D != 0.
- **Offset class**: D and -D together (h = (p-1)/2 classes).
- **Per-offset worst margin**, three versions. Each is >= the margin, so any of them gives a valid CAoff:
  - m^u_D = min over rows of both gap_D and gap_{-D} (unpaired, the smallest). Note min_D m^u_D = margin exactly.
  - m^s_D = min over rows of (gap_D + gap_{-D})/2 at the same row (X_CHARGE's version).
  - m^f_D = the same with the slot-1 flip pairing (rows shifted by +-D/2).
- **Coverage matrix** A_{Dk} = 1 - cos(2 pi k D/p). A clean spectrum y >= 0 covers offset class D by (Ay)_D.
- **Budgets** (l^{2/3} units: a junk-free block spends exactly sum_k |G_k|^{2/3}):
  - diag = sum_k cost_k / 3^{1/3}, where cost_k = sum_i |e1_i^(k) e2_i^(k) w_i^(k)|^{2/3} counts only the diagonal atoms;
  - P = (p/6) P_2 / 3^{1/3}, from the full weight norm P_2. Budget (B) gives diag <= P.
- **need(M)** = min { sum_k y_k^{2/3} : y >= 0, A y >= M }, with M = (p^3/2) m (the smallest budget of a clean spectrum
  covering the profile).
- **CAoff ratio** = (need / budget)^{3/2}. **CAoff** (for a given version and budget) = "ratio <= 1 for every network".
- **Dom(beta)** = {M : need(M) <= beta}, the margin profiles coverable with budget beta. It is star-shaped but NOT convex.
- **T** = gamma/gamma_sym, the target conjecture (T <= 1).

**CAoff (any version, P budget) implies T.** margin = min_D m^u_D <= min_D m_D <= (2/p^3) min_D (Ay)_D
<= (2/p^3) Gamma* (sum y^{2/3})^{3/2} <= (2/p^3) Gamma* beta^{3/2} = gamma_sym P_2^{3/2}. PROVED.

## 1. Stress test (NUMERICAL; `CAOFF/ascent.py`, `lib.py`; logs `batch1.log`, `batch2.log`, `batch3.log`)

Method: Adam ascent directly on the CAoff ratio of unrestricted networks (every weight free: mixing neurons, DC,
junk). The objective uses soft minima over rows and the exact active vertex of need(M) for the gradient. Every
reported number is an EXACT re-evaluation (`lib.evaluate`: exact minima, full vertex enumeration of need).
Seeds: blocks + random perturbation (5-20%), junky local maxima (claude0923 p = 5 s3, p = 7 s9) + perturbation,
single-frequency blocks, random. Optionally the X share and junk share of the logit tensor are forced above set levels
(penalty), to push the search into the mixing/junk regime.

| version | p | runs | max ratio found | where / what | verdict |
|---|---|---|---|---|---|
| s_diag (same-row pairing, diagonal budget) | 5 | 18 | **1.223** | T < 0, u-margins all negative; pairing averages a negative gap_D with a positive gap_{-D} | **FALSE** |
| f_diag (flip pairing, diagonal budget) | 5 | 18 | **1.213** | same kind of network | **FALSE** |
| u_diag (unpaired, diagonal budget) | 5, 7 | 84 forced + 36 free | **1.108** (p = 7), **1.055** (p = 5) | X share 0.40-0.51, junk share 0.24-0.30, T < 0 (lopsided profile: X lifts some classes) | **FALSE** |
| s_P, f_P (paired, P budget) | 5 | 36 | 0.77 | | not violated |
| **u_P (unpaired, P budget)** | 5, 7, 11, 13 | 36 free + 70 forced (p = 5, 7, 11) | **0.9998** free (approaching 1 only at junk-free nets); **0.936** under forced X/junk (p = 11 max 0.90) | | **no violation** |

Free ascents on u_diag / u_P climb to junk-free networks (ratio -> 1 from below, e.g. single-frequency blocks with
T = 0.75 but ratio 0.9999): CAoff is tight exactly on balanced junk-free networks, as X_CHARGE predicted.
Why the diagonal budget fails: pure-X neurons (slots at different frequencies) have no diagonal atom, so they cost
nothing in `diag`, and their X can lift the worst margin on some offset classes (the violators are lopsided,
T < 0). With the P budget they pay their full weight norm, and no violation appears.
**So the correct global target is CAoff-u_P. Any proof for mixing networks must charge X through the full weight norm
(Parseval), not through diagonal atoms.** This matches and extends X_CHARGE's finding that the per-frequency form is
false.

Single-offset check (`single_offset.py`, `single.log`). Can a network with junk hidden at ONE offset class beat the
best single-frequency coverage there? That would break CAoff while saying nothing about T. Tested with progression
supports near maximal coverage at p = 23, 59 and 1-3 neurons per frequency: best 0.96 (a junk-free block). Lone-atom
populations stay negative at that offset. Junk hidden at one offset loses to the concentration price.

## 2. Structure of CAoff (the "dual form")

**Dom is a union of vertex pieces (PROVED).** need(M) is a concave minimisation over the polyhedron {y >= 0, Ay >= M},
so it is attained at a vertex: M in Dom(beta) iff there are a support S and an equally large set T of tight
offset classes with y_S = A_{TS}^{-1} M_T >= 0, A y >= M on the other classes (slack) and sum y_S^{2/3} <= beta.
- A **membership certificate** is one such (S, T, y).
- A **violation certificate** is the full vertex enumeration (finite; `lib.need`).

The Lagrangian dual is useless: y^{2/3} - c y is unbounded below, so the duality gap is infinite. The convex hull of
Dom is too large to imply T: it contains the average of single-frequency coverages, of height p/(p-1) > Gamma*.
Any certificate must stay non-convex.
Full-support reading: when A^{-1} M >= 0, y = A^{-1} M = -(4/p) C M (C = cosine matrix) is the network's
**effective clean spectrum** read off its offset-margin profile. CAoff with full support means
sum_k (A^{-1} M)_k^{2/3} <= beta.

**p = 5 exactly (PROVED).** Here h = 2, A = [[a, b], [b, a]], a = (5 - sqrt5)/4, b = (5 + sqrt5)/4. Dom(beta) is the region under
the curve C(s) = beta^{3/2} (a s^{3/2} + b t^{3/2}, b s^{3/2} + a t^{3/2}), t = 1 - s. Along it
c1 + c2 = (a + b)(s^{3/2} + t^{3/2}) beta^{3/2} >= 2 Gamma(5) beta^{3/2} (Gamma(5) = (a + b)/(2 sqrt2) = 0.8839), with equality only at
the flat point. Consequence: **Dom(beta) contains {M : M1 + M2 <= 2 Gamma beta^{3/2}, max(M1, M2) <= b beta^{3/2}}.**
Proof: if the smaller coordinate is <= a beta^{3/2}, take s = 0. Otherwise pick s with c2(s) equal to it, and use
c1 >= 2 Gamma beta^{3/2} - c2. At the flat point the boundary is tangent to M1 + M2 = const. So near the block, CAoff at p = 5
is a statement about the class-SUM of the margins, while T is a statement about their minimum.

**The X-free barrier persists (PROVED equivalence at p = 5).** The only X-free information at p = 5 is: m^u_D <= clean_D
per class (uniform rows), and the class-sum under shared flip-line aims (zero-sum identity). Together these give
exactly opus55's X-blind bound, which fails by up to 1.115x on heavy-junk profiles. So **CAoff does not escape case B:
any proof of CAoff or of T must use X-visible test distributions and must bound X through the P budget.**

## 3. Proof attempt: what is proved

**C1 (PROVED). CAoff-u holds, even with the diagonal budget, for every junk-free network (any width, mixing neurons,
DC).** Averaging over rows at a fixed offset keeps only the row-mode-(0,0) components, which are the clean ones (X never
has row mode (0,0)). So m^u_D <= average over rows and over +-D = (2/p^3) sum_k cov_k(D) Re G_k. Take y = |G|. Every product
neuron satisfies the sign rule (S) at every frequency, including mixing neurons (its same-|k| components are
e1^(+-k) e2^(+-k) w^(+-k)). So the table's (JF) plus Theorem 6.1 give sum |G_k|^{2/3} <= sum_k cost_k / 3^{1/3}.

**C2 (PROVED). CAoff-u (diagonal budget) holds for single-frequency networks wherever the chain's heavy side holds**
(aim depth >= 0.272 on the junky set: p = 5, junk on <= 2 frequencies, certified supports). Per-offset aims on the same
flip-enlarged body give m^u_D <= (2/p^3) sum_k cov_k(D) y~_k with the same y~_k = (|G_k| - 0.272 eta_k)_+ at every offset.
Take y = y~; the eta-form cost side gives the budget.
So CAoff is exactly the natural output of the existing chain. It is the right global replacement for the false
per-frequency form (X_CHARGE).

**C3 (PROVED). Junk-visibility conservation — X can move junk between offsets but not remove it.** Fix a junk row mode,
e.g. (-2k, 0) (pattern {1} of frequency k). The row-mode coefficient of gap_D is p^{-3} Phi(D), with
Phi(D) = sum_{g != 0} Zhat(-2k - g, -g, g)(1 - w^{-gD}). The junk contributes g = -k, i.e. J_{k,1}; every other g is X.
The offset profiles f_g(D) = 1 - w^{-gD} (g = 1..p-1) have Gram matrix exactly p(I + 11^T) on the nonzero offsets.
Expanding the point mass at D* gives coefficients -w^{g D*}/p. Hence, for EVERY network:
- (i) **J_{k,1} = -(1/p) sum_{D != 0} Phi(D) w^{-kD}** (this is the X-blind functional, i.e. opus55's half-offset line);
- (ii) **sum_{D != 0} |Phi(D)| >= p |J_{k,1}|**, against 2 cot(pi/(2p)) |J| ~ (4/pi) p |J| with no X. So X can cut the l1
  visibility of a junk component over offsets by at most ~21%;
- (iii) **min over X of sum_D |Phi(D)|^2 = p^2/(p-1) |J|^2**, against 2p |J|^2 with no X: at most ~50% in l2;
- (iv) hiding the junk on all offsets but one forces a spike |Phi(D*)| = p |J| there.
The same holds for every junk row mode (patterns {2}, {1,2}). Checks (`conservation.py`, `cons_debug.py`): (i) exact to 1e-13
on random mixing nets at p = 7 (the earlier 0.22 was a relative error on a J of size 1e-13); (ii) observed ratio >= 1.21 on
120 random cases; (iii) the Gram identity and the minimum match p^2/(p-1) exactly at p = 5, 7, 11, 13.
**Reading.** X_CHARGE's "X helps only at some offsets" is now quantitative. X reshapes junk visibility across offsets
at conserved total (up to ~21% in l1), and concentrating it amplifies it. At a flat-coverage optimum (p = 5) there is no
slack offset to dump visibility on, which is the mechanism behind "X cannot help on average".

## 4. What is left (plain words)

The global statement worth proving is **CAoff-u_P**: every network's per-offset worst margins can be covered by a clean
spectrum paid for by the full weight norm. It implies the conjecture, is tight exactly on junk-free networks, holds for
junk-free and for chain-covered single-frequency networks (C1, C2), and survived every adversarial search here. The
diagonal-budget and paired versions are false.

A proof program that uses the new conservation law (C3), in four steps (none done):
1. per-offset exposure >= aim depth x (junk visibility at that offset, summed over junk row modes);
2. conservation: total visibility over offsets >= p x junk size;
3. a coverage-slack programme: the Gamma*-type problem where the adversary may place visibility on slack offsets;
4. a charge for the X that reshapes visibility, through the full weight norm (e.g. the table's Fourier l2 norm is at
   most (p P_2/3)^{3/2}).
Two known obstructions:
- *collision leaks*: X at non-junk row modes that fall in the aim's Fourier spectrum;
- step 3 alone is not enough: at sparse optima there is coverage slack to absorb small junk, so step 4 (X's cost) is
  essential.
At p = 5 the coverage of the optimum is flat (no slack offset), which is why case A covers everything near the block. Case B
(heavy junk) still needs X-visible bounds, as in X_CHARGE.
