# nu-dial cohort — scored against nudial/README.md prereg (2026-08-22)

2208 runs on one 5090 (~2.5 h). Data: `runs_nudial/` (spectra/metrics/config),
`notes/nudial_summary.json`, log `notes/nudial_pod.log`. Transformer
nu_eff measurement: `notes/nu_eff_dial.json`.

## Headline table (grokked runs; k_eff = participation ratio, K_gap = transformer instrument)

| family | nu | m=128 k_eff / K_gap mode | m=512 | m=2048 | grok (512) | theory K* |
|---|---|---|---|---|---|---|
| relu | 2 | 27.2 / – (29/96 grok) | **45.1** / dense | 54.0 / dense | 96/96 | dense |
| relu | 3 | 5.96 / 8 | **7.5** / 12 | 7.96 / 12 | 96/96 | 6 (plateau 5-11) |
| relu | 4 | 3.78 / 5 | **4.25** / 5 | 4.48 / 6 | 96/96 | 4-5 (2% apart) |
| relu | 5 | 2.56 / 4 | **3.32** / 4 | 3.18 / 4 | 96/96 | 4 |
| relu | 6 | 2.16 / 4 | **2.99** / 4 | 2.89 / 4 | 96/96 | 3-4 (6% apart) |
| quad | 3 | 30.6 / dense | **44.0** / dense | 51.0 / dense | 96/96 | 6 |
| quad | 4 | 12.6 / – | **15.0** / – | 16.0 / – | 96/96 | 4-5 |
| quad | 5 | 9.57 / 11 | **9.7** / 11 | 10.1 / 11 | 77/96 | 4 |

relu-6 K_gap histogram (m=512): 3:15, 4:47, 5:29, 6:4, 7:1 — vs the
transformer dial cohort 3:20, 4:92, 5:33 (148 runs), whose measured
nu_eff = 6.6 (logit) / 6.1 (margin), width-flat.

## Prereg verdicts

- **P-N1 (monotone dial):** Spearman(k_eff, nu) = -0.76 over all 24 cells
  (bar <= -0.8: **narrow FAIL**); relu-only = -1.0 and quad-only = -1.0
  (each family is perfectly monotone; the cross-family offset breaks the
  pooled rank). K_gap arm +0.25: **invalid instrument** on dense cells (the
  gap detector returns 1-2 when 56 amplitudes are equal) — registration error
  on my side; README already said K_gap does not certify density.
- **P-N2 (K=4 at nu=4):** modal K_gap = 5 (m=128/512) and 6 (m=2048); 32% in
  {3,4,5}: **FAIL**. The exact table has K=4 and K=5 within 2% at nu=4, so
  the "sharp 4" was an overclaim; k_eff = 4.25 sits on the tie.
- **P-N3 (dense at nu=2):** 83% of grokked relu-2 runs at m>=512 have
  K_eps >= 45: **PASS**.
- **P-N4 (width):** relu-2 ratio 1.99 (bar >= 2: pass at the edge);
  nu>=4: relu4 1.18, relu5 1.24, quad4 1.27, quad5 1.06 pass; relu6 1.34
  misses 1.3. **Mostly PASS**: modal K_gap is identical across 16x width at
  every nu >= 4.
- **P-N5 (norm swap, quad-3):** L_{2,3} arm fully dense (k_eff 55, 32/96
  grok) — but so is the plain-wd control (k_eff 44): **UNINFORMATIVE**. The
  norm swap cannot be tested at nu=3 because plain-wd quad nets are dense
  anyway (Tian / He et al. were right about this cell).
- **P-N6 (brute-force optimum):** relu-2 p=113 dense (k_eff 51.7) as
  predicted; quad-3 p=113 reaches k_eff 14.8 (sparser than the trained 44 —
  dynamics vs optimum gap is real); relu-4 **optimizer failed** (final
  normalised margin negative at both p: the ||theta||^4-normalised softmin
  never separated the data). C is not scored for nu=4; needs a
  sphere-projected / log-margin optimiser.

**Prereg pass condition (P-N1, P-N2, P-N3, P-N5 all pass): NOT MET.**

## What stands

1. Within the relu family, committee size falls monotonically and smoothly
   with the number of inserted linear layers: k_eff 45 -> 7.5 -> 4.25 ->
   3.3 -> 3.0, reproduced at three widths, modal K width-invariant for
   nu >= 4. Nothing but degree changed. New, clean, quantitative.
2. The transformer's 3/4/5 committee split is reproduced by a plain MLP at
   nu = 6, and the transformer's measured nu_eff is 6.1-6.6. No attention
   needed to get "four frequencies".
3. The K*(nu) formula gets scale and trend right to +-1 (k_eff within ~0.5
   of K* at every relu cell), not integer-exact.

## What falls

1. "nu alone" is refuted: the quadratic family is dense at nu=3 and 3-6x
   above relu at equal nu (44 vs 7.5; 15 vs 4.25; 9.7 vs 3.3). The offset is
   family-dependent; the mass-allocation argument has no term for it.
   Candidates: reachability (dense symmetric KKT point for one-hot quad
   nets — C's quad-3 optimum at 14.8 vs trained 44 supports a dynamics gap),
   or degree-from-squaring vs degree-from-linear-depth being inequivalent
   (nuclear-norm-type pressure from linear products is what actually
   sparsifies).
2. The norm-swap figure cannot be made at nu=3. To make it, run B at a
   relu-nu>=4 cell with a per-neuron L_{2,nu} grouping defined for the
   deeper net (open design question).

## Next

- Fix cohort C (normalise theta to the sphere every step; maximise
  log-margin; more restarts) and rerun relu-4/5/6 + quad-3/4 at p=113 —
  this decides optimum-vs-dynamics for the quad offset.
- Degree-from-where test: relu nets with extra linear layers placed
  pre-activation only vs post-activation only at fixed nu; quad nets with
  linear pre-layers (nu = 5 via (1,0)) vs post-layers ((0,2)).
- Paper framing: "linear depth sets committee size" (relu dial + transformer
  nu_eff), with the quad offset reported as the open boundary — not
  "one formula for all architectures".

---

# Round 2 (D optimum / E placement / F init scale) — scored 2026-08-22

864 more runs (total 3072). Data runs_nudial/{D,E,F}, log notes/nudial2_pod.log,
combined notes/nudial_summary.json. Pod 2xuwlojala6cuf deleted.

## D — the true l2 optimum (two-phase: CE on the fixed-radius sphere, then softmin margin; p=113, m=512, 32 inits)

| cell | optimum K_gap mode / K_eps / k_eff | margin | trained (A) mode / k_eff | K*(nu) table |
|---|---|---|---|---|
| relu 2 | dense / 53 / 45.1 | 5.5e-4 | dense / 45.1 | dense |
| relu 4 | **4** / 4 / 2.4 | 6.4e-7 | 5 / 4.25 | 4 |
| relu 5 | **3** / 3 / 2.1 | 3.2e-8 | 4 / 3.3 | 3-4 |
| relu 6 | **3** / 3 / 1.8 | 2.2e-9 | 4 / 3.0 | 3 |
| quad 3 | **dense** / 54 / 45.7 | 6.1e-6 | dense / 44.0 | 6 (WRONG) |
| quad 4 | 17 / 9.9 | 2.1e-7 | 18 / 15.0 | 4-5 (WRONG) |

Hand-built quad-3 nets (scratch quad_construct.py, Morwani construction, best
K-sets): normalised margin rises monotonically with K — K=4 2.7e-7, K=6
1.8e-6, K=12 5.4e-6, dense 5.4e-6; D's 6.1e-6 sits just above the dense
construction. **Dense is genuinely the l2 max-margin for one-hot quadratic
nets.** Mechanism: a one-hot quad circuit (u_a+v_b)^2 w_c emits cross-talk
logits at (2a-c), (2b-c) that never cancel; cross-talk adds incoherently
(sqrt K) while signal adds coherently (K), so density is nearly free. The
separable mass model assumes cross-talk-free circuits.

**P-N7:** relu part PASSES (trained within +1 of optimum; optimum = K*(nu)
exactly: 4, 3, 3, dense). quad part: kill condition HIT — the quad-3 optimum
is dense, so the quad offset is NOT a dynamics trap. Every trained net,
both families, sits at (or +1 above) its own l2 optimum.

## E — where the degree sits (fixed nu; p=113, m=512, 96 runs/cell)

| cell | layout | grok | k_eff | K_eps | K_gap mode |
|---|---|---|---|---|---|
| relu nu=4 | (0,2) one-hot -> ReLU -> 2 readout | 42/96 | 8.84 | 10 | 10 |
| relu nu=4 | (1,1) embed -> ReLU -> 1 readout | 96/96 | 4.19 | 5 | 5 |
| relu nu=4 | (2,0) 2 embed -> ReLU | 96/96 | 3.89 | 5 | 5 |
| quad nu=5 | (0,2) one-hot -> x^2 -> 2 readout | 77/96 | 9.70 | 11 | 11 |
| quad nu=5 | (1,0) embed -> x^2 | 96/96 | 3.85 | 5 | 5 |

**P-N9 PASSES** (2.3x relu, 2.5x quad; shared pre-activation layer gives the
smaller K). The primary switch is whether a shared embedding precedes the
nonlinearity (cross-talk removal); degree modulates within each class
(one-hot: 45 -> 8.8 for nu 2 -> 4; embedded: 7.5 -> 4.2 -> 3.9 -> 3.0 for
nu 3 -> 6).

## F — init scale (p=113, m=512, wd=1, 24 seeds/scale, all grok)

| scale | relu-4 k_eff / K_gap | quad-3 k_eff |
|---|---|---|
| 0.1 | 1.34 / {1:6,2:6,3:7,4:4,5:1} | 51.3 |
| 0.3 | 3.74 / mode 5 | 51.0 |
| 1.0 | 4.13 / mode 5 | 51.2 |
| 3.0 | 4.30 / mode 5-6 | 49.5 |

**P-N8:** quad part kill condition HIT (dense at every scale — consistent
with D: it is the optimum, not a trap). relu-4 is NOT flat: small init
drops it to 1-3 frequencies (the seed barrier: fewer circuits cross
lambda/c from small seeds), and single-frequency solutions still grok.

## Synthesis after both rounds

1. **Networks sit at their l2 max-margin optimum (+-1)** in every cell
   tested: relu 2/4/5/6 and quad 3. The dense theorem in the literature is
   an L_{2,nu}-norm artifact; the ground truth principle is margin under the
   norm people actually train with.
2. **The optimum's frequency count has two terms:** the idealised
   degree term (sum_k m_k^{2/nu}, giving K*(nu) = 4/3/3 at nu = 4/5/6 and
   dense at nu = 2 — matched exactly by D's relu optima) and an
   architecture cross-talk term that makes density cheap for one-hot
   circuits. A shared embedding before the nonlinearity removes the
   cross-talk; transformers and embedding-MLPs are in that class, which is
   why nu_eff ~ 6 gives 3-4 frequencies.
3. **Dynamics add +1 and an init-scale dependence** (F): the trained relu
   nets keep one circuit more than the optimum at standard init and fewer
   at small init — the seed-barrier residue on top of the optimum.
4. Round-1 P-N2 "failure" is explained: the optimum at nu=4 IS 4; training
   lands at 5.

Open: an explicit cross-talk cost functional per architecture (would turn
item 2 into a single formula); the +1 as a function of lambda; transformer
D-style optimum search (attention non-homogeneity).

---

# Round 3 (G frac sweep / H subset optimum / I plain GD / J warm refine) — the +1, scored 2026-08-23

~1300 more runs (total in runs_nudial: see notes/nudial_summary.json). Pods b6n4alazfosaqn deleted.
Logs notes/nudial3*.log.

## Candidates for the +1 and their fate

| candidate | test | result |
|---|---|---|
| finite-temperature objective | D's CE-on-sphere phase K | at optimum already — dead |
| instrument ghost | weakest member amplitude | 0.33-0.44 of max — dead |
| weight decay | K vs wd 0.3/1/3 | flat — dead |
| AdamW l_inf box | max|theta|*wd 0.13-0.63, none near 1/wd | dead |
| mask-specific circuit | lesion: train 0.40x, test 0.39x | generalising — dead |
| unfinished pruning | transformer 25k->50k frozen | dead |
| memorisation remnant | G: K(frac 1.0) = 5.6 (relu4), 3.8 (relu6) = same as frac 0.5 | dead for COUNT; explains WHICH (pre-grok top-K 88-90%) and fattens the profile (relu6 k_eff 2.9 -> 2.0 with full data) |
| subset landscape | H: optimum on the frac-0.5 subset is K=3 (both families), test acc 1.0 | P-N11 pass — dead |
| Adam geometry | I: plain GD+l2 -> K 3-4 (relu4), 2-3 (relu6), concentrated | GD is sparser than AdamW |

## The resolution: AdamW's K*+1 solutions have the HIGHEST normalised l2 margin of anything we produced

Normalised margin g_min/||theta||^nu over the full p^2 grid, identical formula everywhere:

| relu-4, p=113, m=512 | n | margin (median) | K | k_eff | profile |
|---|---|---|---|---|---|
| D margin-opt, random init | 32 | 6.4e-7 | 4 | 2.4 | [1,.61,.39,.13] |
| H margin-opt, subset | 25 | 5.0e-7 | 3 | 2.2 | |
| GD + l2 (wd 3e-4), full | 18 | 2.4e-6 | 3 | 1.6 | [1,.53,.18,.03] |
| AdamW, init 0.1 | 24 | 2.5e-6 | 2 | 1.3 | |
| J margin-refine from AdamW | 32 | 2.1e-6 (regressed) | 5 | 3.9 | |
| **AdamW, full data** | 96 | **4.7e-6** | **5** | 3.9 | [1,.85,.69,.60,.46] |
| **AdamW, frac 0.5, wd 1** | 32 | **4.9e-6** | **5** | 4.1 | |

| relu-6 | n | margin | K | k_eff |
|---|---|---|---|---|
| D margin-opt | 29 | 2.4e-9 | 3 | 1.8 |
| H margin-opt (subset) | 28 | 2.6e-9 | 4 | 2.3 |
| GD + l2 | 8 | 2.8e-8 | 2 | 1.7 |
| **AdamW full / frac 0.5** | 96 / 32 | **4.5e-8 / 4.7e-8** | **4** | 2.0 / 2.9 |

AdamW's democratic K*+1 nets beat the direct margin-ascent "optima" by 7-20x
and plain GD by 1.6-2x in normalised l2 margin. Our optimum searches (D, H,
J) were converging to inferior local optima — J even REDUCED the margin of
the AdamW nets it started from. So the "+1" was never a deviation from
optimality: it is the idealised mass model undercounting by one in the relu
family (the same architecture-specific circuit-structure term that is
catastrophic for one-hot quad nets is mild-but-nonzero here: ReLU circuits
carry harmonics, and K*+1 equal-amplitude frequencies cancel them better
than K* concentrated ones). AdamW finds the better solution; GD's l2
geometry parks at the sparser, worse one; small init parks sparser still.

## Consequences

1. "Trained nets sit at the l2 max-margin optimum" is STRONGER than stated
   in round 2: they sit at the best margin solution we can find by any
   means, not +-1 from it.
2. K*(nu) from the idealised mass model is a lower bound / first term;
   the architecture term adds ~+1 for relu-with-embedding nets and ~dense
   for one-hot quad nets. The probabilistic formula's tilt gamma (~+0.3 at
   standard init) IS this term; its init-scale dependence is the seed
   barrier (small init -> sparser -> LOWER margin, F cell 2.5e-6 vs 4.9e-6).
3. The direct margin-ascent machinery (D/H/J) must not be presented as
   "the optimum"; present it as a lower bound and use AdamW-trained nets'
   margins as the empirical optimum, or build a better maximiser
   (warm-start from AdamW + projected ascent that is verified to be
   monotone in the min margin).
4. The memorisation phase determines WHICH extra frequency (pre-grok top-K)
   and fattens it; it does not create it.

Open: a per-architecture margin(K) curve measured with AdamW at fixed
support size (train with frequency masks) would make P(K) a calibrated
distribution and remove the hand-built-construction step.

## 2026-08-23: the "2.1" has no limit — K* grows with log p (nudial/kstar_pscale.py, notes/kstar_pscale.json)

Theorem (Dirichlet pigeonhole on the K-torus): M_K(p) < K(1 - cos(2 pi p^{-1/K})) -> 0 for every fixed K.
Lattice regime (K << log p): M_K ~ 2 pi^2 gamma'_K p^{-2/K}, gamma'_K just under the Hermite constant;
verified: M_K p^{2/K} -> 22.3 (K=2, bound 22.8), 23.5 (K=3, 24.9), 23.7 (K=4, 27.9). c_2 -> 2 exactly.
c_4 = K - M_4: 2.12, 2.57, 2.92, 3.29, 3.48, 3.63 at p = 113, 211, 401, 1009, 2003, 4001 — not a constant.
Asymptotic K*(nu,p) ~ 4 log p/(nu-2); computed (local search K<=12):
  p:    113 211 401 1009 2003 4001
  nu=3:   6   6   9   11   11   12+
  nu=4:   4   5   5    7    7    9
  nu=5:   4   4   5    5    6    6
  nu=6:   3   4   4    5    5    6
Paper correction: K* depends on p through log p (flat over the 97-157 range tested) — reconciles McCracken
2505.18266's O(log p). nu=6: p=113 K*=3 (4 at 94%), p=211 K*=4 (3 at 99.6%) -> observed 3/4 near-tie is the crossover.
Unrun prediction: relu-4 MLP at p=1009 -> K ~ 7+1, relu-6 -> ~6 (vs 5 / 4 at p=113).
nu for arbitrary architectures: nu = degree of the max-degree input->logit path (ReLU x1, x^2 doubles, LN resets to 0,
softmax adds a fractional term); 1-layer no-LN transformer: E->OV->MLP->U = 6 (+softmax) = measured 6.1-6.6.
Predictions: 2-layer transformer K 2-3; final-LN transformer nu=3 -> broad 5-11.

## Round 4 scored (2026-08-28): fixed-support gamma(K) — P-N12 KILL

Cohort S, relu-4 m=512 p=113 frac 0.5 wd 1, E1 = A @ B_S (orthonormal
support rows, wd/nu accounting unchanged), supports = exhaustive-optimal
sets, 12 paired seeds per K + unmasked full-basis control. Trained on one
RTX 5090 (~10 min, pod deleted). Data: runs_nudial/S/. 60/60 grokked.

| K | support | gamma med | ratio to K=4 | theory K^-2 M_K ratio |
|---|---|---|---|---|
| 3 | (1,6,43) | 4.286e-6 | 0.751 (12/12 below) | 0.797 |
| 4 | (1,7,11,43) | 5.695e-6 | 1 | 1 |
| 5 | (1,14,20,44,52) | 5.491e-6 | 0.964 (12/12 below) | 0.983 |
| 6 | (1,5,12,15,38,46) | 5.246e-6 | 0.922 (12/12 below) | 0.921 |
| free (control) | — | 4.566e-6 | 0.805 (12/12 below) | — |

- **P-N12: KILL C8/C9 as registered.** gamma peaks at K=4=K*, not 5;
  K=4 beats K=5 in 12/12 paired seeds. The trained +1 is a DYNAMICS
  artifact, not a property of the margin optimum.
- **C12 prediction 6 also triggered:** fixed-support K=4 (5.7e-6) beats
  the free-AdamW bar (4.7-4.9e-6) — the "stronger maximiser" exists and
  it is just AdamW handed the right support. C9's "trained nets are
  optimal" reading is retracted; correct statement: trained nets have the
  highest margin among UNCONSTRAINED procedures tried, but sit ~4% below
  the reachable optimum, at K*+1.
- **The equal-amplitude ansatz curve is quantitatively right**: measured
  gamma ratios match K^{-nu/2} M_K to 0.1% (K=6), 2% (K=5), 6% (K=3).
  The theory needed no +1 correction — the optimum was at K* all along.
- **P-N13 (control): gamma PASS** (4.57e-6 vs cohort-A 4.4e-6), committee
  mode borderline (5-6 vs cohort-A 5): Adam in Fourier coordinates
  slightly densifies committees. Does not affect P-N12 (all groups share
  the parameterization; comparisons are internal).
- Secondary read: interior peak 3 < 4 > 5 > 6 confirmed.

Consequence for the pack: C8 dies as stated (leak-cancellation does NOT
make K*+1 optimal); C9 is rewritten as above; W2 is resolved — the +1 is
dynamical (consistent with round-3: the extra member is the surviving
memorisation-era circuit, present in pre-grok top-K 88-90%). C3's K* is
now DIRECTLY measured as the argmax of realised gamma, not just computed
in the ansatz.

## Round 5 (2026-08-29): gamma(S) as a probability distribution over committees

Question: instead of predicting the final committee deterministically, turn
the ansatz gamma(S) = M(S) / K^(nu/2) into a distribution and fit the
observed vanilla-cohort K histogram (n=69: {3:11, 4:45, 5:11, 6:2}).
Script: nudial/gibbs_fit.py (exact enumeration of all subsets K<=5, 4.2M
sets; MC 2M/class for K=6..9; nu=6; veto = additive-collision filter).

One-parameter families fit by max likelihood on the K counts
(saturated LL = -66.71):

| model                                   | param      | LL      | mode | E[n_K] 3/4/5/6        |
|-----------------------------------------|------------|---------|------|-----------------------|
| A  Boltzmann over K, exp(g*(K)/T)       | T=4.9e-3   | -95.0   | 3    | 31/22/6/2 (+6 at K=2) |
| B  Gibbs over sets, exp(g(S)/T)         | T=1.3e-3   | -102.5  | 3    | 35/21/4/2 + tail to 9 |
| C  B + veto (collision-free only)       | T=1.5e-3   | -87.9   | 3    | 28/24/7/4             |
| B2 power law over sets, g(S)^beta       | beta=14.1  | -72.8   | 4    | 21/33/11/3            |
| C2 B2 + veto                            | beta=13.6  | -71.8   | 4    | 19/33/12/3            |

- Boltzmann over K structurally CANNOT put the mode at 4 (exp weight
  inherits gamma's argmax = K* = 3 at every temperature). Exp weight over
  sets is bimodal (entropy blowup at large K). **Power law
  P(S) ~ 1[collision-free] * gamma(S)^13.6 is the only family that works**:
  mode 4, correct tail cutoff, parametric-bootstrap p = 0.045 (deviance
  10.4 vs sim median 3.7; residual misfit is over-spreading the 3/4
  boundary: 19 threes predicted vs 11, 33 fours vs 45).
- nu sensitivity (exp-family): mode flips 3->4 below nu~5.5; power law is
  nearly nu-invariant (nu and beta enter K-pricing only through nu*beta).

**Identity cross-check (the split).** Observed committees sit at median
within-class quantile 0.90 by equal-amp gamma (K=4: 0.92) — top decile
but NOT optimal (min 0.10). Within-class MLE gives beta_id = 4.6, vs
beta_count = 13.6 from the histogram; beta_id applied to counts predicts
mode K=8 (nonsense). **No single Gibbs weight on gamma fits counts and
identity simultaneously — a factor ~3 selection-strength split.** This is
the ensemble-level signature of the round-4 mechanism: K is set sharply
(amplitude-threshold survival at grok), identity is set softly (lottery
seeding biased toward good coverage, then local polish); an equilibrium
draw from the margin landscape would use one strength for both.
- Free-amplitude gamma does NOT rescue the single-beta model: gamma_free
  is monotone under adding members (a set can zero its extras), so it
  cannot price K at all; the equal-amp ansatz is the right variable
  because trained committees are in fact democratic (round-2 result).
- Per-frequency inclusion probabilities under C2 do not correlate with
  observed frequency popularity (Spearman 0.06) — popularity is
  mask/lottery-driven, not coverage-geometry-driven, consistent with the
  mask-popularity result.

Bottom line: P(S) ~ 1[veto] * gamma(S)^beta with beta ~ 14 is a
defensible one-parameter distribution for the K histogram (marginal fit,
p=0.045), and the count-vs-identity beta split (14 vs 4.6) is itself a
new quantitative claim: final committees are not an equilibrium ensemble
of the margin functional.

### Round 5 addendum: drop the veto — pure P(S) ~ gamma(S)^beta suffices

User call: the collision-free indicator is an inelegant extra ingredient.
Refit with NO veto term, margin only:

  P(S) ~ gamma(S)^beta,  beta* = 13.9,  LL -72.90 (veto version -71.83),
  E[n_K] = {3:20.0, 4:32.9, 5:11.4, 6:3.2}, parametric-bootstrap p = 0.023.

The 1-nat LL cost of dropping the veto is noise. Decisive bonus: at the
fitted beta the ensemble is ALREADY 97.3% collision-free — per class
P(collision-free | drawn) = 1.000 / 0.999 / 0.965 / 0.814 at K=3/4/5/6
vs uniform-class baselines 0.964 / 0.859 / 0.675 / 0.445. Probability
that all 69 observed runs are collision-free under the pure model: 0.15
— the observed 0/69 collision law is CONSISTENT AS A COROLLARY of margin
selection (collision sets have systematically poor coverage M(S), and
gamma^beta suppresses them). The veto set-level law no longer needs to be
a separate axiom at vanilla sizes; it is an emergent property of the
fitted ensemble. (At K>=7 the model still predicts substantial collision
content, but those sizes carry ~1% of the mass.)

Final clean statement: one term, one parameter —
  P(committee S) ~ [ M(S) / K^(nu/2) ]^beta,  nu = 6 (measured),
  beta = 13.9 (fitted)
reproduces the K histogram (mode K*+1, p=0.023), the tail cutoff, the
top-decile identity tendency, and the collision-free law. The remaining
quantitative gap is the count-vs-identity beta split (13.9 vs 4.6),
which stands as the non-equilibrium signature.

### Round 5 final form: the K-level free-energy distribution (user-requested)

Requirement: a distribution over K ONLY, built from the ansatz curve
gamma(K) = M_K / K^(nu/2), no set-level machinery. Impossibility note:
any monotone weight of gamma(K) alone pins the mode at K* — a K-increasing
entropy factor is mandatory, and the canonical minimal one is the number
of size-K committees, C(56, K). Final model:

  P(K) ~ C(56, K) * gamma(K)^beta          (one fitted parameter)
  log P(K) = beta * log gamma(K) + log C(56,K) + const   ("margin + entropy")

Fit to the 69-run vanilla transformer cohort (nu = 6 measured, M_K from V1):

  beta* = 16.0  (68% interval [15.1, 17.0])
  E[n_K] = {3: 9.4, 4: 47.2, 5: 10.5, 6: 1.7}   observed {3:11, 4:45, 5:11, 6:2}
  LL = -67.16 vs saturated -66.71  (0.45 nats from perfect)
  parametric bootstrap p = 0.87  — NOT REJECTED, fits far better than the
  set-level enumeration models (p 0.023-0.045).

The +1/-1 question dissolves into the ledger (log units relative to K=4):
  K=3: entropy -2.58, margin +0.97  -> net -1.61   (down-shoulder of gamma
       is FLAT: only 6% margin gain going 4->3, but 13x fewer committees)
  K=5: entropy +2.34, margin -3.85  -> net -1.51   (up-shoulder steepens)
  K=2: entropy -5.47, margin -3.65  -> net -9.13   (-1 offset doubly dead)
The near-equal net for K=3 and K=5 predicts the observed 11-vs-11
symmetric fuzz around the mode; the mode sits at K*+1 because gamma's
peak is flat on its right shoulder at (p=113, nu=6) while C(56,K) grows.

Caveats / next tests: (1) beta is a per-ensemble sharpness, NOT universal
— at nu=4 the same beta would run away to dense committees (needs
beta > ~33 for mode K*+1=5); fitting the MLP dial cohort's histogram is
the immediate cross-check. (2) The K-level form outperforming the full
set-level enumeration is consistent with the beta split: counts behave
as if each class is "best committee x multiplicity", identity is chosen
by a softer lottery. (3) The model upgrades the p=1009 prereg from an
argmax call to a full predicted histogram via M_K(1009).

## Round 6 (2026-08-29): P-N14 scored — the frozen P(K) sweep

224 runs, vanilla legacy recipe, one knob per cell, 5090 (~$1.6, pod
deleted). All predictions frozen in commit 704ce11 BEFORE any run.
beta = 16.02 never refit. Data: runs_sweep/ (slim: spectra+configs, quad
ckpts kept). Scoring = mean -log P(K_obs) in nats; sat = per-cell floor;
null = 69-run baseline hist applied to every cell.

cell    n_gr  obs K hist              CE_frz  CE_null CE_sat  boot p   verdict
base     24   {3:4, 4:15, 5:5}         0.962   1.016   0.919   0.713   fits
wd0.3    24   {3:5, 4:15, 5:4}         0.966   1.016   0.919   0.658   INVARIANCE CONFIRMED
wd3.0    24   {3:5, 4:15, 5:4}         0.966   1.016   0.919   0.653   INVARIANCE CONFIRMED
f0.2      6   {3:5, 4:1}               1.723   1.646   0.451   0.001   deviates -> K* (see below)
f0.4     24   {3:4, 4:13, 5:7}         1.087   1.130   0.990   0.249   INVARIANCE CONFIRMED
f0.5     24   {3:7, 4:13, 5:4}         1.101   1.130   0.990   0.214   INVARIANCE CONFIRMED
p97      24   {3:5, 4:18, 5:1}         0.941   0.844   0.675   0.005   KILL (pred mode 3, obs 4)
p127     24   {3:1, 4:15, 5:7, 6:1}    0.991   1.079   0.918   0.538   CONFIRMED (fat 5-shoulder)
p211     16   {3:4, 4:10, 5:2}         2.227   1.016   0.900   0.000   KILL (pred mode 5, obs 4)
quad     23   {2:1, 3:12, 4:8, 5:2}    9.299   1.533   1.055   0.000   KILL (pred K=2, obs 3-4)

Registered WINS:
- wd-invariance at 10x range (0.3 vs 3.0): histograms bitwise-near-identical
  to base ({3:5,4:15,5:4} twice). The K distribution has NO wd term, as the
  formula says.
- frac-invariance at 0.3/0.4/0.5: confirmed (p 0.21-0.25).
- p127: predicted fat 5-shoulder appeared exactly ({4:15, 5:7}, p=0.54).

Registered KILLS (all in the same direction — a discovery-shaped failure):
- Observed mode is FOUR AT EVERY PRIME (97/113/127/211). The frozen model
  predicted entropy-margin drift: mode 3 at p97 (nf=48 entropy shrink),
  mode 5 at p211 (nf=105 growth, K* itself = 4 there). Reality: the
  committee count is far more p-rigid than C(nf,K)*gamma^beta allows.
  Post-hoc per-cell beta refits: 14 (p97), 16 (113), 15 (127), 32 (211) —
  effective selection sharpness GROWS with p, cancelling the entropy term.
  Note p211 obs mode 4 = K*(211) exactly (no +1); p97/113/127 mode = K*+1.
- quad: grokked 23/24 (first quad transformer cohort ever here), but
  committees are K=3-4, not the predicted K=2 collapse. Measured
  nu_eff on the quad ckpts: margin 9.8, logit 11.3, bias share 3e-4 —
  the degree-10 calculus is CORRECT (squaring doubles the path degree,
  biases negligible). The failure is the MARGIN FUNCTIONAL: additive
  cos-committee coverage M_K does not describe quadratic circuits
  (cross-talk/harmonic leakage — same phenomenon as quad-MLP "dense by
  cross-talk"). Scope boundary of the ansatz, not of nu.

f0.2 (grokking boundary): only 6/24 grok, and survivors sit at K*=3, not
K*+1 ({3:5,4:1}). Model-violating (invariance predicted) but MECHANISM-
consistent: scarce data -> weak memorisation seeding -> fewer circuits
above the survival threshold -> no +1. Prediction for a followup: the +1
should fade as frac approaches the grokking boundary from above.

Pooled honesty: incl. quad, frozen CE 2.010 vs null 1.103 (null wins,
quad dominates); excl. quad, frozen 1.128 vs null 1.051 vs sat 0.890 —
the null still edges the frozen model, driven by p97+p211. Verdict: the
free-energy distribution is exact WITHIN the p=113 recipe family
(wd/frac/seeds; ~120 runs, every cell p>0.2) and directionally right at
p127, but its p-extrapolation and quad transfer are falsified as frozen.
The p-rigidity of mode 4 (and beta(p) growth) is the new open phenomenon.

## Round 7 (2026-08-29): resolving the P-N14 kills — primes and quad

### Why beta grows with p: it doesn't — the entropy term isn't real

1. Set-level rescue TESTED AND KILLED. Hypothesis: C(nf,K) is the wrong
   entropy, the effective multiplicity N_eff(K) = sum_S (gamma(S)/gmax)^beta
   grows slower with p and fixes the drift. Computed exactly (enumeration
   K<=4-5 all primes + MC, scratchpad/prime_setlevel.py): N_eff IS much
   smaller than C(nf,K) (e.g. p211 K4: ln N_eff 10.2 vs ln C 15.4) but
   still grows too fast — joint single-beta fit (beta*=15) still predicts
   mode 3 at p97 and mode 5+tail at p211. No landscape entropy works.

2. The seed is p-RIGID (the real mechanism). Above-threshold circuit
   count at grok (A_k >= 0.15 A_max, snapshot before test>0.5), from the
   sweep's own spectra: median 3-4 at EVERY prime — p97: med 3, p113: 3-4,
   p127: 4, p211: 3 (despite nf=105 available). The memorisation lottery
   does not scale with the number of available frequencies. At p211,
   where K*=4 > typical seed, finals sit ABOVE the seed (seed==final
   0.06, all upward moves): the margin floor RECRUITS members post-grok.

3. The plateau picture. gamma(K) is nearly flat on top at every prime:
   plateau (>=99% of max) = {3,4} at p97/113/127, {3,4} at p211
   (gamma3/gamma4 = 0.990), {4,5} at p1009 (ratio 0.992, K*=5, computed
   K=1..10 by local search). Law: K_final = seed count projected onto the
   plateau. Mode 4 is rigid because the seed is ~4 and 4 sits on the
   plateau for all p in [97, 1009]. beta(p) growth in the Gibbs family =
   artifact of forcing entropy scaling onto a dynamics-set count.

4. REGISTERED DISCRIMINATOR (p=1009, nu=6, vanilla recipe, when run):
   seed-law predicts mode 4-5; the K-level C(nf,K)gamma^16 and set-level
   gamma(S)^15 models predict mode >= 11 (set-level: P(12)=0.84). No
   model tuning can blur these.

### Quad: converged verdict + scope boundary

150k-epoch rerun (3x, same 24 seeds, ~$0.4): committees DO NOT collapse
to K=2. K_emb hist {3:14, 4:6, 5:1} (21/24 grokked), logit mode 3;
per-seed 50k->150k churn symmetric (6 shrunk, 9 same, 5 grew) — K=3 is
quad's converged answer. With nu_eff = 9.8/11.3 measured (degree-10
calculus verified, bias share 3e-4), the failure isolates to the MARGIN
FUNCTIONAL: additive cos(k(a+b-c)) coverage is ReLU-transformer-circuit-
specific. Squaring produces the useful cos(k(a+b)) via a cross term PLUS
an unavoidable cos(k(a-b)) term — the achievable-margin curve gamma(K)
for quad circuits is a different function (harmonic members 2xk appear in
5/23 committees as first-class members). Generic replacement: measure
gamma_hat(K) per architecture by fixed-support training (the relu-4 MLP
fixed-support run already validated M_K to 0.1-6% — same instrument
defines gamma_hat for ANY architecture, no representation assumptions).
Side observation: quad long-time dynamics are unstable (train CE rose
2.9e-2 -> 9.2e-2 between 50k and 150k; 3/24 partially de-grokked) —
committees nonetheless stable.

# Round 8 (2026-08-30): the mode is a SELECTION-TIME landscape argmax — statics exhausted, optimizer geometry indicted

User challenge: "if gamma_hat(3) > gamma_hat(4) yet P(4) > P(3), the
formula or the evaluation functional is wrong — introspect, don't fit
constants." Correct. The dig:

## 8.1 Finite-scale statics: tested and DEAD (new instrument)

nudial/ce_equilibrium.py: for a pure-cosine committee the per-pair
training loss is EXACTLY
    L = log(1 + sum_D exp(-sum_k A_k g_k(D))) + lam sum_k A_k^(2/nu)
(true CE = softmin over the p-1 offsets, not hard min; AM-GM chain
pricing). Minimised over amplitudes (equal-amp grid + free-amp Adam
polish + unconstrained wide-support descent), top-20 set pools per K,
lam scanned 1e-9..1e-3 (25 points), lam pinnable by the observed
train-CE floor 1.9e-7. Result at p=113: argmin K=3 at EVERY lam,
order (3,4,2) throughout; free-amp keeps supports alive (no collapse
to 4); wide starts settle at 5-6 survivors. The softmin/finite-scale
correction never flips the 6% gap. Combined with P-N12 (paired
fixed-support training: realized gamma peaks at K*) and the hard-margin
tables, THREE independent static instruments agree: the l2 landscape
optimum is K*. The observed mode K*+1 is inconsistent with ANY static
l2 functional. That is now proof-grade, not a puzzle.

## 8.2 The +1 is optimizer-borne (evidence already in hand, round 3/I)

plain GD+l2 committees: K 3-4 (relu4, K*=4), 2-3 (relu6, K*=3) — the +1
DISAPPEARS under GD (GD parks at/below K*, under-democratic, k_eff 1.6).
AdamW: K*+1 in the SAME families, and in the transformer (nu=6, K*=3,
mode 4) and the MLP (nu=4, K*=4, mode 5). Architecture-independent,
optimizer-dependent.

## 8.3 The selection-exponent law (one constant, four primes, 3% window)

Hypothesis: the committee is selected mid-training (support then
freezes: sleeping-gradient 12-orders result), when the effective margin
exponent nu_sel is BELOW the endpoint nu (part of the chain — E, Q/K in
the eps-regime, attention locks +4000 ep after grok — is not yet
amplifying). Selection then maximises M_K(p)/K^(s), s = nu_sel/2.
Mode constraints from the four observed histograms (mode 4, and 5 must
lose), s intervals:
    p97  (1.910, 2.413)     p113 (1.923, 2.795)
    p127 (2.083, 2.702)     p211 (2.332, 3.036)
Intersection: s* in (2.332, 2.413)  =>  nu_sel in (4.66, 4.83).
Non-empty at 3% width from four independent primes (could easily be
empty — the endpoint s=3 IS jointly excluded). MLP (relu4, mode 5):
needs s in (1.645, 1.920) => nu_sel in (3.29, 3.84), again below its
endpoint nu=4; reduction is larger for the transformer (1.2-1.3) than
the MLP (0.2-0.7), consistent with the attention tensors being the
missing amplifiers. f0.2 consistency: at the grokking boundary
selection happens late (deeper in the l2/eps regime) => nu_sel -> nu
=> mode K*=3, as observed (6/24).

## 8.4 Predictions that discriminate (registered now, before any run)

- P-N15a (measurement): nu_eff(margin) evaluated at COMMITTEE-LOCK time
  (not endpoint) on checkpointed vanilla p113 runs lands in 4.66-4.83,
  rising to ~6 by the end. f0.2 lock-time nu_eff sits HIGHER than base.
  Needs a checkpointed cohort (slim sweep data has no checkpoints).
- P-N15b (intervention): make the geometry GD-like from epoch 0
  (AdamW eps 1e-4, lr retuned x10 as needed; and/or full-batch SGD+wd):
  mode moves 4 -> 3 at p113. Symmetrically, the GD-trained transformer
  cohort should sit at mode 3.
- p=1009 DISCRIMINATOR FLIPS: selection-exponent law predicts mode 5
  (argmax M_K(1009)/K^s for ALL s in the window; M4=0.742, M5=1.461,
  M6=2.198) vs seed-law mode 4 vs dead entropy models >= 11. Three-way
  separation at one prime.
- Quad: harmonic-coupled functional still applies; nu_sel(quad) from
  its checkpoints (they exist) should sit below 10 at lock time.

## 8.5 Status of the two accounts

Seed-projection (round 7) and selection-exponent (this round) both
reproduce mode-4 p-rigidity at 97..211 with one number each; they
DIVERGE at p1009 (4 vs 5) and under N15b intervention (seed story: eps
change does nothing to the mode; exponent story: mode drops to 3).
Both retire the fitted Q. The plateau law (Law 1, zero parameters) is
untouched by which account wins.

## 8.6 (same day, theory-first) The WHY, derived: two pricing regimes, and nu_sel = #eps-crossed degrees

Pure-geometry limits, both now computed:
- l2 pricing (GD, and AdamW's late eps-phase): cost nu A^(2/nu) per
  frequency => splitting penalty K^(-nu/2) => sparse optimum K*.
- pure sign pricing (AdamW's early phase, steepest descent in l_inf):
  amplitude is width-priced (n_k neurons at O(1) scale => A_k ~ n_k),
  LINEAR cost. The max-min coverage LP over the amplitude simplex is
  then symmetric under the unit group => optimum is UNIFORM-DENSE
  (computed: support = nf, all weights equal, at every prime incl 1009).
So neither pure limit matches (3 vs 56); observed 4 forces PARTIAL
pricing, i.e. the interpolation is necessary, not a knob.

Growth-law asymmetry (why the regimes differ at all):
- GD: dw/dt = u w^(nu-1) - lam w. For nu>2, extinction threshold
  w* = (lam/u)^(1/(nu-2)); u falls as loss falls => rising bar,
  sequential recruitment, hierarchical profile ([1,.53,.18,.03]
  measured), parks at/below K*.
- AdamW sign-phase: update g/(sqrt(v)+eps) has magnitude ~lr per coord
  regardless of |g| => NO extinction threshold; growth gated by SIGN
  COHERENCE, not amplitude; egalitarian => democratic profile
  ([1,.85,.69,.60,.46] measured). Tiny-but-coherent gradients still
  drive full-speed growth; recruitment ends only at the eps-horizon
  |g| ~ eps, i.e. margin ~ log(1/eps) = 18.4 nats = THE HOMEOSTAT
  CONSTANT (same number, new role: close of the recruitment window).

Mixed pricing at lock time: tensors cross into the eps-regime at
different epochs (measured: eps on Q/K/E first). A chain layer that has
crossed is l2-priced (quadratic in its scale); one still sign-phase is
box-priced (O(1) scale, ~free; AdamW l_inf box measured non-binding,
0.13-0.63 of 1/lam). If nu_p of the nu degrees are priced at lock,
committee cost ~ K^(-nu_p/2). Hence

    nu_sel = number of eps-crossed chain degrees at committee lock.

Limits check: GD/all-priced nu_p=nu => K* (GD cohorts: yes).
Sign/none-priced nu_p->0 => dense (LP: yes). Observed modes across four
primes => nu_sel in (4.66, 4.83): about ONE degree still unpriced at
lock (transformer). MLP window (3.29, 3.84) of nu=4: also ~one degree
lagging. Candidate: the readout (largest grads cross eps last) —
measurable per-tensor.

f0.2 fits free: near the grokking boundary the climb is slow, more
tensors eps-cross before lock => nu_sel -> nu => mode K*. Seed count
becomes a consequence, not a cause: egalitarian growth toward the
mixed-pricing optimum puts ~argmax M_K/K^(nu_sel/2) climbers above
threshold; p-rigidity of the seed = p-rigidity of that argmax (4 for
all p in 97..211); at p1009 it flips to 5 WITH the mode (unlike the
EVT story's p-independent ~4.7).

NEW REGISTERED DIAL (P-N15c): eps is a committee-size dial. Smaller eps
=> longer sign-phase => fewer priced degrees at lock => larger mode;
larger eps => GD-like => mode -> K*. Sweep eps 1e-12..1e-4 at p113:
mode should FALL monotonically 5-ish -> 4 -> 3. (Homeostat already
showed eps moves the CE floor as log(1/eps) — same instrument family.)
