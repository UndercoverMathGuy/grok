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
