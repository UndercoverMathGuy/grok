# Two Clocks / Lazy Selection — unified claim ledger

Status date: 2026-08-16, rev 2 (post internal adversarial audit). Every
number below was re-derived from the raw runs or the machine-readable
`notes/*.json` during the audit; numbers tagged **[needs artifact]**
reproduce exactly but currently have no committed script — commit one
before submission. Status tags:

- **PREREG-CONFIRMED** — predicted in a prereg README before training,
  then observed.
- **PREREG-MIXED** — preregistered; some predictions passed, some failed;
  failures reported per the prereg branch letter.
- **POST-HOC** — found in existing data; needs the planned replication
  (degrok2, not yet launched) or is corroborated by a prereg cohort.
- **EXPLORATORY** — reported, never scored.

## 0. The headline claim (final form)

> Training a network on modular arithmetic runs two separable processes:
> **selection** (which Fourier committee will implement the algorithm) and
> **learning** (the trajectory to generalization). The two clocks have
> disjoint causal dials: data quantity moves t_gen 35.7x while t_ident
> stays within ~2x; freezing everything-except-W_E delays grok by
> hundreds-to-thousands of epochs (8/8 runs at N=200/500, p=0.0078) while
> selection stays on schedule. Selection completes in the first ~0.05-3%
> of training, is partially readable from the untrained weights
> (weight AUC ~0.65-0.8), and is stored redundantly across weight factors
> as an alignment — not as a token in any single matrix. On the content
> side the evidence is one-directional: learning never overthrows a
> selected committee (zero recruits in 96/96 installs; post-lock data
> jumps mostly preserve it), and no content coordinate DETECTABLY moves
> learning time once strength is controlled — but the Phase B upper
> bounds (69-103 ep on median t_gen 400) sit above the pre-registered
> 15% band, so content-EQUIVALENCE is not yet established. Phase C
> carries that burden.

Explicitly NOT claimed: that the clocks are independent (per-run
correlation ~+0.5-0.6, C6); that selection strictly gates learning; that
wd-invariance of selection is universal (S5 violates it; the registered
generality verdict is 0/2 tasks fully passing — mod-add-scoped pending
degrok2); that the committee can be steered mid-race (three-way
replicated failure); that exact committees can be WRITTEN at init
(compiler capability preregs failed and are excluded — Section 2); that
learning is proven blind to content (only a detection-null with bounds —
C3); that only-W_E training moves the clocks in OPPOSITE directions (the
sign-flip claim is retracted — Section 2).

## 1. Claim ledger

### C1 — Two clocks: selection time and consolidation time are separately controllable. PREREG-MIXED + POST-HOC

- **wd dissociation (POST-HOC, race join, width 512):** t_grok medians
  fall 22150 -> 6600 -> 1500 across wd 0.3 -> 1 -> 3 (**14.8x**; 5-15x
  across widths) while t_ident medians stay 17.5 / 20 / 20. Stratified
  (within-width) permutation Spearman: wd -> t_grok rho -0.81
  (p < 5e-5, n=147); wd -> t_ident rho +0.13-0.19 (p = 0.13-0.24, n.s.)
  **[needs artifact]**. The wd=0.1 cells are excluded: 30/48 runs never
  generalize within the 50k budget (right-censored). The earlier framing
  — "medians 15/29/30/31", "40.5k -> 1.6k", "~30x" — is retracted: those
  were means mislabeled as medians, anchored on the single wd=0.1
  survivor of 8. Replicated in-family by degrok P-G2/P-G5 (wd 1 -> 3:
  t_gen ~3x, |median t_ident shift| 2.5-15 ep). Scope caveats: t_ident
  here is joined from a 2k-epoch twin of the 50k run (27/32 exact
  committee match); at width 128 t_ident drifts 1.65x with wd —
  invariance is a width >= 256, mod-add fact.
- **Freeze dissociation (prereg cohort `clock/`):** freezing
  everything-except-W_E for N epochs delays grok +585 / +1295 at
  N=200/500 — 8/8 runs delayed, exact Wilcoxon p=0.0078 each — while
  t_ident moves only +7.5-12.5 (n.s.). The N=50 delay (+390) is marginal
  (p=0.055). The delay is **sublinear** in N (~N^0.5) with a large
  intercept (even 50 frozen epochs cost ~390) and rising marginal cost
  (1.3 -> 2.4 ep/ep). Disclosed: the freeze also scrambles WHICH
  committee wins (J_base 0.31-0.37 vs the 0.6 prereg bar — P-T2
  weakened), so this is a timing lever, not a content-preserving one.
- **Only-W_E training (on arms):** delays selection dose-responsively —
  d_t_ident +27.5 / +85 / +100 at N=50/200/500 (Wilcoxon p=0.023 and
  0.016 at N=200/500; pooled dose-response rho +0.575, p=0.0047
  **[needs artifact]**) — while grok timing shows **no consistent
  sign** (medians -490 / -242.5 / +347.5, none significant at n=8).
  What passed is prereg P-T4 as registered — a magnitude dissociation at
  on500: grok delayed >= 250 while selection delay <= 100 (boundary tie).
  The "grok accelerates / opposite directions" sign claim is retracted
  (Section 2).
- **P-T1 ("selection = W_E training dynamics") KILLED by its own
  prereg:** median dt_ident(fz500) = 7.5 vs kill bar 125. The
  pre-declared revision (downstream race over a static tilt) is
  directionally supported but underpowered: epoch-0 W_E AUC vs eventual
  winner 0.734 (base) vs 0.522 (on500), paired p=0.07; the 0.734 -> 0.850
  rise under freeze is monotone in arm means but n.s. at n=8 (p=0.29),
  and W_E@500 predicts the winner in EVERY arm (0.97-1.0), so it does not
  discriminate. Mechanism leg = suggestive, not proven.
  **[needs artifact]**

### C2 — De-grok: the two clocks are a property of training, not of grokking. PREREG-CONFIRMED (the strongest leg)

`degrok/` — 80 runs, fresh META_SEED, fresh-to-this-line prime anchor
(p=127). All five predictions (P-G1..P-G5) passed; audited numbers:

- t_gen spans **35.7x** across frac_train 0.3 -> 0.9 (7057.5 -> 197.5;
  conservative — counting the 2 censored frac-0.3 runs pushes it to
  ~41x) while t_ident spans **2.2x** (27.5 -> 12.5) under the prereg's
  own exclusion rule (label-less non-generalizing runs excluded at
  frac 0.3; the previously quoted "1.6x (20 -> 12.5)" retained them,
  contrary to the prereg — corrected). The grokking gap dissolves
  6938 -> 78 ep while fit time stays 120-160 (all cell medians)
  **[t_fit needs artifact]**. Grokking = the microscope, not the
  subject.
- wd-invariance of t_ident: |median wd1-vs-wd3 shift| 2.5-15 ep at every
  fraction and at p=127, inside the registered <= 25-ep bar. Honest
  footnote: at frac 0.3 the wd ratio is 1.75x — larger than the whole
  data-fraction span — so "invariance" is a bounded-shift statement, not
  a ratio statement.
- Epoch-0 W_E readability: tilt AUC **0.65-0.72** per cell under the
  exclusion rule (the old 0.77 endpoint came from two degenerate K<=2
  memorizing runs the prereg said to drop). Knife edges disclosed: the
  two frac-0.5 cells clear the 0.65 bar by 0.0007-0.0012; the p=127 wd1
  cell alone is 0.60 and passes only pooled with wd3 (the pooling was
  coded pre-results).
- Retracted: "selection slightly FASTER with more data" (paired p=0.064,
  ratio CI includes 1.0; not monotone at wd3). The supported statement
  is: selection never slows while t_gen collapses 35.7x.
- Stated limitation: the t_ident detector starts closer to threshold at
  high frac (epoch-0 AUC 0.85 vs 0.72), 5/80 runs hit t_ident=0, and
  within-cell Spearman(auc0, t_ident) is -0.5 to -0.9 — so "selection
  DYNAMICS are unchanged by data" is not established, only that t_ident
  stays pinned within ~2 snapshot widths.
- Needs the wider replication over more categories (degrok2).

### C3 — Content-level dissociation. DOWNGRADED 2026-08-16: detection-null with bounds; equivalence NOT established. Phase C is the test.

**Retracted (i):** the original identity-within-dose permutation
p = 0.93. `make_arms.make_sweep` redrew the target set inside the
innermost loop; all 96 Phase B target sets are distinct (verified), so
the identity factor carried no content and its null was true by
construction.

**Retracted (ii), this audit:** the rebuilt "nine contrasts all
equivalent at the pre-registered 15% band" verdict. The scoring rule in
`analyze_sside.py` was (p >= 0.05) AND (point estimate < band) — an
absence-of-evidence rule; no confidence interval was computed anywhere.
Under a genuine TOST (within-cell bootstrap and asymptotic 95% upper
bounds, which agree), only 3-4 of 9 contrasts establish equivalence, and
**every genuinely content-bearing coordinate fails**: LP margin
percentile, mean target frequency, target span, and lowest target carry
upper bounds of 69-103 ep against the 60-ep band. A simulated TRUE
band-sized effect is declared "equivalent" 53-56% of the time. The
quoted power ("MDE 47 ep = 12% < 15%") was the MDE of a two-group test
never run; honest per-statistic MDEs are 58-118 ep, above the band for
7/9 tests. Additional defects: the "max T_k rank" contrast is
identically equal to K (a design factor, zero content — the same defect
class as retraction i); lp_relM is rho = 0.94 with K; and the 100-ep
measurement grid (7 distinct t_gen values in the cohort) cannot resolve
a 60-ep band.

**What Phase B DOES establish (verified, byte-reproducible):** among 96
mainline-recipe runs with known installed content x dose (recipe audit
exact: all arms identical except init seed — p=113 add, frac 0.3,
d128/4h/512 ReLU, lr 1e-3, wd 1.0, 30k ep; footprint 1 data mask x 2
base init seeds on QR-flat W_E), once dose AND base are controlled
(R^2 = 0.595 together; dose alone 0.456), **no content coordinate is
detected**: largest point estimate 46 ep on median t_gen 400, all
p >= 0.19; install exactness (54/96 exact) and J(realised, installed)
are also null with tight bounds (UB 20-47 ep). The defensible sentence:
*holding strength fixed, no content effect was detected, with 95% upper
bounds of 69-103 epochs (17-26% of median t_gen) on the content-bearing
coordinates.* That is a real result; it is not equivalence.

**Acceleration caveat, corrected:** compiled Phase B runs grok at
200-1100 (median 400). Phase B used only **2** of the 8 natural bases;
the correctly paired natural baseline is {5100, 8500} (median 6800) —
**17-21x acceleration** (the old "5100-13400, median ~8250, same 8
bases" compared against 6 bases that contributed no arms)
**[natural-base t_gen needs artifact]**. There is no flat-W_E s=0
control, so the acceleration cannot be apportioned between QR-flattening
and installing content. Installing ANY committee at dose d is a massive
head start; dose is the dominant regressor. The claim is CONDITIONAL on
strength by construction — we cannot and do not claim "selection
interventions don't affect learning."

**The L-side (learning -> selection content):** revealing 2.3x more
training data at epoch 1000 (post-lock) accelerates learning 1.5x while
the committee holds at **median J = 0.875 — bimodal: 4/8 exact, 3/8
partially re-selected (J 0.25-0.5)**; self-restart gate 8/8 at J = 1.0.
Prereg P-B7 + P-B8 PASS (`boost/` round 2). Honest form: post-lock
committees are never overthrown wholesale and usually held, occasionally
trimmed or partially re-selected.

### C3-C (Phase C) — the S-side on good runs. PRE-REGISTERED, not yet trained

`compiler/arms/phaseC_manifest.json` — the manifest holds **296 arms**
(192 install + 8 no-install controls + 96 L-side; "200" in earlier
drafts was the S-side subtotal). Design audit (all verified):

- **Crossed by construction:** ONE shared pool of 8 K=4 contents, drawn
  once, installed at every dose on every base — 8 contents x 3 doses x
  8 bases, 24 arms per content. T1 (content identity) becomes a real
  factor; it was undefined on Phase B.
- **Natural substrate** on 192/192 installed arms (base's own W_E kept,
  tilt added); achieved margin == dose exactly (global renorm, verified
  to 7e-15); injected strength at fixed dose varies 1.96x across
  content x base and is recorded per arm as a covariate.
- **Footprint:** all 8 natural p=113 bases (2 data masks x 4 init
  seeds) + one no-install control per base, anchoring the natural t_gen
  in-cohort. Doses s in {1.5, 3, 6}; 30k epochs.
- **Scoring upgrades required before training (from this audit):** the
  verdict must be a genuine CI-based TOST (upper bound < band), not
  p >= 0.05 + point estimate; drop or rename the rankmax contrast
  (K is fixed at 4 here, so it is degenerate anyway); pre-register
  honest per-statistic MDEs; and note the 25-ep spectra grid is a
  LAUNCH FLAG (`--spectra-every 25`), not a manifest field — it is
  load-bearing (a band finer than the grid is unresolvable): do not
  launch without it.

Verdict rule: the S-side stands if T1 plus the content coordinates are
equivalent under the CI-based rule at the 15% band. A detected content
effect at any dose is a first-class finding falsifying the
strength-channel form of C3.

### C4 — The selection state is a W_E<->downstream ALIGNMENT, redundantly stored. PREREG-MIXED

- **Transplant** (`transplant/`): swapping W_E at t=0 into a fresh net
  transfers no detectable committee — median max J 0.21 vs a computed
  fresh-lottery null of 0.14 (p = 0.053), while the same instrument
  detects transfer easily at t >= 50 (p <= 0.0013) **[null needs
  artifact]**. Handoff: recipient-side J_B 0.07 -> 0.33; the donor side
  also gains (mean 0.13 -> 0.28; median flat) — both sides grow legible,
  recipient earlier and faster. Self-restart gate median J = 1.0 (14/16
  exact). "W_E knockout is necessary" is a legacy READOUT result (the
  epoch-0 predictive signal dies, AUC -> 0.51) — necessity of the
  predictive signal, not causal necessity in this cohort.
- **Lesion** (`lesion/`): destroy EITHER factor with norm-matched noise
  at t=150/500 and the surviving factor restores the committee above
  null (rand_we 0.27 -> 0.45, rand_rest 0.50 -> 0.67, all p <= 0.019;
  nulls 0.00-0.13). 31/32 lesioned runs re-grok (one reaches 0.764);
  one of the 8 donors never groks in ANY cohort and is what drags P-L1:
  with it, median 0.45 vs the 0.5 bar — FAIL as registered; without it,
  exactly 0.500. The composite "strict redundancy" verdict is NO by the
  scored rule (it inherits P-L1's bar) and YES by the prereg's
  both-arms-above-null wording — both reported. The redundancy is
  asymmetric: the W_E-copy leg (rand_rest 0.67) is near-implied by
  W_E's own AUC 1.0 at t=500; the downstream copy (0.45) is the weaker,
  load-bearing leg.
- **Isometric swap** (`boost/` round 2): an exactly norm-preserving W_E
  subspace edit at t=20 (orthogonal rotation in Fourier coordinates —
  proven analytically, asserted numerically) scrambles the committee
  (sham J 0.50, n=8). Attached caveats: (i) a run-to-run committee
  instability floor exists — identical re-runs disagree in 2-3 of 8
  seeds (J down to 0.50; median 1.0) **[needs artifact]** — sham 0.50
  clears that floor but the margin is smaller than it reads;
  (ii) **no cold-restart gate exists at t=20** (all gates are at
  t >= 150), so an Adam-restart confound is not excluded mid-selection —
  an 8-run gate20 arm is required before "ANY mid-race W_E surgery
  re-runs the race" can be claimed unconditionally; (iii)
  "content-irrelevant" means the two swapped frequencies are
  non-members of the FINAL committee; at t=20 selection is live and the
  edit does rewrite their direction structure.
- **Engram timeline** (`lesion/engram_curves.py`): **EXPLORATORY,
  n = 8 + 8.** W_E's committee record consolidates before W_U's on any
  fixed threshold (AUC 0.90 at ep 30 vs 70; 0.99 at ep 80 vs 110).
  Masks are epoch-1500 committees (2/8 runs grokked; the 3 label
  mismatches vs 3k-epoch twins are strict subsets, so the bias is
  conservative). **Retracted:** the "order flip" / "first scribe is
  whoever can move" / write-then-replicate moral — freezing W_E leaves
  W_U's write timeline essentially unchanged (0.90 at ep 80 vs 70
  natural), which supports independent PARALLEL writing, not sequential
  replication; the frozen cohort also never grokked (0/8 at 2500 ep).

### C5 — Selection is init-readable and init-influenceable. POST-HOC readout + intervention legs (pruned this audit)

- **Read (POST-HOC, race join):** epoch-0 **logit-spectrum** AUC vs
  final committee rises with width — rho +0.48, p = 0.0007, init-level
  n=48; per-width means 0.64 / 0.57 / 0.76 / 0.80 / 0.78 / 0.79
  (non-monotone: dip at 256, plateau above 1024 — "0.64 -> 0.80" is the
  envelope, not a trend line). The epoch-0 **weight** readout shows the
  same shape (auc_we 0.61 -> 0.80; within-wd rho +0.41, p < 5e-5,
  n=148) **[needs artifact]**. Disclosed: the preregistered form of
  this test (AUC at 0.2*t_grok vs width) was scored KILL; the epoch-0
  form is a post-hoc timepoint/instrument change. t_ident falls with
  width (rho -0.52, p = 5e-5). Prereg corroboration: P-G4 — epoch-0 W_E
  tilt AUC clears 0.65 in every cell, fresh seeds (0.65-0.72).
- **The causal init leg that survives:** installing content into the
  init steers the realized committee — Phase B: realized committee is a
  subset of the installed set in **96/96 arms (zero recruits)**, exact
  in 54/96, K=3 installs exact 31/32. Capability preregs failed their
  bars (Section 2), so no exact-writing claim — but "init content is
  causally upstream of the winner" is carried by these interventions.
- **Removed as evidence this audit (details in Section 2):** orth-init
  "kills the lottery tilt"; the tilted-ERM "nudge at t=5"; unqualified
  single-frequency loser<->winner steering.
- **Three regimes, restated at current strength:** writable-influenceable
  at init (Phase B installs); mid-race edits scramble (swap/lesion —
  pending the t=20 gate control, C4); self-repairing after lock (lesion
  recovery; late-jump median J 0.875 with 3/8 partial re-selections;
  gates J = 1.0 at t >= 150). The programmability-window cohort
  (same-dose edit at t in {0,5,10,20,40}) is still not run; it would
  turn this from argument into figure.

### C6 — The clocks share init luck; "independent" is falsified, "separately controllable" stands. POST-HOC

Directionally robust to pooling; the exact numbers depend on the cell
set and have no committed artifact yet **[needs artifact]**. On the
consistent 14-cell pooling (degrok + mod-96 + S5 frac-0.7), rank-z
within cell: rho(t_ident, t_gen) = **+0.58**; epoch-0 tilt AUC predicts
both clocks (**-0.57 / -0.48**); partial rho given tilt **+0.43**.
(Earlier text mixed a 10-cell tilt estimate with a 14-cell count.)
Per-cell n=8 and only ~4/10 degrok cells are individually significant —
the claim is pooled-with-within-cell-centering, not a per-cell fact.
The claim is NOT independence: two clocks with disjoint causal dials
(data moves only t_gen 35.7x; freezes delay grok while selection holds)
fed by one shared scalar — the same scalar as C3's dose channel.

### C7 — Generality and the honest boundary. PREREG-MIXED; registered verdict: 0/2 tasks fully pass -> mod-add-scoped pending degrok2

- **mod-96 (composite):** clocks replicate — wd moves t_gen 3.15x while
  t_ident holds (0.73x), n=16; K ~ 5/48 at wd1. But P-M4
  init-readability FAILED its registered 0.6 bar (0.568/0.598 — a
  near-miss, reported as the fail it is). "Not prime-specific" holds for
  the clocks only.
- **S5 (non-abelian, irrep instrument):** groks 8/8 at frac 0.7 within
  100k; committees K=2/6, always containing sign, paired with std_sign
  (6/8) or std (2/8). The raw irrep-energy readout FAILED prereg P-M4
  (AUC 0.09-0.27). **Retracted:** the "init-readable AFTER
  rank-normalizing by d_rho^2 (AUC 0.75-1.0)" rescue — it has no code,
  no saved output, the epoch-0 checkpoints are deleted, and the
  statistic is degenerate: a constant 1/d^2 ranking using zero init
  information scores 0.94 on all 8 runs (the committee and the
  normalization are both functions of irrep dimension). S5
  init-readability is OPEN, not rescued. P-M2 (selection early in ALL
  cells) also FAILED (one cell at 0.395 vs the 0.2 bar). **BOUNDARY
  confirmed: wd moves BOTH clocks on S5** (t_ident 2.27x ~ t_gen
  2.14x). degrok2 arm E tests whether the DATA-clock dissociation
  survives on S5, which the strength-channel story predicts.
- **Capacity:** K ~ 4 is task-intrinsic — invariant across 32x width
  (cell means 3.65-4.44; width-vs-K rho +0.096, p = 0.26 **[needs
  artifact]**); width scales lottery DETERMINISM instead (C5 readout).
  EXPLORATORY corroboration from Phase B interventions: zero recruits
  in 96/96 arms; oversized sets pruned down (K=5 -> exact 4/32, mean
  realized 3.8; even K=4 is pruned in 12/32) and undersized sets never
  topped up (K=3: exact 31/32, none grew); an 8-fold dose range
  (s = 3 -> 24) does not rescue. Ceiling ~3.5-4, not a set-point.

## 2. Dead hypotheses and retracted analyses (full disclosure)

Original dead list:

- Selection = W_E training dynamics (P-T1, killed by its own prereg).
- W_E is sufficient to carry the ticket (transplant t=0, killed).
- W_E is the boss engram (killed — and see below: the replacement
  "sequential write / order flip" story is now also retracted).
- Two INDEPENDENT clocks (per-run coupling; revised to C6).
- Amplitude-in-W_E = selection state (isometric swap, killed).
- Mid-race steering via boost (norm saw at alpha >= 4 destroys grokking
  even boosting the run's OWN winner) and via isometric swap (leader
  demoted 8/8, target crowned 0/8). No mid-flight S-lever in W_E space.
- Mask low-order spectral proxies; LP-support / min-feasible-K /
  multiplicative census / surrogate transfer proxies (all null).
- wd and lr as intervention levers (vetoed: global, side-effect-laden).
- Capacity dial (K moves with width) — reversed same day.
- Data jump at t=200 as a pure L-lever (P-B4 fail: re-opens selection,
  J 0.71 — data is a selection input pre-lock; the lever is only clean
  post-lock).
- Compiler capability predictions — DROPPED from evidence (author
  call): Phase A prereg P-C1 FAILED (2/6 exact vs >= 5/6 bar); Phase B
  exact-set 56%. No "we can write exact committees" claim anywhere.
  Phase B runs retained ONLY as the C3 intervention dataset (recipe
  verified mainline) and the C7 pruning by-product.

Added by the 2026-08-16 audit:

- **Phase B S-side "equivalence at the 15% band"** — retracted. The
  scoring rule was p >= 0.05 + point-estimate-under-band (absence of
  evidence, not TOST); a genuine TOST fails 5/9 contrasts including
  every content-bearing coordinate; honest MDEs (58-118 ep) exceed the
  band for 7/9 tests; one contrast was identically K. Downgraded to the
  detection-null-with-bounds in C3.
- **"One knob, opposite directions" (only-W_E sign flip)** — retracted.
  The grok "acceleration" is n.s. at n=8 (Wilcoxon p=0.46/0.95), the
  on200 mean has the opposite sign from its median, and the sign
  reverses at on500 — where prereg P-T4 predicted and observed a grok
  DELAY. What stands is the magnitude dissociation (selection delayed
  dose-responsively; grok moved much more, sign unstable).
- **"Superlinear in freeze length"** — corrected: delay ~ N^0.5,
  sublinear, with a large intercept and rising marginal cost.
- **Orth-init as a causal/necessity leg** — removed. QR rewrites all
  cross-token geometry, not just the tilt; committee change is what ANY
  init edit produces (assembly is chaotic to sub-threshold swaps); n=4;
  flattening does NOT kill init readability (flat-energy AUC 0.637).
- **Tilted-ERM "nudge at t=5" as an init leg** — removed. It is a loss
  modification held for the whole run (t=5 is the tilt hyperparameter,
  not an epoch); the +10.3-percentile effect is carried by 1 of 4 seeds
  and its own noise control reproduces that seed's jump. Belongs to the
  selection-pressure program, not to init writing.
- **Single-frequency loser<->winner steering (unqualified)** — demoted
  to EXPLORATORY legacy. Forward direction real within-context (dose
  threshold in (1.05, 1.10) energy scale) but the universal-threshold
  form was FALSIFIED (T57: one base fails at max dose; threshold varies
  ~10x with context); the winner->loser direction is n=1; nothing is
  machine-readable.
- **Engram "write-then-replicate" / order-flip / "first scribe is
  whoever can move"** — retracted. W_U's write timeline is essentially
  unchanged by freezing W_E (0.90 at ep 80 vs 70), supporting
  independent parallel writes; the frozen cohort never grokked. What
  stands: W_E's record consolidates ~40 ep before W_U's, exploratory.
- **S5 d_rho^2 rank-normalized readability** — retracted (degenerate
  statistic, no artifact, no surviving epoch-0 data).

## 3. Where a reviewer will push, and the current answer

1. **"Toy task, one architecture."** degrok2 (DESIGNED, NOT LAUNCHED —
   no runs, no summary; this answer is a plan) widens to 2 operations,
   3 moduli incl. composite, 2 extra widths, S5. Still one 1-layer
   transformer family — acknowledged; the LM pivot was tried and killed.
   And per the registered task-generality rule the current verdict is
   0/2 tasks fully passing: the paper is mod-add-scoped until degrok2.
2. **"t_ident is an instrument artifact."** The detector is fixed
   (largest-gap, AUC >= 0.9 sustained x2) across cohorts; freezes move
   t_ident by hundreds of epochs when W_E is the only free factor; S5
   uses a different instrument (irreps) and still shows early lock in
   most cells. Conceded and now disclosed: at high data fractions the
   detector starts near threshold (epoch-0 headroom confound, C2) and
   5/80 degrok runs sit at the floor; the claim is stated as "t_ident
   stays pinned", not "selection dynamics are identical".
3. **"Dose confound in the content claim."** Conceded and quantified —
   C3. The claim is conditional-on-strength by construction.
4. **"This is an absence-of-evidence claim."** Now conceded for
   Phase B: it is a detection-null with 95% upper bounds (69-103 ep)
   that sit above the pre-registered band. Equivalence is NOT claimed
   on Phase B. Phase C, scored with a genuine CI-based TOST at a 25-ep
   grid, carries the equivalence burden.
5. **"Per-run correlation contradicts two clocks."** It contradicts
   independence, which we do not claim — C6 is a result, not buried.
6. **"S5 wd-coupling breaks universality."** Scoped in C7; the
   strength-channel form predicts the data-clock dissociation survives
   on S5 (degrok2 P-H6 tests it; a fail scopes the claim to abelian
   tasks and is reported).
7. **"Novelty vs Kunin AGF / 'Two Training Clocks' 2606.05863 / He
   2602.16849."** Carve doc: `notes/two_clocks_related_work.md`. Ours is
   the interventional/causal completion; the title collision is a
   different pair of clocks (optimizer-internal).
8. **"Your mid-race chaos could be an optimizer-restart artifact."**
   Ruled out at t=0 (no optimizer state exists) and at t >= 150 (gates
   J = 1.0); NOT yet ruled out at t=20, where the isometric swap lives.
   The gate20 arm (8 runs) is queued; until it exists the swap leg is
   stated conditionally (C4).

## 4. Repro map

| cohort | dir | runs | key output |
|--------|-----|------|-----------|
| race join | dial/ | 148 kept / 192 | notes/race_summary.json (+ wd test **[needs artifact]**) |
| freeze/clock | clock/ | 56 | notes/clock_summary.json |
| de-grok v1 | degrok/ | 80 | notes/degrok_summary.json (+ t_fit, exclusion-rule numbers **[needs artifact]**) |
| de-grok 2 | degrok2/ | 0 | NOT LAUNCHED — code only |
| transplant | transplant/ | 56 | notes/transplant_summary.json (+ fresh-lottery null **[needs artifact]**) |
| lesion | lesion/ | 48 | notes/lesion_summary.json |
| engram curves | lesion/engram_*.py | 8+8 | notes/engram_curves.json, engram_fz_curves.json (EXPLORATORY) |
| boost r1+r2 | boost/ | 96 | notes/boost_summary.json, boost2_summary.json |
| tilted-init data (C3) | compiler/ | 8+96 | compiler/arms/phaseB_scores.json, notes/sside_phaseB.json (detection-null only; capability claims excluded) |
| phase C | compiler/arms/ | 0 trained | phaseC_manifest.json — 296 arms, inits compiled, UNTRAINED |
| tasks (m96+S5) | tasks/ | 48+ | notes/tasks_summary.json, tasks_ext_summary.json |

Known repro hazards: `safetensors` is imported by the analysis path but
not declared in `pyproject.toml` (use `uv run --with safetensors` or fix
the dependency); S5 epoch-0 checkpoints are deleted (tilt numbers cannot
be recomputed); several headline numbers are tagged **[needs artifact]**
above and must be committed as scripts + JSON before submission.

Trainer: `cloud/train_semifinal_torch.py` (MLX-semantics AdamW, lockstep
batched, bitwise-verified freeze gating, fn_name in {add, sub, s5} with
task-matched spectral instruments). Failures are reported under their
pre-declared branch meanings.
