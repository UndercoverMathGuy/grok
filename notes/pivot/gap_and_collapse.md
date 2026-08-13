# Gap-conditional predictability and the compiled margin curve

2026-08-13. Analysis-only on existing data (no training). Scripts:
`scripts/pivot/gap_determinism.py`, `scripts/pivot/margin_collapse.py`.
Results: `notes/pivot/gap_determinism.json`, `notes/pivot/margin_collapse.json`,
joint summary `notes/pivot/gap_and_collapse.json`.

**Theory under test.** Committee selection is an eigenvalue race in the init
arrival-loudness readout `T_k = Σ_h ||W_O^h W_V^h W_E|_k||²`. Where the
spectrum has a clear GAP at the committee cut, the outcome should be
determined; where it is near-degenerate, a coin-flip. Compiled arms should
then show a single adoption-vs-margin curve that absorbs dose `s` and
committee size `K`.

**Verdict in one line.** The gap→predictability *ladder* is real and spans
five regimes, but neither sharp form of the theory survives: the gap **at the
cut** carries no signal on natural inits (AUC 0.47–0.56), and margin does
**not** collapse the s×K grid — at fixed margin, `K` is a first-order driver
of adoption while an 8× dose increase is worth ~a third as much.

---

## Data actually used

* **Natural runs**: `semifinal/analysis/common.discover()` yields **96 kept
  runs** (p=113 throughout), all with a materialized epoch-0 checkpoint.
  Cohorts: `orth-flat` 54, `surgical` 26, `natural-normal` 8, `double-flat` 8.
* **Same-init grouping** (verified by SHA-1 over all epoch-0 tensors, not by
  name): **8 twin groups of 6–7 runs each** (54 runs) — one orthogonal-W_E
  init trained under 7 recipes (`orthWE`, `dyn-lr3`, `dyn-lrlo`, `dyn-wd04`,
  `dyn-wd25`, `eff-G` grad-noise, `phase2-tilt` tilted ERM). All other runs
  have a unique init: the surgical families (`dosefarm`, `gkrotate`,
  `suppress`, `collisionfarm`, `chaospair`, `transplant`) each *edit* their
  base's init, so they are not twins; `cluster_key` groups them by base
  anyway, which is the right unit for independence but not for a determinism
  measure.
* **Compiled arms**: Phase B 96 arms (4 doses × 3 K × 8 sets) + Phase A 6
  target arms (2 controls skipped) = **102 arms / 408 targets**. Per-target
  adoption taken from the scores JSONs (10×-median-amplitude criterion); it
  agrees with unified-detector committee membership on **98.5 %** of targets.
* **Bonus, recovered not documented**: the single-target promotion arms
  (`dosefarm` ×1.10/1.20/1.50/2.25, `gkrotate`, `collisionfarm`, `chaospair`)
  have no target recorded in their configs. Targets and doses were recovered
  by diffing each arm's epoch-0 `T_k` against its natural base
  (dosefarm/gkrotate target k=37 for seed31831 and k=52 for seed47904, etc.).
  These 18 arms are the only data in the ±0-margin regime and they carry the
  coin-flip boundary.

---

## ANALYSIS A — gap-conditional predictability (96 natural runs)

### A0. The natural spectrum has no gapped regime at all

| statistic (log units, nats) | p10 | median | p90 | max |
|---|---|---|---|---|
| gap at the true cut `log T_(K) − log T_(K+1)` | 0.0009 | **0.0073** | 0.028 | 0.075 |
| largest gap in the head (cut-free) | 0.032 | 0.057 | 0.158 | 0.682 |
| head spread `log T_(1) − log T_(8)` | 0.073 | 0.138 | 0.283 | 0.926 |

The median cut ratio is **T_(K)/T_(K+1) = 1.0073** — a 0.7 % lead. On average
**8.0 background frequencies sit within 5 %** of the K-th place value and 35
within 20 %. For comparison, the compiled arms are built at ratios of 3–24
(1.10–3.18 nats), i.e. **150–400× larger than the median natural cut gap**.
Natural inits are *all* in the degenerate regime; the theory's "gapped" arm
does not exist in this dataset.

Consistently, top-K of `T_k` **never** reproduces a committee exactly:
**exact hit 0/96**, mean Jaccard 0.130, mean membership AUC 0.659 (matching
the known claim-1 ticket strength). Graded outcomes used instead:
top-1 hit 36.5 %, ≥1 hit 60.4 %, ≥half hit 15.6 %.

### A1. The gap AT the cut predicts nothing

| predictor | AUC(top-1 hit) | AUC(≥half hit) | ρ(Jaccard) | ρ partialling out K |
|---|---|---|---|---|
| gap at cut (log) | 0.558 [0.40, 0.76] | 0.591 [0.44, 0.76] | +0.091 (p=0.38) | +0.072 (p=0.49) |
| gap at cut / median T | 0.569 | 0.600 | +0.107 (p=0.30) | — |
| gap at cut / T_(K) | 0.558 | 0.591 | +0.091 (p=0.38) | — |

Gap-quartile table (cut gap) — flat, non-monotone:

| quartile | n | gap range | mean J | exact | mean memb-AUC |
|---|---|---|---|---|---|
| Q1 | 24 | 0.000–0.003 | 0.081 | 0 | 0.620 |
| Q2 | 23 | 0.004–0.007 | 0.154 | 0 | 0.675 |
| Q3 | 25 | 0.007–0.018 | 0.182 | 0 | 0.725 |
| Q4 | 24 | 0.019–0.075 | 0.104 | 0 | 0.615 |

Restricted conditional claim (top vs bottom quartile of cut gap): membership
AUC **0.615 vs 0.620** (Mann-Whitney p = 0.66). Flat null. On the
unedited-init subset it is if anything inverted (AUC(top-1) = 0.468).

### A2. The cut-free spread DOES predict, and survives K-adjustment

| predictor (pooled, n=96) | AUC(top-1) | AUC(≥half) | ρ(J) | partial ρ(J\|K) | ρ(memb-AUC) |
|---|---|---|---|---|---|
| largest head gap | 0.770 [0.57, 0.94] | 0.863 [0.74, 0.96] | +0.515 (8e-08) | +0.521 (5e-08) | +0.264 (0.009) |
| head spread T_(1)/T_(8) | **0.852** [0.69, 0.96] | **0.883** [0.77, 0.97] | **+0.598** (1.2e-10) | +0.598 (1.2e-10) | +0.353 (4e-04) |

Gapped-vs-degenerate split on head spread (top vs bottom quartile):

| | bottom Q (n=23) | top Q (n=26) |
|---|---|---|
| membership AUC | 0.568 [0.51, 0.63] | **0.739** [0.71, 0.79] |
| top-1 hit rate | **0.00** | **0.77** |
| ≥half hit rate | 0.00 | 0.46 |
| mean Jaccard | 0.033 | 0.254 |

Mann-Whitney on membership AUC p = 5.5e-06 (cluster bootstrap CIs above are
over init groups). This is the theory's predicted near-chance vs
well-above-chance split — but at "near-ceiling" it is only 0.74, and the
predictor that works is the *overall* head spread, not the gap at the cut.

**Caveat, and it is a big one.** The effect is strongest in the `surgical`
cohort (ρ = +0.725), whose inits were deliberately spiked (dose ×2.25 etc.),
so "a spike predicts itself" is partly tautological. On unedited inits only
(n=70) it weakens but survives: AUC(top-1) 0.803, ρ(J) +0.365 (p=0.0019),
partial ρ(J|K) +0.382 (p=0.0011); membership-AUC split 0.662 vs 0.601
(p = 0.097, n.s.). Per cohort: natural-normal ρ(J) +0.740 (p=0.036, n=8),
orth-flat +0.341 (p=0.012, n=54), double-flat +0.082 (n.s., n=8).

The cleanest single data point is the **double-flat cohort**: orthogonal W_E
*and* isometric attention make `T_k` exactly flat (head spread 6.6e-09 nats).
There the readout is at chance — membership AUC **0.557**, Jaccard 0.061,
top-1 hit **0/8**. Perfect degeneracy ⇒ perfect blindness, as the theory says.

### A3. Twin determinism: the gap does not predict it

8 identical-init groups; committee agreement across recipes sharing one init:

| init | gap-free head gap | head spread | twin J (all 6–7) | twin J (5 plain dynamics) |
|---|---|---|---|---|
| ds9415/32610 | 0.109 | 0.199 | 0.499 | 0.770 |
| ds3148/81412 | 0.084 | 0.138 | 0.197 | 0.268 |
| ds3148/47904 | 0.057 | 0.121 | 0.424 | 0.740 |
| ds9415/85697 | 0.051 | 0.161 | 0.196 | 0.467 |
| ds9415/19078 | 0.050 | 0.093 | 0.407 | 0.504 |
| ds9415/68549 | 0.039 | 0.096 | 0.281 | 0.471 |
| ds3148/31831 | **0.037** | 0.116 | **0.556** | **1.000** |
| ds3148/69089 | 0.032 | 0.073 | 0.312 | 0.407 |

Mean twin Jaccard 0.359 (0.578 restricted to the five plain lr/wd recipes) —
same init, different dynamics gives *substantially* different committees, so
the init is not a deterministic program even before asking about gaps.
Spearman(gap, twin Jaccard) = **−0.05 (p = 0.91, n = 8)**; with head spread
also −0.05; dynamics-only version +0.05 to +0.17, all n.s. The most
deterministic group (seed31831: identical committee `[6,19,52]` under all five
plain recipes) has the **smallest** cut-free gap in the set — a direct
counterexample. n = 8 is small, but there is no hint of the predicted trend.

---

## ANALYSIS B — margin collapse of the compiled arms

### B0. The margin variable is degenerate by construction (structural finding)

`compiler/core.py::compile_init` scales **every** target to exactly
`T_t = s · max_bg`. Therefore all targets inside an arm are exactly tied:
across 408 targets there are only **10 distinct margin values** (4 of them —
log 3, log 6, log 12, log 24 — cover all 384 Phase-B targets), and the maximum
within-arm margin SD is **4.0e-09**. Consequences:

* `corr(margin, log s) = 0.998`; `corr(margin, K) = 0.012`. Margin *is* dose.
* `m_t` and `m'_t` (vs best background vs vs the (K+1)-th overall) are
  **identical** for every compiled target — all targets occupy the top-K.
* The "is the evicted target the min-margin one?" test is **vacuous**: the
  ranking it needs is decided by float noise at the 1e-9 level.

This was not anticipated in the analysis plan and it invalidates the intended
per-target margin regression. What can still be tested is the across-cell
version, plus a genuinely varying per-target covariate (`need` = the T_k gain
the compiler had to apply, i.e. how quiet the target was on the substrate).

### B1. Margin does not absorb K — the grid does not collapse

Pooled adoption 87.5 % (357/408 targets; Phase B alone 337/384). Logistic fits
(hand-rolled Newton/IRLS, ridge 1e-6, all converge to ‖∇‖ < 1e-8):

| model | k | logLik | AIC | coefficients |
|---|---|---|---|---|
| null | 1 | −153.72 | 309.44 | const +1.95 |
| **margin** | 2 | −152.03 | 308.05 | margin **+0.362** (z = 1.84) |
| margin + log s | 3 | −152.01 | 310.02 | margin −0.11, log s +0.46 (both n.s.) |
| **margin + K** | 3 | −137.32 | **280.64** | margin +0.426 (z=2.05), **K −1.257 (z=−4.71)** |
| log s + K | 3 | −137.18 | **280.35** | log s +0.434 (z=2.11), K −1.268 (z=−4.72) |
| margin + log s + K | 4 | −136.76 | 281.52 | collinear (margin −2.56, log s +2.95) |
| margin + cell FE (s×K) | 13 | −131.87 | 289.74 | — |

Likelihood-ratio tests:

| comparison | χ² | df | p | ΔAIC (favouring the bigger model) |
|---|---|---|---|---|
| margin vs null | 3.39 | 1 | 0.065 | +1.39 |
| **margin + K vs margin** | **29.41** | 1 | **5.9e-08** | **+27.4** |
| margin + log s + K vs margin | 30.53 | 2 | 2.3e-07 | +26.5 |
| margin + cell FE vs margin | 40.31 | 11 | 3.2e-05 | +18.3 |
| **margin + log s + K vs log s + K** | **0.83** | 1 | **0.36** | −1.17 |
| log s + K + log need vs log s + K | 1.43 | 1 | 0.23 | −0.57 |

Reading: adding **K to margin buys 29 χ² units and 27 AIC**; adding **margin
to (log s, K) buys nothing** (p = 0.36, AIC worsens). The margin-only model
misses cells by up to **14 pp**:

| s \ K | 3 | 4 | 5 |
|---|---|---|---|
| 3 | 0.958 (24) | 0.839 (56) | 0.750 (40) |
| 6 | 1.000 (24) | 0.906 (32) | 0.725 (40) |
| 12 | 1.000 (24) | 1.000 (32) | 0.825 (40) |
| 24 | 1.000 (24) | 0.906 (32) | 0.825 (40) |

Marginals: adoption by K = **0.990 / 0.901 / 0.781** for K = 3/4/5; by dose
s = 0.833 / 0.854 / 0.927 / 0.896 for s = 3/6/12/24. At arm level the
contrast is starker — exact-set match **31/32 (K=3)**, 21/38 (K=4),
**4/32 (K=5)**, versus 13/30, 12/24, 17/24, 14/24 across doses. **An 8×
dose increase buys +6 pp of adoption; one extra committee slot costs −21 pp.**

### B2. m* is not identified by the compiled arms; the promotion arms give it

Every compiled margin is ≥ 1.10 nats and every fitted probability lies in
[0.83, 0.91] — the curve never crosses 50 % inside the data. The
margin-only extrapolation gives **m\* = −3.36 nats** with a cluster-bootstrap
CI of **[−23.9, −0.80]** and slope 0.362 logit/nat (CI [0.077, 0.737],
slope at m* = 0.09/nat). That is a non-answer, and honestly labelled as such.

The 18 single-target promotion arms populate margin ∈ [−0.24, +0.68]:

| margin sign | adopted |
|---|---|
| m > 0 (target is loudest at init) | **8 / 8** |
| m < 0 | **2 / 10** |

Fisher exact p = **0.0010**; `T_k`-rank-1 → 8/8. Logistic on these 18:
**m\* = +0.028 nats (dose ×1.03), slope 7.81 logit/nat** (z = 1.86 — steep but
imprecise at n=18). Pooled with the compiled arms plus a substrate indicator:
m\* = −0.37, slope 0.43 (z = 2.16), compiled-substrate offset +0.92.

So the threshold sits essentially **at margin 0 — the promoted frequency must
become the loudest at init** — and the two boundary exceptions are both
"lost from rank 2–4" or "won from rank 4" cases:

* `dosefarm/seed47904/dose_110` (k=52, m = −0.238, rank 4) — not adopted
* `dosefarm/seed31831/dose_110` (k=37, m = −0.093, rank 5) — not adopted
* `dosefarm/seed31831/dose_120` and `gkrotate/seed31831/gain_120`
  (m = −0.006, rank 2) — **both fail**, the sharpest coin-flip case: a 0.6 %
  deficit is enough to lose
* `dosefarm/seed47904/dose_120` (m = −0.151, rank 2) adopted while the
  matched `gkrotate/seed47904/gain_120` at identical margin is not — same
  margin, opposite outcome, i.e. genuine coin-flip behaviour near m ≈ 0
* everything at ×1.50 with rank 1 and all ×2.25 arms adopt (6/6)

### B3. Eviction is not the min-margin target (because margins are tied)

102 multi-target arms; 59 adopt everything, 43 drop ≥1 target, 35 drop
exactly one.

* dropped == min-margin target: **8/35 = 22.9 %**, chance = 22.5 %. Null.
* dropped set == bottom-|D| by margin: 8/43 = 18.6 %.
* dropped == max-`need` target (the quietest target on the substrate before
  compilation, the only per-target quantity with real variation):
  **3/35 = 8.6 %** — *below* chance; `log need` adds nothing to the model
  (χ² = 1.43, p = 0.23).

Which target dies is therefore **not** explained by anything in the T_k
spectrum of the compiled init. Combined with the strong K effect, the natural
reading is a *capacity* constraint (a K=5 committee is harder to sustain
regardless of how loudly it is written in) plus an unmodelled selection among
equals.

---

## Joint picture: one ladder, two failures

| regime | n | head spread (nats) | mean Jaccard | exact | membership AUC |
|---|---|---|---|---|---|
| double-flat (T_k exactly flat) | 8 runs | 6.6e-09 | 0.061 | 0/8 | **0.557** |
| orth-flat | 54 runs | 0.119 | 0.094 | 0/54 | 0.622 |
| natural-normal | 8 runs | 0.162 | 0.137 | 0/8 | 0.758 |
| surgical (spiked inits) | 26 runs | 0.278 | 0.225 | 0/26 | 0.737 |
| compiled s=3 | 30 arms | 1.10 | 0.841 | 43 % | (1 by construction) |
| compiled s=6 | 24 arms | 1.79 | 0.856 | 50 % | — |
| compiled s=12 | 24 arms | 2.48 | 0.923 | 71 % | — |
| compiled s=24 | 24 arms | 3.18 | 0.902 | 58 % | — |

Predictability rises monotonically with spectral spread across ~9 orders of
magnitude of gap, from chance at exact degeneracy to 43–71 % exact-set
dictation once the spread exceeds ~1 nat. That ladder is the honest positive
result, and it is only visible because the compiled arms manufacture a gapped
regime the natural data never reaches.

What fails:

1. **The sharp gap-at-the-cut claim.** Within the natural band the cut gap is
   uninformative (AUC 0.47–0.60, ρ ≈ 0.09 n.s.); only the coarse head spread
   correlates, and it does not distinguish "the race at the cut" from "how
   spiky the init is overall".
2. **Twin determinism.** Gap magnitude does not predict how reproducibly one
   init lands on the same committee across recipes (ρ = −0.05, n = 8), and the
   most deterministic init has the smallest gap.
3. **Margin collapse.** Margin cannot absorb K — it is not even a competitive
   covariate against (log s, K). The compiled design makes margin ≡ log s,
   so the collapse hypothesis is only testable across cells, and across cells
   it is refuted: ΔAIC +27 for adding K, p = 0.36 for adding margin on top of
   (s, K).
4. **Eviction = weakest link.** At chance, because the compiler ties all
   targets exactly. The earlier set-level "eviction = weakest 100 %" result
   applies to natural committees with dispersed amplitudes, not to compiled
   arms.

Design implication if this line continues: compile arms with **deliberately
unequal per-target margins** (e.g. targets at s, s·1.3, s·1.8 within one arm)
and at **doses spanning ~1.0–3.0** rather than 3–24. That is the only way to
separate margin from dose, to test eviction=min-margin non-vacuously, and to
place the adoption curve where it actually bends — the promotion arms say
that is near margin 0, three-fold below the weakest compiled cell.
