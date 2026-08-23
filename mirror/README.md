# Mirror — the S-side content test, matched by symmetry (DRAFT prereg)

STATUS: DRAFT 2026-08-16 — commit before training; nothing trained yet.

## Why this design

Every prior S-side test (Phase B, planned Phase C) tries to prove a null
over hand-picked content coordinates (mean frequency, span, LP margin,
...) and dies by the audit's own criticisms: absence-of-evidence scoring,
MDEs above the band, arbitrary coordinates, coarse t_gen grids. This
cohort replaces the coordinate game with an exact symmetry of the task.

Mod-p addition is isomorphic to itself under token relabeling
x -> g*x mod p (g coprime to p). The relabeling maps frequency k to
fold(g*k) = min(gk mod p, p - gk mod p), a bijection on {1..56}. Hence
for any committee C, the mirror committee gC is guaranteed:

- genuinely different content (g chosen so gC and C are disjoint), and
- intrinsically IDENTICAL: same internal additive structure, same
  violations (relations map k1+k2=k3 -> g k1+g k2=g k3), same LP margin
  (the relabeled solution witnesses it), same size.

Installing C vs gC on the SAME base init, SAME mask, SAME dose therefore
varies content while every intrinsic content property is matched by
mathematics, not by covariates. The test is PAIRED, which removes base
and dose variance — this is where the power comes from.

The equivalence band is EMPIRICAL, not declared: an exact-isomorphism
twin arm (permute the vocab columns of an installed init by pi_g AND
train on the pi_g-image of the mask) has mathematically identical
dynamics in exact arithmetic — in float it differs only through
reduction-order epsilons (CE's vocab-axis logsumexp order and the
train-batch enumeration order; the vocab permutation touches no other
reduction in this architecture: attention softmax is over positions,
there is no LayerNorm, matmul reductions are over d_model). The twin's
|dt_gen| is the chaos noise floor of the training process itself.

## Design

p=113, the 8 natural bases with epoch-0 checkpoints (2 data masks x 4
init seeds, mainline recipe, wd 1.0), 30k epochs, spectra every 5
(t_gen resolution 5 ep — the Phase B grid defect is gone), ckpt at 0.

- MIRROR_SEED-drawn shared pool: 3 strictly-feasible K=4 contents
  (violations-free, LP floor 40) drawn ONCE (crossed factor, the Phase C
  fix). One g for the whole cohort, chosen (smallest such g) so that all
  3 mirrors are disjoint from their originals and violations-free
  (asserted; symmetry guarantees it, the assert is a belt).
- Installer: `compiler.core.compile_init`, natural substrate, energy
  route, renorm="global" (achieved margin == dose exactly), dose s=3.
  Injected energy differs per (content, base) and is recorded as a
  covariate from the compile report.

| arm | count |
|-----|-------|
| install C_i, base b, own mask | 8 x 3 = 24 |
| install gC_i, base b, own mask (mirror) | 8 x 3 = 24 |
| control: base b untouched (natural anchor) | 8 |
| twin: pi_g(install C_0 ckpt), mask pi_g(M_b) (noise floor) | 8 |

64 runs. ~2M run-epochs — one short 5090 session; smoke runs locally.

## Pre-registered predictions (bars set before training)

Metrics: t_gen = first test-acc >= 0.99 sustained 2 snapshots (5-ep
grid); committee = largest-gap detector on the final snapshot;
rel-dev of a pair = |t_gen_1 - t_gen_2| / mean(t_gen_1, t_gen_2).

- **P-S1 (install gate):** realized committee is a subset of the
  installed set in >= 80% of the 48 installs (natural substrate at s=3
  is untested; Phase B flat-substrate precedent is 96/96). Pairs with
  either member failing the gate are excluded PAIRWISE from P-S2 and
  listed. If < 60% pass, the cohort is VOID for content claims and
  reported as "natural-substrate installs do not take at s=3".
- **P-S2 (mirror blindness — the claim):** over the surviving (base,
  content) pairs, the bootstrap 95% upper bound of the median rel-dev
  |t_gen(C) - t_gen(gC)| is <= BAND, where
  BAND = max(2 x twin-floor median rel-dev (P-S3), 2 grid steps / median
  t_gen). A genuine CI-based TOST — no p >= 0.05 scoring.
  **Pre-declared detection branch:** if the 95% CI of the SIGNED paired
  median excludes 0, a content effect exists — reported as a first-class
  falsification of the strength-channel form of C3, not a failure.
- **P-S3 (noise floor):** twin median rel-dev is reported and sets the
  band; twin final committee equals the g-image of its original's
  committee in >= 6/8 (doubling as the run-to-run instability floor
  measurement the audit asked for).
- **P-S4 (anchor):** all 8 controls generalize within 30k; install
  acceleration is reported within-cohort (no cross-cohort baselines).

Exploratory (never scored): paired signed dt_gen vs paired difference in
injected energy (does the strength covariate explain any residual);
committee overlap between realized C-arm and mirror-arm images.

No re-rolling of seeds, contents, g, dose, bands, or detectors after
results. Failures report under their branch letters.

## Files

- mirror/make_mirror.py — draws contents, picks g, compiles the 48
  installed inits + 8 twins (+ twin masks), writes mirror_manifest.json.
  `--smoke` verifies: mirror feasibility identity, permutation algebra
  (round-trip, norm preservation), twin epoch-0 logit spectra equal the
  g-permuted originals.
- mirror/train_mirror.py — manifest-driven lockstep training (local CPU
  or pod). `--smoke` trains an install arm and its twin 150 epochs in
  one batch and reports the float-divergence growth curve.
- mirror/analyze_mirror.py — gate, paired TOST, floor, verdicts ->
  notes/mirror_summary.json.
