# De-grok — the two clocks in ordinary training

Adopted 2026-08-15. Everything so far frames the two-process result inside
grokking (frac_train 0.3, scarce data, long memorize->generalize gap). If
the selection clock is a fact about TRAINING rather than about grokking,
it must survive data abundance: at frac_train 0.7-0.9 mod-add generalizes
almost immediately — no grokking plateau — and the claim becomes "circuit
selection is a distinct early phase of training in general; grokking never
mattered, it just stretches the consolidation clock far enough that the
two processes separate to the naked eye." Grokking demotes from subject to
microscope.

Known confound, designed around: the training mask partially shapes which
committee wins (mask-popularity result, 0.69 vs 0.30 overlap), and
changing frac_train changes the mask wholesale. So NOTHING here predicts
"same committee across fractions" — the predictions are about the CLOCK
(t_ident) and the READABILITY (epoch-0 tilt AUC), which the mask result
does not touch. Committee overlap across fractions is reported as
exploratory only.

This cohort also carries the mandatory replication duties at frac 0.3:
fresh seeds (new META_SEED) and a fresh prime (p=127 batch), replicating
the wd-invariance of selection time and epoch-0 readability out-of-sample.

## Design

Width 512, d_model 128 mainline. FRESH seeds: META_SEED=20260815,
2 data masks x 4 init seeds (drawn, not picked; data seeds from
[20000, 30000) — disjoint from every previously spent mask range).
Within a prime, the SAME init seeds are used at every (frac, wd) — init
does not depend on either, so cells are init-matched twins.

- **frac sweep (p=113):** frac_train in {0.3, 0.5, 0.7, 0.9} x wd in
  {1.0, 3.0} x 8 seed pairs = 64 runs. One lockstep batch per frac
  (train-set size changes tensor shapes; wd is per-run within a batch).
- **prime anchor (p=127):** frac 0.3 x wd {1.0, 3.0} x 8 = 16 runs.

10000 epochs, spectra every 5, checkpoints every 1000 (epoch 0 carries
the tilt readout). 80 runs total, ~30 min on a 5090.

Driver: degrok/train_degrok.py (idempotent). Analyzer:
degrok/analyze_degrok.py (numpy+safetensors, runs locally).

## Metrics (per run, as in dial/race_analyze.py)

- **t_ident**: first epoch with logit-spectrum AUC >= 0.9 for the run's
  own final committee, sustained 2 consecutive snapshots.
- **t_gen**: first snapshot with test acc >= 0.99 sustained 2 snapshots
  (at frac 0.3 this is the grok time; at 0.9 it is just learning time).
- **tilt_auc0**: Mann-Whitney AUC of epoch-0 W_E per-frequency energy
  against the run's own final committee.
- Exploratory: committee size K, cross-frac committee Jaccard for matched
  (mask-half, init) — reported, never scored.

## Pre-registered predictions (committed before training)

Baselines going in (512/wd1/frac0.3, prior cohorts): t_ident median ~20-30,
t_gen ~5-6k, tilt_auc0 ~0.73.

- **P-G1 (selection clock ignores data abundance):** median t_ident at
  (frac 0.9, wd 1) <= 3x median t_ident at (frac 0.3, wd 1).
  **KILL if >= 5x** — selection dynamics are scarcity-dependent and the
  de-grok headline dies (paper stays grokking-scoped).
- **P-G2 (wd-invariance replicates at every fraction, fresh seeds):**
  within each frac, |median t_ident(wd3) - median t_ident(wd1)| <= 25
  epochs, while median t_gen(wd1) / t_gen(wd3) >= 2 at frac 0.3.
- **P-G3 (the dissociation, scored):** across fracs at wd 1, median t_gen
  varies by >= 5x (max/min over fracs) while median t_ident varies < 3x.
- **P-G4 (readability survives abundance):** tilt_auc0 cell means >= 0.65
  at every frac (both wd).
- **P-G5 (prime anchor):** P-G2's wd-invariance holds at p=127
  (|median diff| <= 25 epochs), and tilt_auc0 >= 0.65 there.
- **Branch, pre-declared:** if t_gen <= t_ident at frac 0.9 (the clocks
  cross — the network generalizes before the committee stabilizes), that
  is reported as "the consolidation clock compresses below the selection
  clock while selection is unchanged" — it supports the two-clock law and
  kills only the "selection strictly precedes use" phrasing.

No re-rolling of seeds, fractions, thresholds, or detectors after seeing
results. Runs that fail t_gen within 10k epochs are reported and excluded
pairwise from t_gen tests only (t_ident and tilt_auc0 need no grok at
frac >= 0.5; at frac 0.3 the committee label needs generalization, so
label-less runs are excluded there as in the race analysis).

## RESULT (2026-08-15, 80/80 runs) — ALL FIVE PREREGISTERED PREDICTIONS PASS

First fully confirmatory cohort of the two-clock program: fresh seeds
(META_SEED=20260815), fresh prime anchor, predictions committed above
before training. Full numbers: notes/degrok_summary.json.

- **P-G1 PASS:** t_ident median 12.5 at frac 0.9 vs 20 at frac 0.3
  (ratio 0.625 — selection is if anything slightly FASTER with abundant
  data; bar was <= 3x, kill was >= 5x).
- **P-G2 PASS:** wd-invariance of t_ident at every fraction (|median
  diff| 2.5-15 epochs) while wd moves t_gen ~3x at frac 0.3.
- **P-G3 PASS (the de-grok result):** across fractions at wd 1, t_gen
  spans 7058 -> 198 (**35.7x**) while t_ident spans 20 -> 12.5 (1.6x).
  The consolidation clock is data-controlled; the selection clock is not.
- **P-G4 PASS:** epoch-0 tilt AUC 0.65-0.77 in every cell — the winner
  stays readable from the untrained weights regardless of data abundance.
- **P-G5 PASS:** p=127 anchor replicates wd-invariance (40.0 vs 42.5)
  and readability (0.66) on a fresh prime.
- Branch not triggered: clocks never cross (even at frac 0.9, t_gen ~198
  >> t_ident ~12.5).
- Exploratory: cross-frac committee J ~0.17-0.20 — changing the training
  set changes the winner (as the mask-popularity result predicted; this
  is why committee identity across fractions was never scored).

Headline consequence: the two-clock structure is a property of TRAINING,
not of grokking. Grokking is the microscope — scarce data stretches the
consolidation clock ~36x while leaving the selection clock (and its
init-readability) untouched.

## Post-run addenda (declared before results were seen)

Dense-checkpoint donors for the transplant/handoff experiment are NOT in
this cohort (save cadence too coarse); that experiment gets its own
prereg and a small dedicated donor cohort.
