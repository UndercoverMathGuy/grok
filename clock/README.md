# Clock separation — delaying the selection circuit by freezing its substrate

Adopted 2026-08-15 after the race analysis (dial/race_analyze.py,
notes/race_summary.json — post-hoc): committee selection completes in the
first ~20-80 full-batch steps, is invariant to the weight decay that moves
grok time 30x, and is increasingly readable from the untrained network with
width. That gave one direction of a double dissociation for free: **wd
moves consolidation, not selection.** This experiment supplies the other
direction by tampering with selection's TIMING only: pause the substrate
selection runs on (W_E, knockout-proven locus of the init ticket) and see
whether the selection clock — and only the selection clock — shifts by the
pause length.

Explicitly rejected designs (user call, 2026-08-15): early-lr stretch
(rescales the global time unit — proves nothing about a separate clock);
init-scale dial and noise windows (tamper with the OUTCOME of selection,
which the compiler/kill-switch work already covers). The point here is
timing: if selection can be delayed by a chosen amount while consolidation
runs on schedule, the two are distinct processes and selection comes first.

## Design

One lockstep batch, width 512, wd 3.0 (grok ~1.2-1.9k epochs), 8000
epochs (headroom for delayed grok under serial gating), spectra every 5,
checkpoints every 500 (epoch 0 carries the T_k readout; mid-freeze ckpts
carry the frozen-W_E manipulation check). Same 8 seed
pairs as the dial/race cohorts (META_SEED=20260814: 2 data masks x 4 init
seeds) so every run has natural twins in runs_dial (50k) and runs_race
(fine-grained early). 7 arms x 8 runs = 56:

- **base**: no gating (in-cohort baseline, same batch).
- **fzN** (N in {50, 200, 500}): embed.W_E completely untouched for epochs
  < N — no Adam step, no moment accumulation, **no weight decay** (a
  frozen tilt must not decay) — then normal training. Everything else
  trains normally throughout.
- **onN** (N in {50, 200, 500}): the mirror — ONLY embed.W_E trains for
  epochs < N; every other parameter is untouched; then normal training.

Trainer support: `freeze` argument to `train_batched`
(cloud/train_semifinal_torch.py), per-run gating masks applied inside the
same verified update; `freeze=None` path is bit-identical to before
(equivalence asserted in the driver's --smoke).

## Metrics (per run)

- **t_ident**: first epoch with logit-spectrum AUC >= 0.9 for the run's
  own final committee, sustained 2 consecutive snapshots (as in
  dial/race_analyze.py). Secondary: same vs the base twin's committee.
- **t_grok**: first snapshot with test acc >= 0.99.
- **J_base**: Jaccard(final committee, base twin's final committee).
- Manipulation checks: in fz arms, W_E at the first post-freeze checkpoint
  equals the epoch-0 W_E exactly; train-acc curve during [0, N] tracks the
  base twin (memorization does not run through W_E updates). In on arms,
  train acc stays near chance until N (nothing else can learn).

Baselines going in (512/wd3, n=8, from runs_race join): t_ident mean ~31
(median ~20-30), t_grok mean ~1562. Paired deltas are computed per seed
pair against the in-cohort base arm.

## Pre-registered predictions (committed before training)

- **P-T1 (selection is delayed, dose-response):** paired Δt_ident =
  t_ident(fzN) - t_ident(base) satisfies median Δt_ident >= 0.5·N for
  N=200 and N=500, and Δt_ident increases with N (Spearman over per-run
  points, permutation p<0.05).
  **KILL:** median Δt_ident(fz500) < 125 (0.25·N) — selection is not
  clocked by W_E updates; the model revises to a downstream readout race
  over a static W_E spectrum, and the lazy-selection paper loses its
  substrate claim.
- **P-T2 (identity survives the pause):** median J_base(fz500) >= 0.6 and
  above the shuffled-pair null (permutation p<0.05). The tilt sat
  untouched in the freezer; the same winner should win late.
  Failure is reported as: delay scrambles identity (selection delayed but
  not init-carried — weakens, does not kill).
- **P-T3 (clock relationship — branch, both outcomes reported as THE
  ordering result, neither kills):** compare Δt_grok(fzN) to Δt_ident(fzN)
  at N=500. (i) median Δt_grok <= 0.5·median Δt_ident → independent
  clocks (consolidation does not wait). (ii) ratio in [0.75, 1.25] →
  serial gating: grokking WAITS for selection — the direct "comes before"
  proof. Intermediate → partial coupling, reported with the ratio.
- **P-T4 (mirror dissociation):** in on500, median Δt_grok >= 250 while
  median Δt_ident(on500) <= 100: consolidation is delayed by pausing its
  substrate while selection (running on live W_E) stays near schedule.
  **Caveat pre-declared:** t_ident in on arms is measured through frozen
  random downstream weights; if the logit readout is too dim to cross 0.9
  during [0, N], the fallback readout is AUC at epoch N vs the run's final
  committee (>= the base twin's epoch-5 value counts as "selection ran").

No re-rolling of seeds, arms, thresholds, or detector settings after
seeing results. Non-grokking runs (within 8000 epochs) are reported and
excluded pairwise; if an arm loses >= 4/8 pairs to non-grok, that arm's
predictions are declared unmeasurable at this budget, and the run is
extended once to 16k epochs before judging.

## Files

- `clock/train_clock.py` — driver (idempotent, one lockstep batch of 56,
  RUNS_DIR=runs_clock; --smoke = CPU mechanics + gating-equivalence
  asserts).
- `clock/analyze_clock.py` — numpy-only analyzer, paired stats, writes
  notes/clock_summary.json.
