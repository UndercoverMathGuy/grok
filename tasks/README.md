# Tasks — do the two clocks and the init lottery exist beyond mod-add-prime?

Adopted 2026-08-15, prereg committed before training. Everything so far
is one task family (modular addition, prime modulus). This arm runs the
same instruments on structurally different tasks:

- **S5 composition** (p=120 elements, label = group product). The
  committee universe is the 6 nontrivial irreducible representations of
  S5 (character-based isotypic projectors, verified exact at import —
  cloud/groups.py). This is the real generality test: committees of
  irreps, not Fourier frequencies. Hooks: Chughtai et al. universality;
  Stander et al. coset circuits.
- **Mod-add composite modulus 96** (not prime, 48 frequencies). Nearly
  free: identical instruments; kills "it's special to primes."

Fresh seeds META_SEED=20260817. Width 512, d_model 128 mainline arch.

Grids:
- S5: frac {0.3, 0.5} x wd {0.5, 1.0} x (2 masks x 2 inits) = 16 runs,
  25k epochs, spectra every 25, ckpts at 0/final. (S5 grokking
  hyperparameters are less charted — the frac axis is the hedge.)
- mod-96: frac 0.3 x wd {1.0, 3.0} x (2 masks x 4 inits) = 16 runs,
  10k epochs, spectra every 5, ckpts at 0/final.

## Metrics

As in degrok/analyze_degrok.py: t_ident (AUC >= 0.9 vs own final
committee, sustained 2 snapshots), t_gen (test >= 0.99 sustained 2),
epoch-0 W_E energy AUC (irrep energy for S5 via projectors; Fourier for
mod-96), committee size K.

## Pre-registered predictions (scored per task, in cells with grok rate >= 50%)

- **P-M1 (committees exist):** committees are sparse subsets — S5 mean
  K <= 4 of 6 irreps; mod-96 mean K <= 10 of 48. A dense committee
  (everything wins) makes the selection question degenerate for that
  task; reported as a boundary datum (cf. induction), not hidden.
- **P-M2 (selection early):** median t_ident <= 0.2 x median t_gen in
  every groked cell.
- **P-M3 (wd grabs the learning clock only):** where a frac has two
  groked wd cells: t_gen(wd_low)/t_gen(wd_high) >= 1.5 while the
  t_ident ratio (same direction) <= 1.25.
- **P-M4 (init readability):** cell-mean epoch-0 W_E energy AUC vs own
  final committee >= 0.6.

Generality verdict: a task "passes" if P-M1..P-M4 all hold there. The
paper's generality claim needs >= 1 of the 2 tasks passing; both
failing scopes the paper to modular addition, reported honestly. If S5
grokks nowhere on this grid (rate < 50% in all cells), the task is
declared unmeasurable at this budget and the grid is extended ONCE
(more epochs / frac 0.7) before judging.

No re-rolling of seeds, grids, thresholds, or detectors after results.

## RESULT round 1 (2026-08-15, 32 runs)

- **mod-96: 3 of 4 predictions PASS.** Grok rate 0.75/1.00. Committees
  sparse (K 4.4-5.4 of 48). Selection ultra-early (t_ident/t_gen
  0.004-0.017). wd dissociation clean (t_gen ratio 3.15 vs t_ident
  ratio 0.73). **P-M4 FAIL by threshold**: epoch-0 tilt AUC 0.568/0.598
  vs the 0.6 bar (composite moduli have degenerate frequency structure;
  reported as the near-miss it is). Two-clock structure: NOT
  prime-specific.
- **S5: no cell groks within 25k epochs** (memorizes, never
  generalizes; grok rate 0/16). Extend-once rule triggered as declared:
  ONE extension at 100k epochs, fracs {0.5, 0.7}, wds {1.0, 2.0}, same
  seeds, RUNS_DIR runs_tasks_ext (running 2026-08-15 evening).

## RESULT round 2 — S5 extension (100k epochs, declared extend-once)

S5 GROKS at frac 0.7 (8/8, t_gen 1.5k-12k; frac 0.5 partial 0.5-0.75
rate, very slow 52-83k). Verdicts on groked cells:

- **P-M1 PASS:** committees sparse and consistent — K=2 of 6 irreps,
  always containing sign, paired with std or std_sign.
- **P-M2 mixed:** early at frac 0.7 (t_ident/t_gen 0.06-0.07) but the
  slow frac-0.5 wd2 cell fails (0.395) -> P-M2 FAIL as registered
  (all-cells criterion).
- **P-M3 FAIL — a real boundary datum:** at frac 0.7 weight decay moves
  BOTH clocks together (t_gen ratio 2.14, t_ident ratio 2.27). The
  wd-invariance of selection time, rock-solid in mod-add (prime and
  composite), does NOT hold in S5. First observed coupling of the two
  clocks to the same knob.
- **P-M4 FAIL as registered (raw energy AUC 0.09-0.27), diagnosed as an
  instrument artifact:** raw isotypic energy scales with irrep rank
  (d^2: 1 vs 36), so small-dim committee irreps look energy-poor.
  Rank-normalized (energy per basis function, post-hoc fix reported
  separately): AUC 0.75-1.00 — the committee is strongly readable from
  the untrained W_E. n small (2-4/cell).

Net: S5 has sparse init-readable committees but breaks selection-time
wd-invariance — generality of the two-clock LAW is carried by mod-96;
S5 bounds it. mod-add-family: clocks separate; S5: partially coupled.

## Engram formation curves (exploratory, lesion/engram_curves.py)

Dense-checkpoint donors (ckpt/10, wd 3): the two copies form
SEQUENTIALLY, not in lockstep — W_E's copy is written during the
selection window (AUC 0.82 -> 0.94 by epoch 30, saturated ~60), the
W_U/downstream copy replicates afterward (flat ~0.70 to epoch 40, then
0.85@60 -> 0.99@100). Write-then-replicate, primary copy in the
embedding, backup downstream — consistent with lesion recovery being
weaker at t=150 than t=500 (backup still being written) and with W_U
carrying init signal too (epoch-0 AUC ~0.68: both vocab-facing factors
are born leaning). Balanced-lockstep growth is refuted as the
redundancy mechanism; the "why" is replication of a settled decision,
not simultaneous growth. notes/engram_curves.json.

**Order-flip test (lesion/engram_fz.py): PASSES cleanly.** Same donors
with W_E frozen for 500 epochs: W_E's AUC is pinned flat at 0.894 (its
static init lean) through the whole freeze; W_U writes its record FIRST
(0.69 -> 0.96 by epoch 100); the moment W_E unfreezes it is written in
under 50 epochs (0.894 -> 0.995 by 550). "First scribe" is contingent
on trainability — whichever factor can move records the decision, and
the other replicates when freed. The boss/copy model is dead; the
scribe/replication model is causally confirmed.
notes/engram_fz_curves.json.

## Files

- tasks/train_tasks.py — driver (idempotent; --smoke: CPU mechanics for
  both tasks incl. S5 label/projector sanity).
- tasks/analyze_tasks.py — verdicts; writes notes/tasks_summary.json.
