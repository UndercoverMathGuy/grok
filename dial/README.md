# Capacity dial — when do circuits have identities?

Adopted 2026-08-14 after the induction-pivot closure. The paper's master
claim: **circuit identity is crisp exactly when candidate circuits compete
for scarce capacity, and in that regime the winner is set at init (readable
via T_k, writable via the compiler).** Overprovisioned networks dissolve
identity into a redundant share-vector — the regime documented at LLM scale
as head redundancy (Michel et al. 2019, Voita et al. 2019) and self-repair
(the hydra effect, McGrath et al. 2023). Both ends are already measured in
this repo: mod-add at d_mlp=512/wd=1.0 is crisp (K~4-6, T_k AUC ~0.75,
compiler 97% exact); 2L attention-only induction with free head capacity is
smeared (k_eff flat across 100x wd, `induction/README.md`). This experiment
fills in the middle of the dial **within one system and one instrument**:
a d_mlp x weight-decay grid on mainline mod-add.

Theory stakes: Morwani et al.'s max-margin analysis predicts DENSE frequency
use (all ~56) when capacity is unconstrained, while every empirical mod-add
committee at standard width is sparse — the sparse-vs-dense tension flagged
in the 2026-08 uniqueness audit. The dial measures the interpolation between
the empirical sparse regime and the dense limit, resolving that tension as a
capacity effect rather than a contradiction.

## Design (phase 1: natural sweep)

Grid: d_mlp in {128, 256, 512, 1024, 2048, 4096} x wd in {0.1, 0.3, 1.0,
3.0}; 8 runs per cell = 2 fresh data masks x 4 init seeds, drawn from
META_SEED=20260814 (data seeds from [10000, 20000), disjoint by construction
from every spent mask). Everything else mainline: p=113, frac_train 0.3,
d_model 128, 4 heads, lr 1e-3, AdamW (MLX semantics), 50k epochs, torch
lockstep trainer (`cloud/train_semifinal_torch.py`, verified == MLX).
Within a width, the SAME init seeds are used at every wd — init_params does
not depend on wd, so wd cells are init-matched twins and identity
persistence along the wd axis is directly measurable. Checkpoints at epoch
0 / 25k / 50k (epoch 0 carries the T_k readout), spectra every 100.

Total 192 runs. Driver: `dial/train_dial.py` (idempotent, one lockstep
batch per width, per-run wd inside the batch). Analyzer:
`dial/analyze_dial.py` (numpy-only, runs locally on pulled data).

## Metrics

- **k_eff** — participation ratio (sum E)^2 / sum E^2 of the final logit
  Fourier energy over the 56 frequencies. Primary crispness measure,
  detector-free (lesson from the induction pilot: gap detectors produce
  artifacts on flat profiles).
- **top4_share** — fraction of final logit Fourier energy in the top 4
  frequencies. Secondary crispness.
- **committee** — `committee_from_coeffs` gap detector, used ONLY as the
  membership label for readout AUC, and only meaningful in crisp cells.
- **grok** — final test acc >= 0.99; grok epoch from spectra.
- **readout AUC** — Mann-Whitney AUC of epoch-0 T_k (W_E per-frequency
  energy; OV-transmitted variant secondary) against final committee
  membership, per run, averaged over groked runs per cell.
- **persistence** — Jaccard(committee | same width, same data+init seeds)
  between the wd=1.0 cell and each other wd.

## Pre-registered predictions (committed before training)

- **P-D1 (width axis):** at wd=1.0, k_eff rises monotonically with d_mlp
  (Spearman over groked cell means, p<0.05, permutation test). Direction:
  toward the Morwani dense limit.
- **P-D2 (wd axis):** at d_mlp=512, k_eff falls monotonically with wd among
  groked cells (same test). This is the membership-tax economics the K*
  capacity law fit at fixed width.
- **P-D3 (readability follows crispness):** across all cells with grok rate
  >= 50%, readout AUC correlates negatively with cell-mean k_eff (Spearman
  p<0.05). At the smeared end membership prediction approaches chance
  because membership approaches "everyone".
- **P-D4 (exploratory, non-gating):** cell-mean k_eff collapses onto a
  single curve in a scalar combination of (d_mlp, wd) — candidate x-axis
  d_mlp/wd on log axes. Reported either way; no pass/fail attached.
- **P-D5 (phase 2, steering along the dial):** at three settings chosen
  from the phase-1 curve (crisp / mid / smeared), compiler dose-edits at
  matched dose steer the committee with hit-rate falling monotonically
  along the dial. Phase 2 gets its own prereg (settings, doses, targets,
  n) committed BEFORE its arms train — nothing about it is fixed here
  except the monotone-decline prediction.

## Failure semantics / kill criteria

- Cells that fail to grok (rate < 50%) are excluded from trend tests but
  reported in full — high-wd non-grokking is a boundary datum (over-taxed),
  not a failure of the dial.
- If groked cells span < 3 values on an axis, extend the missing cells to
  100k epochs ONCE before judging that axis; if still < 3, the axis is
  unmeasurable at this budget and is reported as such.
- **Kill:** if k_eff over groked cells is flat on BOTH axes (neither P-D1
  nor P-D2 significant, and cell means within ~15% of the grand mean), the
  dial claim is dead and the paper falls back to the corrected four-claim
  skeleton without the law figure (compiler read/write + kill switch +
  induction boundary). No re-rolling of grids, seeds, or crispness metrics
  after seeing results.
- Prediction conflicts (e.g. P-D1 lands but P-D3 fails: identity smears but
  stays init-readable) are findings, reported as such.

## Anchors from prior cohorts (for sanity, not scored)

d_mlp=512 / wd=1.0 naturals: committees K~4-6, k_eff ~4-7 expected,
T_k AUC ~0.75 (n=8 natural cohort), grok by ~5k epochs. The dial's
(512, 1.0) cell should reproduce these on fresh seeds.
