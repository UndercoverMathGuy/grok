# SEMIFINAL reproduction package

Companion to `../SEMIFINAL.md` (the plain-language claims document).
Layout:

- `analysis/` — analysis-only scripts that back every claim in that
  document. Each auto-discovers every compatible run under `runs_torch/`
  (by config + artifacts, not hardcoded lists), so newly trained runs
  are picked up automatically on re-run.
- `training/` — cleaned training scripts that regenerate the underlying
  runs (see `training/README.md` for order and dependencies).
- `results/` — captured outputs of the analysis scripts.

Run the analyses from the repo root:

```sh
uv run python semifinal/analysis/claim1_readout.py     # ~3 min (forward passes)
uv run python semifinal/analysis/claim1_knockouts.py   # ~10 min (forward passes x scrambles)
uv run python semifinal/analysis/claim2_twins.py       # ~1 min
uv run python semifinal/analysis/claim3_steering.py    # ~1 min
uv run python semifinal/analysis/claim4_floor_repair.py# ~5 min (LP nulls)
```

The current captures in `results/` (2026-08-06) are from the realized v2
dataset in `runs_torch/`; the pre-v2 captures are archived in
`results/legacy-2026-08-03/`.

## The v2 dataset (realized)

98 runs trained on a cloud 4090 by `cloud/train_semifinal_torch.py`
(torch port verified numerically equivalent to the MLX trainer; 50k
epochs each, p=113), landing in `runs_torch/`. Two mask cells
(data_seed 3148 with init seeds 31831/47904/69089/81412; data_seed 9415
with 19078/32610/68549/85697), each cell trained under NINE recipes:
natural (`p-113/`), `orthWE/`, `doubleflat/`, and six orth-flat
dynamics variants (`dyn-lr3`, `dyn-lrlo`, `dyn-wd04`, `dyn-wd25`,
`eff-G` = CVaR, `phase2-tilt`) — 72 runs — plus the steering suite on
two natural bases (seed31831, seed47904): dose x4 each, suppress,
gk-rotate x2, chaos pair, collision trio, and 6 transplants — 26 runs.

Two runs pass the accuracy gate but never consolidated (final test CE
3.8e-2 / 5.5e-3, more than 10x above every other run):
`dyn-lr3/p-113/seed9415/seed85697` and `eff-G/p-113/seed9415/seed19078`.
They are excluded by `analysis/common.py` (`BAD_RUNS`), leaving n=96.

IMPORTANT: run these sequentially, never concurrently — MLX/Metal on this
machine crashes under concurrent GPU processes.

## Claim → script → expected headline numbers

All committee calls use ONE detector (`analysis/common.py`
`committee_from_coeffs`: largest log-gap + 2%-of-max floor; the floor
constant was derived once from the pooled amplitude statistics of the
pre-v2 dataset and is fixed ahead of v2); the same constant is claim 3's
adoption criterion. Statistics are reported per run AND per independent
init cluster (`common.cluster_key`) — cluster-level is primary, since
surgical arms sharing one epoch-0 init are one unit of evidence.

The expected numbers below are PRE-REGISTERED PREDICTIONS for the v2
dataset (written before it trained). The pre-v2 dataset's realized
numbers are archived in `results/legacy-2026-08-03/`.

| SEMIFINAL claim | script | what it computes | pre-registered v2 expectation |
|---|---|---|---|
| 1. Lottery located; ticket = arrival loudness (T_k), two ingredients | `analysis/claim1_readout.py` | 5 epoch-0 committee predictors (emb / T_k / align=T_k÷emb / fwd x2) x every run with an epoch-0 checkpoint, per cohort, run- and cluster-level | natural (n=8): T_k ≈ 0.72, emb ≈ 0.70, align ≈ 0.65; orthWE (n=8, THE decisive test): T_k ≈ 0.66 vs chance 0.5, emb ≈ 0.5; double-flat (n=8): ALL ≈ 0.5 |
| 1 (readout localization — consistency check, not causal proof) | `analysis/claim1_knockouts.py` | agg + per-neuron readouts after scrambling one component (W_E / fresh-frame / attention / W_in), 3 draws each | natural: W_E kills, attn partial; agg W_in row = readout sanity check; neur readout: W_in scramble leaves it intact while W_E scramble kills it; double-flat: nothing to kill |
| 2. Init chooses, not training noise | `analysis/claim2_twins.py` | (A) same-init-across-dynamics twins (v2 has one recipe — section reports from the legacy capture); (B) same-seed across flattening levels, now on TWO cells; (C) paired normal-vs-orth twins | B: same-seed cross-level J ≈ stranger baseline (~0.1) in both cells — every flattening re-rolls the lottery |
| 3. Steering by hand | `analysis/claim3_steering.py` | dose-response on two fresh bases (control = the base run; targets auto-identified), suppression x2, alignment-knob arms x2, chaos pairs x2 | per base: target peak monotone in dose, adoption by 2.25x; suppression evicts the strongest winner; gk arms match energy arms at matched gain; chaos pairs differ (bystander chaos); thresholds may vary by base |
| 4. Floor + repair | `analysis/claim4_floor_repair.py` | LP-margin percentile floor, additive-relation depletion (mask-cluster bootstrap), menu closure (fixed e3000 + half-grok), engineered-trio census — no per-run committee overrides | no freely-trained run below the 25th pctile (forced 2.25x arms are allowed to dip); depletion: final violations well under chance expectation; closure ≈ all runs; collision trios broken 2/2 |

## Realized v2 outcomes (2026-08-06 captures) vs the predictions

- **Claim 1 readout** — natural (n=8): T_k 0.758 (pred 0.72), emb 0.737
  (pred 0.70), align 0.644 (pred 0.65). Orth-flat (n=54 runs, 8
  independent inits — the decisive test): T_k = align 0.620
  cluster-level, p=0.044 (run-level p=6.8e-6) vs pred 0.66, while emb
  collapses to 0.527 ≈ chance exactly as the erasure argument demands.
  Double-flat (n=8): every predictor 0.53–0.60, none significant. All
  three predictions land. Caveat: natural cluster-level p's are weak
  (only 2 mask clusters); run-level carries that cohort.
- **Claim 1 knockouts** — natural: W_E scramble kills agg (0.74→0.48),
  attn partial (−0.11), W_in nothing (+0.004); neur reads the committee
  at baseline (0.68), W_in scramble leaves it intact (−0.007), W_E kills
  it. Orth-flat: W_E, fresh-frame, and attn scrambles ALL kill (→~0.50)
  — the relational carrier. Double-flat: baseline already ≈ chance,
  nothing to kill. Matches the prediction row exactly.
- **Claim 2** — A now runs on FRESH data (6 dynamics families x same 4
  inits per cell, not the legacy capture): within-seed J 0.35–0.37 vs
  across-seed 0.09–0.13, perm p < 1e-4 in both cells at both family and
  recipe-group level. B: same-seed cross-level J 0.139 vs stranger
  baselines 0.090/0.141 — every flattening re-rolls the lottery, both
  cells. Stronger than predicted.
- **Claim 3** — dose-response: seed47904 adopts from 1.20x, seed31831
  from 1.50x (thresholds vary by base, as pre-registered; adoption by
  2.25x in both). Suppression evicts the strongest incumbent 2/2.
  gk-rotate at 2.25x adopts on both bases (energy-free knob works); at
  1.20x neither base adopts while the 1.20x ENERGY arm on seed47904
  does — knob equivalence holds at high gain but is not exact at
  threshold. Transplants: donor-unique kept 1 vs recipient-unique 11 —
  energy copy does not transfer identity. Chaos pairs: seed47904 arms
  diverge ([5,10,11,14,43,45] vs [5,11,14,35]); seed31831 arms land on
  the same committee as each other, both differing from their base
  (+f6) — bystander chaos in one of two pairs.
- **Claim 4** — floor headline is freely-trained runs only (surgical
  arms are init-edited on purpose): 0/70 below the 25th percentile,
  min 43.2. Depletion, run-level on the same 70: 3 runs with >=1
  violation vs 23.5 expected by chance, MC p < 1e-5 (5 total final
  violations vs 72.5 expected over all 96; blind mid-training leaders
  carry 34). The legacy mask-cluster bootstrap was dropped — this
  dataset has 2 mask cells, too few for mask-level inference. Closure:
  93/96 at fixed e3000, 95/96 at half-grok. Engineered collision trios
  broken 2/2. All predictions land.

## Training (regenerating the runs)

**The dataset is built by ONE script:**

- `cloud/train_semifinal_torch.py` — the torch port of the MLX trainer
  (verified numerically equivalent), run on a cloud GPU → the full
  98-run dataset in `runs_torch/` described above, all-new seeds and
  masks. Sized so every section of every analysis script is answered
  from these runs alone.

`training/train_semifinal_v2.py` is the local MLX equivalent of a
44-run subset (kept for provenance / local reproduction). The other
`training/train_*.py` scripts regenerate the archived pre-v2 cohorts
and exist for provenance only — running any local trainer would mix
new runs into the discovered tree, so don't, unless that's what you
want. Everything is
deterministic given the run data; the epoch-0 checkpoints in each run
directory are the exact pre-update inits.

Everything that predates v2 — the old runs, the exploratory lab notebook,
and its logs — is archived under `../legacy/` (not read by any analysis;
pre-v2 captured outputs kept in `results/legacy-2026-08-03/`).

## References

- Nanda, Chan, Lieberum, Smith, Steinhardt — *Progress measures for
  grokking via mechanistic interpretability.* ICLR 2023.
  arXiv:2301.05217. (The model organism and the Fourier algorithm; this
  project answers the frequency-selection question it left open.)
- He et al. — *[two-layer networks on modular addition: per-neuron
  frequency lottery]*, arXiv:2602.16849. (Their per-neuron
  magnitude/phase lottery does not transfer to the transformer — the
  committee signal lives upstream in W_E×OV; even with both carriers
  erased, their variable reads at noise level at n=16.)
- Morwani et al. — *Feature emergence via margin maximization.* ICLR 2024,
  arXiv:2311.07568. (Margin theory for the class-averaged margin over the
  full spectrum; our min-margin floor is a different, empirical object —
  and margin as repair *trigger* was tested and retracted.)
- Varma et al. — *Explaining grokking through circuit efficiency.*
  arXiv:2309.02390. (Efficiency competition one level coarser; related in
  spirit to the amplitude/norm bookkeeping, which we deflated in-house.)
- Ding et al. — *Survival of the fittest representation* (modular-addition
  case study). (Closest prior to the eviction phenomenon; no intrinsic
  set-level selection variable.)
- Liu et al. — *Omnigrok / grokking as delayed generalization* line.
  (Background on grokking dynamics; not selection-specific.)
