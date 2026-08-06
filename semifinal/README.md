# SEMIFINAL reproduction package

Companion to `../SEMIFINAL.md`. This file: the numbers, and how to
regenerate them. Captured outputs (2026-08-06) are in `results/`.

## Results

All predictions below were registered before the dataset trained.

### Init readout — `analysis/claim1_readout.py`

Epoch-0 AUC, predicted committee members vs the rest:

| cohort | n | emb | T_k | OV gain | predicted |
|---|---|---|---|---|---|
| natural | 8 | 0.737 | 0.758 (p=7e-5) | 0.644 | T_k ~0.72 |
| orth-flat (emb erased) | 54 runs / 8 inits | 0.527 ≈ chance | 0.620 (init-level p=0.044, run-level p=7e-6) | = T_k | T_k ~0.66, emb ~0.5 |
| double-flat (both erased) | 8 | 0.50–0.60, none significant | ″ | ″ | all ~0.5 |

### Scramble knockouts — `analysis/claim1_knockouts.py`

Change in epoch-0 readout after scrambling one component at init
(3 draws each; double-flat baseline is already ≈ chance, nothing to
kill):

| component scrambled | natural, aggregate (base 0.74) | natural, per-neuron (base 0.68) | orth-flat |
|---|---|---|---|
| W_emb | → 0.48 (kills) | kills | → ~0.50 (kills) |
| attention | −0.11 (partial) | — | → ~0.50 (kills) |
| embedding frame | — | — | → ~0.50 (kills) |
| MLP input (W_in) | +0.004 (nothing) | −0.007 (nothing) | — |

### Same init, different training — `analysis/claim2_twins.py`

Orth-flat inits, 7 recipes:

| comparison | Jaccard |
|---|---|
| same init, different recipe | 0.35–0.37 (perm p < 1e-4, both cells) |
| different init, same recipe | 0.10 |
| different init, different recipe (stranger baseline) | 0.09–0.13 |
| same seed, different flattening level | 0.14 |
| different seed, same flattening level (stranger baseline) | 0.090 / 0.141 (per cell) |

### Steering — `analysis/claim3_steering.py`

Two fresh bases (seed31831, seed47904):

| intervention at init | arms | result |
|---|---|---|
| energy boost of a doomed frequency | 4 doses × 2 bases | adopted from 1.20× (47904) / 1.50× (31831); by 2.25× on both; peak dose-monotone everywhere |
| energy suppression (0.5× predicted winner) | 2 | evicted, 2/2 |
| rotation: OV fit up, energy change < 1e-7 | 2 gains × 2 bases | adopted at 2.25× on both; at 1.20× neither (1.20× energy arm on 47904 did adopt) |
| energy-profile transplant (donor energies, recipient directions) | 6 | outcome follows recipient: 11 recipient-unique members survive vs 1 donor-unique |

### Committee-set regularities — `analysis/claim4_floor_repair.py`

Supplementary, not a headline claim: 0/70 freely-trained runs below
the 25th percentile of an LP-margin null (min 43.2); 3/70 end with an
additive i±j collision vs 23.5 expected (MC p < 1e-5); implanted
collision trios broken 2/2 (predicted endpoint evicted); final
committees inside the run's own mid-training top-8 in 95/96.

## Replication

Dataset: 98 runs in `runs_torch/` — two mask cells × nine recipes plus
a 26-run steering suite, trained by `cloud/train_semifinal_torch.py`
(50k epochs, p=113). Two never-consolidated runs are excluded by
`analysis/common.py::BAD_RUNS` (test CE 10× above every other run),
leaving n=96. Each run directory's epoch-0 checkpoint is the exact
pre-update init.

Run from the repo root, sequentially (MLX/Metal crashes under
concurrent GPU processes):

```sh
uv run python semifinal/analysis/claim1_readout.py      # ~3 min
uv run python semifinal/analysis/claim1_knockouts.py    # ~10 min
uv run python semifinal/analysis/claim2_twins.py        # ~1 min
uv run python semifinal/analysis/claim3_steering.py     # ~1 min
uv run python semifinal/analysis/claim4_floor_repair.py # ~5 min
```

Scripts auto-discover all compatible runs under `runs_torch/`. One
committee detector everywhere
(`analysis/common.py::committee_from_coeffs`, fixed before v2
trained); statistics reported per run and per independent init. To
retrain the dataset: `cloud/train_semifinal_torch.py`. Pre-v2 material
is archived in `../legacy/` and `results/legacy-2026-08-03/`; no
analysis reads it.
