# SEMIFINAL reproduction package

Companion to `../SEMIFINAL.md` (the plain-language claims document).
Layout:

- `analysis/` — analysis-only scripts that back every claim in that
  document. Each auto-discovers every compatible run under `runs/` (by
  config + artifacts, not hardcoded lists), so newly trained runs are
  picked up automatically on re-run.
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

Fresh captures land in `results/` once the v2 dataset (below) is trained;
the pre-v2 captures are archived in `results/legacy-2026-08-03/`.

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

## Training (regenerating the runs)

**The dataset is built by ONE script:**

- `training/train_semifinal_v2.py` — one script → the full 44-run dataset
  in `runs/`, all-new seeds and masks: two identical cells of 4 natural +
  4 orthWE + 4 double-flat matched triples (masks 4811 and 7207), plus
  the complete steering suite (dose / suppress / gk-rotate / chaos pair /
  collision) duplicated over two fresh bases. Sized so every section of
  every analysis script is answered from these runs alone. Idempotent,
  sequential, ~10 h; kill/resume freely — any prefix is a usable dataset.

The other `training/train_*.py` scripts regenerate the archived pre-v2
cohorts and exist for provenance only — running them would mix legacy-era
runs into `runs/`, so don't, unless that's what you want. Everything is
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
