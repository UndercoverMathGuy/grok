# Circuit compiler

Writes a chosen frequency committee into an epoch-0 init, so training
"develops" a pre-specified circuit. The compiled variable is the claim-1
ticket T_k (OV-transmitted embedding energy); the compiler edits W_E in
Fourier space until the target set tops the T_k profile with a safety
margin, verifies the spec in closed form (no training), and pre-registers
the predicted committee before any arm trains.

## Pipeline

```sh
uv run python compiler/make_arms.py            # compile ckpts + manifest (fast)
nohup uv run python compiler/train_arms.py \
    > compiler/arms/phaseA_train.log 2>&1 &    # detached, batched M=8, ~2h
uv run python compiler/score_arms.py           # exact-set table vs predictions
```

Outputs: `compiler/arms/ckpts/*.safetensors` (compiled inits),
`phaseA_manifest.json` (arms + pre-registered predictions),
`runs_compiler/phaseA/<base>/<set>/` (standard run artifacts),
`phaseA_scores.json`.

## Phase A design (reliable mode)

2 natural p=113 bases x 3 strictly-feasible K=4 target sets, flat
substrate (QR-orthonormalized W_E — the proven-grok-safe orthWE
construction), energy route, safety s=3; plus 1 flat control per base.
Strict feasibility = no additive pair relations inside S (harmonics
included) and LP max-min margin percentile >= 40 vs random same-size sets.

Pre-registered predictions:
- **P-C1** every compiled arm's final committee == its target set (exact,
  unified detector). Success criterion: >= 5/6.
- **P-C4'** any unpredicted recruit is the highest-T_k background frequency
  of that arm's compiled init (recorded in the manifest).
- **P-C5** compiled arms grok (test acc >= 0.99) on the normal schedule.
- Controls re-roll the lottery (committee != any target set); their T_k
  leaders are recorded at compile time as the weak-form readout prediction.

Later phases (built into `core.py`, arms not yet generated): rotation route
(`route="rotate"`, energy-invariant to <1e-7 — invisible to energy
statistics), stealth budget (`energy_cap`), natural substrate, gauge test
(energy-route vs rotate-route to the same T_k spec).

## Notes

- Training is MLX lockstep-batched (`grok.batched`, M<=8, fast_loss);
  sequential batches only — Metal crashes under concurrent GPU processes.
- `runs_compiler/` is a separate dataset root; nothing under
  `semifinal/analysis` discovers it (those scripts own `runs_torch/`).
- All arms share the vanilla recipe; batched training requires identical
  optimizer/epoch settings across a batch (asserted).
