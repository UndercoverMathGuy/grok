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

## Phase C — the content dissociation on good runs (PREREG, 2026-08-16)

248 arms, generated and pre-registered before any training. Phase B is
retained only as the retrospective; Phase C is what the paper reports.

### Why Phase B could not answer the question

Three defects, the first fatal:

1. **Content was never crossed with dose.** `make_sweep` draws its target
   set inside the innermost loop, so all 96 Phase B sets are distinct and
   the `(K, set_i)` label used as the identity factor maps to 8 different
   frequency sets across the dose x base cells. The permuted factor
   carried no content; its null was true by construction. The reported
   `p = 0.93` is retracted (CLAIMS.md C3).
2. **Off-regime.** Phase B groks at median 400 epochs against 5100-13400
   (median ~8250) for the same 8 natural bases — grokking is substantially
   abolished in the cohort carrying the claim.
3. **Substrate and footprint.** QR-flattened W_E, and 2 base inits drawn
   from a single data mask (seed3148).

Rebuilding the analysis (`analyze_sside.py`, run on Phase B) recovers the
conclusion from nine genuinely-varying content contrasts, all equivalent
at a 15% band with MDE 12%. That fixes defect 1's damage retrospectively.
It cannot fix 2 or 3 — only new runs can.

### Design

```sh
uv run python compiler/make_arms.py --crossed 1.5,3,6 --bases 8 \
    --n-contents 8 --k 4 --epochs 30000 --rng-seed 20260816
```

**S-side (200 arms).** ONE shared pool of 8 strictly-feasible K=4 contents,
drawn once, installed at every dose on every base — content identity is a
real factor with 8 observations per level (24 cells x 8). Substrate
`natural`: the base's own W_E draw is kept, so the init is a real natural
init with a tilt added (spec met 192/192; global renorm makes the achieved
margin equal the dose exactly). All 8 natural p=113 bases — 2 data masks x
4 init seeds. Doses s in {1.5, 3, 6}; the injected perturbation
||dW_E||/||W_E|| is 0.12/0.30/0.59 and varies only ~2x across content at
fixed dose, so "holding strength fixed" is meaningful and the realised
perturbation is also recorded per arm as a continuous covariate. One
no-install control per base anchors the cohort's own natural t_gen, so the
acceleration the install buys is measured inside the cohort rather than
against another one. Spectra every 25 epochs (Phase B used 100 on a median
t_gen of 400 — seven distinct t_gen values in the whole cohort).

**L-side (48 arms).** Data reveal frac_train 0.3 -> 0.7 at donor epoch
t in {1000, 2000, 4000}, resuming from the bases' own `save_every=1000`
checkpoints — so the donors are genuine mainline natural runs at wd = 1.0,
not the `boost/` wd = 3.0 donors of which 2 of 8 never reached the
threshold (effective n was 6). Same data_seed: the split is a seeded
shuffle truncated at frac*N, so the larger training set is an exact
SUPERSET (verified: 0 examples change side, 5108 revealed, 2.33x). Every
jump time carries its own self-restart gate, not just the last.

### Pre-registered predictions

S-side, scored by EQUIVALENCE against a band declared before training
(15% of median t_gen), every statistic expressed in epochs so the observed
effect, the permutation noise ceiling and the band share one scale:

- **P-CS1** content identity (T1) equivalent — the test Phase B could not
  run. A detected identity effect falsifies the strength-channel form of
  C3 and is reported as such.
- **P-CS2** the six content coordinates and the Mantel content-similarity
  test all equivalent.
- **P-CS3** MDE below the band. If not, the cohort is declared
  underpowered for an absence claim and the analyzer says so — no absence
  claim is made on a non-significant p-value.

L-side:

- **P-CL1** median J(jump@4000, donor) >= 0.8 with gate@4000 >= 0.8. A
  failing gate VOIDS its jump arm (a cold restart that re-runs selection
  by itself cannot tell us what the reveal did). A low J with a clean gate
  is a first-class finding — lock-in is data-conditional even post-lock —
  not a failure to hide.
- **P-CL2** median absolute t_gen(jump@4000) <= 0.7 x donor t_gen.
- **P-CL3** median J(jump@t) non-decreasing in t (lock hardening).

No re-rolling of seeds, doses, contents, jump times or bands after
results. Both analyzers (`analyze_sside.py`, `analyze_lside.py`) were
committed before training and run unchanged.

### Running it

```sh
# on a CUDA box; base epoch-0 and donor checkpoints come from git LFS,
# compiled inits are regenerated on the pod (deterministic, seconds)
uv run python compiler/make_arms.py --crossed 1.5,3,6 --bases 8 \
    --n-contents 8 --k 4 --epochs 30000 --rng-seed 20260816
nohup python -u cloud/train_compiler_arms.py \
    --manifest compiler/arms/phaseC_manifest.json \
    --spectra-every 25 --batch 8 > phaseC.log 2>&1 &
uv run python compiler/analyze_sside.py
uv run python compiler/analyze_lside.py
```

`--kind sside` / `--kind lside` runs one side alone. The runner groups
lockstep batches by `(frac_train, num_epochs)`, which Phase C needs
because the L-side mixes fracs and `train_batched` asserts they agree.

## Notes

- Training is MLX lockstep-batched (`grok.batched`, M<=8, fast_loss);
  sequential batches only — Metal crashes under concurrent GPU processes.
- `runs_compiler/` is a separate dataset root; nothing under
  `semifinal/analysis` discovers it (those scripts own `runs_torch/`).
- All arms share the vanilla recipe; batched training requires identical
  optimizer/epoch settings across a batch (asserted).
