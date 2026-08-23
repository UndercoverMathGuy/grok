# twin/ — exact-twin experiments on selection vs learning

Replaces the S-side / L-side experiments with **fork-based designs that have no
confound to argue about**: every treatment arm is compared to a control that is
bit-for-bit the same run up to the intervention, trained in the *same* vmap
batch, on the *same* tensor shapes.

Recipe (fixed for all 1400 runs): p=113, `(a+b) mod p`, 1 layer, d_model 128,
4 heads, d_mlp 512, ReLU, n_ctx 3, vocab p+1, AdamW lr 1e-3 (linear warmup 10
steps), wd 1.0, betas (0.9, 0.98), eps 1e-8, fp32, full batch, `META_SEED =
20260816` (all data/init/perturbation seeds drawn from `[40000, 50000)`).

Three engineering choices make the twins exact:

* **Full-grid masked loss.** Every run evaluates all `p^2 = 12769` pairs;
  `loss = sum(ce * train_mask) / train_mask.sum()`. Shapes never depend on the
  train fraction, so revealing/concealing data is a pure mask-bit flip and runs
  at different fractions share one batch.
* **Nested prefix masks.** One permutation per data seed; the last 10% is the
  test set for *every* fraction; `train(f)` is the first `f * p^2` entries.
  `train(0.35) ⊂ train(0.5) ⊂ train(0.7)`, test untouched.
* **Checkpoint = weights + full AdamW state** (`exp_avg`, `exp_avg_sq`, `step`),
  saved *before* the update of epoch `e`, so a fork from `e` replays epoch `e`
  onward and an unedited fork is a bitwise twin (gate G2).

## Cohorts

| cohort | question | design | runs | run-epochs |
|---|---|---|---|---|
| **BASES-A** | — | 2 data × 4 init seeds, frac 0.3, 30k ep, ckpts at T = 0,10,20,40,80,160,320,640,1280,2560 (+end) | 8 | 0.24M |
| **BASES-B** | — | same 8 pairs, frac 0.5, 15k ep, ckpts at T = 10,25,60,150,400 | 8 | 0.12M |
| **FORK** | *when does selection causally lock?* | per base × T × ε∈{0.01,0.03,0.1} × 3 Gaussian draws (globally norm-matched to `ε·‖θ‖_F` over all tensors) + one ε=0 control; continue to 30k | 800 | 23.6M |
| **SWAP** | *is the learning trajectory blind to WHICH committee?* | at t=0: 4 winner↔runner W_E Fourier exchanges, 4 sham (mid-rank non-member ↔ non-member) exchanges, 2 positive controls (halve a winner), + 1 unedited control per base | 88 | 2.6M |
| **DIAL** | *can a single frequency be steered at init?* | at t=0: 6 targets (2 committee members / 2 top non-members / 2 mid-rank) × dose ∈ {0.25,0.5,0.8,1.25,2.0,4.0} on that frequency's W_E cos+sin coefficient vectors, + 1 unedited control per base | 296 | 8.9M |
| **HOLD** | *does mid-run data move learning without rewriting the committee?* | fork BASES-B at T ∈ {10,25,60,150,400}, conceal to 0.35/0.4 or reveal to 0.7/0.9 (mask-bit flip only) + same-data control; 30k budget | 200 | 6.0M |

**1400 runs, 41.4M run-epochs.** At 1000 / 1500 / 2000 batched run-epochs/s that
is **11.5 h / 7.7 h / 5.8 h** of GPU wall clock. FORK is 57% of it.

Exchange and scale edits are done in float64 in the orthonormal vocab-axis
Fourier basis (the `=` token row is never touched) and carry an asserted
**certificate**, checked on the fp32 result that actually trains: for an
exchange, the per-frequency energy profile equals the original with `i, j`
swapped (rel 1e-5), `‖W_E‖_F` is preserved (rel 1e-6), and every other
frequency's coefficient vector is unchanged (≤1e-7 of the coefficient scale).
Observed on a real init: 1.2e-8 / 1.0e-9 / 1.2e-8.

## Determinism, and the G3 result

`gates.py` runs three gates; they are part of the smoke and are re-run on the
GPU before any cohort.

* **G1** repeatability — same config twice, bitwise identical. **PASS**
* **G2** checkpoint/restore — N straight vs N/2 + save + restore + N/2, solo
  *and* batched. **PASS** (this is what licenses FORK and HOLD.)
* **G3** batch invariance — same seeds at vmap widths 1 / 2 / M. **FAIL**, on
  CPU already: vmap dispatches different matmul kernels per batch dim, results
  differ in the last bit, and grokking amplifies 1 ulp into a visibly different
  trajectory within a few hundred epochs. Whether a given (lane, width) pair
  diverges within a horizon is luck, so it is treated as always failing.

**Fallback rule, in force:** each cohort is trained at ONE fixed batch width
(`chunk_jobs` reduces groups-per-chunk to a divisor of the group count so every
chunk of a cohort has the same width), and every treatment arm's control twin —
the ε=0 fork, the unedited SWAP/DIAL arm, the same-data HOLD arm — is trained
**inside the same batch**. `make_twin` puts them in the same `group`, and
`train_twin` never splits a group. What *does* hold, and is all the fallback
needs, is gate `within_batch`: two lanes with identical inputs in one batch are
bitwise identical regardless of lane index.

The analysis therefore uses the within-batch control as the primary reference
and reports the stage-1 base comparison alongside (`chi` vs `chi_base`,
`J_ctrl` vs `J_base`, `ratio_ctrl` vs `ratio_base`), so the size of any
cross-batch offset is visible rather than hidden. `eps0_bitwise_vs_base` /
`ctrl_bitwise_vs_base` count how many cross-width controls happened to
reproduce their base exactly; at matched batch width the reproduction is exact
(verified in the smoke).

**Resuming changes the answer if you change `--width` or `--compile`.** Both are
part of the arithmetic (eager and `torch.compile` fuse differently and differ in
the last bit, which grokking then amplifies exactly as batch width does). Resume
a cohort with the settings it started with, or delete it; each run's
`config.json` records `batch_size`, `compile_mode` and `device`, and
`twin/results/provenance.json` shows what each cohort was actually trained at.

## Running it

Torch is not a default dependency of this repo; it lives in the `convert`
dependency group. Everything runs from the repo root.

```bash
# CPU smoke (p=23, ~45 s) — must pass before anything else
uv run --group convert python twin/smoke_twin.py

# gates on their own
uv run --group convert python twin/gates.py --device cpu --no-compile --epochs 600 \
    --p 23 --d-model 32 --heads 2 --d-mlp 64

# full GPU pipeline: gates -> stage1 -> stage2 manifests -> 4 cohorts -> analysis
bash twin/run_gpu.sh                      # WIDTH=40 default, STAGE=1 stops after bases
```

or by hand:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8    # REQUIRED on GPU (deterministic cuBLAS)
uv run --group convert python -u twin/make_twin.py --stage 1 --out twin/manifests
uv run --group convert python -u twin/train_twin.py --manifest twin/manifests/stage1.json \
    --runs runs_twin --device cuda --width 40
uv run --group convert python -u twin/make_twin.py --stage 2 --out twin/manifests --runs runs_twin
for c in fork swap dial hold; do
  uv run --group convert python -u twin/train_twin.py --manifest twin/manifests/$c.json \
      --runs runs_twin --device cuda --width 40
done
uv run --group convert python -u twin/analyze_twin.py --runs runs_twin --out twin/results
```

Useful flags: `--no-compile` (debugging / CPU; default is
`torch.compile(mode="reduce-overhead")`), `--only REGEX` (filter by run name),
`--force` (retrain finished chunks; a chunk is skipped only when *every* lane in
it already has `spectra.npz`).

**GPU gotchas to expect on the first run.** (a) `CUBLAS_WORKSPACE_CONFIG=:4096:8`
must be exported or `use_deterministic_algorithms(True)` raises on the first
matmul — `run_gpu.sh` does it. (b) The embedding lookup is an `index_select`;
its backward has a deterministic CUDA path, but if inductor emits an atomic
scatter under `--compile`, the gates will fail loudly rather than silently —
fall back to `--no-compile` (or `--compile default`) and note it in
`provenance.json`. (c) Run `gates.py` *before* the cohorts; a hard-gate failure
exits non-zero and `run_gpu.sh` stops there.

**Memory** scales linearly in the batch width: ≈0.19 GB of fp32 activations per
lane at p=113 (12769 examples × d_mlp 512 dominates), so width 40 ≈ 7.5 GB and
width 80 ≈ 15 GB. Group sizes are 8 (bases), 10 (fork), 11 (swap), 37 (dial),
5 (hold), so `--width` must be ≥ 37.

**Disk**: checkpoints are written only for the bases (11 and 6 per run,
weights + optimizer state, ~2.7 MB each) ≈ 350 MB; `spectra.npz` is ~1 MB per
30k-epoch run, ≈ 1.4 GB total. `runs_twin/` is gitignored by the existing
`/runs_*` rule.

## Outputs

`runs_twin/<name>/spectra.npz` per run: `epochs`, `train_acc`, `test_acc`,
`train_loss`, `test_loss` (every 10 epochs) and `coeff_epochs`, `coeffs`,
`energy` (every 25 epochs, f64 Fourier decomposition of the final logits over
the full grid). `config.json` carries the job spec, the recipe, the edit
certificate, the batch width and the batch membership.

`analyze_twin.py` writes `twin/results/{bases,fork,swap,dial,hold}_summary.json`:

* **FORK** — `chi(T, ε) = mean(1 − J)` against the within-batch ε=0 twin, per-base
  lock epoch (first T where all 9 perturbed draws return J = 1), per-arm rows.
* **SWAP** — per arm: did the committee move as commanded (swapped-in frequency
  present, swapped-out absent)? `|Δt_gen|`, max test-acc deviation, area between
  curves; winner-swap vs sham vs positive-control distributions of all three.
* **DIAL** — membership-vs-dose tables split by target class
  (member / runner / mid), background flip rate of untouched frequencies,
  `t_gen` vs dose.
* **HOLD** — remaining-time-to-grok ratio `(t_gen − T)/(t_gen_ctrl − T)` and
  committee Jaccard, tabulated by (T_fork, data level).

`t_gen` = first logged epoch with test acc ≥ 0.99 sustained over 2 consecutive
log points (−1 if never). Committees come from `committee_from_coeffs`, copied
verbatim from `dial/analyze_dial.py`, so they are identical to the rest of the
project.

## Files

| file | what |
|---|---|
| `core.py` | recipe, model, dataset + nested masks, Fourier/spectra, committee detector (copied), edits with certificates, checkpoint io |
| `train_twin.py` | manifest-driven lockstep trainer, batch grouping, fork/checkpoint machinery |
| `make_twin.py` | stage-1 and stage-2 manifest generation (all seeds drawn from `META_SEED`) |
| `analyze_twin.py` | the four analyses → `twin/results/*.json` |
| `gates.py` | G1 / G2 / G3 + `within_batch` |
| `smoke_twin.py` | CPU end-to-end smoke (gates, certificates, mask nesting, all four cohorts, analysis) |
| `run_gpu.sh` | the sequential GPU pipeline |
