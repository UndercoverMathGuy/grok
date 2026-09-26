# TRAINING CODE AUDIT — heracles-emp evaluation runs (E1, E3, E6, E7, E8) and tuning (E0)

Auditor: independent training-code auditor. Written Sat Sep 26 12:59 IST 2026. Read-only on everything except this file.
No training, no network. All local computation < 30 s per call (largest: 3.6 s). Scratch scripts were kept outside
the repo (session scratchpad); every check is described below precisely enough to redo.

Scope: 370 evaluation run directories (E1 180, E3 60, E6 50, E7 32, E8 48) = 358 trained runs + 12 E3 runs stopped at
step 0; the 48 E1 ids marked `invalid` (lost P4 pull) and their 21 surviving unverified copies in
`experiments/E1/lost_P4_unverified/`; the 408 E0 tuning runs. Code: `core/{train,models,measure,metrics,construct,
runner,remote,programme}.py`, `experiments/common/*.py`, `experiments/E0_tuning/*`, `tests/`.

## 1. Verdict

**No defect in the training code path and no undeclared deviation from the frozen recipe.** One code version (identical
to the current `core/*.py`) produced all 370 run directories. Every config matches `RECIPE.json` exactly. The model,
data, loss, penalty and optimiser are what the paper says. The statuses reproduce from the logs, and the stored metrics
reproduce from `final.pt`. The issues below concern disclosure and interpretation, not correctness:

- **2 MEDIUM** (reviewer risk):
  - the convergence filter selects sparser networks;
  - initialisations are shared across cells (undisclosed).
- **8 LOW** (wording or process):
  - the paper describes the tuning rule inaccurately;
  - the paper's convergence criterion is under-specified;
  - the E3 runs called "crashed" were never trained;
  - some evaluation runs are prefix-identical to tuning runs;
  - the lr rule failed at ν=4, p=59;
  - batched training is not bit-identical to solo training;
  - test-suite provenance is weak;
  - the main text omits momentum.
- **10 INFO** items with no measurable effect.

## 2. The 14 checks

| # | check | verdict | evidence |
|---|---|---|---|
| 1 | Models exactly as stated; no bias/scale/normalisation/nonlinearity; one-hot; width ν(p−1) | **PASS** | See the notes after this table. |
| 2 | Full table D_p, no split / subsample / fraction / shuffle / label noise | **PASS** ("0.3 of the data" REFUTED) | See the notes after this table. |
| 3 | Loss = mean CE + λ Σ‖ω_i‖^q, penalty in the gradient; `penalty_grad` exact | **PASS** | See the notes after this table. |
| 4 | Optimiser: full-batch GD, heavy-ball 0.9 (PyTorch convention), no extras; lr, λ formulas | **PASS** (370/370 configs) | See the notes after this table. |
| 5 | Steps 150k (ν=3,4), 90k (ν=4, p=59), 300k (ν=5); all runs ran to the end | **PASS** | See the notes after this table. |
| 6 | Init: N(0,1) entries; init_scale = √p; relabel identity except E6 | **PASS** (+ undisclosed sharing, see M2) | See the notes after this table. |
| 7 | float32 training, TF32 off (set before compute, asserted, recorded); float64 measurement | **PASS** | See the notes after this table. |
| 8 | Batching: runs independent; batched == solo test and outcome | **PASS for independence; ISSUE (LOW) for "identical"** | See the notes after this table. |
| 9 | Seeds / determinism; seed lists pre-declared; `_r1` reruns identical apart from id | **PASS** | See the notes after this table. |
| 10 | Provenance: recorded sha256 = current core/*.py | **PASS** | See the notes after this table. |
| 11 | Convergence criterion = paper; RUNS.csv = metrics.json; no undeclared exclusion or relabel | **PASS** (criterion wording INCOMPLETE; selection effect M1) | See the notes after this table. |
| 12 | Tuning: selection used optimisation quality only; tuning runs are not evaluation data | **PASS** (with L4) | See the notes after this table. |
| 13 | Nothing else in config → final.pt → metrics.json (predictions in training, post-hoc weight edits, measurement on other weights) | **PASS** | See the notes after this table. |
| 14 | Paper vs reality | 2 INACCURATE-level wording issues, several INCOMPLETE | Section 4. |

**Notes on check 1 (models).**
- **Product and power networks.** `models.py:50-62` computes the product network by a split CP contraction, and `models.py:65-72` computes the power network as `(Σ_t U[t,i,a_t])^n · W`. There are no bias or scale parameters anywhere, and no normalisation.
- **Check against a naive loop.** I compared the code against a naive loop over the definitions in float64 for n = 2, 3, 4 (ν = 3, 4, 5), both families. The maximum error is 1.8e-15 (product) and 1.1e-13 (power, n = 3).
- **Labels.** They equal Σa mod p on every row.
- **One-hot inputs.** They are implemented as the indexing `U[..., a]`, which is exactly one-hot input.
- **Width.** `build.py:20` sets `width = 2 ν h`, which is ν(p−1). All 370 configs have `width == nu*(p-1)`.

**Notes on check 2 (data).**
- **Full batch.** `optimizer = "gd"` in 370/370 configs, and `batch_size = None` in 370/370. With `gd`, `rows_idx = None` (`train.py:394-397`), so `ce_margin_form` takes the mean over all p^n rows (`models.py:130`).
- **Guards.** `validate_batch` forbids `gd` together with a `batch_size` (`train.py:241-242`).
- **No other data options.**
  - The trainer has no data-fraction, train/test, noise or shuffle option. `randperm` exists only in the unused SGD branch (`train.py:396`).
  - A grep of `core/train.py` and `core/models.py`, and of all evaluation and tuning batch configs, finds no `frac`/`split` key.
- **Conclusion.** The recollection "trained on 0.3 of the data" does not apply to heracles-emp.

**Notes on check 3 (loss and penalty).**
- **Objective.** `tot = ce + lam_t * pen` is differentiated by autograd (`train.py:397-399`), so the penalty is inside the gradient (not decoupled).
  - `pen` is `Σ_i (Σ_t‖U_ti‖² + ‖W_i‖²)^{q/2}` (`models.py:83-93`).
  - q is checked to lie in [2, ν] (`train.py:239-240`; `tests/test_train.py::test_norms_guards`).
- **Where `penalty_grad` is used.** It (`models.py:96-102`) is used ONLY for the `grad_rel` diagnostic (`train.py:405-412`), never in the update.
- **Derivation.** ∂/∂x Σ_i (sq_i)^{q/2} = q·sq_i^{q/2−1}·x.
- **Numerical check.** I ran a float64 check against `torch.autograd` over q ∈ {2, 2.4, 2.5, 8/3, 3, 3.2, 10/3, 4, 5} ≤ ν, shapes (B, n, m, p) ∈ {(1,2,5,7), (3,3,4,5), (2,4,3,5)}, and per-run mixed q. The maximum relative difference is **0.0**, and the code also matches the closed form exactly.
- **Edge cases.**
  - A zero-norm neuron gives finite and equal gradients at q = 2 (0^0 = 1 → coefficient 2) and at q > 2 (→ 0).
  - q < 2 would give NaN (inf·0) at a zero neuron, but q < 2 is rejected before training.
- **CE form.** `ce_margin_form` equals the standard CE; see I3 for the softplus threshold.

**Notes on check 4 (optimiser).**
- **Update rule.** `buf ← μ·buf + g; θ ← θ − lr·buf`, with `buf` initialised to 0 (`train.py:314-315, 431-440`).
  - This equals `torch.optim.SGD(momentum=0.9, dampening=0, nesterov=False)`: I verified agreement to 6e-17 over 50 steps.
  - `nesterov = False` in 370/370 configs.
- **Nothing else in the loop.** It performs only `U.sub_`/`W.sub_`. There is no clipping, noise, update normalisation, rescaling, re-initialisation, dead-neuron reset, projection, restart, AMP or loss scaling.
- **No early stopping.** The loop exits only on t = T, "all runs diverged", or the 7200 s timeout (`train.py:424-430`). Neither of the last two fired.
- **Schedules.** `lr_schedule` and `lam_schedule` are `{"kind": "constant"}` in 370/370, and the logged `lr`/`lam` are constant in every row of every log.
- **Every config matches the recipe (370/370).** Each has `optimizer gd`, `momentum 0.9`, `lr_c = c` and `lam_tilde = lt` of `RECIPE.json` for its (model, ν).
  - `lr == c·p` to 1e-12 relative.
  - `lam == lt·B_p^{q/ν}`, with `B_p = 2(2pν)^{-ν/2}`, to 1e-12 relative.
  - `recipe_id = FROZEN-2026-09-25-1755`.

**Notes on check 5 (steps).**
- **Configs.** `steps` equals the recipe in 370/370: 150k, or 90k for product ν = 4 at p = 59 (`build.py:20`), or 300k for ν = 5.
- **Logs.** In all 358 trained runs, `log.csv` has rows at exactly 0, 1000, …, T, and the last row is at step = T.
- **Weights and flags.** `final.pt['step'] == steps` in 358/358, and `timed_out = false` in 358/358.
- **The 12 E3 runs stopped at step 0.** They have only `config.json` and a step-0 log row; see L3.
- **Wall time.** The largest batch wall time is 1401 s, under the 7200 s cap.

**Notes on check 6 (init).**
- **What `init_params` does.** It draws U and then W from one `torch.Generator` seeded by `seed`, with std = `init_scale/√p` (`models.py:149-164`). With `init_scale = √p` (370/370, exact), the entries are N(0,1). A sample at p = 113 has std 1.003.
- **Precision of the init.** It is generated in float64 and rounded to float32 precision.
- **Relabelling.**
  - `init_relabel_u` is 1 in all non-E6-pair runs, and the code applies no permutation when u = 1 (`train.py:298-304`).
  - E6 `_u2`/`_u5` (12 runs) carry u = 2 or u = 5, which permutes tokens a ↦ ua in every slot.

**Notes on check 7 (precision).**
- **Where the flags are set.** `set_precision_flags()` is the first action of `train_batch` (`train.py:260`), before any tensor is created or moved (CUDA is first touched at `train.py:307`). It asserts and returns the three flags (`train.py:82-95`).
- **Recorded flags.** All 370 configs record `allow_tf32` false (matmul and cudnn) and precision `highest`.
- **Runtime evidence.** Over every log row of all 349 non-diverged runs, the GPU float32 training CE (`ce`) and the float64 recomputation (`ce64`) agree to ≤ 5.2e-6 relative (median 3.5e-7; ≤ 1.6e-7 at step 0). TF32's 10-bit mantissa would give errors of order 1e-4 to 1e-3.
- **Measurement.** It is float64 (`measure.py:27-30`; `train.py:347`).
- **Exception.** The stationarity statistic comes from the float32 gradient (I2).

**Notes on check 8 (batching).**
- **Independence.** Parameters are per run (B axis), and q, λ, lr and momentum are per-run tensors (`train.py:311-313, 389-393`). `bmm`, logsumexp, mean and penalty act per batch element, and the gradient of `tot.sum()` gives each run its own gradient. Runs couple only through the batch-level exits: all-diverged (never fired alone) and timeout (never fired).
- **NaN containment.** In the E1 ν4 p59 batches, 8 runs went NaN; their neighbours finished finite.
- **Test outcome.** `tests/test_train.py::test_batched_equals_solo[_cuda]` shows **NOT bit-identical**.
  - CPU: relative weight difference 1.7e-7–3.6e-7 after 600 steps at p = 7 (`core/data/test_reports/batched_vs_solo_cpu.json`, `core/TESTS.txt`).
  - CUDA: absolute difference 3.0e-7–6.0e-7 (`core/data/test_reports_pod/batched_vs_solo_cuda.json`, `core/TESTS_pod_RTX4090.txt`, PASSED).
  - The tolerances are 1e-5 relative (CPU) and 1e-4 (CUDA). NORMS F23 asks for "identical" (L6).
- **Production-scale evidence (mine).**
  - `E6_pair_seed{0..5}_u1` (batch of 18) is **bit-identical** to `E1_nu3_p23_s1.5_seed{0..5}_r1` (batch of 12): same γ and same A to the last bit, 6/6.
  - The 21 surviving lost-P4 runs are bit-identical to their `_r1` reruns (21/21).

**Notes on check 9 (seeds).**
- **How the init is seeded.** Each run seeds its own generator from `seed` (`models.py:157-158`). Full-batch GD has no other randomness.
- **Declared seed lists.** The PREREG lists match `build.py:35-83` and the runs:
  - E1: 0..5 (ν = 3) and 0..3 (ν = 4, 5);
  - E3: 0..3;
  - E6: 0..5 × u ∈ {1, 2, 5}, plus 100..131;
  - E7: 0, 1;
  - E8: 0..2.
- **Grid vs run configs.** 0 mismatches between `configs/grid.json` (+ `grid_reruns.json`) and every run `config.json`.
- **Reruns.** 48 E1 and 12 E3 `_r1` configs differ from their originals only by `rerun_of`.
- **Determinism.** It is shown on the same GPU model (check 8).
- **No global seeding.** torch, numpy and python are not seeded globally (I4).

**Notes on check 10 (provenance).**
- **One hash set.** All 370 run configs carry one `code_sha256` set, and it equals the sha256 of the current `core/{__init__,construct,measure,metrics,models,programme,remote,runner,train}.py`. Files were last modified at 17:32–17:38 IST on 09-25, before the first evaluation start at 17:56:32 IST.
- **Tuning code.** E0 R3/R4 (208 runs) used the same `train.py` (981249d3…). E0 R1 used an older one (0522ac69…); R1 is not used for the recipe.
- **Run files unchanged.** The 716 run files (`final.pt` + `metrics.json`) are unchanged since `audit/sha256_frozen_20260926.json` (716/716 verified).

**Notes on check 11 (convergence and statuses).**
- **The criterion.**
  - It is `DEFAULT_CONVERGENCE` (`train.py:164-170`) and `convergence_verdict` (`train.py:173-208`).
  - `grad_rel = ‖g_CE+λg_R‖/(‖g_CE‖+λ‖g_R‖) ≤ 1e-3` at step T.
  - |L64(T)−L64(0.9T)|/L64(T) ≤ 1e-4, where L64 is the float64 regularised objective.
  - |γ(T)−γ(0.9T)|/|γ(T)| ≤ 1e-3.
  - margin > 0; all values finite; step T reached.
- **Criterion recorded.** Every `metrics.json` records exactly this criterion (358/358).
- **Recomputed statuses.** Recomputing the verdict from `log.csv` reproduces status and every check in 358/358.
- **RUNS.csv.** Its last row equals `metrics.json` status in 358/358. The 48 lost ids are `invalid` and the 12 step-0 ids are `crashed`, both with reasons.
- **No undeclared exclusion or relabel.** Each run's status history is `launched → final`, except the 48 lost ids (`invalid` only).
- **Why runs fail.** Among non-diverged runs, γ drift fails in 88, loss plateau in 23, grad_rel in 14, margin in 1.

**Notes on check 12 (tuning).**
- **The selection rule** (`experiments/E0_tuning/select_R3.py:3-4, 27-31`):
  1. the most runs with status `ok`;
  2. ties → smaller λ̃;
  3. ties → lower median final `grad_rel`.
- **What the selection code reads.** It reads no prediction, K*, y*, Γ or count. I re-derived the R4 selection from the run directories and it equals `analysis/selection_R4.json` = `RECIPE.json` for all four (model, ν).
- **No prediction scoring in tuning.** All 408 tuning configs have `score_predictions = false`, and 0 `agreement` blocks exist.
- **Tuning data is not reused.** No analysis, figure or supplement script reads `E0_tuning/runs` (the only reference is the benchmark json in `write_prereg.py:118`).
- **Declared functional forms.** The forms lr = c·p and λ ∝ B_p^{q/ν} were declared in `rounds.py:55-66, 95-103` before launch. B_p depends on (p, ν) only.

**Notes on check 13 (nothing else in the path).**
- **No predictions during training.** `score_predictions = false` in 370/370, and no `metrics.json` has an `agreement` block. `train.py` imports only `models`; `measure` (and, through it, `programme`, used only for `coverage_matrix(p)`) is imported after the loop (`train.py:448`). `full_measurement` is called with `Gamma=None`, so no ceiling or prediction enters.
- **Measurement is on the final weights.**
  - The metrics are computed on the same tensor saved as `final.pt` (`train.py:451-468`).
  - My float64 recomputation of γ from `final.pt` matches `metrics.json` to ≤ 1.1e-15 (349/349).
  - Shapes, dtype float32 and step = T hold in 358/358.
- **No post-hoc weight edits.** The only analysis that loads `final.pt` is E6 (`experiments/E6/analysis/analyze_E6.py:20,25`), and it only reads.
- **Extra config keys are ignored by the trainer.** `s_target`, `lr_c`, `lam_tilde`, `recipe_id` and `rerun_of` are provenance only.

## 3. Issues, ranked

### MEDIUM (no code defect; a hostile reviewer would raise them, and the paper does not address them)

**M1. The convergence filter selects sparser networks.**
- **What the paper does.** Main-text counts use converged runs only, as the PREREG declares (primary = ok).
- **The selection effect.** In the 43 cells that contain both converged and not-converged runs, the not-converged runs have larger counts: thr_20 is higher by +1.28 on average (median +1.0; higher in 29 cells, lower in 6). This is computed from `metrics.json`.
- **E7 is the extreme case** (Fig. 1a; 13 of 32 not converged). In 10 of its 11 mixed s-cells, the not-converged count is at least the converged count (strictly larger in 9); for example, at s = 1.27 it is 11 against 6. The exception is s = 1: 17 against 18.
- **Cells with no converged run:**
  - E1 ν4 p43 s1.5 (0/4);
  - E1 ν4 p59 s1.25 (3 diverged, 1 not converged);
  - E1 ν4 p59 s2 (2 diverged, 2 not converged);
  - E3 power p59, power p113, product ν4 p43, product ν4 p59 (0/4 each);
  - E7 s = 1.373 (0/2).
- **Overall:** 260 converged, 89 not converged and 9 diverged, out of 358 trained runs.
- **Why it matters.** Conditioning on γ-stationarity (γ drift causes 88 of the 89 failures) may bias counts downward, toward K*. The paper should state this and show not-converged counts.

**M2. Initialisations are shared across cells and runs are duplicated. Neither is disclosed.**
- **Mechanism.** The init depends only on (n, p, width, seed): not on q, not on the model (`models.py:149-164`, `build.py:17-26`). All 94 groups of runs with the same (n, p, seed, u) start from identical weights (step-0 margins identical).
- **E7 (fine b-sweep).** Its 32 runs use **2** distinct initialisations. The "83 of 96 pairs" direction test therefore compares runs that start from the same two inits.
- **E1 cells** at the same (ν, p) share inits across s.
- **E3** shares inits with E1 (seeds 0–3 at p = 23, 59, 113 for ν = 3; 23, 43, 59 for ν = 4; 11, 13, 17 for ν = 5). E3 product and power runs share inits with each other.
- **Duplicates.** `E6_pair_seed0..5_u1` are bit-identical duplicates of `E1_nu3_p23_s1.5_seed0..5_r1`. The 12 relabelled E6 runs are their exact symmetry images. The "349 trained runs" therefore contains 18 networks that repeat 6.
- **Assessment.** A paired design is legitimate, but it should be stated. The permutation nulls and the pair counts treat cells as independent.

### LOW

**L1. The paper misdescribes the tuning rule** (`appendix_experiments.tex:3`).
- **Paper:** the rule "scored only the regularized loss reached, stationarity and stability of the normalized margin".
- **Code** (`select_R3.py:3-4, 27-31`): the rule counts runs that pass the convergence criterion, then breaks ties by the smaller λ̃, then by the lower median `grad_rel`. No loss level is scored; the criterion checks that the loss has plateaued, not how low it is.
- **Also not stated:**
  - the lr = c·p and λ ∝ B_p^{b/ν} forms were fixed after R3 by declared reasoning;
  - the steps were set by the 2 h compute budget, not tuned (`RECIPE.json` `steps_rule`).

**L2. The paper under-specifies the convergence criterion** (`appendix_experiments.tex:21`).
- **"Relative gradient"** is ‖∇L‖/(‖∇CE‖+λ‖∇R‖), which the paper does not define.
- **"Stationary over the last 10%"** is a comparison of two points (steps 0.9T and T), with thresholds 1e-4 for the float64 regularised objective and 1e-3 for γ. It is not a check over the whole window.
- **Also required:** all values finite and step T reached. The code uses ≤, the paper "below".

**L3. The E3 runs called "crashed" were never trained** (`appendix_experiments.tex:28`: "23 converged, 24 did not, 1 diverged and 12 crashed").
- **What happened.** The 12 runs were stopped by the operator at step 0 while the pod queues were rebalanced. They were never trained, and were rerun as `_r1` with identical configs (`E3/RUNS.csv` notes; `E3/LOG.md` 18:10, 18:15).
- **Correct count.** 48 E3 runs were trained, as planned. The word "crashed" suggests training failures.

**L4. Some evaluation runs repeat tuning trajectories.**
- **Which runs.** The tuning rounds used seed 0 (and seed 1 for power) at primes that are also evaluation primes, and q ∈ {2, ν}. As a result, 17 trained evaluation runs start with **the same 40k steps, bit for bit,** as tuning runs that fed the selection. Examples:
  - `E1_nu3_p23_s1.5_seed0_r1` ≡ `R4_prod_nu3_p23_006` (γ at step 40k identical);
  - also `E1_nu3_p113_s1.5_seed0`, `E1_nu4_p43_s2_seed0`, `E3_power_nu3_p{23,113}_seed{0,1}`, and others.
- **Consequence.** The recipe was chosen partly to make these trajectories converge, so their convergence status is not independent of tuning.
- **No prediction leakage.** The tuning `metrics.json` contain full measurements (A, counts) that were visible to the tuner, but the selection code does not read them.

**L5. The lr = c·p rule fails at ν = 4, p = 59.**
- **What happened.** 9 of 16 runs diverged (E1 8/12, E3 1/4) at lr = 16.5 (`E1/LOG.md` 18:20).
- **Tuning coverage.** ν = 4 was tuned only at p ≤ 43, and c = 0.28 is the top of that tuning grid. For ν = 3, c = 0.4 is also the top of its grid.
- **What the paper says.** It reports the divergence counts and omits the cell from Fig. 1b, but does not give the cause.

**L6. Batched training is not identical to solo training.**
- **The rule.** NORMS F23 requires batched runs to be identical to solo runs.
- **The test.** It shows only rounding-level agreement (Check 8), at p = 7 over 600 steps. No evaluation run was trained solo.
- **Mitigation.** Production-scale reruns are bit-identical across batch sizes 12 and 18, and across pods. The earlier audit's retraining reproduced all counts (6/6).

**L7. Test-suite provenance is weak.**
- **Local run.** The local suite (`core/TESTS.txt`, 17:08) predates the final `train.py` (17:32).
- **Pod run.** The only full run on the final code is the pod run (`core/TESTS_pod_RTX4090.txt`). It has **1 FAILED** test: `test_run_directory_provenance_and_immutability`. Under root, `final.pt` is writable, so run-directory immutability is not enforced on pods. The pulled files are also writable locally.
- **No code hashes in reports.** Neither report records code hashes.
- **Mitigation.** The sha256 freeze (716/716 unchanged).

**L8. The main text calls the optimiser "gradient descent"** (`experiments.tex:8`). The optimiser is full-batch GD with heavy-ball momentum 0.9. Only the appendix says so.

### INFO (no measurable effect)

- **I1. Float32 hyperparameters.**
  - q, λ, lr and momentum enter training as float32 tensors (`train.py:311-313, 389-390`). For non-dyadic q (2.4, 8/3, 3.2, 10/3, E7 grid), the training exponent is off by about 1e-7 relative.
  - `log.csv` γ uses this float32 q: it differs from `metrics.json` γ by ≤ 2.3e-7 (median 4e-16).
  - `metrics.json` uses the exact q.
- **I2. The stationarity statistic is not float64.** `grad_rel` comes from the float32 autograd gradient and a float32 analytic penalty gradient, cast to float64 (`train.py:405-412`). The paper's "float64 measurement" applies to margins, spectra and ablations only. The float32 floor (~1e-5) is well below the 1e-3 threshold.
- **I3. CE saturation.** `ce_margin_form` uses `softplus`, which saturates for arguments above 20 (`models.py:130`). The per-row CE error is ≤ 2e-9 and occurs only on rows with loss > 20 nats (early training).
- **I4. Seeding scope.** NORMS B10 says "seed torch, numpy, python". Only a per-run `torch.Generator` is seeded. Nothing else is random under full-batch GD.
- **I5. Deterministic algorithms.** `torch.use_deterministic_algorithms` is not set (`train.py:371-373`). Reproducibility is shown empirically on the same GPU model (RTX 4090, torch 2.13.0+cu130, for all 370 runs; the flagged fallback from the 5090).
- **I6. Relabelling.**
  - The code maps tokens a ↦ ua in inputs and readout. This gives v̂′(j) = v̂(uj). The paper says "relabelling j ↦ uj of frequencies"; the direction is a matter of convention.
  - In exact arithmetic, GD is equivariant, so the E6 pairs measure only float32 round-off.
- **I7. The trained networks sit at finite λ.**
  - In converged runs, the penalty is 88–95% of the final objective, and the final CE is 1.6e-3 to 5.2e-3.
  - The networks are stationary points at finite λ, not max-margin solutions. The paper already says trained networks are not maximizers.
- **I8. Large initialisation.**
  - Each weight vector has norm ≈ √p, and lr reaches 45.2 (ν = 3, p = 113).
  - The step-0 CE is ≈ 47 nats. This is declared in the recipe.
- **I9. "349 trained runs"** means 358 trained minus 9 diverged. It includes the E6 duplicates (M2), and it includes not-converged runs, although the paragraph's header says "Converged runs".
- **I10. What `code_sha256` covers.** It covers `core/*.py` only. The config generator (`experiments/common/build.py`) is not hashed, but every resulting config is stored in full.

## 4. Paper vs reality (every training sentence in prelim.tex, experiments.tex, appendix_experiments.tex)

| location | sentence (abridged) | verdict | note |
|---|---|---|---|
| prelim:3 | one-hidden-layer networks, homogeneous neurons, no biases; neuron ω_i=(u_i1..u_in,w_i), ‖ω_i‖_2 Euclidean norm of the stacked vector | ACCURATE | `models.py:83-85` |
| prelim:5 | L_{2,b} norm (Σ‖ω_i‖^b)^{1/b} | ACCURATE | `models.py:88-93,105-109` |
| prelim:7 | D_p = all n-tuples with label Σa mod p | ACCURATE | `models.py:26-38` |
| prelim:7 | inputs one-hot, u acts as u(a) | ACCURATE | indexing ≡ one-hot |
| prelim:7 | product neuron w(c)Π_t u_t(a_t); power neuron w(c)(Σ_t u_t(a_t))^n; ν=n+1 | ACCURATE | verified against a naive loop, n = 2, 3, 4 |
| exp:8 | "We train the networks … with gradient descent" | INCOMPLETE | heavy-ball momentum 0.9, full batch (L8) |
| exp:8 | training details, criterion and count definition in App. E | ACCURATE | |
| exp:19 (Fig. 1a) | "Each marker is one run (19 converged, 13 not)" | ACCURATE | E7 |
| exp:19 (Fig. 1b) | "ν=4, p=59 has one [converged cell] and is omitted" | ACCURATE / INCOMPLETE | 8/12 of that sweep diverged (L5) |
| app:3 | width m=ν(p−1) | ACCURATE | 370/370 |
| app:3 | entries i.i.d. N(0,1) | ACCURATE / INCOMPLETE | init is shared across cells with the same (ν, p, seed) and across models; E7 has 2 inits (M2) |
| app:3 | no biases | ACCURATE | |
| app:3 | full-batch GD with heavy-ball momentum 0.9 | ACCURATE | = torch SGD(momentum=0.9, dampening=0, nesterov=False) |
| app:3 | constant learning rate c·p | ACCURATE | 370/370, constant in logs |
| app:3 | mean CE over D_p plus λΣ‖ω_i‖^b | ACCURATE | |
| app:3 | constant λ=λ̃ B_p^{b/ν} | ACCURATE | 370/370 to 1e-12 |
| app:3 | penalty part of the gradient (not decoupled) | ACCURATE | `train.py:397-399` |
| app:3 | float32 with TF32 disabled | ACCURATE | asserted, recorded, runtime evidence |
| app:3 | margins, spectra, ablations in float64 | ACCURATE / INCOMPLETE | `grad_rel` (stationarity) is from float32 gradients (I2) |
| app:3 | "selection rule scored only the regularized loss reached, stationarity and stability of the normalized margin" | **INACCURATE** | the rule counts converged runs, then smaller λ̃, then median grad_rel; no loss level is scored (L1) |
| app:3 | constants frozen before any evaluation run | ACCURATE | frozen 17:55 IST; first evaluation start 17:56:32 IST |
| app:11-14 (table) | c, λ̃, steps, prime ranges per network | ACCURATE | all four rows |
| app:19 | ν=5 not run at p≥23: no tuning run converged at p=23; longer runs did not fit the 2 h limit | ACCURATE | R3 0/16, R4 0/8 at p = 23 |
| app:21 | trained on the full table D_p, no train/test split | ACCURATE | |
| app:21 | converged if relative gradient < 1e-3, loss and γ stationary over the last 10%, margin > 0; others reported separately | INCOMPLETE | definition, thresholds and the two-point comparison are missing (L2); the selection effect is not discussed (M1) |
| app:21 | grid, seeds, predictions, matching criteria preregistered before any evaluation run | ACCURATE | PREREG stamp 17:55:06 < 17:56:32 |
| app:23 | every launched run listed with status in the supplement; none omitted | CANNOT-VERIFY | supplement not yet built (`supplement/build_supplement.py` only); the RUNS.csv files it copies are complete |
| app:25 | main-sweep grid; 126 converged / 46 not / 8 diverged; 48 lost in transfer, marked invalid, rerun under new ids | ACCURATE | the 21 surviving lost runs are bit-identical to their reruns |
| app:26 | p-sweep grid; 44 converged / 4 not | ACCURATE | |
| app:27 | fine b-sweep: ν=4, p=37, 16 s in [1,2] inside exact plateaus, 2 seeds; 19 / 13 | ACCURATE / INCOMPLETE | only 2 distinct inits (M2) |
| app:28 | dense endpoint grid; "23 converged, 24 did not, 1 diverged and 12 crashed" | **INCOMPLETE (misleading)** | the 12 were never trained (stopped at step 0), then rerun (L3) |
| app:29 | relabelling: 12 pairs, init differs by j↦uj; 50 runs, 48 converged | ACCURATE / INCOMPLETE | u=1 runs duplicate E1 runs; the pairs are exact symmetry images (M2, I6) |
| app:36 | universal bound never exceeded in any of the 349 trained runs | ACCURATE / INCOMPLETE | 358 trained, 9 diverged excluded; includes duplicates and not-converged runs despite the "Converged runs" header (I9) |
