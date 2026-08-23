# Flight-time program — claims ledger for adversarial verification (2026-08-21)

Status after today's weight-space test: the mechanism-correction leg is DEAD
(see §4). What remains is listed by how exposed it is. Every claim names the
script that produces it, the data it reads, and the commit where the
pre-registration landed (where one exists). The reviewer should assume
nothing below is true until reproduced from the named script.

Setup everywhere: 1-layer transformer, d_model 128, 4 heads, d_mlp 512
(unless noted), ReLU, AdamW (MLX semantics: no bias correction, decoupled
wd), lr 1e-3 unless noted, full-batch, modular addition mod 113 unless noted.
t_mem = first 2 consecutive spectra samples with train acc ≥ 0.99; t_gen =
same with test acc ≥ 0.9. Spectra sampled every 25 epochs (`spectra.npz`:
`energy` (T, 56) = logit energy on cos/sin(ω(a+b)) per frequency; `coeffs`;
`train_acc`; `test_acc`). Committee = gap rule on final-epoch log energy
(`committee_final` in `cloud/train_flight_t2.py`).

---

## 1. Per-seed forecasting from the first 500 epochs (the strongest leg)

**Claim 1a.** A ridge model with weights LOCKED before the cohort existed
(`flight/v3_model.json`, sha256 prefix `51494d321f52f587`, commit 90bb97e)
forecasts log10 t_gen of 102 fresh runs from their first 500 epochs only.
Scored (`flight/score_v3.py`, commit da8ed09): 85/102 (83%) within ×1.5,
median miss ×1.26, Spearman(forecast, realized) 0.954, R²(log t_gen) 0.913,
median |err| 0.099 dex, RMSE 0.129 dex. Per registered tier: in-dist 90%,
new-value 80%, extrapolation (wd 5, frac 0.22 — outside training hull) 83%.
Registered bars (`flight/PREREG_V3.md`): ≥60% within ×1.5, median ≤×1.35,
Spearman ≥0.85, in-dist ≥65%. All pass.

- Cohort: `runs_flight/v3/*` (17 cells × 6 seeds; cells in `V3_CELLS`,
  `cloud/train_flight_t2.py`). Seeds from `draw_seeds` (crc32-keyed
  generator, META_SEED 20260820).
- Forecast file `flight/forecasts_v3.json` committed (530bdad) BEFORE
  segment 2 (epochs 500→30k, warm resume with Adam moments) was trained;
  `stage_v3_seg2` asserts the forecast file exists and refuses to proceed
  otherwise; `stage_v3_forecast` refuses to overwrite an existing file.
- Features: 20 observables, `flight/v3_core.py::FEATURES` (law-derived
  t_law + spectrum-shape: comm_hhi, junk_max_share, gap12, gap45, coh_comm,
  coh_junk, g_c, g_tot, lam_early, log_r_mem, t_mem, …). Training set:
  316 historical add113 runs (`flight/v3_features.py` →
  `notes/v3_features.json`). Model fit / held-out estimate:
  `flight/v3_hunt.py` (GroupKFold(8) leave-(cohort,cell)-out; ridge R²(resid)
  0.744, GBM 0.755, MLP 0.766).
- **Attack surface:** (i) is "forecast" genuinely ex-ante — check git
  timestamps of 530bdad vs seg2 run dirs' mtimes; (ii) window violations
  (runs generalizing before 500) are excluded — count them and check the
  exclusion rule was registered; (iii) in-dist tier shares (frac, wd, lr)
  values with training cohorts — is the hull really left in the
  extrapolation tier; (iv) 17 cells × 6 seeds is small per cell; (v) r*
  and floor constants were fit on historical add113 — same task as test.

**Claim 1b (law-anchored vs law-free).** Re-fitting the same ridge without
the law-derived features drops extrapolation-tier accuracy from 89% to 56%
(in-dist roughly unchanged). Script: `flight/v3_hunt.py` ablation branch /
`flight/score_v3.py --lawfree` (verify which; the numbers were produced
from the v3 cohort after unblinding — this is a POST-HOC ablation, not a
pre-registered bar).

**Claim 1c (censoring calls).** Runs forecast > budget are called
"CENSORED"; correct iff the run does not generalize in budget. 41/46
correct pooled over dial cohort (37/42, `flight/t1_dial.py`) and degrok
(4/4). Attack: base rate — how many of the 46 were trivially long (wd 0.1)?

**Claim 1d (earlier failures, disclosed).** v1 forecaster 27% within ×1.5
(`flight/score_t2d.py`), v2 47% (`flight/score_t2dv2.py`); both
pre-registered and failed; v3 is the third attempt. A reviewer may count
this as garden-of-forking-paths. Our defence: each was registered, each
failure is reported, and v3's test cohort is disjoint from everything fit.

## 2. Predictability floor (unique as far as the audit found)

**Claim 2a.** Identical-config replicas (same seeds, separate launches,
GPU nondeterminism only) diverge in t_gen: max/min ratio across 4 replicas
ρ_id = ×1.049 median; 10⁻⁶·σ-perturbed inits ρ_pert = ×1.182 median;
per-run σ ≈ 0.024–0.035 dex. Script `flight/score_floor.py`, prereg
`flight/PREREG_FLOOR.md` (9401cbb), verdict commit 2f596e4. Cohort
`runs_flight/floornoise/*` (8 configs × 4 identical + 4 perturbed = 64
runs).

**Claim 2b.** The floor explains only ~2–4% of the v2 forecaster's residual
variance → under-modelling, not chaos, dominated v2's failure (this is the
result that motivated v3's new observables). Same script.

- **Attack surface:** n=8 configs; "chaotic amplification with flight
  length" is stated from the correlation between ρ_pert and wait length
  across 8 points — weak; replicas ran in different lockstep batches, so
  "identical" also differs in kernel schedule (intended, but say so).

## 3. Predictability onset (unique as far as the audit found)

**Claim 3.** Held-out (leave-cell-out ridge, GroupKFold(8)) R² of log10
t_gen vs forecast epoch FC, fixed population of 207 add113 runs with
t_gen > 500: R² 0.32 / 0.65 / 0.66 / 0.68 / 0.91 at FC 50 / 100 / 150 /
200 / 300 (and ~0.91 at 400, 500). Memorization completes ~ep 100–150, so
~70% of final skill is present before t_mem. Script `flight/onset.py`
(commit a2d5a52). NOT pre-registered. Predictions clipped to
[550, 40000].

- **Attack surface:** features at FC include t_law when memorized, which
  leaks the law's constants (fit on overlapping runs); population selection
  (t_gen > 500) is post hoc; cells are the CV groups but several cohorts
  share (frac, wd) values; the 0.65→0.91 step coincides with t_law becoming
  available, so "two steps" may be "law switched on".

## 4. The flight-time law itself (instrument; NOT novel in form)

**Claim 4a.** t_gen = t_mem + max(log(r_mem/r*)/λ, 132), r = off-committee
share of diagonal logit energy, r* = 10^-2.25 (add113 median), λ = fitted
log-slope of r after t_mem. Retrodictive fit on 80 degrok runs: R²(log)
0.736, median miss ×1.54, within-cell Spearman 0.77 (`flight/flight_fit.py`).
LOO r* 0.729 (`flight/t1_harden.py`). Four 2026 preprints (2603.13331,
2605.18845, 2606.13753, 2606.30388) state the same log-ratio/rate form in a
parameter-norm coordinate; 2603.13331 additionally defines out-of-committee
logit energy R(f_θ). **Whether R(f_θ) ≡ our r is unverified (not yet read
at source).** Treat the law as prior art in form.

**Claim 4b (product law).** λ scales with lr·wd: matched-product cells
(lr 1e-3, wd 3) vs (lr 3e-3, wd 1) give λ = 3.154 vs 3.160 /kep (ratio
1.00); log-log slope of λ on lr·wd 1.16. Script `flight/lambda_evidence.py`
→ `notes/lambda_evidence.json` (455 runs, 7 cohorts); the matched-product
pair comes from `runs_flight/t2c` (grid lr {3e-4,1e-3,3e-3} × wd {0.3,1,3}
× 4 seeds), scored in `flight/score_t2ce.py`. Audit found no peer-reviewed
precedent for a matched-product test (the Liu et al. 2022 attribution was
checked at source and is a symbol collision: `notes/timing_uniqueness_audit.md`
§3). **Attack:** one matched pair, 4 seeds each; the slope 1.16 ≠ 1.

**Claim 4c (r* invariance).** r* ≈ 10^-2.25 across primes 113/127, widths
128–4096, and training eras — BUT add96 gives 10^-1.94 and sub113
10^-1.73 (`flight/t1_*.py`, `notes/lambda_evidence.json`). So "invariant"
is across seeds/width/lr/wd within a task family, task-dependent across
tasks. State it that way.

**Claim 4d (λ₀ floor) — RETRACTED.** λ₀ 0.105–0.191/kep was a regression
intercept over wd ≥ 0.1; at wd 0.1 most dial runs are censored at 50k. No
wd < 0.1 runs exist. Do not use.

**Claim 4e (λ ≡ g_committee − g_junk, corr 0.995).** Arithmetic identity
on the share, not a finding. Fine as exposition.

## 5. DEAD today: "outcompetition, not deletion"

`flight/wspace_local.py` → `notes/wspace_local.json`, figures
`notes/wspace_figs.html`, summary `notes/wspace_summary.json`. 162 add113
waits with ≥3 checkpoints inside [t_mem, t_gen] (cohorts degrok, degrok2,
clock, runs_torch/*, t2e; wd 0.4–3; checkpoints every 500–1000 ep).
Measured with the FINAL committee: off-committee W_E Fourier energy ends
higher in 0/5/19% of waits (wd<1 / =1 / >1); W_U 0/6/0%; ‖θ‖² 0% everywhere;
Nanda excluded loss ≥2× its wait-start value in 96–100%; restricted loss
halves in 96–98%. The memorizer IS shed from the weights. The earlier
"junk grows in 74% of waits" was (a) about other-frequency DIAGONAL logit
energy, not the memorizer, and (b) 17%→88% wd-dependent. At wd>1 the
memorizer's logit energy rises in 87% of waits while its weights shrink —
logit-scale inflation. **Do not pitch against Nanda et al. ICLR 2023.**

Caveat on this test for the verifier: first in-wait checkpoint is typically
epoch 1000 (t_mem ≈ 100–200), so the first ~10–20% of each wait is
unobserved; forward pass is a numpy re-implementation of
`cloud/train_semifinal_torch.py::batched_final_logits` (verify equivalence
on one checkpoint against stored `train_acc`/`test_acc` in spectra — the
script records both).

## 6. What is plausibly unique, honestly

1. Ex-ante, locked, pre-registered per-seed grok-time forecasting with OOD
   tiers and censoring calls (§1). Competitors: retrodictive fits, seed
   averages, forecasts issued at t_mem, no lock, no OOD. **This is the
   paper if it survives verification.** The numbers are not dramatically
   better than 2605.18845's 17.7% MAPE; the protocol is.
2. Predictability floor with a perturbation sweep (§2). Qualitative chaos
   remark exists in 2606.12966; no σ, no replicas, no perturbation scale.
3. Onset curve (§3) — weakest evidentially (not prereg, leakage risk).
4. Matched-product λ test (§4b) — small but clean.

Things that would kill the paper: (§1) any evidence forecasts were written
after seg2 outcomes were visible; (§1) window-violation exclusions being
outcome-dependent; (§3) the onset step being a feature-availability
artefact; (§4a) 2603.13331's R(f_θ) being our r with a threshold law.

## 6b. Required source read (part of the verification brief)

Read arXiv 2603.13331 (Norm-Separation Delay Law, v2 2026-05-02) at source
and answer: (i) is their R(f_θ) = Σ_{k∉κ}|f̂_θ(k)|² the same quantity as our
r (our r is the off-committee SHARE of logit energy on the cos/sin(ω(a+b))
diagonal only, per-output summed, softmax gauge not removed — check their
normalisation, whether they use the full 2D spectrum or the diagonal, and
how κ is chosen); (ii) do they state a threshold law "generalization when
R crosses a constant" and report its value; (iii) is R among the three
inputs to their 34.6%-MAE delay predictor, and is that predictor issued at
t_mem from the same run's data (retrodictive calibration) or from a
held-out cohort with fixed coefficients; (iv) do they run any cell with
matched lr·wd product. If (i) and (ii) are both yes, claim 4a's coordinate
novelty is gone and §1 must be framed purely as protocol.

## 7. Scripts index

| script | produces | reads |
|---|---|---|
| `cloud/train_semifinal_torch.py` | trainer (lockstep batched, MLX-AdamW semantics, spectra tap, warm resume) | — |
| `cloud/train_flight_t2.py` | all flight cohorts (stages t2c, t2d*, t2e, floornoise, v3-*, wspace, wd0) | — |
| `flight/flight_fit.py` | law fit on degrok | `runs_degrok` |
| `flight/t1_harden.py`, `t1_clock.py`, `t1_dial.py`, `t1_s5.py` | LOO r*, freeze retrodiction, dial transfer + censoring, S5 boundary | degrok, clock, dial, tasks |
| `flight/lambda_evidence.py` | 455-run λ dossier, product law | 7 cohorts → `notes/lambda_evidence.json` |
| `flight/score_t2d.py`, `score_t2dv2.py` | v1 / v2 forecast scoring (failed) | `runs_flight/t2d*`, `forecasts_t2d*.json` |
| `flight/score_t2ce.py` | matched-product + eps arms, T2-E task table | `runs_flight/t2c`, `t2e` |
| `flight/score_floor.py` | replica floor | `runs_flight/floornoise` |
| `flight/v3_core.py`, `v3_features.py`, `v3_hunt.py`, `v3_model.json` | features, CV, locked model | `notes/v3_features.json` |
| `flight/score_v3.py` | v3 scoring | `runs_flight/v3`, `forecasts_v3.json` |
| `flight/onset.py` | onset curve | `notes/v3_features.json` + spectra |
| `flight/wspace_local.py`, `wspace_figs.py` | weight-space deletion test | all checkpointed runs |
| `notes/timing_uniqueness_audit.md` | novelty audit (103-agent) | — |
