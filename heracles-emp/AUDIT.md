# AUDIT — independent audit of the heracles-emp empirical record

Auditor: independent AUDIT agent (PLAN.md row AUDIT). Started 2026-09-25 23:59 IST. Own code only, under audit/.

## Pre-declared re-measurement sample (written BEFORE any metric value of these runs was read)

Script: `audit/sample.py` (numpy `default_rng(20260926)`; population = run ids whose last RUNS.csv status is
ok or not_converged; allocation E1 4, E3 2, E6 2, E7 2, E8 2; output `audit/sample.json`). Retrain subset rule
(declared here, config-only): cheapest run per experiment by p^n·p·width·steps, plus the next cheapest.

| # | exp | run_id | status (RUNS.csv) | model | nu | p | q | retrain |
|---|---|---|---|---|---|---|---|---|
| 1 | E1 | E1_nu5_p13_s1.25_seed2 | ok | product | 5 | 13 | 4 | yes |
| 2 | E1 | E1_nu5_p17_s2.5_seed0 | ok | product | 5 | 17 | 2 | no |
| 3 | E1 | E1_nu5_p17_s2_seed1 | ok | product | 5 | 17 | 2.5 | no |
| 4 | E1 | E1_nu5_p17_s2_seed3 | not_converged | product | 5 | 17 | 2.5 | no |
| 5 | E3 | E3_product_nu3_p59_seed3_r1 | ok | product | 3 | 59 | 3 | no |
| 6 | E3 | E3_product_nu5_p11_seed1 | ok | product | 5 | 11 | 5 | yes |
| 7 | E6 | E6_marg_seed108 | ok | product | 3 | 23 | 2 | yes |
| 8 | E6 | E6_marg_seed114 | not_converged | product | 3 | 23 | 2 | yes |
| 9 | E7 | E7_nu4_p37_q2.7586_seed0 | not_converged | product | 4 | 37 | 2.7586 | yes |
| 10 | E7 | E7_nu4_p37_q3.0303_seed0 | ok | product | 4 | 37 | 3.0303 | no |
| 11 | E8 | E8_nu3_p29_s1.25_seed2 | ok | product | 3 | 29 | 2.4 | yes |
| 12 | E8 | E8_nu3_p71_s1.25_seed0 | not_converged | product | 3 | 71 | 2.4 | no |

Disclosure: before drawing, the auditor had seen the metrics.json of one run (E1_nu3_p113_s1.25_seed0) while
learning the file format; it is not in the sample.

## Verdict table

| # | check | verdict | evidence (script -> output) |
|---|---|---|---|
| 1a | PREREG stamped before the earliest evaluation start (E1, E3, E6, E7, E8) | PASS | `audit/check_provenance.py` -> `audit/provenance.json`. Stamp 17:55:06 IST; first starts E1 17:56:32, E3 17:57:43, E7 17:57:21, E8 18:09:37, E6 18:33:40. PREREG mtimes equal the stamps (not edited later). |
| 1b | E4/E5 PREREG timing | CAVEAT | Stamped 18:02:00, after E1 training began (17:56:32) but before the first E1 run finished (18:18:18) and before any E4/E5 analysis. This matches the file's own wording ("before any E1 evaluation run finished"), but it is not "before any evaluation run". |
| 1c | Frozen recipe = every evaluation config (optimizer, momentum, lr = c·p, λ = lt·B_p^{q/ν}, steps incl. ν4 p59 = 90k and ν5 = 300k, N(0,1) init, width 2νh, float32, constant schedules, recipe_id) | PASS | 368 evaluation run configs, 0 violations. Recipe frozen 17:55, 1.5 min before the first evaluation start. |
| 1d | TF32 asserted and recorded | PASS | All 368 configs record allow_tf32 = False (matmul, cudnn) and precision 'highest'; `core/train.py` sets and asserts them (lines 84-93). |
| 1e | Only allowed models / optimisers; full table | PASS | product ν3/4/5 and power ν3 (E3, q = 3 only); optimizer 'gd' (full batch, heavy-ball 0.9) everywhere; the penalty is inside the autograd loss (coupled, train.py:397-399); no SGD row sampling. |
| 1f | Tuning runs not used as evaluation data | PASS | No analysis script reads E0_tuning/runs; all 408 tuning configs have score_predictions = false; the tuning selector never references predictions. (Disclosed deviation: E0 R1 ran on the laptop against NORMS F22; logged; not evaluation data.) |
| 2a | Every run dir in RUNS.csv and vice versa | PASS | 0 dirs without a row. The 48 ids without a dir are exactly the lost-P4 runs (status invalid). |
| 2b | Last-row status = metrics.json convergence checks | PASS | 0 mismatches over 368 dirs (the 12 E3 'crashed' dirs have no metrics.json, as their rows say). |
| 2c | Invalid / crashed / diverged runs explained | CAVEAT | 48 invalid (P4 incident) and 12 E3 crashed (stopped at step 0 for rebalancing) are explained in LOG.md and RUNS.csv notes; E1's 8 diverged (ν4 p59) are explained in E1/LOG.md 18:20. The single E3 divergence (E3_product_nu4_p59_seed2) is listed in REPORT.md but no cause is given in any LOG. |
| 2d | pulled_*.json sha256 manifests match files on disk | **FAIL (provenance gap; no data error found)** | The pulled_*.json files hold only counts ({n_rows, files_verified, queue_manifest_files_checked}), not hashes. pull_fast.py verified against pod-side sha256 lists at pull time, but it deleted its staging directory, so neither the pod lists nor the queue manifests were persisted. The pull-time verification cannot be re-checked. Mitigations: all final.pt / metrics.json mtimes are 18:26-18:59 IST (pull times, before any report was written); my re-measurement reproduces metrics.json to ≤ 1e-14 on 14 runs; the 6-run retraining agrees (section 4b). The auditor froze the current hashes of all 360+96+100+64+96 final.pt / metrics.json files in `audit/sha256_frozen_20260926.json` so that later changes are detectable. Note also: final.pt / metrics.json are writable on disk (train.py makes them read-only on the pod; permissions were not preserved by scp). |
| 2e | Lost-P4 incident handled as LOG.md says | PASS | 86 files / 22 partial dirs in E1/lost_P4_unverified, referenced by no analysis script; 48 ids 'invalid' with the reason; 48 `_r1` reruns whose configs differ from the originals only in the added `rerun_of` key (same seeds). |
| 3 | Headline numbers of every REPORT, REPORT_A1 and RESULTS recomputed from run dirs | PASS (numbers) / CAVEAT (4 text discrepancies) | `audit/recompute_E1.py`, `audit/recompute_rest.py`, `audit/recompute_A1set.py`, `audit/controls_audit.py`. Every table number checked agrees to rounding (A1 E1 per-cell tables: 62 rows, 0 diffs). Text discrepancies listed below. |
| 3b | A1 honestly labelled post hoc; dropped ε = 5% analysis disclosed | PASS / CAVEAT | Labelled POST HOC in AMENDMENTS.md, REPORT_A1.md and the RESULTS A1 column; the ε = 5% drop is disclosed, including that its trained-run numbers were seen. CAVEAT: RESULTS promotes thr_20 as "registered" and "without any amendment", but choosing it as the headline after seeing results is itself a post-hoc choice: it is the best of the 9 registered count definitions (within ±1: 0.51-0.70 across definitions; thr_20 0.698 is the maximum). The RESULTS "usable statement" column for E8 relies on the post-hoc K_2pct count without saying so. |
| 4 | Independent float64 re-measurement of 12 pre-declared runs (+2 power runs) | PASS | `audit/remeasure.py` -> `audit/remeasure.json`: margin, L_{2,q} norm, γ agree to ≤ 2.4e-15 relative (half of them bit-identical); clean amplitudes A_j to ≤ 6.6e-15 of max A; thr_10 / thr_20 counts identical 12/12. Extra: per-offset folded margins (E5 input) ≤ 8.9e-15, frequency ablation (E4 input) ≤ 3.6e-15 (`audit/peroffset_check.py`, `audit/ablation_check.py`); init regeneration matches log.csv step-0 margins to 2e-16. |
| 4b | Retraining of 6 sampled runs on a fresh pod | PASS | Section 4b: statuses, all counts and supports reproduced; γ within 8.6e-6 (2 of 6 bit-identical). Pod $0.38, deleted. |
| 4c | Programme data (Γ, K*, y*, runner-up gaps, vertex cache) | PASS | `audit/brute_programme.py` (own basis enumeration) at p = 5, 7, 11, 13, 17, 19, all s used: Γ to ≤ 3e-10, K* equal, y* equal up to dilation, runner-up gaps equal; vertex cache identical to the brute-force vertex set at p = 11, 13, 17, 19; closed form Γ_1 = p/(p-1) reproduced. |
| 4d | Positive controls (own construction) | PASS | `audit/controls_audit.py`: ν-blocks built from the paper formula at 7 cells (ν = 3, 4, 5; p up to 37) give γ/γ_IF = 1 to 1e-14, A = y* to 7e-14, near-tied offsets = tight set, ablation bound satisfied. This independently backs the E5 control, which in analyze_E5.py is computed from the coverage of y* rather than measured on a network (tautological as coded). |
| 5 | RESULTS summary claims | 2 FAIL, 6 CAVEAT rows (5 distinct issues) | Section 5 below. |

Totals: **FAIL 3** (2d provenance gap; section 5 rows E3 and Reading 3). **CAVEAT 9** (1b, 2c, 3 text discrepancies, 3b, and the section 5 rows E1 count (= Reading 1), E1 shares, E6, E7, E8). No number in any REPORT, REPORT_A1 or RESULTS was found wrong.

## 3. Discrepancies found (all larger than rounding are listed)

Numbers: none. Every recomputed headline agrees to the printed precision, including E1 REG (10.3% joint match; thr_10
0.341 / 0.659 / R² 0.385 / slope 1.36 [1.10, 1.62] / bias +0.94; thr_20 slope 1.01 [0.88, 1.15]; TV 0.418 [0.104, 0.615];
γ/γ_IF 0.716; max γ/ceiling 0.732; 1/36 cells supported; E1b ratios 0.53-1.15), E1 A1 (all six count rows, eff 0.871
[0.850, 0.882], fractions 0.01 / 0.12 / 0.30, 1%-set min-TV 0.288, K_2pct in range 0.58, E1b 0.67 / 8 of 9 / 0.94 over 34), E3
(every cell), E4 (765/768, 0.861, null 0.417, by ν 0.998 / 1.000 / 0.980, the 3 + 1 violations, excess quantiles), E5
(0.016, 0.095, 0.183 / 0.125, 0.008, -3.83, -4.0), E6 (12/12, 1.4e-6, 10 of 12 < 7e-8, drift 1.3e-6 / 4.4e-5, χ² 6.77, p =
0.75), E7 (2/19, all REG sweep numbers, within/adjacent 0.112 / 0.101, max 0.816, A1 rows), E8 (5/14, 3/14, all sweep
numbers, K*/log p ranges, per-p lists, TV ranges), and RESULTS (clean energy 0.47-1.00 median 0.97; E8 23/28 with the
listed misses; E8 efficiency 0.937). My bootstrap CIs use a different RNG and agree to ±0.01.

Nulls: random-allocation TV (median over E1 cells) 0.50 in my draws vs 0.52 reported; random-init (seed 0) TV 0.60 and
efficiency 0.68 over the 33 cells I could build under 2M table entries vs 0.64 / 0.667 reported over 39 cells. E3
random-init claims reproduce (29/29 at p = 59, 54/56 at p = 113; min/max 0.06-0.32; TV 0.13-0.25).

Text discrepancies:
1. **E7/REPORT.md**: "Only s = 1 (q = nu, dense) matches". The 2/19 plateau matches are s = 1.0 seed 0 AND s = 1.226
   seed 1. Also thr_10 = K* at s = 1.27, 1.45, 1.55, 1.6 (seed 1).
2. **E3/REPORT.md**: the per-cell table omits the not-converged rows of 4 cells (power ν3 p23 n = 2; product ν4 p23 n = 1;
   product ν5 p13 n = 3; product ν5 p17 n = 2). They are in REPORT_numbers.md, so nothing is hidden, but the table
   says it lists 'nc' groups.
3. **E1/REPORT.md**: "Not-converged runs (46) ... thr10 exact 0.04, TV 0.51" is computed over the 45 with margin > 0.
4. **RESULTS.md E7 row**: "Spearman 0.79-0.91" covers 4 of the 6 definitions; the full range over the A1 table is
   0.61 (thr_10) to 0.94 (K_5pct).

## 5. Claims (RESULTS.md summary table and Reading)

| row | usable statement (RESULTS) | verdict | problem and safe wording |
|---|---|---|---|
| all | margin never exceeds the ceiling | PASS | Verified on all 349 runs with finite γ (every experiment, every status). Overall max 0.99999 (E3 product ν = 4, p = 23 and ν = 3, p = 23 at q = ν), which is ≤ 1; the RESULTS maxima 0.73 (E1) and 0.82 (E7) are correct. |
| E1 count | "tracks K* within ±1 in ~70% of converged runs, unbiased (slope ~1); exact in ~1/3" | CAVEAT | (i) thr_20 was picked after the results as the best of 9 registered definitions (range 0.51-0.70); the registered primary thr_10 gives 0.66, slope 1.36, bias +0.94. (ii) "unbiased" is a pooled mean: errors are systematic and cancel (overcount at ν = 3 p ≥ 59, undercount at ν = 5 p = 11/13 and ν = 4 p = 23 s = 1.25). (iii) No null was given: the best constant count is within ±1 in 0.50 of the same runs, a permuted K* in 0.36. Safe: "Within ±1 of K* in 70% of converged runs under the registered thr_20 count (66% under the registered primary thr_10; best constant predictor 50%); exact in 33-34%." |
| E1 shares | "~87% of optimum, far above nulls ... Do not claim shares match y*" | CAVEAT | Numbers correct (0.871 [0.850, 0.882]; nulls 0.45 / 0.67). Label efficiency as post hoc, and give the numbers instead of "far above". |
| E1b | "weak: counts agree across ν, contrast small" | PASS | Honest. Lead with the registered result: 0/9 (p, s) met the declared criterion. |
| E3 | "dense demo holds where GD converges" | **FAIL** | False as worded: at ν = 5 runs that converged (p = 11, 13, 17; 7 ok runs) are not flat (TV to uniform 0.10-0.41; p = 17 uses 5.5 of 8). Safe: "q = ν gives dense, near-flat spectra for product ν = 3 (p = 23, 59, 113) and ν = 4 (p = 23), and power ν = 3 (p = 23), at 0.88-1.00 of the ceiling; not at ν = 5 (converged but not flat), and not established at ν = 4 p ≥ 43 or power p ≥ 59 (no run converged)." |
| E4 | "supported as a measurement; bound loose (median excess 0.29)" | PASS | 765/768 active pairs vs a permutation null of 0.42. It is not a theorem check on trained nets (they are not optimal). |
| E5 | "not supported on trained networks ... theorem only on constructed networks" | PASS | "Consistent with sitting at ~0.87 of the optimum" is an interpretation, not a test. |
| E6 | "supported" (equivariance and uniform marginals) | CAVEAT | Equivariance: supported (12/12, drift ≤ 4.4e-5). Marginals: "uniformity not rejected" (χ² p = 0.75, 30 runs, low power; the counts inside a run are not independent, so the p-value is approximate). Do not write "uniform marginals supported". |
| E7 | "count moves in the predicted direction with q; exact plateaus not resolved" | CAVEAT | Correct, but quote the full Spearman range 0.61-0.94 and say the counts are compressed (OLS slope 0.36-0.55; K_2pct bias -2.2). |
| E8 | "the count follows K*(p) up to p = 113 on energy counts. Optimality degrades with p" | CAVEAT | Post hoc (K_2pct) and lenient (per-cell median within ±1). K* only spans 5-10 (s = 1.25) and 2-6 (s = 1.5) and is best-found, not certified, beyond p = 37. Registered result: median thr_10 = K* at 8/28 p-cells, and 1.5-2.5x K* at p ≥ 59 (s = 1.5) and p = 113. A best constant per s reaches 18/28 cells within ±1, against 23/28 for K*. Safe: "post hoc energy count: per-cell median within ±1 of K* at 23/28 (p, s) cells up to p = 113 (constant baseline 18/28); the registered thr_10 count overshoots 1.5-2.5x at p ≥ 59." |
| Reading 1 | "The registered thr_20 count shows this without any amendment" | CAVEAT | Same as the E1 count row: the choice of thr_20 is post hoc. |
| Reading 3 | "Theorems (ceiling, ablation bound, equivariance) hold on every run" | **FAIL** | The ablation bound fails on 4 (run, j) pairs (3 in ok runs, 1 in a not-converged run), and on trained networks it is a measurement, not a theorem. Equivariance was tested on 12 pairs, not on every run. Safe: "The ceiling holds in every run (349/349). The ablation bound holds for 765/768 active pairs of converged runs. Relabelled inits give the relabelled support in 12/12 pairs. Constructed controls are exact everywhere (independently rebuilt by the audit at 7 cells)." |

## 4. Re-measurement of the 12 pre-declared runs (float64, own code: `audit/remeasure.py`)

Definitions implemented from the paper: the full logit table; h = min_x [f_y − max_{c≠y} f_c]; ‖θ‖_{2,q} = (Σ_i (Σ_t‖u_it‖² + ‖w_i‖²)^{q/2})^{1/q};
γ = h/‖θ‖^ν; β_j = 2p^{−ν} Σ_{x,c} f(x)[c] ρ^{−j(Σa−c)} (computed by binning the table on Σa − c, not by the package's FFT); A_j = |β_j|.
The relative differences are |audit − metrics.json| / |metrics.json|; "0" means bit-identical.

| run | h | ‖θ‖_{2,q} | γ | Δh | Δnorm | Δγ | max ΔA / max A | thr_10 audit/metrics | thr_20 audit/metrics |
|---|---|---|---|---|---|---|---|---|---|
| E1_nu5_p13_s1.25_seed2 | 4.96176 | 17.8569 | 2.73275e-06 | 5.4e-16 | 0 | 4.6e-16 | 3.6e-15 | 3/3 | 3/3 |
| E1_nu5_p17_s2.5_seed0 | 4.92323 | 43.4248 | 3.18831e-08 | 0 | 0 | 0 | 6.0e-15 | 3/3 | 1/1 |
| E1_nu5_p17_s2_seed1 | 4.00726 | 33.5645 | 9.40696e-08 | 0 | 0 | 0 | 2.9e-15 | 3/3 | 2/2 |
| E1_nu5_p17_s2_seed3 | 1.50472 | 35.3714 | 2.71768e-08 | 2.4e-15 | 0 | 2.4e-15 | 6.6e-15 | 4/4 | 4/4 |
| E3_product_nu3_p59_seed3_r1 | 9.47508 | 31.877 | 2.92517e-04 | 0 | 0 | 0 | 3.4e-15 | 29/29 | 29/29 |
| E3_product_nu5_p11_seed1 | 8.18223 | 13.6466 | 1.72879e-05 | 2.2e-16 | 0 | 3.9e-16 | 7.4e-16 | 5/5 | 5/5 |
| E6_marg_seed108 | 5.86209 | 31.9302 | 1.80073e-04 | 0 | 0 | 0 | 1.0e-15 | 6/6 | 6/6 |
| E6_marg_seed114 | 5.4393 | 31.8154 | 1.68901e-04 | 0 | 0 | 0 | 1.6e-15 | 5/5 | 5/5 |
| E7_nu4_p37_q2.7586_seed0 | 2.71578 | 39.0682 | 1.16574e-06 | 1.3e-15 | 0 | 1.3e-15 | 3.5e-15 | 11/11 | 8/8 |
| E7_nu4_p37_q3.0303_seed0 | 2.5325 | 34.6095 | 1.7651e-06 | 2.1e-15 | 0 | 2.0e-15 | 2.7e-15 | 12/12 | 11/11 |
| E8_nu3_p29_s1.25_seed2 | 8.28645 | 28.4035 | 3.61622e-04 | 0 | 0 | 0 | 2.7e-15 | 5/5 | 5/5 |
| E8_nu3_p71_s1.25_seed0 | 6.79297 | 48.5375 | 5.94058e-05 | 0 | 0 | 0 | 3.5e-15 | 11/11 | 11/11 |

Extra (not in the sample): the power path on E3_power_nu3_p23_seed2 and E3_power_nu3_p113_seed3 agrees to ≤ 1.2e-14.
The sample contains no ν = 3 E1 run, because the draw put all four E1 picks at ν = 5. The ν = 3 measurement code is
covered by the E3, E6 and E8 picks.

## 6. Numbers and statements that are safe to put in the paper

Each one was recomputed by the audit from the run directories. "REG" means preregistered; "post hoc" means Amendment A1
or an audit check. Qualifiers in brackets are required.

Theorem-level and controls
- The normalised margin never exceeded the universal ceiling B_p Γ_s(p) (n!·B_p Γ_s for power networks) in any of the 349 trained runs with finite γ. Maximum γ/ceiling: 0.73 (E1), 0.82 (E7), 0.76 (E8), 0.56 (E6), and 0.99999 at q = ν (E3).
- The constructed ν-block networks attain γ_IF exactly (to 1e-14), with A = y*, their near-tied offsets equal to the tight set, and the ablation bound satisfied. This was checked independently by the audit at 7 cells.
- The share-programme solutions (Γ, K*, y*, runner-up gaps) are exact at p ≤ 37 (vertex cache, confirmed by brute force at p ≤ 19). At p ≥ 43 they are best-found and not certified, except p = 43, s = 2.

Counts (E1, product networks, 126 converged runs; REG)
- Registered primary thr_10: K = K* in 34%, within ±1 in 66%, R²(identity) 0.39, OLS slope 1.36 [1.10, 1.62], mean signed error +0.94.
- Registered thr_20: K = K* in 33%, within ±1 in 70% [0.60, 0.78], R² 0.63, slope 1.01 [0.88, 1.15], mean signed error +0.16 [the choice of thr_20 as the headline is post hoc; it is the best of the 9 registered definitions, whose within-±1 range is 0.51-0.70].
- Baselines (audit, post hoc): the best constant count is within ±1 in 50% of the same runs; a permuted K* in 36%.
- Declared joint per-run match: 10.3%. Cells supported: 1 of 36.

Shares
- REG: the median TV(w, w*) is 0.42 (10th-90th percentile 0.10-0.62), against 0.52 for the random-allocation null and 0.64 for random init. The registered share test is not met.
- Post hoc: the programme efficiency of the learned allocation is 0.87 [0.85, 0.88] (E1), 0.91 (E6), 0.85 (E7) and 0.94 (E8). The nulls are 0.45 (random allocation) and 0.67 (random init). In E1, 12% of runs reach ≥ 0.95 and 1% reach ≥ 0.99.
- Post hoc, p ≤ 37: the TV to the nearest 1%-optimal vertex is 0.29 (E1), 0.29 (E7) and 0.08 (E8), against random-allocation nulls of 0.43, 0.46 and 0.38.

Dense endpoint (E3, q = ν)
- Dense and near-flat (thr_10 = h; min/max A 0.55-1.00; TV to uniform ≤ 0.034) for product ν = 3 at p = 23, 59, 113 (3-4 converged runs each), product ν = 4 at p = 23, and power ν = 3 at p = 23. γ reaches 0.88-1.00 of the ceiling.
- Not flat at ν = 5, even in converged runs. No converged runs at product ν = 4, p = 43/59, or power p = 59/113.
- The count alone is not diagnostic: random init already has thr_10 = h or h − 2.

Ablation (E4)
- For 765/768 (0.996) (run, j) pairs with w_j ≥ 0.01 in converged runs, rel. margin loss ≥ 1 − (1 − w_j)^s, against a permutation null of 0.42. This is a measurement. The median excess is 0.29.

Bottlenecks (E5)
- Not observed in trained networks: 1.6% per-run match, median Jaccard 0.18.

Equivariance (E6, ν = 3, p = 23, q = 2)
- The relabelled init gives the relabelled support in 12/12 pairs. The amplitude difference is ≤ 1.4e-6 of max A. The parameter drift is ≤ 4.4e-5 relative after 150k float32 steps.
- The frequency-activity marginals over 30 seeds are consistent with uniform (χ² = 6.77, dof 10, p = 0.75; low power).

Plateaus (E7, ν = 4, p = 37)
- REG: the plateau identity matched in 2/19 converged runs, and K = K* in 26%.
- Counts rank with s (Spearman 0.61-0.94 across definitions) but are compressed (slope 0.36-0.55).
- There is no detectable jump at the predicted switch points: within-plateau spread 0.112 vs adjacent-plateau spread 0.101.

K across p (E8 + E1, ν = 3)
- REG: the median thr_10 equals K* at 8/28 (p, s) cells, and trained K reaches 1.5-2.5× K* at p ≥ 59 (s = 1.5) and at p = 113.
- Post hoc: the median energy count K_2pct is within ±1 of K* at 23/28 cells up to p = 113, against 18/28 for the best constant per s.
- K*/log p lies in 1.60-2.67 (s = 1.25) and 0.83-1.38 (s = 1.5). Θ(log p) is not tested.

Junk
- Trained networks are not junk-free: the clean energy fraction over converged runs (E1, E6, E7, E8) is 0.47-1.00, median 0.97.

Do NOT write:
- "dense wherever GD converges".
- "theorems hold on every run" (for the ablation bound).
- "uniform marginals supported".
- "count unbiased" (without saying errors of both signs cancel).
- "follows K*(p) up to p = 113" (without "post hoc energy count, ±1, K* best-found beyond p = 37").
- "shares match y*".
- "the registered thr_20 shows this without amendment".

## 4b. Retraining 6 sampled runs (fresh pod, repository code)

Pod kypd49tv1vs09h: RTX 4090 SECURE at $0.74/h. This is a flagged fallback: 5090 secure was out of stock at every CUDA version.
- Created 00:00 IST, deleted 00:31 IST (0.52 h, $0.38). list-pods returned empty afterwards.
- Code: `core.remote sync` (pinned env: torch 2.13.0+cu130, byte-identical core/*.py).
- The configs were taken verbatim from configs/grid.json (identical to each run's config.json).
- Five detached queues ran concurrently. The original runs were trained in lockstep batches of 4-16; here the runs were solo, except that the two E6 runs were batched together.
- Pull (`audit/pull_retrain.sh`): every file was verified against a fresh pod-side sha256 list and against the queue manifests (24/24). Both lists are kept in `audit/retrain/`.
- Comparison: `audit/compare_retrain.py` -> `audit/retrain/compare.json`. Both sides were re-measured with the audit code.

| run | status orig / retrain | γ orig | γ retrain | rel. diff | thr_10 | thr_20 | K_2pct | same support (thr_10) | max ΔA / max A |
|---|---|---|---|---|---|---|---|---|---|
| E1_nu5_p13_s1.25_seed2 | ok / ok | 2.732755e-06 | 2.732753e-06 | 5.7e-7 | 3 / 3 | 3 / 3 | 3 / 3 | yes | 1.1e-7 |
| E3_product_nu5_p11_seed1 | ok / ok | 1.728790e-05 | 1.728790e-05 | 5.9e-8 | 5 / 5 | 5 / 5 | 5 / 5 | yes | 5.1e-8 |
| E6_marg_seed108 | ok / ok | 1.800732e-04 | 1.800732e-04 | 0 (bit-identical) | 6 / 6 | 6 / 6 | 6 / 6 | yes | 0 |
| E6_marg_seed114 | not_converged / not_converged | 1.689006e-04 | 1.689006e-04 | 0 (bit-identical) | 5 / 5 | 5 / 5 | 5 / 5 | yes | 0 |
| E7_nu4_p37_q2.7586_seed0 | not_converged / not_converged | 1.165736e-06 | 1.165726e-06 | 8.6e-6 | 11 / 11 | 8 / 8 | 8 / 8 | yes | 1.1e-6 |
| E8_nu3_p29_s1.25_seed2 | ok / ok | 3.616218e-04 | 3.616218e-04 | 6.8e-8 | 5 / 5 | 5 / 5 | 5 / 5 | yes | 7.5e-8 |

Verdict: **PASS**. All 6 retrained runs reproduce:
- the status;
- every count;
- the thr_10 support;
- γ to ≤ 8.6e-6 relative (the largest gap is on a not-converged run whose γ was still drifting);
- amplitudes to ≤ 1.1e-6 of max A.

The E6 runs are bit-identical. The others differ only at float32 GPU level.
