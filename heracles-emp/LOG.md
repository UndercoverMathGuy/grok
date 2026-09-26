# LOG (top level, orchestrator)

- 2026-09-25 16:33 IST: created heracles-emp/ with NORMS.md and PLAN.md. The author's decisions:
  - retrain everything; nothing from existing runs is used;
  - models are product (multilinear) and power (polynomial) only; optimisers are GD/SGD with an explicit l2-type penalty only;
  - the primary claim is ansatz validity (q/nu sweeps with every agreement metric), equivalent-s sweeps, and a q = nu dense demo;
  - also run CONSEQUENCES items 1 (plateaus), 3 (ablation), 7 (bottlenecks), 9 (equivariance) and 0/11 (log p); drop item 9 of the earlier list (identity/certificates).
- 2026-09-25 16:33 IST: dispatched the core agent (E0): library, tests, exact programme solver, runner with a 6-slot machine lock, and a tuning phase that produces frozen recipes. The experiment agents (E1/E1b, E3, E6, E7, E8; then E4/E5; then AUDIT) wait on E0.
- 2026-09-25 16:40 IST: the author amended the norms. Training is float32 with TF32 off (measurement stays float64). Pods are allowed (budget USD 40.77; every run ≤ 2 h; best price/performance). p is not restricted to small values. NORMS A6/F, the PLAN p-ranges and COMPUTE.md are updated, and the core agent has been messaged the changes.
- 2026-09-25 16:43 IST: the author made pods secure-cloud only. The default is a 5090 secure ($0.99/h), with a 4090 secure ($0.74/h) as fallback.
- 2026-09-25 17:17 IST: created CPU pod pu7uto001uy9o5 (cpu5g, 8 vCPU / 32 GB, secure, $0.368/h) for the Lean formalization of the paper proofs. It sits outside heracles-emp but draws on the same budget.
- 2026-09-25 17:24 IST: the core agent was stopped by the author's instruction: it had run tuning and a p=41 enumeration on the laptop, and those processes were killed. The core library is kept (101 tests passing). 4 x RTX 4090 secure pods were created (5090 secure out of stock) and handed to a new EMPIRICAL LEAD agent, which owns pods from now on. The full sweep must finish within 2 h.
- 2026-09-25 18:34 IST: deleted the Lean pod pu7uto001uy9o5 (about 1.6 h, about $0.58) after an independent clean rebuild of the local sources on the pod: 117 theorems, standard axioms only.
- 2026-09-25 18:47: INCIDENT (empirical lead error): pod mceuqia93fkz9s (P4) was deleted at ~18:46 while the E1 pull from it (pull_fast P4 E1: batches E1_nu3_p23/p31/p43/p83, 48 runs, queue tag P4s1, training finished) was still transferring (86 of ~192 files staged). The pod-side checksums were never saved, so the 86 files are UNVERIFIED; they are kept in experiments/E1/lost_P4_unverified/ and never used. The 48 runs are recorded in RUNS.csv as 'invalid' with this reason, and rerun under new ids <id>_r1 (configs/grid_reruns.json, plan experiments/common/plan_E1_lostP4_rerun.json) on P1/P2 at 18:47. E7 from the same pod had been pulled and verified at 18:43 (unaffected).
- 2026-09-25 17:25-18:55 IST (empirical lead): full sweep. Summary of actions (details in each experiments/*/LOG.md and COMPUTE.md):
  - pods: used the 4 RTX 4090 secure pods handed over; created 2 more 4090s (17:31) and, when those were not enough for the 2 h limit, 2 more 4090s + 2 RTX PRO 4500 (17:47); deleted 3 of these unused at 17:50 after the author flagged idle pods (5090 secure out of stock throughout).
  - E0: R1 (laptop) 20 runs marked crashed; GPU rounds R3 (128) and R4 (80); recipe frozen 17:55 (experiments/common/RECIPE.json, core/RECIPES.md); pod test suite 102/103 (1 root-permission artifact); CUDA batched-vs-solo gap 3e-7..6e-7 absolute (not bit-identical).
  - core changes: core/train.py init_relabel_u (E6) + test; core/remote.py launch detaches from the ssh channel. New helpers in experiments/common/ (predict, switches, predictions, build, launch, controls, pull_fast, alib, analysis).
  - evaluation launched 17:58 on 7 pods; slow-CPU pods rebalanced into parallel queues 18:07-18:15 (12 E3 ids stopped before training, rerun as _r1).
  - INCIDENT 18:46: pod mceuqia93fkz9s deleted while its E1 pull was running -> 48 E1 runs lost (marked invalid), rerun as _r1 on P1/P2 at 18:47.
- 2026-09-25 23:48 IST (orchestrator): the empirical lead stopped at ~19:05 on the session usage limit. By then every E* REPORT.md was written. AUDIT and the top-level summary were not done, and the lead was not revived.
  - At the author's request ("rework with the right stats"): Amendment A1, a POST-HOC re-scoring, was added to E1, E6, E7 and E8. It comprises energy-share counts (the earlier programme's definitions), programme efficiency, and a share match to the 1%-optimal vertex set, each with nulls. The ε = 5% set was dropped as vacuous (up to 62k orbits).
  - Files: experiments/common/amend_A1.py, experiments/common/write_A1_reports.py, experiments/E*/REPORT_A1.md and AMENDMENTS.md.
  - The REG reports are unchanged apart from a one-line pointer at the end.
  - Top-level RESULTS.md written.
  - A first A1 run with 200 null draws at ε ∈ {1%, 5%} ran more than 5 min on the laptop and was killed (NORMS F22). Final runtime: 7 s.
- 2026-09-26 00:32 IST (AUDIT agent): independent audit written to AUDIT.md; own scripts and outputs in audit/.
  - Checks:
    - prereg/recipe/TF32/scope;
    - completeness and status;
    - every headline number of every REPORT, REPORT_A1 and RESULTS, recomputed with own code (no numeric discrepancy);
    - float64 re-measurement of 12 pre-declared runs (numpy seed 20260926) to ≤ 7e-15;
    - brute-force programme check at p ≤ 19;
    - own ν-block controls;
    - retraining of 6 runs on audit pod kypd49tv1vs09h (RTX 4090 secure, $0.38, deleted 00:31; see COMPUTE.md): all counts, statuses and supports reproduced, γ within 8.6e-6.
  - Verdict: 3 FAIL, 9 CAVEAT.
    - FAIL: pulled_*.json hold no sha256 lists, so pull-time verification is not re-checkable (hashes now frozen in audit/sha256_frozen_20260926.json).
    - FAIL: RESULTS E3 "dense where GD converges" is false at ν=5.
    - FAIL: RESULTS Reading 3 "theorems hold on every run" is false for the ablation bound.
  - The run directories, RUNS.csv, PREREG/REPORT files, core/ and experiments/common/ were not modified.
- 2026-09-26 (orchestrator): RESULTS.md revised to apply AUDIT.md's two FAILs (E3 row, Reading 3) and its caveats (thr_20 choice post hoc, constant baseline 50%, E6 marginals 'not rejected', E7 Spearman range, E8 post-hoc label).

## 2026-09-26 08:35 IST — Amendment A2 (post hoc) + paper figures
- Author request: write the experiments section from the existing data; evidence to come from the sweep shapes, not the pooled count. Wrote experiments/common/amend_A2_shape.py.
  - A first version took 41 s on the laptop (null B = 2000, unvectorised), over the 30 s limit (NORMS F22). Vectorised the null and set B = 1000; it now runs in 19 s.
  - Included the E7 q = 2 cell in T2.
- Results, thr_20 (the range across the 6 count definitions is in A2_summary.json):
  - T1 E1+E8: 30/36 same direction (4 ties); null 0.38 (95%: 0.53); p = 0.001; slope 0.50.
  - T1 E7: 83/96; null 0.47; p = 0.001; slope 0.59.
  - T2: 5/7 (1 tie); p = 0.16 (4/7 to 7/7 across definitions). Not established.
  - T3: same-s |Δ| 1.12 vs across-s 1.31. Not resolved.
  - T4: clean fraction falls with s (Spearman −0.15 / −0.37 / −0.32 at ν = 3/4/5); E3 at q = ν has median 0.99.
- paper_figs/make_figs.py (matplotlib, pinned) writes fig_exponent, fig_prime, fig_spectra and fig_ablation_eff.
  - Its fig_prime per-cell thr_20 agreement is 20/28 vs 16/28 for the best constant per s. The audit's 23/28 vs 18/28 is for K_2pct.

## 2026-09-26 08:49 IST — paper figures redesigned (author: figures unreadable)
- make_figs.py redesigned; the data sources and the definition (thr_20) are unchanged:
  - Fig 1(b): the Δ-scatter is replaced by small multiples of count vs s for every ν=4/5 sweep with ≥2 converged cells (ν=4 p=59 omitted: 1 converged cell).
  - Fig 2: sorted profiles are replaced by Fourier bar spectra of one run per panel (rule: lowest converged seed).
  - Fig 4(a): measured loss vs the bound, against one diagonal (132 losses >1 drawn at 1).
  - Fig 4(b): strip plot of trained runs vs the per-cell nulls and the positive control.

## 2026-09-26 09:03 IST — paper: author cut Fig 4 (ablation + efficiency) and Section 5.4 (ablation, equivariance) from the experiments section. The efficiency numbers stay in the text of 5.1; paper_figs/fig_ablation_eff.pdf is still generated but is not in the paper.

## 2026-09-26 11:31 IST — correction found while verifying the paper's experiments text against E3/analysis/per_run.csv
- AUDIT §6 "min/max A 0.55–1.00; TV ≤ 0.034" and "0.88–1.00 of the ceiling" are per-cell MEDIANS, not per-run bounds.
  - Per run, over converged product ν=3 (p = 23, 59, 113), product ν=4 (p = 23) and power ν=3 (p = 23), min/max A goes down to 0.21 (p=23 seed1, p=59 seed1, p=113 seed3).
  - Per run, TV to uniform goes up to 0.075.
  - Per run, γ/ceiling goes down to 0.866.
  - Every one of those runs is dense (thr_10 = thr_20 = h).
- ν=5 at q=ν: "converged but not flat" was wrong.
  - Seeds 1 and 2 at p=11 are dense and flat (TV 6e-6).
  - p=11 seeds 0/3 miss 1 frequency; p=17 seeds miss 2–3; p=13 seed2 is dense but not flat (TV 0.14).
- RESULTS.md E3 row and the paper text (sections/experiments.tex) are corrected. AUDIT.md is left as written (independent record).

## 2026-09-26 11:45 IST — Amendment A3 (post hoc; see E1/E8 AMENDMENTS.md). Paper Fig 3, the experiments text and the appendix direction table are updated from A3_bestknown.json + A2_summary.json. Also noted: my recomputed best-constant baseline for pooled thr_20 within ±1 is 0.476 (constants 0–19), vs AUDIT's 0.50; the paper now says 'about half'.

## 2026-09-26 12:15 IST — same-s signed test added to amend_A3_bestknown.py (post hoc, author request)
- Result: in 17 same-(p,s) pairs across ν, 12 are within 1. The higher-ν cell is sparser in 13, equal in 3 and denser in 1 (sign test p=0.002).
- count−K*: +0.28 / −0.64 / −1.25 at ν=3/4/5.
- E1 efficiency median by ν: 0.89 / 0.88 / 0.79.
- Paper 5.3 now reports this instead of 'inconclusive'. A2 T3's unsigned |Δ| (1.12 vs 1.31) had hidden the systematic sign.

## 2026-09-26 12:24 IST — copied the 9 Table-1 allocations (q=2, p=59/71/113, ν=3/4/5) from heracles/FINAL_RUNS/LADDER into experiments/common/table1_allocations.json (one-time read outside this dir, for the paper's Table 1; re-evaluated here).

## 2026-09-26 13:04 IST — training-code audit (audit/TRAINING_CODE_AUDIT.md): no defect; 370/370 configs match the recipe. Disclosure fixes applied to the paper's Appendix E (tuning rule, convergence criterion, init sharing, E3 'crashed' wording, ν4 p59 divergence, float32 grad_rel, 349 run records incl. 6 duplicates). Selection check: the direction test including non-converged runs gives 36/41 (null 0.38, p=0.001) — added to A3 script.
