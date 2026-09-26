# E0_tuning — LOG (chronological, IST)

This is the declared TUNING phase of NORMS C14. Every run goes through core/runner.py and is listed in RUNS.csv.
Objective: optimisation quality only (stationary regularised objective, stable normalised margin, loss plateau,
positive margin). No run is scored against the programme's predictions; no ceiling ratio is computed here.
Tuning runs are not evaluation data.

## Declared convergence criterion used for every tuning run (core/train.py DEFAULT_CONVERGENCE)
- grad_rel := ||g_CE + lam g_R|| / (||g_CE|| + lam ||g_R||) <= 1e-3 at the final step (stationarity; scale-free);
- |L64(T) - L64(T - 0.1 T)| / L64(T) <= 1e-4 (loss plateau; window starts after any lambda anneal);
- |gamma(T) - gamma(T - 0.1 T)| / |gamma(T)| <= 1e-3 (normalised margin stable);
- margin > 0, all values finite, final step reached.

## Rounds
- 2026-09-25 17:06: declared R1 (rounds.py): probe at the smallest cells (product nu=3 p=11 q=2 and q=3; product
  nu=4 p=7 q=2; product nu=5 p=7 q=2; power nu=3 p=11 q=3). Grid lr {0.5,1,2,4,8} x momentum {0,0.9} x init_scale
  {1, sqrt(p)} x lam {1e-3,1e-4}, seed 0, 20000 steps, float32 CPU, width 2*nu*h. Purpose: learn how the usable LR,
  momentum and init scale behave before scaling p. 200 runs, batches of 10.
- 2026-09-25 17:08: full test suite passed (101 passed, 1 skipped: no CUDA) -> core/TESTS.txt. Launched R1 (nohup, runner --parallel 5; one slot is held by the p=41 programme cache build).
- 2026-09-25 17:27: (empirical lead) R1 killed mid-run by the orchestrator (laptop compute ban). The 20 R1 runs without a final status are appended to RUNS.csv as 'crashed', note 'killed by orchestrator: laptop compute ban'. The 180 finished R1 runs (102 ok / 48 not_converged / 30 diverged) are kept as tuning data. R2 (declared in rounds.py, CPU) was never launched and is superseded by the GPU round R3 below.
- 2026-09-25 17:35: (empirical lead) declared R3 (GPU) in rounds.py BEFORE launch (see the comment block there: recipe structure, grid, and the mechanical selection rule). make_r3.py writes configs/R3_<cell>.json. GPU step-time benchmark (bench_gpu.py, no run dirs) on pod 0o915f2qtu4sou -> bench_gpu_4090.json (pulled later).
- 2026-09-25 17:41: R3 diagnostics (convergence only): fixed lambda collapses q = nu runs to the origin at nu = 4, 5; gamma_stable marginal at 30k steps. Declared R4 in rounds.py (lambda = lt * B_p^{q/nu}, lt in {1e-2, 3e-3}, 40k steps) BEFORE launch; launched on P1-P4 (P5 gets nu5 p23 after sync). core/remote.py launch fixed to detach from the ssh channel (the old command held the channel open until the queue ended; jobs were unaffected).
- 2026-09-25 18:16: R3/R4 pulled to runs/ with sha256 verification (core.remote pull, 9 tags, 208 runs, all checksums OK). REPORT.md and core/RECIPES.md written.
- 2026-09-25 18:17: NOTE: core.remote.pull appends the pod's WHOLE RUNS.csv for every tag, so pods that ran two tags (R3 and R4) contributed every R3/R4 status row twice (408 rows for 208 runs; identical duplicates, last-row semantics unaffected). Rows are not deleted (append-only history). Evaluation pulls use experiments/common/pull_fast.py, which merges each pod's RUNS.csv once.
