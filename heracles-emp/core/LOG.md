# core/LOG.md — chronological log of the core (E0) agent

All times IST (machine clock, `date`).

- 2026-09-25 16:39: read NORMS.md, PLAN.md, paper sections prelim/approach/cyclic/appendix, CONSEQUENCES.md, and
  (reference only, not imported) heracles/chain/common.py.
- 2026-09-25 16:37: wrote pyproject.toml (torch==2.13.0, numpy==2.5.1, scipy==1.18.0, pandas, pytest, matplotlib;
  requires-python ==3.12.*; [tool.uv] package=false) and .python-version=3.12. `uv sync` created heracles-emp/.venv.
  Verified isolation: `uv run python -c 'import sys; print(sys.prefix)'` -> heracles-emp/.venv, both from inside
  heracles-emp and via `uv run --project heracles-emp` from /tmp. torch 2.13.0, numpy 2.5.1, scipy 1.18.0, pandas 3.0.6.
- 2026-09-25 16:45: wrote core/programme.py (exact vertex enumeration over dilation-canonical supports with batched
  torch.linalg.solve_ex, least-squares refinement on the tight set with a rank check, dedup, orbit canonicalisation,
  per-p cache with sha256 sidecar; solve / gamma / kstar / gamma_restricted / tight_offsets / switch_points /
  canonicalize). Timing check: p=29 2.9M solves in 2.6 s, p=31 10.4M solves in 6.5 s (~1.6M solves/s, 1 thread).
- 16:45-16:58: built caches p=3..37 (core/data/logs/build_3_37.log). p=31: 10095 orbit reps / 151327 vertices, 7.7 s;
  p=37: 110019 reps / 1979271 vertices, 646 s (1 thread).
- ~16:50: NORMS.md was amended by the author (A6: float32 training with TF32 disabled, float64 measurement; F20-F24:
  GPU pods created by the orchestrator, batched lockstep runs with a batched==solo test, 2 h per-run cap;
  PLAN.md: larger p). Implementation changes that followed are logged below.
- 16:55: wrote core/models.py (batched product/power full-table forward; the product CP contraction is split into a
  left block of operands and a right block of operands+readout, so the main cost is one bmm), core/measure.py,
  core/construct.py. Sanity: nu-blocks attain gamma_IF to ~1e-15 relative and measure back beta = y*; the Morwani
  construction attains 2 B_p p/(p-1) to ~1e-15 (p = 5..13).
- 17:00: added to programme.py: kcap (paper cap (2/m)^{1/(s-1)} and a sharper root-based cap, both certified),
  canonical_supports_upto / exact_capped (exact enumeration of supports <= K_max for any p), mm_descent
  (majorise-minimise by LP, cost never increases), best_found (multistart + add/swap moves, labelled
  'best-found (not certified)'), solve_any (picks exact cache / certified cap / best-found and labels it).
  First validation: 150 starts matched exact at 15/16 (p, s) points; the miss (p=31, s=2, 1.2% low) is found with
  400 starts; added swap moves and set the default to 400 starts. SELF_SHA256 of programme.py is recorded at import.
- 17:00: wrote core/train.py (batched lockstep GD/SGD+momentum, schedules, precision flags asserted and recorded,
  float64 logging, convergence verdict, immutable read-only run dirs), core/metrics.py, core/runner.py,
  core/remote.py.
  - Design choice (documented in models.ce_margin_form): CE is computed as softplus(logsumexp_{c!=y}(f_c - f_y)),
    algebraically identical to the standard CE; the standard float32 form rounds the correct-class gradient to 0 once
    the wrong-class mass is < ~6e-8 (tested). f_y is taken by a masked sum instead of gather, so the backward pass is
    elementwise on every device.
  - Performance (per the user's note not to trade GPU speed for reproducibility): no torch.use_deterministic_algorithms;
    no per-step host-device synchronisation (schedules precomputed in chunks on the device; gradient statistics,
    float64 measurements and divergence checks only at log steps). Smoke test at p=11: 4 runs x 4000 steps 6.3 s -> 1.5 s.
- 17:05: tests written (tests/test_programme.py, test_models_measure.py, test_metrics.py, test_train.py,
  test_runner.py). Bugs/fixes found while testing:
  - test_switch_points_consistent compared the plateau representative's cost with 1 at the arithmetic mid-point in s,
    but switch_points normalises at the mid-point in r = 1/s (test bug; the test now compares programme values).
  - budget-normalised metrics of an exact construction agree only to ~1e-8, not 1e-12: off-support float64 FFT
    noise (~1e-17) raised to 1/s enters the budget. Tolerances in test_metrics set to 1e-6 with this reason; the
    'exact' flag for the (float) participation-ratio counts uses |diff| <= 1e-6.
- 17:06: launched the p=41 exact cache build through the runner (one slot; log core/data/logs/build_41.log).
- 17:08: full suite: 101 passed, 1 skipped (CUDA-only test) -> core/TESTS.txt. Reports written by the tests:
  core/data/test_reports/batched_vs_solo_cpu.json (CPU float32 batched vs solo: NOT bit-identical; max relative weight
  difference 1.7e-7..3.5e-7 after 600 steps, gamma relative difference 2e-8..6e-8) and fp32_vs_fp64.json (same init:
  gamma relative gap 1.0e-7..5.7e-7 on three small p=7 runs).
- 17:08: E0 tuning round R1 launched (see experiments/E0_tuning/LOG.md).
