# RECIPES — frozen training recipes (from E0 R3/R4; see experiments/E0_tuning/REPORT.md)

Recipe id FROZEN-2026-09-25-1755 (frozen 2026-09-25 17:55 IST). Machine-readable: experiments/common/RECIPE.json.
Common: full-batch GD, heavy-ball momentum 0.9, init entries N(0,1) (init_scale = sqrt p), width 2 nu h, lr = c p,
lambda = lt * B_p^{q/nu} with B_p = 2 (2 p nu)^{-nu/2}, float32 with TF32 off, float64 measurement, log every 1000 steps,
convergence = core/train.py DEFAULT_CONVERGENCE.

| model, nu | c | lt | steps |
|---|---|---|---|
| product, 3 | 0.4 | 3e-3 | 150000 |
| power, 3 | 0.1 | 1e-2 | 150000 |
| product, 4 | 0.28 | 3e-3 | 150000 (90000 at p = 59) |
| product, 5 | 0.035 | 3e-3 | 300000 (p <= 17 only) |
