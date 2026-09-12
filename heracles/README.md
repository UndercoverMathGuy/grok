# heracles

Isolated line started 2026-09-11: the max-margin count theory in a Morwani-style cubby, with
constant-geometry optimizers only. Nothing here imports from the rest of the repo; nothing from here is
written outside `heracles/`.

- THEORY.md  — the l_q generalisation: K*(p, nu, q) = argmax_K M_K(p) / K^{nu/q}; which optimizers qualify.
- PLAN.md    — cohorts R (recipe sweep) and H (p, nu, wd, frac, geometry, Wei limit) with predictions.
- kstar.py   — the ladder; kstar.json holds the numbers PLAN.md quotes.
- train.py   — batched trainer for the ReLU MLP family; optimizers gd / gdm / sgd / lq / sign / signm.
- score.py   — interface-ablation count, legacy counts, gamma_2 and gamma_q per run.
- jobs.py    — cohort definitions and the resume-safe runner.
- pod.sh     — deploy helper.
- scores/    — scored results (committed). runs/ — raw runs (gitignored).

Smoke: `uv run --with torch python heracles/jobs.py --smoke` (CPU, p = 29, seconds).
