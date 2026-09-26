# heracles-emp — isolated empirical package

Why weight-decayed product/power networks learn sparse Fourier features on modular addition: exact
predictions from the share programme, training under NORMS.md, and every measurement and agreement metric.
Rules: `NORMS.md` (binding). Programme: `PLAN.md`. Theory used by the code: `core/THEORY_SPEC.md`.

## Setup (isolated environment)

The parent repository has its own pyproject/uv.lock. This package has its own `pyproject.toml`
(torch==2.13.0, numpy==2.5.1, scipy==1.18.0, pandas, pytest, matplotlib; Python 3.12) and its own
environment at `heracles-emp/.venv`. Always run from inside `heracles-emp/`:

```bash
cd heracles-emp
uv sync                 # creates/updates heracles-emp/.venv from uv.lock
uv run python -c "import sys; print(sys.prefix)"   # -> .../heracles-emp/.venv
```

From elsewhere, `uv run --project /path/to/heracles-emp python ...` uses the same environment.

## Test

```bash
cd heracles-emp
uv run pytest -v -s -W ignore::RuntimeWarning tests/      # full output is kept in core/TESTS.txt
```

## Library (core/)

| module | what |
|---|---|
| `programme.py` | exact share programme: vertex enumeration + cache (`core/data/vertices_p*.npz`, sha256 sidecars), `solve(p,s)`, `gamma`, `kstar`, `gamma_restricted(p,S,s)`, `tight_offsets`, `switch_points`, dilation tools (`dilate`, `canonicalize`); beyond the cache: `kcap` (certified support cap), `exact_capped`, `best_found`, `solve_any` (labelled 'exact (vertex cache)' / 'exact (certified K cap)' / 'best-found (not certified)') |
| `models.py` | product and power networks (batched full-table forward for any n), penalty, margin-form CE |
| `train.py` | GD / SGD(+momentum) on mean CE + λ Σ‖ω_i‖^q, float32 with TF32 off, float64 measurement, batched lockstep runs on CPU or CUDA, full provenance per run |
| `measure.py` | float64 table, margin, γ, ceilings, clean spectrum β_j, junk modes, energy split, per-neuron Fourier content, budget shares, coverage margins, per-offset margins, frequency ablation, counts |
| `metrics.py` | every agreement metric vs a prediction (after alignment over dilations and optimal vertices) + sweep-level helpers |
| `construct.py` | positive controls: ν-block product networks (any n, any y), Morwani dense power network (n=2, q=3) |
| `runner.py` | local CPU runner: machine-wide 6-slot fcntl lock, 1 thread per job, RUNS.csv bookkeeping |
| `remote.py` | GPU pods created by the orchestrator: sync code, launch a detached queue, poll, pull + verify checksums |

## Running experiments

Local CPU (always through the runner; at most 6 concurrent jobs machine-wide):

```bash
cd heracles-emp
uv run python -m core.runner experiments/EX BATCHES.json --parallel 6 --timeout 7200
```

`BATCHES.json = {"batches": [{"batch_id": ..., "runs": [{"run_id": ..., "config": {...}}]}]}`. Runs in one batch
share the shape (model, n, p, width, steps, dtype, device, optimizer, log_every) and train in lockstep.
Each run gets `experiments/EX/runs/<run_id>/{config.json, log.csv, final.pt, metrics.json}` (read-only once done).

GPU pod (created by the orchestrator only; direct ssh port):

```bash
uv run python -m core.remote sync   HOST PORT
uv run python -m core.remote launch HOST PORT experiments/EX BATCHES.json --tag T   # config "device": "cuda"
uv run python -m core.remote poll   HOST PORT experiments/EX --tag T
uv run python -m core.remote pull   HOST PORT experiments/EX --tag T               # verifies sha256 of every file
```

Programme caches: `uv run python -m core.programme build 43` (exact enumeration; cost grows like C(2h,h)/h).

## Frozen training recipes

`core/RECIPES.md` (from the E0 tuning phase, `experiments/E0_tuning/`).
