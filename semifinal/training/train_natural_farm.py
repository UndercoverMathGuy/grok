"""Natural-run ensemble: 3 primes x 2 data seeds x 2 init seeds = 12 runs
per matrix (the SEMIFINAL natural cohort combines this matrix farm with the
earlier runs/og_seed0 and runs/seed{0,1,2} zoos, 24 runs total).

Run:    caffeinate -i uv run python semifinal/training/train_natural_farm.py
Pause:  touch runs/PAUSE — stops cleanly after the current run finishes
Resume: run again; completed runs (metrics.json present) are skipped.

The matrix is drawn from OS entropy on the FIRST invocation and saved to
runs/farm_matrix_spec.json; later invocations load the spec, so resume
targets exactly the same runs. NOTE: this reproduces the cohort
*statistically*, not the exact named seeds in runs/ — those were one draw
of this same procedure. Runs land in runs/p-<p>/seed<data>/seed<init> with
the spectral logger on (spectra.npz every 50 epochs).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from _shared import ROOT

N_PRIMES = 3
N_DATA_SEEDS = 2
INITS_PER_CELL = 2
PRIME_BAND = (79, 157)
NUM_EPOCHS = 30_000
SPECTRA_EVERY = 50
SAVE_EVERY = 1000

RUNS = ROOT / "runs"
PAUSE = RUNS / "PAUSE"
SPEC = RUNS / "farm_matrix_spec.json"


def primes_in(lo, hi):
    sieve = np.ones(hi + 1, dtype=bool)
    sieve[:2] = False
    for n in range(2, int(hi**0.5) + 1):
        if sieve[n]:
            sieve[n * n:: n] = False
    return [int(n) for n in np.nonzero(sieve)[0] if n >= lo]


def draw_spec():
    # mx.random's global state is entropy-seeded per process — fresh every run
    band = primes_in(*PRIME_BAND)
    order = mx.random.permutation(len(band)).tolist()
    primes = sorted(band[i] for i in order[:N_PRIMES])
    data_seeds = set()
    while len(data_seeds) < N_DATA_SEEDS:
        data_seeds.add(int(mx.random.randint(3, 100_000).item()))
    runs = [
        {"p": p, "data_seed": d, "slot": slot,
         "init_seed": int(mx.random.randint(1, 100_000).item())}
        for slot in range(INITS_PER_CELL)
        for p in primes
        for d in sorted(data_seeds)
    ]
    return {"primes": primes, "data_seeds": sorted(data_seeds), "runs": runs}


def load_spec():
    if SPEC.exists():
        spec = json.loads(SPEC.read_text())
        print(f"loaded spec: primes {spec['primes']}, data seeds {spec['data_seeds']}")
        return spec
    spec = draw_spec()
    SPEC.write_text(json.dumps(spec, indent=2))
    print(f"drew new matrix: primes {spec['primes']}, data seeds {spec['data_seeds']}")
    return spec


def run_dir_of(r):
    return RUNS / f"p-{r['p']}" / f"seed{r['data_seed']}" / f"seed{r['init_seed']}"


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    if PAUSE.exists():
        PAUSE.unlink()
        print("removed stale PAUSE file from a previous stop")

    spec = load_spec()
    todo = [r for r in spec["runs"] if not (run_dir_of(r) / "metrics.json").exists()]
    total = len(spec["runs"])
    print(f"{total - len(todo)}/{total} runs already complete, {len(todo)} to go")

    for i, r in enumerate(todo):
        if PAUSE.exists():
            print("PAUSE file found — stopping at a clean run boundary")
            return
        run_dir = run_dir_of(r)
        print(f"=== p-{r['p']} / seed{r['data_seed']} / slot {r['slot']} / "
              f"init_seed {r['init_seed']} ===  ({time.strftime('%H:%M:%S')})",
              flush=True)
        cmd = [
            sys.executable, "-m", "grok.train",
            "--run-name", str(run_dir.relative_to(RUNS)),
            "--p", str(r["p"]),
            "--init-seed", str(r["init_seed"]),
            "--data-seed", str(r["data_seed"]),
            "--num-epochs", str(NUM_EPOCHS),
            "--save-every", str(SAVE_EVERY),
            "--spectra-every", str(SPECTRA_EVERY),
        ]
        ret = subprocess.run(cmd, cwd=ROOT).returncode
        if ret != 0:
            print(f"run exited with code {ret} — stopping")
            return
    print("matrix complete")


if __name__ == "__main__":
    main()
