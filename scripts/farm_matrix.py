"""Matrix farm: 3 random primes x 3 random data seeds x 4 random init seeds = 36 runs.

Run:    caffeinate -i uv run python scripts/farm_matrix.py
Pause:  touch runs/PAUSE — stops cleanly after the current run finishes
        Ctrl-C           — kills the current run; it is redone on resume
Resume: run the script again. Completed runs (metrics.json present) are kept;
        an interrupted run is retrained from scratch under the same seed.

The whole matrix is drawn from OS entropy on the FIRST invocation and saved
to runs/farm_matrix_spec.json; later invocations load the spec, so resume
targets exactly the same 36 runs. Delete the spec file to draw a fresh matrix
(existing run dirs are never touched or reused — completed runs are skipped
only if they belong to the current spec).

Primes are sampled from PRIME_BAND (kept near 113 so the mainline recipe —
frac_train 0.3, wd 1.0, 30k epochs — transfers; compute scales as p^2).
Runs land in runs/p-<p>/seed<data_seed>/seed<init_seed>, each with the
spectral logger on (spectra.npz, every 50 epochs). Data seeds exclude 0-2,
so no mask collides with the existing p=113 zoo.

Runs are ordered slot-major (one init per cell, then the second, ...), so an
early stop still leaves a balanced matrix with equal inits per cell.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

N_PRIMES = 3
N_DATA_SEEDS = 2
INITS_PER_CELL = 2
PRIME_BAND = (79, 157)
NUM_EPOCHS = 30_000
SPECTRA_EVERY = 50
SAVE_EVERY = 1000

RUNS = Path("runs")
PAUSE = RUNS / "PAUSE"
SPEC = RUNS / "farm_matrix_spec.json"


def primes_in(lo: int, hi: int) -> list[int]:
    sieve = np.ones(hi + 1, dtype=bool)
    sieve[:2] = False
    for n in range(2, int(hi**0.5) + 1):
        if sieve[n]:
            sieve[n * n :: n] = False
    return [int(n) for n in np.nonzero(sieve)[0] if n >= lo]


def draw_spec() -> dict:
    # mx.random's global state is entropy-seeded per process — fresh every run
    band = primes_in(*PRIME_BAND)
    order = mx.random.permutation(len(band)).tolist()
    primes = sorted(band[i] for i in order[:N_PRIMES])
    data_seeds: set[int] = set()
    while len(data_seeds) < N_DATA_SEEDS:
        data_seeds.add(int(mx.random.randint(3, 100_000).item()))
    runs = [
        {
            "p": p,
            "data_seed": d,
            "slot": slot,
            "init_seed": int(mx.random.randint(1, 100_000).item()),
        }
        for slot in range(INITS_PER_CELL)
        for p in primes
        for d in sorted(data_seeds)
    ]
    return {"primes": primes, "data_seeds": sorted(data_seeds), "runs": runs}


def load_spec() -> dict:
    if SPEC.exists():
        spec = json.loads(SPEC.read_text())
        print(f"loaded spec: primes {spec['primes']}, data seeds {spec['data_seeds']}")
        return spec
    spec = draw_spec()
    SPEC.write_text(json.dumps(spec, indent=2))
    print(f"drew new matrix: primes {spec['primes']}, data seeds {spec['data_seeds']}")
    return spec


def run_dir_of(r: dict) -> Path:
    return RUNS / f"p-{r['p']}" / f"seed{r['data_seed']}" / f"seed{r['init_seed']}"


def report(run_dir: Path):
    z = np.load(run_dir / "spectra.npz")
    coeffs = np.abs(z["coeffs"][-1])
    order = np.argsort(coeffs)[::-1]
    top = "  ".join(f"{k + 1}:{coeffs[k]:.0f}" for k in order[:6])
    print(f"    test acc {z['test_acc'][-1]:.3f}   top |coeff|  {top}", flush=True)


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
        print(
            f"=== p-{r['p']} / seed{r['data_seed']} / slot {r['slot']} / "
            f"init_seed {r['init_seed']} ===  "
            f"({time.strftime('%H:%M:%S')}, {i}/{len(todo)} farmed this session)",
            flush=True,
        )
        cmd = [
            sys.executable, "-m", "grok.train",
            "--run-name", str(run_dir.relative_to("runs")),
            "--p", str(r["p"]),
            "--init-seed", str(r["init_seed"]),
            "--data-seed", str(r["data_seed"]),
            "--num-epochs", str(NUM_EPOCHS),
            "--save-every", str(SAVE_EVERY),
            "--spectra-every", str(SPECTRA_EVERY),
        ]
        proc = subprocess.Popen(cmd)
        try:
            ret = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            print(
                f"\ninterrupted — p-{r['p']}/seed{r['data_seed']}/seed{r['init_seed']} "
                "will be redone on resume"
            )
            return
        if ret != 0:
            print(f"run exited with code {ret} — stopping")
            return
        report(run_dir)
    print("matrix complete")


if __name__ == "__main__":
    main()
