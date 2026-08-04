"""Seed farm: keep training fresh-init zoo runs until told to stop.

Run:    uv run python scripts/farm_seeds.py [--data-seed N] [--max-runs K]
Pause:  touch runs/seed<N>/PAUSE — stops cleanly after the current run finishes
        Ctrl-C                   — kills the current run; it is redone on resume
Resume: run the script again. Completed runs (metrics.json present) are kept;
        an interrupted run is retrained from scratch under the same seed.
        Only runs the farm itself created (marked with a .farm file) are ever
        retrained — pre-existing dirs are never touched, only excluded
        from the seed draw.

The zoo is organized by train/test split: runs/seed<data_seed>/ holds all
runs sharing that split (runs/seed0 is the original zoo), each named
seed<init_seed> inside it. Every run has the spectral logger on: spectra.npz
holds per-frequency phase-locked coeffs + energies of the full-grid logits
every --spectra-every epochs. Init seeds are drawn from OS entropy and never
collide with existing dirs under the same data seed.
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

RUNS = Path("runs")


def seed_of(d: Path):
    m = re.fullmatch(r"seed(\d+)", d.name)
    return int(m.group(1)) if m else None


def report(run_dir: Path):
    z = np.load(run_dir / "spectra.npz")
    coeffs = np.abs(z["coeffs"][-1])
    order = np.argsort(coeffs)[::-1]
    top = "  ".join(f"{k + 1}:{coeffs[k]:.0f}" for k in order[:6])
    print(f"    test acc {z['test_acc'][-1]:.3f}   top |coeff|  {top}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-epochs", type=int, default=30000)
    parser.add_argument("--spectra-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()

    zoo = RUNS / f"seed{args.data_seed}"
    zoo.mkdir(parents=True, exist_ok=True)
    pause = zoo / "PAUSE"
    if pause.exists():
        pause.unlink()
        print("removed stale PAUSE file from a previous stop")

    rng = np.random.default_rng()  # OS entropy — fresh seeds every invocation
    done = 0
    while args.max_runs is None or done < args.max_runs:
        if pause.exists():
            print("PAUSE file found — stopping after a clean run boundary")
            break
        dirs = [d for d in zoo.iterdir() if d.is_dir() and seed_of(d) is not None]
        # mainline is init_seed 0; exclude it from the draw along with all seedN
        used = {seed_of(d) for d in dirs} | {0}
        incomplete = [
            d for d in dirs
            if (d / ".farm").exists() and not (d / "metrics.json").exists()
        ]
        if incomplete:
            seed = seed_of(incomplete[0])
            print(f"retraining interrupted {incomplete[0].name}")
        else:
            while (seed := int(rng.integers(1, 100_000))) in used:
                pass
        run_dir = zoo / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / ".farm").touch()  # ownership marker: safe to retrain if interrupted

        print(
            f"=== data_seed {args.data_seed} / init_seed {seed} ===  "
            f"({time.strftime('%H:%M:%S')}, {done} farmed)",
            flush=True,
        )
        cmd = [
            sys.executable, "-m", "grok.train",
            "--run-name", f"seed{args.data_seed}/seed{seed}",
            "--init-seed", str(seed),
            "--data-seed", str(args.data_seed),
            "--num-epochs", str(args.num_epochs),
            "--save-every", str(args.save_every),
            "--spectra-every", str(args.spectra_every),
        ]
        proc = subprocess.Popen(cmd)
        try:
            ret = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            print(f"\ninterrupted — seed {seed} will be redone on resume")
            return
        if ret != 0:
            print(f"seed {seed} exited with code {ret} — stopping")
            return
        done += 1
        report(run_dir)
    print(f"farmed {done} run(s)")


if __name__ == "__main__":
    main()
