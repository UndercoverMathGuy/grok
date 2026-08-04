"""Orth-flat cohort: embed_init='orthogonal' runs (E_k exactly flat), across
one baseline recipe and ten training-dynamics variants.

This reproduces the cohort behind claims 1 and 2 (orth-flat readout 0.66;
across-dynamics twins J 0.354 vs stranger 0.113). The original cohort was
launched with ad-hoc shell loops around grok.train; this script encodes the
exact grid recovered from the runs' config.json files.

Grid (all p=113, lr 1e-3, f64 loss, spectra every 100):
  orthWE          baseline dynamics, cells (ds 2034) x 4 seeds + (ds 3604) x 4
  phase2-tilt     loss_tilt 5
  eff-A           loss_tilt 15
  eff-B           loss_tilt 5, wd 2.5
  eff-D           loss_tilt 5, wd 0.4, 30k epochs
  eff-E           loss_tilt 5, grad_noise 0.2 until 4k
  eff-G           loss_cvar 0.05
  phase2-noise    grad_noise 0.3 until 6k
  phase2-noise2   grad_noise 0.2 until 16k
  combined        loss_tilt 5, adam_eps 1e-11
The dynamics variants share cell (ds 2034) x seeds {4242, 33428, 777,
11285} — that shared cell is what makes the claim-2 twin comparison
possible. (The short runs/phase2-probes/* probes are NOT reproduced here:
they were trained without the spectral logger and no claim analysis reads
them.)

Run:  caffeinate -i uv run python semifinal/training/train_orthflat_farm.py
Idempotent: runs with spectra.npz are skipped. Sequential only (Metal).
DRY_RUN=1 prints which runs would train instead of launching them.
"""
import os
import subprocess
import sys

from _shared import ROOT

DRY_RUN = bool(os.environ.get("DRY_RUN"))

SEEDS = [4242, 33428, 777, 11285]
DS = 2034

# (family, data_seed, init_seed, epochs, extra CLI args)
JOBS = []
for s in SEEDS:
    JOBS.append(("orthWE", DS, s, 30000 if s == 11285 else 20000, []))
for s in [4242, 66433, 777, 54735]:
    JOBS.append(("orthWE", 3604, s, 20000, []))
VARIANTS = [
    ("phase2-tilt", 20000, ["--loss-tilt", "5.0"], SEEDS),
    ("eff-A", 20000, ["--loss-tilt", "15.0"], SEEDS),
    ("eff-B", 20000, ["--loss-tilt", "5.0", "--weight-decay", "2.5"], SEEDS),
    ("eff-D", 30000, ["--loss-tilt", "5.0", "--weight-decay", "0.4"],
     [33428, 777, 11285]),
    ("eff-E", 20000, ["--loss-tilt", "5.0", "--grad-noise", "0.2",
                      "--grad-noise-until", "4000"], SEEDS),
    ("eff-G", 20000, ["--loss-cvar", "0.05"], SEEDS),
    ("phase2-noise", 20000, ["--grad-noise", "0.3",
                             "--grad-noise-until", "6000"], SEEDS),
    ("phase2-noise2", 20000, ["--grad-noise", "0.2",
                              "--grad-noise-until", "16000"], SEEDS),
    ("combined", 20000, ["--loss-tilt", "5.0", "--adam-eps", "1e-11"],
     [33428, 777, 11285]),
]
for fam, ep, extra, seeds in VARIANTS:
    for s in seeds:
        JOBS.append((fam, DS, s, ep, extra))


def launch(run_name, ds, iseed, epochs, extra):
    run_dir = ROOT / "runs" / run_name
    if (run_dir / "spectra.npz").exists():
        print(f"skip {run_name} (exists)", flush=True)
        return 0
    if DRY_RUN:
        print(f"WOULD TRAIN {run_name} ({epochs} epochs, {extra})", flush=True)
        return 0
    print(f"=== {run_name} ===", flush=True)
    cmd = [sys.executable, "-m", "grok.train",
           "--run-name", run_name,
           "--embed-init", "orthogonal",
           "--data-seed", str(ds),
           "--init-seed", str(iseed),
           "--num-epochs", str(epochs),
           "--save-every", "1000",
           "--spectra-every", "100"] + extra
    return subprocess.run(cmd, cwd=ROOT).returncode


for fam, ds, s, ep, extra in JOBS:
    ret = launch(f"{fam}/p-113/seed{ds}/seed{s}", ds, s, ep, extra)
    if ret != 0:
        print(f"run exited with code {ret} — stopping")
        sys.exit(ret)
print("ORTH-FLAT FARM DONE", flush=True)
