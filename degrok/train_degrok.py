"""De-grok cohort: data-fraction sweep + fresh-prime anchor.

Prereg: degrok/README.md (committed before training). Fresh seeds
(META_SEED=20260815). One lockstep batch per (p, frac) — train-set size
changes tensor shapes; wd is per-run within a batch.

Run dirs: runs_degrok/p{p}_f{frac:g}_wd{wd:g}/ds{ds}_is{is}/

On the pod:
    GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --sparse \
        https://github.com/UndercoverMathGuy/grok.git && cd grok \
        && git sparse-checkout set degrok dial cloud
    pip install --break-system-packages safetensors
    setsid nohup python3 -u degrok/train_degrok.py > degrok.log 2>&1 < /dev/null &

Smoke (CPU, ~1 min): python3 degrok/train_degrok.py --smoke
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("RUNS_DIR", str(ROOT / "runs_degrok"))
sys.path.insert(0, str(ROOT / "cloud"))

from train_semifinal_torch import RUNS, Config, report, train_batched  # noqa: E402

META_SEED = 20260815
FRACS = [0.3, 0.5, 0.7, 0.9]
WDS = [1.0, 3.0]


def draw_seeds(n_masks=2, n_inits=4):
    rng = random.Random(META_SEED)
    data_seeds = sorted(rng.sample(range(20_000, 30_000), n_masks))
    init_seeds = sorted(rng.sample(range(100_000, 1_000_000), n_inits))
    return data_seeds, init_seeds


def batches(args):
    """Yield (label, [(cfg, run_dir), ...]) — one lockstep batch per (p, frac)."""
    data_seeds, init_seeds = draw_seeds()
    print(f"degrok seeds: data {data_seeds}  init {init_seeds}", flush=True)
    plan = [(113, f) for f in args.fracs] + [(127, f) for f in args.p127_fracs]
    for p, frac in plan:
        jobs = []
        for wd in args.wds:
            for ds in data_seeds:
                for is_ in init_seeds:
                    rd = RUNS / f"p{p}_f{frac:g}_wd{wd:g}" / f"ds{ds}_is{is_}"
                    if (rd / "spectra.npz").exists():
                        continue
                    jobs.append((Config(
                        p=p, frac_train=frac, d_mlp=512, weight_decay=wd,
                        data_seed=ds, init_seed=is_,
                        num_epochs=args.epochs, save_every=args.ckpt_every), rd))
        if jobs:
            yield f"p{p} frac {frac}", jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10_000)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--compile", default="default",
                    choices=["default", "reduce-overhead", "off"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.fracs, args.p127_fracs, args.wds = FRACS, [0.3], WDS
    if args.smoke:
        args.fracs, args.p127_fracs, args.wds = [0.9], [], [1.0]
        args.epochs, args.ckpt_every, args.compile = 200, 100, "off"

    t0 = time.time()
    n = 0
    for label, jobs in batches(args):
        print(f"batch {label}: {len(jobs)} runs", flush=True)
        train_batched([c for c, _ in jobs], [rd for _, rd in jobs],
                      compile_mode=args.compile, spectra_every=5)
        for _, rd in jobs:
            report(rd)
        n += len(jobs)
    print(f"degrok done: {n} trained, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
