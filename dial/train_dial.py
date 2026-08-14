"""Capacity-dial sweep: d_mlp x weight_decay grid on mainline mod-add.

Prereg and design: dial/README.md (committed before training). Reuses the
verified torch lockstep trainer unchanged — weight decay is per-run inside a
batch, so each width is ONE lockstep batch of all wd x seed runs (split only
by --batch for memory at large widths).

Run dirs: runs_dial/w{d_mlp:04d}_wd{wd:g}/ds{data_seed}_is{init_seed}/
Idempotent: a run dir with spectra.npz is skipped, so re-launching after an
interruption trains only what is missing.

On the pod:
    git clone https://github.com/UndercoverMathGuy/grok.git && cd grok
    pip install torch safetensors numpy   # runpod pytorch image: present
    nohup python -u dial/train_dial.py > dial.log 2>&1 &

Smoke (CPU, ~1 min): python3 dial/train_dial.py --smoke
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# RUNS_DIR must be set before the trainer module reads it at import time.
os.environ.setdefault("RUNS_DIR", str(ROOT / "runs_dial"))
sys.path.insert(0, str(ROOT / "cloud"))

from train_semifinal_torch import RUNS, Config, report, train_batched  # noqa: E402

META_SEED = 20260814
WIDTHS = [128, 256, 512, 1024, 2048, 4096]
WDS = [0.1, 0.3, 1.0, 3.0]


def draw_seeds(n_masks=2, n_inits=4):
    """Cohort seeds drawn from one meta-seed (house discipline: drawn, not
    hand-picked; kept whatever comes out). Data seeds from [10000, 20000) are
    disjoint from every spent mask by construction."""
    rng = random.Random(META_SEED)
    data_seeds = sorted(rng.sample(range(10_000, 20_000), n_masks))
    init_seeds = sorted(rng.sample(range(100_000, 1_000_000), n_inits))
    return data_seeds, init_seeds


def run_dir(width, wd, ds, is_):
    return RUNS / f"w{width:04d}_wd{wd:g}" / f"ds{ds}_is{is_}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default=",".join(str(w) for w in WIDTHS))
    ap.add_argument("--wds", default=",".join(str(w) for w in WDS))
    ap.add_argument("--masks", type=int, default=2)
    ap.add_argument("--inits", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=50_000)
    ap.add_argument("--ckpt-every", type=int, default=25_000)
    ap.add_argument("--batch", type=int, default=32,
                    help="max runs per lockstep batch (widths cannot mix — "
                         "stacked weights need equal shapes; 32 = every "
                         "width is one full batch, ~12-14GB peak at 4096)")
    ap.add_argument("--compile", default="default",
                    choices=["default", "reduce-overhead", "off"])
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU sanity pass: 1 width, 2 wds, 2 runs, "
                         "300 epochs, compile off")
    args = ap.parse_args()

    if args.smoke:
        args.widths, args.wds = "128", "1.0,0.1"
        args.masks = args.inits = 1
        # two runs via two wds on the same (mask, init) pair
        args.epochs, args.ckpt_every, args.compile = 300, 100, "off"

    widths = [int(w) for w in args.widths.split(",")]
    wds = [float(w) for w in args.wds.split(",")]
    data_seeds, init_seeds = draw_seeds(args.masks, args.inits)
    print(f"dial sweep: widths {widths}  wds {wds}\n"
          f"data seeds {data_seeds}  init seeds {init_seeds}  "
          f"epochs {args.epochs}", flush=True)

    t0 = time.time()
    total = done_before = trained = 0
    for width in widths:
        jobs = []
        for wd in wds:
            for ds in data_seeds:
                for is_ in init_seeds:
                    total += 1
                    rd = run_dir(width, wd, ds, is_)
                    if (rd / "spectra.npz").exists():
                        done_before += 1
                        continue
                    jobs.append((Config(
                        d_mlp=width, weight_decay=wd, data_seed=ds,
                        init_seed=is_, num_epochs=args.epochs,
                        save_every=args.ckpt_every), rd))
        if not jobs:
            print(f"width {width}: all runs already complete", flush=True)
            continue
        chunk = args.batch
        for i in range(0, len(jobs), chunk):
            part = jobs[i:i + chunk]
            cfgs = [c for c, _ in part]
            rds = [rd for _, rd in part]
            print(f"width {width}: batch of {len(part)} "
                  f"({i + len(part)}/{len(jobs)} pending)", flush=True)
            train_batched(cfgs, rds, compile_mode=args.compile)
            for rd in rds:
                report(rd)
            trained += len(part)

    print(f"dial sweep done: {trained} trained, {done_before} already "
          f"present, {total} total, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
