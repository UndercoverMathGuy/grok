"""Distribution-sweep driver (prereg: notes/prereg_dist_sweep.md).

Vanilla legacy recipe throughout — Config DEFAULTS of train_semifinal_torch
(p=113, frac 0.3, lr 1e-3, wd 1.0, betas (.9,.98), 1L d128 h4 mlp512, 50k
epochs) — varying exactly ONE knob per cell:

  base   24 x baseline (fresh same-hardware control)
  wd0.3  24 x weight_decay 0.3
  wd3.0  24 x weight_decay 3.0
  f0.2   24 x frac_train 0.2
  f0.4   24 x frac_train 0.4
  f0.5   24 x frac_train 0.5
  p97    24 x p 97
  p127   24 x p 127
  p211   16 x p 211
  quad   24 x act_type Quad (x^2 MLP -> nu = 10 by the degree calculus)

Batches (lockstep; p and act_type must agree, wd/frac are per-run):
  b1 = base+wd0.3+wd3.0+f0.2 (96 runs)   b2 = f0.4+f0.5 (48, bigger N_tr)
  b3 = p97   b4 = p127   b5 = p211   b6 = quad

Run dirs: $RUNS_DIR/<cell>/s<seed>/{config.json, spectra.npz, ...}
Usage: python -u sweep_dist.py [--batch b1] [--dry-run] [--compile default]
"""
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_semifinal_torch import Config, train_batched, RUNS

SEED_DRAW = 20260829


def cells():
    rng = np.random.default_rng(SEED_DRAW)
    seeds = [int(s) for s in rng.choice(100_000, 24 * 10, replace=False)]
    pool = iter(seeds)
    def take(n): return [next(pool) for _ in range(n)]
    return {
        "base":  (take(24), {}),
        "wd0.3": (take(24), dict(weight_decay=0.3)),
        "wd3.0": (take(24), dict(weight_decay=3.0)),
        "f0.2":  (take(24), dict(frac_train=0.2)),
        "f0.4":  (take(24), dict(frac_train=0.4)),
        "f0.5":  (take(24), dict(frac_train=0.5)),
        "p97":   (take(24), dict(p=97)),
        "p127":  (take(24), dict(p=127)),
        "p211":  (take(16), dict(p=211)),
        "quad":  (take(24), dict(act_type="Quad")),
    }


BATCHES = {"b1": ["base", "wd0.3", "wd3.0", "f0.2"], "b2": ["f0.4", "f0.5"],
           "b3": ["p97"], "b4": ["p127"], "b5": ["p211"], "b6": ["quad"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="all",
                    choices=["all"] + sorted(BATCHES))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--compile", default="default")
    args = ap.parse_args()
    C = cells()
    for bname in (sorted(BATCHES) if args.batch == "all" else [args.batch]):
        cfgs, dirs = [], []
        for cell in BATCHES[bname]:
            ss, kw = C[cell]
            for s in ss:
                rd = RUNS / cell / f"s{s}"
                if (rd / "spectra.npz").exists():
                    continue                       # resume: skip finished
                cfgs.append(Config(data_seed=s, init_seed=s, save_every=1000,
                                   **kw))
                dirs.append(rd)
        print(f"== batch {bname}: {len(cfgs)} runs "
              f"({', '.join(BATCHES[bname])})", flush=True)
        if args.dry_run or not cfgs:
            continue
        train_batched(cfgs, dirs, compile_mode=args.compile)


if __name__ == "__main__":
    main()
