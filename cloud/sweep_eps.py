"""Adam-eps dial sweep (prereg: notes/prereg_eps_dial.md).

Vanilla legacy recipe (train_semifinal_torch Config defaults: p=113, frac
0.3, lr 1e-3, wd 1.0, betas (.9,.98), 1L d128 h4 mlp512, 50k epochs, full
batch) varying exactly ONE knob: adam_eps. The 24 seeds are the SAME 24 as
runs_sweep/base (sweep_dist.py, SEED_DRAW 20260829) so every cell is paired
with the existing eps=1e-8 base; eps=1e-8 is re-run here too, with the
per-tensor gradient tap, as the same-instrument control.

Cells (one lockstep batch each; adam_eps must agree within a batch), in
the order trained — most informative extremes first:

  e1e-10  e1e-06  e1e-12  e1e-04  e1e-09  e1e-07  e1e-05  e1e-08

Instrumentation: spectra every 100 epochs (committee readout), tensor_tap
(per-tensor |g| / sqrt(v) quantiles, fraction below eps, l1/l2 — the
eps-crossing chronology), checkpoints every 10k WITH Adam state.

Run dirs: $RUNS_DIR/<cell>/s<seed>/{config.json, spectra.npz, metrics.json}
Usage: python -u sweep_eps.py [--only e1e-10,e1e-06] [--dry-run]
"""
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_semifinal_torch import Config, train_batched, RUNS

SEED_DRAW = 20260829            # == sweep_dist.py; first 24 = its "base" cell
EPS_ORDER = [1e-10, 1e-6, 1e-12, 1e-4, 1e-9, 1e-7, 1e-5, 1e-8]


def base_seeds():
    rng = np.random.default_rng(SEED_DRAW)
    return [int(s) for s in rng.choice(100_000, 24 * 10, replace=False)][:24]


def cell_name(eps):
    return f"e{eps:.0e}".replace("e-0", "e-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated cell names")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--compile", default="default")
    ap.add_argument("--epochs", type=int, default=50_000)
    ap.add_argument("--spectra-every", type=int, default=100)
    args = ap.parse_args()
    seeds = base_seeds()
    print("seeds:", seeds, flush=True)
    only = set(args.only.split(",")) if args.only else None
    for eps in EPS_ORDER:
        cell = cell_name(eps)
        if only and cell not in only:
            continue
        cfgs, dirs = [], []
        for s in seeds:
            rd = RUNS / cell / f"s{s}"
            if (rd / "spectra.npz").exists():
                continue
            cfgs.append(Config(data_seed=s, init_seed=s, adam_eps=eps,
                               num_epochs=args.epochs, save_every=10_000))
            dirs.append(rd)
        print(f"== cell {cell} (adam_eps={eps:g}): {len(cfgs)} runs", flush=True)
        if args.dry_run or not cfgs:
            continue
        train_batched(cfgs, dirs, compile_mode=args.compile,
                      spectra_every=args.spectra_every, save_optim=True,
                      tensor_tap=True)


if __name__ == "__main__":
    main()
