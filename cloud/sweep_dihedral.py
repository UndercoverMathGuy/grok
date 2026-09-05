"""Dihedral D_n multiplication (prereg: notes/prereg_dihedral.md, 2026-09-05).

Same recipe family as the S5 tasks arm (runs_tasks/s5_*): 1L ReLU
transformer d128 h4 mlp512, AdamW lr 1e-3 betas (.9,.98) eps 1e-8, full
batch, MLX-semantics; fn_name 'd<n>' with p = 2n tokens. Per-irrep isotypic
energy is recorded in spectra.npz (coeffs = sqrt(energy) over the 1 + (n-1)/2
nontrivial irreps: sign first, then rho_1..rho_(n-1)/2).

Cells: frac x wd, 8 seeds each. Run dirs: $RUNS_DIR/d<n>_f<frac>_wd<wd>/s<seed>
Usage: python -u sweep_dihedral.py [--n 113] [--fracs 0.5 0.7] [--wds 1] [--epochs 30000]
"""
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_semifinal_torch import Config, train_batched, RUNS

SEED_DRAW = 20260905


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=113)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.5, 0.7])
    ap.add_argument("--wds", type=float, nargs="+", default=[1.0])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30_000)
    ap.add_argument("--spectra-every", type=int, default=100)
    ap.add_argument("--compile", default="default")
    ap.add_argument("--loss", default="f32stable")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED_DRAW)
    seeds = [int(s) for s in rng.choice(100_000, args.seeds, replace=False)]
    print("seeds:", seeds, flush=True)
    for frac in args.fracs:
        for wd in args.wds:
            cell = f"d{args.n}_f{frac:g}_wd{wd:g}"
            cfgs, dirs = [], []
            for s in seeds:
                rd = RUNS / cell / f"s{s}"
                if (rd / "spectra.npz").exists():
                    continue
                cfgs.append(Config(p=2 * args.n, fn_name=f"d{args.n}", frac_train=frac,
                                   data_seed=s, init_seed=s, weight_decay=wd,
                                   num_epochs=args.epochs, save_every=args.epochs))
                dirs.append(rd)
            if not cfgs:
                continue
            print(f"[{cell}] {len(cfgs)} runs", flush=True)
            train_batched(cfgs, dirs, log_every=500, spectra_every=args.spectra_every,
                          loss=args.loss, compile_mode=args.compile)
    print("SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
