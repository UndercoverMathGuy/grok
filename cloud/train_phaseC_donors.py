"""Phase C stage 0: the natural donor cohort, trained fresh on the pod.

Nothing is copied up. These 8 runs are generated here and then serve three
roles, which is why they come first:

  - the S-side base inits: `make_arms.py --crossed` compiles the shared
    content pool onto their epoch-0 checkpoints;
  - the S-side no-install controls' base;
  - the L-side donors: the data reveal resumes from their save_every=1000
    checkpoints, so the donor trajectory is a real mainline natural run at
    wd = 1.0 (not the wd = 3.0 `boost/` donors, 2 of whose 8 never groked).

Layout matches what `compiler/make_arms.natural_bases()` discovers:
runs_torch/p-113/seed<data>/seed<init>/, family "p-113", embed_init normal.

    python -u cloud/train_phaseC_donors.py            # ~8 x 30k run-epochs
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))

from train_semifinal_torch import Config, train_batched

META_SEED = 20260816
N_DATA, N_INIT = 2, 4


def draw_seeds(meta=META_SEED):
    """Fresh draws, fixed by META_SEED: the donors are Phase C's own natural
    cohort, not a post-hoc pick from an existing one."""
    rng = np.random.default_rng(meta)
    data = sorted(rng.choice(np.arange(40_000, 50_000), N_DATA,
                             replace=False).tolist())
    init = sorted(rng.choice(np.arange(10_000, 99_999), N_INIT,
                             replace=False).tolist())
    return data, init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30_000)
    ap.add_argument("--spectra-every", type=int, default=25)
    # 500 so the L-side ladder can start at t=500; the ladder runs
    # 500..10000, which spans pre-lock, post-lock-pre-grok and post-grok.
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--loss", default="f32stable", choices=["f32stable", "f64"])
    ap.add_argument("--compile", dest="compile_mode", default="default")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.save_every, args.spectra_every = 60, 20, 10
        args.compile_mode = "off"

    data_seeds, init_seeds = draw_seeds()
    print(f"META_SEED={META_SEED}  data_seeds={data_seeds}  "
          f"init_seeds={init_seeds}")

    jobs = []
    for ds in data_seeds:
        for is_ in init_seeds:
            rd = ROOT / "runs_torch" / "p-113" / f"seed{ds}" / f"seed{is_}"
            if (rd / "spectra.npz").exists():
                print(f"skip (done): {rd}")
                continue
            jobs.append((Config(p=113, fn_name="add", frac_train=0.3,
                                data_seed=ds, init_seed=is_, d_model=128,
                                num_heads=4, d_mlp=512, act_type="ReLU",
                                lr=1e-3, weight_decay=1.0,
                                betas=(0.9, 0.98), adam_eps=1e-8,
                                num_epochs=args.epochs,
                                save_every=args.save_every), rd))
    if not jobs:
        print("all donors present")
        return

    for i in range(0, len(jobs), args.batch):
        chunk = jobs[i:i + args.batch]
        print(f"=== donor batch {i // args.batch}: {len(chunk)} runs, "
              f"{chunk[0][0].num_epochs} epochs", flush=True)
        train_batched([c for c, _ in chunk], [rd for _, rd in chunk],
                      spectra_every=args.spectra_every, loss=args.loss,
                      compile_mode=args.compile_mode)
    print("DONORS DONE")


if __name__ == "__main__":
    main()
