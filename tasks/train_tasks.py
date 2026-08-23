"""Multi-task generality cohort: S5 composition + mod-96.

Prereg: tasks/README.md (committed before training).

On the pod (after scp of tasks/ and cloud/groups.py, or git pull):
    setsid nohup python3 -u tasks/train_tasks.py > tasks.log 2>&1 < /dev/null &
Smoke (CPU, ~2 min): python3 tasks/train_tasks.py --smoke
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("RUNS_DIR", str(ROOT / "runs_tasks"))
sys.path.insert(0, str(ROOT / "cloud"))

from train_semifinal_torch import RUNS, Config, report, train_batched  # noqa: E402

META_SEED = 20260817


def draw_seeds(n_masks, n_inits):
    rng = random.Random(META_SEED)
    data_seeds = sorted(rng.sample(range(30_000, 40_000), n_masks))
    init_seeds = sorted(rng.sample(range(100_000, 1_000_000), n_inits))
    return data_seeds, init_seeds


def batches(args):
    ds2, is2 = draw_seeds(2, 2)
    ds4, is4 = draw_seeds(2, 4)     # same masks (same rng draw order), 4 inits
    plan = []
    for frac in args.s5_fracs:
        jobs = []
        for wd in args.s5_wds:
            for ds in ds2:
                for is_ in is2:
                    rd = RUNS / f"s5_f{frac:g}_wd{wd:g}" / f"ds{ds}_is{is_}"
                    jobs.append((Config(
                        p=120, fn_name="s5", frac_train=frac, d_mlp=512,
                        weight_decay=wd, data_seed=ds, init_seed=is_,
                        num_epochs=args.s5_epochs,
                        save_every=args.s5_epochs), rd))
        plan.append((f"s5 frac {frac}", jobs, args.s5_spectra))
    jobs = []
    for wd in args.m96_wds:
        for ds in ds4:
            for is_ in is4:
                rd = RUNS / f"m96_f0.3_wd{wd:g}" / f"ds{ds}_is{is_}"
                jobs.append((Config(
                    p=96, frac_train=0.3, d_mlp=512, weight_decay=wd,
                    data_seed=ds, init_seed=is_,
                    num_epochs=args.m96_epochs,
                    save_every=args.m96_epochs), rd))
    plan.append(("mod-96", jobs, 5))
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s5-epochs", type=int, default=25_000)
    ap.add_argument("--m96-epochs", type=int, default=10_000)
    ap.add_argument("--s5-spectra", type=int, default=25)
    ap.add_argument("--compile", default="default",
                    choices=["default", "reduce-overhead", "off"])
    ap.add_argument("--s5-fracs", default="0.3,0.5")
    ap.add_argument("--s5-wds", default="0.5,1.0")
    ap.add_argument("--m96-wds", default="1.0,3.0")
    ap.add_argument("--no-m96", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.s5_fracs = [float(x) for x in args.s5_fracs.split(",")]
    args.s5_wds = [float(x) for x in args.s5_wds.split(",")]
    args.m96_wds = ([] if args.no_m96 else
                    [float(x) for x in args.m96_wds.split(",")])
    if args.smoke:
        args.s5_fracs, args.s5_wds, args.m96_wds = [0.5], [1.0], [1.0]
        args.s5_epochs = args.m96_epochs = 150
        args.s5_spectra, args.compile = 50, "off"

    t0 = time.time()
    for label, jobs, spectra_every in batches(args):
        pend = [(c, rd) for c, rd in jobs if not (rd / "spectra.npz").exists()]
        if not pend:
            continue
        print(f"batch {label}: {len(pend)} runs", flush=True)
        train_batched([c for c, _ in pend], [rd for _, rd in pend],
                      compile_mode=args.compile, spectra_every=spectra_every)
        for _, rd in pend:
            report(rd)

    if args.smoke:
        import numpy as np
        sys.path.insert(0, str(ROOT / "cloud"))
        from groups import MULT, INV, irrep_energy
        # labels sanity: a o a^-1 = identity index
        e = MULT[0, 0]
        assert all(MULT[i, INV[i]] == e for i in range(120))
        z = np.load(next((RUNS).glob("s5_*/*/spectra.npz")))
        assert z["energy"].shape[1] == 6, z["energy"].shape
        print("SMOKE OK: s5 labels + 6-irrep spectra verified")
    print(f"tasks done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
