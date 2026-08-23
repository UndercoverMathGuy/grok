"""Shortcut cohort (prereg: shortcut/README.md).

The mainline mod-add transformer PLUS an explicit memorization pathway: a
zero-init per-pair table added to the logits with gain s (Config.table_scale).
Degree-1 table vs degree->=4 circuit => efficiency theories of grokking
predict a critical gain s* above which the table is the cheaper solution at
the wd-equilibrium margin and generalization never happens. We sweep s x wd
x frac and read (i) whether/when the circuit still generalizes and (ii) the
circuit-only Fourier committee.

Run dirs: runs_shortcut/<tag>/f{frac}_wd{wd}_s{s}/ds{ds}_is{is}/

Pod:    python3 shortcut/train_shortcut.py --tag pilot --scales 0,3,10,30,100
Smoke:  python3 shortcut/train_shortcut.py --smoke
"""
import argparse
import itertools
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("RUNS_DIR", str(ROOT / "runs_shortcut"))
sys.path.insert(0, str(ROOT / "cloud"))

from train_semifinal_torch import RUNS, Config, report, train_batched  # noqa: E402

META_SEED = 20260821


def draw_seeds(n_masks, n_inits):
    rng = random.Random(META_SEED)
    ds = sorted(rng.sample(range(50_000, 60_000), n_masks))   # fresh range
    is_ = sorted(rng.sample(range(100_000, 1_000_000), n_inits))
    return ds, is_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--scales", default="0,3,10,30,100")
    ap.add_argument("--wds", default="1")
    ap.add_argument("--fracs", default="0.3")
    ap.add_argument("--table-wd-mult", type=float, default=1.0)
    ap.add_argument("--masks", type=int, default=1)
    ap.add_argument("--inits", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20_000)
    ap.add_argument("--spectra-every", type=int, default=25)
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--max-batch", type=int, default=48)
    ap.add_argument("--compile", default="reduce-overhead",
                    choices=["default", "reduce-overhead", "off"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    scales = [float(x) for x in a.scales.split(",")]
    wds = [float(x) for x in a.wds.split(",")]
    fracs = [float(x) for x in a.fracs.split(",")]
    if a.smoke:
        a.tag, a.p, a.epochs, a.spectra_every, a.compile = "smoke", 29, 120, 20, "off"
        scales, wds, fracs, a.inits = [0.0, 30.0], [1.0], [0.5], 1
    ds, is_ = draw_seeds(a.masks, a.inits)
    t0 = time.time()
    for frac in fracs:          # one lockstep batch per frac (shared shapes)
        jobs = []
        for wd, s, d, i in itertools.product(wds, scales, ds, is_):
            rd = RUNS / a.tag / f"f{frac:g}_wd{wd:g}_s{s:g}" / f"ds{d}_is{i}"
            if (rd / "spectra.npz").exists():
                continue
            jobs.append((Config(p=a.p, frac_train=frac, weight_decay=wd,
                                table_scale=s, table_wd_mult=a.table_wd_mult,
                                data_seed=d, init_seed=i, num_epochs=a.epochs,
                                save_every=100 if a.smoke else 1000), rd))
        for j in range(0, len(jobs), a.max_batch):
            chunk = jobs[j:j + a.max_batch]
            print(f"batch frac {frac}: {len(chunk)} runs x {a.epochs} ep", flush=True)
            train_batched([c for c, _ in chunk], [rd for _, rd in chunk],
                          compile_mode=a.compile, spectra_every=a.spectra_every)
            for _, rd in chunk:
                try:
                    report(rd)
                except Exception as e:  # report() is the legacy committee printer
                    print(f"report skipped for {rd.name}: {e}")
    print(f"all done in {time.time() - t0:.0f}s", flush=True)

    if a.smoke:
        import numpy as np
        for _, rd in jobs:
            z = np.load(rd / "spectra.npz")
            print(rd.parent.name, "train_acc", z["train_acc"][-1], "full", z["train_acc_full"][-1],
                  "test", z["test_acc"][-1], "tnorm2", z["table_norm2"][-1], "cnorm2", z["circuit_norm2"][-1])
        print("SMOKE OK")


if __name__ == "__main__":
    main()
