"""Transplant/handoff cohort. Prereg: transplant/README.md.

Phase 1: 8 donors (degrok seeds, wd 3, frac 0.3), 3000 epochs, ckpt/50.
Phase 2: build hybrid inits (recipient ckpt-at-t with donor's embed.W_E),
train 32 hybrids + 16 self-restart controls for 3000 epochs, one batch.

On the pod (after scp of transplant/):
    setsid nohup python3 -u transplant/train_transplant.py > transplant.log 2>&1 < /dev/null &
Smoke (CPU): python3 transplant/train_transplant.py --smoke
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("RUNS_DIR", str(ROOT / "runs_transplant"))
sys.path.insert(0, str(ROOT / "cloud"))

from train_semifinal_torch import RUNS, Config, report, train_batched  # noqa: E402

META_SEED = 20260815  # same draws as degrok
SWAP_TS = [0, 50, 150, 500]
CTRL_TS = [150, 500]


def draw_seeds(n_masks=2, n_inits=4):
    rng = random.Random(META_SEED)
    data_seeds = sorted(rng.sample(range(20_000, 30_000), n_masks))
    init_seeds = sorted(rng.sample(range(100_000, 1_000_000), n_inits))
    return data_seeds, init_seeds


def ordered_pairs(init_seeds):
    a, b, c, d = init_seeds
    return [(a, b), (b, a), (c, d), (d, c)]


def ck(rd, ep):
    return rd / "checkpoints" / f"epoch_{ep:05d}.safetensors"


def make_hybrid(donor_rd, recip_rd, t, out_path):
    from safetensors.torch import load_file, save_file
    state = load_file(str(ck(recip_rd, t)))
    state["embed.W_E"] = load_file(str(ck(donor_rd, t)))["embed.W_E"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--compile", default="default",
                    choices=["default", "reduce-overhead", "off"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    swap_ts, ctrl_ts = SWAP_TS, CTRL_TS
    save_every = 50
    if args.smoke:
        args.epochs, args.width, args.compile = 120, 128, "off"
        swap_ts, ctrl_ts, save_every = [0, 40], [40], 20

    data_seeds, init_seeds = draw_seeds()
    cfg_kw = dict(d_mlp=args.width, weight_decay=3.0, num_epochs=args.epochs,
                  save_every=save_every)
    print(f"transplant seeds: data {data_seeds}  init {init_seeds}", flush=True)

    # phase 1: donors
    donor_rd = lambda ds, is_: RUNS / "donor" / f"ds{ds}_is{is_}"
    jobs = [(Config(data_seed=ds, init_seed=is_, **cfg_kw), donor_rd(ds, is_))
            for ds in data_seeds for is_ in init_seeds
            if not (donor_rd(ds, is_) / "spectra.npz").exists()]
    if jobs:
        print(f"donors: {len(jobs)}", flush=True)
        train_batched([c for c, _ in jobs], [rd for _, rd in jobs],
                      compile_mode=args.compile, spectra_every=5)
        for _, rd in jobs:
            report(rd)

    # phase 2: hybrids + controls (one batch; warm-start via init_from)
    hjobs = []  # (cfg, run_dir, init_file)
    for ds in data_seeds:
        for a, b in ordered_pairs(init_seeds):
            for t in swap_ts:
                rd = RUNS / f"hyb_t{t}" / f"A{a}toB{b}_ds{ds}"
                if (rd / "spectra.npz").exists():
                    continue
                init_f = RUNS / "_inits" / f"t{t}_A{a}_B{b}_ds{ds}.safetensors"
                make_hybrid(donor_rd(ds, a), donor_rd(ds, b), t, init_f)
                hjobs.append((Config(data_seed=ds, init_seed=b, **cfg_kw), rd, init_f))
        for is_ in init_seeds:
            for t in ctrl_ts:
                rd = RUNS / f"self_t{t}" / f"ds{ds}_is{is_}"
                if (rd / "spectra.npz").exists():
                    continue
                hjobs.append((Config(data_seed=ds, init_seed=is_, **cfg_kw),
                              rd, ck(donor_rd(ds, is_), t)))
    if hjobs:
        print(f"hybrids+controls: {len(hjobs)}", flush=True)
        train_batched([c for c, _, _ in hjobs], [rd for _, rd, _ in hjobs],
                      init_from=[f for _, _, f in hjobs],
                      compile_mode=args.compile, spectra_every=5)
        for _, rd, _ in hjobs:
            report(rd)

    if args.smoke:
        import numpy as np
        from safetensors.numpy import load_file as npload
        ds, (a, b) = data_seeds[0], ordered_pairs(init_seeds)[0]
        h0 = npload(str(RUNS / "_inits" / f"t0_A{a}_B{b}_ds{ds}.safetensors"))
        da = npload(str(ck(donor_rd(ds, a), 0)))
        db = npload(str(ck(donor_rd(ds, b), 0)))
        assert np.array_equal(h0["embed.W_E"], da["embed.W_E"])
        assert np.array_equal(h0["unembed.W_U"], db["unembed.W_U"])
        assert not np.array_equal(h0["embed.W_E"], db["embed.W_E"])
        print("SMOKE OK: hybrid surgery verified (donor W_E + recipient rest)")
    print("transplant done", flush=True)


if __name__ == "__main__":
    main()
