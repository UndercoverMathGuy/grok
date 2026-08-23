"""Lesion-and-recovery cohort. Prereg: lesion/README.md.

Phase 1: 8 donors (same deterministic draws as transplant), ckpt/50.
Phase 2: lesioned states (noise one factor, norm-matched, no competitor),
32 lesions + 8 controls, 3000 epochs, one batch.

On the pod (after scp of lesion/):
    setsid nohup python3 -u lesion/train_lesion.py > lesion.log 2>&1 < /dev/null &
Smoke (CPU): python3 lesion/train_lesion.py --smoke
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("RUNS_DIR", str(ROOT / "runs_lesion"))
sys.path.insert(0, str(ROOT / "cloud"))

from train_semifinal_torch import RUNS, Config, report, train_batched  # noqa: E402

META_SEED = 20260815      # donor draws — identical to transplant donors
LESION_SEED = 20260816    # lesion noise draws
LESION_TS = [150, 500]
CTRL_T = 500
ZERO_INIT = {"blocks.0.mlp.b_in", "blocks.0.mlp.b_out"}


def draw_seeds(n_masks=2, n_inits=4):
    rng = random.Random(META_SEED)
    data_seeds = sorted(rng.sample(range(20_000, 30_000), n_masks))
    init_seeds = sorted(rng.sample(range(100_000, 1_000_000), n_inits))
    return data_seeds, init_seeds


def ck(rd, ep):
    return rd / "checkpoints" / f"epoch_{ep:05d}.safetensors"


def make_lesion(donor_ck, arm, gen_seed, out_path):
    """arm 'rand_we': noise W_E, keep rest. 'rand_rest': keep W_E, noise rest.
    Noised tensor = fresh Gaussian rescaled to the replaced tensor's
    Frobenius norm; zero-init biases reset to zeros."""
    import torch
    from safetensors.torch import load_file, save_file
    state = dict(load_file(str(donor_ck)))
    g = torch.Generator().manual_seed(gen_seed)
    for k, v in state.items():
        hit = (k == "embed.W_E") if arm == "rand_we" else (k != "embed.W_E")
        if not hit:
            continue
        if k in ZERO_INIT:
            state[k] = torch.zeros_like(v)
            continue
        fresh = torch.randn(v.shape, generator=g, dtype=v.dtype)
        state[k] = fresh * (v.norm() / fresh.norm())
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
    lesion_ts, ctrl_t, save_every = LESION_TS, CTRL_T, 50
    if args.smoke:
        args.epochs, args.width, args.compile = 100, 128, "off"
        lesion_ts, ctrl_t, save_every = [40], 40, 20

    data_seeds, init_seeds = draw_seeds()
    cfg_kw = dict(d_mlp=args.width, weight_decay=3.0, num_epochs=args.epochs,
                  save_every=save_every)
    lrng = random.Random(LESION_SEED)
    print(f"lesion donors: data {data_seeds}  init {init_seeds}", flush=True)

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

    ljobs = []
    for arm in ("rand_we", "rand_rest"):
        for t in lesion_ts:
            for ds in data_seeds:
                for is_ in init_seeds:
                    gseed = lrng.randrange(1, 2**31)  # drawn in fixed order
                    rd = RUNS / f"{arm}_t{t}" / f"ds{ds}_is{is_}"
                    if (rd / "spectra.npz").exists():
                        continue
                    f = RUNS / "_inits" / f"{arm}_t{t}_ds{ds}_is{is_}.safetensors"
                    make_lesion(ck(donor_rd(ds, is_), t), arm, gseed, f)
                    ljobs.append((Config(data_seed=ds, init_seed=is_, **cfg_kw),
                                  rd, f))
    for ds in data_seeds:
        for is_ in init_seeds:
            rd = RUNS / f"ctrl_t{ctrl_t}" / f"ds{ds}_is{is_}"
            if (rd / "spectra.npz").exists():
                continue
            ljobs.append((Config(data_seed=ds, init_seed=is_, **cfg_kw),
                          rd, ck(donor_rd(ds, is_), ctrl_t)))
    if ljobs:
        print(f"lesions+controls: {len(ljobs)}", flush=True)
        train_batched([c for c, _, _ in ljobs], [rd for _, rd, _ in ljobs],
                      init_from=[f for _, _, f in ljobs],
                      compile_mode=args.compile, spectra_every=5)
        for _, rd, _ in ljobs:
            report(rd)

    if args.smoke:
        import numpy as np
        from safetensors.numpy import load_file as npload
        ds, is_ = data_seeds[0], init_seeds[0]
        t = lesion_ts[0]
        d = npload(str(ck(donor_rd(ds, is_), t)))
        w = npload(str(RUNS / "_inits" / f"rand_we_t{t}_ds{ds}_is{is_}.safetensors"))
        r = npload(str(RUNS / "_inits" / f"rand_rest_t{t}_ds{ds}_is{is_}.safetensors"))
        wek = "embed.W_E"
        assert not np.array_equal(w[wek], d[wek])
        assert abs(np.linalg.norm(w[wek]) - np.linalg.norm(d[wek])) < 1e-3
        assert np.array_equal(w["unembed.W_U"], d["unembed.W_U"])
        assert np.array_equal(r[wek], d[wek])
        assert not np.array_equal(r["unembed.W_U"], d["unembed.W_U"])
        assert abs(np.linalg.norm(r["unembed.W_U"]) - np.linalg.norm(d["unembed.W_U"])) < 1e-3
        assert np.all(r["blocks.0.mlp.b_in"] == 0)
        print("SMOKE OK: lesion surgery verified (noised+norm-matched vs kept)")
    print("lesion done", flush=True)


if __name__ == "__main__":
    main()
