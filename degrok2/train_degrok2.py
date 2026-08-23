"""De-grok 2 cohort: pervasiveness sweep (prereg: degrok2/README.md).

Arms: A fine frac grid (add p113), B operation (sub p113), C modulus
(add p96, p127), D width (d_mlp 128/2048), E non-abelian (S5 p120).
Cells sharing (task, p, d_mlp, epochs) are merged into one ragged lockstep
batch across fracs (PAD_CAP-capped padding). Fresh META_SEED=20260818.

Run dirs: runs_degrok2/{fn}{p}_m{d_mlp}_f{frac:g}_wd{wd:g}/ds{ds}_is{is}/

On the pod (see README). Smoke (CPU, ~2 min):
    python3 degrok2/train_degrok2.py --smoke
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("RUNS_DIR", str(ROOT / "runs_degrok2"))
sys.path.insert(0, str(ROOT / "cloud"))

from train_semifinal_torch import RUNS, Config, report, train_batched  # noqa: E402

META_SEED = 20260818


def draw_seeds(n_masks=2, n_inits=4):
    rng = random.Random(META_SEED)
    data_seeds = sorted(rng.sample(range(40_000, 50_000), n_masks))
    init_seeds = sorted(rng.sample(range(100_000, 1_000_000), n_inits))
    return data_seeds, init_seeds


def epochs_for(frac):
    return 30_000 if frac < 0.3 else (15_000 if frac == 0.3 else 10_000)


def plan(args):
    """[(label, fn, p, d_mlp, frac, wds, epochs, spectra, seed_pairs)]"""
    ds, is_ = draw_seeds()
    pairs = [(d, i) for d in ds for i in is_]
    s5_pairs = [(d, i) for d in ds for i in is_[:2]]
    if args.smoke:
        pairs, s5_pairs = pairs[:2], s5_pairs[:2]
    P = []
    for f in args.a_fracs:                                     # arm A
        P.append((f"A add113 f{f:g}", "add", 113, 512, f, args.wds,
                  epochs_for(f), 5, pairs))
    for f in args.b_fracs:                                     # arm B
        P.append((f"B sub113 f{f:g}", "sub", 113, 512, f, args.wds,
                  epochs_for(f), 5, pairs))
    for f in args.b_fracs:                                     # arm C p96
        P.append((f"C add96 f{f:g}", "add", 96, 512, f, args.wds,
                  epochs_for(f), 5, pairs))
    for f in args.c127_fracs:                                  # arm C p127
        P.append((f"C add127 f{f:g}", "add", 127, 512, f, args.wds,
                  epochs_for(f), 5, pairs))
    for m in args.widths:                                      # arm D
        for f in args.d_fracs:
            P.append((f"D add113 m{m} f{f:g}", "add", 113, m, f, [3.0],
                      epochs_for(f), 5, pairs))
    for f in args.e_fracs:                                     # arm E
        P.append((f"E s5 f{f:g}", "s5", 120, 512, f, args.s5_wds,
                  args.s5_epochs, 25, s5_pairs))
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", default="reduce-overhead",
                    choices=["default", "reduce-overhead", "off"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.a_fracs = [0.2, 0.25, 0.3, 0.4, 0.6, 0.8, 0.9]
    args.b_fracs = [0.3, 0.6, 0.9]
    args.c127_fracs = [0.6, 0.9]
    args.d_fracs = [0.3, 0.9]
    args.e_fracs = [0.6, 0.7, 0.8, 0.9]
    args.wds, args.s5_wds = [1.0, 3.0], [1.0, 2.0]
    args.widths = [128, 2048]
    args.s5_epochs = 100_000
    if args.smoke:
        args.a_fracs, args.b_fracs = [0.9], [0.9]
        args.c127_fracs, args.d_fracs, args.e_fracs = [], [], [0.7]
        args.wds, args.s5_wds, args.widths = [1.0], [1.0], []
        args.s5_epochs, args.compile = 150, "off"

    # Cells sharing (task, p, width, epoch budget, spectra grid) are merged
    # into ONE lockstep batch across fracs (ragged padding in the trainer),
    # capped at PAD_CAP train-count waste. 2.5 merges maximally: the four
    # 10k-epoch add-113 fracs (.4-.9) train as one M=64 batch, all S5 fracs
    # as one M=32 batch — VRAM-bound stacking per the 32GB 5090.
    PAD_CAP = 2.5
    t0, n = time.time(), 0
    groups = {}
    for label, fn, p, m, frac, wds, ep, spec, pairs in plan(args):
        jobs = []
        for wd in wds:
            for ds, is_ in pairs:
                rd = (RUNS / f"{fn}{p}_m{m}_f{frac:g}_wd{wd:g}"
                      / f"ds{ds}_is{is_}")
                if (rd / "spectra.npz").exists():
                    continue
                jobs.append((Config(
                    p=p, fn_name=fn, frac_train=frac, d_mlp=m,
                    weight_decay=wd, data_seed=ds, init_seed=is_,
                    num_epochs=200 if args.smoke else ep,
                    save_every=100 if args.smoke else 1000), rd))
        if jobs:
            groups.setdefault((fn, p, m, ep, spec), []).append(
                (frac, label, jobs))

    def flush(chunk, spec):
        labels = " + ".join(lb for _, lb, _ in chunk)
        jobs = [j for _, _, js in chunk for j in js]
        print(f"batch [{labels}]: {len(jobs)} runs, "
              f"{jobs[0][0].num_epochs} ep", flush=True)
        train_batched([c for c, _ in jobs], [rd for _, rd in jobs],
                      compile_mode=args.compile,
                      spectra_every=50 if args.smoke else spec)
        for _, rd in jobs:
            report(rd)
        return len(jobs)

    for (fn, p, m, ep, spec), ents in sorted(groups.items()):
        chunk = []
        for frac, label, jobs in sorted(ents):
            if chunk and frac / chunk[0][0] > PAD_CAP:
                n += flush(chunk, spec)
                chunk = []
            chunk.append((frac, label, jobs))
        if chunk:
            n += flush(chunk, spec)

    if args.smoke:
        import json
        import numpy as np
        from train_semifinal_torch import fourier_basis, make_dataset
        rd = next((RUNS).glob("sub113_*/*/config.json")).parent
        cfg = json.loads((rd / "config.json").read_text())
        assert cfg["fn_name"] == "sub"
        tok, lab = make_dataset(Config(p=113, fn_name="sub"))
        assert (lab == (tok[:, 0] - tok[:, 1]) % 113).all()
        # instrument check: a synthetic sub-solution logit tensor must put
        # ~all its projected energy on its own frequency under the
        # sign-flipped (sub) projectors
        p_, w = 113, 7
        B = fourier_basis(p_)
        a = np.repeat(np.arange(p_), p_)
        b = np.tile(np.arange(p_), p_)
        L = np.cos(2 * np.pi * w * (a[:, None] - b[:, None]
                                    - np.arange(p_)[None, :]) / p_)
        bc, bs = B[1::2], B[2::2]
        uc = (np.einsum("np,nq->npq", bc, bc)
              + np.einsum("np,nq->npq", bs, bs)).reshape(56, -1)
        us = (np.einsum("np,nq->npq", bs, bc)
              - np.einsum("np,nq->npq", bc, bs)).reshape(56, -1)
        E = ((uc @ L) ** 2).sum(-1) + ((us @ L) ** 2).sum(-1)
        assert E.argmax() == w - 1 and E[w - 1] > 1e3 * np.sort(E)[-2]
        z = np.load(rd / "spectra.npz")
        assert z["energy"].shape[1] == 56, z["energy"].shape
        print("SMOKE OK: sub labels + sign-flipped instrument verified")
    print(f"degrok2 done: {n} trained, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
