"""Boost/Jump double-dissociation cohort. Prereg: boost/README.md.

On the pod (after scp of boost/ and lesion/train_lesion_seeds.py):
    setsid nohup python3 -u boost/train_boost.py > boost.log 2>&1 < /dev/null &
Smoke (CPU): python3 boost/train_boost.py --smoke
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("RUNS_DIR", str(ROOT / "runs_boost"))
sys.path.insert(0, str(ROOT / "cloud"))
sys.path.insert(0, str(ROOT / "lesion"))
sys.path.insert(0, str(ROOT / "dial"))

import numpy as np  # noqa: E402
from train_semifinal_torch import (RUNS, Config, fourier_basis, report,  # noqa: E402
                                   train_batched, train_test_split)
from train_lesion_seeds import draw_seeds  # noqa: E402

T_BOOST, T_JUMP = 20, 200
ALPHAS, SHAM_ALPHAS = [2.0, 4.0, 8.0], [4.0, 8.0]


def ck(rd, ep):
    return rd / "checkpoints" / f"epoch_{ep:05d}.safetensors"


def committee_of(rd):
    sys.path.insert(0, str(ROOT / "dial"))
    from analyze_dial import committee_from_coeffs
    z = np.load(rd / "spectra.npz")
    return committee_from_coeffs(z["coeffs"][-1])


def freq_projector(p, k):
    """(p, p) projector onto span(cos_k, sin_k) of the vocab axis."""
    B = fourier_basis(p)
    c, s = B[2 * k - 1], B[2 * k]
    return np.outer(c, c) + np.outer(s, s)


def boosted_state(donor_ck_path, p, k, alpha, out_path):
    import torch
    from safetensors.torch import load_file, save_file
    st = dict(load_file(str(donor_ck_path)))
    W = st["embed.W_E"]                       # (d, p+1)
    P_k = torch.from_numpy(freq_projector(p, k)).to(W.dtype)
    W = W.clone()
    W[:, :p] = W[:, :p] + (alpha - 1.0) * (W[:, :p] @ P_k)
    st["embed.W_E"] = W
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(st, str(out_path))


def freq_energy_at(rd, ep, p):
    from safetensors.numpy import load_file
    B = fourier_basis(p)
    W = load_file(str(ck(rd, ep)))["embed.W_E"].astype(np.float64)[:, :p]
    E = ((B @ W.T) ** 2).sum(1)
    return E[1::2][:p // 2] + E[2::2][:p // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--compile", default="default",
                    choices=["default", "reduce-overhead", "off"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t_boost, t_jump, alphas, shams = T_BOOST, T_JUMP, ALPHAS, SHAM_ALPHAS
    frac2 = 0.7
    if args.smoke:
        args.epochs, args.compile = 120, "off"
        t_boost, t_jump, alphas, shams = 20, 40, [4.0], [4.0]

    data_seeds, init_seeds = draw_seeds()
    if args.smoke:
        data_seeds, init_seeds = data_seeds[:1], init_seeds[:1]
    kw = dict(d_mlp=128 if args.smoke else 512, weight_decay=3.0,
              num_epochs=args.epochs, save_every=10)

    donor_rd = lambda ds, is_: RUNS / "donor" / f"ds{ds}_is{is_}"
    jobs = [(Config(data_seed=ds, init_seed=is_, **kw), donor_rd(ds, is_))
            for ds in data_seeds for is_ in init_seeds
            if not (donor_rd(ds, is_) / "spectra.npz").exists()]
    if jobs:
        print(f"donors: {len(jobs)}", flush=True)
        train_batched([c for c, _ in jobs], [rd for _, rd in jobs],
                      compile_mode=args.compile, spectra_every=5)
        for _, rd in jobs:
            report(rd)

    p = 113
    sjobs, meta = [], {}
    for ds in data_seeds:
        for is_ in init_seeds:
            drd = donor_rd(ds, is_)
            comm = committee_of(drd)
            E = freq_energy_at(drd, t_boost, p)
            order = np.argsort(E)[::-1] + 1
            target = int(next(k for k in order if k not in comm))
            leader = int(next(k for k in order if k in comm))
            meta[f"ds{ds}_is{is_}"] = {"committee": comm, "target": target,
                                       "leader": leader}
            for arm, k, als in (("boost", target, alphas),
                                ("sham", leader, shams)):
                for a in als:
                    rd = RUNS / f"{arm}_a{a:g}" / f"ds{ds}_is{is_}"
                    if (rd / "spectra.npz").exists():
                        continue
                    f = RUNS / "_inits" / f"{arm}_a{a:g}_ds{ds}_is{is_}.safetensors"
                    boosted_state(ck(drd, t_boost), p, k, a, f)
                    sjobs.append((Config(data_seed=ds, init_seed=is_, **kw),
                                  rd, f))
            for arm, fr in (("jump", frac2), ("jumpgate", 0.3)):
                rd = RUNS / arm / f"ds{ds}_is{is_}"
                if (rd / "spectra.npz").exists():
                    continue
                c2 = Config(data_seed=ds, init_seed=is_, **{**kw,
                            "frac_train": fr})
                sjobs.append((c2, rd, ck(drd, t_jump)))
    (RUNS / "boost_meta.json").parent.mkdir(parents=True, exist_ok=True)
    (RUNS / "boost_meta.json").write_text(json.dumps(
        {"t_boost": t_boost, "t_jump": t_jump, "runs": meta}, indent=1))

    # frac differs across stage-2 runs -> group into equal-frac batches
    byfrac = {}
    for c, rd, f in sjobs:
        byfrac.setdefault(c.frac_train, []).append((c, rd, f))
    for fr, js in sorted(byfrac.items()):
        print(f"stage2 frac {fr}: {len(js)} runs", flush=True)
        train_batched([c for c, _, _ in js], [rd for _, rd, _ in js],
                      init_from=[f for _, _, f in js],
                      compile_mode=args.compile, spectra_every=5)
        for _, rd, _ in js:
            report(rd)

    if args.smoke:
        from safetensors.numpy import load_file
        ds, is_ = data_seeds[0], init_seeds[0]
        m = meta[f"ds{ds}_is{is_}"]
        d0 = load_file(str(ck(donor_rd(ds, is_), t_boost)))["embed.W_E"]
        b0 = load_file(str(RUNS / "_inits" /
                           f"boost_a4_ds{ds}_is{is_}.safetensors"))["embed.W_E"]
        diff = (b0 - d0).astype(np.float64)
        B = fourier_basis(113)
        dE = ((B @ diff[:, :113].T) ** 2).sum(1)
        dEf = dE[1::2][:56] + dE[2::2][:56]
        assert dEf.argmax() + 1 == m["target"]
        assert dEf[dEf.argmax()] / dEf.sum() > 0.999, "edit leaked beyond target freq"
        m03 = train_test_split(Config(data_seed=ds, init_seed=is_, frac_train=0.3))
        m07 = train_test_split(Config(data_seed=ds, init_seed=is_, frac_train=0.7))
        assert (m03 & ~m07).sum() == 0 and m07.sum() > m03.sum(), "jump not a superset"
        print("SMOKE OK: single-frequency edit + superset jump verified")
    print("boost done", flush=True)


if __name__ == "__main__":
    main()
