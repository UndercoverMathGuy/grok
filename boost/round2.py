"""Round 2 of the double-dissociation cohort (prereg: boost/README.md,
Round 2 section). Reuses the round-1 donors and their dense checkpoints.

Arms: swap (leader<->target subspace exchange, epoch 20) x8,
swapsham (two non-committee freqs) x8, jump500 x8, jump1000 x8,
gate1000 x8.

On the pod: python3 boost/round2.py         (then boost/analyze2.py)
Smoke (CPU): python3 boost/round2.py --smoke
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("RUNS_DIR", str(ROOT / "runs_boost"))
sys.path.insert(0, str(ROOT / "cloud"))
sys.path.insert(0, str(ROOT / "lesion"))
sys.path.insert(0, str(ROOT / "dial"))

import numpy as np  # noqa: E402
from train_semifinal_torch import (RUNS, Config, fourier_basis, report,  # noqa: E402
                                   train_batched)
from train_lesion_seeds import draw_seeds  # noqa: E402
from train_boost import ck, committee_of, freq_energy_at  # noqa: E402

T_SWAP = 20
JUMP_TS = [500, 1000]


def swap_state(donor_ck_path, p, k1, k2, out_path):
    """Exchange the coefficient contents of frequency subspaces k1<->k2 in
    embed.W_E (orthogonal right-multiplication; Frobenius norm exact)."""
    import torch
    from safetensors.torch import load_file, save_file
    st = dict(load_file(str(donor_ck_path)))
    W = st["embed.W_E"].clone()
    B = fourier_basis(p)
    vecs = {k: (torch.from_numpy(B[2 * k - 1]).to(W.dtype),
                torch.from_numpy(B[2 * k]).to(W.dtype)) for k in (k1, k2)}
    Wp = W[:, :p]
    c1, s1 = vecs[k1]
    c2, s2 = vecs[k2]
    a_c1, a_s1 = Wp @ c1, Wp @ s1          # (d,) coefficients
    a_c2, a_s2 = Wp @ c2, Wp @ s2
    Wp = (Wp - torch.outer(a_c1, c1) - torch.outer(a_s1, s1)
          - torch.outer(a_c2, c2) - torch.outer(a_s2, s2)
          + torch.outer(a_c1, c2) + torch.outer(a_s1, s2)
          + torch.outer(a_c2, c1) + torch.outer(a_s2, s1))
    W[:, :p] = Wp
    st["embed.W_E"] = W
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(st, str(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--compile", default="default",
                    choices=["default", "reduce-overhead", "off"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t_swap, jump_ts = T_SWAP, JUMP_TS
    if args.smoke:
        args.epochs, args.compile = 120, "off"
        jump_ts = [40]

    data_seeds, init_seeds = draw_seeds()
    if args.smoke:
        data_seeds, init_seeds = data_seeds[:1], init_seeds[:1]
    kw = dict(d_mlp=128 if args.smoke else 512, weight_decay=3.0,
              num_epochs=args.epochs, save_every=10)
    p = 113
    donor_rd = lambda ds, is_: RUNS / "donor" / f"ds{ds}_is{is_}"

    meta2, sjobs = {}, []
    for ds in data_seeds:
        for is_ in init_seeds:
            drd = donor_rd(ds, is_)
            comm = committee_of(drd)
            E = freq_energy_at(drd, t_swap, p)
            order = (np.argsort(E)[::-1] + 1).tolist()
            target = int(next(k for k in order if k not in comm))
            leader = int(next(k for k in order if k in comm))
            nonc = [k for k in order if k not in comm and k != target]
            sh1, sh2 = int(nonc[len(nonc) // 2]), int(nonc[len(nonc) // 2 + 1])
            meta2[f"ds{ds}_is{is_}"] = {"committee": comm, "target": target,
                                        "leader": leader, "sham": [sh1, sh2]}
            for arm, (ka, kb) in (("swap", (leader, target)),
                                  ("swapsham", (sh1, sh2))):
                rd = RUNS / arm / f"ds{ds}_is{is_}"
                if (rd / "spectra.npz").exists():
                    continue
                f = RUNS / "_inits" / f"{arm}_ds{ds}_is{is_}.safetensors"
                swap_state(ck(drd, t_swap), p, ka, kb, f)
                sjobs.append((Config(data_seed=ds, init_seed=is_, **kw), rd, f))
            for t in jump_ts:
                for arm, fr in ((f"jump{t}", 0.7), (f"gate{t}", 0.3)):
                    if arm.startswith("gate") and t != jump_ts[-1]:
                        continue
                    rd = RUNS / arm / f"ds{ds}_is{is_}"
                    if (rd / "spectra.npz").exists():
                        continue
                    c2 = Config(data_seed=ds, init_seed=is_,
                                **{**kw, "frac_train": fr})
                    sjobs.append((c2, rd, ck(drd, t)))
    (RUNS / "round2_meta.json").write_text(json.dumps(
        {"t_swap": t_swap, "jump_ts": jump_ts, "runs": meta2}, indent=1))

    byfrac = {}
    for c, rd, f in sjobs:
        byfrac.setdefault(c.frac_train, []).append((c, rd, f))
    for fr, js in sorted(byfrac.items()):
        print(f"round2 frac {fr}: {len(js)} runs", flush=True)
        train_batched([c for c, _, _ in js], [rd for _, rd, _ in js],
                      init_from=[f for _, _, f in js],
                      compile_mode=args.compile, spectra_every=5)
        for _, rd, _ in js:
            report(rd)

    if args.smoke:
        from safetensors.numpy import load_file
        ds, is_ = data_seeds[0], init_seeds[0]
        m = meta2[f"ds{ds}_is{is_}"]
        d0 = load_file(str(ck(donor_rd(ds, is_), t_swap)))["embed.W_E"]
        w0 = load_file(str(RUNS / "_inits" /
                           f"swap_ds{ds}_is{is_}.safetensors"))["embed.W_E"]
        assert abs(np.linalg.norm(w0) - np.linalg.norm(d0)) < 1e-3, "not isometric"
        B = fourier_basis(p)
        e_d = ((B @ d0.astype(np.float64)[:, :p].T) ** 2).sum(1)
        e_w = ((B @ w0.astype(np.float64)[:, :p].T) ** 2).sum(1)
        fe = lambda e: e[1::2][:56] + e[2::2][:56]
        ed, ew = fe(e_d), fe(e_w)
        l, t = m["leader"], m["target"]
        assert abs(ed[l - 1] - ew[t - 1]) < 1e-6 and abs(ed[t - 1] - ew[l - 1]) < 1e-6
        untouched = [k for k in range(56) if k not in (l - 1, t - 1)]
        assert np.allclose(ed[untouched], ew[untouched], atol=1e-6)
        print("SMOKE OK: swap is isometric, exchanges exactly two frequencies")
    print("round2 done", flush=True)


if __name__ == "__main__":
    main()
