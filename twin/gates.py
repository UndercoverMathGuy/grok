"""twin/gates.py — determinism gates G1/G2/G3.  Part of the smoke test, and
re-run on the GPU before any cohort is trained.

G1  repeatability: the same config trained twice (fresh model + optimizer
    objects) must give bitwise identical weights and bitwise identical logs.
    If G1 fails, nothing else in the suite means anything.

G2  checkpoint/restore: N epochs straight == N/2, torch.save, restore into
    fresh objects, N/2 more.  This is the gate that licenses the whole FORK /
    HOLD design: a fork is only an exact twin if weights AND full AdamW state
    (exp_avg, exp_avg_sq, step) round-trip exactly.  Run in BOTH solo and
    batched mode, because the batched path is the one production uses.

G3  batch invariance: the same seeds trained at several vmap widths (1 / 2 / M).
    This one is ALLOWED to fail, and in practice it DOES: vmap dispatches
    different matmul kernels for different batch dims, the results differ in
    the last bit, and grokking dynamics amplify a 1-ulp difference into a
    visibly different trajectory within a few hundred epochs.  Observed on CPU
    (torch 2.13, p=23 toy): lane agreement holds for some (lane, width) pairs
    and breaks for others, with no useful pattern — so the fallback rule is
    treated as ALWAYS in force:

        one fixed batch width per cohort, and every control twin sits in the
        SAME batch as its treatment siblings (make_twin's `group` field, which
        train_twin never splits), so no comparison ever crosses a batch
        boundary.

    What DOES hold, and is what the fallback needs, is gate `within_batch`:
    two lanes with identical inputs inside ONE batch are bitwise identical
    regardless of their lane index.

Usage:
    uv run --group convert python twin/gates.py --device cpu --no-compile
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from core import Recipe, load_ckpt, read_json, setup_determinism
from train_twin import train_batch

LOG_KEYS = ["epochs", "train_acc", "test_acc", "train_loss", "test_loss",
            "coeff_epochs", "coeffs", "energy"]


def _job(name, ds, iseed, start, end, ckpts, frac=0.5, log_every=10,
         spec_every=25, init=None, group=None):
    return {"name": name, "cohort": "gate", "group": group or name,
            "data_seed": ds, "init_seed": iseed, "frac": frac,
            "start_epoch": start, "end_epoch": end, "log_every": log_every,
            "spectra_every": spec_every, "ckpt_epochs": ckpts,
            "init": init or {"kind": "fresh"}, "edit": {"kind": "none"}}


def _weights(runs, name, epoch):
    return load_ckpt(Path(runs) / name / "checkpoints" / f"epoch_{epoch:06d}.pt")


def _same_weights(a, b):
    ok = all(torch.equal(a["weights"][k], b["weights"][k]) for k in a["weights"])
    ok &= all(torch.equal(a["opt"][k][s], b["opt"][k][s])
              for k in a["opt"] for s in ("exp_avg", "exp_avg_sq"))
    return bool(ok)


def _logs(runs, name):
    z = np.load(Path(runs) / name / "spectra.npz")
    return {k: z[k] for k in LOG_KEYS}


def _same_logs(a, b, at=None):
    """Bitwise log comparison, optionally restricted to a shared epoch grid."""
    if at is None:
        return all(np.array_equal(a[k], b[k]) for k in LOG_KEYS)
    ia = np.isin(a["epochs"], at)
    ib = np.isin(b["epochs"], at)
    ja = np.isin(a["coeff_epochs"], at)
    jb = np.isin(b["coeff_epochs"], at)
    ok = np.array_equal(a["epochs"][ia], b["epochs"][ib])
    for k in ("train_acc", "test_acc", "train_loss", "test_loss"):
        ok &= np.array_equal(a[k][ia], b[k][ib])
    ok &= np.array_equal(a["coeff_epochs"][ja], b["coeff_epochs"][jb])
    for k in ("coeffs", "energy"):
        ok &= np.array_equal(a[k][ja], b[k][jb])
    return bool(ok)


def run_gates(recipe, runs, device, compile_mode, n_epochs=200, ds=41001,
              iseed=41002, others=(41003, 41004, 41005), verbose=True):
    runs = Path(runs)
    if runs.exists():
        shutil.rmtree(runs)
    N, H = n_epochs, n_epochs // 2
    kw = dict(recipe=recipe, runs_dir=runs, device=device,
              compile_mode=compile_mode, verbose=False)
    res = {}

    # ---- G1 ------------------------------------------------------------
    train_batch([_job("g1/a", ds, iseed, 0, N, [N])], **kw)
    train_batch([_job("g1/b", ds, iseed, 0, N, [N])], **kw)
    res["G1"] = (_same_weights(_weights(runs, "g1/a", N), _weights(runs, "g1/b", N))
                 and _same_logs(_logs(runs, "g1/a"), _logs(runs, "g1/b")))

    # ---- G2 solo -------------------------------------------------------
    train_batch([_job("g2/straight", ds, iseed, 0, N, [N])], **kw)
    train_batch([_job("g2/half", ds, iseed, 0, H, [H])], **kw)
    train_batch([_job("g2/rest", ds, iseed, H, N, [N],
                      init={"kind": "ckpt",
                            "path": f"g2/half/checkpoints/epoch_{H:06d}.pt"})], **kw)
    st, rt = _logs(runs, "g2/straight"), _logs(runs, "g2/rest")
    res["G2_solo"] = (_same_weights(_weights(runs, "g2/straight", N),
                                    _weights(runs, "g2/rest", N))
                      and _same_logs(st, rt, at=rt["epochs"]))

    # ---- G3 ------------------------------------------------------------
    # Sensitive version: the SAME seeds trained at several vmap widths.  A
    # single (M=1 vs M=4) probe is not enough — the differences start at 1 ulp
    # and only become visible after the dynamics have amplified them, so G3 is
    # run over the full gate horizon at 1 / 2 / M widths and every lane that
    # appears at more than one width must agree.
    seeds = [iseed] + list(others)
    widths = sorted({1, 2, len(seeds)})
    for W in widths:
        train_batch([_job(f"g3/w{W}/lane{i}", ds, seeds[i], 0, N, [N],
                          group=f"g3/w{W}") for i in range(W)], **kw)
    g3, g3_detail = True, {}
    for i in range(len(seeds)):
        ws = [W for W in widths if i < W]
        if len(ws) < 2:
            continue
        ref = _logs(runs, f"g3/w{ws[0]}/lane{i}")
        for W in ws[1:]:
            same = (_same_logs(_logs(runs, f"g3/w{W}/lane{i}"), ref)
                    and _same_weights(_weights(runs, f"g3/w{ws[0]}/lane{i}", N),
                                      _weights(runs, f"g3/w{W}/lane{i}", N)))
            g3_detail[f"lane{i}_w{ws[0]}_vs_w{W}"] = same
            g3 &= same
    res["G3"] = bool(g3)
    res["G3_detail"] = g3_detail

    # ---- G2 batched (the mode production actually runs in) --------------
    M = 1 + len(others)
    seeds = [iseed] + list(others)
    straight = [_job(f"g2b/straight/lane{i}", ds, s, 0, N, [N], group="g2b/straight")
                for i, s in enumerate(seeds)]
    half = [_job(f"g2b/half/lane{i}", ds, s, 0, H, [H], group="g2b/half")
            for i, s in enumerate(seeds)]
    rest = [_job(f"g2b/rest/lane{i}", ds, seeds[i], H, N, [N], group="g2b/rest",
                 init={"kind": "ckpt",
                       "path": f"g2b/half/lane{i}/checkpoints/epoch_{H:06d}.pt"})
            for i in range(M)]
    train_batch(straight, **kw)
    train_batch(half, **kw)
    train_batch(rest, **kw)
    ok = True
    for i in range(M):
        a, b = f"g2b/straight/lane{i}", f"g2b/rest/lane{i}"
        rb = _logs(runs, b)
        ok &= _same_weights(_weights(runs, a, N), _weights(runs, b, N))
        ok &= _same_logs(_logs(runs, a), rb, at=rb["epochs"])
    res["G2_batched"] = bool(ok)

    # ---- within-batch comparison path (the G3 fallback) -----------------
    # Two lanes with the same seed inside ONE batch must agree bitwise; that is
    # all the fallback rule needs, and it holds whether or not G3 does.
    twin = [_job("g3fb/lane0", ds, iseed, 0, N, [N], group="g3fb"),
            _job("g3fb/lane1", ds, iseed, 0, N, [N], group="g3fb"),
            _job("g3fb/lane2", ds, others[0], 0, N, [N], group="g3fb")]
    train_batch(twin, **kw)
    res["within_batch"] = (_same_weights(_weights(runs, "g3fb/lane0", N),
                                         _weights(runs, "g3fb/lane1", N))
                           and _same_logs(_logs(runs, "g3fb/lane0"),
                                          _logs(runs, "g3fb/lane1")))
    if verbose:
        for k in ("G1", "G2_solo", "G2_batched", "G3", "within_batch"):
            note = "" if k != "G3" else ("" if res[k] else
                                         "  (allowed: fallback = within-batch only)")
            print(f"  {'PASS' if res[k] else 'FAIL'}  {k}{note}")
        if not res["G3"]:
            for k, v in res["G3_detail"].items():
                print(f"        {'ok  ' if v else 'diff'} {k}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_twin/_gates")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--compile", default="reduce-overhead")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d-mlp", type=int, default=512)
    args = ap.parse_args()

    setup_determinism()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    recipe = Recipe(p=args.p, d_model=args.d_model, num_heads=args.heads,
                    d_mlp=args.d_mlp)
    res = run_gates(recipe, args.runs, device,
                    "off" if args.no_compile else args.compile, args.epochs)
    hard = ["G1", "G2_solo", "G2_batched", "within_batch"]
    print("GATES:", " ".join(f"{k}={'PASS' if res[k] else 'FAIL'}"
                             for k in hard + ["G3"]))
    raise SystemExit(0 if all(res[k] for k in hard) else 1)


if __name__ == "__main__":
    main()
