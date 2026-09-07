"""Fixed-support cohort for D_n: MEASURE the constrained max normalised margin.

notes/dihedral_results.md closed the D_113 closed form as a registered
failure: committees are hierarchical, K_thr median 5, sign present at 0.065.
The framework's fallback for non-decoupled pricing is to measure the
constrained sup instead of predicting it, i.e. to train nets whose logit
function may only use a FIXED set S of irreps and read off

    gamma_hat(K) = max over that class of  min-margin / ||theta||^nu,   nu = 6

as a function of K = |S| (two-dim irreps rho_j). This is the D_n analogue of
nudial/fixed_support.py (fixed Fourier support for the p=113 MLP).

Mechanism: train_semifinal_torch.train_batched(vocab_proj=...) — an
orthogonal projector P on the group-token axis (vocab columns 0..p-1; the
'=' column stays free) applied to W_E and/or W_U at init and after every
AdamW update. P = PROJ[trivial] + PROJ[sign] + sum_{j in S} PROJ[rho_j]
(cloud/dihedral.py isotypic projectors). Projecting W_U forces the logits'
isotypic energy outside S to zero (measured at ~1e-14 of the total, i.e.
f32 roundoff); projecting W_E in addition restricts the inputs too.

Arms (6 seeds each, frac 0.5, wd 1, otherwise the sweep_dihedral recipe):
  both   E and U projected, K in {2..7}   decoupled-circuit class
  out    only U projected,  K in {2..7}   true constrained sup on the output
  rand4  E and U projected, K = 4, RANDOM support per seed   (control)
  free   P = identity through the same code path             (control)
Supports S_K are the closed-form-optimal Z_113 frequency sets from
nudial/kstar_table.best_subset (exhaustive for K <= 6, local search at 7).

Usage
  RUNS_DIR=runs_dihedral python3 -u cloud/fixed_support_dihedral.py
  RUNS_DIR=runs_dihedral python3 -u cloud/fixed_support_dihedral.py --arms out
  uv run python cloud/fixed_support_dihedral.py --test    # projector unit tests
  uv run python cloud/fixed_support_dihedral.py --smoke   # end-to-end CPU, n=13
Score with cloud/score_fixed_support_dihedral.py --root $RUNS_DIR.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "nudial"))
from dihedral import build
from kstar_table import best_subset
from train_semifinal_torch import Config, PARAM_KEYS, RUNS, train_batched

SEED_DRAW = 20260907
ARMS = {"both": dict(E=True, U=True), "out": dict(E=False, U=True),
        "rand4": dict(E=True, U=True), "free": dict(E=True, U=True)}


def irrep_indices(n, rhos):
    """PROJ row indices for [trivial, sign] + the given rho_j (j = 1..(n-1)/2).
    PROJ order is [trivial, sign, rho_1, ..., rho_(n-1)/2], so rho_j -> j+1."""
    return [0, 1] + [j + 1 for j in sorted(rhos)]


def projector(G, idx):
    """P = sum of the isotypic projectors listed in idx; checked symmetric
    idempotent. rank = trace = sum of d_r^2 = 2 + 4K."""
    P = G["PROJ"][idx].sum(0)
    assert np.allclose(P, P.T, atol=1e-9), "projector not symmetric"
    assert np.allclose(P @ P, P, atol=1e-8), "projector not idempotent"
    rank = int(round(float(np.trace(P))))
    assert rank == int(sum(G["IRREP_DIMS"][i] ** 2 for i in idx))
    return P.astype(np.float32), rank


def plan(args, G, n):
    """One dict per run: cfg, dir, projector, support metadata (resumable)."""
    rng = np.random.default_rng(SEED_DRAW)
    seeds = [int(s) for s in rng.choice(100_000, args.seeds, replace=False)]
    print(f"seeds: {seeds}", flush=True)
    h = (n - 1) // 2
    all_rho = list(range(1, h + 1))
    sup = {}
    for K in args.Ks:
        M_K, S, how = best_subset(n, K)
        sup[K] = tuple(int(j) for j in S)
        print(f"  best_subset({n}, {K}) = {sup[K]}  M_K {M_K:.4f} [{how}]", flush=True)
    jobs = []
    for arm in args.arms:
        Ks = ([args.rand_k] if arm == "rand4" else [0] if arm == "free"
              else args.Ks)
        for K in Ks:
            for s in seeds:
                if arm == "free":
                    rhos = all_rho                       # P = identity
                elif arm == "rand4":
                    rhos = sorted(int(j) for j in np.random.default_rng(
                        SEED_DRAW + s).choice(all_rho, K, replace=False))
                else:
                    rhos = list(sup[K])
                idx = irrep_indices(n, rhos)
                P, rank = projector(G, idx)
                if arm == "free":
                    assert np.allclose(P, np.eye(2 * n), atol=1e-8)
                jobs.append(dict(
                    arm=arm, K=K, seed=s, rhos=rhos, irreps=idx, rank=rank,
                    E=ARMS[arm]["E"], U=ARMS[arm]["U"], P=P,
                    dir=RUNS / f"fs_{arm}_K{K}" / f"s{s}",
                    cfg=Config(p=2 * n, fn_name=f"d{n}", frac_train=args.frac,
                               data_seed=s, init_seed=s, weight_decay=args.wd,
                               num_epochs=args.epochs, save_every=args.epochs)))
    return jobs


def run(args):
    n = args.n
    G = build(n)
    jobs = plan(args, G, n)
    todo = [j for j in jobs if not (j["dir"] / "spectra.npz").exists()]
    print(f"{len(jobs)} runs planned, {len(todo)} to train, "
          f"{args.epochs} epochs each", flush=True)
    groups = {}
    for j in todo:                       # E/U are batch-level flags; P is per run
        groups.setdefault((j["E"], j["U"]), []).append(j)
    for (E, U), js in sorted(groups.items()):
        for b in range(0, len(js), args.batch):
            blk = js[b:b + args.batch]
            names = sorted({f"{j['arm']}_K{j['K']}" for j in blk})
            print(f"\n[E={E} U={U}] batch of {len(blk)}: {names}", flush=True)
            for j in blk:
                j["dir"].mkdir(parents=True, exist_ok=True)
                (j["dir"] / "support.json").write_text(json.dumps(
                    {k: j[k] for k in ("arm", "K", "seed", "rhos", "irreps",
                                       "rank", "E", "U")}, indent=1))
            train_batched([j["cfg"] for j in blk], [j["dir"] for j in blk],
                          log_every=args.log_every,
                          spectra_every=args.spectra_every, loss=args.loss,
                          compile_mode=args.compile, device=args.device,
                          vocab_proj=dict(E=E, U=U, P=np.stack(
                              [j["P"] for j in blk])))
    print("SWEEP DONE", flush=True)


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def test(args):
    """(1) vocab_proj with P = I is bit-identical to vocab_proj=None
        (torch.equal on all 11 tensors after training).
    (2) a rank-deficient P keeps the off-support part of W_E/W_U at zero (to
        f32 roundoff, < 1e-6 relative) and the logits' off-support isotypic
        energy below 1e-10 of the total."""
    import tempfile
    n, dev = 7, "cpu"
    G = build(n)
    p = 2 * n
    base = dict(p=p, fn_name=f"d{n}", frac_train=0.5, data_seed=3, init_seed=3,
                d_model=32, d_mlp=64, num_heads=2, weight_decay=1.0,
                num_epochs=40, save_every=40)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        def go(tag, vp):
            d = td / tag / "s0"
            train_batched([Config(**base)], [d], log_every=1000,
                          spectra_every=20, compile_mode=args.compile,
                          device=dev, vocab_proj=vp)
            from safetensors.torch import load_file
            return load_file(str(d / "checkpoints" / "epoch_00040.safetensors")), d
        a, _ = go("none", None)
        b, _ = go("ident", dict(E=True, U=True, P=np.eye(p, dtype=np.float32)))
        bad = [k for k in PARAM_KEYS if not torch.equal(a[k], b[k])]
        assert not bad, f"P=I is NOT bit-identical; differs on {bad}"
        print("TEST 1 PASS: vocab_proj P=I bit-identical to vocab_proj=None "
              f"({len(PARAM_KEYS)} tensors, torch.equal)")

        rhos = [1]
        idx = irrep_indices(n, rhos)
        P, rank = projector(G, idx)
        c, d = go("rank", dict(E=True, U=True, P=P))
        Q = np.eye(p) - P                       # complementary projector
        for k in ("embed.W_E", "unembed.W_U"):
            off = float(np.abs(c[k].numpy()[:, :p] @ Q).max())
            tot = float(np.abs(c[k].numpy()[:, :p]).max())
            assert off <= 1e-6 * max(tot, 1e-30), (k, off, tot)
            print(f"TEST 2a PASS: {k} off-support max |.| {off:.3e} "
                  f"(on-support scale {tot:.3e})")
        z = np.load(d / "spectra.npz")
        en = z["energy"][-1]                    # [sign, rho_1 .. rho_(n-1)/2]
        on = [0] + list(rhos)                   # sign at 0, rho_j at index j
        off_frac = (en.sum() - en[on].sum()) / max(en.sum(), 1e-300)
        assert off_frac < 1e-10, off_frac
        print(f"TEST 2b PASS: off-support logit isotypic energy fraction "
              f"{off_frac:.3e} (< 1e-10); rank(P) = {rank} = 2 + 4*{len(rhos)}")
    print("ALL TESTS PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=113)
    ap.add_argument("--Ks", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--rand-k", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--wd", type=float, default=1.0)
    # every grokked free run in runs_dihedral/ groks by epoch 5700 (max), so
    # 20k epochs leaves >= 14k epochs of norm-shrinking cleanup after grok.
    ap.add_argument("--epochs", type=int, default=None)   # 20k / 3k in --smoke
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--spectra-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--compile", default="default")
    ap.add_argument("--loss", default="f32stable")
    ap.add_argument("--device", default=None)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.test:
        return test(args)
    if args.smoke:
        global RUNS
        args.n, args.Ks, args.rand_k, args.seeds = 13, [1, 2], 2, 2
        # 3k epochs, not 300: at 300 nothing groks and the scored table is all
        # n/a, which verifies the plumbing but not the readout. 3k x 12 runs
        # is ~90 s on one CPU core and most cells reach the margin readout.
        args.epochs = args.epochs or 3000
        args.batch, args.spectra_every = 32, 100
        args.log_every, args.compile = 1000, "off"
        args.device = args.device or "cpu"
        import train_semifinal_torch as T
        RUNS = Path(os.environ.get("RUNS_DIR", ROOT / "runs_fs_smoke"))
        T.RUNS = RUNS                          # only used for wandb key names
        print(f"SMOKE: n={args.n} p={2 * args.n} epochs={args.epochs} "
              f"RUNS_DIR={RUNS}", flush=True)
        test(args)
        run(args)
        import score_fixed_support_dihedral as S
        S.score(str(RUNS), str(RUNS / "fixed_support_scored.json"))
        return
    args.epochs = args.epochs or 20_000
    run(args)


if __name__ == "__main__":
    main()
