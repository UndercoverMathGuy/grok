"""gopt_local.py -- can anything beat the (n+1)-block?  LOCAL ESCAPE TEST at finite p.

The random-start searches of gopt_atom.py never reach the block, so they cannot decide whether the block
is the global maximiser.  This script starts AT the block (and at block + noise of several sizes) and runs
a long, hard-annealed ascent on the EXACT single-frequency objective

    ratio(c, v) = Lambda(c, v) / ( M_1 * (n+1)^{-(1-r)/r} * ||c||_r ),
    Lambda      = min over u in (mu_p)^n, D != 0  of  Re[ (1 - e^{2 pi i D / p}) h(u) ]
                = min over u of  [ M_1 Re h(u) - sin(2 pi / p) |Im h(u)| ]          (identity, checked)
    h(u)        = sum_l c_l prod_j (1 + u_j v_lj),     M_1 = 1 - cos(2 pi / p).

ratio == 1 exactly at the block.  ratio > 1 anywhere would REFUTE the single-frequency case of the target
(T) at that p (and, if the programme's optimal support has size 1, refute (T) itself).

Plain reading: Lambda is 2^n times the true margin of a one-frequency population of pure cosine neurons,
computed exactly on the full table; the denominator is 2^n times the margin that the target inequality
allows for the same weight-decay cost.

Also reports the two ingredients of the proof in GLOBAL_OPT.md:
  R_grid  = min over the grid of Re h        (what the bijection/Hadamard argument bounds)
  R_torus = min over the full torus of Re h  (the continuum idealisation, always <= R_grid)
"""
import argparse, json, math
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(ROOT))
from gopt_atom import block, R_torus, h_on_points  # noqa: E402


def grid_U(p, n):
    th = torch.arange(p) * (2 * math.pi / p)
    u = torch.polar(torch.ones(p), th)
    idx = torch.cartesian_prod(*[torch.arange(p)] * n) if n > 1 else torch.arange(p)[:, None]
    return u[idx.reshape(-1, n)]


MODE = "margin"   # "margin": true margin (Im part weighted by cot(pi/p));  "regrid": min of Re h alone


def _t(c, v, U, M1, s):
    H = h_on_points(c, v, U)
    return M1 * H.real - (s * H.imag.abs() if MODE == "margin" else 0.0)


def lam_exact(c, v, U, M1, s):
    return _t(c, v, U, M1, s).min()


def lam_soft(c, v, U, M1, s, beta):
    return -torch.logsumexp(-beta * _t(c, v, U, M1, s), 0) / beta


def run(p, n, q, noise, steps, seed, lr):
    nu = n + 1; r = q / nu
    M1 = 1 - math.cos(2 * math.pi / p); s = math.sin(2 * math.pi / p)
    tgt = M1 * (n + 1) ** (-(1 - r) / r)
    U = grid_U(p, n)
    c0, v0 = block(n)
    g = torch.Generator().manual_seed(seed)
    N = c0.shape[0]
    x = torch.cat([c0.real, c0.imag]) + noise * torch.randn(2 * N, generator=g)
    ph = torch.angle(v0) + noise * torch.randn(N, n, generator=g)
    x.requires_grad_(True); ph.requires_grad_(True)
    opt = torch.optim.Adam([x, ph], lr=lr)
    best = -1e18; bestc = None
    for t in range(steps):
        beta = 50.0 * (2e5 / 50.0) ** (t / max(steps - 1, 1)) / M1
        c = torch.complex(x[:N], x[N:]); v = torch.polar(torch.ones(N, n), ph)
        cst = (c.abs() ** 2 + 1e-30).pow(r / 2).sum().pow(1 / r)
        loss = -lam_soft(c, v, U, M1, s, beta) / cst
        opt.zero_grad(); loss.backward(); opt.step()
        if t % 50 == 0 or t == steps - 1:
            with torch.no_grad():
                c = torch.complex(x[:N], x[N:]); v = torch.polar(torch.ones(N, n), ph)
                cst = (c.abs() ** 2).pow(r / 2).sum().pow(1 / r)
                val = float(lam_exact(c, v, U, M1, s)) / float(cst) / tgt
                if val > best:
                    best = val; bestc = (c.clone(), v.clone())
    return best, bestc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", type=int, nargs="+", default=[5, 7, 11, 17, 23, 41])
    ap.add_argument("--ns", type=int, nargs="+", default=[2])
    ap.add_argument("--qs", type=float, nargs="+", default=[2.0])
    ap.add_argument("--noises", type=float, nargs="+", default=[0.0, 1e-4, 1e-2, 0.1, 0.4])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--mode", default="margin", choices=["margin", "regrid"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    import gopt_local as _self; _self.MODE = a.mode
    globals()["MODE"] = a.mode
    rows = []
    for p in a.ps:
        for n in a.ns:
            if p ** n > 4_000_000:
                continue
            for q in a.qs:
                if q > n + 1:
                    continue
                top = -1e18; where = None
                for nz in a.noises:
                    for sd in range(a.seeds if nz > 0 else 1):
                        v, bc = run(p, n, q, nz, a.steps, sd, a.lr)
                        if v > top:
                            top = v; where = (nz, sd)
                # exact block check
                nu = n + 1; r = q / nu
                M1 = 1 - math.cos(2 * math.pi / p); s = math.sin(2 * math.pi / p)
                c0, v0 = block(n); U = grid_U(p, n)
                bl = float(lam_exact(c0, v0, U, M1, s)) / float((c0.abs() ** r).sum()) ** (1 / r) \
                    / (M1 * (n + 1) ** (-(1 - r) / r))
                rows.append(dict(p=p, n=n, q=q, best_ratio=top, block_ratio=bl, at=where))
                print(f"p={p:3d} n={n} q={q:g}   block ratio {bl:.12f}   best local-escape ratio {top:.12f}"
                      f"   (start noise {where[0]}, seed {where[1]})", flush=True)
    d = ROOT / "FINAL_RUNS" / "GOPT"; d.mkdir(parents=True, exist_ok=True)
    (d / f"local{a.tag}.json").write_text(json.dumps(rows, indent=1))
    print("wrote", d / f"local{a.tag}.json")
