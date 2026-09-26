"""lemma_ascent.py -- COUNTEREXAMPLE HUNT: direct maximisation of gamma_q over gated-chain networks.

We maximise the normalised margin itself (no cross-entropy, no weight-decay dynamics):

    gamma_q(theta) = margin(theta) / P_q(theta)^{nu/q},
    margin  = min over all p^n rows and all p-1 wrong classes of (z_correct - z_wrong),
    P_q     = sum_i ||theta_i||^q, ||theta_i||^2 = sum_j ||E_j[i,:]||^2 + ||W[:,i]||^2,

by annealed soft-min + Adam in float64, from many random starts plus a warm start on the
(n+1)-neuron common-phase block.  The conjecture under test is

    gamma_q  <=  (n+1)^{-(nu-q)/q} * B * Gamma_{nu/q}(p),     B = 2 (2 p (n+1))^{-nu/2},

with Gamma_s(p) the free-amplitude programme (computed here by exhaustive support enumeration plus
SLSQP on the shares -- exact for these tiny primes).  ANY run with ratio > 1 refutes the target.

Usage: uv run python heracles/lemma_ascent.py --p 11 --n 2 --q 2
"""
import argparse, itertools, json, math
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

torch.set_default_dtype(torch.float64)
OUT = Path(__file__).resolve().parent / "FINAL_RUNS" / "LEMMA"


# ---------------------------------------------------------------- the programme Gamma_s(p)
def gamma_prog(p, s):
    """Gamma_s(p) = max_{b >= 0, sum b = 1} min_{D != 0} sum_k b_k^s (1 - cos(2 pi k D / p)).
    Exhaustive over supports (nf <= 6 here) x SLSQP on the shares, 60 restarts each."""
    nf = (p - 1) // 2
    D = np.arange(1, p)
    G = 1 - np.cos(2 * np.pi * np.outer(np.arange(1, nf + 1), D) / p)   # (nf, p-1)
    rng = np.random.default_rng(1)
    best, bestb = -1.0, None
    for r in range(1, nf + 1):
        for S in itertools.combinations(range(nf), r):
            Gs = G[list(S)]
            for _ in range(60):
                b0 = rng.dirichlet(np.ones(r))

                def negval(x):
                    return -float(((np.abs(x) ** s) @ Gs).min())

                res = minimize(negval, b0, method="SLSQP",
                               constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}],
                               bounds=[(0, 1)] * r, options={"maxiter": 400, "ftol": 1e-14})
                v = -res.fun
                if v > best:
                    best, bestb = v, (list(np.array(S) + 1), np.abs(res.x).tolist())
    return best, bestb


# ---------------------------------------------------------------- net
def rows(p, n):
    ops = torch.tensor(list(itertools.product(range(p), repeat=n)))
    return ops, ops.sum(1) % p


def gaps(E, W, ops, y, p, n):
    h = torch.ones(ops.shape[0], E.shape[1], dtype=torch.float64)
    for j in range(n):
        h = h * E[j][:, ops[:, j]].T
    z = h @ W.T
    zy = z.gather(1, y[:, None])
    d = zy - z                                   # (rows, p); column y is 0
    mask = torch.ones_like(d, dtype=torch.bool)
    mask.scatter_(1, y[:, None], False)
    return d[mask].reshape(len(y), p - 1)


def pen(E, W, q):
    t2 = E.square().sum(dim=(0, 2)) + W.square().sum(0)
    return (t2 ** (q / 2)).sum()


def ascent(p, n, q, m, steps, seed, warm=False, wnoise=0.02, beta0=1.0, dev="cpu"):
    nu = n + 1
    ops, y = rows(p, n)
    g = torch.Generator().manual_seed(seed)
    if warm:
        a = torch.arange(p) * 2 * math.pi / p
        S = list(range(1, min((p - 1) // 2, max(1, m // (n + 1))) + 1))
        mm = (n + 1) * len(S)
        E = torch.zeros(n, mm, p); W = torch.zeros(p, mm); i = 0
        for k in S:
            for l in range(n + 1):
                ph = math.pi * l / (n + 1)
                for j in range(n):
                    E[j, i] = torch.cos(k * a + ph)
                W[:, i] = torch.cos(k * a + n * ph); i += 1
        E = E + wnoise * torch.randn(E.shape, generator=g)
        W = W + wnoise * torch.randn(W.shape, generator=g)
    else:
        E = torch.randn(n, m, p, generator=g)
        W = torch.randn(p, m, generator=g)
    E.requires_grad_(True); W.requires_grad_(True)
    opt = torch.optim.Adam([E, W], lr=0.05)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)
    b0, b1 = beta0, 3e-3                      # soft-min temperature RELATIVE to the current gap scale
    best = -np.inf
    for t in range(steps):
        beta = b0 * (b1 / b0) ** (t / max(1, steps - 1))
        d = gaps(E, W, ops, y, p, n)
        P = pen(E, W, q)
        with torch.no_grad():
            scale = d.abs().mean().clamp_min(1e-300)
        T = beta * scale
        sm = -T * torch.logsumexp(-d.reshape(-1) / T - math.log(d.numel()), 0)
        obj = sm / P ** (nu / q)
        opt.zero_grad(); (-obj).backward(); opt.step(); sch.step()
        if t % 25 == 0 or t == steps - 1:
            with torch.no_grad():
                hard = float(gaps(E, W, ops, y, p, n).min()) / float(pen(E, W, q)) ** (nu / q)
                best = max(best, hard)
    # polish: fixed very low temperature, small steps -- turns the soft optimum into a hard one
    opt = torch.optim.Adam([E, W], lr=2e-3)
    for t in range(steps // 2):
        d = gaps(E, W, ops, y, p, n)
        P = pen(E, W, q)
        with torch.no_grad():
            scale = d.abs().mean().clamp_min(1e-300)
        T = 1e-3 * scale
        sm = -T * torch.logsumexp(-d.reshape(-1) / T - math.log(d.numel()), 0)
        obj = sm / P ** (nu / q)
        opt.zero_grad(); (-obj).backward(); opt.step()
        if t % 25 == 0 or t == steps // 2 - 1:
            with torch.no_grad():
                hard = float(gaps(E, W, ops, y, p, n).min()) / float(pen(E, W, q)) ** (nu / q)
                best = max(best, hard)
    with torch.no_grad():
        d = gaps(E, W, ops, y, p, n); P = pen(E, W, q)
        hard = float(d.min()) / float(P) ** (nu / q)
        t2 = (E.square().sum(dim=(0, 2)) + W.square().sum(0))
        live = int((t2 > 1e-6 * t2.max()).sum())
    return max(best, hard), live


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=11)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--q", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--local", action="store_true", help="local ascent from the exact (n+1)-block: tiny noise, cold start")
    a = ap.parse_args()
    p, n, q = a.p, a.n, a.q
    nu = n + 1
    s = nu / q
    G, bb = gamma_prog(p, s)
    bb = ([int(u) for u in bb[0]], [float(u) for u in bb[1]])
    B = 2 * (2 * p * (n + 1)) ** (-nu / 2)
    target = (n + 1) ** (-(nu - q) / q) * B * G
    print(f"p={p} n={n} nu={nu} q={q:g}  Gamma_{s:g}({p}) = {G:.8f} on support {bb[0]} shares "
          f"{[round(u,4) for u in bb[1]]}", flush=True)
    print(f"  upper bound B*Gamma            = {B*G:.8e}", flush=True)
    print(f"  conjectured sup (n+1)^-(nu-q)/q = {target:.8e}", flush=True)
    res = []
    for m in (n + 1, 2 * (n + 1), 3 * (n + 1), 4 * (n + 1), 5 * (n + 1)):
        for sd in range(a.seeds):
            gam, live = ascent(p, n, q, m, a.steps, sd, warm=(sd == 0 or a.local),
                               wnoise=(1e-6 if a.local else 0.02), beta0=(0.01 if a.local else 1.0))
            res.append(dict(m=m, seed=sd, warm=sd == 0, gamma=gam, live=live,
                            ratio_target=gam / target, ratio_bound=gam / (B * G)))
            print(f"  m={m:3d} seed={sd:2d}{' (warm)' if sd==0 else '       '}  gamma {gam:.6e}"
                  f"  /conj {gam/target:.6f}  /bound {gam/(B*G):.6f}  live {live}", flush=True)
    bestr = max(r["ratio_target"] for r in res)
    print(f"\n  BEST gamma / conjectured sup = {bestr:.8f}   "
          f"{'*** REFUTED ***' if bestr > 1 + 1e-6 else '(no violation)'}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"ascent_p{p}_n{n}_q{q:g}{'_local' if a.local else ''}.json").write_text(json.dumps(
        dict(p=p, n=n, q=q, Gamma=G, support=bb, B=B, target=target, runs=res, best=bestr), indent=1))
