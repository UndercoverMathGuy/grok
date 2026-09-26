"""gopt_nlp.py -- exact global search for the single-frequency cell, as a smooth NLP.

For a pure single-frequency population of N gated-chain neurons at frequency k, the EXACT margin is

    margin = 2^{-n} * min over u in (mu_p)^n, D = 1..p-1  of  [ (1 - cos th_D) Re h(u) + sin(th_D) Im h(u) ],
    h(u)   = sum_l c_l prod_j (1 + u_j v_lj),    th_D = 2 pi D / p,

and the cost is P_q = ((n+1) p / 2)^{q/2} * sum_l |c_l|^r, r = q/nu.  The target inequality (T) restricted
to a one-frequency support says exactly

    Lambda := 2^n * margin  <=  M_1 * (n+1)^{-(1-r)/r} * ||c||_r,      M_1 = 1 - cos(2 pi / p).

Every constraint above is LINEAR in c for fixed phases, so the whole question is the smooth NLP

    maximise t   s.t.   (1 - cos th_D) Re h(u) + sin(th_D) Im h(u) >= t   for all (u, D),
                        sum_l (|c_l|^2)^{r/2} = 1,
    ratio = t / ( M_1 (n+1)^{-(1-r)/r} ).

ratio > 1 refutes (T) on one-frequency supports.  ratio = 1 is attained by the (n+1)-block.
SLSQP, analytic Jacobians, many random restarts plus block-seeded restarts.  fp64.
"""
import argparse, json, math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

ROOT = Path(__file__).resolve().parent


def build(p, n):
    """Grid points U (M,n) complex and the (P,2) real coefficient pairs (1-cos th_D, sin th_D)."""
    u = np.exp(2j * np.pi * np.arange(p) / p)
    idx = np.stack(np.meshgrid(*[np.arange(p)] * n, indexing="ij"), -1).reshape(-1, n)
    U = u[idx]
    th = 2 * np.pi * np.arange(1, p) / p
    return U, np.stack([1 - np.cos(th), np.sin(th)], 1)


def unpack(x, N, n):
    c = x[:N] + 1j * x[N:2 * N]
    phi = x[2 * N:2 * N + N * n].reshape(N, n)
    return c, np.exp(1j * phi), x[-1]


def cons_and_jac(x, N, n, U, CD, r):
    c, v, t = unpack(x, N, n)
    phi = x[2 * N:2 * N + N * n].reshape(N, n)
    F = 1.0 + U[:, None, :] * v[None, :, :]                 # (M, N, n)
    P = F.prod(-1)                                          # (M, N)
    H = P @ c                                               # (M,)
    g = CD[:, 0][:, None] * H.real[None, :] + CD[:, 1][:, None] * H.imag[None, :] - t   # (P, M)
    # Jacobian
    #   dH/dRe c_l = P[:,l] ; dH/dIm c_l = i P[:,l]
    #   dH/dphi_lj = c_l * i v_lj U[:,j] * prod_{j'!=j} F[:,l,j']
    Q = P[:, :, None] / F                                   # (M, N, n)  = prod over j' != j
    dHdphi = (c[None, :, None] * 1j * v[None, :, :] * U[:, None, :] * Q)   # (M, N, n)
    J = np.empty((CD.shape[0], U.shape[0], x.size))
    a, b = CD[:, 0][:, None], CD[:, 1][:, None]
    for l in range(N):
        J[:, :, l] = a * P[:, l].real[None, :] + b * P[:, l].imag[None, :]
        J[:, :, N + l] = a * (1j * P[:, l]).real[None, :] + b * (1j * P[:, l]).imag[None, :]
        for j in range(n):
            J[:, :, 2 * N + l * n + j] = a * dHdphi[:, l, j].real[None, :] + b * dHdphi[:, l, j].imag[None, :]
    J[:, :, -1] = -1.0
    return g.reshape(-1), J.reshape(-1, x.size)


def norm_and_jac(x, N, n, r):
    c = x[:N] + 1j * x[N:2 * N]
    a2 = c.real ** 2 + c.imag ** 2 + 1e-300
    val = (a2 ** (r / 2)).sum() - 1.0
    j = np.zeros(x.size)
    d = r * a2 ** (r / 2 - 1)
    j[:N] = d * c.real
    j[N:2 * N] = d * c.imag
    return np.array([val]), j[None, :]


def solve(p, n, N, q, restarts, seed):
    nu = n + 1; r = q / nu
    M1 = 1 - math.cos(2 * math.pi / p)
    tgt = M1 * (n + 1) ** (-(1 - r) / r)
    U, CD = build(p, n)
    rng = np.random.default_rng(seed)
    best = (-1e18, None)
    starts = []
    # block-seeded starts (exact + perturbed) and random starts
    nb = min(N, n + 1)
    cb = np.full(N, 0.0, complex); cb[:nb] = 1.0 / (n + 1)
    vb = np.zeros((N, n))
    vb[:nb] = (-2 * np.pi * np.arange(nb) / (n + 1))[:, None]
    for nz in (0.0, 1e-6, 1e-3, 1e-2, 0.05, 0.2):
        for _ in range(1 if nz == 0 else 3):
            c0 = cb + nz * (rng.normal(size=N) + 1j * rng.normal(size=N))
            if N > n + 1 and nz == 0:
                c0[n + 1:] = 1e-3
            p0 = vb + nz * rng.normal(size=(N, n))
            starts.append((c0, p0))
    for _ in range(restarts):
        starts.append(((rng.normal(size=N) + 1j * rng.normal(size=N)) / N,
                       rng.uniform(0, 2 * np.pi, (N, n))))
    for c0, p0 in starts:
        s = (np.abs(c0) ** r).sum() ** (1 / r)
        c0 = c0 / max(s, 1e-12)
        x0 = np.concatenate([c0.real, c0.imag, p0.reshape(-1), [0.0]])
        res = minimize(lambda x: -x[-1], x0, jac=lambda x: np.eye(x.size)[-1] * -1.0,
                       method="SLSQP",
                       constraints=[{"type": "ineq", "fun": lambda x: cons_and_jac(x, N, n, U, CD, r)[0],
                                     "jac": lambda x: cons_and_jac(x, N, n, U, CD, r)[1]},
                                    {"type": "eq", "fun": lambda x: norm_and_jac(x, N, n, r)[0],
                                     "jac": lambda x: norm_and_jac(x, N, n, r)[1]}],
                       options={"maxiter": 400, "ftol": 1e-14})
        x = res.x
        gv, _ = cons_and_jac(x, N, n, U, CD, r)
        nv, _ = norm_and_jac(x, N, n, r)
        # exact value irrespective of SLSQP's feasibility slack
        c, v, _ = unpack(x, N, n)
        F = (1.0 + U[:, None, :] * v[None, :, :]).prod(-1)
        H = F @ c
        lam = (CD[:, 0][:, None] * H.real[None, :] + CD[:, 1][:, None] * H.imag[None, :]).min()
        cost = (np.abs(c) ** r).sum() ** (1 / r)
        val = lam / cost / tgt
        if val > best[0]:
            best = (val, (c.copy(), v.copy()))
    return best, tgt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", type=int, nargs="+", default=[5, 7, 11])
    ap.add_argument("--ns", type=int, nargs="+", default=[2])
    ap.add_argument("--qs", type=float, nargs="+", default=[2.0])
    ap.add_argument("--Ns", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--restarts", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    rows = []
    for p in a.ps:
        for n in a.ns:
            for q in a.qs:
                if q > n + 1:
                    continue
                for N in a.Ns:
                    (val, cv), tgt = solve(p, n, N, q, a.restarts, a.seed)
                    amps = sorted(np.abs(cv[0]).tolist(), reverse=True)
                    rows.append(dict(p=p, n=n, q=q, N=N, ratio=val,
                                     amps=[round(u, 8) for u in amps]))
                    print(f"p={p:3d} n={n} q={q:g} N={N:2d}   max ratio = {val:.15f}   "
                          f"amps {[f'{u:.5f}' for u in amps[:7]]}", flush=True)
    d = ROOT / "FINAL_RUNS" / "GOPT"; d.mkdir(parents=True, exist_ok=True)
    (d / f"nlp{a.tag}.json").write_text(json.dumps(rows, indent=1))
    print("wrote", d / f"nlp{a.tag}.json")
