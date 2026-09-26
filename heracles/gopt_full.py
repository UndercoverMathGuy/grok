"""gopt_full.py -- global test of (T) ITSELF: max-margin over WHOLE gated-chain networks, as a smooth NLP.

No atom parameterisation, no purity assumption, no junk-free assumption, no single-frequency restriction:
the variables are the raw tensors E (n, m, p) and W (p, m), so every gap G1/G2/G3 is exercised at once.
The max-margin problem at fixed penalty is the smooth NLP

    maximise t   s.t.  z(a, s(a)) - z(a, c) >= t   for every row a and every wrong class c,
                       P_q(theta) = sum_i (sum_j ||e_ij||^2 + ||w_i||^2)^{q/2} = 1,
    gamma_q = t  (because P_q = 1),   ratio = gamma_q / [ (n+1)^{-(nu-q)/q} B Gamma_{nu/q}(p) ].

ratio > 1 anywhere REFUTES (T).  ratio = 1 is attained by the (n+1)-block on the programme's optimiser.
SLSQP with analytic Jacobians; starts = the constructed block on every ladder support, block + noise, and
random.  Small primes only (the constraint count is p^n (p-1)).  fp64.
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from kstar import ladder, G  # noqa: E402
from nu4_program import solve as prog_solve  # noqa: E402


def gamma_star(p, s, kmax=8):
    """Gamma_s(p): free-amplitude programme value, s = nu/q (exponent on the shares)."""
    g = G(p)[:, :(p - 1) // 2]
    nf = g.shape[0]
    lad = ladder(p, min(kmax, nf))
    rng = np.random.default_rng(0)
    best = 0.0
    for K in range(1, min(kmax, nf) + 1):
        S = np.array(lad["S"][K]) - 1 if K > 1 else np.array([0])
        v, _ = prog_solve(g[S], s, np.full(len(S), 1.0 / len(S)))
        best = max(best, v)
    for _ in range(40):
        b0 = rng.dirichlet(np.full(nf, 0.5))
        v, _ = prog_solve(g, s, b0)
        best = max(best, v)
    return best


def rows_and_wrong(p, n):
    idx = np.stack(np.meshgrid(*[np.arange(p)] * n, indexing="ij"), -1).reshape(-1, n)
    y = idx.sum(1) % p
    return idx, y


def build_block(p, n, S, m):
    """the (n+1)-neuron common-phase block on support S, padded with zero neurons to width m."""
    a = np.arange(p) * 2 * np.pi / p
    E = np.zeros((n, m, p)); W = np.zeros((p, m)); i = 0
    for k in S:
        for l in range(n + 1):
            ph = np.pi * l / (n + 1)
            for j in range(n):
                E[j, i] = np.cos(k * a + ph)
            W[:, i] = np.cos(k * a + n * ph); i += 1
    return E, W


def pack(E, W):
    return np.concatenate([E.reshape(-1), W.reshape(-1)])


def unpack(x, p, n, m):
    nE = n * m * p
    return x[:nE].reshape(n, m, p), x[nE:nE + p * m].reshape(p, m)


def margin_cons(x, p, n, m, idx, y, t_in_x=True):
    E, W = unpack(x, p, n, m)
    t = x[-1]
    H = np.ones((idx.shape[0], m))
    for j in range(n):
        H = H * E[j][:, idx[:, j]].T                       # (R, m)
    z = H @ W.T                                            # (R, p)
    zy = z[np.arange(len(y)), y]
    g = (zy[:, None] - z) - t                              # (R, p); the c = y column is -t
    # Jacobian
    R = idx.shape[0]
    J = np.zeros((R, p, x.size))
    # d z[a,c] / d W[c,i] = H[a,i]
    nE = n * m * p
    for i in range(m):
        # w.r.t. W
        # d(zy - z_c)/dW[c', i] = H[a,i]*( [c'=y] - [c'=c] )
        pass
    # explicit assembly (loops over m and p are fine at these sizes)
    for i in range(m):
        col = nE + i * 1
        for c in range(p):
            J[:, c, nE + c * m + i] -= H[:, i]
        J[np.arange(R), :, nE + y * m + i] += H[:, i][:, None]
    # w.r.t. E: d z[a,c]/dE[j,i,b] = [a_j = b] * W[c,i] * prod_{j'!=j} E[j',i,a_{j'}]
    for j in range(n):
        Q = np.ones((R, m))
        for j2 in range(n):
            if j2 != j:
                Q = Q * E[j2][:, idx[:, j2]].T
        for i in range(m):
            base = j * m * p + i * p
            # d(zy - z_c) = Q[a,i]*(W[y,i]-W[c,i]) at column base + a_j
            coef = Q[:, i][:, None] * (W[y, i][:, None] - W[:, i][None, :])     # (R, p)
            for b in range(p):
                sel = idx[:, j] == b
                J[sel, :, base + b] += coef[sel]
    J[:, :, -1] = -1.0
    return g.reshape(-1), J.reshape(-1, x.size)


def pen_cons(x, p, n, m, q):
    E, W = unpack(x, p, n, m)
    nrm2 = (E ** 2).sum((0, 2)) + (W ** 2).sum(0)           # (m,)
    val = (nrm2 ** (q / 2)).sum() - 1.0
    d = (q / 2) * nrm2 ** (q / 2 - 1)                       # d/d nrm2
    gE = 2 * E * d[None, :, None]
    gW = 2 * W * d[None, :]
    return np.array([val]), np.concatenate([gE.reshape(-1), gW.reshape(-1), [0.0]])[None, :]


def run(p, n, q, m, restarts, seed, kmax):
    nu = n + 1
    Bc = 2 * (2 * p * (n + 1)) ** (-nu / 2)
    Gs = gamma_star(p, nu / q, kmax)
    tgt = (n + 1) ** (-(nu - q) / q) * Bc * Gs
    idx, y = rows_and_wrong(p, n)
    rng = np.random.default_rng(seed)
    lad = ladder(p, min(kmax, (p - 1) // 2))
    starts = []
    for K in range(1, min(kmax, (p - 1) // 2, m // (n + 1)) + 1):
        S = lad["S"][K] if K > 1 else [1]
        E0, W0 = build_block(p, n, S, m)
        for nz in (0.0, 0.05, 0.3):
            starts.append((E0 + nz * rng.normal(size=E0.shape), W0 + nz * rng.normal(size=W0.shape)))
    for _ in range(restarts):
        starts.append((rng.normal(size=(n, m, p)), rng.normal(size=(p, m))))
    best = (-1e18, None)
    for E0, W0 in starts:
        nrm2 = (E0 ** 2).sum((0, 2)) + (W0 ** 2).sum(0)
        sc = ((nrm2 ** (q / 2)).sum()) ** (-1 / q)
        E0, W0 = E0 * sc, W0 * sc
        x0 = np.concatenate([pack(E0, W0), [0.0]])
        res = minimize(lambda x: -x[-1], x0, jac=lambda x: np.concatenate([np.zeros(x.size - 1), [-1.0]]),
                       method="SLSQP",
                       constraints=[{"type": "ineq", "fun": lambda x: margin_cons(x, p, n, m, idx, y)[0],
                                     "jac": lambda x: margin_cons(x, p, n, m, idx, y)[1]},
                                    {"type": "eq", "fun": lambda x: pen_cons(x, p, n, m, q)[0],
                                     "jac": lambda x: pen_cons(x, p, n, m, q)[1]}],
                       options={"maxiter": 300, "ftol": 1e-12})
        E, W = unpack(res.x, p, n, m)
        H = np.ones((idx.shape[0], m))
        for j in range(n):
            H = H * E[j][:, idx[:, j]].T
        z = H @ W.T
        zy = z[np.arange(len(y)), y]
        zz = z.copy(); zz[np.arange(len(y)), y] = -np.inf
        mg = float((zy - zz.max(1)).min())
        nrm2 = (E ** 2).sum((0, 2)) + (W ** 2).sum(0)
        Pq = float((nrm2 ** (q / 2)).sum())
        val = mg / Pq ** (nu / q) / tgt
        if val > best[0]:
            best = (val, (E.copy(), W.copy()))
    return best[0], tgt, Gs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", type=int, nargs="+", default=[5, 7])
    ap.add_argument("--ns", type=int, nargs="+", default=[2])
    ap.add_argument("--qs", type=float, nargs="+", default=[2.0])
    ap.add_argument("--ms", type=int, nargs="+", default=[6, 9, 12])
    ap.add_argument("--restarts", type=int, default=20)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    rows = []
    for p in a.ps:
        for n in a.ns:
            for q in a.qs:
                if q > n + 1:
                    continue
                for m in a.ms:
                    v, tgt, Gs = run(p, n, q, m, a.restarts, a.seed, a.kmax)
                    rows.append(dict(p=p, n=n, q=q, m=m, ratio=v, target=tgt, Gamma=Gs))
                    print(f"p={p} n={n} q={q:g} width m={m:3d}   Gamma_s={Gs:.6f}  "
                          f"target gamma_q={tgt:.6e}   best/target = {v:.12f}", flush=True)
    d = ROOT / "FINAL_RUNS" / "GOPT"; d.mkdir(parents=True, exist_ok=True)
    (d / f"full{a.tag}.json").write_text(json.dumps(rows, indent=1))
    print("wrote", d / f"full{a.tag}.json")
