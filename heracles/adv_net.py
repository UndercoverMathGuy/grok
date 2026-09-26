"""adv_net.py -- ADVERSARIAL attack on the target (T) over WHOLE gated-chain networks.

(T)      gamma_q(theta)  <=  (n+1)^{-(nu-q)/q} * B * Gamma_{nu/q}(p),    B = 2 (2 p (n+1))^{-nu/2},

for the gated chain: neuron i has operand embeddings E[j, i, :] in R^p (j = 1..n) and
readout W[:, i] in R^p; logit z(a, c) = sum_i W[c, i] prod_j E[j, i, a_j]; the penalty is
P_q = sum_i (sum_j ||E[j,i]||^2 + ||W[:,i]||^2)^{q/2}; the normalised margin is
gamma_q = margin / P_q^{nu/q} with margin = min over all p^n rows and all wrong classes.

Plain reading: (T) says no network can beat the (n+1)-neuron-per-frequency "block" on the
programme's optimal frequency support.  ratio := gamma_q / target > 1 refutes it.

Differences from the inherited search (gopt_full.py, gopt4_trade.py), which are the
biases this script exists to remove:
  * the block is NEVER among the starts (--noblock, the default);
  * primes p = 11 .. 29 instead of 5, 7 (the inherited whole-network NLP stopped at 7);
  * a scale-invariant annealed soft-min ascent with an analytic gradient (L-BFGS-B),
    which does not degrade with p the way SLSQP on p^n (p-1) explicit constraints does;
  * basin hopping, impure-neuron starts, clustered/non-optimal-support starts;
  * every reported ratio is recomputed from the EXACT margin over the full p^n x p table
    in fp64 by an independent code path (brute_ratio).

Usage:  uv run python heracles/adv_net.py --p 11 13 --n 2 --q 2 --m 9 12 --starts 40
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "FINAL_RUNS" / "ADV"

from kstar import ladder, G  # noqa: E402
from nu4_program import solve as prog_solve  # noqa: E402


# ------------------------------------------------------------------ the programme
def gamma_prog(p, s, kmax=10, extra=200, seed=0):
    """Gamma_s(p) = max { min_{D!=0} sum_k x_k (1-cos 2 pi k D/p) : x >= 0, sum_k x_k^{1/s} <= 1 }.

    A maximum, so every feasible point is a lower bound; searching harder can only
    RAISE it, which can only lower the ratios below.  We therefore search hard.
    """
    g = G(p)[:, :(p - 1) // 2]
    nf = g.shape[0]
    lad = ladder(p, min(kmax, nf))
    rng = np.random.default_rng(seed)
    best, bestb, bestS = 0.0, None, None
    for K in range(1, min(kmax, nf) + 1):
        S = np.array(lad["S"][K]) - 1 if K > 1 else np.array([0])
        for b0 in [np.full(len(S), 1.0 / len(S))] + [rng.dirichlet(np.full(len(S), 3.0)) for _ in range(6)]:
            v, b = prog_solve(g[S], s, b0)
            if v > best:
                best, bestb, bestS = v, b, S
    for _ in range(extra):
        K = int(rng.integers(1, min(nf, 12) + 1))
        S = rng.choice(nf, K, replace=False)
        v, b = prog_solve(g[S], s, rng.dirichlet(np.full(K, 1.0)))
        if v > best:
            best, bestb, bestS = v, b, S
    return best, (np.array(bestS) + 1).tolist(), bestb.tolist()


# ------------------------------------------------------------------ the network
class Task:
    def __init__(self, p, n):
        self.p, self.n = p, n
        r = np.arange(p ** n)
        self.idx = np.stack([(r // p ** j) % p for j in range(n)], 1)   # (R, n)
        self.y = self.idx.sum(1) % p
        self.R = p ** n
        self.ar = np.arange(self.R)


def forward(E, W, tk):
    H = np.ones((tk.R, E.shape[1]))
    for j in range(tk.n):
        H = H * E[j][:, tk.idx[:, j]].T
    z = H @ W.T
    return H, z


def exact_margin(E, W, tk):
    _, z = forward(E, W, tk)
    zy = z[tk.ar, tk.y]
    zz = z.copy()
    zz[tk.ar, tk.y] = -np.inf
    return float((zy - zz.max(1)).min())


def penalty(E, W, q):
    nrm2 = (E ** 2).sum((0, 2)) + (W ** 2).sum(0)
    return float((nrm2 ** (q / 2)).sum()), nrm2


def brute_ratio(E, W, tk, q, tgt):
    """Independent, loop-based exact evaluation of gamma_q / target (fp64)."""
    p, n = tk.p, tk.n
    m = E.shape[1]
    nu = n + 1
    best = np.inf
    for a in range(p ** n):
        ops = [(a // p ** j) % p for j in range(n)]
        h = np.ones(m)
        for j in range(n):
            h = h * E[j][:, ops[j]]
        z = W @ h
        y = sum(ops) % p
        for c in range(p):
            if c != y:
                best = min(best, z[y] - z[c])
    P = 0.0
    for i in range(m):
        s = sum(float((E[j, i] ** 2).sum()) for j in range(n)) + float((W[:, i] ** 2).sum())
        P += s ** (q / 2)
    return best / P ** (nu / q) / tgt


# ------------------------------------------------------------------ objective
def neg_obj(x, tk, m, q, beta, tgt):
    """-softmin_beta( gaps / (P_q^{nu/q} tgt) ), degree-0 homogeneous, with gradient."""
    p, n = tk.p, tk.n
    nu = n + 1
    nE = n * m * p
    E = x[:nE].reshape(n, m, p)
    W = x[nE:].reshape(p, m)
    H, z = forward(E, W, tk)
    zy = z[tk.ar, tk.y]
    g = zy[:, None] - z                                   # (R, p); the c = y column is 0
    P, nrm2 = penalty(E, W, q)
    if P <= 0:
        return 1e9, np.zeros_like(x)
    Pn = P ** (nu / q) * tgt
    gh = g / Pn
    mask = np.ones_like(gh, dtype=bool)
    mask[tk.ar, tk.y] = False
    v = gh[mask]
    v0 = v.min()
    e = np.exp(-beta * (v - v0))
    Z = e.sum()
    s = v0 - np.log(Z) / beta
    w = np.zeros_like(gh)
    w[mask] = e / Z
    # d s / d g[a,c] = w[a,c]/Pn ; d s / d Pn = -(sum w gh)/Pn
    cw = float((w * gh).sum())
    # gradient wrt the logits: Q[a,c] = w[a,c] for c != y, Q[a,y] = -sum_c w[a,c]
    Q = w.copy()
    Q[tk.ar, tk.y] = -w.sum(1)
    # g[a,c] = sum_i H[a,i](W[y,i] - W[c,i]) ; d/dW[c',i] = -Q^T H
    gW = -(Q.T @ H) / Pn
    GH = -(Q @ W) / Pn                                    # (R, m)
    gE = np.zeros_like(E)
    for j in range(n):
        Qx = np.ones((tk.R, m))
        for j2 in range(n):
            if j2 != j:
                Qx = Qx * E[j2][:, tk.idx[:, j2]].T
        contrib = GH * Qx                                 # (R, m)
        np.add.at(gE[j].T, tk.idx[:, j], contrib)
    # d Pn / d params
    dP = (q / 2) * nrm2 ** (q / 2 - 1)
    dPn_dE = 2 * E * dP[None, :, None] * (nu / q) * P ** (nu / q - 1) * tgt
    dPn_dW = 2 * W * dP[None, :] * (nu / q) * P ** (nu / q - 1) * tgt
    gE = gE - cw / Pn * dPn_dE
    gW = gW - cw / Pn * dPn_dW
    return -s, -np.concatenate([gE.reshape(-1), gW.reshape(-1)])


def ascend(x0, tk, m, q, tgt, betas, maxiter=400):
    x = x0.copy()
    for b in betas:
        res = minimize(neg_obj, x, args=(tk, m, q, b, tgt), jac=True,
                       method="L-BFGS-B",
                       options={"maxiter": maxiter, "maxfun": 3 * maxiter,
                                "ftol": 1e-16, "gtol": 1e-14})
        x = res.x
        nn = np.linalg.norm(x)
        if nn > 0:
            x = x / nn * np.sqrt(x0.size)       # keep the scale sane (degree-0 objective)
    return x


# ------------------------------------------------------------------ starts
def make_starts(rng, tk, m, nstart, kinds):
    p, n = tk.p, tk.n
    S = []
    a = np.arange(p) * 2 * np.pi / p
    nf = (p - 1) // 2
    for s in range(nstart):
        kind = kinds[s % len(kinds)]
        if kind == "rand":
            E = rng.normal(size=(n, m, p))
            W = rng.normal(size=(p, m))
        elif kind == "randsmall":
            E = rng.normal(size=(n, m, p)) * rng.uniform(0.2, 2.0, (1, m, 1))
            W = rng.normal(size=(p, m)) * rng.uniform(0.2, 2.0, (1, m))
        elif kind == "cos1":                     # one random-phase cosine neuron per slot
            E = np.zeros((n, m, p))
            W = np.zeros((p, m))
            for i in range(m):
                k = int(rng.integers(1, nf + 1))
                ph = rng.uniform(0, 2 * np.pi, n)
                for j in range(n):
                    E[j, i] = np.cos(k * a + ph[j])
                W[:, i] = np.cos(k * a + rng.uniform(0, 2 * np.pi))
        elif kind == "impure":                   # each neuron carries TWO frequencies
            E = np.zeros((n, m, p))
            W = np.zeros((p, m))
            for i in range(m):
                k1, k2 = rng.choice(np.arange(1, nf + 1), 2, replace=False)
                w2 = rng.uniform(0.2, 1.0)
                for j in range(n):
                    E[j, i] = (np.cos(k1 * a + rng.uniform(0, 2 * np.pi))
                               + w2 * np.cos(k2 * a + rng.uniform(0, 2 * np.pi)))
                W[:, i] = (np.cos(k1 * a + rng.uniform(0, 2 * np.pi))
                           + w2 * np.cos(k2 * a + rng.uniform(0, 2 * np.pi)))
        elif kind == "clustered":                # a deliberately NON-optimal support
            E = np.zeros((n, m, p))
            W = np.zeros((p, m))
            k0 = int(rng.integers(1, max(2, nf - 4)))
            for i in range(m):
                k = k0 + i % 4
                ph = rng.uniform(0, 2 * np.pi, n)
                for j in range(n):
                    E[j, i] = np.cos(k * a + ph[j])
                W[:, i] = np.cos(k * a + rng.uniform(0, 2 * np.pi))
        elif kind == "junk":                     # single neuron per frequency = max junk
            E = np.zeros((n, m, p))
            W = np.zeros((p, m))
            ks = rng.choice(np.arange(1, nf + 1), min(m, nf), replace=False)
            for i, k in enumerate(ks):
                for j in range(n):
                    E[j, i] = np.cos(k * a)
                W[:, i] = np.cos(k * a)
            E = E + 0.05 * rng.normal(size=E.shape)
            W = W + 0.05 * rng.normal(size=W.shape)
        else:
            raise ValueError(kind)
        S.append(np.concatenate([E.reshape(-1), W.reshape(-1)]))
    return S


# ------------------------------------------------------------------ driver
def run_cell(p, n, q, m, nstart, seed, betas, hops, kinds, log, tgt_cache):
    nu = n + 1
    key = (p, n, q)
    if key not in tgt_cache:
        Gs, Sb, bb = gamma_prog(p, nu / q)
        Bc = 2 * (2 * p * (n + 1)) ** (-nu / 2)
        tgt_cache[key] = ((n + 1) ** (-(nu - q) / q) * Bc * Gs, Gs, Sb, bb)
    tgt, Gs, Sb, bb = tgt_cache[key]
    tk = Task(p, n)
    rng = np.random.default_rng(seed)
    starts = make_starts(rng, tk, m, nstart, kinds)
    best, bx = -np.inf, None
    t0 = time.time()
    for si, x0 in enumerate(starts):
        x = ascend(x0, tk, m, q, tgt, betas)
        for _ in range(hops):                         # basin hopping
            xn = x + rng.normal(size=x.size) * 0.15 * np.linalg.norm(x) / np.sqrt(x.size)
            xn = ascend(xn, tk, m, q, tgt, betas[-3:])
            if ratio_of(xn, tk, m, q, tgt) > ratio_of(x, tk, m, q, tgt):
                x = xn
        v = ratio_of(x, tk, m, q, tgt)
        if v > best:
            best, bx = v, x.copy()
    return dict(p=p, n=n, q=q, m=m, Gamma=Gs, prog_support=Sb, target=tgt,
                best_ratio=best, nstart=nstart, secs=time.time() - t0), bx


def ratio_of(x, tk, m, q, tgt):
    p, n = tk.p, tk.n
    nE = n * m * p
    E = x[:nE].reshape(n, m, p)
    W = x[nE:].reshape(p, m)
    mg = exact_margin(E, W, tk)
    P, _ = penalty(E, W, q)
    return mg / P ** ((n + 1) / q) / tgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, nargs="+", default=[11, 13])
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--q", type=float, nargs="+", default=[2.0])
    ap.add_argument("--m", type=int, nargs="+", default=[9, 12])
    ap.add_argument("--starts", type=int, default=24)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--kinds", nargs="+",
                    default=["rand", "randsmall", "cos1", "impure", "clustered", "junk"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(OUT / f"net{a.tag}.log", "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    betas = [8.0, 25.0, 80.0, 250.0, 800.0, 2500.0, 8000.0, 25000.0]
    rows = []
    tgt_cache = {}
    for p in a.p:
        for q in a.q:
            if q > a.n + 1:
                continue
            for m in a.m:
                row, bx = run_cell(p, a.n, q, m, a.starts, a.seed, betas, a.hops,
                                   a.kinds, log, tgt_cache)
                # independent exact re-evaluation of the winner
                tk = Task(p, a.n)
                nE = a.n * m * p
                E = bx[:nE].reshape(a.n, m, p)
                W = bx[nE:].reshape(p, m)
                row["brute_ratio"] = float(brute_ratio(E, W, tk, q, row["target"]))
                rows.append(row)
                log(f"p={p} n={a.n} q={q:g} m={m:3d}  Gamma={row['Gamma']:.9f} "
                    f"S*={row['prog_support']}  target={row['target']:.6e}  "
                    f"best ratio = {row['best_ratio']:.12f}  "
                    f"(brute {row['brute_ratio']:.12f})  [{row['secs']:.0f}s]")
                if row["best_ratio"] > 1.0 + 1e-9:
                    np.savez(OUT / f"VIOLATION_p{p}_n{a.n}_q{q}_m{m}.npz", E=E, W=W,
                             ratio=row["best_ratio"], target=row["target"])
                    log(f"*** CANDIDATE VIOLATION saved: ratio {row['best_ratio']:.12f}")
    (OUT / f"net{a.tag}.json").write_text(json.dumps(rows, indent=1, default=float))
    log(f"wrote {OUT / f'net{a.tag}.json'}")


if __name__ == "__main__":
    main()
