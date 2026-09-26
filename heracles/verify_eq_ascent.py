"""verify_eq_ascent.py -- referee task 6: try to BEAT (T) at q = 2 with my own optimiser.

Independent of adv_net.py: different parametrisation (raw tensors, but a log-sum-exp soft-min of the
NORMALISED gap array), different optimiser (Adam with an annealed temperature, then a Nelder-free
L-BFGS polish), different random families, and a completely separate exact re-evaluation.

ratio(theta) = [ margin(theta) / P_q(theta)^{nu/q} ] / [ (n+1)^{-(nu-q)/q} B Gamma_{nu/q}(p) ].
(T) says ratio <= 1.  The share-weighted block gives exactly 1.  Any ratio > 1 is a counterexample.

The block is NEVER used as an initialisation.  Start families:
  gauss     -- i.i.d. normal tensors
  junkmax   -- exactly ONE neuron per frequency (maximal same-|k| junk)
  clustered -- support = four consecutive frequencies, random phases (a deliberately bad support)
  impure    -- every neuron carries two frequencies of comparable amplitude
  sparsecos -- random single-frequency cosine neurons with random independent phases per factor

Usage: uv run python heracles/verify_eq_ascent.py [--p 11 13] [--m 3 6 9 12] [--starts 24]
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "vqb", str(Path(__file__).resolve().parent / "verify_eq_block.py"))
vqb = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(vqb)

OUT = Path(__file__).resolve().parent / "FINAL_RUNS" / "VERIFY"
torch.set_default_dtype(torch.float64)


def rows_of(p, n):
    idx = torch.arange(p ** n)
    ops = torch.stack([(idx // p ** j) % p for j in range(n)], 1)
    return ops, ops.sum(1) % p


def ratio_torch(E, W, ops, y, p, n, q, rhs):
    """Differentiable soft version is built by the caller; this is the EXACT ratio."""
    h = torch.ones(ops.shape[0], E.shape[1])
    for j in range(n):
        h = h * E[j][:, ops[:, j]].T
    z = h @ W.T
    zy = z.gather(1, y[:, None])
    gaps = zy - z                                   # 0 in the correct column
    gaps = gaps.scatter(1, y[:, None], float("inf"))
    mg = gaps.min()
    nrm2 = (E ** 2).sum(dim=(0, 2)) + (W ** 2).sum(0)
    Pq = (nrm2 ** (q / 2)).sum()
    return mg / Pq ** (n + 1) ** 0 / Pq ** ((n + 1) / q - 1) / rhs if False else \
        (mg / Pq ** ((n + 1) / q)) / rhs, gaps, Pq


def soft_ratio(E, W, ops, y, p, n, q, rhs, beta):
    h = torch.ones(ops.shape[0], E.shape[1])
    for j in range(n):
        h = h * E[j][:, ops[:, j]].T
    z = h @ W.T
    zy = z.gather(1, y[:, None])
    gaps = zy - z
    nrm2 = (E ** 2).sum(dim=(0, 2)) + (W ** 2).sum(0) + 1e-300
    Pq = (nrm2 ** (q / 2)).sum()
    g = gaps / (Pq ** ((n + 1) / q) * rhs)          # degree-0 homogeneous
    g = g.scatter(1, y[:, None], 1e6)               # drop the correct class
    return -torch.logsumexp(-beta * g.reshape(-1), 0) / beta


def init(fam, p, n, m, rng):
    nf = (p - 1) // 2
    a = np.arange(p) * 2 * np.pi / p
    E = np.zeros((n, m, p)); W = np.zeros((p, m))
    if fam == "gauss":
        E = rng.normal(size=(n, m, p)); W = rng.normal(size=(p, m))
    elif fam == "junkmax":
        for i in range(m):
            k = 1 + (i % nf)
            for j in range(n):
                E[j, i] = np.cos(k * a + rng.uniform(0, 2 * np.pi))
            W[:, i] = np.cos(k * a + rng.uniform(0, 2 * np.pi))
    elif fam == "clustered":
        for i in range(m):
            k = 1 + (i % min(4, nf))
            for j in range(n):
                E[j, i] = np.cos(k * a + rng.uniform(0, 2 * np.pi))
            W[:, i] = np.cos(k * a + rng.uniform(0, 2 * np.pi))
    elif fam == "impure":
        for i in range(m):
            k1, k2 = rng.choice(np.arange(1, nf + 1), size=2, replace=False)
            for j in range(n):
                E[j, i] = np.cos(k1 * a + rng.uniform(0, 2 * np.pi)) + \
                          0.8 * np.cos(k2 * a + rng.uniform(0, 2 * np.pi))
            W[:, i] = np.cos(k1 * a + rng.uniform(0, 2 * np.pi)) + \
                      0.8 * np.cos(k2 * a + rng.uniform(0, 2 * np.pi))
    elif fam == "sparsecos":
        for i in range(m):
            k = int(rng.integers(1, nf + 1))
            for j in range(n):
                E[j, i] = rng.uniform(0.5, 1.5) * np.cos(k * a + rng.uniform(0, 2 * np.pi))
            W[:, i] = rng.uniform(0.5, 1.5) * np.cos(k * a + rng.uniform(0, 2 * np.pi))
    return E + 1e-3 * rng.normal(size=E.shape), W + 1e-3 * rng.normal(size=W.shape)


def _run_stages(E0, W0, ops, y, p, n, q, rhs, adam_steps):
    """Adam warm-up at a low inverse temperature, then annealed L-BFGS.  Returns (ratio, E, W)."""
    E = torch.tensor(np.ascontiguousarray(E0), requires_grad=True)
    W = torch.tensor(np.ascontiguousarray(W0), requires_grad=True)
    best, bestEW = -np.inf, (np.array(E0), np.array(W0))
    if adam_steps:
        opt = torch.optim.Adam([E, W], lr=0.05)
        for t in range(adam_steps):
            beta = 5.0 * (200.0 / 5.0) ** (t / max(adam_steps - 1, 1))
            opt.zero_grad()
            (-soft_ratio(E, W, ops, y, p, n, q, rhs, beta)).backward()
            opt.step()
    for beta in (50.0, 200.0, 800.0, 3000.0, 12000.0, 50000.0, 200000.0):
        with torch.no_grad():   # renormalise the scale (the objective is scale invariant)
            sc = float(torch.sqrt((E ** 2).sum() + (W ** 2).sum()))
            if sc > 0:
                E /= sc; W /= sc
        opt = torch.optim.LBFGS([E, W], max_iter=250, history_size=30,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            l = -soft_ratio(E, W, ops, y, p, n, q, rhs, beta)
            l.backward()
            return l
        try:
            opt.step(closure)
        except Exception:
            break
        with torch.no_grad():
            r, _, _ = ratio_torch(E, W, ops, y, p, n, q, rhs)
            if float(r) > best:
                best = float(r); bestEW = (E.detach().numpy().copy(), W.detach().numpy().copy())
    return best, bestEW[0], bestEW[1]


def ascend(p, n, q, m, fam, seed, steps=1800, hops=3, prog=None, rhs=None):
    rng = np.random.default_rng(seed)
    nu = n + 1
    if prog is None:
        prog = vqb.gamma_s(p, nu / q, starts=60)
        rhs = vqb.target_rhs(p, n, q, prog["value"])
    ops, y = rows_of(p, n)
    E0, W0 = init(fam, p, n, m, rng)
    best, bE, bW = _run_stages(E0, W0, ops, y, p, n, q, rhs, steps // 3)
    for _ in range(hops):                      # basin hopping around the incumbent
        sc = math.sqrt((bE ** 2).sum() + (bW ** 2).sum()) / math.sqrt(bE.size + bW.size)
        r2, E2, W2 = _run_stages(bE + 0.35 * sc * rng.normal(size=bE.shape),
                                 bW + 0.35 * sc * rng.normal(size=bW.shape),
                                 ops, y, p, n, q, rhs, 0)
        if r2 > best:
            best, bE, bW = r2, E2, W2
    return best, rhs, prog, bE, bW


def brute_ratio(E, W, p, n, q, rhs):
    """Completely separate numpy re-evaluation (loop-based margin), no torch."""
    z, y, _ = vqb.logit_table(E, W, p, n)
    mg = vqb.exact_margin(z, y)
    Pq = vqb.Pq_of(E, W, q)
    return (mg / Pq ** ((n + 1) / q)) / rhs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, nargs="+", default=[11, 13])
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--q", type=float, default=2.0)
    ap.add_argument("--m", type=int, nargs="+", default=[3, 6, 9, 12])
    ap.add_argument("--starts", type=int, default=6)
    ap.add_argument("--steps", type=int, default=1800)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fams = ["gauss", "junkmax", "clustered", "impure", "sparsecos"]
    rows, gmax = [], -np.inf
    t0 = time.time()
    for p in a.p:
        prog0 = vqb.gamma_s(p, (a.n + 1) / a.q, starts=80)
        rhs0 = vqb.target_rhs(p, a.n, a.q, prog0["value"])
        print(f"  [p={p}] Gamma_{{{(a.n+1)/a.q:g}}} = {prog0['value']:.12f}  RHS = {rhs0:.6e}", flush=True)
        for m in a.m:
            cell_best, cell_arg = -np.inf, None
            for fam in fams:
                for s in range(a.starts):
                    r, rhs, prog, E, W = ascend(p, a.n, a.q, m, fam, seed=1000 * s + hash(fam) % 997,
                                                steps=a.steps, prog=prog0, rhs=rhs0)
                    rb = brute_ratio(E, W, p, a.n, a.q, rhs)
                    r = max(r, rb)
                    if r > cell_best:
                        cell_best, cell_arg = r, (fam, s, rb)
                    if r > 1.0:
                        np.savez(OUT / f"VIOLATION_p{p}_m{m}_{fam}_s{s}.npz", E=E, W=W, ratio=r)
                        print(f"  !!! VIOLATION ratio {r:.15f} at p={p} m={m} {fam} seed {s}")
            gmax = max(gmax, cell_best)
            print(f"  p={p} m={m:2d}: best ratio {cell_best:.12f}  ({cell_arg[0]}, seed {cell_arg[1]}, "
                  f"brute re-eval {cell_arg[2]:.12f})", flush=True)
            rows.append(dict(p=p, n=a.n, q=a.q, m=m, best_ratio=cell_best,
                             family=cell_arg[0], seed=cell_arg[1], brute=cell_arg[2]))
    print(f"\nMAX RATIO over all cells: {gmax:.15f}   (>1 would break (T))   [{time.time()-t0:.0f}s]")
    (OUT / "eq_ascent.json").write_text(json.dumps(dict(rows=rows, max_ratio=gmax), indent=1))
    print("wrote", OUT / "eq_ascent.json")
