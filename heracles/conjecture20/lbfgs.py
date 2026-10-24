# Annealed soft-min normalized margin with L-BFGS, then SLSQP polish (min P s.t. gaps>=1).
import torch, numpy as np, itertools, math, sys
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from slsqp import make, iff
from gamma import Gamma
from scipy.optimize import minimize
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

def ds(p, n):
    X = torch.tensor(list(itertools.product(range(p), repeat=n)))
    return X, X.sum(1) % p

def ngaps(th, p, n, b, m, X, y):
    U = th[: m * n * p].view(m, n, p); W = th[m * n * p:].view(m, p)
    Z = torch.ones(m, X.shape[0])
    for t in range(n): Z = Z * U[:, t, X[:, t]]
    f = Z.T @ W
    g = f.gather(1, y[:, None]) - f
    mask = torch.ones_like(g, dtype=torch.bool); mask.scatter_(1, y[:, None], False)
    sq = (U ** 2).sum((1, 2)) + (W ** 2).sum(1)
    P = (sq ** (b / 2)).sum()
    return g[mask] / P ** ((n + 1) / b)

def run(p, n, b, m, seed, T):
    torch.manual_seed(seed)
    X, y = ds(p, n)
    th = torch.randn(m * (n * p + p), requires_grad=True)
    for kappa in [1.0, 0.3, 0.1, 0.03, 0.01, 0.003]:
        tau = kappa * T
        opt = torch.optim.LBFGS([th], lr=1, max_iter=400, line_search_fn="strong_wolfe", tolerance_grad=1e-14, tolerance_change=1e-16)
        def closure():
            opt.zero_grad()
            g = ngaps(th, p, n, b, m, X, y)
            loss = tau * torch.logsumexp(-g / tau, 0)
            loss.backward(); return loss
        try: opt.step(closure)
        except Exception: pass
        with torch.no_grad():  # keep scale sane
            sq = (th ** 2).sum(); th.mul_(1 / sq.sqrt() * math.sqrt(m))
    with torch.no_grad():
        g = ngaps(th, p, n, b, m, X, y)
    return g.min().item(), th.detach().numpy().copy()

def polish(p, n, b, m, th):
    gaps, gj, pen, pj = make(p, n, b, m)
    g = gaps(th)
    if g.min() <= 0: return -1, th
    th = th / g.min() ** (1 / (n + 1))
    r = minimize(pen, th, jac=pj, method="SLSQP",
                 constraints=[{"type": "ineq", "fun": lambda t: gaps(t) - 1.0, "jac": gj}],
                 options={"maxiter": 3000, "ftol": 1e-15})
    g = gaps(r.x); P = pen(r.x)
    return (g.min() / P ** ((n + 1) / b) if g.min() > 0 else -1), r.x

def job(a):
    p, n, b, m, sd = a
    T = iff(p, n, b, Gamma(p, (n + 1) / b))
    g0, th = run(p, n, b, m, sd, T)
    g1, th1 = polish(p, n, b, m, th)
    return a, g0 / T, g1 / T, th1

if __name__ == "__main__":
    from multiprocessing import Pool
    cfgs = eval(sys.argv[1]); seeds = int(sys.argv[2])
    jobs = [(p, n, b, m, sd) for (p, n, b, m) in cfgs for sd in range(seeds)]
    out = {}; best = {}
    with Pool(4) as pool:
        for a, r0, r1, th in pool.imap_unordered(job, jobs):
            k = a[:4]; out.setdefault(k, []).append((max(r0, r1), r0))
            if max(r0, r1) > best.get(k, (-9,))[0]: best[k] = (max(r0, r1), th)
            print(a, f"lbfgs={r0:.5f} polished={r1:.6f}", flush=True)
    for k in sorted(out):
        v = sorted(out[k], reverse=True)
        print("SUMMARY", k, "top", [round(x[0], 5) for x in v[:6]], flush=True)
        np.save("lb_%s.npy" % "_".join(map(str, k)), best[k][1])
