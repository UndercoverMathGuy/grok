# Cross-feature test: start from the two-feature IF network (programme weights y*) at p, n, b=2,
# perturb (optionally drop neurons), optimise the true unrestricted margin (general networks, all inputs).
import sys, math, numpy as np, torch
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..'))
from lbfgs import ngaps, ds
from slsqp import iff
from gamma import Gamma
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
p, n, drop, noise, seeds = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
b = 2.0; nu = n + 1; s = nu / b
G, y = Gamma(p, s, True); T = iff(p, n, b, G)
X, Y = ds(p, n)
a = np.arange(p); blocks = []
for j, yj in enumerate(y, start=1):
    if yj <= 1e-12: continue
    lam = (2 ** n * yj / nu) ** (1 / nu)
    for l in range(nu):
        al = math.pi * l / nu
        blocks.append(([lam * np.cos(2 * math.pi * j * a / p + al)] * n, lam * np.cos(2 * math.pi * j * a / p + n * al)))
m0 = len(blocks)
def pack(bl):
    U = np.array([[u for u in B[0]] for B in bl]); W = np.array([B[1] for B in bl])
    return torch.tensor(np.concatenate([U.reshape(-1), W.reshape(-1)]))
th0 = pack(blocks)
print("IF network: m=%d ratio %.6f" % (m0, ngaps(th0, p, n, b, m0, X, Y).min().item() / T))
best = -9
for sd in range(seeds):
    rng = np.random.default_rng(sd)
    keep = sorted(rng.choice(m0, m0 - drop, replace=False)); m = len(keep)
    th = pack([blocks[k] for k in keep]) + noise * torch.tensor(rng.standard_normal(m * (n * p + p)))
    th = th.clone().requires_grad_()
    for tau in (0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4):
        opt = torch.optim.LBFGS([th], lr=1, max_iter=500, line_search_fn="strong_wolfe", tolerance_grad=1e-15, tolerance_change=1e-18)
        def closure():
            opt.zero_grad(); g = ngaps(th, p, n, b, m, X, Y) / T
            loss = tau * torch.logsumexp(-g / tau, 0); loss.backward(); return loss
        try: opt.step(closure)
        except Exception: pass
    r = ngaps(th.detach(), p, n, b, m, X, Y).min().item() / T
    best = max(best, r)
    print("seed %d m=%d ratio %.6f" % (sd, m, r), flush=True)
print("BEST p=%d n=%d drop=%d noise=%.2f: %.6f" % (p, n, drop, noise, best))
