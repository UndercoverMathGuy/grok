"""Frontier of the decoupling violation: maximise R_Q = h / RHS(D_Q) subject to gamma/gamma_IF >= c (quadratic penalty),
p = 5, n = 2, b = 2, starting from the saved violators and from random starts.  Reports, for each c, the best R_Q found
and the gamma/gamma_IF of that network (exact margins recomputed in float64).
usage: python adv_frontier.py p n b m c_list restarts
"""
import sys, json
import numpy as np, torch
from netopt import *

p, n, b, m = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
cs = [float(v) for v in sys.argv[5].split(',')]; R = int(sys.argv[6])
nu = n + 1; s = nu / b; r = 1 / s
Gam, ystar = programme(p, s)
gIF = gamma_IF(p, n, b, Gam)
Bp_ = Bp(p, n); Cb = Bp_ ** (1 / s)
model = Full(p, n)
F = list(range(1, (p - 1) // 2 + 1))
Kt = torch.tensor(kappa_mat(p))


def quantities(U, W, tau=None):
    G = model.gaps(U, W); P = model.cost(U, W, b)
    Uh = torch.fft.fft(U.to(torch.complex128), dim=-1); Wh = torch.fft.fft(W.to(torch.complex128), dim=-1)
    A = torch.stack([2 * p ** (-nu) * torch.conj(Wh[:, j]) * torch.prod(Uh[:, :, j], dim=1) for j in F], 0)
    Q = ((A.abs() ** 2 + 1e-30) ** (r / 2)).sum(1)
    cov = Kt @ (Q ** s)
    if tau is None:
        h = G.min(); cm = cov.min()
    else:
        sc = G.detach().abs().mean(); h = -tau * sc * torch.logsumexp(-G.reshape(-1) / (tau * sc), 0)
        cc = cov.detach().mean(); cm = -tau * cc * torch.logsumexp(-cov / (tau * cc), 0)
    rhs = nu ** (1 - s) * (Cb * P / Q.sum()) ** s * cm
    return h / rhs, h / P ** s / gIF


starts = []
try:
    d = np.load(f'viol_p{p}_n{n}_b{int(b) if b == int(b) else b}.npz')
    for k in range(4):
        if f'U{k}' in d:
            starts.append((d[f'U{k}'], d[f'W{k}']))
except Exception:
    pass
rng = np.random.default_rng(3)
for k in range(R):
    U0, W0, h, P = optimise(model, rng.normal(size=(m, n, p)), rng.normal(size=(m, p)), b, stages=(0.3, 0.1, 0.03))
    starts.append((U0, W0))
res = {}
for c in cs:
    bestc = (-1, None)
    for (U0, W0) in starts:
        U = torch.tensor(U0 / np.abs(U0).max(), requires_grad=True); W = torch.tensor(W0 / np.abs(W0).max(), requires_grad=True)
        for tau in [0.03, 0.01, 3e-3, 1e-3]:
            o = torch.optim.LBFGS([U, W], lr=1, max_iter=300, line_search_fn='strong_wolfe')
            def closure():
                o.zero_grad()
                rr, g = quantities(U, W, tau)
                loss = -torch.log(torch.clamp(rr, min=1e-9)) + 200.0 * torch.relu(c - g) ** 2
                loss.backward(); return loss
            try:
                o.step(closure)
            except Exception:
                pass
        with torch.no_grad():
            rr, g = quantities(U, W)
        if float(g) >= c - 1e-4 and float(rr) > bestc[0]:
            bestc = (float(rr), float(g))
    res[c] = bestc
    print(f"c = {c:.3f}: best R_Q = {bestc[0]:.5f} with gamma/gamma_IF = {bestc[1]}", flush=True)
json.dump({str(k): v for k, v in res.items()}, open(f'adv_frontier_p{p}_n{n}_b{b}.json', 'w'))
