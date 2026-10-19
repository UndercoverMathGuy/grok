# N5 (p=3): the certified p=3 networks (heracles/conjecture20/oned_certified/p3_n*.json, b = 2).
#   F = {1}: every network is single-feature, so maximisers of gamma = optimal single features, H^dec = H^(1)_min,
#   (E), (FE) and the weighted identity (H1) are all vacuous.  We record H, nu*E, output DC, gamma/gamma_IF.
import json, glob, os
import numpy as np
from elib import Net, Bp, entropy, Evec, freq_proj
base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'oned_certified')
for fn in sorted(glob.glob(os.path.join(base, 'p3_n*.json')), key=lambda f: int(f.split('_n')[1].split('.')[0])):
    d = json.load(open(fn)); p, n = d['p'], d['n']; nu = n + 1
    lam, lv, av = map(np.array, (d['lam'], d['logV'], d['argV'])); lv = lv - lv.max()
    eps = np.array([1.0 if a < p / 2 else -1.0 for a in range(p)])
    e = np.array([2 * a if a < p / 2 else 2 * a - p for a in range(p)], float)
    rho = np.exp(2j * np.pi * np.arange(p) / p)
    U, W = [], []
    for i in range(len(lam)):
        u = eps * np.exp(lam[i] * e); w = np.real(np.exp(lv[i] + 1j * av[i]) * rho)
        t = (np.linalg.norm(w) * np.linalg.norm(u) ** n) ** (1 / nu)
        U.append([u / np.linalg.norm(u) * t] * n); W.append(w / np.linalg.norm(w) * t)
    U = np.array(U); W = np.array(W)
    s = nu / 2
    # margin via slot-symmetric count enumeration (p=3): f depends on counts only
    from itertools import product
    best = None
    def comps(n, k):
        if k == 1: yield (n,); return
        for v in range(n + 1):
            for r in comps(n - v, k - 1): yield (v,) + r
    h = None
    for N in comps(n, p):
        y = sum(a * N[a] for a in range(p)) % p
        z = np.array([np.prod([U[i, 0, a] ** N[a] for a in range(p)]) for i in range(len(lam))])
        f = W.T @ z; g = min(f[y] - f[c] for c in range(p) if c != y); h = g if h is None else min(h, g)
    P = np.sum(((U ** 2).sum((1, 2)) + (W ** 2).sum(1)))
    gam = h / P ** s; gIF = nu ** (1 - s) * Bp(p, n) * (1 - np.cos(2 * np.pi / p))
    sq = (U ** 2).sum((1, 2)) + (W ** 2).sum(1); w = sq / sq.sum(); H = -(w * np.log(w)).sum()
    dc = np.sum((W @ freq_proj(p, 0)) ** 2) / np.sum(W ** 2)
    nuE = nu * np.sum(w * ((W @ freq_proj(p, 1)) ** 2).sum(1) / sq)
    print(f"p=3 n={n}: gamma/gamma_IF={gam/gIF:.6f}; m={len(lam)}; H={H:.5f} (log m={np.log(len(lam)):.5f}, log nu={np.log(nu):.5f}); "
          f"nu*E_1={nuE:.6f}; output-DC fraction {dc:.1e}; H(nuE)=0 so (FE) reads H >= H^(1)_min: vacuous for an optimal single feature")
