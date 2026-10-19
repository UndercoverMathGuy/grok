"""E1: random mixed networks and IF+mixed-noise starts, optimised for the normalised margin (full tables).
For every optimised theta we record gamma/gamma_IF, gamma(theta_split)/gamma_IF, gamma(theta_live)/gamma_IF,
energy fractions (DC / dead / live), shared neurons, features, cheap features, and
Delta = gamma(theta) - max(gamma_IF, gamma(theta_split)).
usage: python exp_random.py p n b widths seeds [init=rand|ifnoise] [eps]
"""
import sys, json, time, math
import numpy as np, torch
from netopt import *

p, n, b = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
widths = [int(v) for v in sys.argv[4].split(',')]
seeds = int(sys.argv[5])
init = sys.argv[6] if len(sys.argv) > 6 else 'rand'
eps = float(sys.argv[7]) if len(sys.argv) > 7 else 0.3
nu = n + 1; s = nu / b; r = 1 / s
Gam, ystar = programme(p, s)
gIF = gamma_IF(p, n, b, Gam)
K = kappa_mat(p)
model = Full(p, n)
F = list(range(1, (p - 1) // 2 + 1))
print(f"p={p} n={n} b={b} s={s:.3f}: Gamma_s={Gam:.6f} y*={np.round(ystar, 4)} K*={int((ystar > 1e-9).sum())}", flush=True)


def if_network(p, n, y):
    """IF optimum: one Prop-9 block per support frequency, scaled so that its clean amplitude is prop. to y_j."""
    Us, Ws = [], []
    a = np.arange(p)
    for j, yj in zip(F, y):
        if yj <= 1e-12:
            continue
        lam = (2 ** n * yj / nu) ** (1 / nu)
        for l in range(nu):
            al = np.pi * l / nu
            Us.append(np.array([lam * np.cos(2 * np.pi * j * a / p + al) for t in range(n)]))
            Ws.append(lam * np.cos(2 * np.pi * j * a / p + n * al))
    return np.array(Us), np.array(Ws)


def gam(U, W):
    with torch.no_grad():
        G = model.gaps(torch.tensor(U), torch.tensor(W)); P = model.cost(torch.tensor(U), torch.tensor(W), b)
    return float(G.min()), float(P)


def analyse(U, W):
    h, P = gam(U, W)
    Us, Ws, live = split_full(U, W, p)
    hs, Ps = gam(Us, Ws)
    Pl = float(((((Us[live] ** 2).sum((1, 2)) + (Ws[live] ** 2).sum(1)) ** (b / 2)).sum())) if live.any() else 0.0
    at = own(U, W, p, n)
    beta = {j: at[j].sum() for j in F}
    Q = {j: (np.abs(at[j]) ** r).sum() for j in F}
    amax = max(abs(beta[j]) for j in F)
    feats = [j for j in F if abs(beta[j]) > 1e-6 * amax]
    cheap = [j for j in F if abs(beta[j]) > nu ** (1 - s) * Q[j] ** s * (1 + 1e-6)]
    st = energy_stats(U, W, p)
    g = h / P ** s / gIF
    gs = hs / Ps ** s / gIF
    gl = h / Pl ** s / gIF if Pl > 0 else float('nan')
    return dict(gamma=g, gamma_split=gs, gamma_live_cost=gl, P_split_over_P=Ps / P, P_live_over_P=Pl / P,
                feats=feats, cheap=cheap, shares={j: float(abs(beta[j])) for j in F},
                Delta=g - max(1.0, gs), **st)


rng = np.random.default_rng(int(1000 * b) + 17 * p + n)
res = []
t0 = time.time()
U_if, W_if = if_network(p, n, ystar)
print("IF network check: gamma/gamma_IF =", round(gam(U_if, W_if)[0] / gam(U_if, W_if)[1] ** s / gIF, 6), flush=True)
for m in widths:
    for sd in range(seeds):
        if init == 'rand':
            U0 = rng.normal(size=(m, n, p)); W0 = rng.normal(size=(m, p))
        else:
            mm = max(m, len(U_if))
            U0 = np.zeros((mm, n, p)); W0 = np.zeros((mm, p))
            U0[:len(U_if)] = U_if; W0[:len(W_if)] = W_if
            sc = np.abs(U_if).max()
            U0 += eps * sc * rng.normal(size=U0.shape); W0 += eps * sc * rng.normal(size=W0.shape)
        U, W, h, P = optimise(model, U0, W0, b)
        a = analyse(U, W); a.update(m=m, seed=sd, init=init)
        res.append(a)
        print(f"m={m:2d} sd={sd} gamma={a['gamma']:+.4f} split={a['gamma_split']:+.4f} live-cost={a['gamma_live_cost']:+.4f} "
              f"dc={a['dc']:.3f} dead={a['dead']:.3f} sec={a.get('secondary', float('nan')):.1e} shared={a['shared_neurons']} feats={a['feats']} cheap={a['cheap']} "
              f"Delta={a['Delta']:+.4f} ({time.time() - t0:.0f}s)", flush=True)
best = max(res, key=lambda d: d['gamma'])
print(f"SUMMARY p={p} n={n} b={b} init={init}: best gamma/gamma_IF={best['gamma']:.5f}; max Delta={max(d['Delta'] for d in res):+.5f}; "
      f"runs with gamma>1: {sum(d['gamma'] > 1 + 1e-6 for d in res)}; runs with gamma > gamma_split + 1e-6 and gamma>0.9: "
      f"{sum((d['gamma'] > d['gamma_split'] + 1e-6) and d['gamma'] > 0.9 for d in res)}", flush=True)
json.dump(res, open(f"exp_random_p{p}_n{n}_b{b}_{init}.json", "w"), indent=1, default=float)
