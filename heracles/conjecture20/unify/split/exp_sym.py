"""E3: slot-symmetric networks at p = 5, 7 and larger n (Section 6 of the Note: sign-vector inputs, outputs on several
frequencies), optimised for the normalised margin; compare gamma(theta) with gamma(theta_split) and gamma_IF.
Inits: '1d' = 1-D shared-sign inputs u_i(a) = eps_a exp(lam_i e_a) with random slopes, outputs on all frequencies;
       'rand' = random; 'ifn' = IF blocks (slot-symmetric, Prop 9) + mixed noise.
usage: python exp_sym.py p n b widths seeds init [eps]
"""
import sys, json, time
import numpy as np, torch
from netopt import *

p, n, b = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
widths = [int(v) for v in sys.argv[4].split(',')]
seeds = int(sys.argv[5]); init = sys.argv[6]; eps = float(sys.argv[7]) if len(sys.argv) > 7 else 0.3
nu = n + 1; s = nu / b; r = 1 / s
Gam, ystar = programme(p, s)
gIF = gamma_IF(p, n, b, Gam)
model = Sym(p, n)
F = list(range(1, (p - 1) // 2 + 1))
print(f"p={p} n={n} b={b} s={s:.3f}: Gamma={Gam:.6f} y*={np.round(ystar, 4)} K*={int((ystar > 1e-9).sum())} count classes={len(model.Nc)}", flush=True)


def gam(U, W):
    with torch.no_grad():
        G = model.gaps(torch.tensor(U), torch.tensor(W)); P = model.cost(torch.tensor(U), torch.tensor(W), b)
    return float(G.min()), float(P)


def if_sym(y):
    a = np.arange(p); Us, Ws = [], []
    for j, yj in zip(F, y):
        if yj <= 1e-12:
            continue
        lam = (2 ** n * yj / nu) ** (1 / nu)
        for l in range(nu):
            al = np.pi * l / nu
            Us.append(lam * np.cos(2 * np.pi * j * a / p + al)); Ws.append(lam * np.cos(2 * np.pi * j * a / p + n * al))
    return np.array(Us), np.array(Ws)


rng = np.random.default_rng(int(10 * b) + 31 * n + p)
U_if, W_if = if_sym(ystar)
h0, P0 = gam(U_if, W_if)
print("IF (slot-symmetric blocks) gamma/gamma_IF =", round(h0 / P0 ** s / gIF, 6), flush=True)
eps_a = np.array([1.0 if a < p / 2 else -1.0 for a in range(p)])
e_a = np.array([2 * a if a < p / 2 else 2 * a - p for a in range(p)], float)
res = []; t0 = time.time()
for m in widths:
    for sd in range(seeds):
        if init == '1d':
            lam = np.sort(rng.uniform(-1.5, 1.5, m))
            U0 = np.array([eps_a * np.exp(l * e_a) for l in lam]); U0 /= np.linalg.norm(U0, axis=1, keepdims=True)
            W0 = rng.normal(size=(m, p))
        elif init == 'rand':
            U0 = rng.normal(size=(m, p)); W0 = rng.normal(size=(m, p))
        else:
            mm = max(m, len(U_if)); U0 = np.zeros((mm, p)); W0 = np.zeros((mm, p))
            U0[:len(U_if)] = U_if; W0[:len(W_if)] = W_if
            sc = np.abs(U_if).max(); U0 += eps * sc * rng.normal(size=U0.shape); W0 += eps * sc * rng.normal(size=W0.shape)
        U, W, h, P = optimise(model, U0, W0, b)
        g = h / P ** s / gIF
        Us, Ws, live = split_sym(U, W, p)
        hs, Ps = gam(Us, Ws)
        gs = hs / Ps ** s / gIF
        st = energy_stats(U, W, p, sym=True, n=n)
        at = own(U, W, p, n, sym=True)
        beta = {j: at[j].sum() for j in F}; Q = {j: (np.abs(at[j]) ** r).sum() for j in F}
        amax = max(abs(beta[j]) for j in F) + 1e-300
        feats = [j for j in F if abs(beta[j]) > 1e-6 * amax]
        cheap = [j for j in F if abs(beta[j]) > nu ** (1 - s) * Q[j] ** s * (1 + 1e-6)]
        st.pop('m', None); d = dict(m=m, seed=sd, gamma=g, gamma_split=gs, feats=feats, cheap=cheap, **st)
        res.append(d)
        print(f"m={m:2d} sd={sd} gamma={g:+.4f} split={gs:+.4f} dc={st['dc']:.3f} dead={st['dead']:.3f} sec={st['secondary']:.1e} "
              f"feats={feats} cheap={cheap} ({time.time() - t0:.0f}s)", flush=True)
best = max(res, key=lambda d: d['gamma'])
print(f"SUMMARY p={p} n={n} b={b} init={init}: best gamma/gamma_IF={best['gamma']:.5f} (split {best['gamma_split']:.5f}); "
      f"runs with gamma>1+1e-6: {sum(d['gamma'] > 1 + 1e-6 for d in res)}; max gamma-max(1,split): "
      f"{max(d['gamma'] - max(1, d['gamma_split']) for d in res):+.5f}", flush=True)
json.dump(res, open(f"exp_sym_p{p}_n{n}_b{b}_{init}.json", "w"), indent=1, default=float)
