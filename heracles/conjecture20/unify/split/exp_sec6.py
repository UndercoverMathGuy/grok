"""E4: the compensated networks of Note Section 6, rebuilt.  p = 5 (or 7), slot-symmetric 1-D shared-sign networks
(inputs u_i(a) = eps_a exp(lam_i e_a), single output frequency 1), optimised in the 1-D field model (mechanism_scripts/
oned.py).  Then:
  (a) gamma of the base network (outputs at frequency 1) and of its split;
  (b) outputs released to all frequencies with the inputs FIXED (the Note's joint LP, here by smoothed-min ascent);
  (c) everything released.
For each network report gamma/gamma_IF, gamma(split)/gamma_IF and the energy fractions.
usage: python exp_sec6.py p n m trials [json]   (json: stored 1-D parameters, e.g. ../../mechanism_scripts/kkt/oned_p5_n6.json)
"""
import sys, json, math
import numpy as np, torch
sys.path.insert(0, '../../mechanism_scripts')
import oned
from netopt import *

p, n, m, trials = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
b = 2.0; nu = n + 1; s = nu / b; r = 1 / s
Gam, ystar = programme(p, s)
gIF = gamma_IF(p, n, b, Gam)
model = Sym(p, n)
F = list(range(1, (p - 1) // 2 + 1))
print(f"p={p} n={n} m={m}: Gamma={Gam:.6f} y*={np.round(ystar, 4)} (K*={int((ystar > 1e-9).sum())})", flush=True)


def build(x):
    lam, lv, av = x[:m], x[m:2 * m], x[2 * m:]
    lv = lv - lv.max()
    eps = np.array([1.0 if a < p / 2 else -1.0 for a in range(p)])
    e = np.array([2 * a if a < p / 2 else 2 * a - p for a in range(p)], float)
    rho = np.exp(2j * np.pi * np.arange(p) / p)
    U, W = [], []
    for i in range(m):
        u = eps * np.exp(lam[i] * e); w = np.real(np.exp(lv[i] + 1j * av[i]) * rho)
        nu_, nw = np.linalg.norm(u), np.linalg.norm(w)
        t = (nw * nu_ ** n) ** (1.0 / (n + 1))
        U.append(u / nu_ * t); W.append(w / nw * t)
    return np.array(U), np.array(W)


def gam(U, W):
    with torch.no_grad():
        G = model.gaps(torch.tensor(U), torch.tensor(W)); P = model.cost(torch.tensor(U), torch.tensor(W), b)
    return float(G.min()) / float(P) ** s / gIF


def report(tag, U, W):
    g = gam(U, W)
    Us, Ws, live = split_sym(U, W, p)
    gs = gam(Us, Ws)
    st = energy_stats(U, W, p, sym=True, n=n)
    at = own(U, W, p, n, sym=True)
    amp = {j: abs(at[j].sum()) for j in F}
    print(f"  {tag:28s} gamma/gIF={g:+.4f}  split={gs:+.4f}  dc={st['dc']:.3f} dead={st['dead']:.3f} "
          f"sec={st['secondary']:.2e}  |beta_j|/max: {[round(amp[j] / max(amp.values()), 4) for j in F]}", flush=True)
    return dict(tag=tag, gamma=g, gamma_split=gs, **{k: v for k, v in st.items() if k != 'm'})


rng = np.random.default_rng(5)
best = (-1e9, None)
if len(sys.argv) > 5:
    d = json.load(open(sys.argv[5])); assert d['p'] == p and d['n'] == n and d['width'] == m
    best = (d['numerical_ratio'], np.concatenate([d['lam'], d['logV'], d['argV']])); trials = 0
for tr in range(trials):
    curv = [0.5, 1.0, 2.0, 4.0, 0.25, 8.0][tr % 6] * (p / ((p - 1) * n / m)) / 3
    x0 = oned.init(m, p, n, curv, rng, 0.05 * (tr // 6))
    v, x = oned.optimise(x0, m, p, n, 2.0)
    if v > best[0]:
        best = (v, x)
print(f"best 1-D efficiency (field model) = {best[0]:.5f}", flush=True)
U, W = build(best[1])
out = [report('(a) base 1-D (output freq 1)', U, W)]
# (b) outputs released, inputs fixed
mask = torch.ones(U.shape, dtype=torch.bool)
W0 = W + 1e-3 * np.abs(W).max() * rng.normal(size=W.shape)
U2, W2, h, P = optimise(model, U, W0, b, fixed_mask=mask)
out.append(report('(b) outputs on all freqs', U2, W2))
# (c) everything released
U3, W3, h, P = optimise(model, U2, W2, b)
out.append(report('(c) all weights released', U3, W3))
json.dump(out, open(f'exp_sec6_p{p}_n{n}_m{m}.json', 'w'), indent=1)
