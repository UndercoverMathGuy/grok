"""z12_check.py -- Z_12, n = 3 (nu = 4), q = 2: (1) the (H)-ceiling is attained by ELEGANT blocks (f3, f5) + one sign neuron
(amplitudes optimised) -- validates the abelian pricing; (2) an unrestricted ascent net exceeds it (saved, independently re-verified);
(3) where its junk lives (sign-consistent vs inconsistent triples, per real irrep)."""
import sys, itertools, json
from pathlib import Path
import numpy as np, torch
from scipy.optimize import minimize
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gt_core as gc
from junk_nu4 import ceil
from verify_fourier import row_products
torch.set_default_dtype(torch.float64)
G = gc.groups_mod.abelian((12,), "z12"); N = 12; n = 3; nu = 4
ce = ceil(G, nu, 2.0); gs = ce["school"][0]; y = row_products(G, n)
x = np.arange(N)
def gam(E, W):
    m = W.shape[0]; h = np.ones((m, 1))
    for j in range(n): h = (h[:, :, None] * E[j][:, None, :]).reshape(m, -1)
    Z = h.T @ W; zz = Z.copy(); zy = zz[np.arange(len(y)), y]; zz[np.arange(len(y)), y] = -np.inf
    return float((zy - zz.max(1)).min()) / float((E ** 2).sum() + (W ** 2).sum()) ** (nu / 2)
def H_net(rs, rsg, ks=(3, 5)):
    Es, Ws = [], []
    for k, r in zip(ks, rs):
        th = 2 * np.pi * k * x / N
        for l in range(n + 1):
            ph = np.pi * l / (n + 1)
            Es.append([r * np.cos(th + ph)] * n); Ws.append(r * np.cos(th + n * ph))
    s = (-1.0) ** x; Es.append([rsg * s] * n); Ws.append(rsg * s)
    return np.array(Es).transpose(1, 0, 2), np.array(Ws)
best = None
for st in range(8):
    x0 = np.random.default_rng(st).normal(0, 0.3, 3)
    r = minimize(lambda v: -gam(*H_net(np.exp(v[:2]), np.exp(v[2]))), x0, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-9, fatol=1e-16))
    if best is None or r.fun < best.fun: best = r
gH = -best.fun
print(f"(H)-ceiling {gs:.8e} on {ce['school'][1]};  ELEGANT-block + sign construction, amplitudes optimised: {gH:.8e}  ratio {gH / gs:.8f}")
# ascent
yt = torch.from_numpy(y); bestA = (-1, None)
for seed in (0, 3, 5, 7):
    g = torch.Generator().manual_seed(seed); m = 24
    E = (torch.randn(n, m, N, generator=g) / N ** 0.5).requires_grad_(True); W = (torch.randn(m, N, generator=g) / N ** 0.5).requires_grad_(True)
    opt = torch.optim.Adam([E, W], lr=0.01); steps = 6000
    for it in range(steps):
        h = torch.ones(m, 1)
        for j in range(n): h = (h[:, :, None] * E[j][:, None, :]).reshape(m, -1)
        Z = h.T @ W; P2 = (E ** 2).sum() + (W ** 2).sum(); Zn = Z / P2 ** 2
        gaps = Zn.gather(1, yt[:, None]) - Zn; gaps = gaps + torch.zeros_like(gaps).scatter_(1, yt[:, None], 1e6)
        mn = float(gaps.min().detach())
        if mn > bestA[0]: bestA = (mn, (E.detach().numpy().copy(), W.detach().numpy().copy()), seed)
        tau = max(abs(mn), 1e-12) * (0.3 * (1 - it / steps) + 0.01)
        soft = -tau * torch.logsumexp(-gaps.reshape(-1) / tau, 0)
        opt.zero_grad(); (-soft).backward(); opt.step()
        if it == steps // 2:
            for gp in opt.param_groups: gp["lr"] = 0.003
    print(f"   ascent seed {seed}: running best gamma/ceiling {bestA[0] / gs:.5f}", flush=True)
val, (E, W), seed = bestA
np.savez(HERE / "out" / "counterexample_z12_nu4.npz", E=E, W=W)
worst = np.inf
for a in itertools.product(range(N), repeat=n):
    ct = sum(a) % N
    logit = np.array([sum(W[i, c] * E[0, i, a[0]] * E[1, i, a[1]] * E[2, i, a[2]] for i in range(W.shape[0])) for c in range(N)])
    worst = min(worst, min(logit[ct] - logit[c] for c in range(N) if c != ct))
gA = worst / float((E ** 2).sum() + (W ** 2).sum()) ** 2
print(f"saved ascent net (seed {seed}): independently recomputed gamma {gA:.8e} = {gA / gs:.5f} x (H)-ceiling")
# junk location
sg = (-1.0) ** x
aa = np.array(list(itertools.product(range(N), repeat=n))); P = aa.sum(1) % N
cons = (sg[P][:, None] == sg[None, :])
for r_, nm in enumerate(G.irrep_names):
    if nm == "trivial": continue
    Pr = G.PROJ[r_]; Ep = np.einsum("jix,xy->jiy", E, Pr.T); Wp = W @ Pr.T
    h = np.ones((W.shape[0], 1))
    for j in range(n): h = (h[:, :, None] * Ep[j][:, None, :]).reshape(W.shape[0], -1)
    z = h.T @ Wp; D = (P[:, None] - np.arange(N)[None, :]) % N
    clean = np.zeros_like(z)
    for d in range(N): clean[D == d] = z[D == d].mean()
    J = z - clean; tot = np.linalg.norm(z) + 1e-300
    if tot > 1e-8 * np.abs(Z := z).max() + 1e-300 and np.linalg.norm(z) > 1e-6:
        print(f"   {nm:6s}: |z_pi| share {np.linalg.norm(z):.3e}; junk on consistent {np.linalg.norm(J[cons]) / tot:.4f}, inconsistent {np.linalg.norm(J[~cons]) / tot:.4f}")
json.dump(dict(ceiling=gs, H_construction=gH, ascent=gA, ratio=gA / gs), open(HERE / "out" / "z12_check.json", "w"), indent=1)
