# Case B at p = 5, seeded properly. Phase 1 (torch Adam, soft-min, as in opus55/xblind_relax.py): pure nets with Nper
# atoms per frequency, maximise the X-free bound B_XB / P^{3/2}  -> lands in case B (B_XB up to ~1.115 x target).
# Phase 2 (SLSQP, unrestricted: mixing and DC allowed, width 9): maximise the TRUE score subject to B_XB >= (1+delta) gs.
import sys, math, numpy as np, torch
from scipy.optimize import minimize
sys.path.insert(0, '.')
_argv = sys.argv; sys.argv = [_argv[0], "9"]
from caseB import gap3, xblind_vals, gs, p
sys.argv = _argv
torch.set_default_dtype(torch.float64)
N = 9; delta = float(sys.argv[1]); seeds = int(sys.argv[2]); h = 2
x = torch.arange(p); i2 = pow(2, -1, p); Y = (x[:, None] + x[None, :]) % p
def tgap(E1, E2, W):
    Z = torch.einsum('ia,ib,ic->abc', E1, E2, W); corr = Z[x[:, None], x[None, :], Y]
    return torch.stack([corr - Z[x[:, None], x[None, :], (Y - D) % p] for D in range(p)], -1)
Ds = torch.arange(1, p)
S0 = torch.arange(p)
L1 = ((S0[:, None] + Ds[None, :] * i2) % p)                      # [s, D] row index for the line
A_ = torch.arange(p)
def txb(g):
    gD = g[:, :, 1:]                                                # [a, b, D-1]
    clean = gD.mean((0, 1))                                         # [D-1]
    t1 = gD.mean(1)[L1, Ds[None, :] - 1].mean(1)                    # type {1}: x1 = s + D/2, x2 uniform
    t2 = gD.mean(0)[L1, Ds[None, :] - 1].mean(1)                    # type {2}
    bidx = (L1[:, :, None] - A_[None, None, :]) % p                 # [s, D, a] -> b = s + D/2 - a
    t12 = gD[A_[None, None, :].expand_as(bidx), bidx, (Ds[None, :, None] - 1).expand_as(bidx)].mean((1, 2))
    return torch.cat([clean, t1, t2, t12])
res = []
for s in range(seeds):
    g0 = torch.Generator().manual_seed(100 + s); Nper = [2, 3][s % 2]
    par = torch.randn(h * Nper * 6, generator=g0).requires_grad_()
    ks = torch.arange(1, h + 1).repeat_interleave(Nper)[:, None]
    C = torch.cos(2 * math.pi * ks * x / p); Sn = torch.sin(2 * math.pi * ks * x / p)
    opt = torch.optim.Adam([par], lr=1e-2)
    for it in range(1500):
        a = par.view(h * Nper, 3, 2); E = [a[:, j, 0:1] * C + a[:, j, 1:2] * Sn for j in range(3)]
        P = sum((e ** 2).sum() for e in E); b = txb(tgap(*E)) / P ** 1.5 / gs
        tau = 0.3 * 0.01 ** (it / 1500); loss = tau * torch.logsumexp(-b / tau, 0)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        a = par.view(h * Nper, 3, 2); E = [a[:, j, 0:1] * C + a[:, j, 1:2] * Sn for j in range(3)]
        V = np.zeros((3, N, p)); V[:, :h * Nper] = np.stack([e.numpy() for e in E])
    v1 = V.ravel() / np.linalg.norm(V)
    bxb = xblind_vals(gap3(v1)).min() / gs; T1 = gap3(v1)[:, :, 1:].min() / gs
    if bxb < 1 + delta:
        print(f"seed {s} Nper={Nper}: phase 1 B_XB {bxb:.4f} (true T {T1:.3f}) < 1+delta, skip", flush=True); continue
    c2 = [dict(type='ineq', fun=lambda z: gap3(z[:-1])[:, :, 1:].ravel() - z[-1] * gs),
          dict(type='ineq', fun=lambda z: xblind_vals(gap3(z[:-1])) - (1 + delta) * gs),
          dict(type='eq', fun=lambda z: z[:-1] @ z[:-1] - 1)]
    v1p = v1 + 1e-3 * np.random.default_rng(s).normal(size=v1.shape)   # let the idle neurons / mixing directions move
    r2 = minimize(lambda z: -z[-1], np.r_[v1p, gap3(v1p)[:, :, 1:].min() / gs], constraints=c2, method='SLSQP', options=dict(maxiter=2000, ftol=1e-12))
    v2 = r2.x[:-1] / np.linalg.norm(r2.x[:-1]); g = gap3(v2); T = g[:, :, 1:].min() / gs; feas = xblind_vals(g).min() / gs
    res.append((T, feas, s)); np.save(f'caseB3_d{delta}_s{s}.npy', v2)
    print(f"seed {s} Nper={Nper}: phase 1 B_XB {bxb:.4f} (true T {T1:.3f}) | phase 2 true T {T:.4f}, B_XB {feas:.4f}", flush=True)
ok = [r for r in res if r[1] >= 1 + delta - 1e-6]
print("MAX true T in case B:", max(ok) if ok else None)
