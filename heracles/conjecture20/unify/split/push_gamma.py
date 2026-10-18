"""Falsification probe: start from mixing-driven decoupling violators (saved nets), push them up the frontier
(maximise R_Q subject to gamma >= c), then maximise gamma itself.  If mixing could beat the IF value, this is where
it should show.  Reports gamma/gamma_IF, gamma(split)/gamma_IF and the secondary-energy fraction at each stage.
usage: python push_gamma.py p n b nets.npz c
"""
import sys
import numpy as np, torch
from netopt import *
import lib

p, n, b = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]); f = sys.argv[4]; c = float(sys.argv[5])
nu = n + 1; s = nu / b; r = 1 / s
Gam, ystar = programme(p, s); gIF = gamma_IF(p, n, b, Gam)
Cb = Bp(p, n) ** (1 / s)
model = Full(p, n); F = list(range(1, (p - 1) // 2 + 1)); Kt = torch.tensor(kappa_mat(p))


def quantities(U, W, tau=None):
    G = model.gaps(U, W); P = model.cost(U, W, b)
    Uh = torch.fft.fft(U.to(torch.complex128), dim=-1); Wh = torch.fft.fft(W.to(torch.complex128), dim=-1)
    A = torch.stack([2 * p ** (-nu) * torch.conj(Wh[:, j]) * torch.prod(Uh[:, :, j], dim=1) for j in F], 0)
    Q = ((A.abs() ** 2 + 1e-30) ** (r / 2)).sum(1); cov = Kt @ (Q ** s)
    if tau is None:
        h = G.min(); cm = cov.min()
    else:
        sc = G.detach().abs().mean(); h = -tau * sc * torch.logsumexp(-G.reshape(-1) / (tau * sc), 0)
        cc = cov.detach().mean(); cm = -tau * cc * torch.logsumexp(-cov / (tau * cc), 0)
    return h / (nu ** (1 - s) * (Cb * P / Q.sum()) ** s * cm), h / P ** s / gIF


def stats(U, W):
    Us, Ws = lib.split(U, W)
    gs = lib.margin(Us, Ws) / lib.cost(Us, Ws, b) ** s / gIF
    st = energy_stats(U, W, p)
    return gs, st['secondary'], st['dc']


d = np.load(f)
for k in range(4):
    if f'U{k}' not in d:
        break
    U = torch.tensor(d[f'U{k}'] / np.abs(d[f'U{k}']).max(), requires_grad=True)
    W = torch.tensor(d[f'W{k}'] / np.abs(d[f'W{k}']).max(), requires_grad=True)
    for tau in [0.03, 0.01, 3e-3, 1e-3]:
        o = torch.optim.LBFGS([U, W], lr=1, max_iter=300, line_search_fn='strong_wolfe')
        def closure():
            o.zero_grad(); rr, g = quantities(U, W, tau)
            loss = -torch.log(torch.clamp(rr, min=1e-9)) + 200.0 * torch.relu(c - g) ** 2
            loss.backward(); return loss
        try:
            o.step(closure)
        except Exception:
            pass
    with torch.no_grad():
        rr, g = quantities(U, W)
    gs, sec, dc = stats(U.detach().numpy(), W.detach().numpy())
    print(f"net {k}: after frontier push  R_Q={float(rr):.5f} gamma={float(g):.5f} split={gs:.5f} secondary={sec:.3e}", flush=True)
    U2, W2, h, P = optimise(model, U.detach().numpy(), W.detach().numpy(), b)
    g2 = h / P ** s / gIF
    gs2, sec2, dc2 = stats(U2, W2)
    with torch.no_grad():
        rr2, _ = quantities(torch.tensor(U2), torch.tensor(W2))
    print(f"        after gamma ascent    R_Q={float(rr2):.5f} gamma={g2:.5f} split={gs2:.5f} secondary={sec2:.3e} dc={dc2:.2e}", flush=True)
