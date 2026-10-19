"""Control for R10: the same adversarial search for violations of (D_Q), restricted to class C (every neuron supported
on one frequency plane, no DC).  At n = 2, b in [2,3] the Note's Theorem 'decoupling when few features are bought
cheaply' proves h <= nu^{1-s} min_delta sum_l Q_l^s kappa_l(delta) for every class-C network with <= 2 cheap features,
so at p = 5 (|F| = 2) no violation may appear.
usage: python adv_dq_classC.py p n b neurons_per_freq restarts
"""
import sys
import numpy as np, torch
from netopt import *

p, n, b, mper, R = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
nu = n + 1; s = nu / b; r = 1 / s
Gam, ystar = programme(p, s); gIF = gamma_IF(p, n, b, Gam); Cb = Bp(p, n) ** (1 / s)
model = Full(p, n); F = list(range(1, (p - 1) // 2 + 1)); Kt = torch.tensor(kappa_mat(p))
a = np.arange(p)
basis = {k: torch.tensor(np.stack([np.cos(2 * np.pi * k * a / p), np.sin(2 * np.pi * k * a / p)])) for k in F}
freq = [k for k in F for _ in range(mper)]
m = len(freq)


def build(cu, cw):
    U = torch.stack([torch.stack([cu[i, t] @ basis[freq[i]] for t in range(n)]) for i in range(m)])
    W = torch.stack([cw[i] @ basis[freq[i]] for i in range(m)])
    return U, W


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


rng = np.random.default_rng(2)
best = (-9, None)
for rep in range(R):
    cu = torch.tensor(rng.normal(size=(m, n, 2)), requires_grad=True); cw = torch.tensor(rng.normal(size=(m, 2)), requires_grad=True)
    for stage, tau in enumerate([0.1, 0.03, 0.01, 3e-3, 1e-3, 1e-3, 3e-4]):
        o = torch.optim.LBFGS([cu, cw], lr=1, max_iter=300, line_search_fn='strong_wolfe')
        def closure():
            o.zero_grad(); U, W = build(cu, cw); rr, g = quantities(U, W, tau)
            # first stages: raise gamma (to get h > 0), later stages: raise R_Q
            loss = -g if stage < 3 else (-torch.log(torch.clamp(rr, min=1e-9)) if rr > 0 else -rr)
            loss.backward(); return loss
        try:
            o.step(closure)
        except Exception:
            pass
    with torch.no_grad():
        U, W = build(cu, cw); rr, g = quantities(U, W)
    print(f"rep {rep}: class C  R_Q = {float(rr):.5f}  gamma/gamma_IF = {float(g):.5f}", flush=True)
    if float(g) > 0 and float(rr) > best[0]:
        best = (float(rr), float(g))
print(f"SUMMARY class C p={p} n={n} b={b}: max R_Q over positive-margin runs = {best[0]:.5f} (gamma/gamma_IF {best[1]})")
