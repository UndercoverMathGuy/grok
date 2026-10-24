"""Adversarial search for violations of the 'atom-cost decoupling' (D_Q) with the IF price (p >= 5):
     R_Q(theta) = h(theta) / [ nu^{1-s} (C_b P / sum_l Q_l)^s  min_delta sum_l Q_l^s kappa_l(delta) ]  <= 1 ?
At n = 2 the price of a feature is PROVED to equal nu^{1-s} (Note, Thm 'price of a feature at n=2'), so any network
with R_Q > 1 refutes decoupling-for-all-networks with the true price.  We maximise log R_Q (smoothed min) from random
starts and report the best ratio together with gamma/gamma_IF of the violating networks.
usage: python adv_dq.py p n b width restarts
"""
import sys, json
import numpy as np, torch
from netopt import *

p, n, b, m, R = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
nu = n + 1; s = nu / b; r = 1 / s
Gam, ystar = programme(p, s)
gIF = gamma_IF(p, n, b, Gam)
Bp_ = Bp(p, n); Cb = Bp_ ** (1 / s)
model = Full(p, n)
F = list(range(1, (p - 1) // 2 + 1))
Kt = torch.tensor(kappa_mat(p))


def atoms_t(U, W):
    Uh = torch.fft.fft(U.to(torch.complex128), dim=-1); Wh = torch.fft.fft(W.to(torch.complex128), dim=-1)
    out = []
    for j in F:
        out.append(2 * p ** (-nu) * torch.conj(Wh[:, j]) * torch.prod(Uh[:, :, j], dim=1))
    return torch.stack(out, 0)            # |F| x m


def logR(U, W, tau=None):
    G = model.gaps(U, W); P = model.cost(U, W, b)
    A = atoms_t(U, W)
    Q = ((A.abs() ** 2 + 1e-30) ** (r / 2)).sum(1)
    cov = Kt @ (Q ** s)
    if tau is None:
        h = G.min(); cm = cov.min()
    else:
        sc = G.detach().abs().mean()
        h = -tau * sc * torch.logsumexp(-G.reshape(-1) / (tau * sc), 0)
        cs = cov.detach().mean()
        cm = -tau * cs * torch.logsumexp(-cov / (tau * cs), 0)
    rhs = nu ** (1 - s) * (Cb * P / Q.sum()) ** s * cm
    return h / rhs, h / P ** s / gIF


rng = np.random.default_rng(1)
best = []; nets = []
for rep in range(R):
    U = torch.tensor(rng.normal(size=(m, n, p)), requires_grad=True); W = torch.tensor(rng.normal(size=(m, p)), requires_grad=True)
    # first get a positive margin network (normalised margin ascent), then push the ratio
    U0, W0, h, P = optimise(model, U.detach().numpy(), W.detach().numpy(), b, stages=(0.3, 0.1, 0.03, 0.01))
    U = torch.tensor(U0, requires_grad=True); W = torch.tensor(W0, requires_grad=True)
    if h <= 0:
        continue
    for tau in [0.1, 0.03, 0.01, 3e-3, 1e-3]:
        o = torch.optim.LBFGS([U, W], lr=1, max_iter=300, line_search_fn='strong_wolfe')
        def closure():
            o.zero_grad()
            rr, g = logR(U, W, tau)
            loss = -torch.log(torch.clamp(rr, min=1e-12)) if rr > 0 else -rr
            loss.backward(); return loss
        try:
            o.step(closure)
        except Exception:
            pass
    with torch.no_grad():
        rr, g = logR(U, W)
    best.append((float(rr), float(g)))
    nets.append((float(rr), float(g), U.detach().numpy().copy(), W.detach().numpy().copy()))
    print(f"rep {rep}: R_Q = {float(rr):.5f}   gamma/gamma_IF = {float(g):.5f}", flush=True)
best.sort(reverse=True)
print(f"SUMMARY p={p} n={n} b={b} m={m}: max R_Q = {best[0][0]:.5f} at gamma/gamma_IF = {best[0][1]:.5f}; "
      f"max R_Q among networks with gamma/gamma_IF >= 0.95: {max([x[0] for x in best if x[1] >= 0.95], default=float('nan')):.5f}")
json.dump(best, open(f'adv_dq_p{p}_n{n}_b{b}_m{m}.json', 'w'))
nets.sort(key=lambda t: -t[0])
np.savez(f'adv_dq_p{p}_n{n}_b{b}_m{m}_nets.npz', **{f'U{k}': t[2] for k, t in enumerate(nets[:4])}, **{f'W{k}': t[3] for k, t in enumerate(nets[:4])})
