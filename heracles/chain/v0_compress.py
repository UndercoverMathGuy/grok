# V0: width compression (astra ARGUMENT sec 4). Any net -> <= p^2 (p-1) neurons, same gaps, no larger cost.
import numpy as np
from common import *

def compress(p, net):
    cen = [(e1, e2, w - w.mean()) for e1, e2, w in net]                      # step a: centre readouts
    alpha, U = [], []
    for e1, e2, w in cen:
        a = np.linalg.norm(e1) * np.linalg.norm(e2) * np.linalg.norm(w)
        if a > 1e-14:
            alpha.append(a)
            U.append(np.einsum('a,b,c->abc', e1, e2, w).ravel() / a)
    alpha, U = np.array(alpha), np.array(U).T
    while True:                                                               # step c: walk to a vertex
        live = alpha > 1e-13
        sv = np.linalg.svd(U[:, live], compute_uv=True)
        if sv[1][-1] > 1e-10 * sv[1][0] and live.sum() <= U.shape[0]:
            break
        beta = np.zeros_like(alpha)
        beta[live] = sv[2][-1]                                                # null direction
        f = lambda t: (np.clip(alpha + t * beta, 0, None) ** (2 / 3)).sum()
        ends = []
        for sgn in (+1, -1):
            b = sgn * beta
            neg = b < -1e-15
            if neg.any():
                ends.append(sgn * (alpha[neg] / -b[neg]).min())
        t = min(ends, key=f)                                                  # concave on the segment: an endpoint is no worse
        alpha = np.clip(alpha + t * beta, 0, None)
        alpha[alpha < 1e-13] = 0
    new = []
    for a, u in zip(alpha, U.T):                                              # step d: rebalance as rank-one neurons
        if a <= 0:
            continue
        T = u.reshape(p, p, p)
        x, s, _ = np.linalg.svd(T.reshape(p, p * p))                          # T is rank one: recover factors
        e1 = x[:, 0]; rest = (s[0] * _[0]).reshape(p, p)
        y2, s2, v2 = np.linalg.svd(rest)
        e2 = y2[:, 0]; w = s2[0] * v2[0]
        n = [np.linalg.norm(e1), np.linalg.norm(e2), np.linalg.norm(w)]
        c = (a * np.prod(n) / np.prod(n)) ** (1 / 3)
        new.append((e1 / n[0] * c, e2 / n[1] * c, w / n[2] * c * np.sign(a)))
    return new

rng = np.random.default_rng(1)
for p, width in ((3, 40), (5, 140)):
    net = random_net(p, width, rng, pure=False)
    new = compress(p, net)
    g0, g1 = gaps(table(p, net)), gaps(table(p, new))
    print(f"p={p} width {width} -> {len(new)} (bound d={p*p*(p-1)}), max gap change {np.abs(g0 - g1)[:, :, 1:].max():.1e}, "
          f"cost {cost2(net):.4f} -> {cost2(new):.4f}, gamma {gamma(p, net):.3e} -> {gamma(p, new):.3e}")
