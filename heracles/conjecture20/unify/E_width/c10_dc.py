# W8 with DC inputs (single-frequency + DC, circular at p >= 5); and the atom cancellation 1-|A|/L of the p=3 winners.
import sys, numpy as np, itertools
from elib import *
from c8_aligned import analyse
rng = np.random.default_rng(10)
worst_cone = np.inf; worst_final = -np.inf; cnt = 0
for (p, n) in [(5, 2), (7, 2), (5, 3), (7, 3)]:
    nu = n + 1; a = np.arange(p)
    for trial in range(60):
        m = nu + rng.integers(0, 3)
        base = np.array([np.pi * (l % nu) / nu for l in range(m)])
        ph = np.repeat(base[:, None], n, axis=1) + rng.normal(scale=0.15, size=(m, n))
        amp = np.exp(rng.normal(scale=0.2, size=m))
        dc = rng.normal(scale=[0.0, 0.1, 0.4][trial % 3], size=(m, n))
        U = np.zeros((m, n, p)); W = np.zeros((m, p))
        for i in range(m):
            for t in range(n):
                U[i, t] = amp[i] * np.cos(2 * np.pi * a / p + ph[i, t]) + dc[i, t]
            W[i] = amp[i] * np.cos(2 * np.pi * a / p + ph[i].sum())
        net = Net(U, W)
        if net.margin() <= 0: continue
        for s in [1.0, nu / 2]:
            ratio, bound, coneL2, c, A, L, kp = analyse(net, p, n, s)
            worst_cone = min(worst_cone, coneL2); worst_final = max(worst_final, ratio - bound, ratio - 1); cnt += 1
print(f"single-frequency + DC, aligned: {cnt} cases; min cone-L2 slack {worst_cone:.3e} (>=0); max of (ratio-bound, ratio-1) {worst_final:.3e} (<=0)")
sys.path.insert(0, '/home/user/grok/heracles/conjecture20')
import exact_certificate as ec
for n in [11, 13]:
    d = ec.NETWORKS[n]; p = 3; nu = n + 1
    U1 = np.array([[float(x) for x in row] for row in d['u']]); W = np.array([[float(x) for x in row] for row in d['w']])
    Uh = np.fft.fft(U1, axis=1); Wh = np.fft.fft(W, axis=1)
    beta = 2 * p ** (-nu) * np.conj(Wh[:, 1]) * Uh[:, 1] ** n
    print(f"p=3 certificate n={n}: atom cancellation 1-|A|/L = {1 - abs(beta.sum()) / np.abs(beta).sum():.4f}; atom phases (deg) = {np.round(np.degrees(np.angle(beta)), 1)}")
