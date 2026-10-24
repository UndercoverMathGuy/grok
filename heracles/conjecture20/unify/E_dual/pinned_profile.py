"""Pinned-profile classes C = {theta : |beta|/sum|beta| = yhat_v}: is W_C <= D_C = pi B_p M(yhat)/||yhat||_r ?
Uses the certified (5,2,2) violators of SPLIT R10 (pi_{2,b} = 3^{1-s}, Note Thm q1n2).  Ratios are to gamma_IF."""
import numpy as np
from elib import G
d = np.load('../split/viol_p5_n2_b2.npz')
p, n, b = 5, 2, 2.0; s = 1.5; r = 1 / s
Gam, y, _ = G.programme(p, s); K = G.kappa_mat(p)[:2]
for k in range(3):
    net = G.Net(d['U%d' % k], d['W%d' % k], b)
    A = np.abs(net.clean()); yh = A / A.sum()
    Dc = (K @ yh).min() / ((yh ** r).sum() ** s) / Gam          # D_C / gamma_IF  (pi = 3^{1-s} cancels)
    # closed eta-neighbourhood of the profile: max of M/||.||_r over |yhat_1 - yv_1| <= eta
    out = []
    for eta in [0.0, 0.01, 0.02, 0.05]:
        ts = np.linspace(max(0, yh[0] - eta), min(1, yh[0] + eta), 20001)
        vals = [(K @ np.array([t, 1 - t])).min() / ((np.array([t, 1 - t]) ** r).sum() ** s) / Gam for t in ts]
        out.append('eta=%.2f: %.5f' % (eta, max(vals)))
    print('violator %d: gamma/gIF=%.6f profile=(%.4f, %.4f)  D_C/gIF=%.5f  |  ' % (k, net.gamma() / G.gamma_if(p, n, b, Gam), yh[0], yh[1], Dc) + '; '.join(out))
