"""Entropy of the certified (5,2,2) decoupling violator (SPLIT R10) and of the IF optimum, at s = 3/2."""
import numpy as np
from elib import G, nu_costs, shares, shannon, renyi
d = np.load('../split/viol_p5_n2_b2.npz')
p, n, b = 5, 2, 2.0; s = 1.5; r = 1 / s
Gam, y, _ = G.programme(p, s); gIF = G.gamma_if(p, n, b, Gam)
for k in range(4):
    net = G.Net(d['U%d' % k], d['W%d' % k], b)
    a = nu_costs(net); w = shares(a, s); pp = a / a.sum()
    A, Q, aa = net.cheapness()
    print('net%d m=%d gamma/gIF=%.6f  H(w_s)=%.4f  H_r(p)=%.4f  K_r=%.3f  cost shares=%s  |beta|=%s  cheap=%s' %
          (k, net.m, net.gamma() / gIF, shannon(w), renyi(pp, r), np.exp(renyi(pp, r)), np.round(np.sort(w)[::-1], 3),
           np.round(A, 5), np.round(A / aa, 3)))
ifn = G.if_network(p, n, y, b)
a = nu_costs(ifn); w = shares(a, s)
print('IF optimum: gamma/gIF=%.6f H=%.4f (log3+log2=%.4f)' % (ifn.gamma() / gIF, shannon(w), np.log(6)))
# single IF block (one feature): gamma/gIF and entropy
ifn1 = G.if_network(p, n, np.array([1.0, 0.0]), b)
a = nu_costs(ifn1); w = shares(a, s)
print('one-feature IF block: gamma/gIF=%.6f H=%.4f' % (ifn1.gamma() / gIF, shannon(w)))
# least-entropy two-block IF allocation reaching level lam: scan y2/y1
K = G.kappa_mat(p)[:2]
for lam in [0.79, 0.80, 0.85, 0.9, 0.95]:
    best = None
    for t in np.linspace(0, 1, 200001):
        yy = np.array([1.0, t]); val = (K @ yy).min() / (yy ** r).sum() ** s
        if val >= lam * Gam:
            z = yy ** r / (yy ** r).sum(); H = np.log(3) + shannon(z)
            if best is None or H < best[0]: best = (H, t)
    print('  IF two-block, gamma>=%.2f gIF: min entropy %.4f at y2/y1=%.4f' % (lam, best[0], best[1]))
