# For IF block unions built on EVERY vertex v of the coverage polyhedron (n = 2, so pi = 3^{1-s}), where is the
# RD slack S_v(s) = log Gamma_s - phi_v(s) smallest?  (phi_v(s) = -s log sum v^{1/s}; log Gamma_s = max_v phi_v.)
import numpy as np, math
from sclib import *
for p in [11, 13, 17]:
    ss = np.round(np.linspace(1.0, 1.5, 101), 4)
    _, _, best = ER.programme_vertices(p, 1.2)
    V = [v for _, v in best]
    Phi = np.array([[-s * math.log((v[v > 0] ** (1 / s)).sum()) for s in ss] for v in V])
    env = Phi.max(0); win = Phi.argmax(0)
    kinks = [ss[i] for i in range(1, len(ss)) if win[i] != win[i - 1]]
    cnt = dict(at1=0, at15=0, kink=0, interior=0)
    for k in range(len(V)):
        S = env - Phi[k]; i = int(np.argmin(S))
        if S.min() < 1e-12 and (S < 1e-12).sum() > 1: i = int(np.nonzero(S < 1e-12)[0][-1])
        if i == 0: cnt['at1'] += 1
        elif i == len(ss) - 1: cnt['at15'] += 1
        elif min(abs(ss[i] - kk) for kk in kinks) <= 0.006 if kinks else False: cnt['kink'] += 1
        else: cnt['interior'] += 1
    print('p=%d: %d vertices; envelope kinks at %s; argmin_s S_v: %s' % (p, len(V), kinks, cnt))
