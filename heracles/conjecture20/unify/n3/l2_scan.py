"""L2-currency relaxation (chain (0,1,1,2)):  with R = ||(beta_T)_T||_2 / L,
  Pbar = R^2 + delta2(a,R),  delta2 = max_{u^2+w^2<=R^2} [h(u)+h(w)],  h(t) = G(a,t)^2 - t^2,
  Tbar = -6 R^2 Gmax(a,R),  Gmax = max_{t<=R} G(a,t),
  bonus/L <= a - F_r(min(Pbar,6), Tbar).   K_L2(s) = sup (a - F)/R   (floating point)."""
import sys
import numpy as np
from ptinner import F
from kbound import G
def h(a, t): return G(a, t)**2 - t*t
def delta2(a, R):
    # max over the DISK u^2+w^2 <= R^2, u,w in [0,1]
    best = 0.0
    ts = np.linspace(0, min(R, 1.0), 61)
    for u in ts:
        wmax = min(np.sqrt(max(R*R - u*u, 0.0)), 1.0)
        for w in np.linspace(0, wmax, 31):
            best = max(best, h(a, u) + h(a, w))
    return best
for s in [float(x) for x in sys.argv[1:]]:
    r = 1/s
    best = (-np.inf, None)
    for a in np.linspace(0.3, 1.0, 141):
        for R in np.linspace(0.01, 2.6, 260):
            P = min(R*R + delta2(a, R), 6.0)
            Gm = G(a, min(R, 1.0))
            v = (a - F(P, -6*R*R*Gm, r))/R
            if v > best[0]: best = (v, (a, R, P))
    print(f"s={s}: K_L2 ~ {best[0]:.5f} at (a,R,P)={np.round(best[1],4)}", flush=True)
