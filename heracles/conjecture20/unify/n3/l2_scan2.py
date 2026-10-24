"""Float scan of the L2 relaxation used by cert_l2.py: Pbar = R^2 + 2 h_a(min(R,t*)), Tbar = -6R^2 G(a,min(R,1))."""
import sys
import numpy as np
from ptinner import F
from kbound import G
def Pbar(a, R):
    al = np.arccos(a); ts = np.cos(np.pi/4 + al/2); t = min(R, ts)
    g = a*t + np.sqrt(max((1-a*a)*(1-t*t), 0)); return R*R + 2*(g*g - t*t)
for s in [float(x) for x in sys.argv[1:]]:
    r = 1/s; best = (-np.inf, None)
    for a in np.linspace(0.3, 1.0, 141):
        for R in np.linspace(0.005, 2.65, 266):
            v = (a - F(min(Pbar(a, R), 6.0), -6*R*R*G(a, min(R, 1.0)), r))/R
            if v > best[0]: best = (v, (a, R))
    a0, R0 = best[1]
    for it in range(3):
        d = 0.005/(3**it)
        for a in np.linspace(max(0.3, a0-5*d), min(1, a0+5*d), 21):
            for R in np.linspace(max(0.002, R0-5*d), R0+5*d, 21):
                v = (a - F(min(Pbar(a, R), 6.0), -6*R*R*G(a, min(R, 1.0)), r))/R
                if v > best[0]: best = (v, (a, R))
        a0, R0 = best[1]
    print(f"s={s}: K_L2 ~ {best[0]:.5f} at a={a0:.4f} R={R0:.4f}", flush=True)
