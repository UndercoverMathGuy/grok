"""PT-relaxation constant K(s) via the closed-form inner solver (floating point), coarse-then-fine grid."""
import sys
import numpy as np
from ptinner import F
from kbound import G
cur = sys.argv[1]; s = float(sys.argv[2]); r = 1/s
kP, kT = (4, 24) if cur == 'pt' else (3, 12)
def val(a, m):
    g = G(a, m); return (a - F(kP*m*m + 2*g*g, -kT*m*m*g, r))/m
best = (-np.inf, None)
for a in np.linspace(0.3, 1.0, 141):
    for m in np.linspace(0.005, 1.0, 200):
        v = val(a, m)
        if v > best[0]: best = (v, (a, m))
# refine
a0, m0 = best[1]
for it in range(3):
    da, dm = 0.005/(3**it), 0.005/(3**it)
    for a in np.linspace(max(0.3, a0-5*da), min(1, a0+5*da), 21):
        for m in np.linspace(max(0.001, m0-5*dm), min(1, m0+5*dm), 21):
            v = val(a, m)
            if v > best[0]: best = (v, (a, m))
    a0, m0 = best[1]
print(f"{cur} s={s}: K_PT ~ {best[0]:.5f} at a={best[1][0]:.4f} m={best[1][1]:.4f}", flush=True)
