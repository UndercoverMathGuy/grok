"""Phase-blind payment curve from the decoupled twisted bound (float exploration):
   F_tw(x; s) = min over |B|,|C|,|D| in [0, x], cos(phi) in [-1, 1] of pay_tw2(1, |B|, |C|, |D|, cos phi, s)
compared with c1sharp (1 - c_1 x), the quadratic bound (1 - 3 K_2 x^2) and the trivial bound 3^{1-s} max(1, x).
Usage: python ftw_curve.py s
"""
import sys
import numpy as np
from scipy.optimize import minimize
from tw2 import pay_tw2

s = float(sys.argv[1])
c1 = 1 - 3 ** (1 - s); K2 = 2 * (s - 1) / (7 - 4 * s)
rng = np.random.default_rng(0)


def Ftw(x, nL=150):
    best = np.inf
    grid = np.linspace(0, x, 4)
    for b in grid:
        for c in grid:
            for d in grid:
                for cp in (-1.0, -0.5, 0.0, 0.5, 1.0):
                    v = pay_tw2(1.0, b, c, d, cp, s, nL=nL)
                    if v < best:
                        best, arg = v, (b, c, d, cp)
    # local refinement
    f = lambda z: pay_tw2(1.0, min(abs(z[0]), x), min(abs(z[1]), x), min(abs(z[2]), x), np.clip(z[3], -1, 1), s, nL=nL)
    res = minimize(f, np.array(arg), method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 400})
    return min(best, res.fun), res.x


for x in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2):
    v, z = Ftw(x)
    print(f"s={s} x={x:.2f}: F_tw={v:.4f}  c1sharp={1 - c1 * x:.4f}  quad={1 - 3 * K2 * x * x:.4f}  trivial={3 ** (1 - s) * max(1, x):.4f}   argmin={np.round(z, 3)}", flush=True)
