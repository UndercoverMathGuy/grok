"""Local maximisation of the phase-blind semi-rigid bound over (tau, x1, x2) at fixed s (Nelder-Mead from several starts).
Usage: python sup72.py s ledger"""
import sys
import rig72, numpy as np
from scipy.optimize import minimize
s = float(sys.argv[1]); ledger = sys.argv[2] if len(sys.argv) > 2 else 'nopaid'
ms = {c: rig72.Model(c, ledger) for c in rig72.CASES}
def f(v):
    tau, x1, x2 = v
    if not (0 <= tau <= 1 and 0 <= x1 <= 1.75 and 0 <= x2 <= 1.75):
        return 0
    return -max(m.solve(s, tau, x1, x2, False, False)[2] for m in ms.values())
best = (0, None)
for v0 in [(0.532, 1.012, 0.66), (0.55, 1.0, 0.6), (0.5, 1.0, 0.8), (0.6, 1.0, 0.5), (0.45, 1.0, 1.0), (0.532, 1.0, 0.66)]:
    res = minimize(f, np.array(v0), method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-7, 'maxiter': 300})
    if -res.fun > best[0]:
        best = (-res.fun, res.x)
    print(v0, '->', -res.fun, res.x, flush=True)
print('SUP', best)
