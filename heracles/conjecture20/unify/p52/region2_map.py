"""Map Region II = {own data with best provable payment F = max(P_lb, decoupled twisted) < thr = kappa_a/Gamma} at s=1.5.
Grid over (m1 >= m2, m12, cos phi_pay)."""
import numpy as np
from p52lib import *
from tw2 import pay_tw2

s = 1.5; G = Gamma(s); c1 = 1 - 3 ** (1 - s); thr = KA / G
ms = np.linspace(0, 1.36, 35)
cps = np.linspace(-1, 1, 21)
cnt = 0; tot = 0; ext = {'m1': [9, -9], 'm2': [9, -9], 'm12': [9, -9], 'cos': [9, -9]}
minF = 9
for i, m1 in enumerate(ms):
    for m2 in ms[:i + 1]:
        for m12 in ms:
            M = max(m1, m2, m12); RT = np.sqrt(m1 ** 2 + m2 ** 2 + m12 ** 2)
            plb = max(1 - c1 * M, 1 - RT / (2 * np.sqrt(3)), 3 ** (1 - s) * max(1, M))
            if plb >= thr:
                tot += len(cps); continue
            for cp_ in cps:
                tot += 1
                F = max(plb, pay_tw2(1.0, m1, m2, m12, cp_, s, nL=120))
                minF = min(minF, F)
                if F < thr:
                    cnt += 1
                    for k, v in zip(['m1', 'm2', 'm12', 'cos'], [m1, m2, m12, cp_]):
                        ext[k][0] = min(ext[k][0], v); ext[k][1] = max(ext[k][1], v)
print(f"Region II grid points: {cnt}/{tot}; min F = {minF:.4f}; extents: {ext}")
