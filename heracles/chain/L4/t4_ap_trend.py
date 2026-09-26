# Depth trend on progressions as length grows (p = 113 and 1009):
#  one-sided A*(AP_L) <= 1/L (PROVED, Fejer construction; checked here by evaluating that z),
#  two-sided A_s(AP_L) in [1/sqrt(2L), 2/sqrt(L)] (PROVED: Parseval / Rudin-Shapiro on a power-of-2 sub-progression),
#  search upper estimates of A_s, transfer c(A*, A_s) using A* <= 1/L as an optimistic input (so c here is an UPPER estimate
#  of what the transfer can certify), and (p = 113 only) the search estimate of the true flip-enlarged l2-ball depth.
import sys, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '..')
from transfer_flip import depth as transfer_depth
from t3_race import enlarged_search
from l4lib import two_sided_depth
rng = np.random.default_rng(4)
for p in (113, 1009):
    for L in (3, 4, 5, 6, 8, 10, 12, 16):
        S = np.arange(1, L + 1); y = np.arange(p)
        z = -2 * (1 - S / (L + 1))                                   # Fejer direction
        fz = np.real(z @ np.exp(-2j * np.pi * 2 * np.outer(S, y) / p))
        fejer = fz.max() / np.abs(z).sum()
        As = two_sided_depth(p, list(S), rng, starts=8)
        c, _ = transfer_depth(As, min(fejer, 1.0))
        es = enlarged_search(p, list(S), extra=6) if (p == 113 and L <= 10) else float('nan')
        print(f"p={p} AP L={L:2d}: one-sided <= {fejer:.3f} (1/L={1/L:.3f}) | two-sided <= {As:.3f} in-theory [{(2*L)**-0.5:.3f}, {2*L**-0.5:.3f}] | transfer c <= {c:.3f} | enlarged (search) <= {es:.3f}", flush=True)
