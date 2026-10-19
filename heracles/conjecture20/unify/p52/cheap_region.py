"""Where can feature 1 be 'very cheap'?  The trivial bound h <= kappa_a Re A gives ratio <= kappa_a |A| / (Gamma P_1),
so only own data with P_1 < (kappa_a/Gamma)|A| = 0.7818|A| (s=1.5) need detection.  Map the set where the best provable
payment F = max(P_lb, twisted) is below that threshold, in torus invariants (m1, m2, m12, phi), |A| = 1, A real.
"""
import sys
import numpy as np
from p52lib import *
from twisted_pay import twisted_lb

s = 1.5; G = Gamma(s); c1 = 1 - 3 ** (1 - s); thr = KA / G
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
Nsamp = int(sys.argv[2]) if len(sys.argv) > 2 else 400
rows = []
for k in range(Nsamp):
    m = rng.uniform(0, 1.75, 3)
    phi = rng.uniform(-np.pi, np.pi)
    B, C, D = m[0], m[1], m[2] * np.exp(1j * phi)
    M = m.max(); RT = np.sqrt((m ** 2).sum())
    plb = max(1 - c1 * M, 1 - RT / (2 * np.sqrt(3)), 3 ** (1 - s) * max(1, M))
    tw = twisted_lb(1.0, B, C, D, s, tries=8, seed=k)
    F = max(plb, tw)
    rows.append((m[0], m[1], m[2], phi, plb, tw, F))
    if F < thr:
        print(f"CHEAP: m=({m[0]:.3f},{m[1]:.3f},{m[2]:.3f}) phi={phi:.3f} P_lb={plb:.4f} tw={tw:.4f}", flush=True)
rows = np.array(rows)
np.save(f"cheap_region_seed{sys.argv[1] if len(sys.argv)>1 else 0}.npy", rows)
print("fraction cheap:", np.mean(rows[:, 6] < thr), " min F:", rows[:, 6].min())
