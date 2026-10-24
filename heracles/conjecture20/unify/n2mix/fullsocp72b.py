"""Phase-aware (full test) channel-resolved bound at (7,2) on the worst phase-blind configuration family:
feature 1 = single atom (x_1 = 1), feature 2 = random 2-3 atom system, amplitude ratio tau ~ 0.54 (phase-blind worst).
Exact atom payments P_j = 3^{1-s} Q_j^s (best case for the method).  Reports the full-SOCP bound with ledgers
'nopaid' (L-pure networks) and 'pat' (general networks), and the phase-blind bound at the same (tau, x1, x2).
Usage: python fullsocp72b.py s ntrials"""
import sys
import numpy as np
from fullsocp72 import FullBound, feature_from_atoms
import rig72

s = float(sys.argv[1]); NT = int(sys.argv[2]) if len(sys.argv) > 2 else 6
r = 1 / s
rng = np.random.default_rng(11)
fbs = {lg: FullBound(s, lg) for lg in ('nopaid', 'pat')}
pbs = {lg: {c: rig72.Model(c, lg) for c in rig72.CASES} for lg in ('nopaid', 'pat')}
for trial in range(NT):
    xi1 = np.exp(1j * rng.uniform(0, 2 * np.pi, (1, 2)))
    k2 = int(rng.integers(2, 4))
    beta2 = rng.uniform(0.3, 1, k2) * np.exp(1j * rng.uniform(-0.5, 0.5, k2))
    xi2 = np.exp(1j * rng.uniform(0, 2 * np.pi, (k2, 2)))
    A1, b1, P1 = feature_from_atoms(np.array([1.0 + 0j]), xi1, s)
    A2u, b2u, P2u = feature_from_atoms(beta2, xi2, s)
    # scale feature 2 so that tau = |A2|/(|A1|+|A2|) = 0.54
    tau = 0.54
    sc = tau / (1 - tau) / abs(A2u)
    A2 = A2u * sc; b2 = [v * sc for v in b2u]; P2 = P2u * sc
    x2 = max(abs(v) for v in b2) / abs(A2)
    F = (P1 ** r + P2 ** r) ** s
    res = {}
    for lg in ('nopaid', 'pat'):
        res[lg] = fbs[lg]([A1, A2], [b1, b2], F)
        # phase-blind bound at the same (tau, x1=1, x2) but with the c1sharp payment curve
        res[lg + '_pb'] = max(m.solve(s, tau, 1.0, min(x2, 1.75), False, x2 > 1.75)[2] for m in pbs[lg].values())
    print(f"trial {trial}: k2={k2} x2={x2:.3f} P2/A2={P2/abs(A2):.3f}  full: nopaid {res['nopaid']:.4f} pat {res['pat']:.4f}"
          f"   phase-blind: nopaid {res['nopaid_pb']:.4f} pat {res['pat_pb']:.4f}", flush=True)
