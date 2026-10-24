"""Phase-aware channel-resolved bound at (7,2), b = 2 (s = 1.5): sample and locally maximise over atom systems.
Families: both features 1-3 atoms with random phases and amplitudes; amplitude ratio free.  Exact atom payments.
Ledger 'pat' (K-plan, general networks).  Usage: python fullsocp72c.py s nsamples nlocal"""
import sys
import numpy as np
from scipy.optimize import minimize
from fullsocp72 import FullBound, feature_from_atoms

s = float(sys.argv[1]); NS = int(sys.argv[2]); NL = int(sys.argv[3])
r = 1 / s
rng = np.random.default_rng(21)
fb = FullBound(s, 'pat')


def decode(v, k1, k2):
    i = 0
    def take(n):
        nonlocal i
        out = v[i:i + n]; i += n
        return out
    amp1 = np.abs(take(k1)) + 1e-9; ph1 = take(k1); xi1 = take(2 * k1).reshape(k1, 2)
    amp2 = np.abs(take(k2)) + 1e-9; ph2 = take(k2); xi2 = take(2 * k2).reshape(k2, 2)
    return (amp1 * np.exp(1j * ph1), np.exp(1j * xi1)), (amp2 * np.exp(1j * ph2), np.exp(1j * xi2))


def value(v, k1, k2):
    (b1, x1), (b2, x2) = decode(v, k1, k2)
    A1, bt1, P1 = feature_from_atoms(b1, x1, s); A2, bt2, P2 = feature_from_atoms(b2, x2, s)
    if abs(A1) < 1e-6 or abs(A2) < 1e-6:
        return 0.0
    F = (P1 ** r + P2 ** r) ** s
    return fb([A1, A2], [bt1, bt2], F)


best = (0, None)
vals = []
for t in range(NS):
    k1 = int(rng.integers(1, 4)); k2 = int(rng.integers(1, 4))
    v = np.concatenate([rng.uniform(0.3, 1, k1), rng.uniform(-0.4, 0.4, k1), rng.uniform(0, 6.3, 2 * k1),
                        rng.uniform(0.3, 1, k2) * rng.uniform(0.5, 2.0), rng.uniform(-0.4, 0.4, k2), rng.uniform(0, 6.3, 2 * k2)])
    val = value(v, k1, k2)
    vals.append(val)
    if val > best[0]:
        best = (val, (v, k1, k2))
print(f"s={s}: {NS} random samples: max {max(vals):.4f}, mean {np.mean(vals):.4f}", flush=True)
# local maximisation from the best few samples
v, k1, k2 = best[1]
res = minimize(lambda vv: -value(vv, k1, k2), v, method='Nelder-Mead', options={'maxiter': NL, 'xatol': 1e-3, 'fatol': 1e-4})
print(f"local max from best sample (k1={k1},k2={k2}): {-res.fun:.4f}", flush=True)
for rnd in range(3):
    res = minimize(lambda vv: -value(vv, k1, k2), res.x, method='Nelder-Mead', options={'maxiter': NL, 'xatol': 1e-4, 'fatol': 1e-5})
    print(f"  continued round {rnd}: {-res.fun:.5f}", flush=True)
np.save('out/fullsocp72c_best.npy', np.concatenate([[k1, k2], res.x]))
(b1, x1), (b2, x2) = decode(res.x, k1, k2)
A1, bt1, P1 = feature_from_atoms(b1, x1, s); A2, bt2, P2 = feature_from_atoms(b2, x2, s)
print("best config: |beta| atoms f1", np.round(np.abs(b1), 4), "f2", np.round(np.abs(b2), 4))
print("  A1", np.round(A1, 4), "|B,C,D|1", np.round(np.abs(bt1), 4), "P1/|A1|", round(P1 / abs(A1), 4))
print("  A2", np.round(A2, 4), "|B,C,D|2", np.round(np.abs(bt2), 4), "P2/|A2|", round(P2 / abs(A2), 4), "tau", round(abs(A2) / (abs(A1) + abs(A2)), 4))
