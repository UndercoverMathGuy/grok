# Riesz-with-errors bound vs grid-LP bracket [lower, upper] of the true depth (HOLES_aiming method, fp64) on explicit sets.
import sys, itertools, numpy as np
from scipy.optimize import linprog
from riesz_bound import bounds
from fractions import Fraction as Fr
def depth(p, S, L, sym):
    K = len(S); x = np.arange(p); chi = np.exp(2j * np.pi * 2 * np.outer(S, x) / p)
    if sym: chi = np.hstack([chi, -chi])
    n = chi.shape[1]; Ab = np.vstack([chi.real, chi.imag, np.ones(n)]); tmin = 9
    for idx in itertools.product(range(L), repeat=K - 1):
        t = np.r_[0.0, 2 * np.pi * np.array(idx) / L]
        r = linprog(np.r_[np.zeros(n), -1], A_eq=np.column_stack([Ab, -np.r_[np.cos(t), np.sin(t), 0.0]]),
                    b_eq=np.r_[np.zeros(2 * K), 1], bounds=[(0, None)] * n + [(None, None)], method='highs')
        tmin = min(tmin, -r.fun)
    return np.cos(np.pi / L) * tmin, tmin
p = int(sys.argv[1]); H = [int(v) for v in sys.argv[2].split(',')]; L = int(sys.argv[3])
b1, b2 = bounds(p, H, [Fr(i, 200) for i in range(1, 101)])[:2]
o = depth(p, H, L, False); t = depth(p, H, L, True)
print(f"p={p} H={H}: one-sided Riesz {float(b1[0]):.3f} vs true in [{o[0]:.3f},{o[1]:.3f}] | two-sided Riesz {float(b2[0]):.3f} vs true in [{t[0]:.3f},{t[1]:.3f}]", flush=True)
