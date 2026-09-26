# 3a.8(b): independent re-derivation of certified depth lower bounds (fp64, not interval).
# Polydisc of radius t0 cos(pi/L) sits inside the product of regular L-gons of radius t0, whose vertices are the phase grid;
# so depth >= cos(pi/L) * min over grid directions of the largest reachable radius.
import itertools, sys, numpy as np
from scipy.optimize import linprog
p = int(sys.argv[1]); S = [int(s) for s in sys.argv[2].split(',')]; L = int(sys.argv[3])
K = len(S); x = np.arange(p)
chi = np.exp(2j * np.pi * 2 * np.outer(S, x) / p)          # moment of w at mode 2k (either sign convention: set is symmetric under conj)
Aeq_base = np.vstack([chi.real, chi.imag, np.ones(p)])
tmin, arg = 9, None
for idx in itertools.product(range(L), repeat=K - 1):
    th = np.r_[0.0, 2 * np.pi * np.array(idx) / L]             # overall rotation is NOT a symmetry, so this fixes theta_1 = 0 only
    for th0 in [0.0]:                                         # cyclic symmetry (HOLES_aiming sec 2): theta_1 = 0 suffices
        t = th.copy(); t[0] = th0
        col = -np.r_[np.cos(t), np.sin(t), 0.0]
        r = linprog(np.r_[np.zeros(p), -1], A_eq=np.column_stack([Aeq_base, col]), b_eq=np.r_[np.zeros(2 * K), 1],
                    bounds=[(0, None)] * p + [(None, None)], method='highs')
        if -r.fun < tmin:
            tmin, arg = -r.fun, t
print(f"p={p} S={S} L={L}: min grid radius {tmin:.6f} -> depth >= {np.cos(np.pi / L) * tmin:.6f} (fp64); worst grid phases {np.round(arg / (2 * np.pi / L)).astype(int)}")
