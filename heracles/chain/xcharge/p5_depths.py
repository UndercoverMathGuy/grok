# p = 5, S = {1,2}: one-sided depth A*, two-sided depth A_s (1-D K-mode), by full phase grid (no cyclic reduction:
# that reduction needs L <= p).  lower = cos(pi/L) * min over grid, upper = min over grid (explicit directions).
import sys, numpy as np
from scipy.optimize import linprog
p = 5; S = [1, 2]; x = np.arange(p)
chi = np.exp(2j * np.pi * 2 * np.outer(S, x) / p)
def grid_depth(sym, L):
    c = np.hstack([chi, -chi]) if sym else chi
    n = c.shape[1]; Ab = np.vstack([c.real, c.imag, np.ones(n)]); tmin = 9
    for i in range(L):
        for j in range(L):
            t = 2 * np.pi * np.array([i, j]) / L
            r = linprog(np.r_[np.zeros(n), -1], A_eq=np.column_stack([Ab, -np.r_[np.cos(t), np.sin(t), 0.0]]),
                        b_eq=np.r_[0, 0, 0, 0, 1], bounds=[(0, None)] * n + [(None, None)], method='highs')
            tmin = min(tmin, -r.fun)
    return np.cos(np.pi / L) * tmin, tmin
for L in (40, 200):
    print("L", L, "one-sided [lo, up]:", np.round(grid_depth(False, L), 5), " two-sided:", np.round(grid_depth(True, L), 5), flush=True)
