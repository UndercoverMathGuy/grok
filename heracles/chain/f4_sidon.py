# f.4: one-sided depth (moment body of the rows) vs two-sided depth (symmetric hull = rows and their negatives).
# Lower bounds by the HOLES_aiming grid method (cyclic symmetry fixes theta_1 = 0); upper bounds = min grid radius.
import itertools, sys, numpy as np
from scipy.optimize import linprog
def depth(p, S, L, sym):
    K = len(S); x = np.arange(p)
    chi = np.exp(2j * np.pi * 2 * np.outer(S, x) / p)
    if sym:
        chi = np.hstack([chi, -chi])
    n = chi.shape[1]; Ab = np.vstack([chi.real, chi.imag, np.ones(n)]); tmin = 9
    for idx in itertools.product(range(L), repeat=K - 1):
        t = np.r_[0.0, 2 * np.pi * np.array(idx) / L]
        r = linprog(np.r_[np.zeros(n), -1], A_eq=np.column_stack([Ab, -np.r_[np.cos(t), np.sin(t), 0.0]]),
                    b_eq=np.r_[np.zeros(2 * K), 1], bounds=[(0, None)] * n + [(None, None)], method='highs')
        tmin = min(tmin, -r.fun)
    return np.cos(np.pi / L) * tmin, tmin
for p, S, L in [(11, [1, 2, 3, 4, 5], 8), (13, [1, 3, 4], 12), (13, [1, 2, 3, 4, 5, 6], 6), (59, [1, 6, 14, 16, 19], 6),
                (59, [1, 4, 12, 18, 25, 27], 6), (113, [1, 5, 12, 15, 38, 46], 6)]:
    one = depth(p, S, L, False); two = depth(p, S, L, True)
    print(f"p={p:3d} S={S} L={L}: one-sided in [{one[0]:.3f}, {one[1]:.3f}]   two-sided in [{two[0]:.3f}, {two[1]:.3f}]", flush=True)
