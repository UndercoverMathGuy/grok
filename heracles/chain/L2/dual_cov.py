# Item (c) diagnostic: at the optimal heavy spectrum, how well can one spread the test over its binding offsets?
#   W_k(lam) = E_lam cov_k(D), lam a probability on the binding offsets (all give the same heavy value).
#   A_dual = min_lam max_{k not in support} W_k(lam)   (vs the worst-case constant 2 used in the proof).
import json, numpy as np
from scipy.optimize import linprog
from progopt import cover
for p in (59, 113):
    hb = json.load(open(f'heavy_{p}.json')); x = np.array(hb['x']); A = cover(p)
    f = A @ (x ** 1.5 / np.sqrt(3)); bind = np.nonzero(f <= f.min() * (1 + 1e-7))[0]
    off = np.nonzero(x <= 1e-7)[0]; B = A[np.ix_(bind, off)]          # rows: binding offsets, cols: candidate light freqs
    nb = len(bind)
    r = linprog(np.r_[np.zeros(nb), 1], A_ub=np.c_[B.T, -np.ones(len(off))], b_ub=np.zeros(len(off)),
                A_eq=[np.r_[np.ones(nb), 0]], b_eq=[1], bounds=[(0, None)] * nb + [(None, None)])
    single = B.max(0).max()
    print(f"p={p}: {nb} binding offsets {list(bind + 1)}; worst single-offset light coverage {single:.3f}; "
          f"best spread over binding offsets: max_k W_k = {r.fun:.3f}")
