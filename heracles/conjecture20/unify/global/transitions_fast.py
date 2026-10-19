# Transition points of K*(s,p): per support size K the best programme value (exhaustive over supports containing 1,
# which suffices by dilation symmetry), then bisection on the s-grid where the argmax over K changes.
import sys, itertools, numpy as np
from scipy.optimize import linprog
from lib import kappa_mat


def best_support_value(p, s, S, starts=4):
    F = (p - 1) // 2; r = 1 / s; K = kappa_mat(p)[:F][:, [j - 1 for j in S]]; best = -1
    for t in range(starts):
        w = np.ones(len(S)) if t == 0 else np.random.default_rng(t).uniform(0.3, 1.7, len(S))
        res = None
        for it in range(200):
            res = linprog(w, A_ub=-K, b_ub=-np.ones(F), bounds=(0, None), method='highs')
            if res.status != 0: break
            v = res.x; wn = (v + 1e-12) ** (r - 1)
            if np.allclose(wn, w, rtol=1e-10): break
            w = wn
        if res is None or res.status != 0 or np.any(v < 1e-9): continue
        best = max(best, np.sum(v ** r) ** (-s))
    return best


def table(p, s):
    F = (p - 1) // 2; out = {}
    for k in range(1, F + 1):
        out[k] = max(best_support_value(p, s, (1,) + S) for S in itertools.combinations(range(2, F + 1), k - 1))
    return out


def kset(p, s):
    t = table(p, s); G = max(t.values())
    return [k for k, v in t.items() if v >= G * (1 - 1e-10)], t


if __name__ == "__main__":
    p = int(sys.argv[1]); grid = np.arange(float(sys.argv[2]), float(sys.argv[3]) + 1e-9, float(sys.argv[4]))
    prev = None
    for s in grid:
        ks, t = kset(p, s)
        if prev is not None and ks != prev[1]:
            lo, hi = prev[0], s
            for it in range(25):
                mid = (lo + hi) / 2; km, _ = kset(p, mid)
                if km == prev[1]: lo = mid
                else: hi = mid
            print(f"p={p}: K* {prev[1]} -> {ks} at s = {(lo + hi) / 2:.6f}", flush=True)
        prev = (s, ks)
    print(f"p={p}: done; K* at s={grid[-1]:.2f}: {prev[1]}", flush=True)
