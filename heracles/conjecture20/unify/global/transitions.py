# Transition points of K*(s,p): for each support size K the best value Gamma^{(K)}_s(p) (exact enumeration),
# and the values of s where the argmax over K changes (bisection).
import sys, numpy as np
from lib import exhaustive_programme

def best_by_K(p, s):
    ex = exhaustive_programme(p, s)
    return {k: v[0] for k, v in ex.items() if v[0] > 0}, ex

def kstar(p, s):
    vals, ex = best_by_K(p, s)
    G = max(vals.values())
    return [k for k, v in vals.items() if v >= G * (1 - 1e-12)], G, ex

if __name__ == "__main__":
    for p in [int(a) for a in sys.argv[1:]]:
        grid = np.round(np.arange(1.0, 3.01, 0.02), 3)
        prev = None; out = []
        for s in grid:
            ks, G, ex = kstar(p, s)
            if prev is not None and ks != prev:
                lo, hi = s - 0.02, s
                for it in range(30):
                    mid = (lo + hi) / 2
                    if kstar(p, mid)[0] == prev: lo = mid
                    else: hi = mid
                vals_lo, _ = best_by_K(p, lo); vals_hi, _ = best_by_K(p, hi)
                out.append((round((lo + hi) / 2, 6), prev, ks))
                print(f"p={p}: transition at s={(lo+hi)/2:.6f}: K* {prev} -> {ks};  supports: {ex[ks[0]][1]}", flush=True)
            prev = ks
        print(f"p={p}: K* at s=1: {kstar(p,1.0001)[0]}, at s=3: {kstar(p,3.0)[0]}")
