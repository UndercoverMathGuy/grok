# Support gap delta_min(s,p) = 1 - max{ Gamma^{(S)}_s / Gamma_s : S contains no support of a programme maximiser }.
# By the slack principle, approximate decoupling with relative error eps < delta_min/(1-delta_min) at a global maximiser
# forces its feature set to contain an optimal support (lower half of (C)).  Exhaustive over S for small p.
import sys, itertools, numpy as np
from transitions_fast import best_support_value

def all_supports(p, s):
    F = (p - 1) // 2; vals = {}
    for k in range(1, F + 1):
        for S in itertools.combinations(range(1, F + 1), k):
            vals[S] = best_support_value(p, s, S)          # value with support exactly S (all entries > 0), -1 if none
    return vals

def gamma_restricted(vals, S):
    """Gamma^{(S)}: best over supports contained in S."""
    return max(v for T, v in vals.items() if set(T) <= set(S))

if __name__ == "__main__":
    p = int(sys.argv[1]); grid = [float(v) for v in sys.argv[2].split(",")]
    F = (p - 1) // 2
    for s in grid:
        vals = all_supports(p, s); G = max(vals.values())
        opt = [T for T, v in vals.items() if v >= G * (1 - 1e-9)]
        worst = (0.0, None)
        for k in range(1, F + 1):
            for S in itertools.combinations(range(1, F + 1), k):
                if any(set(T) <= set(S) for T in opt): continue
                r = gamma_restricted(vals, S) / G
                if r > worst[0]: worst = (r, S)
        dmin = 1 - worst[0] if worst[1] is not None else float('inf')
        eps = dmin / (1 - dmin) if dmin < 1 else float('inf')
        print(f"p={p} s={s:.3f}: K*={sorted(set(len(T) for T in opt))} optimal supports {opt[:3]}  delta_min={dmin:.4f} (best non-containing S={worst[1]})  tolerated eps={eps:.4f}", flush=True)
