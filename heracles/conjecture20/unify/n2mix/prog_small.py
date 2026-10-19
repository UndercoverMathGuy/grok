"""Programme Gamma_s(p) and restricted Gamma_s^{(S)}(p) for small p (exploration; reweighted LP, many restarts).
Prints K*, the binding set, and the dense-regime slack ratio Gamma^{(F minus one)}/Gamma."""
import sys
import numpy as np
from scipy.optimize import linprog


def kappa(p):
    F = (p - 1) // 2
    j = np.arange(1, F + 1); d = np.arange(1, F + 1)
    return 1 - np.cos(2 * np.pi * np.outer(d, j) / p)


def solve(K, s, restarts=30, seed=0, support=None):
    r = 1 / s; nF = K.shape[1]
    idx = np.arange(nF) if support is None else np.array(support)
    Ks = K[:, idx]
    rng = np.random.default_rng(seed); best = (np.inf, None)
    for t in range(restarts):
        w = rng.uniform(0.2, 2.0, len(idx)) if t else np.ones(len(idx))
        for it in range(200):
            res = linprog(w, A_ub=-Ks, b_ub=-np.ones(Ks.shape[0]), bounds=(0, None), method='highs')
            y = res.x; wn = (y + 1e-10) ** (r - 1)
            if np.allclose(wn, w, rtol=1e-9): break
            w = wn
        Q = np.sum(y[y > 1e-12] ** r)
        if Q < best[0]: best = (Q, y.copy())
    Q, y = best
    yfull = np.zeros(nF); yfull[idx] = y / Q ** s
    return Q ** (-s), yfull


if __name__ == '__main__':
    p = int(sys.argv[1])
    K = kappa(p); nF = K.shape[1]
    for s in np.arange(1.0, 1.501, 0.025):
        G, y = solve(K, s)
        Ksupp = int(np.sum(y > 1e-9))
        Gs = max(solve(K, s, support=[i for i in range(nF) if i != j], restarts=12)[0] for j in [0])
        print(f"p={p} s={s:.3f} b={3/s:.3f} Gamma={G:.6f} K*={Ksupp} y*={np.round(y,4)}  Gamma(F-1)/Gamma={Gs/G:.5f}  3^(s-1)*ratio={3**(s-1)*Gs/G:.5f}", flush=True)
