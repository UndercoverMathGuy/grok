"""V_A: atomic free relaxation with atoms only on the feature set A (all fields j in F carry free non-own modes;
fields j not in A have zero own coefficients).  At a dense prime (K* = F), V_A < 1 for every proper A implies (C) for all
networks (REPORT, Proposition D).  Usage: python freesub.py p n b restarts seed natoms sym A...
"""
import sys, time
import numpy as np
from freeatoms2 import FA2

def run(p, n, b, restarts, seed, natoms, sym, A, verbose=True):
    s = (n + 1) / b; F = (p - 1) // 2; S = list(range(1, F + 1))
    M = FA2(p, n, s, S, natoms, sym)
    for j in S:
        if j not in A:
            M.na[j] = 0
    rng = np.random.default_rng(seed)
    best = (-np.inf, None); t0 = time.time()
    for k in range(restarts):
        th = []
        for j in S:
            m = M.na[j]
            if m == 0:
                continue
            y = max(M.ystar[j - 1], 0.1)
            beta = y / m * (rng.normal(size=m) + 1j * rng.normal(size=m) + (1.5 if k % 2 else 0))
            ph = rng.uniform(0, 2 * np.pi, m if sym else m * n)
            if k % 5 == 4:   # IF block start
                beta = np.full(m, y / (n + 1), dtype=complex); beta[n + 1:] = 0.01 * y
                ph = np.array([2 * np.pi * i / (n + 1) for i in range(m)]) if sym else np.repeat([2 * np.pi * i / (n + 1) for i in range(m)], n)
                ph = ph + 0.3 * rng.normal(size=ph.shape)
            th += list(beta.real) + list(beta.imag) + list(ph)
        th = np.array(th)
        try:
            val, th2 = M.opt(th)
        except Exception as e:
            continue
        if val > best[0]:
            best = (val, th2)
        if verbose and (k % 10 == 9 or val > 1.0 + 1e-6):
            print(f"  restart {k}: {val:.6f}   best {best[0]:.6f}  ({time.time()-t0:.0f}s)", flush=True)
    return M, best

if __name__ == '__main__':
    p, n, b = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    restarts, seed, natoms, sym = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
    A = [int(v) for v in sys.argv[8:]]
    print(f"V_A search p={p} n={n} b={b} A={A} natoms={natoms} sym={sym}", flush=True)
    M, (val, th) = run(p, n, b, restarts, seed, natoms, sym, A)
    np.save(f"fs_p{p}_n{n}_b{b}_A{''.join(map(str,A))}_m{natoms}_sym{sym}_s{seed}.npy", th)
    print(f"BEST V_A {val:.6f}")
