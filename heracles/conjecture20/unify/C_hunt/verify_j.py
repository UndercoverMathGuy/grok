# Exact (mpmath, 40 digits) re-evaluation of saved jtest2 designs: paired loss / L1 at delta0 = 1, real clean part (A = 0).
# A value below theta_P(n,s) = 1 - (n/(n+1))^{s-1} refutes (J'_{theta_P}) at that p (explicit counterexample).
import sys, glob, itertools
import numpy as np, mpmath as mp
mp.mp.dps = 40
for f in sys.argv[1:]:
    z = np.load(f); p = int(z['p']); n = int(z['n']); psi = z['psi']; r = z['r']; F = (p - 1) // 2
    Ts = [T for T in itertools.product([0, 1], repeat=n) if any(T)]
    cols = [(l, T) for l in range(1, F + 1) for T in Ts]
    if len(r) > len(cols): print(f, 'cplx design: skipped (has clean part)'); continue
    a = [(cols[k], mp.mpf(float(r[k])) * mp.expj(mp.mpf(float(psi[k])))) for k in range(len(cols)) if r[k] > 1e-14]
    pi = mp.pi
    L1 = mp.fsum(abs(ak) * abs(1 - mp.expj(-2 * pi * l / p)) for (l, T), ak in a)
    worst = mp.mpf(0)
    for d in (1, p - 1):
        cd = {l: 1 - mp.expj(-2 * pi * l * d / p) for l in range(1, F + 1)}
        for x in itertools.product(range(p), repeat=n):
            val = mp.fsum(mp.re(ak * cd[l] * mp.expj(2 * pi * 2 * l * sum(x[t] for t in range(n) if T[t]) / p)) for (l, T), ak in a)
            if -val > worst: worst = -val
    print(f"{f}: p={p} n={n} exact paired loss/L1 = {mp.nstr(worst / L1, 12)}  (stored rho/2 = {float(z['rho']) / 2:.12f})", flush=True)
