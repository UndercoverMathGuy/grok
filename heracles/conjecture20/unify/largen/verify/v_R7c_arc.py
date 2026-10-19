# Probe of R7c's p=7 statement ("the same holds for s up to about 15", i.e. k*(2s-1,s) < k_7 = 1.81987 for s <= ~15).
# Independent implementation of the arc family: m phases equally spaced with step Delta = 2pi/(c n + 2), and the
# least-squares atoms (E minimal for A = 1):  beta = conj(H^{-1} 1)/(1^T H^{-1} 1),  H_ij = (1 + xi_i conj xi_j)^n - 1.
# Then L-BFGS polish (double, moment form) and rigorous re-evaluation by v_R5.kstar_iv (interval arithmetic).
import sys, os, math
import numpy as np, mpmath as mp
from math import comb
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v_R5 import kstar_iv
K7 = math.sqrt(1 / math.tan(math.pi / 7) ** 2 - 1)

def arc(n, m, delta, s, dps):
    mp.mp.dps = dps
    xi = [mp.expj(j * delta) for j in range(m)]
    H = mp.matrix(m, m)
    for i in range(m):
        for j in range(m):
            H[i, j] = (1 + xi[i] * mp.conj(xi[j])) ** n - 1
    y = mp.lu_solve(H, mp.matrix([1] * m))
    tot = mp.fsum([y[i] for i in range(m)])
    beta = [mp.conj(y[i]) / mp.conj(tot) for i in range(m)]
    A = mp.fsum(beta); E = 1 / mp.re(tot); nu = n + 1
    Q = mp.fsum([abs(b) ** (1 / mp.mpf(s)) for b in beta])
    r = (mp.re(A) - mp.mpf(nu) ** (1 - mp.mpf(s)) * Q ** mp.mpf(s)) / mp.sqrt(E / 2)
    return r, beta, [j * delta for j in range(m)]

def ratio_sym(v, m, n, s):
    rho, al, ph = v[:m], v[m:2 * m], v[2 * m:3 * m]
    beta = np.exp(rho + 1j * al); xi = np.exp(1j * ph)
    A = beta.sum(); Q = np.exp(rho / s).sum(); nu = n + 1
    K = np.arange(1, n + 1)
    mk = (beta[None, :] * xi[None, :] ** K[:, None]).sum(1)
    E = np.sum(np.array([comb(n, k) for k in K], float) * np.abs(mk) ** 2)
    val = (A.real - nu ** (1 - s) * Q ** s) / math.sqrt(E / 2)
    return val if np.isfinite(val) else -1e9

for n in [26, 27, 28, 29, 30]:
    s = (n + 1) / 2
    cands = []
    for f in [0.72, 0.75, 0.78, 0.81, 0.84]:
        m = int(round(f * n))
        for c in np.arange(1.15, 1.46, 0.05):
            r, beta, ph = arc(n, m, 2 * math.pi / (c * n + 2), s, 40 + n)
            cands.append((float(r), m, c, beta, ph))
    cands.sort(key=lambda t: -t[0])
    best_iv = None
    for r0, m, c, beta, ph in cands[:3]:
        z0 = np.concatenate([[float(mp.log(abs(b))) for b in beta], [float(mp.arg(b)) for b in beta], [float(x) for x in ph]])
        res = minimize(lambda v: -ratio_sym(v, m, n, s), z0, method='L-BFGS-B', options=dict(maxiter=3000, maxcor=30))
        z = res.x if -res.fun > r0 else z0
        rv, _, _, _ = kstar_iv(n, z[:m], z[m:2 * m], z[2 * m:])
        if best_iv is None or rv.a > best_iv[0].a:
            best_iv = (rv, m, c, r0, -res.fun)
            np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'arcref_n{n}.npy'), np.concatenate([[m], z]))
    rv, m, c, r0, rp = best_iv
    print(f"n={n} (s={s}): best arc start {cands[0][0]:.5f}; after polish (double) {rp:.5f}; certified k* >= {mp.nstr(rv.a, 8)}"
          f"  (m={m}, c={c:.2f});  exceeds k_7={K7:.5f}: {rv.a > K7}", flush=True)
