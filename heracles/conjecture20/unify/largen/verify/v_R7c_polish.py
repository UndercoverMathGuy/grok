# Analytic-gradient polish of the arc atoms (v_R7c_arc.py output arcref_n*.npy) for k*(n, nu/2), then interval
# re-certification (v_R5.kstar_iv).  Purpose: locate where certified lower bounds cross k_7 = 1.81987 (R7c's p=7 claim).
import sys, os, math
import numpy as np, mpmath as mp
from math import comb
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v_R5 import kstar_iv
K7 = math.sqrt(1 / math.tan(math.pi / 7) ** 2 - 1)

def fg(v, m, n, s):
    rho, al, ph = v[:m], v[m:2 * m], v[2 * m:]
    beta = np.exp(rho + 1j * al); xi = np.exp(1j * ph); nu = n + 1
    K = np.arange(1, n + 1); C = np.array([comb(n, k) for k in K], float)
    P = beta[None, :] * xi[None, :] ** K[:, None]          # (n, m): beta_i xi_i^k
    mk = P.sum(1)
    E = np.sum(C * np.abs(mk) ** 2)
    ReA = np.sum(beta.real); Q = np.sum(np.exp(rho / s)); IF = nu ** (1 - s) * Q ** s
    num = ReA - IF; den = math.sqrt(E / 2); r = num / den
    w = (C * np.conj(mk))[:, None]                          # (n,1)
    dE_rho = 2 * np.real(np.sum(w * P, 0)); dE_al = 2 * np.real(np.sum(w * 1j * P, 0)); dE_ph = 2 * np.real(np.sum(w * 1j * K[:, None] * P, 0))
    dnum_rho = beta.real - nu ** (1 - s) * Q ** (s - 1) * np.exp(rho / s); dnum_al = -beta.imag
    g_rho = dnum_rho / den - num / den ** 3 * dE_rho / 4
    g_al = dnum_al / den - num / den ** 3 * dE_al / 4
    g_ph = -num / den ** 3 * dE_ph / 4
    return r, np.concatenate([g_rho, g_al, g_ph])

if __name__ == '__main__':
    # gradient check
    rng = np.random.default_rng(1); m, n, s = 5, 7, 4.0
    v = rng.standard_normal(3 * m); r, g = fg(v, m, n, s); h = 1e-6
    gn = np.array([(fg(v + h * e, m, n, s)[0] - fg(v - h * e, m, n, s)[0]) / (2 * h) for e in np.eye(3 * m)])
    print("gradient check max abs err:", np.abs(g - gn).max())
    for n in [26, 27, 28, 29]:
        f = os.path.join(HERE, f'arcref_n{n}.npy')
        if not os.path.exists(f): continue
        d = np.load(f); m = int(d[0]); z = d[1:]; s = (n + 1) / 2
        r0 = fg(z, m, n, s)[0]
        res = minimize(lambda v: tuple(-x for x in fg(v, m, n, s)), z, jac=True, method='L-BFGS-B',
                       options=dict(maxiter=20000, maxcor=40, gtol=1e-12, ftol=1e-15))
        rv, _, _, _ = kstar_iv(n, res.x[:m], res.x[m:2 * m], res.x[2 * m:])
        np.save(os.path.join(HERE, f'polished_n{n}.npy'), np.concatenate([[m], res.x]))
        print(f"n={n} (s={s}): start {r0:.5f} -> polished {-res.fun:.5f}; certified k* >= {mp.nstr(rv.a, 8)};"
              f" exceeds k_7: {rv.a > K7}", flush=True)
