# k*(n,s) = sup (Re A - nu^{1-s} Q^s) / sqrt(E/2) over atom systems (beta_i in C, xi_i in T^n).
#   A = sum beta_i, Q = sum |beta_i|^r, E = sum_{T != 0} |sum_i beta_i prod_{t in T} xi_it|^2
#     = sum_{ij} beta_i conj(beta_j) (prod_t (1 + xi_it conj xi_jt) - 1).
# Companion note Thm 14: single-frequency networks at p>=5 cannot beat IF when k_p >= k*(n,s),
# k_p = sqrt(cot^2(pi/p)-1)  (k_5 = 0.946, k_7 = 1.825, k_11 = 3.32).
# Usage: python kstar_large.py n1,n2,... m1,m2,... starts [b]
import sys, math, numpy as np
from scipy.optimize import minimize

def unpack(z, m, n):
    rho = z[:m]; al = z[m:2*m]; th = z[2*m:].reshape(m, n)
    return rho, al, th

def obj(z, m, n, s):
    nu = n + 1; r = 1.0 / s
    rho, al, th = unpack(z, m, n)
    b = np.exp(rho + 1j * al)
    A = b.sum()
    Q = np.exp(r * rho).sum()
    X = np.exp(1j * th)                                   # m x n
    F = 1 + X[:, None, :] * X.conj()[None, :, :]          # m x m x n
    # prefix / suffix products for derivative of prod over t
    pre = np.ones((m, m, n + 1), complex); suf = np.ones((m, m, n + 1), complex)
    for t in range(n):
        pre[:, :, t + 1] = pre[:, :, t] * F[:, :, t]
        suf[:, :, n - t - 1] = suf[:, :, n - t] * F[:, :, n - t - 1]
    M = pre[:, :, n]
    Mb = M @ b.conj()                                     # (M bbar)_k
    E = np.real(b @ Mb) - abs(A) ** 2
    E = max(E, 1e-300)
    IF = nu ** (1 - s) * Q ** s
    num = A.real - IF
    # gradients
    dnum_rho = np.real(b) - nu ** (1 - s) * s * Q ** (s - 1) * r * np.exp(r * rho)
    dnum_al = np.real(1j * b)
    dE_rho = 2 * np.real(b * Mb) - 2 * np.real(b * np.conj(A))
    dE_al = -2 * np.imag(b * Mb) + 2 * np.imag(b * np.conj(A))
    Mminus = pre[:, :, :n] * suf[:, :, 1:]                # m x m x n : prod over s != t
    dM = 1j * X[:, None, :] * X.conj()[None, :, :] * Mminus   # d M_kj / d theta_kt
    dE_th = 2 * np.real(b[:, None] * np.einsum('kjt,j->kt', dM, b.conj()))
    sq = math.sqrt(E / 2)
    f = num / sq
    g_rho = (dnum_rho - num * dE_rho / (2 * E)) / sq
    g_al = (dnum_al - num * dE_al / (2 * E)) / sq
    g_th = (-num * dE_th / (2 * E)) / sq
    grad = np.concatenate([g_rho, g_al, g_th.ravel()])
    return f, grad, dict(A=A, Q=Q, E=E, num=num, IF=IF, L=np.abs(b).sum())

def search(n, m, s, starts, seed=0, iters=3000):
    rng = np.random.default_rng(seed)
    best = (-np.inf, None)
    for k in range(starts):
        z0 = np.concatenate([rng.uniform(-0.5, 0.5, m), rng.uniform(-0.6, 0.6, m) if k % 2 else rng.uniform(-3, 3, m),
                             rng.uniform(0, 2 * np.pi, m * n)])
        def fun(z):
            f, g, _ = obj(z, m, n, s)
            if not np.isfinite(f): return 1e6, np.zeros_like(z)
            return -f, -g
        try:
            res = minimize(fun, z0, jac=True, method='L-BFGS-B', options=dict(maxiter=iters, gtol=1e-10))
        except Exception:
            continue
        f, _, info = obj(res.x, m, n, s)
        if info['num'] > 0 and f > best[0]:
            best = (f, res.x.copy(), info)
    return best

if __name__ == "__main__":
    ns = [int(v) for v in sys.argv[1].split(',')]
    ms = [int(v) for v in sys.argv[2].split(',')]
    starts = int(sys.argv[3])
    bval = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    for n in ns:
        nu = n + 1; s = nu / bval
        row = {}
        bestall = (-np.inf, None, None, None)
        for m in ms:
            v, z, info = search(n, m, s, starts, seed=1000 * n + m)
            row[m] = v
            if v > bestall[0]: bestall = (v, z, info, m)
        print(f"n={n} b={bval} s={s:.3f}: k* by width " + " ".join(f"{m}:{row[m]:.4f}" for m in ms) +
              f"  max {bestall[0]:.4f} at m={bestall[3]}", flush=True)
        v, z, info, m = bestall
        if z is not None:
            rho, al, th = unpack(z, m, n)
            mod = np.exp(rho); mod = mod / mod.sum()
            print(f"   |beta|/L = {np.round(np.sort(mod)[::-1],4)}  ReA/L={info['A'].real/info['L']:.4f} |A|/L={abs(info['A'])/info['L']:.4f}"
                  f"  IF/L={info['IF']/info['L']:.4f}  E/|A|^2={info['E']/abs(info['A'])**2:.4f}", flush=True)
