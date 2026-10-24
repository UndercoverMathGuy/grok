"""Equality configurations of Paper Lemma 7 (tight case).
At equality all beta_l = beta/nu (shown in the report: H normal + triangular => H = I => d_l = 1),
so we need nu unimodular points xi_l in T^n with sum_l prod_{t in T} xi_lt = 0 for every nonempty T subset [n].
We search numerically and test whether every solution is equivalent to the Fourier one
(xi_lt = c_t * omega^{pi_t(l)}, omega = e^{2 pi i/nu}, with the structure forced by Z_nu)."""
import numpy as np, itertools
from scipy.optimize import least_squares
rng = np.random.default_rng(0)

def subsets(n):
    for k in range(1, n+1):
        for T in itertools.combinations(range(n), k):
            yield T

def resid(phi, n):
    nu = n+1
    th = phi.reshape(nu, n)
    out = []
    for T in subsets(n):
        z = np.exp(1j*th[:, list(T)].sum(axis=1)).sum()
        out += [z.real, z.imag]
    return np.array(out)

def classify(th, n):
    nu = n+1
    # normalise: rotate each slot so that point 0 has phase 0
    th = th - th[0:1, :]
    z = np.exp(1j*th)
    # check whether each slot's points are nu-th roots of unity
    k = np.angle(z)*nu/(2*np.pi)
    dev_roots = np.max(np.abs(k - np.round(k)))
    return dev_roots, np.round(np.mod(np.round(k), nu)).astype(int)

for n in [2, 3, 4, 5]:
    nu = n+1
    sols = 0; nonroot = 0; worst = 0.0
    for trial in range(300 if n < 5 else 150):
        x0 = rng.uniform(0, 2*np.pi, nu*n)
        r = least_squares(resid, x0, args=(n,), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=4000)
        if np.max(np.abs(r.fun)) < 1e-10:
            sols += 1
            d, K = classify(r.x.reshape(nu, n), n)
            worst = max(worst, d)
            if d > 1e-6:
                nonroot += 1
                if nonroot <= 2:
                    print("  non-root solution n=%d:" % n, np.round(np.angle(np.exp(1j*(r.x.reshape(nu,n)-r.x.reshape(nu,n)[0:1])))/np.pi, 4))
    print(f"n={n} nu={nu}: solutions {sols}, non-root-of-unity solutions {nonroot}, max deviation from nu-th roots {worst:.2e}")
