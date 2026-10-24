"""Is the ~4e-9 'violation' of D5 seen in v2's adversarial search real?  Re-run the adversarial search, keep the
worst points, and evaluate lhs - rhs at 60 digits (sqrt(2(1-|A|/L)) loses half the digits in double precision)."""
import numpy as np, mpmath as mp
from scipy.optimize import minimize
mp.mp.dps = 60
rng = np.random.default_rng(11)


def intervals(n):
    return [(k0, k1) for k0 in range(n + 1) for k1 in range(k0 + 1, n + 1)]


def d5_viol(beta, xi):
    N, n = xi.shape; nu = n + 1
    Lm = np.abs(beta).sum(); A = beta.sum(); al = min(1.0, abs(A) / Lm)
    bI = np.array([np.sum(beta * np.prod(np.conj(xi[:, k0:k1]), axis=1)) for (k0, k1) in intervals(n)])
    lhs = np.sum((np.abs(beta) / Lm) ** 2)
    rhs = 1 / nu + 2 / nu ** 2 * np.sum((np.abs(bI) / Lm + np.sqrt(2 * (1 - al))) ** 2)
    return lhs - rhs


def d5_viol_mp(z, N, n):
    nu = n + 1
    mod = [mp.e ** mp.mpf(v) for v in z[:N]]; ph = [mp.mpf(v) for v in z[N:2 * N]]
    xph = [[mp.mpf(z[2 * N + l * n + t]) for t in range(n)] for l in range(N)]
    beta = [mod[l] * mp.expj(ph[l]) for l in range(N)]
    Lm = sum(abs(b) for b in beta); A = sum(beta)
    tot = mp.mpf(0)
    for (k0, k1) in intervals(n):
        bI = sum(beta[l] * mp.expj(-sum(xph[l][k0:k1])) for l in range(N))
        tot += (abs(bI) / Lm + mp.sqrt(2 * (1 - abs(A) / Lm))) ** 2
    lhs = sum((abs(b) / Lm) ** 2 for b in beta)
    return lhs - (mp.mpf(1) / nu + mp.mpf(2) / nu ** 2 * tot)


def unpack(z, N, n):
    mod = np.exp(z[:N]); ph = z[N:2 * N]; xph = z[2 * N:].reshape(N, n)
    return mod * np.exp(1j * ph), np.exp(1j * xph)


res_all = []
for trial in range(150):
    n = int(rng.integers(1, 6)); nu = n + 1; N = int(rng.integers(1, 3 * nu + 1))
    z = np.concatenate([rng.normal(0, 1, N), rng.normal(0, rng.uniform(0, 2), N), rng.uniform(0, 2 * np.pi, N * n)])
    res = minimize(lambda zz: -d5_viol(*unpack(zz, N, n)), z, method='Nelder-Mead',
                   options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-14})
    res_all.append((-res.fun, N, n, res.x))
res_all.sort(key=lambda t: -t[0])
for v, N, n, z in res_all[:8]:
    print('double: %.3e   60-digit: %s   (N=%d, n=%d)' % (v, mp.nstr(d5_viol_mp(z, N, n), 5), N, n))
