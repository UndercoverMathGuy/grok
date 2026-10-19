"""Order of the cheapness gain g = |A| - nu^{1-s} Q^s in the interference R = (sum_{T != 0} |beta_T|^2)^{1/2}.
Aligned atoms: g <= 2(s-1) R_I^2/(nu |A|) (D5 corollary, second order).  Misaligned atoms: maximise g/R near IF."""
import itertools, numpy as np
from scipy.optimize import minimize
rng = np.random.default_rng(5)


def data(x, N, n, aligned):
    if aligned:
        mod = np.exp(x[:N]); ph = np.zeros(N); xi = np.exp(1j * x[N:].reshape(N, n))
    else:
        mod = np.exp(x[:N]); ph = x[N:2 * N]; xi = np.exp(1j * x[2 * N:].reshape(N, n))
    beta = mod * np.exp(1j * ph)
    bT = {T: np.sum(beta * np.prod(np.where(np.array(T)[None, :] == 1, np.conj(xi), 1.0), axis=1))
          for T in itertools.product([0, 1], repeat=n)}
    return beta, bT


for n in [2, 3]:
    nu = n + 1
    for s in [1.25, nu / 2]:
        r = 1 / s
        for aligned in [True, False]:
            for Rtarget in [1e-1, 1e-2, 1e-3]:
                best = -np.inf
                for t in range(12):
                    N = nu + int(rng.integers(0, 3))
                    om = np.exp(2j * np.pi / nu)
                    xi0 = np.array([[np.angle(om ** (l % nu))] * n for l in range(N)])
                    x0 = np.concatenate([np.log(np.full(N, 1.0 / nu) * rng.uniform(0.5, 1.5, N))] +
                                        ([] if aligned else [0.1 * rng.standard_normal(N)]) +
                                        [(xi0 + 0.3 * rng.standard_normal((N, n))).ravel()])

                    def obj(x):
                        beta, bT = data(x, N, n, aligned)
                        A = abs(bT[(0,) * n]); R = np.sqrt(sum(abs(v) ** 2 for T, v in bT.items() if any(T)))
                        g = A - nu ** (1 - s) * np.sum(np.abs(beta) ** r) ** s
                        return -(g / A) + 1e3 * (R / A - Rtarget) ** 2
                    res = minimize(obj, x0, method='Nelder-Mead' if len(x0) < 8 else 'Powell',
                                   options={'maxiter': 20000, 'xtol': 1e-10, 'ftol': 1e-14})
                    beta, bT = data(res.x, N, n, aligned)
                    A = abs(bT[(0,) * n]); R = np.sqrt(sum(abs(v) ** 2 for T, v in bT.items() if any(T)))
                    g = A - nu ** (1 - s) * np.sum(np.abs(beta) ** r) ** s
                    best = max(best, (g / A) / (R / A))
                print('n=%d s=%.2f %-9s R/|A|~%.0e : max gain/R = %.4e' % (n, s, 'aligned' if aligned else 'misaligned',
                      Rtarget, best), flush=True)
