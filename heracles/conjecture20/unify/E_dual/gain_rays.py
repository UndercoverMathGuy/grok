"""Gain g = |A| - nu^{1-s} Q^s versus interference R along rays from the KM2 block (n=2,3), aligned vs misaligned
perturbations: g/R -> 0 (second order) or -> const (first order)?"""
import itertools, numpy as np
rng = np.random.default_rng(11)


def gain_R(beta, xi, nu, s):
    n = xi.shape[1]; r = 1 / s
    bT = {T: np.sum(beta * np.prod(np.where(np.array(T)[None, :] == 1, np.conj(xi), 1.0), axis=1))
          for T in itertools.product([0, 1], repeat=n)}
    A = abs(bT[(0,) * n]); R = np.sqrt(sum(abs(v) ** 2 for T, v in bT.items() if any(T)))
    return A - nu ** (1 - s) * np.sum(np.abs(beta) ** r) ** s, R, A


for n in [2, 3]:
    nu = n + 1
    for s in [1.25, nu / 2]:
        om = np.exp(2j * np.pi / nu)
        beta0 = np.full(nu, 1.0 / nu, dtype=complex); xi0 = np.array([[om ** l] * n for l in range(nu)])
        for kind in ['aligned', 'misaligned']:
            rows = []
            for eps in [1e-1, 1e-2, 1e-3, 1e-4]:
                best = -np.inf
                for t in range(4000):
                    dmod = rng.standard_normal(nu); dph = rng.standard_normal(nu) if kind == 'misaligned' else 0
                    dxi = rng.standard_normal((nu, n))
                    beta = beta0 * np.exp(eps * dmod + 1j * eps * dph); xi = xi0 * np.exp(1j * eps * dxi)
                    g, R, A = gain_R(beta, xi, nu, s)
                    if R > 1e-14: best = max(best, g / R)
                rows.append('eps=%.0e: %.3e' % (eps, best))
            print('n=%d s=%.2f %-10s max g/R over 4000 random rays: ' % (n, s, kind) + '; '.join(rows), flush=True)
