"""Sanity checks of the n=3 payment chain on random and adversarial atom systems (floating point).

For each atom system and s:
 (C1) theta Q^s >= L * M_r(spec Hhat)          (Rotfel'd, chain (0,1,1,2), order (1,2,3))
 (C2) |Hhat_32| = |u_2| <= G(a,|b_2|), |Hhat_41| = |w_123| <= G(a,|b_123|)   (Fubini-Study triangle)
 (C3) tr E = 0, tr E^2 = 2P <= 2(4m^2+2G^2), tr E^3 >= -24 m^2 G,  lam >= -1
 (C4) M_r(spec) >= F_r(Pbar, Tbar)   (closed-form inner solver)
 (TH) bonus <= c * M  for the certified (s*, c) pairs:  pt (1.8553,0.845), (2,0.955), (1.82,0.809), (1.94,0.9);
      cf (1.98, 0.8506), each tested at s in [1, s*].
Adversarial part: maximise bonus - c*M by local search (all widths 1..5) for each certified pair.
"""
import itertools
import numpy as np
from scipy.optimize import minimize
from n3lib import coeffs, unpack, TNAMES
from kbound import G
from ptinner import F

IX = {nm: k for k, nm in enumerate(TNAMES)}
CERTS = [('pt', 1.8553, 0.845), ('pt', 2.0, 0.955), ('pt', 1.82, 0.809), ('pt', 1.94, 0.9), ('cf', 1.98, 0.8506)]


def Mcur(bT, cur):
    m = np.abs(bT)
    if cur == 'pt':
        return m.max()
    return max(m.max(), m[IX['12']] + m[IX['13']], m[IX['12']] + m[IX['23']], m[IX['13']] + m[IX['23']])


def chain(beta, ang):
    rho = np.abs(beta); p = beta / rho; xi = np.exp(1j * ang)
    U = np.stack([np.ones_like(p), p * xi[:, 0], p * xi[:, 0] * xi[:, 1], p ** 2 * xi[:, 0] * xi[:, 1] * xi[:, 2]], axis=1)
    L = rho.sum()
    H = (U.T * rho) @ U.conj() / L
    return H, L


def checks(beta, ang, s, worst):
    r = 1 / s
    A, bT = coeffs(beta, ang)
    rho = np.abs(beta); L = rho.sum(); Q = np.sum(rho ** r)
    theta = 4.0 ** (1 - s)
    H, _ = chain(beta, ang)
    nu = np.linalg.eigvalsh(H)
    nu = np.where(nu < 1e-12, 0.0, nu)  # eigenvalue noise of rank-deficient Gram matrices
    Mr = np.mean(nu ** r) ** (1 / r)
    worst['C1'] = max(worst.get('C1', -np.inf), L * Mr - theta * Q ** s)
    a = abs(A) / L
    b = np.abs(bT) / L
    worst['C2u'] = max(worst.get('C2u', -np.inf), abs(H[2, 1]) - G(a, b[IX['2']]))
    worst['C2w'] = max(worst.get('C2w', -np.inf), abs(H[3, 0]) - G(a, b[IX['123']]))
    m = max(b[IX[t]] for t in ['1', '12', '23', '3', '2', '123'])
    g = G(a, m)
    E = H - np.eye(4)
    lam = np.linalg.eigvalsh(E)
    P = np.sum(lam ** 2) / 2
    worst['C3P'] = max(worst.get('C3P', -np.inf), P - (4 * m * m + 2 * g * g))
    worst['C3T'] = max(worst.get('C3T', -np.inf), -24 * m * m * g - np.sum(lam ** 3))
    Fv = F(min(4 * m * m + 2 * g * g, 6.0), -24 * m * m * g, r)
    worst['C4'] = max(worst.get('C4', -np.inf), Fv - Mr)
    bonus = abs(A) - theta * Q ** s
    for cur, sst, c in CERTS:
        if s <= sst + 1e-12:
            key = f"TH_{cur}_{sst}_{c}"
            worst[key] = max(worst.get(key, -np.inf), (bonus - c * Mcur(bT, cur)) / L)
    return worst


if __name__ == '__main__':
    rng = np.random.default_rng(11)
    worst = {}
    for t in range(4000):
        N = rng.integers(1, 7)
        beta = rng.exponential(1, N) * np.exp(1j * rng.uniform(0, 2 * np.pi, N)) * (rng.random() < 0.5 or np.exp(1j * 0.3 * rng.normal(size=N)))
        if rng.random() < 0.5:   # near-aligned phases (large |A|/L)
            beta = np.abs(beta) * np.exp(1j * 0.3 * rng.normal(size=N))
        ang = rng.uniform(0, 2 * np.pi, (N, 3))
        if rng.random() < 0.3:   # slot-symmetric
            ang = np.repeat(rng.uniform(0, 2 * np.pi, (N, 1)), 3, axis=1)
        s = rng.uniform(1.0, 2.0)
        checks(beta, ang, s, worst)
    print("random systems (max violation; must be <= ~1e-12):")
    for k, v in sorted(worst.items()):
        print(f"   {k}: {v:.3e}")
    # adversarial: maximise (bonus - c M)/L
    print("adversarial local search, max (bonus - c*M)/L (must be <= 0):")
    for cur, sst, c in CERTS:
        best = -np.inf
        for s in [sst, 0.5 * (1 + sst) + 0.25 * (sst - 1)]:
            r = 1 / s
            for N in range(1, 6):
                for rep in range(6):
                    v0 = np.concatenate([rng.uniform(0.2, 1, N), rng.uniform(-np.pi, np.pi, 4 * N)])
                    def obj(v):
                        beta, ang = unpack(v, N)
                        A, bT = coeffs(beta, ang)
                        L = np.abs(beta).sum()
                        return -(abs(A) - 4 ** (1 - s) * np.sum(np.abs(beta) ** r) ** s - c * Mcur(bT, cur)) / L
                    res = minimize(obj, v0, method='Nelder-Mead', options=dict(maxiter=6000, xatol=1e-10, fatol=1e-13))
                    best = max(best, -res.fun)
        print(f"   {cur} s*={sst} c={c}: {best:.3e}")
