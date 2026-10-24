"""Decoupled twisted payment (rigorous reduction of the twisted Rotfel'd bound to one variable L).

Spectrum of a unit-diagonal PSD 3x3 matrix with off-diagonal (b, c, d'): nu = 1 + mu, mu^3 - P mu - q = 0,
P = |b|^2+|c|^2+|d'|^2, q = 2 Re(b c conj(d')).  S(P,q) = sum (1+mu_k)^r is decreasing in P and increasing in q on the
feasible set {0<=P<=3, max(P-1, -2(P/3)^{3/2}) <= q <= 2 (P/3)^{3/2}}.
With b = B/L, c = C/L, d' = D'/L, |D' - alpha D| <= rho_L = sqrt(2L(L-|A|)), |D'| <= L:
   P <= Pmax(L) = (|B|^2+|C|^2+min(|D|+rho_L, L)^2)/L^2,
   q >= qmin(L) = 2 (Re(B C conj(alpha D)) - rho_L |B||C|)/L^3,
   payment:  P_1 >= min_L  L * ( S(Pmax, q*)/3 )^s,  q* = max(qmin, Pmax - 1, -2 (Pmax/3)^{3/2})  (Pmax capped at 3).
"""
import numpy as np


def roots(P, q):
    """three real roots of mu^3 - P mu - q (trigonometric form); assumes 27 q^2 <= 4 P^3"""
    if P <= 1e-15:
        return np.zeros(3)
    m = 2 * np.sqrt(P / 3)
    arg = np.clip(3 * q / (P * m), -1, 1)
    th = np.arccos(arg) / 3
    return np.array([m * np.cos(th - 2 * np.pi * k / 3) for k in range(3)])


def S(P, q, r):
    mu = roots(P, q)
    return np.sum(np.clip(1 + mu, 0, None) ** r)


def pay_tw2(Aabs, Babs, Cabs, Dabs, cosphi, s, nL=400):
    """lower bound on P_1 from the decoupled twisted bound; cosphi = cos(arg(B C conj(alpha D)))"""
    r = 1 / s
    Lmin = max(Aabs, Babs, Cabs, Dabs)
    best = np.inf
    # L range: beyond L >= thr-independent cap the bound 3^{1-s} L is large anyway; scan geometric grid
    for L in Lmin * np.concatenate([[1.0], 1 + np.geomspace(1e-5, 3, nL)]):
        rho = np.sqrt(max(2 * L * (L - Aabs), 0))
        Dm = min(Dabs + rho, L)
        Pm = min((Babs ** 2 + Cabs ** 2 + Dm ** 2) / L ** 2, 3.0)
        qm = 2 * (Babs * Cabs * Dabs * cosphi - rho * Babs * Cabs) / L ** 3
        qs = max(qm, Pm - 1, -2 * (Pm / 3) ** 1.5)
        qs = min(qs, 2 * (Pm / 3) ** 1.5)
        val = L * (S(Pm, qs, r) / 3) ** s
        best = min(best, val)
    return best


if __name__ == '__main__':
    from twisted_pay import twisted_lb
    s = 1.5
    rng = np.random.default_rng(0)
    print("single atom:", pay_tw2(1, 1, 1, 1, 1.0, s), " IF:", pay_tw2(1, 0, 0, 0, 1.0, s))
    for k in range(12):
        m = rng.uniform(0.3, 1.2, 3); phi = rng.normal(0, 0.5)
        B, C, D = m[0], m[1], m[2] * np.exp(1j * phi)
        full = twisted_lb(1.0, B, C, D, s, tries=8)
        dec = pay_tw2(1.0, m[0], m[1], m[2], np.cos(-phi), s)
        print(f"m={np.round(m,3)} phi={phi:.3f}: full twisted {full:.4f}  decoupled {dec:.4f}")
