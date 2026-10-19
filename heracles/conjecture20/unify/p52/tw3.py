"""Circle-coupled twisted payment, best of the three chain variants (float version, for exploration).
Uncertain entry X (one of |B|,|C|,|D|), exact Y, Z:
   P(L,psi) = (mY^2 + mZ^2 + mX^2 + rho^2 + 2 mX rho cos(psi - phi))/L^2,   q = 2 mY mZ (mX cos(phi) + rho cos(psi))/L^3,
   rho = sqrt(2 L (L - |A|)),   payment >= min_{L, psi} L (Smin(P, q)/3)^{3/2}.
"""
import numpy as np
from tw2 import S


def Smin(P, q, r):
    P = min(P, 3.0)
    dh = 2 * (P / 3) ** 1.5
    if q > dh:
        return np.inf
    qs = max(q, P - 1, -dh)
    return S(P, qs, r)


def pay_tw3(m1, m2, m12, phi, s=1.5, nL=120, npsi=72):
    r = 1 / s
    best_chain = 0
    for (mX, mY, mZ) in [(m12, m1, m2), (m2, m1, m12), (m1, m2, m12)]:
        Lmin = max(1.0, m1, m2, m12)
        best = np.inf
        for L in Lmin * np.concatenate([[1.0], 1 + np.geomspace(1e-6, 2, nL)]):
            rho = np.sqrt(max(2 * L * (L - 1.0), 0))
            for psi in np.linspace(0, 2 * np.pi, npsi, endpoint=False):
                P = (mY ** 2 + mZ ** 2 + mX ** 2 + rho ** 2 + 2 * mX * rho * np.cos(psi - phi)) / L ** 2
                q = 2 * mY * mZ * (mX * np.cos(phi) + rho * np.cos(psi)) / L ** 3
                v = L * (Smin(P, q, r) / 3) ** s
                best = min(best, v)
        best_chain = max(best_chain, best)
    return best_chain


if __name__ == '__main__':
    from tw2 import pay_tw2
    for (m1, m2, m12, phi) in [(0.524, 0.497, 0.635, 0.117), (0.555, 0.582, 0.856, 0.269), (0.628, 0.663, 0.881, 0.021), (1, 1, 1, 0)]:
        print((m1, m2, m12, phi), "decoupled", round(pay_tw2(1.0, m1, m2, m12, np.cos(phi), 1.5, nL=300), 4),
              " circle+3chains", round(pay_tw3(m1, m2, m12, phi), 4))
