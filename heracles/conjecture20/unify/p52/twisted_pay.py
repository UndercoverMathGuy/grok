"""Twisted-Rotfel'd payment at n=2 (phase-aware lower bound on P_1 = 3^{1-s} Q^s).

For atoms beta_i = rho_i p_i (rho_i > 0, |p_i| = 1), xi_i, eta_i in T, put u_i = (1, p_i xi_i, p_i^2 xi_i eta_i),
H = sum rho_i u_i u_i^* (PSD, diagonal L = sum rho_i).  Entries below the diagonal: H21 = B, H32 = C, H31 = D' :=
sum rho_i p_i^2 xi_i eta_i, and |D' - alpha D| <= sqrt(2 L (L - |A|)), alpha = A/|A|.
Rotfel'd (r <= 1): 3^r Q >= sum_k sigma_k(H)^r = L^r sum nu_k^r (nu = eigenvalues of H/L), hence
     P_1 = 3^{1-s} Q^s >= L ( (1/3) sum_k nu_k^r )^s .
Lower bound on P_1 from own data d=(A,B,C,D):  min over L >= |A|, D' in the disk, H/L PSD.
"""
import numpy as np
from scipy.optimize import minimize


def Psi(b, c, dp, r, s, penal=False):
    M = np.array([[1, np.conj(b), np.conj(dp)], [b, 1, np.conj(c)], [dp, c, 1]], dtype=complex)
    ev = np.linalg.eigvalsh(M)
    if ev.min() < -1e-12 and not penal:
        return None
    viol = max(0.0, -ev.min())
    ev = np.clip(ev, 0, None)
    val = (np.sum(ev ** r) / 3) ** s
    return val + (1e3 * viol if penal else 0.0)


def twisted_lb(A, B, C, D, s, tries=30, seed=0):
    r = 1 / s
    a = abs(A); alpha = A / a
    rng = np.random.default_rng(seed)
    best = np.inf

    Lmin = max(a, abs(B), abs(C), abs(D))   # L = sum rho_i >= |any own coefficient|

    def f(v):
        L = Lmin * (1 + v[0] ** 2)
        rad = np.sqrt(2 * L * (L - a))
        w = v[1] + 1j * v[2]
        if abs(w) > 1:
            w = w / abs(w)
        dp = alpha * D + rad * w
        val = Psi(B / L, C / L, dp / L, r, s, penal=True)
        return L * val
    for t in range(tries):
        v0 = np.array([rng.uniform(0, 1.2), rng.uniform(-1, 1), rng.uniform(-1, 1)])
        if t == 0:
            v0 = np.array([0.0, 0.0, 0.0])
        res = minimize(f, v0, method='Nelder-Mead', options={'maxiter': 4000, 'xatol': 1e-9, 'fatol': 1e-12})
        # accept only feasible minimisers
        Lr = Lmin * (1 + res.x[0] ** 2); rad = np.sqrt(2 * Lr * (Lr - a)); w = res.x[1] + 1j * res.x[2]
        if abs(w) > 1:
            w = w / abs(w)
        chk = Psi(B / Lr, C / Lr, (alpha * D + rad * w) / Lr, r, s)
        if chk is not None:
            best = min(best, Lr * chk)
    return best


if __name__ == '__main__':
    import sys
    s = 1.5; c1 = 1 - 3 ** (1 - s)
    pts = [(1, -0.312 + 0.364j, -0.03 - 0.465j, 0.493 + 0.384j),
           (1, 0.03 + 0.651j, -0.482 + 0.029j, -0.063 - 0.492j),
           (1, -0.293 - 0.368j, -0.207 + 0.48j, 0.656 - 0.061j),
           (1, -0.42 + 0.28j, 0.502 + 0.036j, -0.569 + 0.347j)]
    for (A, B, C, D) in pts:
        M = max(abs(B), abs(C), abs(D)); RT = np.sqrt(abs(B) ** 2 + abs(C) ** 2 + abs(D) ** 2)
        plb = max(abs(A) - c1 * M, abs(A) - RT / (2 * np.sqrt(3)), 3 ** (1 - s) * max(abs(A), M))
        tw = twisted_lb(A, B, C, D, s)
        print(f"|B,C,D|=({abs(B):.3f},{abs(C):.3f},{abs(D):.3f}) arg(A D/(B C))={np.angle(A*D/(B*C)):.3f}:  P_lb={plb:.4f}  twisted={tw:.4f}")
    # tightness checks: single atom and IF block and two aligned atoms
    xi = np.exp(1j * 0.7); eta = np.exp(-1j * 1.1)
    print("single atom:", twisted_lb(1, xi, eta, xi * eta, s), "expected", 3 ** (1 - s))
    print("IF block:", twisted_lb(1, 0, 0, 0, s), "expected 1")
