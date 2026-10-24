"""How tight are 4x4 twisted-Gram (Rotfel'd) cost bounds on the n=3 payment extremals?

For a chain with twist exponents e=(e_1..e_4) and nested slot order sigma, u_i = (p_i^{e_k} xi_i^{v_k})_k,
H = sum rho_i u_i u_i^*, Hhat = H/L, nu = spec(Hhat).  Rotfel'd: theta Q^s >= L M_r(nu).
Prints, for the 1-atom, 2-atom (L2) and 3-atom (pointwise) extremals at s=2 (and s=1.8552):
  theta Q^s, L*M_r(nu) for each chain, and the entries of Hhat (moduli) vs exact |beta_T|/L.
"""
import itertools
import numpy as np
from n3lib import coeffs, TNAMES

def two_atom(phi):
    th = np.pi / 2 - phi
    beta = np.array([np.exp(-1j * phi), np.exp(1j * phi)]) / (2 * np.cos(phi))
    ang = np.array([[0, 0, 0], [th, th, th]])
    return beta, ang

def three_atom(d, c, psi, th):
    beta = np.array([d, c * np.exp(1j * psi), c * np.exp(-1j * psi)])
    ang = np.array([[0, 0, 0], [th] * 3, [-th] * 3])
    return beta, ang

def chain_vectors(beta, ang, e, order):
    rho = np.abs(beta); p = beta / rho
    xi = np.exp(1j * ang)
    N = len(beta)
    U = np.ones((N, 4), dtype=complex)
    for i in range(N):
        cum = 1.0 + 0j
        for k in range(4):
            if k > 0:
                cum = cum * xi[i, order[k - 1]]
            U[i, k] = p[i] ** e[k] * cum
    return rho, U

def Mr(nu, r):
    nu = np.clip(nu, 0, None)
    return (np.mean(nu ** r)) ** (1 / r)

def report(beta, ang, s, label):
    r = 1 / s
    A, bT = coeffs(beta, ang)
    rho = np.abs(beta); L = rho.sum(); Q = np.sum(rho ** r)
    th = 4.0 ** (1 - s)
    print(f"== {label}: s={s}  |A|/L={abs(A)/L:.5f}  thetaQ^s/L={th*Q**s/L:.5f}  bonus/L={(abs(A)-th*Q**s)/L:.5f}"
          f"  M/L={np.abs(bT).max()/L:.5f}  R/L={np.sqrt(np.sum(np.abs(bT)**2))/L:.5f}")
    print("   exact |beta_T|/L:", dict(zip(TNAMES, np.round(np.abs(bT) / L, 4))))
    for e in [(0, 1, 2, 3), (0, 1, 1, 2), (0, 0, 1, 1)]:
        for order in [(0, 1, 2), (1, 0, 2), (0, 2, 1)]:
            rho, U = chain_vectors(beta, ang, e, order)
            H = (U.T * rho) @ U.conj() / L
            nu = np.linalg.eigvalsh(H)
            ent = {}
            for k in range(4):
                for j in range(k):
                    Tset = tuple(sorted(order[j:k]))
                    ent[''.join(str(t + 1) for t in Tset) + f"[{e[k]-e[j]}]"] = round(abs(H[k, j]), 4)
            print(f"   e={e} order={tuple(o+1 for o in order)}: L*M_r/L = {Mr(nu, r):.5f}  (a - M_r = {abs(A)/L - Mr(nu, r):.5f})  entries {ent}")

for s in [2.0, 1.8552]:
    r = 1 / s
    report(np.array([1.0 + 0j]), np.zeros((1, 3)), s, "single atom")
    # 2-atom L2 extremal: optimise phi
    best = max(((np.cos(f) - 2 ** (1 - s)) / (2 * np.cos(np.pi / 4 + f / 2)), f) for f in np.linspace(0, 1.2, 12001))
    b, a = two_atom(best[1]); report(b, a, s, f"2-atom L2 extremal phi={best[1]:.4f} cL2={best[0]:.5f}")
b, a = three_atom(0.37174, 0.3266, 0.2773, 1.432)
report(b, a, 2.0, "3-atom pointwise extremal (s=2)")
