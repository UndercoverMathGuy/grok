"""Block-injective paid ledger at (7,2), L={1,2}.
tau_q(w,u1,u2) = sum_{x,d} q(x,d) (w(y)-w(y-d)) u1(x1) u2(x2) restricted to frequency-class subspaces.
Paid part = blocks A (w in freq-3 plane) + B (w in L-planes, u1 in off-planes {0,3}) + C (w in L, u1 in L, u2 in off).
K <= sqrt(2 p^3) sqrt(N_A^2 + N_B^2 + N_C^2),  N_X = injective norm of tau on block X (orthonormal bases).
"""
import numpy as np
from socp72 import *
from kq72 import Tvec, Kq, PAIDMASK

p = 7
a = np.arange(p)


def basis(k):
    if k == 0:
        return [np.ones(p) / np.sqrt(p)]
    return [np.cos(2 * np.pi * k * a / p) * np.sqrt(2 / p), np.sin(2 * np.pi * k * a / p) * np.sqrt(2 / p)]


def tau_tensor(q):
    """full trilinear form as a p x p x p tensor: tau[w-index c, u1-index a1, u2-index a2]"""
    Q = q.reshape(p * p, p - 1)
    tau = np.zeros((p, p, p))
    for xi, (x1, x2) in enumerate(G.X):
        y = (x1 + x2) % p
        for di, d in enumerate(G.DELS):
            tau[y, x1, x2] += Q[xi, di]
            tau[(y - d) % p, x1, x2] -= Q[xi, di]
    return tau


def restrict(tau, kw, k1, k2):
    Bw = np.array(sum([basis(k) for k in kw], [])).T
    B1 = np.array(sum([basis(k) for k in k1], [])).T
    B2 = np.array(sum([basis(k) for k in k2], [])).T
    return np.einsum('cab,ci,aj,bk->ijk', tau, Bw, B1, B2)


def inj_norm(Tn, iters=200, starts=40, seed=0):
    """numerical injective norm (lower estimate via alternating maximisation, many starts)"""
    rng = np.random.default_rng(seed)
    best = 0
    for _ in range(starts):
        x = rng.normal(size=Tn.shape[0]); x /= np.linalg.norm(x)
        y = rng.normal(size=Tn.shape[1]); y /= np.linalg.norm(y)
        z = rng.normal(size=Tn.shape[2]); z /= np.linalg.norm(z)
        for _ in range(iters):
            x = np.einsum('ijk,j,k->i', Tn, y, z); x /= np.linalg.norm(x) + 1e-300
            y = np.einsum('ijk,i,k->j', Tn, x, z); y /= np.linalg.norm(y) + 1e-300
            z = np.einsum('ijk,i,j->k', Tn, x, y); z /= np.linalg.norm(z) + 1e-300
        best = max(best, abs(np.einsum('ijk,i,j,k->', Tn, x, y, z)))
    return best


if __name__ == '__main__':
    import sys
    s = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
    r = 1 / s; c = 3.0 ** (2 - 2 * s) / 4
    bf = BoundFam(s)
    A2 = 2 * np.cos(np.pi / 7)
    xi = np.exp(1j * np.array([[0.3, 1.1]]))
    a1, bt1 = own_from_atoms(np.array([1.0 + 0j]), xi)
    F = ((3 ** (1 - s)) ** r + A2 ** r) ** s
    v, q = bf([a1, A2], [bt1, [0, 0, 0]], F)
    tau = tau_tensor(q)
    L = [1, 2]; OFF = [0, 3]; ALL = [0, 1, 2, 3]
    NA = inj_norm(restrict(tau, [3], ALL, ALL))
    NB = inj_norm(restrict(tau, L, OFF, ALL))
    NC = inj_norm(restrict(tau, L, L, OFF))
    Kb = np.sqrt(2 * p ** 3) * np.sqrt(NA ** 2 + NB ** 2 + NC ** 2)
    # check the free blocks vanish and the own pairing
    FREEB = [restrict(tau, [j], [k1], [k2]) for j in L for k1 in L for k2 in L if not (k1 == j and k2 == j)]
    print("max |free block| entries:", max(np.max(np.abs(Fb)) for Fb in FREEB))
    T = Tvec(q)
    Ktrue, _ = Kq(T, starts=40)
    print(f"N_A={NA:.5f} N_B={NB:.5f} N_C={NC:.5f};  block-injective K <= {Kb:.4f};  true K ~ {Ktrue:.4f}; 2n = {2*np.sqrt(np.sum(np.abs(T[PAIDMASK])**2)):.4f}")
    a_ = own_functional([a1, A2], [bt1, [0, 0, 0]]) @ q
    for name, K in (('block', Kb), ('true', Ktrue)):
        print(f"  bound with {name} K: {np.sqrt(a_**2/F**2 + K**2/(4*c))/Gamma7(s):.5f}")
