"""Block-injective paid ledger at (7,2), L={1,2}, built from the paid mode transform (clean freq-3 member excluded).
tau_paid[c,a1,a2] = Re sum_{paid (j,k)} (2/p^3) rho^{-jc - k1 a1 - k2 a2} T_j(k + j 1)."""
import numpy as np
from socp72 import *
from kq72 import Tvec, Kq, PAIDMASK, tau_paid
from blockinj72 import basis, inj_norm

p = 7
a = np.arange(p)


def paid_tensor(T):
    tau = np.zeros((p, p, p))
    for ji, j in enumerate(G.F):
        for k1 in range(p):
            for k2 in range(p):
                c1, c2 = (k1 + j) % p, (k2 + j) % p
                if not PAIDMASK[ji, c1, c2]:
                    continue
                coef = 2 * p ** (-3) * T[ji, c1, c2]
                tau += np.real(coef * np.einsum('c,a,b->cab', G.rho ** (-j * a), G.rho ** (-k1 * a), G.rho ** (-k2 * a)))
    return tau


def restrict(tau, kw, k1, k2):
    Bw = np.array(sum([basis(k) for k in kw], [])).T
    B1 = np.array(sum([basis(k) for k in k1], [])).T
    B2 = np.array(sum([basis(k) for k in k2], [])).T
    return np.einsum('cab,ci,aj,bk->ijk', tau, Bw, B1, B2)


def block_K(T, partitions=None):
    tau = paid_tensor(T)
    L = [1, 2]; OFF = [0, 3]; ALL = [0, 1, 2, 3]
    out = {}
    # three product partitions of the paid set (by which factor is 'off' first)
    parts = {
        'w-first': [([3], ALL, ALL), (L, OFF, ALL), (L, L, OFF)],
        'u1-first': [(ALL[1:], OFF, ALL), ([3], L, ALL), (L, L, OFF)],
        'u2-first': [(ALL[1:], ALL, OFF), ([3], ALL, L), (L, OFF, L)],
    }
    best = np.inf
    for name, P in parts.items():
        Ns = [inj_norm(restrict(tau, *blk)) for blk in P]
        K = np.sqrt(2 * p ** 3) * np.sqrt(sum(N ** 2 for N in Ns))
        out[name] = (K, Ns)
        best = min(best, K)
    return best, out


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
    T = Tvec(q)
    # consistency: tau_paid via tensor equals kq72.tau_paid on random neurons
    rng = np.random.default_rng(0)
    tau = paid_tensor(T)
    err = 0
    for _ in range(5):
        om = rng.normal(size=21)
        v1 = np.einsum('cab,c,a,b->', tau, om[14:21], om[:7], om[7:14])
        err = max(err, abs(v1 - tau_paid(om, T)))
    print("tensor vs mode formula:", err)
    Kb, out = block_K(T)
    for k, (K, Ns) in out.items():
        print(f"  {k}: K <= {K:.4f}  (N = {', '.join(f'{x:.5f}' for x in Ns)})")
    a_ = own_functional([a1, A2], [bt1, [0, 0, 0]]) @ q
    print(f"2n = {2*np.sqrt(np.sum(np.abs(T[PAIDMASK])**2)):.4f};  best block K = {Kb:.4f}")
    print(f"bound with block K: {np.sqrt(a_**2/F**2 + Kb**2/(4*c))/Gamma7(s):.5f}  (family {v:.5f})")
