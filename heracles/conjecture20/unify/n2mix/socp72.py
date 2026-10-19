"""Channel-resolved certificate bound at (p,n) = (7,2), features L = {1,2} (beta_3 = 0).

For a probability q on Z_7^2 x Z_7^*:  sum q g = Re sum_{j,chi} phi_j(chi) T_j(chi),
   T_j(chi) = sum_{x,d} q(x,d) c_j(d) rho^{<chi,x>}.
Ledger 1:  h <= a(q) + n(q) R_M,  a(q) = own pairing,  n(q) = ||T_paid(q)||_2, provided T_free(q) = 0.
Ledger 2:  3^{1-s} B_7 P^s >= sqrt(F^2 + c_s R_M^2),  c_s = 3^{2-2s}/4,  F = (P_1^r + P_2^r)^s,  P_j >= payment of feature j.
=> gamma/gamma_IF <= min_q sqrt(a(q)_+^2/F^2 + n(q)^2/c_s) / Gamma_s(7),  Gamma_s(7) = (7/2) 3^{-s}.
"""
import itertools
import numpy as np
import cvxpy as cp
from n2lib import Grid, classify

p = 7
G = Grid(p)
NX = p * p
ND = p - 1
NQ = NX * ND
X = G.X


def build_maps(L=(1, 2)):
    lab = classify(G, list(L))
    # T_j(chi) = sum_{x,d} q[x,d] c_j(d) rho^{<chi,x>};  q index = xi*ND + di
    rows = {}
    for ji, j in enumerate(G.F):
        for c1 in range(p):
            for c2 in range(p):
                ph = G.rho ** ((c1 * X[:, 0] + c2 * X[:, 1]) % p)          # NX
                cd = np.array([G.cfac(j, d) for d in G.DELS])               # ND
                rows[(ji, c1, c2)] = np.outer(ph, cd).reshape(-1)          # NQ complex
    free = [k for k, v in lab.items() if v == 'free']
    paid = [k for k, v in lab.items() if v == 'paid']
    Mfree = np.array([rows[k] for k in free])
    Mpaid = np.array([rows[k] for k in paid])
    return rows, lab, free, paid, Mfree, Mpaid


ROWS, LAB, FREE, PAID, MFREE, MPAID = build_maps()


def own_functional(A, BT):
    """real vector w (NQ) with a(q) = w.q.  A = [A1, A2], BT[j][T] (T in G.TS order) for j = 1,2."""
    w = np.zeros(NQ)
    for ji, j in enumerate((1, 2)):
        w += np.real(np.conj(A[ji]) * ROWS[(ji, 0, 0)])
        for ti, T in enumerate(G.TS):
            key = (ji, (2 * j * T[0]) % p, (2 * j * T[1]) % p)
            assert LAB[key] == 'own'
            w += np.real(np.conj(BT[ji][ti]) * ROWS[key])
    return w


def Gamma7(s):
    return 3.5 * 3.0 ** (-s)


class Bound:
    def __init__(self, s, blind=True, paid_ok=True):
        self.s = s
        self.c = 3.0 ** (2 - 2 * s) / 4
        self.q = cp.Variable(NQ, nonneg=True)
        self.u = cp.Variable(nonneg=True)
        self.wF = cp.Parameter(NQ)
        cons = [cp.sum(self.q) == 1, self.u >= self.wF @ self.q]
        if blind:
            cons += [np.real(MFREE) @ self.q == 0, np.imag(MFREE) @ self.q == 0]
        Pm = np.vstack([np.real(MPAID), np.imag(MPAID)]) / np.sqrt(self.c)
        if paid_ok:
            obj = cp.norm(cp.hstack([self.u, Pm @ self.q]), 2)
        else:   # paid modes treated as free (must be annihilated)
            cons += [Pm @ self.q == 0]
            obj = self.u
        self.prob = cp.Problem(cp.Minimize(obj), cons)

    def __call__(self, A, BT, F, verbose=False):
        w = own_functional(A, BT)
        self.wF.value = w / F
        try:
            self.prob.solve(solver=cp.CLARABEL)
        except Exception:
            self.prob.solve(solver=cp.SCS, eps=1e-7)
        val = self.prob.value
        return val / Gamma7(self.s), self.q.value


def own_from_atoms(beta, xi):
    """atoms beta (k,), xi (k,2) unit -> A, [B, C, D]"""
    A = beta.sum()
    return A, [np.sum(beta * xi[:, 0]), np.sum(beta * xi[:, 1]), np.sum(beta * xi[:, 0] * xi[:, 1])]


if __name__ == '__main__':
    import sys, time
    s = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
    r = 1 / s
    bd = Bound(s)
    # IF configuration at the two-feature optimum: A2/A1 = 2 cos(pi/7)
    A1 = 1.0; A2 = 2 * np.cos(np.pi / 7)
    F = (A1 ** r + A2 ** r) ** s
    t0 = time.time()
    v, q = bd([A1, A2], [[0, 0, 0], [0, 0, 0]], F)
    print(f"s={s}: IF two-feature tie point: bound {v:.6f} (expected Gamma12/Gamma)  [{time.time()-t0:.2f}s]")
    # single-atom feature 1 (cheap), IF feature 2
    xi = np.exp(1j * np.array([[0.3, 1.1]]))
    a1, bt1 = own_from_atoms(np.array([1.0 + 0j]), xi)
    P1 = 3 ** (1 - s) * 1.0
    P2 = A2
    F = (P1 ** r + P2 ** r) ** s
    v, q = bd([a1, A2], [bt1, [0, 0, 0]], F)
    print(f"single-atom feature 1 + IF feature 2: bound {v:.6f}")
    print(f"#free constraints {len(FREE)}, #paid {len(PAID)}")


# ------------------------------------------------------------------ family-aware paid norm
def families(lab_kind='paid'):
    """paid families: (jidx, k1cls, k2cls) -> list of member keys (jidx, chi1, chi2), excluding the zero clean member."""
    fam = {}
    for (ji, c1, c2), v in LAB.items():
        if v != lab_kind:
            continue
        j = G.F[ji]
        k1, k2 = (c1 - j) % p, (c2 - j) % p
        key = (ji, G.cls(k1), G.cls(k2))
        fam.setdefault(key, []).append((ji, c1, c2))
    return fam


FAM = families('paid')


def nmembers(key):
    ji, a, b = key
    return 2 ** ((a != 0) + (b != 0))


class BoundFam(Bound):
    """same as Bound but with the family-aware paid norm
       n''(q)^2 = sum_f (sum_{mu in f} |T_mu|)^2 / n_f  (n_f = 2^{#nonzero input classes})."""
    def __init__(self, s, blind=True):
        self.s = s
        self.c = 3.0 ** (2 - 2 * s) / 4
        self.q = cp.Variable(NQ, nonneg=True)
        self.u = cp.Variable(nonneg=True)
        self.wF = cp.Parameter(NQ)
        cons = [cp.sum(self.q) == 1, self.u >= self.wF @ self.q]
        if blind:
            cons += [np.real(MFREE) @ self.q == 0, np.imag(MFREE) @ self.q == 0]
        terms = []
        for key, mem in FAM.items():
            absT = []
            for mk in mem:
                row = ROWS[mk]
                absT.append(cp.norm(cp.hstack([np.real(row) @ self.q, np.imag(row) @ self.q]), 2))
            terms.append(cp.sum(cp.hstack(absT)) / np.sqrt(nmembers(key) * self.c))
        obj = cp.norm(cp.hstack([self.u] + terms), 2)
        self.prob = cp.Problem(cp.Minimize(obj), cons)
