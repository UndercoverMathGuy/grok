"""Full (phase-aware) channel-resolved test at (7,2), L = {1,2}: the test q is an arbitrary probability on
Z_7^2 x Z_7^*, blind to free pairs, paid pairing priced by the K-plan block bound (neuron level).
For given exact own data (A_j, B_j, C_j, D_j) and payments P_j:
   gamma/gamma_IF <= min_q sqrt( (a(q)/F)^2 + 3^{2s-2} K(q)^2 ) / Gamma_s,   F = (P_1^r + P_2^r)^s.
Used for exploration only (exact atom costs).  Usage: python fullsocp72.py s config
"""
import sys
import numpy as np
import cvxpy as cp
from socp72 import ROWS, LAB, NQ, own_functional, own_from_atoms, Gamma7, G
from kplan import make_plan, eval_cvx

p = 7


def paid_families():
    fam = {}
    for (ji, c1, c2), v in LAB.items():
        if v != 'paid':
            continue
        j = G.F[ji]
        k1, k2 = (c1 - j) % p, (c2 - j) % p
        key = (j, G.cls(k1), G.cls(k2))
        z = int(k1 == 0) + int(k2 == 0)
        fam.setdefault(key, []).append(((ji, c1, c2), z))
    return fam


FAMS = paid_families()
FREE = [k for k, v in LAB.items() if v == 'free']


class FullBound:
    def __init__(self, s, ledger='pat'):
        self.s = s
        self.q = cp.Variable(NQ, nonneg=True)
        self.wF = cp.Parameter(NQ)
        cons = [cp.sum(self.q) == 1]
        for k in FREE:
            row = ROWS[k]
            cons += [np.real(row) @ self.q == 0, np.imag(row) @ self.q == 0]
        entry = {}
        for f, mem in FAMS.items():
            terms = []
            for (key, z) in mem:
                row = ROWS[key]
                terms.append(2 ** (z / 2) * cp.norm(cp.hstack([np.real(row) @ self.q, np.imag(row) @ self.q]), 2))
            entry[f] = cp.sum(cp.hstack(terms)) if len(terms) > 1 else terms[0]
        self.entry = entry
        if ledger == 'pat':
            K = eval_cvx(make_plan(entry.keys()), entry)
        elif ledger == 'fam':
            K = cp.norm(cp.hstack(list(entry.values())), 2)
        else:
            K = cp.Constant(0.0)
        self.K = K
        t = cp.Variable(nonneg=True)
        cons.append(t >= self.wF @ self.q)
        self.prob = cp.Problem(cp.Minimize(cp.norm(cp.hstack([t, 3.0 ** (s - 1) * K]), 2)), cons)

    def __call__(self, A, BT, F):
        self.wF.value = own_functional(A, BT) / F
        self.prob.solve(solver=cp.CLARABEL)
        return self.prob.value / Gamma7(self.s)


def feature_from_atoms(beta, xi, s):
    A, bt = own_from_atoms(beta, xi)
    Q = np.sum(np.abs(beta) ** (1 / s))
    return A, bt, 3 ** (1 - s) * Q ** s


if __name__ == '__main__':
    s = float(sys.argv[1]); cfg = sys.argv[2] if len(sys.argv) > 2 else 'atomIF'
    r = 1 / s
    rng = np.random.default_rng(0)
    out = []
    for ledger in ('pat', 'nopaid'):
        fb = FullBound(s, ledger)
        worst = 0
        for trial in range(12):
            t = 2 * np.cos(np.pi / 7) * rng.uniform(0.6, 1.4)
            ph = rng.uniform(0, 2 * np.pi, 4)
            xi1 = np.exp(1j * np.array([[ph[0], ph[1]]])); xi2 = np.exp(1j * np.array([[ph[2], ph[3]]]))
            if cfg == 'atomIF':
                A1, b1, P1 = feature_from_atoms(np.array([1.0 + 0j]), xi1, s); A2, b2, P2 = t + 0j, [0, 0, 0], t
            elif cfg == 'atomatom':
                A1, b1, P1 = feature_from_atoms(np.array([1.0 + 0j]), xi1, s); A2, b2, P2 = feature_from_atoms(np.array([t + 0j]), xi2, s)
            elif cfg == 'IFatom':
                A1, b1, P1 = 1.0 + 0j, [0, 0, 0], 1.0; A2, b2, P2 = feature_from_atoms(np.array([t + 0j]), xi2, s)
            F = (P1 ** r + P2 ** r) ** s
            v = fb([A1, A2], [b1, b2], F)
            worst = max(worst, v)
        out.append((ledger, worst))
        print(f"s={s} cfg={cfg} ledger={ledger}: max over 12 random phase/ratio draws = {worst:.5f}", flush=True)
