"""Phase-blind semi-rigid bound at (7,2), L = {1,2}, in the pairing-vector parametrisation (chanlib conventions).
Component j at chi_{j,T_j}: pairing vector w^{(j)} in C^6 (m = 1..6); own detection Re w^{(j)}_{-j};
free indices forced to 0; delta-profile z_j = IDFT w^{(j)};  positivity mu_d >= 2|z_1(d)| + 2|z_2(d)|.
gamma/gamma_IF <= sqrt((a_+/F)^2 + 3^{2s-2} K^2)/Gamma_s,  a = K_1(mu)A_1 + K_2(mu)A_2 - S_1 M_1 - S_2 M_2.
"""
import itertools
import numpy as np
import cvxpy as cp
from chanlib import Chan, block_assign

p = 7
CH = Chan(7, (1, 2))
CASES = [('1', '1'), ('1', '2'), ('1', '12'), ('12', '1'), ('12', '12'), ('2', '1'), ('2', '2'), ('2', '12'), ('12', '2')]
CASES5 = [('1', '1'), ('1', '2'), ('1', '12'), ('12', '1'), ('12', '12')]


def Gamma7(s):
    return 3.5 * 3.0 ** (-s)


def pay_simple(x, s):
    c1 = 1 - 3.0 ** (1 - s)
    return max(1 - c1 * x, 3.0 ** (1 - s) * max(1.0, x))


def comp_info(j, T):
    info = CH.component(j, T)
    own = [m for (m, role, f, d) in info if role == 'own'][0]
    free = [m for (m, role, f, d) in info if role == 'free']
    paid = [(m, f, d) for (m, role, f, d) in info if role == 'paid']
    return own, free, paid


class SR72:
    def __init__(self, s, ledger, case, sym=True):
        self.s = s; self.case = case; self.ledger = ledger
        self.mu = cp.Variable(6, nonneg=True)
        self.wr = {j: cp.Variable(6) for j in (1, 2)}
        self.wi = {j: cp.Variable(6) for j in (1, 2)}
        self.Apar = cp.Parameter(2, nonneg=True); self.Mpar = cp.Parameter(2, nonneg=True)
        cons = [cp.sum(self.mu) == 1]
        if sym:
            cons += [self.mu[k] == self.mu[5 - k] for k in range(3)]
        IR, II = CH.IDFT.real, CH.IDFT.imag
        zs = {}
        for j in (1, 2):
            zr = IR @ self.wr[j] - II @ self.wi[j]
            zi = IR @ self.wi[j] + II @ self.wr[j]
            zs[j] = (zr, zi)
        for d in range(6):
            cons.append(self.mu[d] >= 2 * cp.norm(cp.hstack([zs[1][0][d], zs[1][1][d]]), 2)
                        + 2 * cp.norm(cp.hstack([zs[2][0][d], zs[2][1][d]]), 2))
        self.S = {}; self.paid = {}
        ent = {}
        allabs = []
        for j, T in ((1, case[0]), (2, case[1])):
            own, free, paid = comp_info(j, T)
            self.S[j] = self.wr[j][own - 1]
            cons.append(self.wi[j][own - 1] == 0)
            for m in free:
                cons += [self.wr[j][m - 1] == 0, self.wi[j][m - 1] == 0]
            self.paid[j] = paid
            for (m, f, d) in paid:
                ab = cp.norm(cp.hstack([self.wr[j][m - 1], self.wi[j][m - 1]]), 2)
                allabs.append(ab)
                ent.setdefault(tuple(int(t) for t in f), []).append((2 ** (d / 2), ab))
        entry = {f: sum(cf * ab for cf, ab in lst) for f, lst in ent.items()}
        self.entry_fams = list(entry.keys())
        if ledger == 'l2':
            K = 2 * cp.norm(cp.hstack(allabs), 2)
        elif ledger == 'fam':
            K = cp.norm(cp.hstack(list(entry.values())), 2)
        elif ledger == 'blk':
            Us = []
            blocks = {}
            for f in entry:
                blocks.setdefault(block_assign(f, (1, 2)), []).append(f)
            for bi, fl in blocks.items():
                slices = {}
                for f in fl:
                    slices.setdefault(f[0], []).append(f)
                disjoint = len(slices) > 1 and any(
                    all(a.isdisjoint(b) for a, b in itertools.combinations([set(f[ax] for f in sl) for sl in slices.values()], 2))
                    for ax in (1, 2))
                if disjoint:
                    Us.append(cp.max(cp.hstack([cp.norm(cp.hstack([entry[f] for f in sl]), 2) for sl in slices.values()])))
                else:
                    Us.append(cp.norm(cp.hstack([entry[f] for f in fl]), 2))
            K = cp.norm(cp.hstack(Us), 2)
        elif ledger == 'free':
            K = cp.Constant(0.0)
        self.K = K
        Kmu = CH.KM[:2] @ self.mu
        a = self.Apar[0] * Kmu[0] + self.Apar[1] * Kmu[1] - self.S[1] * self.Mpar[0] - self.S[2] * self.Mpar[1]
        t = cp.Variable(nonneg=True)
        cons.append(t >= a)
        self.a = a
        self.prob = cp.Problem(cp.Minimize(cp.norm(cp.hstack([t, 3.0 ** (s - 1) * K]), 2)), cons)

    def solve(self, A1, A2, M1, M2, F):
        self.Apar.value = np.array([A1, A2]) / F
        self.Mpar.value = np.array([M1, M2]) / F
        self.prob.solve(solver=cp.CLARABEL)
        return self.prob.value / Gamma7(self.s)

    def w(self, j):
        return self.wr[j].value + 1j * self.wi[j].value


def config_bound(models, s, tau, x1, x2, pay=pay_simple):
    r = 1 / s
    A1, A2 = 1 - tau, tau
    F = ((pay(x1, s) * A1) ** r + (pay(x2, s) * A2) ** r) ** s
    vals = {case: m.solve(A1, A2, x1 * A1, x2 * A2, F) for case, m in models.items()}
    return max(vals.values()), vals


if __name__ == '__main__':
    import sys, time
    s = float(sys.argv[1]); ledger = sys.argv[2]; NG = int(sys.argv[3]) if len(sys.argv) > 3 else 9
    models = {case: SR72(s, ledger, case) for case in CASES5}
    t0 = time.time(); best = (0, None)
    for tau in np.linspace(0, 1, 2 * NG - 1):
        for x1 in np.linspace(0, 2, NG):
            for x2 in np.linspace(0, 2, NG):
                v, _ = config_bound(models, s, tau, x1, x2)
                if v > best[0]:
                    best = (v, (round(tau, 3), round(x1, 3), round(x2, 3)))
    print(f"s={s} ledger={ledger} NG={NG}: grid sup {best[0]:.5f} at {best[1]}  [{time.time()-t0:.0f}s]", flush=True)
