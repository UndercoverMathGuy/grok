"""Phase-blind semi-rigid bound at (7,2), L={1,2}, with neuron-level paid ledgers.
  gamma/gamma_IF <= sqrt(a_+^2/F^2 + K^2/(4c))/Gamma_s,   paid pairing <= (K/2) R,  K >= sup_neuron pairing/rho.
ledgers: 'l2'  K = 2 ||L||_2   (p52 R3 ledger)
         'fam' K = ||Tn||_F    (family ledger: one mode modulus per (output, input classes) family)
         'blk' K = sqrt(sum_blocks U_X^2), U_X = max over output slices of Frobenius (if slices have disjoint
               supports in an input axis) else block Frobenius  (block order X,Y,Z)
         'free' K = 0 (cap)
Cases (T1,T2): the slot-sets of the detected coefficients (worst case taken).
Usage: python semirigid72c.py s ledger NG"""
import sys, time, itertools
import numpy as np
import cvxpy as cp
from semirigid72 import CM, KM, Gamma7, payment_simple
from leaks72 import leak_positions

CASES = [('1', '1'), ('1', '2'), ('1', '12'), ('12', '1'), ('12', '12')]
NONL = [{3}, {0, 3}, {0, 3}]


def lincomb(zr, zi, vec):
    return (zr @ vec.real - zi @ vec.imag, zr @ vec.imag + zi @ vec.real)


def coeff_vec(j, key):
    sgn, l = key
    v = CM[l - 1]
    return v if sgn == '+' else np.conj(v)


def build(s, ledger, case):
    r = 1 / s; c = 3.0 ** (2 - 2 * s) / 4
    mu = cp.Variable(6, nonneg=True)
    z = {1: (cp.Variable(6), cp.Variable(6)), 2: (cp.Variable(6), cp.Variable(6))}
    Apar = cp.Parameter(2, nonneg=True); Mpar = cp.Parameter(2, nonneg=True)
    cons = [cp.sum(mu) == 1] + [mu[k] == mu[5 - k] for k in range(3)]
    for d in range(6):
        cons.append(mu[d] >= 2 * cp.norm(cp.hstack([z[1][0][d], z[1][1][d]]), 2)
                    + 2 * cp.norm(cp.hstack([z[2][0][d], z[2][1][d]]), 2))
    S = {j: lincomb(*z[j], CM[j - 1])[0] for j in (1, 2)}
    fams = {}
    allL = []
    for j, T in ((1, case[0]), (2, case[1])):
        for d in leak_positions(j, T):
            re, im = lincomb(*z[j], coeff_vec(j, d['key']))
            if d['free']:
                cons += [re == 0, im == 0]
                continue
            nrm = cp.norm(cp.hstack([re, im]), 2)
            allL += [re, im]
            fams.setdefault(tuple(int(v) for v in d['fam']), []).append((2 ** (d['z'] / 2), nrm))
    entry = {f: sum(cf * nr for cf, nr in lst) for f, lst in fams.items()}
    if ledger == 'l2':
        K = 2 * cp.norm(cp.hstack(allL), 2)
    elif ledger == 'fam':
        K = cp.norm(cp.hstack(list(entry.values())), 2)
    elif ledger == 'blk':
        blocks = {0: [], 1: [], 2: []}
        for f in entry:
            idx = f
            first = next(a for a in range(3) if idx[a] in NONL[a])
            blocks[first].append(f)
        Us = []
        for bi, fl in blocks.items():
            if not fl:
                continue
            slices = {}
            for f in fl:
                slices.setdefault(f[0], []).append(f)
            disjoint = False
            for ax in (1, 2):
                sups = [set(f[ax] for f in sl) for sl in slices.values()]
                if all(a.isdisjoint(b) for a, b in itertools.combinations(sups, 2)):
                    disjoint = True
            if disjoint and len(slices) > 1:
                U = cp.max(cp.hstack([cp.norm(cp.hstack([entry[f] for f in sl]), 2) for sl in slices.values()]))
            else:
                U = cp.norm(cp.hstack([entry[f] for f in fl]), 2)
            Us.append(U)
        K = cp.norm(cp.hstack(Us), 2)
    elif ledger == 'free':
        K = cp.Constant(0.0)
    Kmu = KM @ mu
    a = Apar[0] * Kmu[0] + Apar[1] * Kmu[1] - S[1] * Mpar[0] - S[2] * Mpar[1]
    t = cp.Variable(nonneg=True)
    cons.append(t >= a)
    prob = cp.Problem(cp.Minimize(cp.norm(cp.hstack([t, K / (2 * np.sqrt(c))]), 2)), cons)
    return prob, Apar, Mpar


def make(s, ledger):
    return {case: build(s, ledger, case) for case in CASES}


def bound(probs, s, tau, x1, x2, pay=payment_simple):
    r = 1 / s
    A1, A2 = 1 - tau, tau
    F = ((pay(x1, s) * A1) ** r + (pay(x2, s) * A2) ** r) ** s
    worst = 0
    for case, (prob, Apar, Mpar) in probs.items():
        Apar.value = np.array([A1, A2]) / F; Mpar.value = np.array([x1 * A1, x2 * A2]) / F
        prob.solve(solver=cp.CLARABEL)
        worst = max(worst, prob.value / Gamma7(s))
    return worst


if __name__ == '__main__':
    s = float(sys.argv[1]); ledger = sys.argv[2]; NG = int(sys.argv[3]) if len(sys.argv) > 3 else 9
    probs = make(s, ledger)
    t0 = time.time(); best = (0, None)
    for tau in np.linspace(0, 1, 2 * NG - 1):
        for x1 in np.linspace(0, 2, NG):
            for x2 in np.linspace(0, 2, NG):
                v = bound(probs, s, tau, x1, x2)
                if v > best[0]:
                    best = (v, (round(tau, 3), round(x1, 3), round(x2, 3)))
    print(f"s={s} ledger={ledger} NG={NG}: grid sup {best[0]:.5f} at (tau,x1,x2)={best[1]}  [{time.time()-t0:.0f}s]", flush=True)
