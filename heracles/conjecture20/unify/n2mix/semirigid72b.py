"""Variants of the phase-blind semi-rigid bound at (7,2), L = {1,2}:
  ledger = 'l2'   : leakage priced by the l2 Pythagorean ledger (as semirigid72.py)
  ledger = 'free' : leakage costless (cap of any paid ledger for single-character phase-blind tests)
  ledger = 'none' : no leakage allowed (rigid isolated tests: R12 / p52 R14 regime)
Usage: python semirigid72b.py s ledger [grid]"""
import sys, time
import numpy as np
import cvxpy as cp
from semirigid72 import CM, KM, Gamma7, payment_simple

s = float(sys.argv[1]); ledger = sys.argv[2]; NG = int(sys.argv[3]) if len(sys.argv) > 3 else 11
r = 1 / s; c = 3.0 ** (2 - 2 * s) / 4

mu = cp.Variable(6, nonneg=True)
z1r, z1i, z2r, z2i = (cp.Variable(6) for _ in range(4))
Apar = cp.Parameter(2, nonneg=True); Mpar = cp.Parameter(2, nonneg=True)
cons = [cp.sum(mu) == 1] + [mu[k] == mu[5 - k] for k in range(3)]
for d in range(6):
    cons.append(mu[d] >= 2 * cp.norm(cp.hstack([z1r[d], z1i[d]]), 2) + 2 * cp.norm(cp.hstack([z2r[d], z2i[d]]), 2))
def lin(zr, zi, vec):
    return (zr @ vec.real - zi @ vec.imag, zr @ vec.imag + zi @ vec.real)
S1 = lin(z1r, z1i, CM[0])[0]; S2 = lin(z2r, z2i, CM[1])[0]
for l in (0, 1):
    re, im = lin(z2r, z2i, np.conj(CM[l])); cons += [re == 0, im == 0]
leaks = []
for l in (1, 2): leaks += list(lin(z1r, z1i, CM[l]))
for l in (0, 1, 2): leaks += list(lin(z1r, z1i, np.conj(CM[l])))
for l in (0, 2): leaks += list(lin(z2r, z2i, CM[l]))
leaks += list(lin(z2r, z2i, np.conj(CM[2])))
t = cp.Variable(nonneg=True)
Kmu = KM @ mu
a = Apar[0] * Kmu[0] + Apar[1] * Kmu[1] - S1 * Mpar[0] - S2 * Mpar[1]
cons.append(t >= a)
if ledger == 'l2':
    obj = cp.norm(cp.hstack([t, cp.norm(cp.hstack(leaks), 2) / np.sqrt(c)]), 2)
elif ledger == 'free':
    obj = t
elif ledger == 'none':
    cons += [x == 0 for x in leaks]; obj = t
prob = cp.Problem(cp.Minimize(obj), cons)


def bound(tau, x1, x2):
    A1, A2 = 1 - tau, tau
    F = ((payment_simple(x1, s) * A1) ** r + (payment_simple(x2, s) * A2) ** r) ** s
    Apar.value = np.array([A1, A2]) / F; Mpar.value = np.array([x1 * A1, x2 * A2]) / F
    prob.solve(solver=cp.CLARABEL)
    return prob.value / Gamma7(s)


t0 = time.time(); best = (0, None)
for tau in np.linspace(0, 1, 2 * NG - 1):
    for x1 in np.linspace(0, 2, NG):
        for x2 in np.linspace(0, 2, NG):
            v = bound(tau, x1, x2)
            if v > best[0]: best = (v, (round(tau, 3), round(x1, 3), round(x2, 3)))
print(f"s={s} ledger={ledger}: grid sup {best[0]:.5f} at (tau,x1,x2)={best[1]}  [{time.time()-t0:.0f}s]", flush=True)
