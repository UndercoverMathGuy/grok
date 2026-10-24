"""Survey of the channel-resolved bound at (7,2) over structured own-data families.
For each configuration: family-ledger SOCP bound (test optimised), then the block-injective K of that test and the
resulting bound sqrt(a^2/F^2 + K^2/(4c))/Gamma.  F uses EXACT atom costs (P_l = 3^{1-s} Q_l^s), i.e. the best case
for the method.  Usage: python survey72.py s family nsamp seed
families: a = feature-1 single atom + feature-2 IF;  b = feature-2 single atom + feature-1 IF;  c = both single atoms;
          d = random 1-2 atoms per feature;  e = two aligned atoms (P12 extremal) on one feature + IF other.
"""
import sys, time
import numpy as np
from socp72 import *
from kq72 import Tvec
from blockinj72b import block_K

s = float(sys.argv[1]); fam = sys.argv[2]; NS = int(sys.argv[3]); seed = int(sys.argv[4])
r = 1 / s; c = 3.0 ** (2 - 2 * s) / 4
rng = np.random.default_rng(seed)
bf = BoundFam(s)


def atomfeat(k, amp):
    beta = amp * rng.uniform(0.3, 1, k) * np.exp(1j * rng.uniform(-0.6, 0.6, k))
    xi = np.exp(1j * rng.uniform(0, 2 * np.pi, (k, 2)))
    return beta, xi


def featdata(beta, xi):
    A, bt = own_from_atoms(beta, xi)
    Q = np.sum(np.abs(beta) ** r)
    return A, bt, 3 ** (1 - s) * Q ** s


def IF(amp):
    return amp + 0j, [0, 0, 0], amp


def sample():
    t = rng.uniform(0.3, 3.5)          # A2/A1 scale
    if fam == 'a':
        b1, x1 = atomfeat(1, 1.0); f1 = featdata(b1, x1); f2 = IF(t)
    elif fam == 'b':
        f1 = IF(1.0); b2, x2 = atomfeat(1, t); f2 = featdata(b2, x2)
    elif fam == 'c':
        b1, x1 = atomfeat(1, 1.0); b2, x2 = atomfeat(1, t); f1 = featdata(b1, x1); f2 = featdata(b2, x2)
    elif fam == 'd':
        b1, x1 = atomfeat(rng.integers(1, 3), 1.0); b2, x2 = atomfeat(rng.integers(1, 3), t)
        f1 = featdata(b1, x1); f2 = featdata(b2, x2)
    elif fam == 'e':
        # two aligned atoms: beta = (1/2,1/2), xi = (i,-i) pattern with random rotation on slot phases
        ph = rng.uniform(0, 2 * np.pi, 2)
        xi = np.array([[1j * np.exp(1j * ph[0]), 1j * np.exp(1j * ph[1])], [-1j * np.exp(1j * ph[0]), -1j * np.exp(1j * ph[1])]])
        beta = np.array([0.5, 0.5]) + 0j
        if rng.uniform() < 0.5:
            f1 = featdata(beta, xi); f2 = IF(t)
        else:
            f1 = IF(1.0); f2 = featdata(t * beta, xi)
    return f1, f2, t


worst = (0, None)
t0 = time.time()
for k in range(NS):
    f1, f2, t = sample()
    A = [f1[0], f2[0]]; BT = [f1[1], f2[1]]
    F = (f1[2] ** r + f2[2] ** r) ** s
    v, q = bf(A, BT, F)
    T = Tvec(q)
    Kb, _ = block_K(T)
    a_ = own_functional(A, BT) @ q
    vK = np.sqrt(max(a_, 0) ** 2 / F ** 2 + Kb ** 2 / (4 * c)) / Gamma7(s)
    if vK > worst[0]:
        worst = (vK, (A, BT, F, t))
    print(f"{k}: t={t:.3f} |A|=({abs(A[0]):.3f},{abs(A[1]):.3f}) fam-bound {v:.5f}  K-bound {vK:.5f}   worst {worst[0]:.5f}  [{time.time()-t0:.0f}s]", flush=True)
print("WORST", worst[0], worst[1])
