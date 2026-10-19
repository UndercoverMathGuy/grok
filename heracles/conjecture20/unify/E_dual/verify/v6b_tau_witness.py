"""Explicit witness for the p=13, s=2 entry: take an optimal y* with support {2,3}, add the floor on feature 1,
rescale to coverage 1; the value gives a LOWER bound on Gamma^{C(S,eps)} hence an UPPER bound on tau."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlib import *
p, s = 13, 2.0; r = 1 / s; F = 6; K = kap(p)[:F]
G0, y0 = programme_vertices(p, s)
# map support {4,6} -> {2,3} by dilation x7 (4*7=28=2, 6*7=42=3 mod 13)
perm = np.zeros(F)
for j in range(1, F + 1):
    jj = (j * 7) % p; jj = min(jj, p - jj)
    perm[jj - 1] = y0[j - 1]
print('Gamma=%.6f, y* (support {2,3}) =' % G0, np.round(perm, 6), ' coverage min', (K @ perm).min())
for eps in [1e-6, 1e-4, 1e-3]:
    y = perm.copy()
    # y_1 = eps * sum(y) solved: y_1 (1 - eps) = eps * sum(others)
    y[0] = eps * y[1:].sum() / (1 - eps)
    cov = (K @ y).min(); y = y / cov
    val = (y[y > 0] ** r).sum() ** (-s)
    print('eps=%g: floor witness value %.6f -> tau <= %.5f   (check floor: yhat_1 = %.2e)' % (eps, val, G0 / val - 1, y[0] / y.sum()))
