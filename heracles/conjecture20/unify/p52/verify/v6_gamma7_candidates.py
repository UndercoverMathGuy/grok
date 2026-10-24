"""Repair check for R11(i): by the KKT/strict-convexity argument (see referee notes), for s>1 a maximiser of
min_delta (M y)_delta on {sum y^r = 1} has an active set whose tie equations fix y up to scale within its support.
Candidates: uniform (support 3), for each support pair and each pair of rows the tie point, and vertices.
Compute max candidate value / (3.5 3^{-s}) over a fine s-grid in (1, 1.5]."""
import numpy as np, itertools
p = 7
kap = lambda j, d: 1 - np.cos(2*np.pi*j*d/p)
M = np.array([[kap(j, d) for j in (1, 2, 3)] for d in (1, 2, 3)])
worst = 0; arg = None
for s in np.linspace(1.0005, 1.5, 2000):
    r = 1/s; uni = 3.5*3**-s
    vals = [M[0, 0]]   # vertex value: min_delta kappa_j(delta) = kappa_min for every j
    for J in itertools.combinations(range(3), 2):
        for d1, d2 in itertools.combinations(range(3), 2):
            a = M[d1, list(J)] - M[d2, list(J)]
            # tie: a[0] y0 + a[1] y1 = 0 with y >= 0, y != 0
            if a[0]*a[1] >= 0: continue
            y = np.array([-a[1], a[0]]) if a[0] > 0 or a[1] < 0 else np.array([a[1], -a[0]])
            y = np.abs(y); y = y/np.sum(y**r)**s
            yy = np.zeros(3); yy[list(J)] = y
            vals.append(np.min(M @ yy))
    worst = max(worst, max(vals)/uni)
    if worst == max(vals)/uni: arg = s
print(f"max over s in (1,1.5] of (best non-uniform candidate)/(3.5*3^-s) = {worst:.5f} at s={arg:.4f}  (<1 => uniform is the unique maximiser)")
