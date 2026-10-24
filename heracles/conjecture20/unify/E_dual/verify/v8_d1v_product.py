"""Referee check of D1v with a NON-IF single feature theta_1 (frequency-1 outputs, arbitrary inputs incl. DC and
partners): relabelled copies u(j.), w(j.) scaled by y_j^{1/nu} -> profile yhat, p_Theta = yhat (x) p_1, Renyi additivity,
and the price identity e(theta;s) = e(theta;1) K_{1/s}(p)^{1-s}."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlib import *
rng = np.random.default_rng(3)
for (p, n) in [(7, 2), (11, 2), (7, 3)]:
    nu = n + 1; F = (p - 1) // 2; x = np.arange(p); m = 5
    U = rng.standard_normal((m, n, p))                               # arbitrary inputs
    W = np.array([np.cos(2 * np.pi * x / p + rng.uniform(0, 7)) * rng.uniform(0.5, 2) + rng.normal() for _ in range(m)])
    f1 = table(U, W); b1 = clean(f1, p, n)
    assert np.all(abs(b1[1:]) < 1e-12 * abs(b1[0]))                 # single feature (frequency 1)
    y = rng.uniform(0, 1, F) * (rng.random(F) < 0.8); y[0] = max(y[0], 0.3)
    Us, Ws = [], []
    for j in range(1, F + 1):
        if y[j - 1] == 0: continue
        perm = (j * x) % p
        Us.append(U[:, :, perm] * y[j - 1] ** (1 / nu)); Ws.append(W[:, perm] * y[j - 1] ** (1 / nu))
    UT = np.concatenate(Us); WT = np.concatenate(Ws)
    fT = table(UT, WT); bT = clean(fT, p, n)
    yh = y / y.sum(); prof = abs(bT) / abs(bT).sum()
    a1 = acost(U, W); p1 = a1 / a1.sum(); aT = acost(UT, WT); pT = aT / aT.sum()
    prod = np.outer(yh[y > 0], p1).ravel()
    errs = [abs(prof - yh).max(), abs(pT - prod).max()]
    for al in [0.4, 1.0, 2.0]:
        errs.append(abs(renyi(pT, al) - renyi(yh, al) - renyi(p1, al)))
    print('p=%d n=%d: |profile - yhat| %.1e ; |p_Theta - yhat x p1| %.1e ; Renyi additivity errs (0.4,1,2) %.1e %.1e %.1e'
          % (p, n, *errs))
