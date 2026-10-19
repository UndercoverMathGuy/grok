"""Explicit non-aligned interference-free atom configuration (n = 2, three atoms) whose atom-modulus entropy is
below log nu = log 3: entropic Lemma 7 needs alignment."""
import itertools, math
import numpy as np
from scipy.optimize import least_squares
rng = np.random.default_rng(2)
n, m = 2, 3
def unpack(x):
    w = np.exp(x[:m]) / np.exp(x[:m]).sum(); th = x[m:m + m * n].reshape(m, n); ph = x[m + m * n:]
    return w, th, ph
def res(x, target):
    w, th, ph = unpack(x); beta = w * np.exp(1j * ph)
    R = []
    for T in [(0,), (1,), (0, 1)]:
        v = (beta * np.exp(1j * th[:, list(T)].sum(1))).sum(); R += [v.real, v.imag]
    H = -(w * np.log(w)).sum()
    R += [10 * (H - target)]
    return np.array(R)
for trial in range(200):
    x0 = np.concatenate([rng.normal(0, 1, m), rng.uniform(0, 6.3, m * n), rng.uniform(0, 6.3, m)])
    sol = least_squares(res, x0, args=(math.log(3) - 0.15,), xtol=1e-15, ftol=1e-15, gtol=1e-15)
    if np.abs(sol.fun).max() < 1e-13:
        w, th, ph = unpack(sol.x); beta = w * np.exp(1j * ph)
        print("weights |beta_l| =", np.round(w, 12)); print("phases arg beta_l =", np.round(ph % (2*np.pi), 12))
        print("xi_lt angles =", np.round(th % (2*np.pi), 12))
        for T in [(0,), (1,), (0, 1)]:
            print("T =", T, "|sum_l beta_l prod xi| =", abs((beta * np.exp(1j * th[:, list(T)].sum(1))).sum()))
        print("|beta| = |sum beta_l| =", abs(beta.sum()), " (sum |beta_l| = 1)")
        print("Shannon H(|beta_l|) =", -(w * np.log(w)).sum(), " log 3 =", math.log(3))
        for r in [0.5, 2/3, 1.0]:
            e = w ** r / (w ** r).sum(); print(f"escort r={r:.3f}: H = {-(e*np.log(e)).sum():.6f};  Lemma 7: sum|b_l|^r = {(w**r).sum():.6f} >= 3^(1-r)|beta|^r = {3**(1-r)*abs(beta.sum())**r:.6f}")
        break
