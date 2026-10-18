"""Hiding vs the spread of the test measure mu: for the certified decoupling violator (p=5, n=2, b=2), compare
mu = uniform on Delta(theta) (the exact binding pair) with mu = uniform on all delta != 0 (all directions; this costs the
'excess' sum_delta mu_delta (sum_l a_l kappa_l(delta) - min)).  Reports hidden fraction and the MI slack in each case."""
import numpy as np, lib
from netopt import programme, gamma_IF, kappa_mat, own
p, n, b = 5, 2, 2.0; nu = 3; s = 1.5; r = 1 / s
Gam, y = programme(p, s); gIF = gamma_IF(p, n, b, Gam); Bp = lib.Bp(p, n); Cb = Bp ** (1 / s)
K = kappa_mat(p); F = lib.Fset(p)
for fname, k in [('viol_p5_n2_b2.npz', 0), ('viol_p5_n2_b2.npz', 1)]:
    d = np.load(fname); U, W = d[f'U{k}'], d[f'W{k}']
    Us, Ws = lib.split(U, W)
    P = lib.cost(U, W, b); sc = gIF * P ** s
    at = own(U, W, p, n); beta = np.array([at[j].sum() for j in F]); Q = np.array([(np.abs(at[j]) ** r).sum() for j in F])
    a = nu ** (1 - s) * Q ** s; cov_a = K @ a
    G1 = lib.gaps_from_table(lib.table(U, W), p, n)[:, 1:]; G2 = lib.gaps_from_table(lib.table(Us, Ws), p, n)[:, 1:]
    for name, mu in [('mu on Delta(theta)', None), ('mu uniform on all delta', np.ones(p - 1) / (p - 1))]:
        if mu is None:
            D = np.where(cov_a <= cov_a.min() * (1 + 1e-9))[0]; mu = np.zeros(p - 1); mu[D] = 1; mu[p - 2 - D] = 1; mu /= mu.sum()
        Kmu = mu @ K
        Det1 = mu @ (G1.mean(0) - G1.min(0)); Det2 = mu @ (G2.mean(0) - G2.min(0))
        excess = mu @ cov_a - cov_a.min()
        bonus = ((beta.real - a) * Kmu).sum()
        print(f"{fname}[{k}] {name:26s}: Det(split)={Det2 / sc:.4f} Det(theta)={Det1 / sc:.4f} hidden fraction={(Det2 - Det1) / Det2:.3f} "
              f"| bonus={bonus / sc:.4f} excess={excess / sc:.4f}  -> bonus - Det(theta) - excess = {(bonus - Det1 - excess) / sc:+.4f}")
