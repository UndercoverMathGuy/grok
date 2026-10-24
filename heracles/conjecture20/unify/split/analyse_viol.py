"""Independent (numpy, lib.py) re-evaluation of the (D_Q)/(D_split) violators saved by adv_dq.py, and of their splits."""
import sys, numpy as np
import lib
from netopt import programme, gamma_IF, kappa_mat, own, energy_stats
f = sys.argv[1]; p, n, b = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
nu = n + 1; s = nu / b; r = 1 / s
Gam, y = programme(p, s); gIF = gamma_IF(p, n, b, Gam); Bp = lib.Bp(p, n); Cb = Bp ** (1 / s)
K = kappa_mat(p); F = lib.Fset(p)
d = np.load(f)
def RQ(U, W):
    h = lib.margin(U, W); P = lib.cost(U, W, b)
    at = own(U, W, p, n); Q = np.array([(np.abs(at[j]) ** r).sum() for j in F]); beta = np.array([at[j].sum() for j in F])
    rhs = nu ** (1 - s) * (Cb * P / Q.sum()) ** s * (K @ (Q ** s)).min()
    a = nu ** (1 - s) * Q ** s
    return h / rhs, h / P ** s / gIF, np.abs(beta) / a, (Q ** s) / (Q ** s).sum()
for k in range(4):
    if f'U{k}' not in d: break
    U, W = d[f'U{k}'], d[f'W{k}']
    R, g, cheapness, alloc = RQ(U, W)
    Us, Ws = lib.split(U, W)
    Rs, gs, _, _ = RQ(Us, Ws)
    st = energy_stats(U, W, p)
    print(f"net {k}: R_Q={R:.5f} gamma/gIF={g:.5f} | split: R_Q={Rs:.5f} gamma/gIF={gs:.5f} | |beta_l|/a_l={np.round(cheapness,4)} "
          f"alloc(Q^s share)={np.round(alloc,3)} dc={st['dc']:.3f} dead={st['dead']:.3f} secondary={st['secondary']:.3f}")
