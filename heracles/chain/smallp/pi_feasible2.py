# NUMERICAL: PI feasibility with the exact p-gon mean in the averaged orders (instead of the Hoelder bound).
import sys, numpy as np
from scipy.optimize import linprog
from transfer_cert import ORDERS, beta
from pi_feasible import N
def mean_p(x, w, p, nps=64):
    ps = np.linspace(0, 2 * np.pi / p, nps)[:, None] + 2 * np.pi * np.arange(p)[None, :] / p
    return np.abs(x + w * np.exp(1j * ps)).mean(axis=1).min()
def betas(m, As, Ao, p):
    b = [float(beta(m, o, As, Ao)) for o in ORDERS]
    b += [As * mean_p(m[2], m[1], p), As * mean_p(m[2], m[0], p), Ao * mean_p(m[0], m[1], p)]
    return np.array(b)
def run(As, Ao, p, n=26):
    grid = np.linspace(0, np.sqrt(3), n); pts, needs = [], []
    for j1 in grid:
        for j2 in grid:
            for j12 in grid:
                j = (j1, j2, j12); v = N(j)
                if v > 1e-12: pts.append(j); needs.append(v)
    B = np.array([betas(np.array(j), As, Ao, p) for j in pts]); k = B.shape[1]; needs = np.array(needs)
    r = linprog(np.r_[np.zeros(k), -1], A_ub=np.c_[-B, np.ones(len(B))], b_ub=-needs,
                A_eq=[np.r_[np.ones(k), 0]], b_eq=[1], bounds=[(0, None)] * k + [(None, None)])
    return -r.fun, np.round(r.x[:k], 3)
if __name__ == '__main__':
    for name, As, Ao, p in [('p13 dense shift L6', 0.355114, 0.075851, 13), ('p11 dense shift L10', 0.423078, 0.0968349, 11)]:
        t, lam = run(As, Ao, p); print(f"{name}: min slack {t:+.4f} lam={list(lam)}", flush=True)
