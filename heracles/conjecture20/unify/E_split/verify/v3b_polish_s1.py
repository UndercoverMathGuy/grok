"""E3 (D'): epigraph SLSQP maximisation of gamma(.;1) at (5,2) and (7,2) from random starts; at near-maximisers check
H(theta;1) >= H^dec(1) = log nu + min H(y*), and the structural conditions (single plane, IF)."""
import math
import numpy as np
from scipy.optimize import minimize, linprog
from vlib import *
rng = np.random.default_rng(11)
out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)

def Gamma1(p):
    S = F_of(p); k = len(S)
    c = np.zeros(k + 1); c[-1] = -1
    A = [np.concatenate([-np.array([kappa(j, d, p) for j in S]), [1]]) for d in range(1, p)] + [np.concatenate([np.ones(k), [0]])]
    r = linprog(c, A_ub=np.array(A), b_ub=np.array([0] * (p - 1) + [1]), bounds=[(0, None)] * k + [(None, None)], method='highs')
    return -r.fun, r.x[:k]

for (p, n, m, runs) in [(5, 2, 6, 6), (5, 2, 7, 4), (7, 2, 9, 3)]:
    nu = n + 1
    G1, y = Gamma1(p); Hdec = math.log(nu) + ent(y / y.sum())
    for run in range(runs):
        x0 = rng.standard_normal(m * nu * p)
        def unpack(x):
            X = x[:m * nu * p].reshape(m, nu, p); return X[:, :n, :], X[:, n, :]
        def gvec(x):
            U, W = unpack(x); return gaps(table(U, W), p, n).ravel()
        U, W = unpack(x0); x0 = x0 / (sqnorm(U, W) ** (nu / 2)).sum() ** (1 / nu)
        z0 = np.concatenate([x0, [gvec(x0).min()]])
        cons = [{'type': 'ineq', 'fun': lambda z: gvec(z[:-1]) - z[-1]},
                {'type': 'eq', 'fun': lambda z: (sqnorm(*unpack(z[:-1])) ** (nu / 2)).sum() - 1}]
        best = None
        for it in range(3):
            r = minimize(lambda z: -z[-1], z0, constraints=cons, method='SLSQP', options={'maxiter': 2000, 'ftol': 1e-13})
            z0 = r.x
        U, W = unpack(z0[:-1])
        h = margin(U, W); g1 = h / (sqnorm(U, W) ** (nu / 2)).sum()
        _, w = shares(U, W, 1.0)
        keep = w > 1e-6
        e = plane_energy(U, W); rho = e / e.sum(1, keepdims=True)
        offdom = float(((1 - rho.max(1)) * w).sum())
        beta, betaT, _ = atoms(U, W)
        mb = max(abs(v) for v in beta.values()); mbt = max(abs(v) for v in betaT.values())
        say(f"(p,n,m)=({p},{n},{m}) run {run}: gamma/(B Gamma_1) = {g1/(Bp(p,n)*G1):.6f}, H - H^dec(1) = {ent(w)-Hdec:+.5f}, "
            f"off-dominant-plane share = {offdom:.1e}, max|beta_T|/max|beta| = {mbt/mb:.1e}")
open('v3b_polish_s1.out', 'w').write("\n".join(out) + "\n")
