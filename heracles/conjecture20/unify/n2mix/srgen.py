"""Phase-blind semi-rigid channel-resolved bound at (p, n=2) for a general feature set L (exploration).
Networks with features in L.  Per feature l: amplitude A_l, x_l = max_T |beta_{l,T}| / A_l, payment
F_l >= A_l * pay(x_l) (c1sharp / trivial / two-atom cap).  Test: mean measure mu (symmetric) + one single-character
component per feature (pairing vector w^(l) in C^{p-1}; free indices zero; own index real).  Positivity
mu_d >= 2 sum_l |z_l(d)|.  Paid leakage priced by the l2 ledger K = 2 ||(w_m)_{paid}||  (T-independent) or 0 ('nopaid').
gamma/gamma_IF <= sqrt((a_+/F)^2 + 3^{2s-2} K^2) / Gamma_s(p)   (Gamma from the full programme).
Usage: python srgen.py p s ledger nstarts [Lmissing]"""
import sys
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from chanlib import Chan
from prog_small import kappa, solve as prog_solve


def pay(x, s):
    c1 = 1 - 3.0 ** (1 - s); r = 1 / s
    v = max(1 - c1 * x, 3.0 ** (1 - s) * max(1.0, x))
    if x <= 1:
        v = max(v, 3.0 ** (1 - s) * (((1 + x) / 2) ** r + ((1 - x) / 2) ** r) ** s)
    return v


class Gen:
    def __init__(self, p, L, s, ledger):
        self.p = p; self.L = tuple(L); self.s = s
        CH = Chan(p, L); self.CH = CH
        nL = len(L); half = (p - 1) // 2
        self.mu = cp.Variable(half, nonneg=True)
        mu = cp.hstack([self.mu[min(d, p - 1 - d) - 1] if False else self.mu[min(d, p - d) - 1] for d in range(1, p)])
        self.wr = {l: cp.Variable(p - 1) for l in L}; self.wi = {l: cp.Variable(p - 1) for l in L}
        self.cA = cp.Parameter(nL, nonneg=True); self.cM = cp.Parameter(nL, nonneg=True)
        cons = [cp.sum(mu) == 1]
        IR, II = CH.IDFT.real, CH.IDFT.imag
        za = {}
        for l in L:
            zr = IR @ self.wr[l] - II @ self.wi[l]; zi = IR @ self.wi[l] + II @ self.wr[l]
            za[l] = [cp.norm(cp.hstack([zr[d], zi[d]]), 2) for d in range(p - 1)]
        for d in range(p - 1):
            cons.append(mu[d] >= 2 * sum(za[l][d] for l in L))
        S = []; leaks = []
        for l in L:
            info = CH.component(l, '1')
            own = [m for (m, role, f, dd) in info if role == 'own'][0]
            S.append(self.wr[l][own - 1]); cons.append(self.wi[l][own - 1] == 0)
            for (m, role, f, dd) in info:
                if role == 'free':
                    cons += [self.wr[l][m - 1] == 0, self.wi[l][m - 1] == 0]
                elif role == 'paid':
                    leaks += [self.wr[l][m - 1], self.wi[l][m - 1]]
        K = 2 * cp.norm(cp.hstack(leaks), 2) if (ledger == 'l2' and leaks) else cp.Constant(0.0)
        KM = np.array([[1 - np.cos(2 * np.pi * l * d / p) for d in range(1, p)] for l in L])
        Kmu = KM @ mu
        a = self.cA @ Kmu - self.cM @ cp.hstack(S)
        t = cp.Variable(nonneg=True)
        cons.append(t >= a)
        self.prob = cp.Problem(cp.Minimize(cp.norm(cp.hstack([t, 3.0 ** (s - 1) * K]), 2)), cons)
        self.Gamma = prog_solve(kappa(p), s, restarts=20)[0]

    def bound(self, A, x):
        s = self.s; r = 1 / s
        A = np.asarray(A, float); x = np.asarray(x, float)
        F = np.sum((np.array([pay(xx, s) for xx in x]) * A) ** r) ** s
        self.cA.value = A / F; self.cM.value = x * A / F
        self.prob.solve(solver=cp.CLARABEL)
        return self.prob.value / self.Gamma


if __name__ == '__main__':
    p = int(sys.argv[1]); s = float(sys.argv[2]); ledger = sys.argv[3]; NS = int(sys.argv[4])
    miss = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    F = list(range(1, (p - 1) // 2 + 1)); L = [l for l in F if l != miss]
    g = Gen(p, L, s, ledger)
    nL = len(L)
    # restricted programme optimum for L (IF point)
    GL, yL = prog_solve(kappa(p), s, restarts=20, support=[l - 1 for l in L])
    print(f"p={p} L={L} s={s}: Gamma={g.Gamma:.5f} Gamma_L/Gamma={GL/g.Gamma:.5f}", flush=True)
    rng = np.random.default_rng(0)
    def f(v):
        A = np.abs(v[:nL]) + 1e-12; A = A / A.sum(); x = np.clip(np.abs(v[nL:]), 0, 1.75)
        return -g.bound(A, x)
    best = (0, None)
    yLs = yL[[l - 1 for l in L]]
    for k in range(NS):
        if k == 0:
            v0 = np.concatenate([yLs ** (1), np.zeros(nL)])
        else:
            v0 = np.concatenate([yLs * rng.uniform(0.5, 1.5, nL), rng.uniform(0, 1.2, nL) * (rng.uniform(size=nL) < 0.6)])
        res = minimize(f, v0, method='Nelder-Mead', options={'maxiter': 600, 'xatol': 1e-4, 'fatol': 1e-6})
        if -res.fun > best[0]:
            best = (-res.fun, res.x)
        print(f"  start {k}: {-res.fun:.5f}", flush=True)
    A = np.abs(best[1][:nL]); A /= A.sum()
    print(f"SUP {best[0]:.5f} at A={np.round(A,4)} x={np.round(np.clip(np.abs(best[1][nL:]),0,1.75),4)}", flush=True)
